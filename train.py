
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import read_yaml
from dataset import HeadlineDataset, collate_fn
from models import TokenGenerator

from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def model_size(model):
    num_params = sum(param.numel() for param in model.parameters())
    return num_params

def train(model, loader, optimizer, scheduler, log_interval):
    train_loss = 0.0
    model.train()

    batch_t0 = time.time()

    for batch, (seqs, mask) in enumerate(loader):
        seqs = seqs.to(device)
        mask = mask.to(device)

        inp_tokens = seqs[:, :-1]
        tar_tokens = seqs[:, 1:]

        token_mask = mask[:, 1:]

        pred = model(inp_tokens, token_mask)
        loss = masked_loss(pred, tar_tokens, token_mask)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch % log_interval == 0:
            t_batch = (time.time() - batch_t0) * 1000 / log_interval
            current_lr = optimizer.param_groups[0]['lr']

            print(f'batch {batch:5d}/{len(loader)} | {int(t_batch):5d} ms/batch | learning rate {current_lr:.4g} | training loss {loss.item():.4f}')
            batch_t0 = time.time()

    return train_loss / len(loader)

@torch.no_grad()
def validate(model, loader):
    val_loss = 0.0
    model.eval()
    
    for seqs, mask in loader:
        seqs = seqs.to(device)
        mask = mask.to(device)

        inp_tokens = seqs[:, :-1]
        tar_tokens = seqs[:, 1:]

        token_mask = mask[:, 1:]

        pred = model(inp_tokens, token_mask)
        loss = masked_loss(pred, tar_tokens, token_mask)
        
        val_loss += loss.item()

    return val_loss / len(loader)


def masked_loss(y_pred, y_true, mask):
    loss = F.cross_entropy(y_pred.permute(0, 2, 1), y_true, reduction='none')
    mask = (mask == 0).to(loss.dtype)
    return torch.sum(loss * mask) / torch.sum(mask).to(loss.dtype)

def step_lr(step, d_model, warmup_steps=4000):
    # learning rate from the original attention paper
    arg1 = torch.tensor(1 / math.sqrt(step)) if step > 0 else torch.tensor(float('inf'))
    arg2 = torch.tensor(step * warmup_steps**-1.5)
    
    return 1 / math.sqrt(d_model) * torch.minimum(arg1, arg2)
    

def main(config_fn='settings.yaml'):
    cfg = read_cfg(config_fn)

    headlines_dir = cfg.get('headlines_dir')
    train_size = cfg.get('train_size', 0.9)

    max_length = cfg.get('max_sequence_length', 256)
    num_workers = cfg.get('num_dataloader_workers', 4)

    batch_size = cfg.get('batch_size', 32)
    epochs = cfg.get('epochs', 20)
    max_patience = cfg.get('patience', 5)

    log_interval = cfg.get('log_interval', 100)

    d_model = cfg.get('d_model', 256)
    d_ff = cfg.get('d_ff??', 1024)
    n_heads = cfg.get('n_heads', 8)
    n_layers = cfg.get('n_layers', 8)
    dropout = cfg.get('dropout', 0.2)

    model_path = Path(cfg.get('model_path', 'data/model.pt'))
    warmup_steps = cfg.get('warmup_steps', 4000)

    # create datasets
    filenames = sorted(Path(headlines_dir).glob('otsikot-*.txt'))
    train_fns, val_fns = train_test_split(filenames, train_size=train_size)
    print(f'{len(train_fns)} files for training, {len(val_fns)} files for validation')

    train_ds = HeadlineDataset(train_fns, max_length=max_length)
    val_ds = HeadlineDataset(val_fns, max_length=max_length)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    
    print(f'd_model={d_model}, d_ff={d_ff}, n_heads={n_heads}, n_layers={n_layers}, dropout={dropout}')
    model = TokenGenerator(
        vocab_size=train_ds.vocab_size(),
        max_sequence_length=max_length,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout)

    print(f'model has {model_size(model)/1e6:.1f}M params')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
            lambda step: step_lr(step, d_model, warmup_steps=warmup_steps))

    model_path.parent.mkdir(exist_ok=True, parents=True)

    patience = max_patience
    best_loss = float('inf')

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train(model, train_loader, optimizer, scheduler, log_interval)
        val_loss = validate(model, val_loader)

        dt = time.time() - t0
        print(f'epoch {epoch:3d}/{epochs} | {dt:.1f} s | training loss {train_loss:.4f} | validation loss {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience = max_patience
            torch.save(model.state_dict(), model_path)

        else:
            patience -= 1
            if patience <= 0:
                print('results not improving, stopping...')
                break

if __name__ == '__main__':
    main()
