
from pathlib import Path
from tqdm import tqdm

import sys
import torch
import torch.nn.functional as F

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from common import read_yaml
from models import TokenGenerator

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def get_tokenizer(dir_path):
    dir_path = Path(dir_path)

    vocab_fn = str(dir_path / 'tokenizer-vocab.json')
    merges_fn = str(dir_path / 'tokenizer-merges.txt')

    tokenizer = ByteLevelBPETokenizer(vocab_fn, merges_fn)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ('</s>', tokenizer.token_to_id('</s>')),
        ('<s>', tokenizer.token_to_id('<s>')),
    )

    return tokenizer


@torch.no_grad()
def generate_headline(model, tokenizer, max_length=128, prompt=None):
    if prompt is None:
        prompt = ''

    tokens = tokenizer.encode(prompt).ids

    end_token = tokens[-1]
    new_token = -1
    k = 1
    tokens = torch.tensor(tokens[:-1], dtype=torch.int64)

    while new_token != end_token:
        if k < max_length:
            #mask = torch.zeros(1, k)
            preds = model(tokens.unsqueeze(0), mask=None)
            probs = F.softmax(preds[0, -1, :] * (0.5 * k + 0.5), dim=0)
            new_token = torch.multinomial(probs, 1)[0]
            #new_token = torch.argmax(preds[0, -1:, :])
        else:
            new_token = end_token

        tokens = torch.cat([tokens, new_token.unsqueeze(0)])
        k = tokens.size(0)

    return tokenizer.decode(tokens[1:-1].detach().numpy())


def main(config_fn='settings.yaml', prompt=None):
    cfg = read_yaml(config_fn)
    # tokenizer
    tokenizer = get_tokenizer(cfg.get('tokenizer_dir'))

    max_length = cfg.get('max_sequence_length', 128)

    d_model = cfg.get('d_model', 256)
    d_ff = cfg.get('d_ff', 1024)
    n_heads = cfg.get('n_heads', 8)
    n_layers = cfg.get('n_layers', 8)
    dropout = cfg.get('dropout', 0.2)
    
    #print(f'd_model={d_model}, d_ff={d_ff}, n_heads={n_heads}, n_layers={n_layers}, dropout={dropout}')

    model_path = cfg.get('model_path', 'data/model.pt')

    model = TokenGenerator(
        vocab_size=tokenizer.get_vocab_size(),
        max_sequence_length=max_length,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout)

    model = model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    headlines = []
    for _ in tqdm(range(20)):
        hl = generate_headline(model, tokenizer, prompt=prompt)
        headlines.append(hl)
    
    hl_fn = f'headlines-{prompt.lower().replace(" ", "_")}.txt' if prompt is not None else 'headlines.txt'
    Path(hl_fn).write_text('\n'.join(headlines))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = None

    main(prompt=prompt)

