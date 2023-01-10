
from pathlib import Path

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import torch




class HeadlineDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, max_length=512):

        # get files:
        self.headlines = []
        for fn in filenames:
            lines = Path(fn).read_text(encoding='utf8').splitlines()
            headlines = list(map(lambda hl: hl.split('|')[-1], lines))
            self.headlines.extend(headlines)

        # get tokenizer
        self.tokenizer = ByteLevelBPETokenizer('data/tokenizer-vocab.json', 'data/tokenizer-merges.txt')
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ('</s>', self.tokenizer.token_to_id('</s>')),
            ('<s>', self.tokenizer.token_to_id('<s>')),
        )

        if max_length is not None:
            self.tokenizer.enable_truncation(max_length=max_length)

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, item):
        return self.tokenizer.encode(self.headlines[item]).ids

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


def collate_fn(batch):
    batch_size = len(batch)
    lens = list(map(len, batch))

    seqs = torch.zeros(batch_size, max(lens), dtype=torch.int64)
    mask = -float('inf') * torch.ones_like(seqs, dtype=torch.float32)
    
    for k in range(batch_size):
        seqs[k, :lens[k]] = torch.tensor(batch[k], dtype=torch.int64)
        mask[k, :lens[k]] = 0.0

    return seqs, mask    

