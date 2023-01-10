
import math

import torch
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_sequence_length, d_model, dropout=0.1):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) 
        pe = torch.zeros(1, max_sequence_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenGenerator(torch.nn.Module):
    def __init__(self, vocab_size, max_sequence_length, d_model, d_ff, n_heads, n_layers, dropout=0.1):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_sequence_length, d_model)

        # lookback mask
        mask = torch.empty(max_sequence_length, max_sequence_length).fill_(-float('inf')).triu(1)
        self.register_buffer('mask', mask.to(torch.bool), persistent=False)

        enc_layer = torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(enc_layer, n_layers)
        self.layernorm = torch.nn.LayerNorm(d_model)

    def forward(self, tokens, mask):
        batch_size = tokens.size(0)
        seq_len = tokens.size(-1)

        x = self.embedding(tokens)
        x = self.positional_encoding(x)
        
        x = self.encoder(x, mask=self.mask[:seq_len, :seq_len], src_key_padding_mask=mask)

        x = self.layernorm(x)
        logits = x @ torch.transpose(self.embedding.weight, 0, 1)

        return logits


