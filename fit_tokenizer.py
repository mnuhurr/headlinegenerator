
from pathlib import Path
import tokenizers

from common import read_yaml

def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)

    filenames = Path(cfg.get('headlines_dir')).glob('otsikot-*.txt')
    filenames = sorted(map(lambda fn: str(fn), filenames))

    vocab_size = cfg.get('vocab_size', 30000)
    min_frequency = 2
    special_tokens = ['<s>', '</s>', '<pad>', '<unk>', '<mask>']
    
    tokenizer = tokenizers.ByteLevelBPETokenizer()

    tokenizer.train(files=filenames, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

    tokenizer.save_model(cfg.get('tokenizer_dir', '.'), 'tokenizer')

if __name__ == '__main__':
    main()
