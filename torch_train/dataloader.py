"""WikiText-2 raw -> GPT-2 tokens -> fixed-size causal LM blocks.

Tokenizes one time and caches int64 IDs locally. Each batch is [B, T+1],
allowing exact next-token targets via inputs=batch[:,:-1], labels=batch[:,1:].
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TOKENIZER_NAME = "openai-community/gpt2-medium"


class TokenBlockDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.ids = token_ids.long().contiguous()
        self.seq_len = seq_len
        # Disjoint blocks of T+1 tokens with a one-token overlap for labels.
        self.count = max(0, (self.ids.numel() - 1) // seq_len)
        if not self.count:
            raise ValueError("Dataset too short for selected seq_len")

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.count:
            raise IndexError(idx)
        start = idx * self.seq_len
        return self.ids[start:start + self.seq_len + 1]


def get_tokens(split: str, cache_dir: str = './data_cache', tokenizer=None):
    if split not in ('train', 'validation', 'test'):
        raise ValueError(f'Unsupported split {split!r}')
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    file = cache / f'wikitext2_raw_gpt2_{split}.pt'
    if file.exists():
        saved = torch.load(file, map_location='cpu', weights_only=True)
        return saved['ids']

    from datasets import load_dataset
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f'Downloading/tokenizing {DATASET_NAME}/{DATASET_CONFIG}: {split}', flush=True)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    # Keep article text and newline boundaries. Blank lines are not training examples.
    texts = [row['text'] for row in ds if row['text'].strip()]
    ids = []
    for start in range(0, len(texts), 256):
        chunk = '\n'.join(texts[start:start + 256]) + '\n'
        ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
    ids = torch.tensor(ids, dtype=torch.long)
    torch.save({'ids': ids}, file)
    print(f'Cached {ids.numel():,} tokens to {file}', flush=True)
    return ids


def make_dataloader(split, seq_len=128, batch_size=1, cache_dir='./data_cache',
                    shuffle=None, num_workers=0, max_examples=0, pin_memory=False):
    tokens = get_tokens(split, cache_dir)
    ds = TokenBlockDataset(tokens, seq_len)
    if max_examples:
        from torch.utils.data import Subset
        ds = Subset(ds, range(min(max_examples, len(ds))))
    if shuffle is None:
        shuffle = split == 'train'
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory, num_workers=num_workers, drop_last=False)
