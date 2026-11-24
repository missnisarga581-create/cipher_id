"""Dataset generation: English text -> random-key ciphertexts of 20-500 chars.

Leakage control
---------------
The corpus is a folder of documents (books / articles).  Whole DOCUMENTS are
held out for testing, so the test set contains writing the model has never
seen - this measures real generalisation, not memorisation of the corpus.

Length sampling
---------------
Lengths are drawn log-uniformly from 20-500, so short ciphertexts (the hard
case this project is about) are well represented in training and testing.
"""
import random
from pathlib import Path

import numpy as np

from .ciphers import CLASSES, clean, encrypt_random

MIN_LEN, MAX_LEN = 20, 500
BLOCK = 1500  # letters per block (must be > MAX_LEN)

LENGTH_BUCKETS = {
    "short (20-50)": (20, 50),
    "medium (51-200)": (51, 200),
    "long (201-500)": (201, 500),
}
GRID_LENGTHS = [20, 30, 40, 50, 75, 100, 150, 200, 300, 500]


def _documents(corpus_path: str):
    p = Path(corpus_path)
    files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
    docs = []
    for f in files:
        letters = clean(f.read_text(encoding="utf-8", errors="ignore"))
        if len(letters) >= BLOCK:
            docs.append((f.stem, letters))
    return docs


def _blocks(letters: str):
    return [letters[i:i + BLOCK] for i in range(0, len(letters) - BLOCK + 1, BLOCK)]


def load_blocks(corpus_path: str, seed: int = 42, test_fraction: float = 0.2):
    """Return (train_blocks, test_blocks), split by whole document."""
    docs = _documents(corpus_path)
    if len(docs) == 1:               # single file: fall back to block split
        blocks = _blocks(docs[0][1])
        if len(blocks) < 10:
            raise ValueError(
                f"Corpus too small ({len(docs[0][1])} letters). Use at least ~50,000 "
                "letters of English text.")
        rng = random.Random(seed)
        rng.shuffle(blocks)
        n_test = max(1, int(len(blocks) * test_fraction))
        return blocks[n_test:], blocks[:n_test]
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_test = max(1, int(len(docs) * test_fraction))
    test_docs, train_docs = docs[:n_test], docs[n_test:]
    train = [b for _, l in train_docs for b in _blocks(l)]
    test = [b for _, l in test_docs for b in _blocks(l)]
    return train, test


def sample_length(rng: random.Random) -> int:
    """Log-uniform length in [MIN_LEN, MAX_LEN]."""
    import math
    return int(round(math.exp(rng.uniform(math.log(MIN_LEN), math.log(MAX_LEN)))))


def generate(blocks, n_samples: int, seed: int, fixed_length: int = None):
    """Balanced set of (ciphertext, label)."""
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n_samples):
        label = i % len(CLASSES)                      # perfectly balanced
        length = fixed_length or sample_length(rng)
        block = rng.choice(blocks)
        start = rng.randint(0, len(block) - length)
        plain = block[start:start + length]
        texts.append(encrypt_random(plain, label, rng))
        labels.append(label)
    return texts, np.array(labels)


def bucket_of(length: int) -> str:
    for name, (lo, hi) in LENGTH_BUCKETS.items():
        if lo <= length <= hi:
            return name
    return "other"
