"""Classical cipher implementations used to generate the labelled dataset.

All functions work on plain A-Z text (see `clean`). Non-letters are dropped
because the classifier only ever sees the letter stream.
"""
import random
import string

ALPHABET = string.ascii_uppercase
CLASSES = ["caesar", "monoalphabetic", "vigenere"]  # label ids 0, 1, 2


def clean(text: str) -> str:
    """Keep letters only, upper-cased."""
    return "".join(ch for ch in text.upper() if ch in ALPHABET)


def caesar_encrypt(text: str, shift: int) -> str:
    return "".join(ALPHABET[(ord(c) - 65 + shift) % 26] for c in text)


def random_substitution_key(rng: random.Random) -> str:
    """Random permutation of the alphabet with no fixed point and which is
    not a plain Caesar shift (so the two classes are truly different)."""
    while True:
        perm = list(ALPHABET)
        rng.shuffle(perm)
        if any(p == a for p, a in zip(perm, ALPHABET)):
            continue
        shifts = {(ord(p) - ord(a)) % 26 for p, a in zip(perm, ALPHABET)}
        if len(shifts) > 1:
            return "".join(perm)


def mono_encrypt(text: str, key: str) -> str:
    table = {a: k for a, k in zip(ALPHABET, key)}
    return "".join(table[c] for c in text)


def vigenere_encrypt(text: str, key: str) -> str:
    k = [ord(c) - 65 for c in key]
    return "".join(ALPHABET[(ord(c) - 65 + k[i % len(k)]) % 26] for i, c in enumerate(text))


def random_vigenere_key(rng: random.Random, min_len: int = 2, max_len: int = 12) -> str:
    """Key of length >= 2 (length 1 would just be a Caesar cipher)."""
    while True:
        key = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(min_len, max_len)))
        if len(set(key)) > 1:          # all-same-letter key is also just Caesar
            return key


def encrypt_random(text: str, label: int, rng: random.Random) -> str:
    """Encrypt `text` with the cipher given by `label` using a random key."""
    if label == 0:
        return caesar_encrypt(text, rng.randint(1, 25))
    if label == 1:
        return mono_encrypt(text, random_substitution_key(rng))
    if label == 2:
        return vigenere_encrypt(text, random_vigenere_key(rng))
    raise ValueError(f"unknown label {label}")
