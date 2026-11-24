"""Sanity check on hand-written sentences (NOT from the training corpus).

Encrypts each sentence with every cipher (random keys, several seeds) and reports how
often predict.py gets it right, per cipher and per sentence length.
    python selftest.py
"""
import random
from collections import defaultdict

from cipher_id.ciphers import CLASSES, clean, encrypt_random
from predict import classify, load

SENTENCES = [
    "The quick brown fox jumps over the lazy dog and runs away from the farmer",
    "We will meet at the main gate of the college tomorrow before the first lecture begins",
    "Please send me the report by Friday evening so that I can review it over the weekend",
    "Security of a system depends on the weakest link and not on the strongest lock",
    "It was a bright cold day in April and the clocks were striking thirteen",
    "Attack the northern wall at dawn and hold the bridge until reinforcements arrive",
    "My professor said that the examination will cover cryptography and network security",
    "The train to Bengaluru leaves at six in the morning from platform number four",
    "Meet me",
    "Hello how are you doing today my friend",
    "Cryptanalysis is the art of reading messages that were never meant for you",
    "The committee will announce its final decision after reviewing all of the proposals submitted this year",
]


def main(trials: int = 40):
    bundle = load("results/rf_model.joblib")
    short = load("results/short_model.joblib")
    load("results/rf_model.joblib")
    rng = random.Random(7)
    by_cls, by_len = defaultdict(list), defaultdict(list)
    for s in SENTENCES:
        plain = clean(s)
        for label, cls in enumerate(CLASSES):
            for _ in range(trials):
                c = encrypt_random(plain, label, rng)
                m = short if len(c) <= short["max_len"] else bundle
                ok = classify(c, m)["ranked"][0][0] == cls
                by_cls[cls].append(ok)
                by_len[len(plain)].append(ok)
    print("Accuracy per cipher:")
    for c in CLASSES:
        print(f"  {c:<15} {100*sum(by_cls[c])/len(by_cls[c]):5.1f}%")
    print("Accuracy per sentence length (letters):")
    for L in sorted(by_len):
        print(f"  {L:>4}  {100*sum(by_len[L])/len(by_len[L]):5.1f}%")
    allr = [x for v in by_cls.values() for x in v]
    print(f"Overall: {100*sum(allr)/len(allr):.1f}%")


if __name__ == "__main__":
    main()
