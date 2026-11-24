"""Run the identifier on the predefined dataset (no typing needed).

Takes plaintext from data/corpus/ (documents the models never saw in training),
encrypts it with random Caesar / Monoalphabetic / Vigenere keys, classifies every
ciphertext and prints true vs predicted plus accuracy.

Usage:
    python predict_dataset.py                    # 30 samples, random lengths 20-500
    python predict_dataset.py -n 100 --length 40 # 100 samples, all 40 letters long
    python predict_dataset.py --show 0           # only the summary, no per-sample rows
    python predict_dataset.py --model results/ann_model.joblib
"""
import argparse
import os
from collections import defaultdict

import joblib
import numpy as np

from cipher_id.ciphers import CLASSES
from cipher_id.dataset import bucket_of, generate, load_blocks
from cipher_id.features import extract_features, set_bigram_logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--model", default="results/rf_model.joblib")
    ap.add_argument("-n", "--samples", type=int, default=30)
    ap.add_argument("--length", type=int, help="fixed ciphertext length (default: random 20-500)")
    ap.add_argument("--seed", type=int, default=7, help="change for a different random sample")
    ap.add_argument("--show", type=int, default=30, help="how many sample rows to print")
    a = ap.parse_args()

    # same document split as train.py (seed 42) -> test blocks are unseen documents
    _, test_blocks = load_blocks(a.corpus, seed=42)
    texts, y = generate(test_blocks, a.samples, seed=a.seed, fixed_length=a.length)

    main_b = joblib.load(a.model)
    short_b = None
    sp = os.path.join(os.path.dirname(a.model) or ".", "short_model.joblib")
    if a.model.endswith("rf_model.joblib") and os.path.exists(sp):
        short_b = joblib.load(sp)

    ok = 0
    by_class = defaultdict(lambda: [0, 0])
    by_bucket = defaultdict(lambda: [0, 0])
    print(f"{'#':>3} {'len':>4}  {'true':<15} {'predicted':<15} {'conf':>6}  ciphertext")
    for i, (t, label) in enumerate(zip(texts, y)):
        b = short_b if (short_b is not None and len(t) <= short_b["max_len"]) else main_b
        set_bigram_logp(b["bigram_logp"])
        probs = b["model"].predict_proba(extract_features(t).reshape(1, -1))[0]
        k = int(np.argmax(probs))
        pred, true = b["classes"][k], CLASSES[label]
        hit = pred == true
        ok += hit
        by_class[true][0] += hit; by_class[true][1] += 1
        bk = bucket_of(len(t))
        by_bucket[bk][0] += hit; by_bucket[bk][1] += 1
        if i < a.show:
            snippet = t[:28] + ("..." if len(t) > 28 else "")
            print(f"{i+1:>3} {len(t):>4}  {true:<15} {pred:<15} {probs[k]*100:5.1f}%  "
                  f"{snippet} {'' if hit else '<-- wrong'}")

    n = len(texts)
    print(f"\nAccuracy: {ok}/{n} = {ok/n*100:.1f}%")
    for c in CLASSES:
        h, t = by_class[c]
        if t:
            print(f"  {c:<15} {h}/{t}  ({h/t*100:.0f}%)")
    if a.length is None:
        print("By length:")
        for bk, (h, t) in by_bucket.items():
            print(f"  {bk:<16} {h}/{t}  ({h/t*100:.0f}%)")


if __name__ == "__main__":
    main()
