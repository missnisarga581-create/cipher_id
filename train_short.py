"""Train a dedicated SHORT-ciphertext model (10-60 letters).

Short text is the hard case (mostly Monoalphabetic vs Vigenere), and a model trained
only on short samples, with many more of them, beats the general model there.
predict.py automatically uses this model for inputs of <= 60 letters.

    python train_short.py            # writes results/short_model.joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import cipher_id.dataset as D
from cipher_id.ciphers import CLASSES
from cipher_id.features import FEATURE_NAMES, build_bigram_logp, extract_matrix, set_bigram_logp

SHORT_MIN, SHORT_MAX = 10, 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--train-samples", type=int, default=36000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    train_blocks, test_blocks = D.load_blocks(a.corpus, seed=a.seed)
    logp = build_bigram_logp(train_blocks)
    set_bigram_logp(logp)
    D.MIN_LEN, D.MAX_LEN = SHORT_MIN, SHORT_MAX

    print("Generating short-text training data ...")
    t, y = D.generate(train_blocks, a.train_samples, seed=a.seed + 7)
    X = extract_matrix(t)
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                           random_state=a.seed).fit(X, y)

    res = {}
    for L in (10, 15, 20, 25, 30, 40, 50, 60):
        tt, yy = D.generate(test_blocks, 1500, seed=900 + L, fixed_length=L)
        res[str(L)] = round(float((model.predict(extract_matrix(tt)) == yy).mean()), 4)
    print("Short model accuracy by length:", res)

    out = Path(a.out)
    joblib.dump({"model": model, "classes": CLASSES, "features": FEATURE_NAMES,
                 "bigram_logp": logp, "max_len": SHORT_MAX}, out / "short_model.joblib")
    m = json.loads((out / "metrics.json").read_text())
    m["Short model (HGB, 10-60)"] = {"by_length": res}
    (out / "metrics.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
