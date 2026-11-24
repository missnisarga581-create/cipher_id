"""Classify an unknown ciphertext (without decrypting it).

Usage:
    python predict.py                        # interactive: asks for input repeatedly
    python predict.py "WKH TXLFN EURZQ ARB ..."
    python predict.py --file cipher.txt
    python predict.py --model results/ann_model.joblib "..."
    python predict.py --explain "..."        # show the evidence behind the decision

Spaces, punctuation and case are ignored: only the letter stream is analysed.
"""
import argparse

import joblib
import numpy as np

from cipher_id.ciphers import clean
from cipher_id.features import (extract_features, period_features, set_bigram_logp,
                                to_array, bigram_shift_scores, index_of_coincidence,
                                letter_counts)


def load(model_path: str):
    bundle = joblib.load(model_path)
    set_bigram_logp(bundle["bigram_logp"])      # same English model used in training
    return bundle


def classify(text: str, bundle) -> dict:
    """Return {'letters', 'ranked': [(class, prob)...]} for any ciphertext string."""
    letters = clean(text)
    probs = bundle["model"].predict_proba(extract_features(letters).reshape(1, -1))[0]
    ranked = sorted(zip(bundle["classes"], probs), key=lambda x: -x[1])
    return {"letters": letters, "ranked": ranked}


def explain(letters: str) -> list:
    a = to_array(letters)
    ioc = index_of_coincidence(letter_counts(letters))
    scores = [0] * 26
    from cipher_id.features import get_bigram_logp
    logp = get_bigram_logp()
    sc = np.array([logp[(a[:-1] - s) % 26, (a[1:] - s) % 26].mean() for s in range(26)])
    best = int(sc.argmax())
    c_ioc, _ = period_features(a)
    per = int(c_ioc.argmax()) + 2
    return [
        f"Index of coincidence : {ioc:.3f}   (English-like ~0.066, flat/Vigenere-like ~0.04)",
        f"Best single shift    : {best}  English-bigram score {sc[best]:.2f} "
        f"vs runner-up {np.sort(sc)[-2]:.2f}   (big gap => Caesar)",
        f"Best column period   : {per}  mean column IoC {c_ioc.max():.3f}   "
        f"(high => periodic Vigenere key)",
    ]


def run_one(raw, a, bundle, short_bundle):
    n = len(clean(raw))
    if n < 10:
        print("Need at least 10 letters.\n")
        return
    use = bundle
    if short_bundle is not None and n <= short_bundle["max_len"]:
        use = short_bundle
        print("(using the short-ciphertext model)")
    set_bigram_logp(use["bigram_logp"])
    res = classify(raw, use)
    letters, ranked = res["letters"], res["ranked"]

    print(f"Letters analysed: {len(letters)}")
    print(f"Prediction: {ranked[0][0].upper()}")
    for name, p in ranked:
        print(f"  {name:<15} {p*100:5.1f}%")
    if a.explain:
        print("Evidence:")
        for line in explain(letters):
            print("  " + line)
    if len(letters) < 50:
        print("Note: very short ciphertext - treat the result with caution.")
    if ranked[0][1] < 0.6:
        print("Note: low confidence - the top two classes are close.")
    print()


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="ciphertext string (omit for interactive mode)")
    ap.add_argument("--file", help="read ciphertext from a file")
    ap.add_argument("--model", default="results/rf_model.joblib")
    ap.add_argument("--explain", action="store_true", help="show the evidence")
    a = ap.parse_args()

    bundle = load(a.model)
    short_bundle = None
    short_path = os.path.join(os.path.dirname(a.model) or ".", "short_model.joblib")
    if a.model.endswith("rf_model.joblib") and os.path.exists(short_path):
        short_bundle = load(short_path)

    if a.file or a.text:
        raw = open(a.file, encoding="utf-8").read() if a.file else a.text
        run_one(raw, a, bundle, short_bundle)
        return

    # Interactive mode: keep asking until the user quits
    print("Cipher identifier - paste a ciphertext and press Enter.")
    print("Type 'q' to quit. Spaces, punctuation and case are ignored.\n")
    while True:
        try:
            raw = input("Enter ciphertext: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw.lower() in ("q", "quit", "exit"):
            break
        if not raw:
            continue
        run_one(raw, a, bundle, short_bundle)


if __name__ == "__main__":
    main()
