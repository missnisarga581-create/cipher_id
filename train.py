"""Train and evaluate the cipher identifiers.

Pipeline (Phase 1 architecture):
  generate dataset -> extract features -> train classifiers -> classify -> evaluate

Models compared on ONE common dataset and ONE evaluation method:
  1. IoC-only baseline   (decision tree on the IoC feature alone)
  2. Random Forest       (main model, all features)
  3. ANN                 (small MLP, all features)
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from cipher_id.ciphers import CLASSES
from cipher_id.dataset import GRID_LENGTHS, LENGTH_BUCKETS, bucket_of, generate, load_blocks
from cipher_id.features import (FEATURE_NAMES, build_bigram_logp, extract_matrix,
                                set_bigram_logp)

IOC_COL = FEATURE_NAMES.index("ioc")
LEN_COL = FEATURE_NAMES.index("length")
TEAL, ORANGE, NAVY = "#1F7A8C", "#F4A442", "#12263A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--train-samples", type=int, default=50000)
    ap.add_argument("--test-samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # 1. Generate dataset ------------------------------------------------
    print("[1/5] Generating dataset ...")
    train_blocks, test_blocks = load_blocks(args.corpus, seed=args.seed)
    # English bigram model learned from TRAIN text only, stored inside the saved model
    bigram_logp = build_bigram_logp(train_blocks)
    set_bigram_logp(bigram_logp)
    tr_txt, y_tr = generate(train_blocks, args.train_samples, seed=args.seed)
    te_txt, y_te = generate(test_blocks, args.test_samples, seed=args.seed + 1)
    print(f"      train={len(tr_txt)}  test={len(te_txt)}  "
          f"(train/test plaintext blocks: {len(train_blocks)}/{len(test_blocks)}, "
          "test = documents never seen in training)")

    # 2. Extract features ------------------------------------------------
    print("[2/5] Extracting features ...")
    X_tr, X_te = extract_matrix(tr_txt), extract_matrix(te_txt)
    print(f"      {X_tr.shape[1]} features per ciphertext")

    # fixed-length test sets -> clean accuracy-vs-length curve
    grid = {L: generate(test_blocks, 900, seed=1000 + L, fixed_length=L) for L in GRID_LENGTHS}
    grid_X = {L: (extract_matrix(t), y) for L, (t, y) in grid.items()}

    # 3. Train classifiers ----------------------------------------------
    print("[3/5] Training models ...")
    models = {
        "IoC-only baseline": DecisionTreeClassifier(max_depth=2, random_state=args.seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=1, n_jobs=-1, random_state=args.seed),
        "ANN (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                          early_stopping=True, random_state=args.seed)),
    }
    models["IoC-only baseline"].fit(X_tr[:, [IOC_COL]], y_tr)
    models["Random Forest"].fit(X_tr, y_tr)
    models["ANN (MLP)"].fit(X_tr, y_tr)

    def predict(name, X):
        if name == "IoC-only baseline":
            return models[name].predict(X[:, [IOC_COL]])
        return models[name].predict(X)

    # 4/5. Classify + evaluate ------------------------------------------
    print("[4/5] Evaluating ...")
    lengths = X_te[:, LEN_COL].astype(int)
    buckets = np.array([bucket_of(l) for l in lengths])
    results, preds = {}, {}
    for name in models:
        p = predict(name, X_te)
        preds[name] = p
        results[name] = {"overall": round(accuracy_score(y_te, p), 4)}
        for b in LENGTH_BUCKETS:
            m = buckets == b
            results[name][b] = round(accuracy_score(y_te[m], p[m]), 4)

    # accuracy on fixed lengths (this is the curve shown in the slides)
    for name in models:
        results[name]["by_length"] = {
            str(L): round(accuracy_score(y, predict(name, X)), 4) for L, (X, y) in grid_X.items()}

    # per-class recall for the main model (Caesar vs mono is the hard pair)
    cm_rf = confusion_matrix(y_te, preds["Random Forest"])
    results["Random Forest"]["per_class_recall"] = {
        c: round(float(cm_rf[i, i] / cm_rf[i].sum()), 4) for i, c in enumerate(CLASSES)}

    # Console table
    cols = ["overall"] + list(LENGTH_BUCKETS)
    print("\n" + "Model".ljust(20) + "".join(c.ljust(18) for c in cols))
    for name, r in results.items():
        print(name.ljust(20) + "".join(f"{r[c]*100:5.1f}%".ljust(18) for c in cols))
    print("\nAccuracy at fixed lengths:")
    print("Model".ljust(20) + "".join(str(L).ljust(8) for L in GRID_LENGTHS))
    for name, r in results.items():
        print(name.ljust(20) + "".join(f"{r['by_length'][str(L)]*100:5.1f}%".ljust(8) for L in GRID_LENGTHS))
    print("\nRandom Forest per-class recall:", results["Random Forest"]["per_class_recall"])

    # 5. Save artefacts --------------------------------------------------
    print("\n[5/5] Saving results ...")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    joblib.dump({"model": models["Random Forest"], "classes": CLASSES,
                 "features": FEATURE_NAMES, "bigram_logp": bigram_logp}, out / "rf_model.joblib")
    joblib.dump({"model": models["ANN (MLP)"], "classes": CLASSES,
                 "features": FEATURE_NAMES, "bigram_logp": bigram_logp}, out / "ann_model.joblib")

    plot_confusions(models.keys(), preds, y_te, out / "confusion_matrices.png")
    plot_accuracy_vs_length(results, out / "accuracy_by_length.png")
    plot_importance(models["Random Forest"], out / "feature_importance.png")
    print(f"      saved to {out}/  (metrics.json, *.png, rf_model.joblib, ann_model.joblib)")


def plot_confusions(names, preds, y, path):
    names = list(names)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.4))
    for ax, name in zip(np.atleast_1d(axes), names):
        cm = confusion_matrix(y, preds[name])
        ax.imshow(cm, cmap="GnBu")
        ax.set_title(name, color=NAVY, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(["Caesar", "Mono", "Vigenère"])
        ax.set_yticklabels(["Caesar", "Mono", "Vigenère"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_accuracy_vs_length(results, path):
    colors = {"IoC-only baseline": "#888888", "Random Forest": TEAL, "ANN (MLP)": ORANGE}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, r in results.items():
        xs = [int(k) for k in r["by_length"]]
        ax.plot(xs, list(r["by_length"].values()), "o-", label=name,
                color=colors.get(name), lw=2)
    ax.set_xscale("log")
    ax.set_xticks(GRID_LENGTHS); ax.set_xticklabels(GRID_LENGTHS)
    ax.set_xlabel("Ciphertext length (characters)")
    ax.set_ylabel("Accuracy (held-out documents)")
    ax.set_title("Accuracy vs ciphertext length", color=NAVY, fontweight="bold")
    ax.set_ylim(0.3, 1.02); ax.grid(alpha=.3); ax.legend()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(rf, path, top=15):
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh([FEATURE_NAMES[i] for i in idx], imp[idx], color=TEAL)
    ax.set_title("Random Forest: top features", color=NAVY, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
