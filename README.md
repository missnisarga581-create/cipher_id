# AI-Based Identification of Classical Ciphers

Phase 1 project – Cryptography and Network Security, Ramaiah Institute of Technology.

Identifies whether a ciphertext was produced by a **Caesar**, **Monoalphabetic**
or **Vigenère** cipher, *without decrypting it*, with a focus on **short ciphertext**
(20–500 characters).

## Pipeline (matches the slides)

```
Generate dataset -> Extract features -> Train classifier -> Classify ciphertext -> Evaluate
```

| Stage | File | What it does |
|---|---|---|
| Data | `cipher_id/ciphers.py`, `cipher_id/dataset.py` | Encrypts English text with all 3 ciphers using random keys, lengths 20–500. Train/test plaintext blocks are split *before* sampling (no leakage). |
| Features | `cipher_id/features.py` | IoC, entropy, chi-squared, sorted letter frequencies, repeated bigrams/trigrams, lag coincidences, best-shift score (84 features). |
| Models | `train.py` | IoC-only baseline (decision tree), **Random Forest** (main), small **ANN** (MLP). |
| Evaluate | `train.py` | Accuracy + confusion matrix overall and for short / medium / long text. |
| Classify | `predict.py` | Predict the cipher type of any ciphertext. |

### Key feature ideas
- **Vigenère** flattens frequencies -> IoC near 0.04, plus coincidence spikes at multiples of the key length.
- **Caesar vs Monoalphabetic** (both IoC ≈ 0.066): the **best-shift score** tries all 26 shifts; if one shift makes the text fit English (low chi-squared, large gap to the runner-up) it is Caesar, otherwise monoalphabetic.

## Run

```bash
pip install -r requirements.txt

# 1. (corpus already in data/corpus/)
# 2. Train + evaluate (writes results/)
python train.py

# 3. Classify a ciphertext
python predict.py "WKHFRPPLWWHHZLOOPHHWDJDLQ"
python predict.py --file cipher.txt
```

Options: `--corpus PATH  --train-samples 50000  --test-samples 6000  --seed 42  --out results`

Outputs in `results/`: `metrics.json`, `confusion_matrices.png`,
`accuracy_by_length.png`, `feature_importance.png`, `rf_model.joblib`, `ann_model.joblib`.

## Corpus
`data/corpus/` holds 518 letters-only documents (~13.7M letters): Project Gutenberg books
and Brown-corpus articles (rebuild with `tools/build_corpus.py`). **Whole documents are held
out for testing**, so test accuracy is on writing the model never saw. Add your own `.txt`
files to the folder to extend it.

## Results (50k train / 6k test, unseen documents)

| Model | Overall | Short (20-50) | Medium (51-200) | Long (201-500) |
|---|---|---|---|---|
| IoC-only baseline | 60.4% | 55.5% | 61.4% | 63.8% |
| Random Forest | 96.1% | 87.7% | 99.3% | 100% |
| ANN (MLP) | 96.3% | 88.2% | 99.4% | 100% |

Random Forest at fixed lengths: 20 -> 79.9%, 30 -> 88.0%, 50 -> 96.7%, 100+ -> >=99.7%.
Per-class recall: Caesar 99.9%, Monoalphabetic 95.5%, Vigenere 92.9%.
`python selftest.py` checks hand-written sentences (about 89% overall).

## Short ciphertext
`python train_short.py` trains a dedicated model on 10-60 letter samples; `predict.py` uses it
automatically for inputs of <= 60 letters. Accuracy by length: 10 -> 67%, 15 -> 75%, 20 -> 83%,
25 -> 86%, 30 -> 90%, 40 -> 93%, 50 -> 97%, 60 -> 98%. Below ~25 letters Monoalphabetic and
Vigenere are genuinely hard to separate (there is little statistical evidence), so read the
printed probabilities rather than just the label.

## What was fixed
The original corpus was licence text, so ordinary English was misclassified (a Caesar-shifted
sentence came out as Vigenere at 98%). Now: real prose, document-level split, log-uniform
lengths, and new features (bigram Caesar fit, column IoC/fit for key lengths 2-12, bigram IoC,
substitution hill-climb "mono_fit").

## Limitations (mention in the report)
- Synthetic, clean, uppercase A–Z text; no spaces/punctuation/case in the ciphertext.
- Vigenère keys are 2–12 letters; very short Vigenère vs monoalphabetic is the hardest case.
- Plaintext with unusually flat letter frequencies (pangrams) and ciphertext under ~30 letters can still be misclassified (mono vs Vigenere).
- Results depend on the corpus and are not directly comparable with other papers' datasets.

## Ideas for Phase 2
- Add Playfair / Atbash / Affine / transposition classes (as in Abd & Al-Janabi, 2019).
- Try an LSTM/GRU on raw ciphertext and compare with the feature-based models.
- Add noise (typos, missing letters) and non-English text for robustness tests.
