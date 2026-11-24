"""Statistical feature extraction from ciphertext (v2).

Feature groups
--------------
  * IoC, entropy, chi-squared vs English, distinct letters
  * sorted letter frequencies (shape of the distribution, key-independent)
  * repeated patterns: repeated bigrams / trigrams, lag coincidences, bigram IoC
  * Mono fit    : hill-climb a substitution key with the bigram model; how English can it get?
  * Caesar fit  : best of 26 shifts, scored by letter frequencies AND by an
                  English bigram model (much sharper on short text)
  * Vigenere fit: for every key length 2..12, the average column IoC and how well
                  each column fits English after its own best shift

The bigram model is learned from the TRAINING text (see train.py) and stored
inside the saved model, so predict.py uses exactly the same table.
"""
from collections import Counter

import numpy as np

# Standard English letter frequencies (A..Z), sums to ~1.
ENGLISH_FREQ = np.array([
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094,
    0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929,
    0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978, 0.02360, 0.00150,
    0.01974, 0.00074,
])
ENGLISH_FREQ = ENGLISH_FREQ / ENGLISH_FREQ.sum()

MAX_LAG = 16
MAX_PERIOD = 12

# MIC_MATRIX[i, s] = ENGLISH_FREQ[(i - s) % 26]: how likely cipher letter i is if the shift is s
_idx = (np.arange(26)[:, None] - np.arange(26)[None, :]) % 26
MIC_MATRIX = ENGLISH_FREQ[_idx]

# ---- learned English bigram model (set from training data) -----------------
_BIGRAM_LOGP = None


def build_bigram_logp(blocks) -> np.ndarray:
    """26x26 matrix of log P(next letter | letter) from English text blocks."""
    counts = np.ones((26, 26))                      # add-one smoothing
    for b in blocks:
        a = np.frombuffer(b.encode(), dtype=np.uint8).astype(int) - 65
        np.add.at(counts, (a[:-1], a[1:]), 1)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def set_bigram_logp(logp: np.ndarray):
    global _BIGRAM_LOGP
    _BIGRAM_LOGP = np.asarray(logp)


def get_bigram_logp() -> np.ndarray:
    if _BIGRAM_LOGP is None:
        raise RuntimeError("Bigram model not set: call set_bigram_logp() or load a trained model.")
    return _BIGRAM_LOGP


# ---- basic statistics ------------------------------------------------------
def to_array(text: str) -> np.ndarray:
    return np.frombuffer(text.encode(), dtype=np.uint8).astype(int) - 65


def letter_counts(text: str) -> np.ndarray:
    return np.bincount(to_array(text), minlength=26).astype(float)


def index_of_coincidence(counts: np.ndarray) -> float:
    n = counts.sum()
    if n < 2:
        return 0.0
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def entropy(freq: np.ndarray) -> float:
    p = freq[freq > 0]
    return float(-(p * np.log2(p)).sum())


def chi_squared(counts: np.ndarray, expected_freq: np.ndarray) -> float:
    n = counts.sum()
    expected = expected_freq * n
    return float(((counts - expected) ** 2 / np.maximum(expected, 1e-9)).sum())


def repeated_ngram_ratio(text: str, n: int) -> float:
    """Fraction of n-grams that occur more than once (repeated patterns)."""
    if len(text) < n + 1:
        return 0.0
    grams = Counter(text[i:i + n] for i in range(len(text) - n + 1))
    total = sum(grams.values())
    return sum(c for c in grams.values() if c > 1) / total


def bigram_ioc(a: np.ndarray) -> float:
    """IoC over letter PAIRS. Mono keeps English pair structure; Vigenere destroys it."""
    if len(a) < 3:
        return 0.0
    pairs = np.bincount(a[:-1] * 26 + a[1:], minlength=676).astype(float)
    n = pairs.sum()
    return float((pairs * (pairs - 1)).sum() / (n * (n - 1)))


def lag_coincidences(a: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Fraction of positions where letter[i] == letter[i+k]; spikes at multiples of
    the key length for Vigenere."""
    out = np.zeros(max_lag - 1)
    for k in range(2, max_lag + 1):
        if len(a) > k:
            out[k - 2] = float((a[:-k] == a[k:]).mean())
    return out


# ---- Caesar fit ------------------------------------------------------------
def best_shift_scores(counts: np.ndarray):
    """Try all 26 shifts. Returns (min chi2, gap to 2nd best, max mutual IoC).
    Caesar -> one shift fits English; Mono -> none does."""
    n = counts.sum()
    if n == 0:
        return 0.0, 0.0, 0.0
    chis = [chi_squared(np.roll(counts, -s), ENGLISH_FREQ) for s in range(26)]
    mic = float(((counts / n) @ MIC_MATRIX).max())
    cs = sorted(chis)
    return cs[0], cs[1] - cs[0], mic


def bigram_shift_scores(a: np.ndarray):
    """Mean English bigram log-likelihood under each of 26 shifts.
    Returns (best, gap to 2nd best, best - mean of the rest)."""
    if len(a) < 2:
        return 0.0, 0.0, 0.0
    logp = get_bigram_logp()
    sc = np.array([logp[(a[:-1] - s) % 26, (a[1:] - s) % 26].mean() for s in range(26)])
    order = np.sort(sc)[::-1]
    return float(order[0]), float(order[0] - order[1]), float(order[0] - order[1:].mean())


# ---- Monoalphabetic fit ----------------------------------------------------
_ER_ORDER = np.argsort(-ENGLISH_FREQ)           # English letters, most -> least frequent
_PAIRS = np.triu_indices(26, 1)


def mono_fit_score(a: np.ndarray, iters: int = 6) -> float:
    """How well can ONE substitution alphabet make this text look like English?

    Start from the frequency-rank guess (commonest cipher letter -> E, ...) and
    greedily swap letter pairs while the English bigram log-likelihood improves.
    A real monoalphabetic ciphertext climbs close to English; a Vigenere
    ciphertext (several alphabets mixed) cannot. Works even when the plaintext
    has a flat letter distribution, where plain IoC fails."""
    if len(a) < 3:
        return 0.0
    logp = get_bigram_logp()
    C = np.zeros((26, 26))
    np.add.at(C, (a[:-1], a[1:]), 1)
    cnt = np.bincount(a, minlength=26)
    key = np.empty(26, dtype=int)
    key[np.argsort(-cnt, kind="stable")] = _ER_ORDER
    pi, pj = _PAIRS
    cur = float((C * logp[np.ix_(key, key)]).sum())
    for _ in range(iters):
        keys = np.tile(key, (len(pi), 1))
        rows = np.arange(len(pi))
        keys[rows, pi], keys[rows, pj] = key[pj], key[pi]
        scores = (C[None] * logp[keys[:, :, None], keys[:, None, :]]).sum(axis=(1, 2))
        b = int(scores.argmax())
        if scores[b] <= cur + 1e-9:
            break
        cur, key = float(scores[b]), keys[b]
    return cur / (len(a) - 1)


# ---- Vigenere fit ----------------------------------------------------------
def period_features(a: np.ndarray):
    """For key length p = 2..MAX_PERIOD:
       col_ioc[p] : length-weighted mean IoC of the p columns
       col_fit[p] : length-weighted mean of (best-shift mutual IoC with English) per column
    A Vigenere ciphertext lights up at the true key length (and its multiples)."""
    n = len(a)
    col_ioc, col_fit = [], []
    idx = np.arange(n)
    for p in range(2, MAX_PERIOD + 1):
        counts = np.bincount((idx % p) * 26 + a, minlength=p * 26).reshape(p, 26).astype(float)
        sizes = counts.sum(axis=1)
        ok = sizes >= 2
        if not ok.any():
            col_ioc.append(0.0); col_fit.append(0.0)
            continue
        c, s = counts[ok], sizes[ok]
        ioc = (c * (c - 1)).sum(axis=1) / (s * (s - 1))
        fit = ((c / s[:, None]) @ MIC_MATRIX).max(axis=1)
        col_ioc.append(float((ioc * s).sum() / s.sum()))
        col_fit.append(float((fit * s).sum() / s.sum()))
    return np.array(col_ioc), np.array(col_fit)


FEATURE_NAMES = (
    ["length", "ioc", "entropy", "chi2_english", "distinct_letters"]
    + [f"sorted_freq_{i}" for i in range(26)]
    + ["rep_bigram", "rep_trigram", "bigram_ioc", "lag_max", "lag_mean"]
    + [f"lag_{k}" for k in range(2, MAX_LAG + 1)]
    + ["best_shift_chi2", "best_shift_gap", "best_shift_mic"]
    + ["bigram_best", "bigram_gap", "bigram_lead", "mono_fit", "mono_fit_vs_caesar"]
    + [f"col_ioc_{p}" for p in range(2, MAX_PERIOD + 1)]
    + [f"col_fit_{p}" for p in range(2, MAX_PERIOD + 1)]
    + ["col_ioc_max", "col_fit_max", "col_fit_gain"]
)


def extract_features(text: str) -> np.ndarray:
    a = to_array(text)
    counts = np.bincount(a, minlength=26).astype(float)
    n = max(counts.sum(), 1)
    freq = counts / n
    lags = lag_coincidences(a)
    ioc = index_of_coincidence(counts)
    bs_chi, bs_gap, bs_mic = best_shift_scores(counts)
    bg_best, bg_gap, bg_lead = bigram_shift_scores(a)
    mono_fit = mono_fit_score(a)
    c_ioc, c_fit = period_features(a)

    feats = [len(a), ioc, entropy(freq), chi_squared(counts, ENGLISH_FREQ),
             float((counts > 0).sum())]
    feats += sorted(freq, reverse=True)
    feats += [repeated_ngram_ratio(text, 2), repeated_ngram_ratio(text, 3),
              bigram_ioc(a), float(lags.max()), float(lags.mean())]
    feats += list(lags)
    feats += [bs_chi, bs_gap, bs_mic, bg_best, bg_gap, bg_lead, mono_fit, mono_fit - bg_best]
    feats += list(c_ioc) + list(c_fit)
    feats += [float(c_ioc.max()), float(c_fit.max()), float(c_fit.max() - bs_mic)]
    return np.array(feats, dtype=float)


def extract_matrix(texts) -> np.ndarray:
    return np.vstack([extract_features(t) for t in texts])
