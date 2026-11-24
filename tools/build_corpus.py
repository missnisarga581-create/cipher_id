"""Build data/corpus/ (one letters-only file per document) from real English text.

Sources (both free, from the NLTK data repo on GitHub):
  * Project Gutenberg sample  - novels, plays, stories (gutenberg.zip)
  * Brown corpus              - modern news / fiction / science / religion ... (brown.zip)

Usage:
    python tools/build_corpus.py --gutenberg path/to/gutenberg --brown path/to/brown
"""
import argparse
import re
from pathlib import Path

LETTERS = re.compile(r"[^A-Z]")


def letters_only(text: str) -> str:
    return LETTERS.sub("", text.upper())


def write_doc(out: Path, name: str, letters: str):
    if len(letters) < 3000:          # skip tiny files
        return 0
    lines = [letters[i:i + 100] for i in range(0, len(letters), 100)]
    (out / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    return len(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gutenberg", required=True)
    ap.add_argument("--brown", required=True)
    ap.add_argument("--out", default="data/corpus")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0

    for p in sorted(Path(a.gutenberg).glob("*.txt")):
        raw = p.read_text(encoding="latin-1")
        total += write_doc(out, "gut_" + p.stem.replace("-", "_"), letters_only(raw))

    for p in sorted(Path(a.brown).glob("c[a-r]??")):
        # Brown is tagged "word/TAG"; keep only the word part.
        words = [tok.rsplit("/", 1)[0] for tok in p.read_text(encoding="latin-1").split()]
        total += write_doc(out, "brown_" + p.name, letters_only(" ".join(words)))

    print(f"wrote {len(list(out.glob('*.txt')))} documents, {total:,} letters -> {out}")


if __name__ == "__main__":
    main()
