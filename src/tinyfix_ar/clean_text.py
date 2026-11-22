"""Writes held-out clean text for measuring false changes: python -m tinyfix_ar.clean_text --corpus artifacts/corpus --output artifacts/clean-text

The documents come from the corpus's held-out split, so no run trained on that corpus has seen
them. They are cut into sentence-sized rows in the hand-written eval format (typed = corrected),
alternating between a calibration file and a reporting file.
"""

import argparse
import csv
from pathlib import Path

from tinyfix_ar.corpus import Corpus
from tinyfix_ar.text import FOREIGN, NUMBER, SPECIAL_TOKENS, START
from tinyfix_ar.train import VALIDATION_SHARE

ROW_TOKENS = 40
RETYPED = {SPECIAL_TOKENS[NUMBER]: "1", SPECIAL_TOKENS[FOREIGN]: "x"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--words", type=int, default=120_000, help="Words across both files.")
    args = parser.parse_args()

    corpus = Corpus.load(args.corpus)
    _, held_out = corpus.split(VALIDATION_SHARE)
    rows, words = [], 0
    for index in held_out:
        tokens = [RETYPED.get(token, token) for token in (corpus.raw_tokens[i] for i in corpus.documents[index]) if token != SPECIAL_TOKENS[START]]
        for start in range(0, len(tokens), ROW_TOKENS):
            rows.append((corpus.sources[index], " ".join(tokens[start : start + ROW_TOKENS])))
            words += min(ROW_TOKENS, len(tokens) - start)
        if words >= args.words:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    for half, name in ((0, "calibration"), (1, "heldout")):
        with (args.output / f"clean_{name}.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, quotechar=None, escapechar="\\")
            writer.writerow(["id", "category", "typed", "corrected", "note"])
            for number, (source, text) in enumerate(rows[half::2]):
                writer.writerow([f"clean-{name}-{number:05d}", "clean", text, text, source])
    print(f"{len(rows):,} rows, {words:,} words from {args.corpus}")


if __name__ == "__main__":
    main()
