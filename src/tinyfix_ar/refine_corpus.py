"""Drops documents that misspell too many words: python -m tinyfix_ar.refine_corpus --corpus artifacts/corpus --output artifacts/corpus-refined

The first pass in build_corpus only knows a few dozen telltale words. This pass checks every
document against all known misspellings, which are only known once a corpus exists.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tinyfix_ar.corpus import Corpus
from tinyfix_ar.lexicon import Curated, known_misspellings, mine_real_mistakes
from tinyfix_ar.text import is_arabic_word


def misspelling_density(corpus: Corpus, misspellings: set[str]) -> np.ndarray:
    is_misspelled = np.array([token in misspellings for token in corpus.raw_tokens])
    is_arabic = np.array([is_arabic_word(token) for token in corpus.raw_tokens])
    return np.array([is_misspelled[document].sum() / max(is_arabic[document].sum(), 1) for document in corpus.documents])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-density", type=float, default=0.005)
    parser.add_argument("--groups", type=Path, default=Path("data/confusions/groups.tsv"))
    parser.add_argument("--one-way", type=Path, default=Path("data/confusions/one_way.tsv"))
    parser.add_argument("--dialect", type=Path, default=Path("data/confusions/dialect_words.tsv"), help="Valid dialect spellings; skipped if missing.")
    parser.add_argument("--contexts", type=Path, default=Path("data/contexts"), help="Audited correct sentences; their words count as valid.")
    args = parser.parse_args()

    corpus = Corpus.load(args.corpus)
    counts = np.bincount(np.concatenate(corpus.documents), minlength=len(corpus.raw_tokens))
    misspellings = known_misspellings(
        corpus.raw_tokens,
        counts,
        mine_real_mistakes(),
        Curated.read(args.groups, args.one_way, args.dialect if args.dialect.exists() else None, args.contexts if args.contexts.exists() else None),
    )
    density = misspelling_density(corpus, misspellings)
    keep = density <= args.max_density

    refined = Corpus(
        [document for document, kept in zip(corpus.documents, keep, strict=True) if kept],
        corpus.raw_tokens,
        [source for source, kept in zip(corpus.sources, keep, strict=True) if kept],
    )
    refined.save(args.output)
    lengths = np.array([len(document) for document in corpus.documents])
    report = {"max_density": args.max_density, "misspellings": len(misspellings), "sources": {}}
    for source in sorted(set(corpus.sources)):
        mine = np.array(corpus.sources) == source
        report["sources"][source] = {
            "documents_kept": int((mine & keep).sum()),
            "documents_dropped": int((mine & ~keep).sum()),
            "tokens_kept": int(lengths[mine & keep].sum()),
            "misspelling_rate_before": round(float((density[mine] * lengths[mine]).sum() / lengths[mine].sum()), 5),
            "misspelling_rate_after": round(float((density[mine & keep] * lengths[mine & keep]).sum() / lengths[mine & keep].sum()), 5),
        }
    (args.output / "refine-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
