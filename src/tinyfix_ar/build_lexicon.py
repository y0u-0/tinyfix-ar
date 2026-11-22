"""Builds the vocabulary and candidate tables: python -m tinyfix_ar.build_lexicon --corpus artifacts/corpus --output artifacts/lexicon"""

import argparse
import json
from pathlib import Path

import numpy as np

from tinyfix_ar.corpus import Corpus
from tinyfix_ar.lexicon import MAX_NEIGHBORS, Curated, build_lexicon, is_target, mine_real_mistakes
from tinyfix_ar.text import PAD, UNK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--groups", type=Path, default=Path("data/confusions/groups.tsv"))
    parser.add_argument("--one-way", type=Path, default=Path("data/confusions/one_way.tsv"))
    parser.add_argument("--dialect", type=Path, default=Path("data/confusions/dialect_words.tsv"), help="Valid dialect spellings; skipped if missing.")
    parser.add_argument("--contexts", type=Path, default=Path("data/contexts"), help="Audited correct sentences; their words count as valid.")
    parser.add_argument("--min-real-count", type=int, default=2)
    args = parser.parse_args()

    corpus = Corpus.load(args.corpus)
    counts = np.bincount(np.concatenate(corpus.documents), minlength=len(corpus.raw_tokens))
    real = mine_real_mistakes()
    curated = Curated.read(args.groups, args.one_way, args.dialect if args.dialect.exists() else None, args.contexts if args.contexts.exists() else None)
    lexicon = build_lexicon(corpus.raw_tokens, counts, args.size, real, curated, args.min_real_count)
    real_mistakes = real.frequent(args.min_real_count)
    lexicon.save(args.output)

    lookup = lexicon.lookup_table(corpus.raw_tokens)
    mapped = lookup[np.concatenate(corpus.documents)]
    has_neighbors = (lexicon.neighbors[:, 0] != PAD).sum()
    report = {
        "vocabulary": len(lexicon.words),
        "unknown_token_rate": round(float((mapped == UNK).mean()), 4),
        "words_with_neighbors": int(has_neighbors),
        "mean_neighbors": round(float((lexicon.neighbors != PAD).sum(1)[lexicon.neighbors[:, 0] != PAD].mean()), 2),
        "full_neighbor_lists": int(((lexicon.neighbors != PAD).sum(1) == MAX_NEIGHBORS).sum()),
        "real_mistake_pairs": len(real_mistakes),
        "real_mistake_pairs_in_vocabulary": sum(typed in lexicon.ids and intended in lexicon.ids for typed, intended in real_mistakes),
        "curated_groups": len(curated.groups),
        "curated_one_way": len(curated.one_way),
        "never_suggested": int(sum(1 for index, word in enumerate(lexicon.words) if is_target(word) and not lexicon.valid[index])),
        "typed_words_with_confusions": len(lexicon.confusions),
        "top_real_mistakes": [f"{typed}->{intended} {count}" for (typed, intended), count in real_mistakes.most_common(15)],
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
