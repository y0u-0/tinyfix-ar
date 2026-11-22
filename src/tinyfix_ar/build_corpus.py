"""Builds the cleaned, tokenized training corpus: python -m tinyfix_ar.build_corpus --output artifacts/corpus"""

import argparse
import json
from pathlib import Path

from tinyfix_ar.corpus import SOURCES, Corpus, build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", action="append", default=[], metavar="SOURCE=TOKENS", help=f"Repeatable. Sources: {', '.join(SOURCES)}")
    parser.add_argument("--base", type=Path, help="An existing corpus to extend with these budgets.")
    parser.add_argument("--max-error-rate", type=float, default=0.02)
    parser.add_argument("--min-evidence", type=int, default=3)
    args = parser.parse_args()

    budgets = {name: int(float(tokens)) for name, tokens in (item.split("=", 1) for item in args.budget)}
    unknown = set(budgets) - set(SOURCES)
    if unknown:
        parser.error(f"unknown sources: {sorted(unknown)}")

    base = Corpus.load(args.base) if args.base else None
    corpus, report = build_corpus(budgets, args.max_error_rate, args.min_evidence, base)
    corpus.save(args.output)
    (args.output / "report.json").write_text(
        json.dumps(
            {"base": str(args.base or ""), "budgets": budgets, "max_error_rate": args.max_error_rate, "min_evidence": args.min_evidence, "sources": report},
            indent=2,
        )
    )
    print(f"saved {sum(len(document) for document in corpus.documents):,} tokens to {args.output}", flush=True)


if __name__ == "__main__":
    main()
