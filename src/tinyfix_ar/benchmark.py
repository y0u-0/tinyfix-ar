"""Scores a trained run on real writing: python -m tinyfix_ar.benchmark --run runs/v1

Thresholds are calibrated on QALB-2014 dev and held-out clean text, then held fixed for every other
set. A valid typed word gets one threshold per kind of edit, set so false changes stay within
--false-rate-target on both. A typed word that is a known misspelling (ان, الى) is wrong whatever it
is swapped for, so its threshold only has to keep --misspelled-precision of those fixes picking the
right word. A word is in scope when the writer typed a word in the vocabulary that has candidates;
other mistakes are left to the spell checker.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinyfix_ar.engine import Fixer, Judgement, Thresholds
from tinyfix_ar.lexicon import is_target
from tinyfix_ar.qalb import AlignedSentence, align, aligned_sentences

REAL_SETS = ("qalb14-dev", "qalb14-test", "qalb15-test-l1", "zaebuc-dev", "zaebuc-test", "wiki-dev", "wiki-test")
CALIBRATION_SET = "qalb14-dev"
DEV_SETS = sorted(Path("data/dev").glob("*.tsv"))
CLEAN_TEXT = Path("artifacts/clean-text")
# Each change the budget fill adds must be right at least twice as often as it is wrong.
FILL_PRECISION = 2 / 3
ANALYSIS_SETS = ("qalb14-dev", "zaebuc-dev", "wiki-dev", *(path.stem for path in DEV_SETS))


@dataclass(frozen=True)
class Outcome:
    """A judged word: a real mistake (intended is the gold word) or a word the annotator left alone."""

    judgement: Judgement
    is_error: bool
    intended: str
    category: str
    reachable: bool

    @property
    def right(self) -> bool:
        return self.is_error and self.judgement.best == self.intended


def handwritten_sentences(path: Path) -> list[tuple[AlignedSentence, str]]:
    with path.open(encoding="utf-8") as rows:
        return [(align(row["typed"], row["corrected"]), row["category"]) for row in csv.DictReader(rows, delimiter="\t", quoting=csv.QUOTE_NONE)]


def outcomes(fixer: Fixer, sentences: list[tuple[AlignedSentence, str | None]]) -> tuple[list[Outcome], Counter]:
    """Judges every in-scope word: real mistakes, and words the annotator left alone.

    A gold correction to a known misspelling (QALB annotators sometimes wrote الإحتلال for
    الاحتلال), and a known misspelling the annotator left alone, are annotation errors, so those
    words are left out of scoring entirely.
    """
    lexicon = fixer.lexicon
    results, scope = [], Counter()
    for sentence, category in sentences:
        errors = {}
        for substitution in sentence.substitutions:
            intended_id = lexicon.ids.get(substitution.intended)
            if intended_id is not None and not lexicon.valid[intended_id] and is_target(substitution.intended):
                scope["annotation_noise"] += 1
            else:
                errors[substitution.position] = substitution
        scope["errors"] += len(errors)
        scope["untouched"] += len(sentence.untouched)
        for judgement in fixer.judge(sentence.typed):
            if judgement.position in errors:
                substitution = errors[judgement.position]
                candidates, _ = lexicon.candidates(lexicon.ids[judgement.typed])
                reachable = lexicon.ids.get(substitution.intended, -1) in candidates[1:]
                results.append(Outcome(judgement, True, substitution.intended, category or substitution.kind.value, reachable))
            elif judgement.position in sentence.untouched and judgement.misspelled:
                scope["annotation_missed"] += 1
            elif judgement.position in sentence.untouched:
                results.append(Outcome(judgement, False, judgement.typed, category or "clean", False))
    return results, scope


def summarize(results: list[Outcome], scope: Counter, thresholds: Thresholds) -> dict:
    errors = [outcome for outcome in results if outcome.is_error]
    fixes = [outcome for outcome in results if thresholds.changes(outcome.judgement)]
    right = [outcome for outcome in fixes if outcome.right]
    false_changes = sum(not outcome.is_error for outcome in fixes)
    by_kind = defaultdict(lambda: [0, 0])
    for outcome in errors:
        by_kind[outcome.category][0] += 1
        by_kind[outcome.category][1] += thresholds.changes(outcome.judgement) and outcome.right
    return {
        "errors": scope["errors"],
        "annotation_noise_skipped": scope["annotation_noise"] + scope["annotation_missed"],
        "errors_in_scope": len(errors),
        "untouched_in_scope": sum(not outcome.is_error for outcome in results),
        "fixed": round(len(right) / max(scope["errors"], 1), 4),
        "fixed_in_scope": round(len(right) / max(len(errors), 1), 4),
        "candidate_coverage": round(sum(outcome.reachable for outcome in errors) / max(len(errors), 1), 4),
        "fixes_right": round(len(right) / max(len(fixes), 1), 4),
        "false_changes_per_1000_words": round(1000 * false_changes / max(scope["untouched"], 1), 3),
        "by_kind": {kind: {"cases": total, "fixed": round(fixed / total, 3)} for kind, (total, fixed) in sorted(by_kind.items(), key=lambda item: -item[1][0])},
    }


def failures(results: list[Outcome], thresholds: Thresholds, top: int = 60) -> dict:
    """The most common mistakes left unfixed, fixed wrongly, and correct words changed."""
    missed, wrong, false_changes = Counter(), Counter(), Counter()
    for outcome in results:
        judgement, changed = outcome.judgement, thresholds.changes(outcome.judgement)
        if outcome.is_error and not changed:
            missed[f"{judgement.typed}->{outcome.intended}{'' if outcome.reachable else ' (not a candidate)'}"] += 1
        elif outcome.is_error and not outcome.right:
            wrong[f"{judgement.typed}->{judgement.best} (meant {outcome.intended})"] += 1
        elif not outcome.is_error and changed:
            false_changes[f"{judgement.typed}->{judgement.best}"] += 1
    return {name: counter.most_common(top) for name, counter in (("missed", missed), ("wrong_fix", wrong), ("false_change", false_changes))}


def _best_prefix(outcomes: list[Outcome], penalty: float) -> tuple[float, int, int]:
    """Threshold maximizing right fixes minus penalty times bad changes; returns it with its right and false counts."""
    ordered = sorted(outcomes, key=lambda outcome: -outcome.judgement.margin)
    right = np.cumsum([outcome.right for outcome in ordered])
    false = np.cumsum([not outcome.is_error for outcome in ordered])
    bad = np.arange(1, len(ordered) + 1) - right
    gain = right - penalty * bad
    best = int(np.argmax(gain))
    if gain[best] <= 0:
        return float("inf"), 0, 0
    return ordered[best].judgement.margin, int(right[best]), int(false[best])


def _false_changes(results: list[Outcome], thresholds: Thresholds) -> int:
    return sum(not outcome.is_error and thresholds.changes(outcome.judgement) for outcome in results)


def _count_at_least(descending: np.ndarray, thresholds: np.ndarray | float) -> np.ndarray:
    return np.searchsorted(-descending, -np.asarray(thresholds), side="right")


def _fill_budget(tuned: dict[str, list[Outcome]], valid: dict[str, float], budgets: list[tuple[list[Outcome], int]], min_precision: float) -> dict[str, float]:
    """Lowers one kind's threshold at a time into the false-change budget the shared-penalty sweep left unspent.

    The sweep moves every kind together, so one large step in a single kind can stop it well short
    of the budget. Each step here takes the lowering that adds the most right fixes net of bad
    changes, among those where at least min_precision of the added changes are right.
    """
    valid = dict(valid)
    false_margins = [
        {
            kind: np.sort([o.judgement.margin for o in results if not o.is_error and not o.judgement.misspelled and o.judgement.kind == kind])[::-1]
            for kind in tuned
        }
        for results, _ in budgets
    ]
    tables = {}
    for kind, outcomes in tuned.items():
        ordered = sorted(outcomes, key=lambda outcome: -outcome.judgement.margin)
        margins = np.array([outcome.judgement.margin for outcome in ordered])
        tables[kind] = (margins, np.cumsum([outcome.right for outcome in ordered]))
    while True:
        thresholds = Thresholds(0.0, valid)
        spent = [_false_changes(results, thresholds) for results, _ in budgets]
        best_score, best_step = 0, None
        for kind, (margins, rights) in tables.items():
            start = int(_count_at_least(margins, valid[kind]))
            ends = np.flatnonzero(np.r_[margins[1:] != margins[:-1], True]) + 1
            ends = ends[ends > start]
            if not len(ends):
                continue
            levels = margins[ends - 1]
            added = ends - start
            right = rights[ends - 1] - (rights[start - 1] if start else 0)
            allowed = right >= min_precision * added
            for (_, budget), used, margins_by_kind in zip(budgets, spent, false_margins, strict=True):
                extra = _count_at_least(margins_by_kind[kind], levels) - _count_at_least(margins_by_kind[kind], valid[kind])
                allowed &= used + extra <= budget
            score = np.where(allowed, 2 * right - added, 0)
            pick = int(np.argmax(score))
            if score[pick] > best_score:
                best_score, best_step = int(score[pick]), (kind, float(levels[pick]))
        if best_step is None:
            return valid
        valid[best_step[0]] = best_step[1]


def calibrate(
    judged: tuple[list[Outcome], Counter], guards: list[tuple[list[Outcome], Counter]], false_rate: float, misspelled_precision: float, min_cases: int = 40
) -> Thresholds:
    """Per-kind thresholds for valid typed words that fix the most mistakes in judged within the false-change budget.

    Each guard set (clean text) must also stay within the budget, so thresholds tuned on writers who
    make many mistakes still leave correct text alone.
    """
    results = judged[0]
    valid = [outcome for outcome in results if not outcome.judgement.misspelled]
    budgets = [(guard_results, int(false_rate * guard_scope["untouched"])) for guard_results, guard_scope in [judged, *guards]]

    def default_for(outcomes: list[Outcome], allowed: int) -> float:
        margins = sorted((outcome.judgement.margin for outcome in outcomes if not outcome.is_error and not outcome.judgement.misspelled), reverse=True)
        return margins[allowed] + 1e-6 if allowed < len(margins) else float("-inf")

    default = max(default_for(outcomes, allowed) for outcomes, allowed in budgets)
    by_kind = defaultdict(list)
    for outcome in valid:
        by_kind[outcome.judgement.kind].append(outcome)
    tuned = {kind: outcomes for kind, outcomes in by_kind.items() if len(outcomes) >= min_cases}
    best_right, best_valid = -1, {}
    for penalty in np.geomspace(0.05, 100, 80):
        chosen = {kind: _best_prefix(outcomes, penalty) for kind, outcomes in tuned.items()}
        candidate = Thresholds(0.0, {kind: result[0] for kind, result in chosen.items()} | {"default": default})
        right = sum(result[1] for result in chosen.values())
        if right > best_right and all(_false_changes(outcomes, candidate) <= allowed for outcomes, allowed in budgets):
            best_right, best_valid = right, candidate.valid
    if best_valid:
        best_valid = _fill_budget(tuned, best_valid, budgets, FILL_PRECISION)

    misspelled = sorted((outcome for outcome in results if outcome.judgement.misspelled), key=lambda outcome: -outcome.judgement.margin)
    lowest, right = float("inf"), 0
    for count, outcome in enumerate(misspelled, start=1):
        right += outcome.right
        if right / count >= misspelled_precision:
            lowest = outcome.judgement.margin
    return Thresholds(lowest, best_valid or {"default": default})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--handwritten", type=Path, nargs="*", default=sorted(Path("data/eval").glob("*.tsv")))
    parser.add_argument("--onnx", type=Path, help="Score this exported model instead of the run's PyTorch weights.")
    parser.add_argument("--visible-right", type=int, default=2)
    parser.add_argument("--false-rate-target", type=float, default=1 / 2000)
    parser.add_argument("--misspelled-precision", type=float, default=0.95)
    parser.add_argument(
        "--clean-text",
        type=Path,
        default=CLEAN_TEXT if CLEAN_TEXT.exists() else None,
        help="Folder from tinyfix_ar.clean_text: calibration half guards the thresholds, held-out half is reported.",
    )
    args = parser.parse_args()

    fixer = Fixer.load(args.run, onnx=args.onnx, visible_right=args.visible_right)
    sets = {name: [(sentence, None) for sentence in aligned_sentences(name)] for name in REAL_SETS}
    handwritten = [*args.handwritten, *DEV_SETS]
    clean = [args.clean_text / "clean_calibration.tsv", args.clean_text / "clean_heldout.tsv"] if args.clean_text else []
    for path in [*handwritten, *clean]:
        sets[path.stem] = handwritten_sentences(path)
    judged = {name: outcomes(fixer, sentences) for name, sentences in sets.items()}

    guards = [judged["clean_calibration"]] if args.clean_text else []
    thresholds = calibrate(judged[CALIBRATION_SET], guards, args.false_rate_target, args.misspelled_precision)
    report = {
        "thresholds": thresholds.to_json(),
        "visible_right": args.visible_right,
        "onnx": str(args.onnx or ""),
        "sets": {name: summarize(*judged[name], thresholds) for name in judged},
    }
    variant = "onnx-" if args.onnx else ""
    (args.run / f"benchmark-{variant}right{args.visible_right}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    report_failures = {name: failures(results, thresholds) for name, (results, _) in judged.items() if name in ANALYSIS_SETS}
    (args.run / f"failures-{variant}right{args.visible_right}.json").write_text(json.dumps(report_failures, ensure_ascii=False, indent=1))

    valid = ", ".join(f"{kind} {value:.2f}" for kind, value in sorted(thresholds.valid.items()))
    print(f"thresholds on {CALIBRATION_SET}: known misspellings {thresholds.misspelled:.2f}; valid words by kind: {valid}")
    print(f"{'set':18} {'errors':>7} {'in scope':>8} {'coverage':>8} {'fixed':>7} {'in-scope':>8} {'right':>7} {'false/1k':>8}")
    for name, summary in report["sets"].items():
        print(
            f"{name:18} {summary['errors']:7} {summary['errors_in_scope']:8} {summary['candidate_coverage']:8.1%} {summary['fixed']:7.1%} "
            f"{summary['fixed_in_scope']:8.1%} {summary['fixes_right']:7.1%} {summary['false_changes_per_1000_words']:8.2f}"
        )


if __name__ == "__main__":
    main()
