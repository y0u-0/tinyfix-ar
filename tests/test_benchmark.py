import math

from tinyfix_ar.benchmark import Outcome, _false_changes, _fill_budget
from tinyfix_ar.engine import Judgement, Thresholds


def outcome(margin: float, kind: str = "hamza", *, error: bool, right: bool = True) -> Outcome:
    best = "أن" if right else "آن"
    judgement = Judgement(position=0, typed="ان", best=best, margin=margin, kind=kind, misspelled=False)
    return Outcome(judgement, error, "أن" if error else "ان", "test", True)


def test_fill_lowers_a_threshold_into_unspent_budget():
    calibration = [outcome(margin, error=True) for margin in (5.0, 4.0, 3.0, 2.0)] + [outcome(1.0, error=False)]
    filled = _fill_budget({"hamza": calibration}, {"hamza": 4.5, "default": 9.0}, [(calibration, 1)], min_precision=2 / 3)
    assert filled["hamza"] == 2.0
    assert filled["default"] == 9.0


def test_fill_never_spends_more_than_any_budget():
    calibration = [outcome(margin, error=True) for margin in (5.0, 4.0, 3.0, 2.0)]
    guard = [outcome(3.5, error=False), outcome(1.0, error=False)]
    filled = _fill_budget({"hamza": calibration}, {"hamza": 4.5, "default": 9.0}, [(calibration, 0), (guard, 0)], min_precision=2 / 3)
    assert filled["hamza"] == 4.0
    assert _false_changes(guard, Thresholds(math.inf, filled)) == 0


def test_fill_skips_steps_that_are_wrong_too_often():
    calibration = [outcome(4.0, error=True), outcome(3.0, error=True, right=False), outcome(2.0, error=True), outcome(1.0, error=True)]
    filled = _fill_budget({"hamza": calibration}, {"hamza": 4.5, "default": 9.0}, [(calibration, 10)], min_precision=0.9)
    assert filled["hamza"] == 4.0
