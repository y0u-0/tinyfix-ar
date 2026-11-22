from tinyfix_ar.engine import Fixer
from tinyfix_ar.model import LEFT_WINDOW, RIGHT_WINDOW
from tinyfix_ar.text import PAD, START


def fixer_with(visible_right: int) -> Fixer:
    fixer = Fixer.__new__(Fixer)
    fixer.visible_right = visible_right
    return fixer


def test_context_matches_training_layout():
    left, right = fixer_with(visible_right=2).context([10, 11, 12, 13, 14], position=1)
    assert left == [PAD] * (LEFT_WINDOW - 2) + [START, 10]
    assert right == [PAD] * (RIGHT_WINDOW - 2) + [13, 12]


def test_context_at_end_of_text_has_no_right_words():
    left, right = fixer_with(visible_right=4).context([10, 11], position=1)
    assert left[-2:] == [START, 10]
    assert right == [PAD] * RIGHT_WINDOW
