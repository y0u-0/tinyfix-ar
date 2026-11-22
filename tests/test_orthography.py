import pytest

from tinyfix_ar.orthography import ErrorKind, NeighborIndex, classify, same_skeleton, within_one_edit


@pytest.mark.parametrize(
    ("typed", "intended", "kind"),
    [
        ("ان", "أن", ErrorKind.HAMZA),
        ("مسئول", "مسؤول", ErrorKind.HAMZA),
        ("مسوول", "مسؤول", ErrorKind.HAMZA),
        ("شي", "شيء", ErrorKind.HAMZA),
        ("كتابه", "كتابة", ErrorKind.TAA_MARBUTA),
        ("علي", "على", ErrorKind.ALIF_MAQSURA),
        ("كتبو", "كتبوا", ErrorKind.ALIF_FARIQA),
        ("الي", "إلى", ErrorKind.SPELLING_MIXED),
        ("ضل", "ظل", ErrorKind.SOUND_ALIKE),
        ("هدا", "هذا", ErrorKind.SOUND_ALIKE),
        ("سورة", "صورة", ErrorKind.SOUND_ALIKE),
        ("ذل", "ظل", ErrorKind.SOUND_ALIKE),
        ("رحمت", "رحمة", ErrorKind.TAA_MARBUTA),
        ("مرت", "مرة", ErrorKind.TAA_MARBUTA),
        ("عصا", "عصى", ErrorKind.ALIF_MAQSURA),
        ("لا", "ولا", ErrorKind.CLITIC),
        ("قلب", "قلت", ErrorKind.TYPO),
        ("هيك", "هكذا", ErrorKind.OTHER),
    ],
)
def test_classify(typed, intended, kind):
    assert classify(typed, intended) is kind


def test_classify_rejects_identical_words():
    with pytest.raises(ValueError):
        classify("في", "في")


def test_same_skeleton_does_not_merge_different_letters():
    assert same_skeleton("أمام", "إمام")
    assert not same_skeleton("كتب", "كتف")


def test_within_one_edit_handles_adjacent_swaps_only():
    assert within_one_edit("قلب", "قبل")
    assert not within_one_edit("كتاب", "باتك")
    assert not within_one_edit("في", "في")


def test_neighbors_put_spelling_variants_before_typos_and_skip_self():
    index = NeighborIndex(["من", "أن", "إن", "عن", "آن", "ان"])
    assert index.neighbors("ان", limit=8) == ["أن", "إن", "آن", "من", "عن"]


def test_neighbors_put_known_mixups_first_but_only_if_suggestable():
    index = NeighborIndex(["على", "علي", "إلى"])
    assert index.neighbors("علي", limit=2, known_mixups=["إلى", "غير-موجود"]) == ["إلى", "على"]


def test_typos_leave_out_clitic_swaps():
    index = NeighborIndex(["لا", "ولا", "ما"])
    assert index.typos("لا") == ["ما"]


def test_neighbors_can_skip_typos_for_known_misspellings():
    index = NeighborIndex(["إلا", "ألا", "لا"])
    assert index.neighbors("الا", limit=8, with_typos=False) == ["إلا", "ألا"]


def test_neighbors_include_ending_swaps_before_typos():
    index = NeighborIndex(["مدت", "مرة", "مرت"])
    assert index.neighbors("مرت", limit=8) == ["مرة", "مدت"]
