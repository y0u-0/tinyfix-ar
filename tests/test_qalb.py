from tinyfix_ar.orthography import ErrorKind
from tinyfix_ar.qalb import align


def test_align_finds_one_for_one_swaps_and_untouched_words():
    sentence = align("اريد ان اذهب الى البيت", "أريد أن أذهب إلى البيت .")
    assert [(s.position, s.typed, s.intended) for s in sentence.substitutions] == [(0, "اريد", "أريد"), (1, "ان", "أن"), (2, "اذهب", "أذهب"), (3, "الى", "إلى")]
    assert sentence.untouched == {4}


def test_align_skips_merges_and_unrelated_rewrites():
    sentence = align("لوكان الجو جميل", "لو كان الطقس جميلا")
    assert all(s.kind is not ErrorKind.OTHER for s in sentence.substitutions)
    assert 0 not in {s.position for s in sentence.substitutions}
