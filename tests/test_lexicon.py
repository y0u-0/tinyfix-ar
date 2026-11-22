from collections import Counter

import numpy as np

from tinyfix_ar.lexicon import Curated, RealMistakes, build_lexicon, known_misspellings, unknown_class


def test_unknown_class_keeps_clitics_and_endings():
    assert unknown_class("والمستشفيات") == "<unk:وال-ات>"
    assert unknown_class("كتبهم") == "<unk:-هم>"
    assert unknown_class("زرد") == "<unk:->"


def test_misspellings_need_annotators_and_clean_text_to_agree():
    real = RealMistakes(
        pairs=Counter({("ان", "أن"): 100, ("والدي", "والذي"): 50, ("علي", "على"): 30}),
        kept=Counter({"علي": 20}),
    )
    clean = Counter({"ان": 1, "أن": 1000, "والدي": 80, "والذي": 900, "علي": 5, "على": 1000})
    assert real.misspellings(clean) == {"ان"}


def test_dialect_words_are_never_changed_and_never_suggested():
    raw_tokens = ["<s>", "مرة", "مره", "حلو", "مرت"]
    counts = np.array([0, 900, 5, 500, 400])
    real = RealMistakes(pairs=Counter(), kept=Counter())
    lexicon = build_lexicon(raw_tokens, counts, size=200, real=real, curated=Curated(dialect={"مره"}), min_real_count=2)
    ids = lexicon.ids
    assert lexicon.valid[ids["مره"]]
    assert ids["مره"] not in lexicon.neighbors[ids["مرة"]]
    assert not lexicon.neighbors[ids["مره"]].any()


def test_corpus_can_be_extended_without_renumbering_existing_tokens(tmp_path, monkeypatch):
    from tinyfix_ar import corpus as corpus_module

    monkeypatch.setitem(
        corpus_module.SOURCES, "fake", corpus_module.Source("fake", lambda: (text for text in ["ذهبت إلى السوق مع أخي الصغير صباح اليوم"]), trusted=True)
    )
    base, _ = corpus_module.build_corpus({"fake": 100}, max_error_rate=0.02, min_evidence=3)
    extended, _ = corpus_module.build_corpus({"fake": 100}, max_error_rate=0.02, min_evidence=3, base=base)
    assert extended.raw_tokens[: len(base.raw_tokens)] == base.raw_tokens
    assert len(extended.documents) == 2 and (extended.documents[0] == extended.documents[1]).all()


def test_a_dialect_word_that_is_also_msa_can_still_be_suggested():
    raw_tokens = ["<s>", "كذا", "كدا", "هكذا"]
    counts = np.array([0, 900, 300, 500])
    real = RealMistakes(pairs=Counter(), kept=Counter())
    curated = Curated(one_way=[("كدا", "كذا")], dialect={"كذا"})
    lexicon = build_lexicon(raw_tokens, counts, size=200, real=real, curated=curated, min_real_count=2)
    ids = lexicon.ids
    assert ids["كذا"] in lexicon.neighbors[ids["كدا"]]


def test_words_in_audited_sentences_are_never_treated_as_misspellings():
    real = RealMistakes(pairs=Counter({("مذ", "منذ"): 40}), kept=Counter())
    raw_tokens = ["<s>", "مذ", "منذ"]
    counts = np.array([0, 3, 9000])
    assert known_misspellings(raw_tokens, counts, real, Curated()) == {"مذ"}
    assert known_misspellings(raw_tokens, counts, real, Curated(attested={"مذ"})) == set()


def test_a_dialect_word_common_in_the_corpus_can_still_be_suggested():
    raw_tokens = ["<s>", "واجد", "واحد", "وش"]
    counts = np.array([0, 300, 9000, 5])
    real = RealMistakes(pairs=Counter(), kept=Counter())
    lexicon = build_lexicon(raw_tokens, counts, size=200, real=real, curated=Curated(dialect={"واجد", "وش"}), min_real_count=2)
    ids = lexicon.ids
    assert ids["واجد"] in lexicon.neighbors[ids["واحد"]]
