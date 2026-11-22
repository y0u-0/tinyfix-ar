import gzip
import json
from collections import Counter
from dataclasses import asdict

import pytest

from tinyfix_ar.qalb import sentence_pairs
from tinyfix_ar.wiki_edits import (
    HISTORY_COLUMNS,
    LINK_PLACEHOLDER,
    Candidate,
    _texts,
    build,
    changed_lines,
    deduplicated,
    fetch_order,
    is_candidate,
    is_misspelling,
    prose,
    public_comment,
    revision_pairs,
    sentence_fixes,
    split_of,
    summary_kind,
    word_swaps,
)

SENTENCE = "ذهب الطالب الى المدرسة في الصباح الباكر مع اخيه"
FIXED = "ذهب الطالب إلى المدرسة في الصباح الباكر مع أخيه"


def candidate(rev_id: int = 2, parent_id: int = 1, page_id: int = 7, editor: int = 5, comment: str = "تدقيق إملائي") -> Candidate:
    return Candidate(rev_id, parent_id, page_id, "2015-01-01 00:00:00.0", 0, False, editor, comment)


def history_row(**overrides: str) -> list[str]:
    values = {
        "event_entity": "revision",
        "page_namespace": "0",
        "page_is_redirect": "false",
        "revision_id": "2",
        "revision_parent_id": "1",
        "revision_text_bytes_diff": "2",
        "revision_is_identity_reverted": "false",
        "revision_is_identity_revert": "false",
        "event_comment": "تدقيق إملائي",
    }
    values.update(overrides)
    return [values.get(name, "") for name in HISTORY_COLUMNS]


@pytest.mark.parametrize(
    ("summary", "kind"),
    [
        ("تدقيق إملائي باستخدام أوب", "spelling"),
        ("/* التاريخ */ تصحيح املائي", "spelling"),
        ("typo", "spelling"),
        ("تصحيح", "correction"),
        ("تصحيح وصلة", None),
        ("/* أشهر الخطاطين */", None),
        ("رجوع عن تعديل إملائي", None),
        ("إضافة مرجع", None),
    ],
)
def test_summary_kind_separates_spelling_from_other_corrections(summary, kind):
    assert summary_kind(summary) == kind


@pytest.mark.parametrize(
    "overrides",
    [
        {"page_namespace": "1"},
        {"revision_is_identity_reverted": "true"},
        {"revision_is_identity_revert": "true"},
        {"revision_parent_id": ""},
        {"revision_text_bytes_diff": "-900"},
        {"event_comment": "إضافة تصنيف"},
        {"event_entity": "page"},
    ],
)
def test_is_candidate_rejects_talk_pages_reverts_big_edits_and_unrelated_summaries(overrides):
    assert is_candidate(history_row())
    assert not is_candidate(history_row(**overrides))


def test_fetch_order_lets_each_editor_take_a_turn_before_a_prolific_bot_continues():
    bot = [candidate(rev_id=100 + index, editor=1) for index in range(5)]
    people = [candidate(rev_id=200 + index, editor=10 + index) for index in range(3)]
    anonymous = [candidate(rev_id=300 + index, editor=0) for index in range(2)]
    corrections = [candidate(rev_id=400, editor=20, comment="تصحيح")]
    order = fetch_order(corrections + bot + people + anonymous)
    assert order[-1] == corrections[0]
    assert Counter(c.editor for c in order[:6]) == {1: 1, 10: 1, 11: 1, 12: 1, 0: 2}


def test_prose_keeps_reader_visible_text_and_drops_markup():
    line = "'''الرباط''' هي [[عاصمة]] [[المغرب|المملكة المغربية]]<ref>{{استشهاد|عنوان=x}}</ref> {{مرجع}} [[ملف:a.jpg|تصغير]] [http://x.org موقع]"
    assert prose(line) == "الرباط هي عاصمة المملكة المغربية موقع"
    assert prose(line, show_links=False) == f"الرباط هي {LINK_PLACEHOLDER} {LINK_PLACEHOLDER} موقع"


@pytest.mark.parametrize("line", ["== التاريخ ==", "| الاسم = الرباط", "{| class=wikitable", "{{صندوق معلومات", "نص [[رابط مكسور"])
def test_prose_rejects_headings_tables_and_broken_markup(line):
    assert prose(line) is None


def test_word_swaps_accepts_only_sentences_whose_every_difference_is_a_word_swap():
    assert [(s.typed, s.intended, s.kind) for s in word_swaps(SENTENCE, FIXED)] == [("الى", "إلى", "hamza"), ("اخيه", "أخيه", "hamza")]
    assert word_swaps(SENTENCE, SENTENCE) is None
    assert word_swaps(SENTENCE, FIXED.replace("الصباح", "المساء")) is None
    assert word_swaps(SENTENCE + " عام 1990", FIXED + " عام 1991") is None
    assert word_swaps(SENTENCE, FIXED + " اليوم") is None


def test_sentence_fixes_keeps_only_the_changed_sentence_and_skips_link_edits():
    other = "وكان الجو جميلا في ذلك اليوم المشمس الطويل."
    fixes = list(sentence_fixes(f"{other} {SENTENCE}.", f"{other} {FIXED}."))
    assert [(old, new) for old, new, _ in fixes] == [(f"{SENTENCE}.", f"{FIXED}.")]
    in_link = f"[[{SENTENCE.split()[0]}]] {SENTENCE.split(maxsplit=1)[1].replace('الى', '[[الى]]')}"
    assert list(sentence_fixes(in_link, in_link.replace("[[الى]]", "[[إلى]]"))) == []


def test_changed_lines_pairs_lines_edited_in_place():
    old = "مقدمة\nسطر أول الى\nسطر ثان\nخاتمة"
    new = "مقدمة\nسطر أول إلى\nسطر ثان\nسطر جديد\nخاتمة"
    assert list(changed_lines(old, new)) == [("سطر أول الى", "سطر أول إلى")]


def test_is_misspelling_catches_wrong_direction_hamza_fixes():
    assert is_misspelling("الإنتخابات", frozenset())
    assert is_misspelling("للإستخدام", frozenset())
    assert is_misspelling("والأعتماد", frozenset())
    assert is_misspelling("الى", frozenset({"الى"}))
    assert not any(is_misspelling(word, frozenset()) for word in ("الإنتاج", "الإنسان", "إطلاق", "إتمام", "إستونيا", "أستار", "أحداث", "أختام"))


def test_revision_pairs_drops_fixes_that_introduce_a_misspelling():
    old, new = f"مقدمة\n{SENTENCE}\n", f"مقدمة\n{FIXED}\n"
    assert len(revision_pairs(candidate(), old, new, frozenset())) == 1
    assert revision_pairs(candidate(), old, new, frozenset({"أخيه"})) == []


def test_deduplicated_keeps_the_earliest_revision_of_a_normalized_pair():
    pairs = [revision_pairs(candidate(rev_id=rev_id), f"\n{SENTENCE}", f"\n{FIXED}{suffix}", frozenset())[0] for rev_id, suffix in ((9, ""), (3, "َ"))]
    assert [pair.rev_id for pair in deduplicated(pairs)] == [3]


def test_deduplicated_counts_a_fix_repeated_through_one_page_once():
    other_sentence = SENTENCE.replace("الصباح", "المساء")
    pairs = [
        revision_pairs(candidate(page_id=page_id), f"\n{typed}", f"\n{typed.replace('الى', 'إلى').replace('اخيه', 'أخيه')}", frozenset())[0]
        for page_id, typed in ((1, SENTENCE), (1, other_sentence), (2, other_sentence))
    ]
    assert [(pair.page_id, pair.raw) for pair in deduplicated(pairs)] == [(1, SENTENCE), (2, other_sentence)]


def test_public_comment_removes_user_names_and_addresses():
    assert public_comment("قائمة تدقيق ([[User:SomeBot/إملاء|المزيد]]) شكرا [[مستخدم:فلان|فلان]] و 10.1.2.3") == "قائمة تدقيق ([user]) شكرا [user] و [ip]"
    assert public_comment("تدقيق إملائي باستخدام أوب") == "تدقيق إملائي باستخدام أوب"


def test_split_of_is_deterministic_and_close_to_eighty_ten_ten():
    shares = Counter(split_of(page_id) for page_id in range(20_000))
    assert split_of(12345) == split_of(12345)
    assert abs(shares["train"] / 20_000 - 0.8) < 0.01
    assert abs(shares["dev"] / 20_000 - 0.1) < 0.01


def test_texts_reads_content_and_marks_missing_or_hidden_revisions():
    answer = {
        "query": {
            "badrevids": {"9": {"revid": 9, "missing": True}},
            "pages": [
                {
                    "revisions": [
                        {"revid": 1, "slots": {"main": {"contentmodel": "wikitext", "content": "نص"}}},
                        {"revid": 2, "slots": {"main": {"texthidden": True}}},
                    ]
                }
            ],
        }
    }
    assert _texts(answer) == {9: None, 1: "نص", 2: None}


def test_build_writes_splits_that_qalb_sentence_pairs_can_read(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "out"
    (cache / "candidates").mkdir(parents=True)
    (cache / "revisions").mkdir()
    edits = [candidate(rev_id=2 * page, parent_id=2 * page - 1, page_id=page) for page in range(1, 41)]
    (cache / "candidates" / "2015.jsonl").write_text("".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in edits), encoding="utf-8")
    revisions = [
        {"revid": rev_id, "slots": {"main": {"contentmodel": "wikitext", "content": f"مقدمة\n{text} {'جدا ' * c.page_id}\n"}}}
        for c in edits
        for rev_id, text in ((c.parent_id, SENTENCE), (c.rev_id, FIXED))
    ]
    (cache / "revisions" / "1-80.json.gz").write_bytes(gzip.compress(json.dumps({"query": {"pages": [{"revisions": revisions}]}}).encode()))

    stats = build(cache, output)

    assert stats["pairs"] == 40
    assert stats["kinds"] == {"hamza": 80}
    rows = [json.loads(line) for line in (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(rows[0]) == {"raw", "cor", "page_id", "rev_id", "comment"}
    split_rows = [pair for name in ("train", "dev", "test") for pair in sentence_pairs(output / "splits" / f"{name}.jsonl")]
    assert sorted(split_rows) == sorted((row["raw"], row["cor"]) for row in rows)
