"""Contract checks on every data file the pipeline reads, so a bad merge fails loudly instead of quietly corrupting training."""

import csv
import re
from pathlib import Path

import pytest

from tinyfix_ar.corpus import TELLTALE_MISSPELLINGS
from tinyfix_ar.lexicon import Curated
from tinyfix_ar.text import is_arabic_word, normalize, tokenize

DATA = Path(__file__).resolve().parent.parent / "data"
DIACRITICS = re.compile("[ً-ْٰـ]")
EVAL_FILES = sorted([*DATA.glob("eval/*.tsv"), *DATA.glob("dev/*.tsv")])
CONTEXT_FILES = sorted(DATA.glob("contexts/*.txt"))


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as rows:
        return list(csv.DictReader(rows, delimiter="\t", quoting=csv.QUOTE_NONE))


@pytest.fixture(scope="module")
def curated() -> Curated:
    dialect = DATA / "confusions/dialect_words.tsv"
    return Curated.read(DATA / "confusions/groups.tsv", DATA / "confusions/one_way.tsv", dialect if dialect.exists() else None, DATA / "contexts")


@pytest.fixture(scope="module")
def misspellings(curated: Curated) -> set[str]:
    return ({typed for typed, _ in curated.one_way} | set(TELLTALE_MISSPELLINGS)) - curated.valid_words


def test_group_words_are_single_arabic_words(curated: Curated):
    bad = [group for group in curated.groups if len(group) < 2 or not all(is_arabic_word(word) and tokenize(word) == [word] for word in group)]
    assert not bad


def test_no_word_is_both_valid_and_a_misspelling(curated: Curated):
    assert not curated.valid_words & {typed for typed, _ in curated.one_way}


def test_one_way_pairs_point_to_valid_spellings(curated: Curated):
    typed_forms = {typed for typed, _ in curated.one_way}
    assert not [(typed, intended) for typed, intended in curated.one_way if intended in typed_forms or typed == intended]
    assert not [(typed, intended) for typed, intended in curated.one_way if not (is_arabic_word(typed) and is_arabic_word(intended))]


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda path: path.name)
def test_eval_rows_differ_by_exactly_one_word(path: Path):
    problems = []
    for row in read_tsv(path):
        typed, corrected = tokenize(row["typed"]), tokenize(row["corrected"])
        expected = 0 if row["category"].endswith("clean") else 1
        if len(typed) != len(corrected) or sum(a != b for a, b in zip(typed, corrected, strict=False)) != expected:
            problems.append(row["id"])
        if DIACRITICS.search(row["typed"] + row["corrected"]):
            problems.append(f"{row['id']} has diacritics")
    assert not problems


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda path: path.name)
def test_eval_ids_are_unique(path: Path):
    ids = [row["id"] for row in read_tsv(path)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("path", CONTEXT_FILES, ids=lambda path: path.name)
def test_training_contexts_contain_no_known_misspelling(path: Path, misspellings: set[str]):
    """These sentences are trained on as correct text, so one misspelling teaches the model the mistake."""
    flagged = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and set(tokenize(line)) & misspellings]
    assert not flagged, flagged[:5]


@pytest.mark.parametrize("path", CONTEXT_FILES, ids=lambda path: path.name)
def test_training_contexts_are_plain_sentences(path: Path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert not [line for line in lines if DIACRITICS.search(line) or normalize(line) != line or not 6 <= len(tokenize(line)) <= 60]
    assert len(lines) == len(set(lines))


LEAK_NGRAM = 6


def ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[start : start + size]) for start in range(len(tokens) - size + 1)}


@pytest.fixture(scope="module")
def evaluation_ngrams() -> set[tuple[str, ...]]:
    found = set()
    for path in EVAL_FILES:
        for row in read_tsv(path):
            found |= ngrams(tokenize(row["typed"]), LEAK_NGRAM) | ngrams(tokenize(row["corrected"]), LEAK_NGRAM)
    return found


@pytest.mark.parametrize("path", CONTEXT_FILES, ids=lambda path: path.name)
def test_training_contexts_do_not_copy_evaluation_sentences(path: Path, evaluation_ngrams: set[tuple[str, ...]]):
    """A training sentence sharing a long run of words with a dev or test row leaks its answer."""
    leaked = [line for line in path.read_text(encoding="utf-8").splitlines() if ngrams(tokenize(line), LEAK_NGRAM) & evaluation_ngrams]
    assert not leaked, leaked[:5]
