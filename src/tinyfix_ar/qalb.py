"""Real Arabic writing mistakes from QALB-2014, QALB-2015 and ZAEBUC.

The sentence pairs come from CAMeL Lab's arabic-gec repository (pinned below), which publishes
the diacritic-free versions of the shared-task splits. QALB-2014 is news-site comments by native
writers, QALB-2015 adds learner essays, and ZAEBUC is university essays by native writers.
"""

import json
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from tinyfix_ar.orthography import ErrorKind, classify
from tinyfix_ar.text import is_arabic_word, tokenize

_COMMIT = "8c7fb84f3ed84d1d30beb7080b02d0f6bfe1c5bf"
_BASE_URL = f"https://raw.githubusercontent.com/CAMeL-Lab/arabic-gec/{_COMMIT}/data/gec/modeling"
CACHE_DIR = Path(".cache/qalb")
SPLITS = {
    "qalb14-train": "qalb14/wo_camelira/binary/train.json",
    "qalb14-dev": "qalb14/wo_camelira/binary/dev.json",
    "qalb14-test": "qalb14/wo_camelira/binary/test.json",
    "qalb15-dev": "qalb15/wo_camelira/binary/dev.json",
    "qalb15-test-l1": "qalb15/wo_camelira/binary/test_L1.json",
    "zaebuc-dev": "zaebuc/wo_camelira/binary/dev.json",
    "zaebuc-test": "zaebuc/wo_camelira/binary/test.json",
}


@dataclass(frozen=True)
class Substitution:
    position: int
    typed: str
    intended: str
    kind: ErrorKind


@dataclass(frozen=True)
class AlignedSentence:
    typed: list[str]
    substitutions: list[Substitution]
    untouched: frozenset[int]


# Real spelling fixes mined from Arabic Wikipedia edit history (data/wiki_edits/README.md), split by page.
LOCAL_SPLITS = {f"wiki-{name}": Path(f"data/wiki_edits/splits/{name}.jsonl") for name in ("dev", "test")}


def download(split: str) -> Path:
    if split in LOCAL_SPLITS:
        return LOCAL_SPLITS[split]
    path = CACHE_DIR / f"{split}.jsonl"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        urllib.request.urlretrieve(f"{_BASE_URL}/{SPLITS[split]}", partial)
        partial.rename(path)
    return path


def sentence_pairs(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                row = json.loads(line)
                yield row["raw"], row["cor"]


def align(typed_text: str, corrected_text: str) -> AlignedSentence:
    """Keeps one-for-one word swaps a writer could plausibly make, and the words the annotator left alone."""
    typed, corrected = tokenize(typed_text), tokenize(corrected_text)
    substitutions, untouched = [], set()
    for operation, typed_start, typed_end, corrected_start, corrected_end in SequenceMatcher(a=typed, b=corrected, autojunk=False).get_opcodes():
        if operation == "equal":
            untouched.update(range(typed_start, typed_end))
        elif operation == "replace" and typed_end - typed_start == corrected_end - corrected_start:
            for offset in range(typed_end - typed_start):
                typed_word, intended = typed[typed_start + offset], corrected[corrected_start + offset]
                if not (is_arabic_word(typed_word) and is_arabic_word(intended)):
                    continue
                kind = classify(typed_word, intended)
                if kind is not ErrorKind.OTHER:
                    substitutions.append(Substitution(typed_start + offset, typed_word, intended, kind))
    return AlignedSentence(typed, substitutions, frozenset(untouched))


def aligned_sentences(split: str) -> Iterator[AlignedSentence]:
    for typed_text, corrected_text in sentence_pairs(download(split)):
        yield align(typed_text, corrected_text)
