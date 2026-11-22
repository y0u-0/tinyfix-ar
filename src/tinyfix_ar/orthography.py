"""How Arabic writers turn one real word into another, and how to find the words they meant.

Three kinds of neighbors, from most to least common in real writing:

* spelling variants: words that differ only in hamza, taa marbuta, alif maqsura or alif fariqa
  (أن/إن/ان, مدرسة/مدرسه, على/علي). They share a spelling skeleton. A last letter swapped
  between ة and ت (رحمة/رحمت) or ى and ا (عصى/عصا) counts too.
* sound-alikes: one letter swapped for a letter that sounds the same in many dialects
  (ظل/ضل, هذا/هدا, ثمن/سمن, صورة/سورة).
* typos: one deletion, insertion, substitution or swap of adjacent letters.
"""

from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum


class ErrorKind(StrEnum):
    HAMZA = "hamza"
    TAA_MARBUTA = "taa_marbuta"
    ALIF_MAQSURA = "alif_maqsura"
    ALIF_FARIQA = "alif_fariqa"
    SPELLING_MIXED = "spelling_mixed"
    SOUND_ALIKE = "sound_alike"
    CLITIC = "clitic"
    TYPO = "typo"
    OTHER = "other"


_KIND_BY_LETTER_PAIR = {frozenset("ةه"): ErrorKind.TAA_MARBUTA, frozenset("ىي"): ErrorKind.ALIF_MAQSURA}
_DROPPED_HAMZA = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ؤ": "و", "ئ": "ي", "ى": "ي", "ة": "ه", "ء": None})
_ANY_HAMZA_SEAT = str.maketrans({"أ": "ء", "إ": "ء", "آ": "ء", "ؤ": "ء", "ئ": "ء", "ى": "ي", "ة": "ه"})
_SOUND_ALIKE_PAIRS = [("ض", "ظ"), ("ذ", "ز"), ("ذ", "د"), ("ظ", "ز"), ("ظ", "ذ"), ("ث", "س"), ("ث", "ت"), ("ص", "س")]
_ENDING_SWAPS = {"ة": ("ت", ErrorKind.TAA_MARBUTA), "ت": ("ة", ErrorKind.TAA_MARBUTA), "ى": ("ا", ErrorKind.ALIF_MAQSURA), "ا": ("ى", ErrorKind.ALIF_MAQSURA)}
_PREFIX_CLITICS = frozenset("وفبلك")

SOUND_ALIKES: dict[str, tuple[str, ...]] = {}
for _first, _second in _SOUND_ALIKE_PAIRS:
    SOUND_ALIKES[_first] = (*SOUND_ALIKES.get(_first, ()), _second)
    SOUND_ALIKES[_second] = (*SOUND_ALIKES.get(_second, ()), _first)


def _without_alif_fariqa(word: str) -> str:
    return word[:-1] if len(word) > 2 and word.endswith("وا") else word


def skeletons(word: str) -> tuple[str, str]:
    """Two keys that collapse spellings differing only in hamza, ة/ه, ى/ي or a final silent alif.

    The first drops hamza entirely (مسؤول and مسوول match); the second keeps that a hamza is
    there but forgets its seat (مسؤول and مسئول match).
    """
    return _without_alif_fariqa(word.translate(_DROPPED_HAMZA)), "ء:" + _without_alif_fariqa(word.translate(_ANY_HAMZA_SEAT))


def same_skeleton(first: str, second: str) -> bool:
    return any(a == b for a, b in zip(skeletons(first), skeletons(second), strict=True))


def ending_variant(word: str) -> str | None:
    """The word with its last letter swapped as writers do: ة/ت (رحمة/رحمت, مرة/مرت) or ى/ا (عصى/عصا)."""
    swap = _ENDING_SWAPS.get(word[-1:])
    return word[:-1] + swap[0] if swap and len(word) > 2 else None


def sound_alike_variants(word: str) -> set[str]:
    return {word[:index] + other + word[index + 1 :] for index, letter in enumerate(word) for other in SOUND_ALIKES.get(letter, ())}


def deletions(word: str) -> set[str]:
    return {word[:index] + word[index + 1 :] for index in range(len(word))}


def within_one_edit(first: str, second: str) -> bool:
    if first == second or abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        differences = [index for index, (a, b) in enumerate(zip(first, second, strict=True)) if a != b]
        if len(differences) == 1:
            return True
        return (
            len(differences) == 2
            and differences[1] == differences[0] + 1
            and first[differences[0]] == second[differences[1]]
            and first[differences[1]] == second[differences[0]]
        )
    shorter, longer = sorted((first, second), key=len)
    return shorter in deletions(longer)


def _spelling_kind(typed: str, intended: str) -> ErrorKind:
    if len(typed) != len(intended):
        longer = max(typed, intended, key=len)
        return ErrorKind.ALIF_FARIQA if longer.endswith("وا") and longer[:-1] in (typed, intended) else ErrorKind.HAMZA
    kinds = {_KIND_BY_LETTER_PAIR.get(frozenset(pair), ErrorKind.HAMZA) for pair in zip(typed, intended, strict=True) if pair[0] != pair[1]}
    return kinds.pop() if len(kinds) == 1 else ErrorKind.SPELLING_MIXED


def classify(typed: str, intended: str) -> ErrorKind:
    if typed == intended:
        raise ValueError("a word cannot be a mistake for itself")
    if same_skeleton(typed, intended):
        return _spelling_kind(typed, intended)
    if ending_variant(typed) == intended:
        return _ENDING_SWAPS[typed[-1]][1]
    if intended in sound_alike_variants(typed):
        return ErrorKind.SOUND_ALIKE
    shorter, longer = sorted((typed, intended), key=len)
    if len(longer) == len(shorter) + 1 and longer[0] in _PREFIX_CLITICS and longer[1:] == shorter:
        return ErrorKind.CLITIC
    if within_one_edit(typed, intended):
        return ErrorKind.TYPO
    return ErrorKind.OTHER


class NeighborIndex:
    """Finds real words a writer may have meant, ordered by kind and then by how common they are."""

    def __init__(self, words_by_frequency: Iterable[str]):
        self.rank: dict[str, int] = {}
        self._by_skeleton: dict[str, list[str]] = defaultdict(list)
        self._by_deletion: dict[str, list[str]] = defaultdict(list)
        for word in words_by_frequency:
            if word in self.rank:
                continue
            self.rank[word] = len(self.rank)
            for key in skeletons(word):
                self._by_skeleton[key].append(word)
            for key in deletions(word) | {word}:
                self._by_deletion[key].append(word)

    def spelling_variants(self, word: str) -> list[str]:
        found = {other for key in skeletons(word) for other in self._by_skeleton.get(key, [])}
        return sorted(found - {word}, key=self.rank.__getitem__)

    def sound_alikes(self, word: str) -> list[str]:
        return sorted((other for other in sound_alike_variants(word) if other in self.rank), key=self.rank.__getitem__)

    def typos(self, word: str) -> list[str]:
        """One-edit words, except a clitic added or dropped (لا/ولا), which is rarely a typo and often a style edit."""
        nearby = {other for key in deletions(word) | {word} for other in self._by_deletion.get(key, [])}
        return sorted((other for other in nearby if within_one_edit(word, other) and classify(word, other) is not ErrorKind.CLITIC), key=self.rank.__getitem__)

    def neighbors(self, word: str, limit: int, known_mixups: Iterable[str] = (), with_typos: bool = True) -> list[str]:
        ordered = [*known_mixups, *self.spelling_variants(word), ending_variant(word), *self.sound_alikes(word), *(self.typos(word) if with_typos else [])]
        return [other for other in dict.fromkeys(ordered) if other != word and other in self.rank][:limit]
