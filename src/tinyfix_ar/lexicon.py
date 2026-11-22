"""The model's vocabulary and, for every word in it, the words a writer may have meant instead.

Everything here is shipped with the model, so the browser engine never has to rebuild it:
the candidate list and its kinds for a typed word are a single table lookup.

A word can be in the vocabulary without ever being suggested. Common misspellings (ان, الى)
must be in the vocabulary so the model can judge them when typed, but must never be offered as
the fix for something else.
"""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from tinyfix_ar.orthography import ErrorKind, NeighborIndex, classify
from tinyfix_ar.qalb import aligned_sentences
from tinyfix_ar.text import FOREIGN, NUMBER, SPECIAL_TOKENS, START, UNK, is_arabic_word, tokenize

MAX_NEIGHBORS = 8
MIN_TARGET_LENGTH = 2
CURATED_WEIGHT = 20
MIN_PUNCTUATION_COUNT = 100
MISSPELLING_MIN_COUNT = 3
MISSPELLING_MIN_SHARE = 0.9
MISSPELLING_MAX_CLEAN_RATIO = 0.05
# A dialect spelling this common in the mostly-MSA corpus is also an MSA word (واجد, بليلة), so it may be suggested.
DIALECT_AS_MSA_MIN_COUNT = 20

TYPED_KIND = 0
KIND_IDS = {kind: index + 1 for index, kind in enumerate(ErrorKind)}
KIND_COUNT = len(KIND_IDS) + 1

_UNKNOWN_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل")
_UNKNOWN_SUFFIXES = ("ات", "ون", "ين", "ان", "ة", "ها", "هم", "ه", "ي", "ا")
UNKNOWN_CLASSES = [f"<unk:{prefix}-{suffix}>" for prefix in ("", *_UNKNOWN_PREFIXES) for suffix in ("", *_UNKNOWN_SUFFIXES)]


def unknown_class(word: str) -> str:
    """Keeps the clitics and ending of a word outside the vocabulary, which still say a lot about grammar."""
    prefix = next((prefix for prefix in _UNKNOWN_PREFIXES if word.startswith(prefix) and len(word) - len(prefix) >= 2), "")
    stem = word[len(prefix) :]
    suffix = next((suffix for suffix in _UNKNOWN_SUFFIXES if stem.endswith(suffix) and len(stem) - len(suffix) >= 2), "")
    return f"<unk:{prefix}-{suffix}>"


def is_target(word: str) -> bool:
    return len(word) >= MIN_TARGET_LENGTH and is_arabic_word(word)


@dataclass(frozen=True)
class Curated:
    """Hand-checked mix-ups: groups of valid words, and misspellings that are always wrong."""

    groups: list[list[str]] = field(default_factory=list)
    one_way: list[tuple[str, str]] = field(default_factory=list)
    dialect: set[str] = field(default_factory=set)
    attested: set[str] = field(default_factory=set)

    @classmethod
    def read(cls, groups: Path, one_way: Path, dialect: Path | None = None, contexts: Path | None = None) -> "Curated":
        """contexts is a folder of audited correct sentences; every Arabic word in them is a valid spelling."""
        with groups.open(encoding="utf-8") as rows:
            group_rows = [row["words"].split() for row in csv.DictReader(rows, delimiter="\t", quoting=csv.QUOTE_NONE)]
        with one_way.open(encoding="utf-8") as rows:
            one_way_rows = [(row["typed"], row["intended"]) for row in csv.DictReader(rows, delimiter="\t", quoting=csv.QUOTE_NONE)]
        dialect_words = set()
        if dialect:
            with dialect.open(encoding="utf-8") as rows:
                dialect_words = {row["word"] for row in csv.DictReader(rows, delimiter="\t", quoting=csv.QUOTE_NONE)}
        attested = set()
        for path in sorted(contexts.glob("*.txt")) if contexts else []:
            attested |= {token for line in path.read_text(encoding="utf-8").splitlines() for token in tokenize(line) if is_arabic_word(token)}
        return cls(group_rows, one_way_rows, dialect_words, attested)

    def pairs(self) -> list[tuple[str, str]]:
        within_groups = [(typed, intended) for group in self.groups for typed in group for intended in group if typed != intended]
        return within_groups + self.one_way

    @property
    def valid_words(self) -> set[str]:
        """Words that are correct as typed: group members, dialect spellings (ابغى, وش) and words in audited sentences."""
        return {word for group in self.groups for word in group} | self.dialect | self.attested


@dataclass(frozen=True)
class RealMistakes:
    """What native writers typed and what annotators changed it to, plus how often each word was left alone."""

    pairs: Counter
    kept: Counter

    def frequent(self, min_count: int) -> Counter:
        return Counter({pair: count for pair, count in self.pairs.items() if count >= min_count})

    @property
    def corrected(self) -> Counter:
        """How often annotators changed each typed word, to anything."""
        counts = Counter()
        for (typed, _), count in self.pairs.items():
            counts[typed] += count
        return counts

    def misspellings(self, clean_counts: Counter) -> set[str]:
        """Words annotators nearly always corrected and that clean text almost never uses, like ان and الى.

        The second test keeps valid words that QALB writers happened to misuse, like والدي typed for والذي.
        """
        top_intended = {}
        for (typed, intended), _ in self.pairs.most_common():
            top_intended.setdefault(typed, intended)
        return {
            word
            for word, count in self.corrected.items()
            if count >= MISSPELLING_MIN_COUNT
            and count / (count + self.kept[word]) >= MISSPELLING_MIN_SHARE
            and clean_counts[word] < MISSPELLING_MAX_CLEAN_RATIO * clean_counts[top_intended[word]]
        }


def mine_real_mistakes(split: str = "qalb14-train") -> RealMistakes:
    pairs, kept = Counter(), Counter()
    for sentence in aligned_sentences(split):
        pairs.update((substitution.typed, substitution.intended) for substitution in sentence.substitutions)
        kept.update(sentence.typed[position] for position in sentence.untouched)
    return RealMistakes(pairs, kept)


@dataclass
class Lexicon:
    words: list[str]
    neighbors: np.ndarray
    neighbor_kinds: np.ndarray
    mistakes: np.ndarray
    mistake_weights: np.ndarray
    valid: np.ndarray
    confusions: dict[str, list[str]]

    _ARRAYS = ("neighbors", "neighbor_kinds", "mistakes", "mistake_weights", "valid")

    @cached_property
    def ids(self) -> dict[str, int]:
        return {word: index for index, word in enumerate(self.words)}

    def token_id(self, token: str) -> int:
        if token in self.ids:
            return self.ids[token]
        return self.ids[unknown_class(token)] if is_arabic_word(token) else UNK

    def candidates(self, typed_id: int) -> tuple[np.ndarray, np.ndarray]:
        """The typed word first, then its neighbors, padded with PAD."""
        ids = np.concatenate([[typed_id], self.neighbors[typed_id]]).astype(np.int64)
        kinds = np.concatenate([[TYPED_KIND], self.neighbor_kinds[typed_id]]).astype(np.int64)
        return ids, kinds

    def lookup_table(self, raw_tokens: list[str]) -> np.ndarray:
        specials = {SPECIAL_TOKENS[START]: START, SPECIAL_TOKENS[NUMBER]: NUMBER, SPECIAL_TOKENS[FOREIGN]: FOREIGN}
        return np.array([specials[token] if token in specials else self.token_id(token) for token in raw_tokens], dtype=np.int32)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "vocab.txt").write_text("\n".join(self.words) + "\n", encoding="utf-8")
        for name in self._ARRAYS:
            np.save(folder / f"{name}.npy", getattr(self, name))
        (folder / "confusions.json").write_text(json.dumps(self.confusions, ensure_ascii=False, indent=0), encoding="utf-8")
        (folder / "kinds.json").write_text(json.dumps({"typed": TYPED_KIND, **{kind.value: index for kind, index in KIND_IDS.items()}}, indent=2))

    @classmethod
    def load(cls, folder: Path) -> "Lexicon":
        arrays = {name: np.load(folder / f"{name}.npy") for name in cls._ARRAYS}
        words = (folder / "vocab.txt").read_text(encoding="utf-8").splitlines()
        confusions = json.loads((folder / "confusions.json").read_text(encoding="utf-8"))
        return cls(words=words, confusions=confusions, **arrays)


def known_misspellings(raw_tokens: list[str], counts: np.ndarray, real: RealMistakes, curated: Curated) -> set[str]:
    """Spellings that are always wrong: attested in QALB and rare in clean text, or listed by hand."""
    clean_counts = Counter({token: int(counts[index]) for index, token in enumerate(raw_tokens) if is_arabic_word(token)})
    return (real.misspellings(clean_counts) | {typed for typed, _ in curated.one_way}) - curated.valid_words


def rank_words(raw_tokens: list[str], counts: np.ndarray) -> tuple[list[str], list[str]]:
    words = sorted(((token, counts[index]) for index, token in enumerate(raw_tokens) if is_arabic_word(token)), key=lambda pair: (-pair[1], pair[0]))
    punctuation = [token for index, token in enumerate(raw_tokens) if len(token) == 1 and not token.isalnum() and counts[index] >= MIN_PUNCTUATION_COUNT]
    return [word for word, _ in words], sorted(punctuation)


def choose_vocabulary(ranked: list[str], punctuation: list[str], size: int, forced: list[str], required: list[str] = ()) -> tuple[list[str], list[str]]:
    """The most common words, plus forced words too rare to make the cut on their own (returned separately).

    A forced word is added only if the corpus has it; a required word (a curated misspelling such as
    الرياظ, which a cleaned corpus no longer contains) is added either way, so a writer who types it
    can be corrected.
    """
    reserved = SPECIAL_TOKENS + UNKNOWN_CLASSES + punctuation
    room = size - len(reserved)
    common = set(ranked[:room])
    present = set(ranked)
    candidates = [*(word for word in forced if word in present), *(word for word in required if is_arabic_word(word))]
    extra = [word for word in dict.fromkeys(candidates) if word not in common]
    if len(extra) > room // 10:
        raise ValueError(f"{len(extra)} forced words would crowd out the common vocabulary")
    return reserved + ranked[: room - len(extra)] + extra, extra


def build_confusions(typeable: set[str], suggestable: set[str], real_mistakes: Counter, curated: list[tuple[str, str]]) -> dict[str, list[tuple[str, int]]]:
    """Maps a typed word to words people really meant by it, most likely first."""
    counts = Counter({pair: count for pair, count in real_mistakes.items() if pair[0] in typeable and pair[1] in suggestable})
    for typed, intended in curated:
        if typed in typeable and intended in suggestable:
            counts[(typed, intended)] += CURATED_WEIGHT
    confusions = defaultdict(list)
    for (typed, intended), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        confusions[typed].append((intended, count))
    return {typed: options[:MAX_NEIGHBORS] for typed, options in confusions.items()}


def build_lexicon(raw_tokens: list[str], counts: np.ndarray, size: int, real: RealMistakes, curated: Curated, min_real_count: int) -> Lexicon:
    ranked, punctuation = rank_words(raw_tokens, counts)
    mined = Counter({pair: count for pair, count in real.frequent(min_real_count).items() if classify(*pair) is not ErrorKind.CLITIC})
    curated_pairs = curated.pairs()
    pairs = [*curated_pairs, *mined]
    words, forced_in = choose_vocabulary(
        ranked, punctuation, size, [*(word for pair in mined for word in pair), *sorted(curated.dialect)], [word for pair in curated_pairs for word in pair]
    )
    ids = {word: index for index, word in enumerate(words)}

    meant_somewhere = {intended for _, intended in pairs} | curated.valid_words
    never_suggest = known_misspellings(raw_tokens, counts, real, curated) | (set(forced_in) - meant_somewhere)
    valid = [word for word in words if is_target(word) and word not in never_suggest]
    corpus_counts = dict(zip(raw_tokens, counts.tolist(), strict=True))
    msa_words = {intended for _, intended in pairs} | {word for group in curated.groups for word in group}
    common_in_msa = {word for word in curated.dialect if corpus_counts.get(word, 0) >= DIALECT_AS_MSA_MIN_COUNT}
    dialect_only = curated.dialect - msa_words - common_in_msa
    suggestable = [word for word in valid if word not in dialect_only]
    confusions = build_confusions(set(words), set(suggestable), mined, curated_pairs)

    index = NeighborIndex(suggestable)
    neighbors = np.zeros((len(words), MAX_NEIGHBORS), dtype=np.int32)
    neighbor_kinds = np.zeros((len(words), MAX_NEIGHBORS), dtype=np.int8)
    for word_id, word in enumerate(words):
        # A typed dialect word (مره "very", تكفى) is what the writer meant, so it gets no candidates and is never judged.
        if not is_target(word) or word in curated.dialect:
            continue
        known_mixups = [intended for intended, _ in confusions.get(word, [])]
        found = index.neighbors(word, MAX_NEIGHBORS, known_mixups=known_mixups, with_typos=word not in never_suggest)
        neighbors[word_id, : len(found)] = [ids[other] for other in found]
        neighbor_kinds[word_id, : len(found)] = [KIND_IDS[classify(word, other)] for other in found]

    typed_for = defaultdict(list)
    for typed, options in confusions.items():
        for intended, count in options:
            typed_for[ids[intended]].append((ids[typed], count))
    mistakes = np.zeros((len(words), MAX_NEIGHBORS), dtype=np.int32)
    mistake_weights = np.zeros((len(words), MAX_NEIGHBORS), dtype=np.float32)
    for intended_id, options in typed_for.items():
        options = sorted(options, key=lambda option: -option[1])[:MAX_NEIGHBORS]
        mistakes[intended_id, : len(options)] = [typed for typed, _ in options]
        mistake_weights[intended_id, : len(options)] = [count**0.5 for _, count in options]

    valid_mask = np.zeros(len(words), dtype=bool)
    valid_mask[[ids[word] for word in valid]] = True
    return Lexicon(
        words,
        neighbors,
        neighbor_kinds,
        mistakes,
        mistake_weights,
        valid_mask,
        {typed: [intended for intended, _ in options] for typed, options in confusions.items()},
    )
