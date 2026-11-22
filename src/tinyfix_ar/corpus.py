"""Clean Arabic training text, tokenized once and cached as flat integer arrays.

The model learns that the word it sees in clean text is the right one, so a document that writes
ان for أن teaches the opposite of what we want. Web text is full of those, so every document is
scored on how it spells a handful of very common words and dropped if it gets them wrong.
"""

import ast
import contextlib
import itertools
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from datasets import load_dataset

from tinyfix_ar.qalb import download, sentence_pairs
from tinyfix_ar.text import SPECIAL_TOKENS, START, tokenize

MIN_DOCUMENT_TOKENS = 8

# Misspellings of very common words, mined from QALB-2014: typed form -> what was meant.
TELLTALE_MISSPELLINGS = {
    "ان": "أن",
    "الى": "إلى",
    "او": "أو",
    "الا": "إلا",
    "اي": "أي",
    "انه": "أنه",
    "اذا": "إذا",
    "اكثر": "أكثر",
    "انا": "أنا",
    "الان": "الآن",
    "انها": "أنها",
    "انهم": "أنهم",
    "ايضا": "أيضا",
    "اول": "أول",
    "اخرى": "أخرى",
    "فى": "في",
    "التى": "التي",
    "الذى": "الذي",
    "هى": "هي",
    "حتي": "حتى",
    "هذة": "هذه",
    "ذالك": "ذلك",
    "اللذي": "الذي",
    "اللتي": "التي",
    "لاكن": "لكن",
    "شئ": "شيء",
    "بان": "بأن",
}
_TELLTALE_CORRECT = frozenset(TELLTALE_MISSPELLINGS.values())


@dataclass(frozen=True)
class Hygiene:
    correct: int
    misspelled: int

    @property
    def evidence(self) -> int:
        return self.correct + self.misspelled

    @property
    def error_rate(self) -> float:
        return self.misspelled / self.evidence if self.evidence else 0.0


def hygiene(tokens: list[str]) -> Hygiene:
    correct = sum(token in _TELLTALE_CORRECT for token in tokens)
    misspelled = sum(token in TELLTALE_MISSPELLINGS for token in tokens)
    return Hygiene(correct, misspelled)


@dataclass(frozen=True)
class Source:
    name: str
    documents: Callable[[], Iterator[str]]
    trusted: bool


def _wikipedia() -> Iterator[str]:
    stream = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True).shuffle(seed=7, buffer_size=10_000)
    for row in stream:
        yield row["text"]


def _fineweb2_hq() -> Iterator[str]:
    stream = load_dataset("epfml/FineWeb2-HQ", "arb_Arab", split="train", streaming=True, columns=["text"]).shuffle(seed=7, buffer_size=10_000)
    for row in stream:
        yield row["text"]


SAUDI_NEWS_HOSTS = (
    "spa.gov.sa",
    "alriyadh.com",
    "okaz.com.sa",
    "sabq.org",
    "alwatan.com.sa",
    "al-madina.com",
    "aawsat.com",
    "al-jazirah.com",
    "aleqt.com",
    "makkahnewspaper.com",
    "almarsd.com",
    "ajel.sa",
    "alweeam.com.sa",
    "arriyadiyah.com",
    "slaati.com",
    "almowaten.net",
    "mz-mz.net",
    "akhbaar24.argaam.com",
    "argaam.com",
)
# These hosts drop hamza far more often than other Saudi sites.
_DIRTY_SAUDI_HOSTS = ("alyaum.com", "archive.aawsat.com", "classic.aawsat.com")


def is_clean_saudi_host(url: str) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if host.startswith("faculty.") or any(host == dirty or host.endswith("." + dirty) for dirty in _DIRTY_SAUDI_HOSTS):
        return False
    return host.endswith(".sa") or any(host == known or host.endswith("." + known) for known in SAUDI_NEWS_HOSTS)


def _fineweb2_saudi() -> Iterator[str]:
    """Saudi news, government, university and religious pages from FineWeb2's Arabic split."""
    stream = load_dataset("HuggingFaceFW/fineweb-2", "arb_Arab", split="train", streaming=True, columns=["text", "url"])
    for row in stream:
        if is_clean_saudi_host(row["url"] or ""):
            yield row["text"]


def _aya() -> Iterator[str]:
    for row in load_dataset("CohereLabs/aya_dataset", split="train"):
        if row["language_code"] == "arb":
            yield f"{row['inputs']}\n{row['targets']}"


def _streamed(name: str, config: str | None = None) -> Iterator[dict]:
    yield from load_dataset(name, config, split="train", streaming=True).shuffle(seed=7, buffer_size=5_000)


def _messages(conversation: str | list) -> str:
    turns = ast.literal_eval(conversation) if isinstance(conversation, str) else conversation
    return "\n".join(turn["value"] for turn in turns)


def _hindawi_books() -> Iterator[str]:
    for row in _streamed("Bayanat-Labs/open-arabic-books", "chapters"):
        yield row["text_raw"]


def _aljazeera() -> Iterator[str]:
    for row in _streamed("Skiittoo/aljazeera-news-arabic"):
        yield row["article_text_content"]


def _un_general_assembly() -> Iterator[str]:
    """Resolution sentences, grouped forty at a time so each document carries some context."""
    rows = (row["translation"]["ar"] for row in load_dataset("Helsinki-NLP/un_ga", "ar_to_en", split="train", streaming=True))
    while chunk := list(itertools.islice(rows, 40)):
        yield "\n".join(chunk)


def _openhermes_arabic() -> Iterator[str]:
    for row in _streamed("2A2I/Arabic-OpenHermes-2.5"):
        yield f"{row['user']}\n{row['gpt']}"


def _translated_chat() -> Iterator[str]:
    for name in ("FreedomIntelligence/alpaca-gpt4-arabic", "FreedomIntelligence/sharegpt-arabic"):
        for row in _streamed(name):
            yield _messages(row["conversations"])
    for row in _streamed("arbml/CIDAR"):
        yield f"{row['instruction']}\n{row['output']}"


def _curated_contexts() -> Iterator[str]:
    """Hand-written correct sentences for confusable words that are rare in the web and book sources."""
    for path in sorted(Path("data/contexts").glob("*.txt")):
        yield from (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _qalb_corrected() -> Iterator[str]:
    for _, corrected in sentence_pairs(download("qalb14-train")):
        yield corrected


SOURCES = {
    source.name: source
    for source in (
        Source("wikipedia", _wikipedia, trusted=True),
        Source("fineweb2-hq", _fineweb2_hq, trusted=False),
        Source("fineweb2-saudi", _fineweb2_saudi, trusted=False),
        Source("aya", _aya, trusted=False),
        Source("hindawi-books", _hindawi_books, trusted=True),
        Source("aljazeera", _aljazeera, trusted=True),
        Source("un-general-assembly", _un_general_assembly, trusted=True),
        Source("openhermes-arabic", _openhermes_arabic, trusted=True),
        Source("translated-chat", _translated_chat, trusted=True),
        Source("qalb-corrected", _qalb_corrected, trusted=True),
        Source("curated-contexts", _curated_contexts, trusted=True),
    )
}


def is_clean(tokens: list[str], trusted: bool, max_error_rate: float, min_evidence: int) -> bool:
    score = hygiene(tokens)
    if score.evidence < min_evidence:
        return trusted
    return score.error_rate <= max_error_rate


@dataclass
class Corpus:
    documents: list[np.ndarray]
    raw_tokens: list[str]
    sources: list[str]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "tokens.npy", np.concatenate(self.documents))
        np.save(folder / "lengths.npy", np.array([len(document) for document in self.documents], dtype=np.int64))
        (folder / "raw_tokens.json").write_text(json.dumps(self.raw_tokens, ensure_ascii=False))
        (folder / "sources.json").write_text(json.dumps(self.sources))

    def split(self, held_out_share: float) -> tuple[np.ndarray, np.ndarray]:
        """Document indices for training and for held-out evaluation, the same for every run on this corpus."""
        order = np.random.default_rng(7).permutation(len(self.documents))
        held_out = max(1, int(len(order) * held_out_share))
        return order[held_out:], order[:held_out]

    @classmethod
    def load(cls, folder: Path) -> "Corpus":
        tokens, lengths = np.load(folder / "tokens.npy"), np.load(folder / "lengths.npy")
        return cls(
            documents=np.split(tokens, np.cumsum(lengths)[:-1]),
            raw_tokens=json.loads((folder / "raw_tokens.json").read_text()),
            sources=json.loads((folder / "sources.json").read_text()),
        )


def build_corpus(budgets: dict[str, int], max_error_rate: float, min_evidence: int, base: Corpus | None = None) -> tuple[Corpus, dict[str, dict]]:
    """Tokenizes each source until it has contributed its token budget of clean documents, appending to base if given."""
    raw_ids = {token: index for index, token in enumerate(base.raw_tokens)} if base else {SPECIAL_TOKENS[START]: 0}
    documents, document_sources, report = (list(base.documents), list(base.sources), {}) if base else ([], [], {})
    for name, budget in budgets.items():
        source = SOURCES[name]
        started, used, kept, dropped = time.time(), 0, 0, 0
        with contextlib.closing(source.documents()) as texts:
            for text in texts:
                tokens = tokenize(text)
                if len(tokens) < MIN_DOCUMENT_TOKENS:
                    continue
                if not is_clean(tokens, source.trusted, max_error_rate, min_evidence):
                    dropped += 1
                    continue
                documents.append(np.array([0, *(raw_ids.setdefault(token, len(raw_ids)) for token in tokens)], dtype=np.int32))
                document_sources.append(name)
                kept += 1
                used += len(tokens)
                if used >= budget:
                    break
        report[name] = {"tokens": used, "documents_kept": kept, "documents_dropped": dropped, "seconds": round(time.time() - started)}
        print(f"{name}: {report[name]}", flush=True)
    return Corpus(documents, list(raw_ids), document_sources), report
