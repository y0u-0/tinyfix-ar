"""Real Arabic spelling fixes mined from Arabic Wikipedia edit history: python -m tinyfix_ar.wiki_edits {scan,fetch,build,sample}

scan   streams Wikimedia's MediaWiki history dumps (one row per revision, with its edit summary, size
       change and revert flags) and keeps article revisions whose summary says they fix spelling.
fetch  asks the MediaWiki API for each candidate's diff, one request a second, caching every answer.
build  turns the cached diffs into sentence pairs that differ only by one or two plausible word swaps,
       then writes pairs.jsonl, page-disjoint train/dev/test splits and their statistics.
sample prints random pairs for checking by hand.

Everything downloaded is cached under --cache (outside the repository), so every stage can be
stopped and rerun without repeating requests.
"""

import argparse
import bz2
import gzip
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path

from tinyfix_ar.lexicon import Curated
from tinyfix_ar.qalb import align
from tinyfix_ar.text import is_arabic_word, tokenize

USER_AGENT = "tinyfix-ar-research/0.1 (spelling-error research)"
API_URL = "https://ar.wikipedia.org/w/api.php"
HISTORY_SNAPSHOT = "2026-10"
HISTORY_URL = f"https://dumps.wikimedia.org/other/mediawiki_history/{HISTORY_SNAPSHOT}/arwiki/"
DEFAULT_CACHE = Path("/tmp/wikiedits-cache")
DEFAULT_OUTPUT = Path("data/wiki_edits")
GROUPS = Path("data/confusions/groups.tsv")
ONE_WAY = Path("data/confusions/one_way.tsv")

HISTORY_COLUMNS = (
    "wiki_db", "event_log_id", "event_entity", "event_type", "event_timestamp", "event_comment",
    "event_user_id", "event_user_central_id", "event_user_text_historical", "event_user_text",
    "event_user_blocks_historical", "event_user_blocks", "event_user_groups_historical", "event_user_groups",
    "event_user_is_bot_by_historical", "event_user_is_bot_by", "event_user_is_created_by_self",
    "event_user_is_created_by_system", "event_user_is_created_by_peer", "event_user_is_anonymous",
    "event_user_is_temporary", "event_user_is_permanent", "event_user_is_cross_wiki",
    "event_user_registration_timestamp", "event_user_creation_timestamp", "event_user_first_edit_timestamp",
    "event_user_revision_count", "event_user_seconds_since_previous_revision",
    "page_id", "page_title_historical", "page_title", "page_namespace_historical",
    "page_namespace_is_content_historical", "page_namespace", "page_namespace_is_content", "page_is_redirect",
    "page_is_deleted", "page_creation_timestamp", "page_first_edit_timestamp", "page_revision_count",
    "page_seconds_since_previous_revision",
    "user_id", "user_central_id", "user_text_historical", "user_text", "user_blocks_historical", "user_blocks",
    "user_groups_historical", "user_groups", "user_is_bot_by_historical", "user_is_bot_by",
    "user_is_created_by_self", "user_is_created_by_system", "user_is_created_by_peer", "user_is_anonymous",
    "user_is_temporary", "user_is_permanent", "user_registration_timestamp", "user_creation_timestamp",
    "user_first_edit_timestamp",
    "revision_id", "revision_parent_id", "revision_minor_edit", "revision_deleted_parts",
    "revision_deleted_parts_are_suppressed", "revision_text_bytes", "revision_text_bytes_diff",
    "revision_text_sha1", "revision_content_model", "revision_content_format",
    "revision_is_deleted_by_page_deletion", "revision_deleted_by_page_deletion_timestamp",
    "revision_is_identity_reverted", "revision_first_identity_reverting_revision_id",
    "revision_seconds_to_identity_revert", "revision_is_identity_revert", "revision_is_from_before_page_creation",
    "revision_tags",
)  # fmt: skip
_COLUMN = {name: index for index, name in enumerate(HISTORY_COLUMNS)}

SPELLING_SUMMARY = re.compile(r"إملا|املا|تهجئ|تهجي|همز|مطبع|typo|spell|hamza", re.IGNORECASE)
"""Summaries that name spelling itself: إملائي, تدقيق إملائي, تهجئة, همزات, خطأ مطبعي, typo, spelling."""
CORRECTION_SUMMARY = re.compile(r"تدقيق|تصحيح|تصويب|أخطاء|اخطاء|خطأ|خطا|لغوي|correct", re.IGNORECASE)
"""Summaries that announce a correction of some kind; many are spelling fixes, many are not."""
NOT_SPELLING_SUMMARY = re.compile(
    r"وصل|رابط|روابط|تحويل|تصنيف|قالب|{{|مرجع|مراجع|استشهاد|تاريخ|معلوم|صور|اسم|إسم|ترجم|interwiki|link|redirect|image|\biw\b|ref|cat",
    re.IGNORECASE,
)
"""Corrections that say what they correct, when that is not spelling (تصحيح وصلة, تصحيح تاريخ)."""
REVERT_SUMMARY = re.compile(r"رجوع|استرجاع|استرجع|تراجع|إلغاء تعديل|الغاء تعديل|تخريب|undo|undid|revert|\brv\b|vandal", re.IGNORECASE)
_SECTION_NAME = re.compile(r"/\*.*?\*/")
MAX_BYTES_CHANGED = 300


@dataclass(frozen=True)
class Candidate:
    rev_id: int
    parent_id: int
    page_id: int
    timestamp: str
    bytes_diff: int
    is_bot: bool
    editor: int
    """The editor's numeric id (0 when anonymous), kept in the cache only to measure how concentrated the data is."""
    comment: str

    @property
    def summary_kind(self) -> str | None:
        return summary_kind(self.comment)


def is_candidate(fields: list[str]) -> bool:
    def column(name: str) -> str:
        return fields[_COLUMN[name]]

    return (
        column("event_entity") == "revision"
        and column("page_namespace") == "0"
        and column("page_is_redirect") != "true"
        and column("revision_parent_id") not in ("", "0")
        and column("revision_is_deleted_by_page_deletion") != "true"
        and column("revision_deleted_parts") in ("", "array()")
        and column("revision_is_identity_reverted") != "true"
        and column("revision_is_identity_revert") != "true"
        and column("revision_text_bytes_diff") != ""
        and abs(int(column("revision_text_bytes_diff"))) <= MAX_BYTES_CHANGED
        and summary_kind(column("event_comment")) is not None
    )


def summary_kind(summary: str) -> str | None:
    """Whether an edit summary names spelling, announces some other likely correction, or neither."""
    summary = _SECTION_NAME.sub(" ", summary)
    if REVERT_SUMMARY.search(summary):
        return None
    if SPELLING_SUMMARY.search(summary):
        return "spelling"
    if CORRECTION_SUMMARY.search(summary) and not NOT_SPELLING_SUMMARY.search(summary):
        return "correction"
    return None


def to_candidate(fields: list[str]) -> Candidate:
    def column(name: str) -> str:
        return fields[_COLUMN[name]]

    return Candidate(
        rev_id=int(column("revision_id")),
        parent_id=int(column("revision_parent_id")),
        page_id=int(column("page_id")),
        timestamp=column("event_timestamp"),
        bytes_diff=int(column("revision_text_bytes_diff")),
        is_bot=column("event_user_is_bot_by_historical") not in ("", "array()"),
        editor=int(column("event_user_id") or 0),
        comment=column("event_comment"),
    )


def history_files() -> list[str]:
    with urllib.request.urlopen(urllib.request.Request(HISTORY_URL, headers={"User-Agent": USER_AGENT}), timeout=60) as response:
        listing = response.read().decode("utf-8")
    return sorted(set(re.findall(rf'href="({re.escape(HISTORY_SNAPSHOT)}\.arwiki\.\d{{4}}\.tsv\.bz2)"', listing)))


def scan_history_file(name: str) -> Iterator[Candidate]:
    request = urllib.request.Request(HISTORY_URL + name, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, bz2.open(response, "rt", encoding="utf-8", errors="replace") as rows:
        for row in rows:
            fields = row.rstrip("\n").split("\t")
            if len(fields) == len(HISTORY_COLUMNS) and is_candidate(fields):
                yield to_candidate(fields)


def candidates_dir(cache: Path) -> Path:
    return cache / "candidates"


def scan_one(cache: Path, name: str) -> str:
    path = candidates_dir(cache) / f"{name.removesuffix('.tsv.bz2')}.jsonl"
    if path.exists():
        return f"{name}: already scanned"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    found = 0
    with partial.open("w", encoding="utf-8") as output:
        for candidate in scan_history_file(name):
            output.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
            found += 1
    partial.rename(path)
    return f"{name}: {found:,} candidates"


def scan(cache: Path, files: list[str], jobs: int) -> None:
    """Dump files are scanned a few at a time; dumps.wikimedia.org asks for at most two or three connections."""
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for message in pool.map(scan_one, [cache] * len(files), files):
            print(message, flush=True)


def load_candidates(cache: Path) -> list[Candidate]:
    found = {}
    for path in sorted(candidates_dir(cache).glob("*.jsonl")):
        with path.open(encoding="utf-8") as rows:
            for row in rows:
                candidate = Candidate(**json.loads(row))
                found[candidate.rev_id] = candidate
    return list(found.values())


def fetch_order(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Spelling summaries first, then other corrections, with editors taking turns within each.

    A few typo-fixing bots made most of these edits; taking turns keeps them from crowding out the
    thousands of people who fixed spelling by hand. Anonymous edits count as one editor each.
    """
    tier = {"spelling": 0, "correction": 1}
    by_editor: dict[tuple[int, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_editor[tier[candidate.summary_kind], candidate.editor or -candidate.rev_id].append(candidate)
    turns = []
    for (rank, _), edits in by_editor.items():
        edits.sort(key=lambda candidate: _stable_hash(f"rev:{candidate.rev_id}"))
        turns.extend(((rank, turn, _stable_hash(f"rev:{candidate.rev_id}")), candidate) for turn, candidate in enumerate(edits))
    return [candidate for _, candidate in sorted(turns, key=lambda turn: turn[0])]


def _stable_hash(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


class Api:
    """A sequential MediaWiki API client that spaces out requests, honours maxlag and Retry-After, and backs off on errors.

    Clients whose User-Agent carries no contact address get 10 requests a minute, so the default
    interval stays under that.
    """

    def __init__(self, url: str = API_URL, interval: float = 6.5, attempts: int = 8):
        self.url, self.interval, self.attempts = url, interval, attempts
        self._last_request = 0.0

    def get(self, **params: str | int) -> dict:
        query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2, "maxlag": 5})
        request = urllib.request.Request(f"{self.url}?{query}", headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        for attempt in range(self.attempts):
            self._wait()
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    answer = json.loads(body)
                    retry_after = response.headers.get("Retry-After")
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504):
                    raise
                self._back_off(attempt, error.headers.get("Retry-After"), f"HTTP {error.code}")
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                self._back_off(attempt, None, str(error))
                continue
            if answer.get("error", {}).get("code") == "maxlag":
                self._back_off(attempt, retry_after, "maxlag")
                continue
            return answer
        raise RuntimeError(f"gave up after {self.attempts} attempts")

    def _wait(self) -> None:
        pause = self._last_request + self.interval - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        self._last_request = time.monotonic()

    @staticmethod
    def _back_off(attempt: int, retry_after: str | None, reason: str) -> None:
        delay = int(retry_after) + 1 if retry_after and retry_after.isdigit() else min(2**attempt * 10, 600)
        print(f"{reason}; retrying in {delay}s", file=sys.stderr, flush=True)
        time.sleep(delay)


REVISIONS_PER_REQUEST = 50


def revisions_dir(cache: Path) -> Path:
    return cache / "revisions"


def resolved_revisions(cache: Path) -> set[int]:
    """Revisions already answered for: their text, or the API's word that it is gone or hidden."""
    index = revisions_dir(cache) / "index.jsonl"
    if not index.exists():
        return set()
    with index.open(encoding="utf-8") as rows:
        return {rev_id for row in rows if row.strip() for rev_id in json.loads(row)["resolved"]}


def _texts(answer: dict) -> dict[int, str | None]:
    """Wikitext by revision id from a prop=revisions answer; None for revisions that are deleted or hidden."""
    texts: dict[int, str | None] = {int(rev_id): None for rev_id in answer.get("query", {}).get("badrevids", {})}
    for page in answer.get("query", {}).get("pages", []):
        for revision in page.get("revisions", []):
            main = revision.get("slots", {}).get("main", {})
            texts[revision["revid"]] = main.get("content") if main.get("contentmodel", "wikitext") == "wikitext" else None
    return texts


def fetch(cache: Path, max_requests: int, hours: float) -> None:
    """Downloads each candidate's text and its parent's, fifty revisions a request, in fetch_order."""
    directory = revisions_dir(cache)
    directory.mkdir(parents=True, exist_ok=True)
    resolved = resolved_revisions(cache)
    batches, batch = [], []
    for candidate in fetch_order(load_candidates(cache)):
        needed = [rev_id for rev_id in (candidate.parent_id, candidate.rev_id) if rev_id not in resolved and rev_id not in batch]
        if len(batch) + len(needed) > REVISIONS_PER_REQUEST:
            batches.append(batch)
            batch = []
        batch.extend(needed)
    batches.append(batch)
    batches = [batch for batch in batches if batch][:max_requests]
    print(f"{len(resolved):,} revisions cached, {len(batches):,} requests to go", flush=True)

    api, deadline = Api(), time.monotonic() + hours * 3600
    with (directory / "index.jsonl").open("a", encoding="utf-8") as index:
        for number, batch in enumerate(batches, 1):
            if time.monotonic() > deadline:
                break
            answer = api.get(action="query", prop="revisions", revids="|".join(map(str, batch)), rvprop="ids|content", rvslots="main")
            name = f"{batch[0]}-{len(batch)}.json.gz"
            partial = directory / f"{name}.partial"
            partial.write_bytes(gzip.compress(json.dumps(answer, ensure_ascii=False).encode("utf-8")))
            partial.rename(directory / name)
            index.write(json.dumps({"file": name, "resolved": sorted(_texts(answer))}) + "\n")
            index.flush()
            if number % 50 == 0:
                print(f"{number:,} requests, {number * REVISIONS_PER_REQUEST:,} revisions", flush=True)


def load_texts(cache: Path) -> dict[int, str | None]:
    texts = {}
    for path in sorted(revisions_dir(cache).glob("*.json.gz")):
        texts.update(_texts(json.loads(gzip.decompress(path.read_bytes()))))
    return texts


def changed_lines(old_text: str, new_text: str) -> Iterator[tuple[str, str]]:
    """Lines edited in place: blocks of lines replaced by as many lines, paired in order."""
    old, new = old_text.split("\n"), new_text.split("\n")
    for operation, old_start, old_end, new_start, new_end in SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if operation == "replace" and old_end - old_start == new_end - new_start:
            yield from zip(old[old_start:old_end], new[new_start:new_end], strict=True)


_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|$)", re.DOTALL)
_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_WIKILINK = re.compile(r"\[\[([^\[\]]*)\]\]")
_NOT_ARTICLE_LINK = re.compile(r"^\s*:?\s*(?:ملف|صورة|تصنيف|file|image|category|[a-z][a-z-]{1,11})\s*:", re.IGNORECASE)
_EXTERNAL_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:\s+([^\]]*))?\]")
_URL = re.compile(r"https?://\S+")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_EMPHASIS = re.compile(r"'{2,}")
_MAGIC_WORD = re.compile(r"__[A-Zا-ي_]+__")
_LIST_MARKER = re.compile(r"^[*#:;]+")
_LEFTOVER_MARKUP = re.compile(r"\{\{|\}\}|\[\[|\]\]|\{\||\|\}|<|>")
_NOT_PROSE_LINE = re.compile(r"^\s*(?:[|!=]|\{\||#REDIRECT|#تحويل)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")
LINK_PLACEHOLDER = "§"


def prose(wikitext: str, show_links: bool = True) -> str | None:
    """The reader-visible text of one wikitext line, or None for headings, tables, galleries and broken markup.

    With show_links=False each article link becomes LINK_PLACEHOLDER, so an edit to the text of a
    link (which also renames the page it points to) can be told apart from an edit to plain prose.
    """
    if _NOT_PROSE_LINE.match(wikitext):
        return None
    text = _REF.sub("", _HTML_COMMENT.sub("", wikitext))
    text = _repeat(_TEMPLATE, "", text)

    def link(match: re.Match) -> str:
        target, _, label = match.group(1).partition("|")
        if _NOT_ARTICLE_LINK.match(target):
            return ""
        return (label or target) if show_links else LINK_PLACEHOLDER

    text = _repeat(_WIKILINK, link, text)
    text = _URL.sub("", _EXTERNAL_LINK.sub(lambda match: match.group(1) or "", text))
    text = _EMPHASIS.sub("", _HTML_TAG.sub("", _MAGIC_WORD.sub("", text)))
    if _LEFTOVER_MARKUP.search(text):
        return None
    return " ".join(_LIST_MARKER.sub("", unescape(text).strip()).split())


def _repeat(pattern: re.Pattern, replacement, text: str) -> str:
    """Applies a substitution until nothing matches, so nested markup is removed from the inside out."""
    while True:
        text, replaced = pattern.subn(replacement, text)
        if not replaced:
            return text


def sentences(paragraph: str) -> list[str]:
    return [sentence for sentence in _SENTENCE_END.split(paragraph) if sentence]


@dataclass(frozen=True)
class Swap:
    typed: str
    intended: str
    kind: str


def word_swaps(old: str, new: str) -> list[Swap] | None:
    """The word swaps that turn old into new, or None unless every difference is a plausible one-word swap.

    Differences the tokenizer hides (digits, Latin words, diacritics) must not appear either: the two
    sentences may differ in as many space-separated words as there are swaps and no more.
    """
    aligned, corrected = align(old, new), tokenize(new)
    if len(aligned.typed) != len(corrected):
        return None
    changed = {index for index, (typed, intended) in enumerate(zip(aligned.typed, corrected, strict=True)) if typed != intended}
    if not changed or changed != {substitution.position for substitution in aligned.substitutions}:
        return None
    old_words, new_words = old.split(), new.split()
    if len(old_words) != len(new_words) or sum(a != b for a, b in zip(old_words, new_words, strict=True)) > len(changed):
        return None
    return [Swap(substitution.typed, substitution.intended, substitution.kind.value) for substitution in aligned.substitutions]


MAX_SWAPS = 2
MIN_WORDS = 5
MAX_WORDS = 120
MIN_ARABIC_SHARE = 0.6


def is_sentence(text: str) -> bool:
    tokens = tokenize(text)
    words = [token for token in tokens if token.isalpha() or token.startswith("<")]
    arabic = sum(is_arabic_word(token) for token in tokens)
    return MIN_WORDS <= len(words) <= MAX_WORDS and arabic >= MIN_ARABIC_SHARE * len(words)


_HAMZA_ON_WASL = re.compile(r"^[وف]?(?:[بكل]?ال|لل|[بكل])?[أإ](?:ست[^ا]{2}ا|ن[^ا]{2}ا|[^ا][تطد][^ا]ا)")
"""Masdars of verb forms VII, VIII and X written with hamza (الإنتخاب, الإختبار, والأعتماد): their alif is hamzat wasl."""


def is_misspelling(word: str, known_misspellings: frozenset[str]) -> bool:
    """Editors sometimes "fix" a correct word into a common misspelling; such a pair teaches the wrong direction."""
    return word in known_misspellings or _HAMZA_ON_WASL.match(word) is not None


def known_misspellings(one_way: Path) -> frozenset[str]:
    return frozenset(typed for typed, _ in Curated.read(GROUPS, one_way).one_way)


@dataclass(frozen=True)
class Pair:
    raw: str
    cor: str
    page_id: int
    rev_id: int
    comment: str
    swaps: tuple[Swap, ...]

    def record(self) -> dict:
        return {"raw": self.raw, "cor": self.cor, "page_id": self.page_id, "rev_id": self.rev_id, "comment": public_comment(self.comment)}


_USER_LINK = re.compile(
    r"\[\[\s*:?\s*(?:user|user talk|مستخدم|مستخدمة|نقاش المستخدم|نقاش المستخدمة|special:contrib\w*|خاص:مساهمات)\s*[:/][^\]]*\]\]",
    re.IGNORECASE,
)
_BARE_USER = re.compile(r"(?:user|مستخدم|مستخدمة)\s*:\s*[^\s|\]/]+", re.IGNORECASE)
_IP_ADDRESS = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b", re.IGNORECASE)


def public_comment(comment: str) -> str:
    """The edit summary without the user names and IP addresses it sometimes mentions."""
    return _IP_ADDRESS.sub("[ip]", _BARE_USER.sub("[user]", _USER_LINK.sub("[user]", comment)))


def sentence_fixes(old_line: str, new_line: str) -> Iterator[tuple[str, str, list[Swap]]]:
    """Sentences of one edited line whose only changes are one or two word swaps outside links and markup."""
    old_text, new_text = prose(old_line), prose(new_line)
    old_masked, new_masked = prose(old_line, show_links=False), prose(new_line, show_links=False)
    if None in (old_text, new_text, old_masked, new_masked) or old_text == new_text:
        return
    split = [sentences(text) for text in (old_text, new_text, old_masked, new_masked)]
    if len({len(part) for part in split}) != 1:
        return
    for old, new, masked_old, masked_new in zip(*split, strict=True):
        if old == new:
            continue
        swaps = word_swaps(old, new)
        if not swaps or len(swaps) > MAX_SWAPS or not is_sentence(old):
            continue
        masked_swaps = word_swaps(masked_old, masked_new)
        if masked_swaps is None or Counter(masked_swaps) != Counter(swaps):
            continue
        yield old, new, swaps


def revision_pairs(candidate: Candidate, old_text: str, new_text: str, misspellings: frozenset[str]) -> list[Pair]:
    return [
        Pair(old, new, candidate.page_id, candidate.rev_id, candidate.comment, tuple(swaps))
        for old_line, new_line in changed_lines(old_text, new_text)
        for old, new, swaps in sentence_fixes(old_line, new_line)
        if not any(is_misspelling(swap.intended, misspellings) for swap in swaps)
    ]


def deduplicated(pairs: Iterable[Pair]) -> list[Pair]:
    """Keeps the earliest revision of each (typed, corrected) sentence, compared after normalization, and one
    sentence per page for each set of swaps, so a fix replaced all through one article counts once."""
    seen, kept = set(), []
    for pair in sorted(pairs, key=lambda pair: pair.rev_id):
        sentence = (" ".join(tokenize(pair.raw)), " ".join(tokenize(pair.cor)))
        swaps_on_page = (pair.page_id, tuple(sorted((swap.typed, swap.intended) for swap in pair.swaps)))
        if sentence not in seen and swaps_on_page not in seen:
            seen.update((sentence, swaps_on_page))
            kept.append(pair)
    return kept


SPLIT_SHARES = (("train", 8), ("dev", 1), ("test", 1))


def split_of(page_id: int) -> str:
    """Assigns whole pages to splits by hash, so no article's sentences are in two splits."""
    bucket = _stable_hash(f"page:{page_id}") % sum(share for _, share in SPLIT_SHARES)
    for name, share in SPLIT_SHARES:
        if bucket < share:
            return name
        bucket -= share
    raise AssertionError("unreachable")


def build(cache: Path, output: Path, one_way: Path = ONE_WAY, top: int = 50) -> dict:
    misspellings = known_misspellings(one_way)
    candidates = {candidate.rev_id: candidate for candidate in load_candidates(cache)}
    texts = load_texts(cache)
    fetched = [candidate for candidate in candidates.values() if texts.get(candidate.rev_id) and texts.get(candidate.parent_id)]
    found = [pair for candidate in fetched for pair in revision_pairs(candidate, texts[candidate.parent_id], texts[candidate.rev_id], misspellings)]
    pairs = deduplicated(found)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "pairs.jsonl", (pair.record() for pair in pairs))
    by_split = {name: [pair for pair in pairs if split_of(pair.page_id) == name] for name, _ in SPLIT_SHARES}
    (output / "splits").mkdir(exist_ok=True)
    for name, members in by_split.items():
        _write_jsonl(output / "splits" / f"{name}.jsonl", ({"raw": pair.raw, "cor": pair.cor} for pair in members))

    editors = Counter(candidates[pair.rev_id].editor for pair in pairs)
    anonymous = editors.pop(0, 0)
    stats = {
        "candidates": len(candidates),
        "revisions_fetched": len(fetched),
        "revisions_with_pairs": len({pair.rev_id for pair in pairs}),
        "pages": len({pair.page_id for pair in pairs}),
        "pairs_before_dedup": len(found),
        "pairs": len(pairs),
        "substitutions": sum(len(pair.swaps) for pair in pairs),
        "registered_editors": len(editors),
        "anonymous_share": round(anonymous / max(len(pairs), 1), 3),
        "bot_share": round(sum(candidates[pair.rev_id].is_bot for pair in pairs) / max(len(pairs), 1), 3),
        "top_editor_share": round(max(editors.values(), default=0) / max(len(pairs), 1), 3),
        "top10_editor_share": round(sum(count for _, count in editors.most_common(10)) / max(len(pairs), 1), 3),
        "splits": {name: _split_stats(members, top) for name, members in by_split.items()},
        "kinds": dict(Counter(swap.kind for pair in pairs for swap in pair.swaps).most_common()),
        "top_swaps": _top_swaps(pairs, top),
    }
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def _split_stats(pairs: list[Pair], top: int) -> dict:
    return {
        "pairs": len(pairs),
        "pages": len({pair.page_id for pair in pairs}),
        "substitutions": sum(len(pair.swaps) for pair in pairs),
        "kinds": dict(Counter(swap.kind for pair in pairs for swap in pair.swaps).most_common()),
        "top_swaps": _top_swaps(pairs, top),
    }


def _top_swaps(pairs: list[Pair], top: int) -> list[tuple[str, int]]:
    return Counter(f"{swap.typed}→{swap.intended}" for pair in pairs for swap in pair.swaps).most_common(top)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample(output: Path, count: int, seed: int) -> None:
    with (output / "pairs.jsonl").open(encoding="utf-8") as rows:
        pairs = [json.loads(row) for row in rows]
    for number, pair in enumerate(random.Random(seed).sample(pairs, min(count, len(pairs))), 1):
        swaps = word_swaps(pair["raw"], pair["cor"]) or []
        print(f"{number:3d} rev {pair['rev_id']} | {' ; '.join(f'{swap.typed}→{swap.intended} ({swap.kind})' for swap in swaps)} | {pair['comment'][:80]}")
        print(f"    {pair['raw']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan", help="Find candidate revisions in the MediaWiki history dumps.")
    scan_parser.add_argument("--files", nargs="*", help="Dump file names; all years by default.")
    scan_parser.add_argument("--jobs", type=int, default=2)
    fetch_parser = commands.add_parser("fetch", help="Download the diffs of candidate revisions.")
    fetch_parser.add_argument("--max-requests", type=int, default=1_500)
    fetch_parser.add_argument("--hours", type=float, default=3.0)
    commands.add_parser("build", help="Extract, filter, deduplicate and split the sentence pairs.")
    sample_parser = commands.add_parser("sample", help="Print random pairs for checking by hand.")
    sample_parser.add_argument("--count", type=int, default=100)
    sample_parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.command == "scan":
        scan(args.cache, args.files or history_files(), args.jobs)
    elif args.command == "fetch":
        fetch(args.cache, args.max_requests, args.hours)
    elif args.command == "build":
        stats = build(args.cache, args.output)
        print(json.dumps({key: value for key, value in stats.items() if key not in ("splits", "top_swaps")}, ensure_ascii=False, indent=2))
    else:
        sample(args.output, args.count, args.seed)


if __name__ == "__main__":
    main()
