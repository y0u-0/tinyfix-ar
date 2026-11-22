"""Arabic text normalization and word tokenization.

Normalization removes only what never changes the spelling of a word (diacritics, tatweel,
presentation forms, Persian look-alike letters). Hamza, taa marbuta and alif maqsura are kept
exactly as typed, because those are the letters the model learns to fix.
"""

import re
import unicodedata

PAD, UNK, NUMBER, START, FOREIGN = 0, 1, 2, 3, 4
SPECIAL_TOKENS = ["<pad>", "<unk>", "<num>", "<s>", "<lat>"]

ARABIC_LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"

_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL = "\u0640"
_LOOKALIKES = str.maketrans({"ی": "ي", "ې": "ي", "ک": "ك", "ە": "ه", "ۀ": "ة", "ٱ": "ا", "ٲ": "أ", "ٳ": "إ"})
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_TOKEN = re.compile(rf"[{ARABIC_LETTERS}]+|[A-Za-z]+|\d+(?:[.,٫٬]\d+)*|[^\s\w]")
ARABIC_WORD = re.compile(rf"^[{ARABIC_LETTERS}]+$")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    return text.translate(_LOOKALIKES).translate(_ARABIC_DIGITS)


def tokenize(text: str) -> list[str]:
    """Splits normalized text into Arabic words, digits, Latin words and single punctuation marks."""
    tokens = []
    for raw in _TOKEN.findall(normalize(text)):
        if raw[0].isdigit():
            tokens.append(SPECIAL_TOKENS[NUMBER])
        elif raw[0].isascii() and raw[0].isalpha():
            tokens.append(SPECIAL_TOKENS[FOREIGN])
        else:
            tokens.append(raw)
    return tokens


def is_arabic_word(token: str) -> bool:
    return ARABIC_WORD.match(token) is not None
