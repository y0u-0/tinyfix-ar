"""Writes the fixtures the TypeScript engine is checked against: .venv/bin/python -I web/scripts/make-parity-fixtures.py

Everything comes from the Python reference: tokens and token ids from tinyfix_ar.text and
tinyfix_ar.lexicon, model inputs and margins from Fixer.judge running the exported ONNX model,
and the text as the typing app fixes it (left context already fixed, right context as typed).
The character tables pin down which code points Python's re treats as \\w, \\s and \\d, since
JavaScript's classes differ and the tokenizer must split text exactly as Python does.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from itertools import islice
from pathlib import Path

from tinyfix_ar.engine import INPUTS, Fixer, Thresholds, onnx_scorer
from tinyfix_ar.lexicon import Lexicon
from tinyfix_ar.qalb import download, sentence_pairs
from tinyfix_ar.text import tokenize

REPO = Path(__file__).resolve().parents[2]
SPLIT = "qalb14-dev"
LEXICON = Path("artifacts/lexicon")
BUNDLE = Path("model")
OUTPUT = Path("web/test/fixtures.json")
VISIBLE_RIGHT = 2
MARGIN_DECIMALS = 5

EDGE_TEXTS = [
    "",
    " \t\n ",
    "ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ صَبَاحًا",
    "كـــتاب جمـيل جداًّ",
    "ﻻ أعرف ﷲ ﷺ ﺑﻴﺖ",
    "ا\u0654ن الكتاب و\u0654 ي\u0654 ا\u0653 ا\u0655",
    "علی کتابی ٱلحمد",
    "عام ٢٠٢٤ و ١٢٫٥ و 3,000.25 و ۱۴۰۲ و 12.",
    "hello مرحبا world, café ڤيديو پ",
    "مرحبا\xa0بك\u2009يا\u3000صديقي",
    "\u200cمي\u200fرحبا\u200d\u200e\ufeffسلام",
    "😀 انا ❤\ufe0f هنا 👩\u200d👩\u200d👧 الى",
    "\x1c\x1dسلام\x1e\x1f\x85عليكم",
    "ﹰ ﹱ ﹲ ﹴ",
    "انا ذهبت الى المدرسه ولاكن لم اجد احد هناك",
    "قال إن الطالب ذهب الي البيت فى المساء",
    "؟!..،؛ «هذا» (ذلك) [تلك] - — …",
    "ـــ",
    "عَلَى",
    "Ⅻ ① ² ½ ٣٤٥",
]


def character_ranges(test) -> list[list[int]]:
    """Inclusive [first, last] runs of code points (surrogates excluded) for which test holds."""
    ranges: list[list[int]] = []
    for code_point in range(sys.maxunicode + 1):
        if 0xD800 <= code_point <= 0xDFFF or not test(chr(code_point)):
            continue
        if ranges and ranges[-1][1] == code_point - 1:
            ranges[-1][1] = code_point
        else:
            ranges.append([code_point, code_point])
    return ranges


def character_classes() -> dict:
    word, space, digit = re.compile(r"\w"), re.compile(r"\s"), re.compile(r"\d")
    return {
        "unicode_version": unicodedata.unidata_version,
        "unassigned": character_ranges(lambda char: unicodedata.category(char) == "Cn"),
        "word": character_ranges(lambda char: word.fullmatch(char) is not None),
        "space": character_ranges(lambda char: space.fullmatch(char) is not None),
        "digit": character_ranges(lambda char: digit.fullmatch(char) is not None),
    }


def judged(fixer: Fixer, tokens: list[str]) -> dict:
    positions, inputs = fixer.inputs(tokens)
    judgements = fixer.judge(tokens)
    return {
        "positions": positions,
        **{name: inputs[name].ravel().tolist() if inputs else [] for name in INPUTS},
        "best": [judgement.best for judgement in judgements],
        "best_kinds": [judgement.kind for judgement in judgements],
        "margins": [round(judgement.margin, MARGIN_DECIMALS) for judgement in judgements],
    }


def fix_as_typed(fixer: Fixer, tokens: list[str], thresholds: Thresholds) -> tuple[list[str], list[dict]]:
    """Fixes left to right: each word is judged after the words before it have been fixed, as in the typing app.

    The whole text is judged again after every change, because the int8 model quantizes its activations per batch:
    a word's margin depends on the other words scored with it, so only the same batches give the same margins.
    """
    fixed = list(tokens)
    judgements = {judgement.position: judgement for judgement in fixer.judge(fixed)}
    changes = []
    for position in sorted(judgements):
        judgement = judgements[position]
        if thresholds.changes(judgement):
            fixed[position] = judgement.best
            changes.append({"position": position, "best": judgement.best, "margin": round(judgement.margin, MARGIN_DECIMALS)})
            judgements.update({later.position: later for later in fixer.judge(fixed) if later.position > position})
    return fixed, changes


def case(fixer: Fixer, text: str, thresholds: Thresholds) -> dict:
    tokens = tokenize(text)
    fixed, changes = fix_as_typed(fixer, tokens, thresholds)
    batch_fixed = fixer.fix(text, thresholds)
    return {
        "text": text,
        "tokens": tokens,
        "token_ids": [fixer.lexicon.token_id(token) for token in tokens],
        "judge": judged(fixer, tokens),
        "changes": changes,
        "fixed_like_batch": fixed == batch_fixed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sentences", type=int, default=300)
    parser.add_argument("--stride", type=int, default=3, help="Take every n-th sentence of the split, so the fixtures span all of it.")
    args = parser.parse_args()

    os.chdir(REPO)

    calibrated = json.loads((BUNDLE / "manifest.json").read_text())["thresholds"]
    thresholds = Thresholds.from_json(calibrated)
    fixer = Fixer(onnx_scorer(BUNDLE / "model.onnx"), Lexicon.load(LEXICON), visible_right=VISIBLE_RIGHT)
    typed_texts = [typed for typed, _ in islice(sentence_pairs(download(SPLIT)), 0, None, args.stride)][: args.sentences]

    fixtures = {
        "source": {"split": SPLIT, "lexicon": str(LEXICON), "model": str(BUNDLE / "model.onnx"), "visible_right": fixer.visible_right},
        "thresholds": calibrated,
        "characters": character_classes(),
        "edge_cases": [case(fixer, text, thresholds) for text in EDGE_TEXTS],
        "sentences": [case(fixer, text, thresholds) for text in typed_texts],
    }
    OUTPUT.write_text(json.dumps(fixtures, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")

    sentences = fixtures["sentences"]
    rows = sum(len(sentence["judge"]["positions"]) for sentence in sentences)
    changes = sum(len(sentence["changes"]) for sentence in sentences)
    differ = sum(not sentence["fixed_like_batch"] for sentence in sentences)
    print(f"{len(sentences)} sentences, {rows} judged words, {changes} changes; {differ} fix differently from Fixer.fix (typed left context)")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
