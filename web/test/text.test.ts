import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { applyEdits, edits, normalize, tokenize, type TokenizedText } from "../src/text.js";
import { fixtures } from "./helpers.js";

const inRanges = (ranges: ReadonlyArray<readonly [number, number]>, codePoint: number): boolean => {
  let low = 0;
  let high = ranges.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    const [first, last] = ranges[middle]!;
    if (codePoint < first) high = middle - 1;
    else if (codePoint > last) low = middle + 1;
    else return true;
  }
  return false;
};

const ARABIC_LETTERS = new Set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي");

/** What tinyfix_ar.text.tokenize makes of a single normalized character, from how Python's re classifies it. */
function pythonTokens(char: string, codePoint: number): string[] {
  const { word, space, digit } = fixtures.characters;
  if (inRanges(space, codePoint)) return [];
  if (inRanges(digit, codePoint)) return ["<num>"];
  if (/^[A-Za-z]$/.test(char)) return ["<lat>"];
  if (ARABIC_LETTERS.has(char)) return [char];
  return inRanges(word, codePoint) ? [] : [char];
}

const tokenAt = (tokenized: TokenizedText, text: string, index: number) => {
  const token = tokenized.tokens[index]!;
  return { value: token.value, typed: text.slice(token.start, token.end) };
};

describe("tokenize", () => {
  it(`splits every character the way Python's re does (Unicode ${fixtures.characters.unicode_version})`, () => {
    const mismatches: string[] = [];
    for (let codePoint = 0; codePoint <= 0x10ffff; codePoint++) {
      if ((codePoint >= 0xd800 && codePoint <= 0xdfff) || inRanges(fixtures.characters.unassigned, codePoint)) continue;
      const char = String.fromCodePoint(codePoint);
      if (normalize(char) !== char) continue;
      const actual = tokenize(char).tokens.map((token) => token.value);
      if (actual.join("\n") !== pythonTokens(char, codePoint).join("\n"))
        mismatches.push(`U+${codePoint.toString(16)}`);
    }
    assert.deepEqual(mismatches.slice(0, 20), []);
  });

  it("normalizes each character with its marks the same as the whole text", () => {
    for (const { text } of [...fixtures.edge_cases, ...fixtures.sentences]) {
      assert.equal(tokenize(text).normalized, normalize(text), JSON.stringify(text));
    }
  });

  it("points every token at the original characters it came from", () => {
    for (const { text } of [...fixtures.edge_cases, ...fixtures.sentences]) {
      const tokenized = tokenize(text);
      for (const token of tokenized.tokens) {
        assert.equal(
          normalize(text.slice(token.start, token.end)),
          tokenized.normalized.slice(token.spanStart, token.spanEnd),
        );
      }
    }
  });

  it("keeps diacritics, tatweel and the typed letters inside a word's span", () => {
    const text = "ذَهَبَ  الوَلَـــدُ،";
    const tokenized = tokenize(text);
    assert.deepEqual(
      tokenized.tokens.map((_, index) => tokenAt(tokenized, text, index)),
      [
        { value: "ذهب", typed: "ذَهَبَ" },
        { value: "الولد", typed: "الوَلَـــدُ" },
        { value: "،", typed: "،" },
      ],
    );
  });

  it("composes a combining hamza or madda with its letter", () => {
    const text = "أن آخر";
    const tokenized = tokenize(text);
    assert.deepEqual(tokenAt(tokenized, text, 0), { value: "أن", typed: "أن" });
    assert.deepEqual(tokenAt(tokenized, text, 1), { value: "آخر", typed: "آخر" });
  });

  it("maps presentation forms and ligatures back to the character typed", () => {
    const text = "ﻻ ﺑﻴﺖ ﷺ";
    const tokenized = tokenize(text);
    assert.deepEqual(
      tokenized.tokens.map((_, index) => tokenAt(tokenized, text, index)),
      [
        { value: "لا", typed: "ﻻ" },
        { value: "بيت", typed: "ﺑﻴﺖ" },
        { value: "صلى", typed: "ﷺ" },
        { value: "الله", typed: "ﷺ" },
        { value: "عليه", typed: "ﷺ" },
        { value: "وسلم", typed: "ﷺ" },
      ],
    );
  });

  it("counts positions in UTF-16 units past emoji and other astral characters", () => {
    const text = "👩‍👩‍👧 انا 😀 هنا";
    const tokenized = tokenize(text);
    const words = tokenized.tokens.filter((token) => token.value === "انا" || token.value === "هنا");
    assert.deepEqual(
      words.map((token) => text.slice(token.start, token.end)),
      ["انا", "هنا"],
    );
  });
});

describe("edits", () => {
  it("replaces only the word, keeping the spacing, punctuation and diacritics around it", () => {
    const text = "  قال  انَّ   الطالبَ ذهب الي البيتِ؟";
    const tokenized = tokenize(text);
    const index = (value: string) => tokenized.tokens.findIndex((token) => token.value === value);
    const changes = edits(tokenized, [
      { index: index("الي"), word: "إلى" },
      { index: index("ان"), word: "إن" },
    ]);
    assert.equal(applyEdits(text, changes), "  قال  إن   الطالبَ ذهب إلى البيتِ؟");
  });

  it("rewrites a typed character once when several of the words it expands to change", () => {
    const text = "قال ﷺ.";
    const tokenized = tokenize(text);
    const changes = edits(tokenized, [
      { index: 2, word: "اللة" },
      { index: 4, word: "و سلم" },
    ]);
    assert.deepEqual(changes, [{ start: 4, end: 5, replacement: "صلى اللة عليه و سلم", indices: [2, 4] }]);
    assert.equal(applyEdits(text, changes), "قال صلى اللة عليه و سلم.");
  });

  it("refuses overlapping edits", () => {
    assert.throws(() =>
      applyEdits("abc", [
        { start: 0, end: 2, replacement: "x" },
        { start: 1, end: 3, replacement: "y" },
      ]),
    );
  });
});
