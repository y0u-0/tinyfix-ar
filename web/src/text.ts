/**
 * Arabic normalization and word tokenization, matching tinyfix_ar/text.py exactly, plus where each token sits in the
 * text the user typed. The model reads normalized tokens; fixes are written back into the original text.
 */

export const PAD = 0;
export const START = 3;

const NUMBER_TOKEN = "<num>";
const FOREIGN_TOKEN = "<lat>";

const ARABIC_LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي";
/** Diacritics, and tatweel (U+0640). */
const NOT_SPELLING = /[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]/gu;
const LOOKALIKES: Readonly<Record<string, string>> = {
  ی: "ي",
  ې: "ي",
  ک: "ك",
  ە: "ه",
  ۀ: "ة",
  ٱ: "ا",
  ٲ: "أ",
  ٳ: "إ",
};
const EASTERN_DIGITS = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹";
const EASTERN_DIGIT = new RegExp(`[${EASTERN_DIGITS}]`, "gu");
const LOOKALIKE = new RegExp(`[${Object.keys(LOOKALIKES).join("")}]`, "gu");

/** Python's \s, which unlike JavaScript's includes U+001C–U+001F and U+0085 and leaves out U+FEFF. */
const PYTHON_SPACE = "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
/** Python's \w and \d over Unicode, which JavaScript's \w and \d only cover for ASCII. */
const PYTHON_WORD = "\\p{L}\\p{N}_";
const PYTHON_DIGIT = "\\p{Nd}";

const TOKEN = new RegExp(
  `[${ARABIC_LETTERS}]+|(?<latin>[A-Za-z]+)|(?<number>${PYTHON_DIGIT}+(?:[.,٫٬]${PYTHON_DIGIT}+)*)|[^${PYTHON_SPACE}${PYTHON_WORD}]`,
  "gu",
);
const ARABIC_WORD = new RegExp(`^[${ARABIC_LETTERS}]+$`, "u");

/**
 * A base character and the marks that follow it. Normalization composes only within these (ا followed by a combining
 * hamza becomes أ), so normalizing them one at a time gives the same text as normalizing the whole string. Hangul
 * vowel and final jamo compose with the letter before them without being marks, so they stay with it too.
 */
const CLUSTER = /\P{M}[\p{M}\u1160-\u11ff]*|[\p{M}\u1160-\u11ff]+/gu;

export function normalize(text: string): string {
  return text
    .normalize("NFKC")
    .replace(NOT_SPELLING, "")
    .replace(LOOKALIKE, (letter) => LOOKALIKES[letter]!)
    .replace(EASTERN_DIGIT, (digit) => String(EASTERN_DIGITS.indexOf(digit) % 10));
}

export function isArabicWord(token: string): boolean {
  return ARABIC_WORD.test(token);
}

export interface Token {
  /** What the model reads: the normalized word, a punctuation mark, or <num> / <lat>. */
  value: string;
  /** The original characters the token came from, as string indices into the typed text. */
  start: number;
  end: number;
  /** The token in the normalized text. */
  normalizedStart: number;
  normalizedEnd: number;
  /**
   * The normalized text of the token's original characters. It reaches past the token when one typed character
   * expands into several words (ﷺ), so a fix of one of them has to rewrite that whole character.
   */
  spanStart: number;
  spanEnd: number;
}

export interface TokenizedText {
  normalized: string;
  tokens: Token[];
}

/** A token to change into another word. */
export interface Replacement {
  index: number;
  word: string;
}

/** A rewrite of part of the original text, made of the replacements whose original characters overlap. */
export interface Edit {
  start: number;
  end: number;
  replacement: string;
  /** The token indices of the replacements it makes. */
  indices: number[];
}

interface Cluster {
  start: number;
  end: number;
  normalizedStart: number;
  normalizedEnd: number;
}

/** The last cluster that starts at or before a normalized index, which skips clusters that normalized to nothing. */
function clusterAt(clusters: readonly Cluster[], normalizedIndex: number): Cluster {
  let low = 0;
  let high = clusters.length - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (clusters[middle]!.normalizedStart <= normalizedIndex) low = middle;
    else high = middle - 1;
  }
  return clusters[low]!;
}

/** Splits text the way tinyfix_ar.text.tokenize does, keeping where each token came from in the original text. */
export function tokenize(text: string): TokenizedText {
  const clusters: Cluster[] = [];
  let normalized = "";
  for (const match of text.matchAll(CLUSTER)) {
    const piece = normalize(match[0]);
    clusters.push({
      start: match.index,
      end: match.index + match[0].length,
      normalizedStart: normalized.length,
      normalizedEnd: normalized.length + piece.length,
    });
    normalized += piece;
  }

  const tokens: Token[] = [];
  for (const match of normalized.matchAll(TOKEN)) {
    const first = clusterAt(clusters, match.index);
    const last = clusterAt(clusters, match.index + match[0].length - 1);
    const groups = match.groups!;
    tokens.push({
      value: groups["number"] ? NUMBER_TOKEN : groups["latin"] ? FOREIGN_TOKEN : match[0],
      start: first.start,
      end: last.end,
      normalizedStart: match.index,
      normalizedEnd: match.index + match[0].length,
      spanStart: first.normalizedStart,
      spanEnd: last.normalizedEnd,
    });
  }
  return { normalized, tokens };
}

/**
 * Turns token replacements into edits of the original text. A replaced word takes the place of its original
 * characters; replacements that share an original character become one edit.
 */
export function edits({ normalized, tokens }: TokenizedText, replacements: readonly Replacement[]): Edit[] {
  const groups: Replacement[][] = [];
  for (const replacement of [...replacements].sort((first, second) => first.index - second.index)) {
    const group = groups.at(-1);
    if (group && tokens[group.at(-1)!.index]!.end > tokens[replacement.index]!.start) group.push(replacement);
    else groups.push([replacement]);
  }
  return groups.map((group) => {
    const first = tokens[group[0]!.index]!;
    const last = tokens[group.at(-1)!.index]!;
    let replacement = "";
    let consumed = first.spanStart;
    for (const { index, word } of group) {
      replacement += normalized.slice(consumed, tokens[index]!.normalizedStart) + word;
      consumed = tokens[index]!.normalizedEnd;
    }
    replacement += normalized.slice(consumed, last.spanEnd);
    return { start: first.start, end: last.end, replacement, indices: group.map(({ index }) => index) };
  });
}

/** Makes edits, which must be in text order and must not overlap. */
export function applyEdits(text: string, textEdits: readonly Pick<Edit, "start" | "end" | "replacement">[]): string {
  let result = "";
  let consumed = 0;
  for (const { start, end, replacement } of textEdits) {
    if (start < consumed) throw new Error("Edits must be in text order and must not overlap.");
    result += text.slice(consumed, start) + replacement;
    consumed = end;
  }
  return result + text.slice(consumed);
}
