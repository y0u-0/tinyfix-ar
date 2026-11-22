/** The model's vocabulary and, for each word, the words a writer may have meant: tinyfix_ar/lexicon.py, read from the bundle. */

import { PAD, isArabicWord } from "./text.js";

const UNK = 1;
const TYPED_KIND = 0;
const MIN_TARGET_LENGTH = 2;
const UNKNOWN_PREFIXES = ["وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل"] as const;
const UNKNOWN_SUFFIXES = ["ات", "ون", "ين", "ان", "ة", "ها", "هم", "ه", "ي", "ا"] as const;
const MIN_STEM_LENGTH = 2;

/** The vocabulary class of a word outside it, which keeps its clitics and ending. */
export function unknownClass(word: string): string {
  const prefix = UNKNOWN_PREFIXES.find((p) => word.startsWith(p) && word.length - p.length >= MIN_STEM_LENGTH) ?? "";
  const stem = word.slice(prefix.length);
  const suffix = UNKNOWN_SUFFIXES.find((s) => stem.endsWith(s) && stem.length - s.length >= MIN_STEM_LENGTH) ?? "";
  return `<unk:${prefix}-${suffix}>`;
}

export function isTarget(token: string): boolean {
  return token.length >= MIN_TARGET_LENGTH && isArabicWord(token);
}

export interface LexiconTables {
  words: readonly string[];
  /** [vocabulary, neighborsPerWord] word ids, PAD where a word has fewer neighbors. */
  neighbors: Uint32Array;
  /** The kind of edit linking each word to each of its neighbors, same shape as neighbors. */
  neighborKinds: Uint8Array;
  /** 0 for known misspellings (ان, الى): a typed word that is wrong whatever replaces it. */
  valid: Uint8Array;
  /** The name of each edit kind, by id: "typed", "hamza", "taa_marbuta", … */
  kindNames: readonly string[];
}

export class Lexicon {
  readonly words: readonly string[];
  readonly neighborsPerWord: number;
  private readonly ids: ReadonlyMap<string, number>;
  private readonly neighbors: Uint32Array;
  private readonly neighborKinds: Uint8Array;
  private readonly valid: Uint8Array;
  private readonly kindNames: readonly string[];

  constructor({ words, neighbors, neighborKinds, valid, kindNames }: LexiconTables) {
    if (words.length === 0 || neighbors.length % words.length !== 0) {
      throw new Error(`The neighbor table (${neighbors.length}) doesn't fit a vocabulary of ${words.length} words.`);
    }
    if (neighborKinds.length !== neighbors.length || valid.length !== words.length) {
      throw new Error("The lexicon tables have different sizes.");
    }
    this.words = words;
    this.neighborsPerWord = neighbors.length / words.length;
    this.ids = new Map(words.map((word, id) => [word, id]));
    this.neighbors = neighbors;
    this.neighborKinds = neighborKinds;
    this.valid = valid;
    this.kindNames = kindNames;
  }

  /** The vocabulary id of a word, or undefined when the word is outside the vocabulary. */
  idOf(token: string): number | undefined {
    return this.ids.get(token);
  }

  tokenId(token: string): number {
    const id = this.ids.get(token);
    if (id !== undefined) return id;
    return isArabicWord(token) ? (this.ids.get(unknownClass(token)) ?? UNK) : UNK;
  }

  /** Whether the model judges a typed word: an Arabic word in the vocabulary with at least one neighbor. */
  isJudged(token: string): boolean {
    const id = this.ids.get(token);
    return id !== undefined && isTarget(token) && this.neighbors[id * this.neighborsPerWord] !== PAD;
  }

  isValid(id: number): boolean {
    return this.valid[id] !== 0;
  }

  kindName(kind: number): string {
    const name = this.kindNames[kind];
    if (name === undefined) throw new Error(`Edit kind ${kind} is not in kinds.json.`);
    return name;
  }

  /** The typed word first, then its neighbors, padded with PAD; and the edit kind of each. */
  candidates(typedId: number): { ids: number[]; kinds: number[] } {
    const from = typedId * this.neighborsPerWord;
    const to = from + this.neighborsPerWord;
    return {
      ids: [typedId, ...this.neighbors.subarray(from, to)],
      kinds: [TYPED_KIND, ...this.neighborKinds.subarray(from, to)],
    };
  }
}
