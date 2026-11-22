/** Scores each word against its neighbors, as tinyfix_ar/engine.py does, and fixes text the way the typing app does. */

import type { Lexicon } from "./lexicon.js";
import { PAD, START } from "./text.js";

/** Rows of token ids, one per judged word, flattened row by row. */
export interface ModelInputs {
  rows: number;
  left: BigInt64Array;
  right: BigInt64Array;
  candidates: BigInt64Array;
  kinds: BigInt64Array;
}

/** Runs the model: a score for each candidate of each row, flattened row by row. */
export type Scorer = (inputs: ModelInputs) => Promise<Float32Array>;

/** How far the best candidate must beat the typed word before it is swapped in, as in tinyfix_ar.engine.Thresholds. */
export interface Thresholds {
  /** For a known misspelling (ان, الى), which is wrong whatever replaces it. */
  misspelled: number;
  /** For a valid word, by the kind of swap ("hamza", "typo", …), with a default for kinds not listed. */
  valid: Readonly<Record<string, number>> & { readonly default: number };
}

export interface EngineSettings {
  leftWindow: number;
  rightWindow: number;
  /** How many words after a word the model reads. The app sees only what has been typed so far. */
  visibleRight: number;
  thresholds: Thresholds;
  batchSize: number;
}

export interface Judgement {
  /** Index of the word in the token list. */
  position: number;
  typed: string;
  /** The candidate the model likes best, other than the typed word. */
  best: string;
  /** How much better best scores than the typed word. */
  margin: number;
  /** The kind of edit that links best to the typed word. */
  kind: string;
  /** The typed word is a known misspelling. */
  misspelled: boolean;
}

/** The value numpy uses for a padded candidate, so it never wins. */
const PADDED_SCORE = -1e4;
const PAD_ID = BigInt(PAD);

export class Engine {
  constructor(
    private readonly lexicon: Lexicon,
    private readonly score: Scorer,
    private readonly settings: EngineSettings,
  ) {}

  /** The words before a word, oldest first after START, and the visible words after it, nearest last. */
  context(tokenIds: readonly number[], position: number): { left: number[]; right: number[] } {
    const { leftWindow, rightWindow, visibleRight } = this.settings;
    const left = [START, ...tokenIds].slice(Math.max(0, position + 1 - leftWindow), position + 1);
    const right = tokenIds.slice(position + 1, position + 1 + Math.min(visibleRight, rightWindow)).reverse();
    return {
      left: [...Array<number>(leftWindow - left.length).fill(PAD), ...left],
      right: [...Array<number>(rightWindow - right.length).fill(PAD), ...right],
    };
  }

  /** The words the model judges: Arabic words in the vocabulary that have neighbors. */
  judgedPositions(tokens: readonly string[]): number[] {
    return tokens.flatMap((token, position) => (this.lexicon.isJudged(token) ? [position] : []));
  }

  /** Model inputs for the words at these positions, reading the words around them as they are in tokens. */
  inputs(tokens: readonly string[], positions: readonly number[]): ModelInputs {
    const tokenIds = tokens.map((token) => this.lexicon.tokenId(token));
    const rows = positions.map((position) => {
      const { left, right } = this.context(tokenIds, position);
      const { ids, kinds } = this.lexicon.candidates(tokenIds[position]!);
      return { left, right, candidates: ids, kinds };
    });
    const column = (name: keyof (typeof rows)[number]) =>
      BigInt64Array.from(
        rows.flatMap((row) => row[name]),
        BigInt,
      );
    return {
      rows: rows.length,
      left: column("left"),
      right: column("right"),
      candidates: column("candidates"),
      kinds: column("kinds"),
    };
  }

  /** Judges every word with the words around it as they are in tokens, like Fixer.judge, in batches of the same size. */
  async judge(tokens: readonly string[]): Promise<Judgement[]> {
    const positions = this.judgedPositions(tokens);
    const judgements: Judgement[] = [];
    for (let start = 0; start < positions.length; start += this.settings.batchSize) {
      const batch = positions.slice(start, start + this.settings.batchSize);
      const inputs = this.inputs(tokens, batch);
      judgements.push(...this.read(tokens, batch, inputs, await this.score(inputs)));
    }
    return judgements;
  }

  /**
   * The judgements that change the text, made left to right: each word is judged after the words before it have been
   * fixed, and against the words after it as typed.
   *
   * The whole text is judged again after every change, not just the words that read the changed one. The int8 model
   * quantizes its activations per batch, so a word's margin depends on the words scored alongside it, and only the
   * same batches give the same margins as the Python engine.
   */
  async fix(tokens: readonly string[]): Promise<Judgement[]> {
    const fixed = [...tokens];
    let judgements = await this.judge(fixed);
    const changes: Judgement[] = [];
    for (let index = 0; index < judgements.length; index++) {
      const judgement = judgements[index]!;
      if (!this.changes(judgement)) continue;
      changes.push(judgement);
      fixed[judgement.position] = judgement.best;
      const later = (await this.judge(fixed)).filter(({ position }) => position > judgement.position);
      judgements = [...judgements.slice(0, index + 1), ...later];
    }
    return changes;
  }

  /** Whether the thresholds swap the best candidate in. */
  changes(judgement: Judgement): boolean {
    const { valid, misspelled } = this.settings.thresholds;
    return judgement.margin >= (judgement.misspelled ? misspelled : (valid[judgement.kind] ?? valid.default));
  }

  private read(
    tokens: readonly string[],
    positions: readonly number[],
    inputs: ModelInputs,
    scores: Float32Array,
  ): Judgement[] {
    const width = this.lexicon.neighborsPerWord + 1;
    if (scores.length !== positions.length * width) {
      throw new Error(
        `The model returned ${scores.length} scores for ${positions.length} words of ${width} candidates.`,
      );
    }
    return positions.map((position, row) => {
      const scoreAt = (column: number) =>
        inputs.candidates[row * width + column] === PAD_ID ? PADDED_SCORE : scores[row * width + column]!;
      let best = 1;
      for (let column = 2; column < width; column++) if (scoreAt(column) > scoreAt(best)) best = column;
      const typed = tokens[position]!;
      return {
        position,
        typed,
        best: this.lexicon.words[Number(inputs.candidates[row * width + best])]!,
        margin: Math.fround(scoreAt(best) - scoreAt(0)),
        kind: this.lexicon.kindName(Number(inputs.kinds[row * width + best])),
        misspelled: !this.lexicon.isValid(this.lexicon.idOf(typed)!),
      };
    });
  }
}
