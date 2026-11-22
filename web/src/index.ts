import { fetchFrom, readBundle } from "./bundle.js";
import { Engine, type Thresholds } from "./engine.js";
import { createModel, webRuntime, type OnnxRuntime } from "./runtime.js";
import { applyEdits, edits, tokenize } from "./text.js";

export { readBundle, type Bundle, type ReadFile } from "./bundle.js";
export type { Thresholds } from "./engine.js";
export type { OnnxRuntime } from "./runtime.js";
export { normalize, tokenize, type Token, type TokenizedText } from "./text.js";

/** One change tinyfix-ar makes, located in the text as it was typed. */
export interface Change {
  /** Where the changed characters start in the original text, as a string index. */
  start: number;
  end: number;
  /** The original characters, diacritics and all. */
  typed: string;
  replacement: string;
  /**
   * How much better the model scores the replacement than the typed word. When one typed character held several
   * changed words (ﷺ), the change rewrites it once and this is the smallest of their margins.
   */
  margin: number;
}

export interface FixResult {
  /** The original text with every change made, and everything else, spacing and diacritics included, left alone. */
  text: string;
  /** In text order, never overlapping. */
  changes: Change[];
}

/** What the model thinks of one word, reading the words around it as typed. */
export interface WordJudgement {
  start: number;
  end: number;
  typed: string;
  /** The candidate the model likes best, other than the typed word. */
  best: string;
  margin: number;
  /** The kind of edit from the typed word to best: "hamza", "taa_marbuta", "typo", … */
  kind: string;
  /** The typed word is a known misspelling (ان, الى), held to its own threshold. */
  misspelled: boolean;
  /** Whether the thresholds would swap best in. */
  changes: boolean;
}

export interface LoadOptions {
  /**
   * The folder holding the exported bundle: model.onnx, vocab.txt, the lexicon tables, kinds.json and manifest.json.
   * Defaults to the model repo on Hugging Face, at the revision this version of the package was tested with.
   */
  baseUrl?: string | URL;
  /** ONNX Runtime to run the model with. Defaults to onnxruntime-web's WebAssembly build. */
  ort?: OnnxRuntime;
  /** Where onnxruntime-web's .wasm files are, for apps that host them. */
  wasmPaths?: string;
  /** How many words after a word the model reads. Defaults to 2, what the thresholds are calibrated with. */
  visibleRight?: number;
  /** Overrides the thresholds in manifest.json. */
  thresholds?: Thresholds;
}

export interface TinyfixAr {
  /** Fixes a text left to right, reading each word after the fixes before it, as while typing. */
  fix(text: string): Promise<FixResult>;
  /** Judges every word the model can judge, reading the words around it as typed. Changes nothing. */
  judge(text: string): Promise<WordJudgement[]>;
  /** Frees the model. Nothing can be checked afterwards. */
  dispose(): Promise<void>;
}

const DEFAULT_BASE_URL = "https://huggingface.co/ali77a/tinyfix-ar/resolve/v1.0.0/";
const DEFAULT_VISIBLE_RIGHT = 2;
/** Fixer.batch_size. Margins depend on the batch (see Engine.fix), so it has to match. */
const BATCH_SIZE = 2048;

function resolveBase(baseUrl: string | URL): URL {
  const base = String(baseUrl).replace(/\/?$/, "/");
  return typeof location === "undefined" ? new URL(base) : new URL(base, location.href);
}

/** Loads the model and lexicon from an exported bundle. Call it once and keep the result. */
export async function loadTinyfixAr(options: LoadOptions = {}): Promise<TinyfixAr> {
  const [bundle, ort] = await Promise.all([
    readBundle(fetchFrom(resolveBase(options.baseUrl ?? DEFAULT_BASE_URL))),
    options.ort ?? webRuntime(options.wasmPaths),
  ]);
  const thresholds = options.thresholds ?? bundle.thresholds;
  if (!thresholds) throw new Error("The bundle has no thresholds: pass them in the options.");
  const model = await createModel(ort, bundle.model);
  const engine = new Engine(bundle.lexicon, model.score, {
    leftWindow: bundle.leftWindow,
    rightWindow: bundle.rightWindow,
    visibleRight: options.visibleRight ?? DEFAULT_VISIBLE_RIGHT,
    thresholds,
    batchSize: BATCH_SIZE,
  });

  return {
    async fix(text) {
      const tokenized = tokenize(text);
      const fixes = await engine.fix(tokenized.tokens.map((token) => token.value));
      const margins = new Map(fixes.map((judgement) => [judgement.position, judgement.margin]));
      const replacements = fixes.map((judgement) => ({ index: judgement.position, word: judgement.best }));
      const changes = edits(tokenized, replacements).map(({ start, end, replacement, indices }): Change => ({
        start,
        end,
        typed: text.slice(start, end),
        replacement,
        margin: Math.min(...indices.map((index) => margins.get(index)!)),
      }));
      return { text: applyEdits(text, changes), changes };
    },
    async judge(text) {
      const { tokens } = tokenize(text);
      const judgements = await engine.judge(tokens.map((token) => token.value));
      return judgements.map((judgement) => {
        const { start, end } = tokens[judgement.position]!;
        const { best, margin, kind, misspelled } = judgement;
        return {
          start,
          end,
          typed: text.slice(start, end),
          best,
          margin,
          kind,
          misspelled,
          changes: engine.changes(judgement),
        };
      });
    },
    dispose: () => model.release(),
  };
}
