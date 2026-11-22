/** Reads a bundle written by tinyfix_ar.export: the ONNX model, the lexicon tables and manifest.json. */

import type { Thresholds } from "./engine.js";
import { Lexicon } from "./lexicon.js";

/** Reads one file of the bundle. */
export type ReadFile = (file: string) => Promise<Uint8Array>;

export interface Bundle {
  leftWindow: number;
  rightWindow: number;
  /** The calibrated thresholds, or null when the bundle was exported without a benchmark. */
  thresholds: Thresholds | null;
  lexicon: Lexicon;
  model: Uint8Array;
}

/** The thresholds benchmark.py calibrates, as Thresholds.to_json writes them: null is a threshold that never fires. */
interface ThresholdsJson {
  misspelled: number | null;
  valid: Record<string, number | null>;
}

interface Manifest {
  left_window: number;
  right_window: number;
  candidates: number;
  vocabulary: number;
  thresholds: ThresholdsJson | null;
}

const neverWhenNull = (value: number | null): number => value ?? Infinity;

export function toThresholds(thresholds: ThresholdsJson): Thresholds {
  const valid = Object.fromEntries(
    Object.entries(thresholds.valid).map(([kind, value]) => [kind, neverWhenNull(value)]),
  );
  if (valid["default"] === undefined) throw new Error("The thresholds have no default for valid words.");
  return { misspelled: neverWhenNull(thresholds.misspelled), valid: valid as Thresholds["valid"] };
}

function readUint32LittleEndian(bytes: Uint8Array): Uint32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Uint32Array.from({ length: bytes.byteLength / 4 }, (_, index) => view.getUint32(index * 4, true));
}

function parseVocabulary(text: string, expected: number): string[] {
  const words = text.split("\n");
  if (words.at(-1) === "") words.pop();
  if (words.length !== expected) throw new Error(`vocab.txt has ${words.length} words, the manifest says ${expected}.`);
  return words;
}

/** Kind names by id, from kinds.json's name → id map. */
function parseKinds(kinds: Record<string, number>): string[] {
  const names: string[] = [];
  for (const [name, id] of Object.entries(kinds)) names[id] = name;
  return names;
}

export async function readBundle(read: ReadFile): Promise<Bundle> {
  const decoder = new TextDecoder();
  const readText = async (file: string) => decoder.decode(await read(file));
  const [manifestText, kindsText, vocabulary, neighbors, neighborKinds, valid, model] = await Promise.all([
    readText("manifest.json"),
    readText("kinds.json"),
    readText("vocab.txt"),
    read("neighbors.u32"),
    read("neighbor_kinds.u8"),
    read("valid.u8"),
    read("model.onnx"),
  ]);
  const manifest = JSON.parse(manifestText) as Manifest;
  const lexicon = new Lexicon({
    words: parseVocabulary(vocabulary, manifest.vocabulary),
    neighbors: readUint32LittleEndian(neighbors),
    neighborKinds,
    valid,
    kindNames: parseKinds(JSON.parse(kindsText) as Record<string, number>),
  });
  if (lexicon.neighborsPerWord + 1 !== manifest.candidates) {
    throw new Error(
      `The lexicon has ${lexicon.neighborsPerWord} neighbors per word, the manifest says ${manifest.candidates - 1}.`,
    );
  }
  return {
    leftWindow: manifest.left_window,
    rightWindow: manifest.right_window,
    thresholds: manifest.thresholds && toThresholds(manifest.thresholds),
    lexicon,
    model,
  };
}

/** Reads bundle files over HTTP from a folder URL. */
export function fetchFrom(base: URL): ReadFile {
  return async (file) => {
    const response = await fetch(new URL(file, base));
    if (!response.ok) throw new Error(`tinyfix-ar couldn't load ${file} from ${base.href} (${response.status}).`);
    return new Uint8Array(await response.arrayBuffer());
  };
}
