import { readFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import * as ortNode from "onnxruntime-node";
import { readBundle, toThresholds, type Bundle } from "../src/bundle.js";
import { Engine, type Thresholds } from "../src/engine.js";
import { createModel, type Model } from "../src/runtime.js";

/** The exported bundle in the repository, which is far too large to copy into the package. */
export const BUNDLE = new URL("../../model/", import.meta.url);

export interface Fixtures {
  source: { visible_right: number };
  thresholds: { misspelled: number | null; valid: Record<string, number | null> };
  characters: {
    unicode_version: string;
    unassigned: Array<[number, number]>;
    word: Array<[number, number]>;
    space: Array<[number, number]>;
    digit: Array<[number, number]>;
  };
  edge_cases: Case[];
  sentences: Case[];
}

export interface Case {
  text: string;
  tokens: string[];
  token_ids: number[];
  judge: {
    positions: number[];
    left: number[];
    right: number[];
    candidates: number[];
    kinds: number[];
    best: string[];
    best_kinds: string[];
    margins: number[];
  };
  changes: Array<{ position: number; best: string; margin: number }>;
  fixed_like_batch: boolean;
}

/** Written by make-parity-fixtures.py; a null threshold is a kind of swap that is never made. */
export const fixtures = JSON.parse(readFileSync(new URL("fixtures.json", import.meta.url), "utf-8")) as Fixtures;

/** The thresholds the fixtures were fixed with: the run's per-kind calibration. */
export const thresholds: Thresholds = toThresholds(fixtures.thresholds);

export const readBundleFromDisk = (): Promise<Bundle> =>
  readBundle(async (file) => new Uint8Array(await readFile(new URL(file, BUNDLE))));

/** The engine on onnxruntime-node, reading the bundle straight from disk. */
export async function nodeEngine(): Promise<{ engine: Engine; model: Model; bundle: Bundle }> {
  const bundle = await readBundleFromDisk();
  const model = await createModel(ortNode, bundle.model);
  const engine = new Engine(bundle.lexicon, model.score, {
    leftWindow: bundle.leftWindow,
    rightWindow: bundle.rightWindow,
    visibleRight: fixtures.source.visible_right,
    thresholds,
    batchSize: 2048,
  });
  return { engine, model, bundle };
}

/** Serves the bundle over HTTP, as a site hosting it would. */
export async function serveBundle(): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server: Server = createServer((request, response) => {
    readFile(new URL(`.${request.url ?? "/"}`, BUNDLE)).then(
      (contents) => response.end(contents),
      () => {
        response.statusCode = 404;
        response.end();
      },
    );
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${port}/`,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}
