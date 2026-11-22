import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import type { Engine } from "../src/engine.js";
import type { Model } from "../src/runtime.js";
import { tokenize } from "../src/text.js";
import type { Lexicon } from "../src/lexicon.js";
import { fixtures, nodeEngine, readBundleFromDisk, type Case } from "./helpers.js";

/** The ONNX model run by onnxruntime-node here and onnxruntime in Python. */
const MARGIN_TOLERANCE = 1e-3;

const cases: Array<[group: string, Case[]]> = [
  ["edge cases", fixtures.edge_cases],
  ["QALB-2014 dev sentences", fixtures.sentences],
];
const values = (text: string) => tokenize(text).tokens.map((token) => token.value);

describe("tokens and token ids match tinyfix_ar", () => {
  let lexicon: Lexicon;
  before(async () => ({ lexicon } = await readBundleFromDisk()));
  for (const [group, groupCases] of cases) {
    it(`for ${groupCases.length} ${group}`, () => {
      for (const { text, tokens, token_ids } of groupCases) {
        assert.deepEqual(values(text), tokens, `tokens of ${JSON.stringify(text)}`);
        assert.deepEqual(
          tokens.map((token) => lexicon.tokenId(token)),
          token_ids,
          `token ids of ${JSON.stringify(text)}`,
        );
      }
    });
  }
});

describe("model inputs, margins and fixes match tinyfix_ar.engine.Fixer", () => {
  let engine: Engine;
  let model: Model;
  before(async () => ({ engine, model } = await nodeEngine()));
  after(() => model.release());

  for (const [group, groupCases] of cases) {
    it(`builds the same model inputs for ${group}`, () => {
      for (const { text, tokens, judge } of groupCases) {
        const positions = engine.judgedPositions(tokens);
        assert.deepEqual(positions, judge.positions, `positions in ${JSON.stringify(text)}`);
        const inputs = engine.inputs(tokens, positions);
        for (const name of ["left", "right", "candidates", "kinds"] as const) {
          assert.deepEqual(Array.from(inputs[name], Number), judge[name], `${name} in ${JSON.stringify(text)}`);
        }
      }
    });

    it(`scores every word of ${group} within ${MARGIN_TOLERANCE} and picks the same candidate`, async (t) => {
      let worst = 0;
      for (const { text, tokens, judge } of groupCases) {
        const judgements = await engine.judge(tokens);
        assert.deepEqual(
          judgements.map((judgement) => judgement.best),
          judge.best,
          `best candidates in ${JSON.stringify(text)}`,
        );
        assert.deepEqual(
          judgements.map((judgement) => judgement.kind),
          judge.best_kinds,
          `kinds of the best candidates in ${JSON.stringify(text)}`,
        );
        for (const [row, judgement] of judgements.entries()) {
          worst = Math.max(worst, Math.abs(judgement.margin - judge.margins[row]!));
        }
      }
      t.diagnostic(`largest margin difference ${worst.toExponential(2)}`);
      assert.ok(worst <= MARGIN_TOLERANCE, `largest margin difference ${worst}`);
    });

    it(`fixes ${group} word for word like the Python reference`, async (t) => {
      let worst = 0;
      for (const { text, tokens, changes } of groupCases) {
        const fixes = await engine.fix(tokens);
        assert.deepEqual(
          fixes.map(({ position, best }) => ({ position, best })),
          changes.map(({ position, best }) => ({ position, best })),
          `changes in ${JSON.stringify(text)}`,
        );
        for (const [index, fix] of fixes.entries())
          worst = Math.max(worst, Math.abs(fix.margin - changes[index]!.margin));
      }
      t.diagnostic(
        `${groupCases.reduce((total, { changes }) => total + changes.length, 0)} changes, largest margin difference ${worst.toExponential(2)}`,
      );
      assert.ok(worst <= MARGIN_TOLERANCE, `largest margin difference ${worst}`);
    });
  }

  it("covers sentences where fixing left to right reads differently from fixing as typed", () => {
    assert.ok(fixtures.sentences.some((sentence) => !sentence.fixed_like_batch));
  });
});
