import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import * as ortNode from "onnxruntime-node";
import { toThresholds } from "../src/bundle.js";
import { loadTinyfixAr, tokenize, type TinyfixAr } from "../src/index.js";
import { fixtures, serveBundle, thresholds } from "./helpers.js";

const MARGIN_TOLERANCE = 1e-3;

let server: Awaited<ReturnType<typeof serveBundle>>;
let tinyfix: TinyfixAr;

before(async () => {
  server = await serveBundle();
  tinyfix = await loadTinyfixAr({ baseUrl: server.baseUrl, ort: ortNode, thresholds });
});
after(async () => {
  await tinyfix.dispose();
  await server.close();
});

describe("fix", () => {
  it("fixes QALB-2014 dev sentences into the words the Python reference fixes them to", async () => {
    for (const { text, tokens, changes } of fixtures.sentences) {
      const expected = [...tokens];
      for (const { position, best } of changes) expected[position] = best;
      const result = await tinyfix.fix(text);
      assert.deepEqual(
        tokenize(result.text).tokens.map((token) => token.value),
        expected,
        JSON.stringify(text),
      );
    }
  });

  it("reports each change where it is in the typed text, and leaves the rest of the text as it was", async () => {
    for (const { text } of fixtures.sentences.slice(0, 50)) {
      const result = await tinyfix.fix(text);
      let rebuilt = "";
      let consumed = 0;
      for (const change of result.changes) {
        assert.equal(text.slice(change.start, change.end), change.typed);
        assert.ok(change.start >= consumed, "changes are in text order and don't overlap");
        rebuilt += text.slice(consumed, change.start) + change.replacement;
        consumed = change.end;
      }
      assert.equal(rebuilt + text.slice(consumed), result.text);
    }
  });

  it("keeps the diacritics, spacing and punctuation of the words it doesn't change", async () => {
    const text = "ذهبتُ  الى المدرسةِ صباحاً،\tوقال انه سيعود .";
    const result = await tinyfix.fix(text);
    assert.deepEqual(
      result.changes.map(({ typed, replacement }) => [typed, replacement]),
      [
        ["الى", "إلى"],
        ["انه", "إنه"],
      ],
    );
    assert.equal(result.text, "ذهبتُ  إلى المدرسةِ صباحاً،\tوقال إنه سيعود .");
  });

  it("rewrites a changed word's own diacritics and tatweel along with it", async () => {
    const result = await tinyfix.fix("ذهبتُ الـــى المدرسةِ");
    assert.equal(result.text, "ذهبتُ إلى المدرسةِ");
    assert.deepEqual(
      result.changes.map(({ start, end, typed }) => [start, end, typed]),
      [[6, 12, "الـــى"]],
    );
  });

  it("leaves text without Arabic words alone", async () => {
    for (const text of ["", "   ", "hello world 123", "😀 ✓"])
      assert.deepEqual(await tinyfix.fix(text), { text, changes: [] });
  });
});

describe("judge", () => {
  it("judges every word as Fixer.judge does, at its place in the typed text", async () => {
    for (const { text, judge } of fixtures.sentences.slice(0, 50)) {
      const judgements = await tinyfix.judge(text);
      const { tokens } = tokenize(text);
      assert.deepEqual(
        judgements.map(({ start }) => start),
        judge.positions.map((position) => tokens[position]!.start),
      );
      assert.deepEqual(
        judgements.map(({ best }) => best),
        judge.best,
      );
      for (const [row, judgement] of judgements.entries()) {
        assert.ok(Math.abs(judgement.margin - judge.margins[row]!) <= MARGIN_TOLERANCE);
      }
    }
  });
});

describe("loading", () => {
  it("reads per-kind thresholds, with null for a kind of swap that is never made", () => {
    assert.deepEqual(toThresholds({ misspelled: 3, valid: { hamza: 1, typo: null, default: 4 } }), {
      misspelled: 3,
      valid: { hamza: 1, typo: Infinity, default: 4 },
    });
    assert.throws(() => toThresholds({ misspelled: 3, valid: { hamza: 1 } }));
  });

  it("uses the bundle's thresholds unless told otherwise", async () => {
    const withManifestThresholds = await loadTinyfixAr({ baseUrl: server.baseUrl, ort: ortNode });
    try {
      assert.ok((await withManifestThresholds.judge("ذهبت الى البيت")).length > 0);
    } finally {
      await withManifestThresholds.dispose();
    }
  });

  it("says which file is missing", async () => {
    await assert.rejects(loadTinyfixAr({ baseUrl: `${server.baseUrl}missing/`, ort: ortNode }), /manifest\.json/);
  });

  it("loads from a pinned revision of the Hugging Face model repo when given no baseUrl", async (t) => {
    const requested: string[] = [];
    t.mock.method(globalThis, "fetch", async (url: URL) => {
      requested.push(url.href);
      return new Response(null, { status: 404 });
    });
    await assert.rejects(loadTinyfixAr({ ort: ortNode }));
    assert.ok(requested.length > 0);
    for (const url of requested) {
      assert.match(url, /^https:\/\/huggingface\.co\/ali77a\/tinyfix-ar\/resolve\/v\d+\.\d+\.\d+\/[\w.]+$/);
    }
  });
});

/**
 * onnxruntime-web's WebAssembly int8 kernels round differently from the CPU kernels onnxruntime-node and Python use, so
 * its margins are close to the Python engine's rather than equal to them.
 */
describe("onnxruntime-web, as in the browser", () => {
  const MEAN_MARGIN_TOLERANCE = 0.01;
  const MIN_BEST_AGREEMENT = 0.995;

  it("fixes QALB-2014 dev sentences like the Python reference, with margins close to it", async (t) => {
    const web = await loadTinyfixAr({ baseUrl: server.baseUrl, thresholds });
    let rows = 0;
    let marginDifference = 0;
    let worst = 0;
    let sameBest = 0;
    try {
      for (const { text, tokens, judge, changes } of fixtures.sentences) {
        for (const [row, judgement] of (await web.judge(text)).entries()) {
          const difference = Math.abs(judgement.margin - judge.margins[row]!);
          rows++;
          marginDifference += difference;
          worst = Math.max(worst, difference);
          sameBest += Number(judgement.best === judge.best[row]);
        }
        const expected = [...tokens];
        for (const { position, best } of changes) expected[position] = best;
        assert.deepEqual(
          tokenize((await web.fix(text)).text).tokens.map((token) => token.value),
          expected,
          JSON.stringify(text),
        );
      }
    } finally {
      await web.dispose();
    }
    const mean = marginDifference / rows;
    t.diagnostic(
      `${rows} words: mean margin difference ${mean.toExponential(2)}, largest ${worst.toFixed(3)}, same best ${(sameBest / rows).toFixed(4)}`,
    );
    assert.ok(mean <= MEAN_MARGIN_TOLERANCE, `mean margin difference ${mean}`);
    assert.ok(sameBest / rows >= MIN_BEST_AGREEMENT, `same best candidate for ${sameBest} of ${rows} words`);
  });
});
