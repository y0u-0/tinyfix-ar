/** Runs the exported ONNX model with whichever ONNX Runtime build the caller has: onnxruntime-web or onnxruntime-node. */

import type * as Ort from "onnxruntime-web/wasm";
import type { ModelInputs, Scorer } from "./engine.js";

/** The part of ONNX Runtime the engine uses. onnxruntime-web and onnxruntime-node both provide it. */
export type OnnxRuntime = Pick<typeof Ort, "InferenceSession" | "Tensor">;

export interface Model {
  score: Scorer;
  release(): Promise<void>;
}

/** The WebAssembly build of onnxruntime-web, on one thread so the page needs no cross-origin isolation headers. */
export async function webRuntime(wasmPaths: string | undefined): Promise<OnnxRuntime> {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.numThreads = 1;
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
  return ort;
}

export async function createModel(ort: OnnxRuntime, bytes: Uint8Array): Promise<Model> {
  const session = await ort.InferenceSession.create(bytes);
  return {
    async score(inputs: ModelInputs) {
      const tensor = (data: BigInt64Array) => new ort.Tensor("int64", data, [inputs.rows, data.length / inputs.rows]);
      const output = await session.run({
        left: tensor(inputs.left),
        right: tensor(inputs.right),
        candidates: tensor(inputs.candidates),
        kinds: tensor(inputs.kinds),
      });
      const scores = output["scores"];
      if (!(scores?.data instanceof Float32Array)) {
        throw new Error("The model didn't return float scores.");
      }
      return scores.data;
    },
    release: () => session.release(),
  };
}
