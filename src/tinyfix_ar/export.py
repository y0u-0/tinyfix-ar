"""Exports a run for the browser: python -m tinyfix_ar.export --run runs/v1 --output bundle

The bundle holds the quantized ONNX model and the lexicon tables as flat little-endian arrays,
so the app looks up candidates for a typed word without recomputing anything.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from tinyfix_ar.engine import INPUTS, Fixer, load_model, onnx_scorer, torch_scorer
from tinyfix_ar.lexicon import Lexicon
from tinyfix_ar.model import LEFT_WINDOW, RIGHT_WINDOW, WrongWordModel
from tinyfix_ar.qalb import aligned_sentences
from tinyfix_ar.text import PAD

PARITY_SPLIT = "zaebuc-dev"


def parity_inputs(fixer: Fixer) -> dict[str, np.ndarray]:
    """Real model inputs from held-out sentences, built exactly as the engine builds them."""
    batches = [inputs for _, inputs in (fixer.inputs(sentence.typed) for sentence in aligned_sentences(PARITY_SPLIT)) if inputs]
    return {name: np.concatenate([batch[name] for batch in batches]) for name in INPUTS}


def margins(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    scores = np.where(candidates != PAD, scores, -1e4)
    return scores[:, 1:].max(1) - scores[:, 0]


def export_model(model: WrongWordModel, fixer: Fixer, output: Path) -> dict:
    float_path, quantized_path = output / "model-fp32.onnx", output / "model.onnx"
    example = tuple(
        torch.ones(2, width, dtype=torch.long)
        for width in (LEFT_WINDOW, RIGHT_WINDOW, fixer.lexicon.neighbors.shape[1] + 1, fixer.lexicon.neighbors.shape[1] + 1)
    )
    torch.onnx.export(
        model.eval(),
        example,
        float_path,
        input_names=list(INPUTS),
        output_names=["scores"],
        dynamic_axes={name: {0: "batch"} for name in (*INPUTS, "scores")},
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(float_path, quantized_path, weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gather"])

    inputs = parity_inputs(fixer)
    reference = fixer.score(inputs)
    report = {"parity_cases": len(reference)}
    for path in (float_path, quantized_path):
        scores = onnx_scorer(path)(inputs)
        report[path.name] = {
            "mb": round(path.stat().st_size / 1e6, 2),
            "max_score_difference": round(float(np.abs(scores - reference).max()), 4),
            "pick_agreement": round(float((scores.argmax(1) == reference.argmax(1)).mean()), 4),
            "mean_margin_difference": round(float(np.abs(margins(scores, inputs["candidates"]) - margins(reference, inputs["candidates"])).mean()), 4),
        }
    float_path.unlink()
    return report


def export_lexicon(lexicon: Lexicon, output: Path) -> None:
    (output / "vocab.txt").write_text("\n".join(lexicon.words) + "\n", encoding="utf-8")
    lexicon.neighbors.astype("<u4").tofile(output / "neighbors.u32")
    lexicon.neighbor_kinds.astype("u1").tofile(output / "neighbor_kinds.u8")
    lexicon.valid.astype("u1").tofile(output / "valid.u8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = load_model(args.run)
    fixer = Fixer(torch_scorer(model), Lexicon.load(Path(json.loads((args.run / "config.json").read_text())["lexicon"])))
    args.output.mkdir(parents=True, exist_ok=True)
    report = export_model(model, fixer, args.output)
    export_lexicon(fixer.lexicon, args.output)
    lexicon_folder = Path(json.loads((args.run / "config.json").read_text())["lexicon"])
    shutil.copy(lexicon_folder / "kinds.json", args.output / "kinds.json")
    benchmark = args.run / "benchmark-right2.json"
    manifest = {
        "left_window": LEFT_WINDOW,
        "right_window": RIGHT_WINDOW,
        "candidates": fixer.lexicon.neighbors.shape[1] + 1,
        "vocabulary": len(fixer.lexicon.words),
        "thresholds": json.loads(benchmark.read_text())["thresholds"] if benchmark.exists() else None,
        "export": report,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
