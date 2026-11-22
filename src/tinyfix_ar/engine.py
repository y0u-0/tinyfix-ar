"""Runs the model the way the typing app does: score each word against its neighbors, swap on a clear win."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from tinyfix_ar.lexicon import KIND_IDS, TYPED_KIND, Lexicon, is_target
from tinyfix_ar.model import LEFT_WINDOW, RIGHT_WINDOW, WrongWordModel
from tinyfix_ar.text import PAD, START, tokenize

INPUTS = ("left", "right", "candidates", "kinds")
Scorer = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class Judgement:
    position: int
    typed: str
    best: str
    margin: float
    kind: str
    misspelled: bool


@dataclass(frozen=True)
class Thresholds:
    """How far the best candidate must beat the typed word before it is swapped in.

    A typed word that is a known misspelling is wrong whatever replaces it, so it has one
    threshold. For a valid typed word the risk depends on the kind of swap (a restored hamza is
    safer than a one-letter slip), so each kind has its own, with a default for the rest.
    """

    misspelled: float
    valid: dict[str, float]

    def changes(self, judgement: Judgement) -> bool:
        threshold = self.misspelled if judgement.misspelled else self.valid.get(judgement.kind, self.valid["default"])
        return judgement.margin >= threshold

    def to_json(self) -> dict:
        """Plain JSON: a threshold that never fires (infinity) is written as null."""
        return {"misspelled": _threshold_to_json(self.misspelled), "valid": {kind: _threshold_to_json(value) for kind, value in self.valid.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "Thresholds":
        return cls(_threshold_from_json(data["misspelled"]), {kind: _threshold_from_json(value) for kind, value in data["valid"].items()})


def _threshold_to_json(value: float) -> float | None:
    return None if value == math.inf else round(value, 4)


def _threshold_from_json(value: float | None) -> float:
    return math.inf if value is None else value


def torch_scorer(model: WrongWordModel) -> Scorer:
    model.eval()

    @torch.no_grad()
    def score(inputs: dict[str, np.ndarray]) -> np.ndarray:
        return model(*(torch.from_numpy(inputs[name]) for name in INPUTS)).numpy()

    return score


def onnx_scorer(path: Path) -> Scorer:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return lambda inputs: session.run(None, inputs)[0]


def load_model(run: Path) -> WrongWordModel:
    config = json.loads((run / "config.json").read_text())
    model = WrongWordModel(config["vocabulary_size"], config["embedding"], config["hidden"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model


class Fixer:
    def __init__(self, score: Scorer, lexicon: Lexicon, visible_right: int = 2, batch_size: int = 2048):
        self.score = score
        self.lexicon = lexicon
        self.kind_names = {index: kind.value for kind, index in KIND_IDS.items()} | {TYPED_KIND: "typed"}
        self.visible_right = visible_right
        self.batch_size = batch_size

    @classmethod
    def load(cls, run: Path, onnx: Path | None = None, **options) -> "Fixer":
        lexicon = Lexicon.load(Path(json.loads((run / "config.json").read_text())["lexicon"]))
        return cls(onnx_scorer(onnx) if onnx else torch_scorer(load_model(run)), lexicon, **options)

    def context(self, token_ids: list[int], position: int) -> tuple[list[int], list[int]]:
        left = [START, *token_ids][max(0, position + 1 - LEFT_WINDOW) : position + 1]
        left = [PAD] * (LEFT_WINDOW - len(left)) + left
        right = token_ids[position + 1 : position + 1 + min(self.visible_right, RIGHT_WINDOW)]
        right = [PAD] * (RIGHT_WINDOW - len(right)) + right[::-1]
        return left, right

    def inputs(self, tokens: list[str]) -> tuple[list[int], dict[str, np.ndarray]]:
        """Model inputs for every word that has neighbors, reading the words around it as typed."""
        token_ids = [self.lexicon.token_id(token) for token in tokens]
        positions, rows = [], []
        for position, (token, token_id) in enumerate(zip(tokens, token_ids, strict=True)):
            if is_target(token) and token in self.lexicon.ids and self.lexicon.neighbors[token_id, 0] != PAD:
                positions.append(position)
                rows.append((*self.context(token_ids, position), *self.lexicon.candidates(token_id)))
        if not rows:
            return [], {}
        return positions, {name: np.array(column, dtype=np.int64) for name, column in zip(INPUTS, zip(*rows, strict=True), strict=True)}

    def judge(self, tokens: list[str]) -> list[Judgement]:
        positions, inputs = self.inputs(tokens)
        results = []
        for start in range(0, len(positions), self.batch_size):
            chunk = {name: values[start : start + self.batch_size] for name, values in inputs.items()}
            scores = np.where(chunk["candidates"] != PAD, self.score(chunk), -1e4)
            best = scores[:, 1:].argmax(1)
            margins = scores[np.arange(len(best)), best + 1] - scores[:, 0]
            for row, position in enumerate(positions[start : start + self.batch_size]):
                typed_id = chunk["candidates"][row, 0]
                results.append(
                    Judgement(
                        position,
                        tokens[position],
                        self.lexicon.words[chunk["candidates"][row, best[row] + 1]],
                        float(margins[row]),
                        self.kind_names[chunk["kinds"][row, best[row] + 1]],
                        not self.lexicon.valid[typed_id],
                    )
                )
        return results

    def fix(self, text: str, thresholds: Thresholds) -> list[str]:
        tokens = tokenize(text)
        for judgement in self.judge(tokens):
            if thresholds.changes(judgement):
                tokens[judgement.position] = judgement.best
        return tokens
