"""Scores the typed word and its neighbors against the words on both sides.

Same shape as tinyfix (two small GRUs, dot product against candidate embeddings), plus two things
Arabic needs: each candidate carries the kind of edit that links it to the typed word (a restored
hamza is a far likelier fix than a one-letter slip), and the context summary sees the typed word.
"""

import torch
import torch.nn as nn

from tinyfix_ar.lexicon import KIND_COUNT, TYPED_KIND
from tinyfix_ar.text import PAD

LEFT_WINDOW = 16
RIGHT_WINDOW = 4


class MatMulLinear(nn.Module):
    """Linear layer that exports as MatMul so ONNX Runtime dynamic quantization applies."""

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(inputs, outputs)))
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.matmul(self.weight) + self.bias


class Gru(nn.Module):
    """GRU built from MatMul layers because ONNX Runtime cannot quantize its GRU operator."""

    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.reset_gate = MatMulLinear(inputs + hidden, hidden)
        self.update_gate = MatMulLinear(inputs + hidden, hidden)
        self.proposal_input = MatMulLinear(inputs, hidden)
        self.proposal_state = MatMulLinear(hidden, hidden)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        state = sequence[:, 0, :1].mul(0).expand(-1, self.hidden)
        for step in range(sequence.shape[1]):
            current = sequence[:, step, :]
            joined = torch.cat([current, state], dim=-1)
            reset = torch.sigmoid(self.reset_gate(joined))
            update = torch.sigmoid(self.update_gate(joined))
            proposal = torch.tanh(self.proposal_input(current) + reset * self.proposal_state(state))
            state = (1 - update) * proposal + update * state
        return state


class WrongWordModel(nn.Module):
    def __init__(self, vocabulary_size: int, embedding: int, hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding, padding_idx=PAD)
        self.word_bias = nn.Embedding(vocabulary_size, 1, padding_idx=PAD)
        self.kind_embedding = nn.Embedding(KIND_COUNT, embedding)
        self.kind_bias = nn.Embedding(KIND_COUNT, 1)
        self.left = Gru(embedding, hidden)
        self.right = Gru(embedding, hidden)
        self.mix = MatMulLinear(hidden * 2 + embedding, hidden)
        self.project = MatMulLinear(hidden, embedding)
        nn.init.zeros_(self.kind_embedding.weight)
        nn.init.zeros_(self.kind_bias.weight)
        with torch.no_grad():
            self.kind_bias.weight[TYPED_KIND] = 1.0

    def forward(self, left: torch.Tensor, right: torch.Tensor, candidates: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        typed = self.embedding(candidates[:, 0])
        context = torch.cat([self.left(self.embedding(left)), self.right(self.embedding(right)), typed], dim=-1)
        summary = self.project(torch.tanh(self.mix(context)))
        vectors = self.embedding(candidates) + self.kind_embedding(kinds)
        scores = (vectors * summary.unsqueeze(1)).sum(-1)
        return scores + self.word_bias(candidates).squeeze(-1) + self.kind_bias(kinds).squeeze(-1)


def mask_empty(scores: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(candidates == PAD, -1e4)
