"""Draws training cases by swapping words in clean text for the words Arabic writers type instead."""

from dataclasses import dataclass, field

import numpy as np
import torch

from tinyfix_ar.lexicon import KIND_COUNT, KIND_IDS, Lexicon, is_target
from tinyfix_ar.model import LEFT_WINDOW, RIGHT_WINDOW
from tinyfix_ar.orthography import ErrorKind
from tinyfix_ar.text import PAD, START

DEFAULT_KIND_WEIGHTS = {
    ErrorKind.HAMZA: 3.0,
    ErrorKind.TAA_MARBUTA: 2.0,
    ErrorKind.ALIF_MAQSURA: 2.0,
    ErrorKind.ALIF_FARIQA: 1.0,
    ErrorKind.SPELLING_MIXED: 1.0,
    ErrorKind.SOUND_ALIKE: 1.0,
    ErrorKind.CLITIC: 0.3,
    ErrorKind.TYPO: 0.5,
    ErrorKind.OTHER: 1.0,
}


@dataclass
class SamplerOptions:
    corrupt_rate: float = 0.5
    real_share: float = 0.6
    group_share: float = 0.25
    group_power: float = 0.5
    short_share: float = 0.2
    context_noise: float = 0.03
    kind_weights: dict[ErrorKind, float] = field(default_factory=lambda: dict(DEFAULT_KIND_WEIGHTS))


@dataclass
class Batch:
    left: torch.Tensor
    right: torch.Tensor
    candidates: torch.Tensor
    kinds: torch.Tensor
    label: torch.Tensor
    is_corrupted: torch.Tensor


class Sampler:
    def __init__(self, tokens: torch.Tensor, lexicon: Lexicon, options: SamplerOptions, device: str):
        self.tokens = tokens.to(device)
        self.options = options
        self.device = device
        self.neighbors = torch.from_numpy(lexicon.neighbors).to(device).long()
        self.neighbor_kinds = torch.from_numpy(lexicon.neighbor_kinds).to(device).long()
        self.neighbor_counts = (self.neighbors != PAD).sum(1)
        kind_weight = torch.zeros(KIND_COUNT, device=device)
        for kind, weight in options.kind_weights.items():
            kind_weight[KIND_IDS[kind]] = weight
        weights = kind_weight[self.neighbor_kinds] * (self.neighbors != PAD)
        self.neighbor_weights = torch.where(weights.sum(1, keepdim=True) > 0, weights, torch.ones_like(weights))
        self.mistakes = torch.from_numpy(lexicon.mistakes).to(device).long()
        mistake_weights = torch.from_numpy(lexicon.mistake_weights).to(device)
        self.has_mistakes = mistake_weights.sum(1) > 0
        self.mistake_weights = torch.where(self.has_mistakes.unsqueeze(1), mistake_weights, torch.ones_like(mistake_weights))
        self.is_target = torch.tensor([is_target(word) for word in lexicon.words], device=device)
        self.valid = torch.from_numpy(lexicon.valid).to(device)
        self._index_groups(lexicon)

    def _index_groups(self, lexicon: Lexicon) -> None:
        """Positions of words that take part in a known mix-up, so a share of cases can center on them."""
        ids = lexicon.ids
        group_words = {ids[word] for typed, intended in lexicon.confusions.items() for word in (typed, *intended)}
        if not group_words or self.options.group_share <= 0:
            self.group_positions = None
            return
        tokens = self.tokens.cpu().numpy()
        where = np.nonzero(np.isin(tokens, np.fromiter(group_words, dtype=tokens.dtype)))[0]
        where = where[(where > LEFT_WINDOW) & (where < len(tokens) - RIGHT_WINDOW - 1)]
        where = where[np.argsort(tokens[where], kind="stable")]
        type_counts = np.bincount(tokens[where], minlength=len(lexicon.words))
        self.group_positions = torch.from_numpy(where).to(self.device)
        self.group_starts = torch.from_numpy(np.cumsum(type_counts) - type_counts).to(self.device)
        self.group_counts = torch.from_numpy(type_counts).to(self.device)
        self.group_type_weights = self.group_counts.float().pow(self.options.group_power)

    def _positions(self, size: int) -> torch.Tensor:
        positions = torch.randint(LEFT_WINDOW + 1, len(self.tokens) - RIGHT_WINDOW - 1, (size,), device=self.device)
        if self.group_positions is None:
            return positions
        types = torch.multinomial(self.group_type_weights, size, replacement=True)
        offsets = (torch.rand(size, device=self.device) * self.group_counts[types]).long()
        chosen = self.group_positions[(self.group_starts[types] + offsets).clamp(max=len(self.group_positions) - 1)]
        return torch.where(torch.rand(size, device=self.device) < self.options.group_share, chosen, positions)

    def corrupt(self, words: torch.Tensor) -> torch.Tensor:
        """A plausible mistake for each word: a real one from QALB when known, else a weighted neighbor."""
        neighbor_choice = torch.multinomial(self.neighbor_weights[words], 1).squeeze(1)
        corrupted = self.neighbors[words, neighbor_choice]
        mistake_choice = torch.multinomial(self.mistake_weights[words], 1).squeeze(1)
        uses_real = self.has_mistakes[words] & (torch.rand(len(words), device=self.device) < self.options.real_share)
        corrupted = torch.where(uses_real, self.mistakes[words, mistake_choice], corrupted)
        return torch.where(corrupted == PAD, words, corrupted)

    def _noisy_context(self, context: torch.Tensor) -> torch.Tensor:
        if self.options.context_noise <= 0:
            return context
        flat = context.reshape(-1)
        noisy = (torch.rand(flat.shape, device=self.device) < self.options.context_noise) & self.is_target[flat] & (self.neighbor_counts[flat] > 0)
        return torch.where(noisy, self.corrupt(flat), flat).reshape(context.shape)

    def batch(self, size: int, corrupt_rate: float | None = None, context_noise: bool = True) -> Batch:
        corrupt_rate = self.options.corrupt_rate if corrupt_rate is None else corrupt_rate
        positions = self._positions(size)
        target = self.tokens[positions]
        is_corrupted = torch.rand(size, device=self.device) < corrupt_rate
        corrupted = self.corrupt(target)
        is_corrupted &= corrupted != target
        typed = torch.where(is_corrupted, corrupted, target)
        candidates = torch.cat([typed.unsqueeze(1), self.neighbors[typed]], dim=1)
        kinds = torch.cat([torch.zeros_like(typed).unsqueeze(1), self.neighbor_kinds[typed]], dim=1)
        matches = candidates == target.unsqueeze(1)
        valid = self.valid[target] & self.is_target[typed] & (self.neighbor_counts[typed] > 0) & matches.any(1)

        positions, candidates, kinds, matches, is_corrupted = positions[valid], candidates[valid], kinds[valid], matches[valid], is_corrupted[valid]
        count = len(positions)
        left_offsets = torch.arange(-LEFT_WINDOW, 0, device=self.device)
        left = self.tokens[positions.unsqueeze(1) + left_offsets]
        right = self.tokens[positions.unsqueeze(1) + torch.arange(1, RIGHT_WINDOW + 1, device=self.device)]
        if context_noise:
            left, right = self._noisy_context(left), self._noisy_context(right)

        visible_right = torch.randint(0, RIGHT_WINDOW + 1, (count, 1), device=self.device)
        right = right.masked_fill(torch.arange(RIGHT_WINDOW, device=self.device).unsqueeze(0) >= visible_right, PAD).flip(1)

        is_short = torch.rand(count, device=self.device) < self.options.short_share
        kept = torch.randint(0, LEFT_WINDOW, (count, 1), device=self.device)
        distance_from_end = (-left_offsets).unsqueeze(0)
        left = left.masked_fill(is_short.unsqueeze(1) & (distance_from_end > kept + 1), PAD)
        left = left.masked_fill(is_short.unsqueeze(1) & (distance_from_end == kept + 1), START)
        return Batch(left, right, candidates, kinds, matches.long().argmax(1), is_corrupted)
