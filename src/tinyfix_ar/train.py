"""Trains the wrong-word model: python -m tinyfix_ar.train --corpus artifacts/corpus --lexicon artifacts/lexicon --output runs/v1"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinyfix_ar.corpus import Corpus
from tinyfix_ar.lexicon import Lexicon
from tinyfix_ar.model import WrongWordModel, mask_empty
from tinyfix_ar.sampler import Sampler, SamplerOptions
from tinyfix_ar.text import UNK

VALIDATION_SHARE = 0.02
FALSE_RATES = (0.005, 0.001, 0.0005)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def split_tokens(corpus: Corpus, lexicon: Lexicon) -> tuple[torch.Tensor, torch.Tensor]:
    lookup = lexicon.lookup_table(corpus.raw_tokens)

    def flatten(indices: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(lookup[np.concatenate([corpus.documents[index] for index in indices])].astype(np.int64))

    train, held_out = corpus.split(VALIDATION_SHARE)
    return flatten(train), flatten(held_out)


@torch.no_grad()
def evaluate_synthetic(model: WrongWordModel, sampler: Sampler, batches: int, batch_size: int) -> dict:
    """Share of corruptions caught and fixed, at the thresholds that change 0.5%, 0.1% and 0.05% of clean words."""
    model.eval()
    clean_margins, corrupted_margins, corrupted_right = [], [], []
    for _ in range(batches):
        batch = sampler.batch(batch_size, corrupt_rate=0.5, context_noise=False)
        scores = mask_empty(model(batch.left, batch.right, batch.candidates, batch.kinds), batch.candidates)
        best = scores[:, 1:].max(1)
        margin = best.values - scores[:, 0]
        clean_margins.append(margin[~batch.is_corrupted])
        corrupted_margins.append(margin[batch.is_corrupted])
        corrupted_right.append(best.indices[batch.is_corrupted] + 1 == batch.label[batch.is_corrupted])
    model.train()
    clean, margin, right = torch.cat(clean_margins).float().cpu(), torch.cat(corrupted_margins).float().cpu(), torch.cat(corrupted_right).cpu()
    results = {"clean_cases": len(clean), "corrupted_cases": len(margin), "top1_on_corrupted": round(right.float().mean().item(), 4)}
    for false_rate in FALSE_RATES:
        threshold = torch.quantile(clean[:500_000], 1 - false_rate).item()
        results[f"caught_at_false_{false_rate}"] = round(((margin >= threshold) & right).float().mean().item(), 4)
        results[f"threshold_at_false_{false_rate}"] = round(threshold, 3)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--corrupt-rate", type=float, default=0.5)
    parser.add_argument("--real-share", type=float, default=0.6)
    parser.add_argument("--group-share", type=float, default=0.25)
    parser.add_argument("--short-share", type=float, default=0.2)
    parser.add_argument("--context-noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"device={device} args={vars(args)}", flush=True)

    lexicon = Lexicon.load(args.lexicon)
    train_tokens, validation_tokens = split_tokens(Corpus.load(args.corpus), lexicon)
    print(f"train={len(train_tokens):,} validation={len(validation_tokens):,} unknown={(train_tokens == UNK).float().mean().item():.4f}", flush=True)

    options = SamplerOptions(
        corrupt_rate=args.corrupt_rate, real_share=args.real_share, group_share=args.group_share, short_share=args.short_share, context_noise=args.context_noise
    )
    train_sampler = Sampler(train_tokens, lexicon, options, device)
    validation_sampler = Sampler(validation_tokens, lexicon, SamplerOptions(**{**asdict(options), "group_share": 0.0}), device)

    model = WrongWordModel(len(lexicon.words), args.embedding, args.hidden).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameters:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=args.steps, pct_start=0.1)

    started, running_loss = time.time(), 0.0
    report_every = max(args.steps // 100, 1)
    for step in range(1, args.steps + 1):
        batch = train_sampler.batch(args.batch)
        loss = F.cross_entropy(mask_empty(model(batch.left, batch.right, batch.candidates, batch.kinds), batch.candidates), batch.label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        running_loss += loss.item()
        if step % report_every == 0:
            print(f"step {step}/{args.steps} loss={running_loss / report_every:.4f} elapsed={time.time() - started:.0f}s", flush=True)
            running_loss = 0.0
        if step % max(args.steps // 5, 1) == 0 and step < args.steps:
            print(f"validation step {step}: {evaluate_synthetic(model, validation_sampler, 40, args.batch)}", flush=True)
            torch.save(model.state_dict(), args.output / f"model-step{step}.pt")

    validation = evaluate_synthetic(model, validation_sampler, 150, args.batch)
    print(f"final validation: {validation}", flush=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    config = {"vocabulary_size": len(lexicon.words), "embedding": args.embedding, "hidden": args.hidden, "lexicon": str(args.lexicon)}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    metrics = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "parameters": parameters,
        "validation": validation,
        "train_minutes": round((time.time() - started) / 60, 1),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
