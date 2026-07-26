from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset


def _token_stream(
    tokenizer,
    *,
    split: str,
    target_tokens: int,
    cache_dir: str,
) -> torch.Tensor:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        cache_dir=cache_dir,
    )
    tokens: list[int] = []
    eos = tokenizer.eos_token_id

    for row in dataset:
        text = row["text"]
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        tokens.extend(encoded)
        if eos is not None:
            tokens.append(eos)
        if len(tokens) >= target_tokens:
            break

    if len(tokens) < target_tokens:
        raise RuntimeError(
            f"{split} yielded {len(tokens):,} tokens, fewer than {target_tokens:,}"
        )
    return torch.tensor(tokens[:target_tokens], dtype=torch.long)


def _blocks_for_predictions(
    tokens: torch.Tensor,
    *,
    prediction_tokens: int,
    sequence_length: int,
) -> torch.Tensor:
    predictions_per_block = sequence_length - 1
    n_blocks = (prediction_tokens + predictions_per_block - 1) // predictions_per_block
    required = n_blocks * sequence_length
    if tokens.numel() < required:
        raise ValueError(f"need {required:,} raw tokens, received {tokens.numel():,}")
    return tokens[:required].reshape(n_blocks, sequence_length)


def prepare_corpus(
    tokenizer,
    *,
    output_dir: Path,
    cache_dir: str,
    calibration_tokens: int,
    evaluation_tokens: int,
    calibration_seeds: Iterable[int],
    sequence_length: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(calibration_seeds)
    paths = {
        "evaluation": output_dir / "evaluation_blocks.pt",
        **{f"calibration_{seed}": output_dir / f"calibration_seed_{seed}_blocks.pt" for seed in seeds},
    }
    if all(path.exists() for path in paths.values()):
        return paths

    predictions_per_block = sequence_length - 1
    calibration_blocks = (calibration_tokens + predictions_per_block - 1) // predictions_per_block
    raw_per_calibration = calibration_blocks * sequence_length
    train_needed = raw_per_calibration * len(seeds)

    evaluation_blocks = (evaluation_tokens + predictions_per_block - 1) // predictions_per_block
    validation_needed = evaluation_blocks * sequence_length

    train = _token_stream(
        tokenizer,
        split="train",
        target_tokens=train_needed,
        cache_dir=cache_dir,
    )
    validation = _token_stream(
        tokenizer,
        split="validation",
        target_tokens=validation_needed,
        cache_dir=cache_dir,
    )

    for index, seed in enumerate(seeds):
        raw = train[index * raw_per_calibration : (index + 1) * raw_per_calibration]
        blocks = _blocks_for_predictions(
            raw,
            prediction_tokens=calibration_tokens,
            sequence_length=sequence_length,
        )
        torch.save(blocks, paths[f"calibration_{seed}"])

    evaluation = _blocks_for_predictions(
        validation,
        prediction_tokens=evaluation_tokens,
        sequence_length=sequence_length,
    )
    torch.save(evaluation, paths["evaluation"])

    manifest = {
        "dataset": "Salesforce/wikitext",
        "subset": "wikitext-103-raw-v1",
        "calibration_split": "train",
        "evaluation_split": "validation",
        "calibration_tokens": calibration_tokens,
        "evaluation_tokens": evaluation_tokens,
        "sequence_length": sequence_length,
        "calibration_seeds": list(seeds),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def iter_batches(blocks: torch.Tensor, batch_size: int):
    for start in range(0, blocks.shape[0], batch_size):
        yield blocks[start : start + batch_size]
