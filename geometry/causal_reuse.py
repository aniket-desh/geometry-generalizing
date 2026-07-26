from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from geogen.model import GeometryTransformer, ModelConfig
from geogen.tasks import make_task
from operator_reuse import orthogonal_fit
from train import TokenLayout, build_tokens


POSITIONS = {"node": 2, "output": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a cyclic activation transport on held-in states and aliases, "
            "then causally test it on unseen operation-table entries."
        )
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--checkpoint-glob",
        help=(
            "Checkpoint glob. By default dense weights are preferred, with "
            "ordinary checkpoints used as a fallback."
        ),
    )
    parser.add_argument(
        "--steps", help="Optional comma-separated checkpoint steps."
    )
    parser.add_argument(
        "--positions",
        default="node,output",
        help="Comma-separated patch positions: node and/or output.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated residual-stream layers, or all.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-seed", type=int, default=2027)
    parser.add_argument("--state-train-fraction", type=float, default=0.5)
    parser.add_argument("--alias-train-fraction", type=float, default=0.5)
    parser.add_argument("--max-dimension", type=int, default=16)
    parser.add_argument("--fit-contexts", type=int, default=4)
    parser.add_argument("--fit-right-aliases", type=int, default=2)
    parser.add_argument("--eval-contexts", type=int, default=4)
    parser.add_argument("--eval-right-aliases", type=int, default=2)
    parser.add_argument("--max-fit-examples", type=int, default=16_384)
    parser.add_argument("--max-eval-examples", type=int, default=8_192)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _parse_ints(value: str | None) -> set[int] | None:
    if value is None:
        return None
    parsed = {int(item) for item in value.split(",") if item.strip()}
    if not parsed or min(parsed) < 0:
        raise ValueError("integer lists must contain non-negative values")
    return parsed


def _split_indices(
    size: int, fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("a held-out split requires at least two items")
    count = min(max(round(size * fraction), 1), size - 1)
    permutation = rng.permutation(size)
    return np.sort(permutation[:count]), np.sort(permutation[count:])


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def discover_checkpoints(
    run_dir: Path, pattern: str | None, steps: set[int] | None
) -> list[Path]:
    if pattern:
        paths = sorted(run_dir.glob(pattern))
    else:
        dense = sorted(run_dir.glob("weights-*.pt"))
        paths = dense if dense else sorted(run_dir.glob("checkpoint-*.pt"))
    if steps is not None:
        paths = [path for path in paths if _checkpoint_step(path) in steps]
    if not paths:
        raise FileNotFoundError("no matching checkpoints found")
    return paths


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a checkpoint dictionary")
    return checkpoint


def make_patch_specs(
    *,
    depth: int,
    positions: str,
    layers: str,
) -> list[tuple[str, int]]:
    requested_positions = [
        item.strip() for item in positions.split(",") if item.strip()
    ]
    invalid = set(requested_positions) - set(POSITIONS)
    if invalid or not requested_positions:
        raise ValueError(f"invalid patch positions: {sorted(invalid)}")
    requested_layers = (
        set(range(depth + 1)) if layers == "all" else _parse_ints(layers)
    )
    assert requested_layers is not None
    invalid_layers = requested_layers - set(range(depth + 1))
    if invalid_layers:
        raise ValueError(f"layers outside the model: {sorted(invalid_layers)}")

    specs: list[tuple[str, int]] = []
    for position in requested_positions:
        for layer in sorted(requested_layers):
            # A node patch after the last block cannot reach the output token.
            if position == "node" and layer == depth:
                continue
            # Before the first block, the equals token is state-independent.
            if position == "output" and layer == 0:
                continue
            specs.append((position, layer))
    if not specs:
        raise ValueError("the requested position/layer combinations have no effect")
    return specs


def _deranged_control(
    successor: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    order = successor.size
    for _ in range(10_000):
        candidate = rng.permutation(order)
        if np.all(candidate != successor) and np.all(
            candidate != np.arange(order)
        ):
            return candidate
    raise RuntimeError("could not sample a scrambled-successor control")


def _subsample_examples(
    examples: dict[str, np.ndarray],
    limit: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    count = len(examples["left"])
    if limit > 0 and count > limit:
        chosen = np.sort(rng.choice(count, size=limit, replace=False))
        return {key: values[chosen] for key, values in examples.items()}
    return examples


def _expand_pairs(
    pairs: np.ndarray,
    *,
    left_aliases: np.ndarray,
    contexts: np.ndarray,
    right_aliases: np.ndarray,
) -> dict[str, np.ndarray]:
    pair_id, left_alias, context, right_alias = np.meshgrid(
        np.arange(len(pairs)),
        left_aliases,
        contexts,
        right_aliases,
        indexing="ij",
    )
    pair_id = pair_id.ravel()
    return {
        "left": pairs[pair_id, 0],
        "right": pairs[pair_id, 1],
        "left_alias": left_alias.ravel(),
        "context": context.ravel(),
        "right_alias": right_alias.ravel(),
    }


def make_fold_examples(
    *,
    order: int,
    aliases: int,
    contexts: int,
    train_mask: np.ndarray,
    successor: np.ndarray,
    state_train_fraction: float,
    alias_train_fraction: float,
    fit_contexts: int,
    fit_right_aliases: int,
    eval_contexts: int,
    eval_right_aliases: int,
    max_fit_examples: int,
    max_eval_examples: int,
    rng: np.random.Generator,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    state_train, state_test = _split_indices(
        order, state_train_fraction, rng
    )
    alias_train, alias_test = _split_indices(
        aliases, alias_train_fraction, rng
    )

    row, relation = np.indices((order, order))
    shifted_row = successor[row]
    fit_pair_mask = (
        np.isin(row, state_train)
        & train_mask
        & train_mask[shifted_row, relation]
    )
    eval_pair_mask = (
        np.isin(row, state_test)
        & ~train_mask
        & ~train_mask[shifted_row, relation]
    )
    fit_pairs = np.column_stack(np.nonzero(fit_pair_mask))
    eval_pairs = np.column_stack(np.nonzero(eval_pair_mask))
    if not len(fit_pairs):
        raise ValueError("the fold contains no jointly trained transition pairs")
    if not len(eval_pairs):
        raise ValueError("the fold contains no jointly held-out transition pairs")

    fit_context_ids = rng.choice(
        contexts, size=min(max(fit_contexts, 1), contexts), replace=False
    )
    fit_right_alias_ids = rng.choice(
        aliases,
        size=min(max(fit_right_aliases, 1), aliases),
        replace=False,
    )
    eval_context_ids = rng.choice(
        contexts, size=min(max(eval_contexts, 1), contexts), replace=False
    )
    eval_right_alias_ids = rng.choice(
        aliases,
        size=min(max(eval_right_aliases, 1), aliases),
        replace=False,
    )
    fit = _expand_pairs(
        fit_pairs,
        left_aliases=alias_train,
        contexts=fit_context_ids,
        right_aliases=fit_right_alias_ids,
    )
    evaluation = _expand_pairs(
        eval_pairs,
        left_aliases=alias_test,
        contexts=eval_context_ids,
        right_aliases=eval_right_alias_ids,
    )
    fit = _subsample_examples(fit, max_fit_examples, rng)
    evaluation = _subsample_examples(evaluation, max_eval_examples, rng)
    return fit, evaluation, state_train, alias_train


def _replace_left(
    examples: dict[str, np.ndarray], mapping: np.ndarray
) -> dict[str, np.ndarray]:
    result = {key: values.copy() for key, values in examples.items()}
    result["left"] = mapping[result["left"]]
    return result


@torch.no_grad()
def extract_activations(
    model: GeometryTransformer,
    *,
    examples: dict[str, np.ndarray],
    layout: TokenLayout,
    specs: list[tuple[str, int]],
    batch_size: int,
    device: torch.device,
    return_logits: bool,
) -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray | None]:
    model.eval()
    collected = {spec: [] for spec in specs}
    logits_collected: list[np.ndarray] = []
    count = len(examples["left"])
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        tensors = {
            key: torch.as_tensor(values[start:stop], device=device)
            for key, values in examples.items()
        }
        tokens = build_tokens(
            layout,
            tensors["left"],
            tensors["right"],
            contexts=tensors["context"],
            left_alias=tensors["left_alias"],
            right_alias=tensors["right_alias"],
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, states = model(tokens, return_states=True)
        for spec in specs:
            position, layer = spec
            values = states[layer][:, POSITIONS[position]].float()
            collected[spec].append(values.cpu().numpy())
        if return_logits:
            logits_collected.append(logits.float().cpu().numpy())
    activations = {
        spec: np.concatenate(parts, axis=0)
        for spec, parts in collected.items()
    }
    logits_array = (
        np.concatenate(logits_collected, axis=0) if return_logits else None
    )
    return activations, logits_array


@torch.no_grad()
def patched_logits(
    model: GeometryTransformer,
    *,
    examples: dict[str, np.ndarray],
    patches: np.ndarray,
    layout: TokenLayout,
    position: str,
    layer: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    collected: list[np.ndarray] = []
    count = len(examples["left"])
    depth = len(model.blocks)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        tensors = {
            key: torch.as_tensor(values[start:stop], device=device)
            for key, values in examples.items()
        }
        tokens = build_tokens(
            layout,
            tensors["left"],
            tensors["right"],
            contexts=tensors["context"],
            left_alias=tensors["left_alias"],
            right_alias=tensors["right_alias"],
        )
        patch = torch.as_tensor(patches[start:stop], device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            x = (
                model.token_embedding(tokens)
                + model.position_embedding[None, : tokens.shape[1], :]
            )
            if layer == 0:
                x = x.clone()
                x[:, POSITIONS[position]] = patch.to(x.dtype)
            mask = torch.triu(
                torch.full(
                    (tokens.shape[1], tokens.shape[1]),
                    float("-inf"),
                    device=device,
                    dtype=x.dtype,
                ),
                diagonal=1,
            )
            for block_index, block in enumerate(model.blocks):
                x = block(x, src_mask=mask, is_causal=True)
                residual_layer = block_index + 1
                if residual_layer == layer and residual_layer < depth:
                    x = x.clone()
                    x[:, POSITIONS[position]] = patch.to(x.dtype)
            x = model.final_norm(x)
            if layer == depth:
                x = x.clone()
                x[:, POSITIONS[position]] = patch.to(x.dtype)
            logits = model.output(x[:, -1])
        collected.append(logits.float().cpu().numpy())
    return np.concatenate(collected, axis=0)


@dataclass(frozen=True)
class Transport:
    center: np.ndarray
    basis: np.ndarray
    operator: np.ndarray
    dimension: int

    def patch(self, activations: np.ndarray) -> np.ndarray:
        coordinates = (activations - self.center) @ self.basis
        delta = (coordinates @ self.operator - coordinates) @ self.basis.T
        return activations + delta


def fit_transport(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_dimension: int,
) -> Transport:
    source64 = np.asarray(source, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    center = np.concatenate((source64, target64), axis=0).mean(
        axis=0, keepdims=True
    )
    pooled = np.concatenate((source64, target64), axis=0) - center
    _, singular, vt = np.linalg.svd(pooled, full_matrices=False)
    if not singular.size or singular[0] <= 1e-12:
        raise ValueError("activation transport has zero rank")
    rank = int(np.sum(singular > singular[0] * 1e-6))
    dimension = min(max_dimension, rank, source.shape[-1])
    basis = vt[:dimension].T
    x = (source64 - center) @ basis
    y = (target64 - center) @ basis
    operator = orthogonal_fit(x, y)
    return Transport(
        center=center,
        basis=basis,
        operator=operator,
        dimension=dimension,
    )


def fit_control_operator(
    transport: Transport,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    x = (np.asarray(source, dtype=np.float64) - transport.center) @ transport.basis
    y = (np.asarray(target, dtype=np.float64) - transport.center) @ transport.basis
    return orthogonal_fit(x, y)


def apply_operator(
    transport: Transport,
    activations: np.ndarray,
    operator: np.ndarray,
) -> np.ndarray:
    values = np.asarray(activations, dtype=np.float64)
    coordinates = (values - transport.center) @ transport.basis
    delta = (coordinates @ operator - coordinates) @ transport.basis.T
    return values + delta


def random_norm_matched_patch(
    transport: Transport,
    activations: np.ndarray,
    learned_patch: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(activations, dtype=np.float64)
    coordinates = (values - transport.center) @ transport.basis
    random_matrix, _ = np.linalg.qr(
        rng.normal(size=(transport.dimension, transport.dimension))
    )
    random_delta = coordinates @ random_matrix - coordinates
    learned_delta = (learned_patch - values) @ transport.basis
    random_norm = np.linalg.norm(random_delta, axis=1, keepdims=True)
    learned_norm = np.linalg.norm(learned_delta, axis=1, keepdims=True)
    random_delta *= learned_norm / np.maximum(random_norm, 1e-12)
    return values + random_delta @ transport.basis.T


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def behavioral_metrics(
    logits: np.ndarray,
    *,
    original_target: np.ndarray,
    desired_target: np.ndarray,
    source_logits: np.ndarray,
    natural_logits: np.ndarray,
) -> dict[str, float | int]:
    probabilities = _softmax(logits)
    source_probabilities = _softmax(source_logits)
    natural_probabilities = _softmax(natural_logits)
    indices = np.arange(len(logits))
    desired_probability = probabilities[indices, desired_target]
    source_desired_probability = source_probabilities[indices, desired_target]
    natural_desired_probability = natural_probabilities[indices, desired_target]
    desired_logits = logits[indices, desired_target]
    source_desired_logits = source_logits[indices, desired_target]
    natural_desired_logits = natural_logits[indices, desired_target]
    best_other = logits.copy()
    best_other[indices, desired_target] = -np.inf
    natural_gain = float(
        np.mean(natural_desired_probability - source_desired_probability)
    )
    intervention_gain = float(
        np.mean(desired_probability - source_desired_probability)
    )
    natural_logit_gain = float(
        np.mean(natural_desired_logits - source_desired_logits)
    )
    intervention_logit_gain = float(
        np.mean(desired_logits - source_desired_logits)
    )
    predictions = np.argmax(logits, axis=1)
    source_predictions = np.argmax(source_logits, axis=1)
    natural_predictions = np.argmax(natural_logits, axis=1)
    qualified = (source_predictions == original_target) & (
        natural_predictions == desired_target
    )
    return {
        "desired_accuracy": float(
            np.mean(predictions == desired_target)
        ),
        "original_accuracy": float(
            np.mean(predictions == original_target)
        ),
        "baseline_original_accuracy": float(
            np.mean(source_predictions == original_target)
        ),
        "natural_shift_accuracy": float(
            np.mean(natural_predictions == desired_target)
        ),
        "qualified_examples": int(np.sum(qualified)),
        "qualified_desired_accuracy": (
            float(np.mean(predictions[qualified] == desired_target[qualified]))
            if np.any(qualified)
            else float("nan")
        ),
        "desired_probability": float(np.mean(desired_probability)),
        "desired_logit_margin": float(
            np.mean(desired_logits - best_other.max(axis=1))
        ),
        "desired_probability_gain": intervention_gain,
        "desired_logit_gain": intervention_logit_gain,
        "probability_recovery": (
            intervention_gain / natural_gain
            if abs(natural_gain) > 1e-9
            else float("nan")
        ),
        "logit_recovery": (
            intervention_logit_gain / natural_logit_gain
            if abs(natural_logit_gain) > 1e-9
            else float("nan")
        ),
    }


def target_centroid_patch(
    target: np.ndarray, target_states: np.ndarray
) -> np.ndarray:
    values = np.asarray(target, dtype=np.float64)
    labels = np.asarray(target_states, dtype=np.int64)
    centroids = np.zeros((int(labels.max()) + 1, values.shape[1]))
    counts = np.bincount(labels, minlength=len(centroids))
    np.add.at(centroids, labels, values)
    occupied = counts > 0
    centroids[occupied] /= counts[occupied, None]
    return centroids[labels]


def manifold_metrics(
    patch: np.ndarray,
    *,
    source: np.ndarray,
    target: np.ndarray,
    target_states: np.ndarray,
) -> dict[str, float]:
    patch64 = np.asarray(patch, dtype=np.float64)
    source64 = np.asarray(source, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    paired = np.linalg.norm(patch64 - target64, axis=1)
    source_to_target = np.linalg.norm(source64 - target64, axis=1)

    target_centroids = target_centroid_patch(target64, target_states)
    patch_centroid_squared = np.sum((patch64 - target_centroids) ** 2, axis=1)
    natural_centroid_squared = np.sum(
        (target64 - target_centroids) ** 2, axis=1
    )
    patch_rms = math.sqrt(float(np.mean(patch_centroid_squared)))
    natural_rms = math.sqrt(float(np.mean(natural_centroid_squared)))
    return {
        "patch_l2": float(np.mean(np.linalg.norm(patch64 - source64, axis=1))),
        "paired_target_distance": float(np.mean(paired)),
        "paired_target_distance_ratio": float(
            np.mean(paired / np.maximum(source_to_target, 1e-12))
        ),
        "target_centroid_distance_rms": patch_rms,
        "natural_target_radius_rms": natural_rms,
        "off_manifold_ratio": (
            patch_rms / natural_rms if natural_rms > 1e-12 else float("nan")
        ),
    }


def write_outputs(
    *,
    prefix: Path,
    metadata: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    jsonl_path = prefix.with_suffix(".jsonl")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(_json_safe({"metadata": metadata, "records": records}), indent=2)
        + "\n"
    )
    jsonl_path.write_text(
        "".join(json.dumps(_json_safe(record)) + "\n" for record in records)
    )
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: _json_safe(value)
                for key, value in record.items()
                if not isinstance(value, (dict, list))
            }
            for record in records
        )
    print(f"wrote {json_path}, {jsonl_path}, and {csv_path}")


def _behavior_at_steps(run_dir: Path) -> dict[int, dict[str, object]]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    result: dict[int, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result[int(record["step"])] = {
            key: record.get(key)
            for key in (
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
            )
        }
    return result


def evaluate_checkpoint(
    *,
    model: GeometryTransformer,
    checkpoint_path: Path,
    step: int,
    layout: TokenLayout,
    specs: list[tuple[str, int]],
    table: np.ndarray,
    train_mask: np.ndarray,
    successor: np.ndarray,
    folds: int,
    fold_seed: int,
    state_train_fraction: float,
    alias_train_fraction: float,
    max_dimension: int,
    fit_contexts: int,
    fit_right_aliases: int,
    eval_contexts: int,
    eval_right_aliases: int,
    max_fit_examples: int,
    max_eval_examples: int,
    batch_size: int,
    device: torch.device,
    behavior: dict[str, object],
) -> list[dict[str, object]]:
    order = table.shape[0]
    records: list[dict[str, object]] = []
    for fold in range(folds):
        rng = np.random.default_rng(fold_seed + fold)
        fit, evaluation, state_train, alias_train = make_fold_examples(
            order=order,
            aliases=layout.aliases,
            contexts=layout.contexts,
            train_mask=train_mask,
            successor=successor,
            state_train_fraction=state_train_fraction,
            alias_train_fraction=alias_train_fraction,
            fit_contexts=fit_contexts,
            fit_right_aliases=fit_right_aliases,
            eval_contexts=eval_contexts,
            eval_right_aliases=eval_right_aliases,
            max_fit_examples=max_fit_examples,
            max_eval_examples=max_eval_examples,
            rng=rng,
        )
        scrambled = _deranged_control(successor, rng)
        fit_target = _replace_left(fit, successor)
        fit_scrambled = _replace_left(fit, scrambled)
        eval_target = _replace_left(evaluation, successor)

        fit_source_acts, _ = extract_activations(
            model,
            examples=fit,
            layout=layout,
            specs=specs,
            batch_size=batch_size,
            device=device,
            return_logits=False,
        )
        fit_target_acts, _ = extract_activations(
            model,
            examples=fit_target,
            layout=layout,
            specs=specs,
            batch_size=batch_size,
            device=device,
            return_logits=False,
        )
        fit_scrambled_acts, _ = extract_activations(
            model,
            examples=fit_scrambled,
            layout=layout,
            specs=specs,
            batch_size=batch_size,
            device=device,
            return_logits=False,
        )
        eval_source_acts, source_logits = extract_activations(
            model,
            examples=evaluation,
            layout=layout,
            specs=specs,
            batch_size=batch_size,
            device=device,
            return_logits=True,
        )
        eval_target_acts, natural_logits = extract_activations(
            model,
            examples=eval_target,
            layout=layout,
            specs=specs,
            batch_size=batch_size,
            device=device,
            return_logits=True,
        )
        assert source_logits is not None and natural_logits is not None
        original_target = table[evaluation["left"], evaluation["right"]]
        desired_target = table[eval_target["left"], evaluation["right"]]

        for position, layer in specs:
            spec = (position, layer)
            source_activations = eval_source_acts[spec]
            target_activations = eval_target_acts[spec]
            target_states = (
                eval_target["left"] if position == "node" else desired_target
            )
            transport = fit_transport(
                fit_source_acts[spec],
                fit_target_acts[spec],
                max_dimension=max_dimension,
            )
            learned_patch = transport.patch(source_activations)
            scrambled_operator = fit_control_operator(
                transport,
                fit_source_acts[spec],
                fit_scrambled_acts[spec],
            )
            scrambled_patch = apply_operator(
                transport, source_activations, scrambled_operator
            )
            random_patch = random_norm_matched_patch(
                transport, source_activations, learned_patch, rng
            )
            centroid_patch = target_centroid_patch(
                target_activations, target_states
            )
            controls: dict[str, tuple[np.ndarray, np.ndarray]] = {
                "source": (source_activations, source_logits),
                "natural_shift": (target_activations, natural_logits),
            }
            for name, patch in (
                ("learned_generator", learned_patch),
                ("exact_state_swap", target_activations),
                ("target_centroid", centroid_patch),
                ("scrambled_successor", scrambled_patch),
                ("random_orthogonal", random_patch),
            ):
                controls[name] = (
                    patch,
                    patched_logits(
                        model,
                        examples=evaluation,
                        patches=patch,
                        layout=layout,
                        position=position,
                        layer=layer,
                        batch_size=batch_size,
                        device=device,
                    ),
                )

            for control, (patch, logits) in controls.items():
                record: dict[str, object] = {
                    "checkpoint": checkpoint_path.name,
                    "step": step,
                    "fold": fold,
                    "position": position,
                    "layer": layer,
                    "control": control,
                    "dimension": transport.dimension,
                    "fit_examples": len(fit["left"]),
                    "eval_examples": len(evaluation["left"]),
                    "fit_states": len(state_train),
                    "fit_aliases": len(alias_train),
                    "canonical_shift_fraction": float(
                        np.mean(desired_target == successor[original_target])
                    ),
                    **behavior,
                    **behavioral_metrics(
                        logits,
                        original_target=original_target,
                        desired_target=desired_target,
                        source_logits=source_logits,
                        natural_logits=natural_logits,
                    ),
                    **manifold_metrics(
                        patch,
                        source=source_activations,
                        target=target_activations,
                        target_states=target_states,
                    ),
                }
                records.append(record)
        print(
            f"{checkpoint_path.name} fold={fold}: "
            f"{len(evaluation['left'])} jointly held-out examples, "
            f"{len(specs)} patch sites",
            flush=True,
        )
    return records


def run_self_test(device: torch.device) -> None:
    rng = np.random.default_rng(11)
    order = 17
    angles = 2.0 * np.pi * np.arange(order) / order
    circle = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    embedding, _ = np.linalg.qr(rng.normal(size=(12, 2)))
    source = circle @ embedding.T
    target = np.roll(source, -1, axis=0)
    transport = fit_transport(source, target, max_dimension=2)
    relative_error = np.linalg.norm(transport.patch(source) - target) / np.linalg.norm(
        target
    )
    if relative_error > 1e-8:
        raise AssertionError(f"synthetic transport failed: {relative_error}")

    layout = TokenLayout(
        7, 2, 2, seed=3, device=device
    )
    model = GeometryTransformer(
        vocab_size=layout.vocab_size,
        output_classes=7,
        config=ModelConfig(width=16, depth=1, heads=4),
    ).to(device)
    examples = {
        "left": np.arange(7),
        "right": np.arange(7),
        "left_alias": np.zeros(7, dtype=np.int64),
        "right_alias": np.ones(7, dtype=np.int64),
        "context": np.zeros(7, dtype=np.int64),
    }
    shifted = _replace_left(examples, (np.arange(7) + 1) % 7)
    specs = [("node", 0), ("output", 1)]
    _, source_logits = extract_activations(
        model,
        examples=examples,
        layout=layout,
        specs=specs,
        batch_size=7,
        device=device,
        return_logits=True,
    )
    shifted_acts, shifted_logits = extract_activations(
        model,
        examples=shifted,
        layout=layout,
        specs=specs,
        batch_size=7,
        device=device,
        return_logits=True,
    )
    assert source_logits is not None and shifted_logits is not None
    for position, layer in specs:
        exact_logits = patched_logits(
            model,
            examples=examples,
            patches=shifted_acts[(position, layer)],
            layout=layout,
            position=position,
            layer=layer,
            batch_size=7,
            device=device,
        )
        if not np.allclose(exact_logits, shifted_logits, atol=1e-5):
            raise AssertionError(
                f"{position} layer {layer} exact swap did not reproduce logits"
            )
    print(
        "self-test passed: synthetic generator recovered and input-node plus "
        "final-output swaps exactly reproduced the shifted computation"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.self_test:
        run_self_test(device)
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --self-test is used")
    if not (0.0 < args.state_train_fraction < 1.0):
        raise ValueError("--state-train-fraction must be between zero and one")
    if not (0.0 < args.alias_train_fraction < 1.0):
        raise ValueError("--alias-train-fraction must be between zero and one")
    if min(
        args.folds,
        args.max_dimension,
        args.batch_size,
        args.fit_contexts,
        args.fit_right_aliases,
        args.eval_contexts,
        args.eval_right_aliases,
    ) < 1:
        raise ValueError("fold, dimension, batch, and sample counts must be positive")

    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text())
    table = np.load(run_dir / "operation_table.npy")
    train_mask = np.load(run_dir / "train_mask.npy").astype(bool)
    if config.get("task_family") not in {"cycle", "broken_cycle"}:
        raise ValueError("causal successor interventions require a cyclic task")
    order = int(config["task_order"])
    if table.shape != (order, order) or train_mask.shape != table.shape:
        raise ValueError("the saved operation table or split has the wrong shape")
    task = make_task(
        config["task"],
        seed=int(
            config.get("task_seed", config.get("split_seed", config["seed"]))
        ),
        corruption=float(
            config.get(
                "task_corruption_fraction",
                config.get("corruption", 0.0),
            )
        ),
    )
    if task.generator is None:
        raise ValueError(f"{config['task']} has no designated successor")
    successor = np.asarray(table[:, task.generator], dtype=np.int64)

    layout = TokenLayout(
        order,
        int(config["aliases"]),
        int(config["contexts"]),
        seed=int(config.get("token_seed", config["seed"])),
        device=device,
    )
    saved_layout = np.load(run_dir / "token_layout.npz")
    if not np.array_equal(layout.node_tokens.cpu().numpy(), saved_layout["node"]):
        raise ValueError("reconstructed node-token layout does not match the run")
    if not np.array_equal(
        layout.relation_tokens.cpu().numpy(), saved_layout["relation"]
    ):
        raise ValueError("reconstructed relation-token layout does not match the run")

    model_config = ModelConfig(**config["model"])
    model = GeometryTransformer(
        vocab_size=layout.vocab_size,
        output_classes=order,
        config=model_config,
    ).to(device)
    specs = make_patch_specs(
        depth=model_config.depth,
        positions=args.positions,
        layers=args.layers,
    )
    requested_steps = _parse_ints(args.steps)
    checkpoints = discover_checkpoints(
        run_dir, args.checkpoint_glob, requested_steps
    )
    behavior = _behavior_at_steps(run_dir)
    records: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        step = int(checkpoint.get("step", _checkpoint_step(checkpoint_path)))
        del checkpoint
        records.extend(
            evaluate_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                step=step,
                layout=layout,
                specs=specs,
                table=table,
                train_mask=train_mask,
                successor=successor,
                folds=args.folds,
                fold_seed=args.fold_seed,
                state_train_fraction=args.state_train_fraction,
                alias_train_fraction=args.alias_train_fraction,
                max_dimension=args.max_dimension,
                fit_contexts=args.fit_contexts,
                fit_right_aliases=args.fit_right_aliases,
                eval_contexts=args.eval_contexts,
                eval_right_aliases=args.eval_right_aliases,
                max_fit_examples=args.max_fit_examples,
                max_eval_examples=args.max_eval_examples,
                batch_size=args.batch_size,
                device=device,
                behavior=behavior.get(step, {}),
            )
        )

    prefix = args.output_prefix or (run_dir / "causal_reuse")
    metadata = {
        "run_name": config["run_name"],
        "task": config["task"],
        "order": order,
        "generator_relation": task.generator,
        "checkpoints": [path.name for path in checkpoints],
        "patch_sites": [
            {"position": position, "layer": layer}
            for position, layer in specs
        ],
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "state_train_fraction": args.state_train_fraction,
        "alias_train_fraction": args.alias_train_fraction,
        "max_dimension": args.max_dimension,
        "fit_pair_rule": (
            "the source and canonically shifted operation-table entries were "
            "both included in model training"
        ),
        "evaluation_pair_rule": (
            "the source and canonically shifted operation-table entries were "
            "both excluded from model training"
        ),
        "transport": (
            "orthogonal Procrustes in a PCA subspace fitted only on held-in "
            "states and aliases; the orthogonal delta is inserted into the "
            "original full residual stream"
        ),
        "off_manifold_ratio": (
            "RMS distance to the natural target-state centroid divided by the "
            "natural within-target-state RMS radius"
        ),
        "controls": {
            "exact_state_swap": (
                "the activation from the naturally shifted input under matched "
                "context and aliases"
            ),
            "target_centroid": (
                "the desired state's centroid from natural counterfactual "
                "activations in the held-out evaluation fold; this is a "
                "label-informed upper bound"
            ),
            "scrambled_successor": (
                "an orthogonal transport fitted to a frequency-matched random "
                "derangement of state successors"
            ),
            "random_orthogonal": (
                "a random orthogonal subspace displacement matched per example "
                "to the learned intervention norm"
            ),
        },
    }
    write_outputs(prefix=prefix, metadata=metadata, records=records)


if __name__ == "__main__":
    main()
