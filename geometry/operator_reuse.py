from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from geogen.model import GeometryTransformer, ModelConfig
from geogen.tasks import make_task
from train import TokenLayout, build_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure when cyclic activations admit one reusable generator "
            "instead of an independently fitted operator for every shift."
        )
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--checkpoint-glob", default="checkpoint-*.pt")
    parser.add_argument(
        "--steps",
        help="Optional comma-separated checkpoint steps to analyze.",
    )
    parser.add_argument("--views", default="node,output")
    parser.add_argument("--max-dimension", type=int, default=16)
    parser.add_argument(
        "--powers",
        help="Comma-separated shifts; defaults to powers of two below the order.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1729)
    parser.add_argument("--state-train-fraction", type=float, default=0.5)
    parser.add_argument("--alias-train-fraction", type=float, default=0.5)
    parser.add_argument("--context-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--precision", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _parse_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parsed = sorted({int(item) for item in value.split(",") if item.strip()})
    if not parsed or min(parsed) < 1:
        raise ValueError("integer lists must contain positive values")
    return parsed


def default_powers(order: int) -> list[int]:
    powers: list[int] = []
    value = 1
    while value < order:
        powers.append(value)
        value *= 2
    return powers


def orthogonal_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return u @ vt


def normalized_error(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[float, float, int]:
    residual = np.asarray(prediction) - np.asarray(target)
    sse = float(np.sum(residual**2))
    denominator = float(np.sum(np.asarray(target) ** 2))
    error = math.sqrt(sse / max(denominator, 1e-30))
    return error, sse, int(residual.size)


def bic_code_bits(
    *,
    sse: float,
    scalar_count: int,
    parameter_count: int,
    precision: float,
) -> float:
    """Two-part Gaussian residual code with a BIC parameter penalty.

    The fixed quantization term makes the differential Gaussian length an
    ordinary positive bit count. It cancels when the two models are compared.
    """

    if scalar_count < 1:
        return float("nan")
    variance_floor = precision**2 / 12.0
    variance = max(sse / scalar_count, variance_floor)
    residual_bits = (
        0.5
        * scalar_count
        * math.log2(2.0 * math.pi * math.e * variance)
        - scalar_count * math.log2(precision)
    )
    parameter_bits = 0.5 * parameter_count * math.log2(max(scalar_count, 2))
    return float(residual_bits + parameter_bits)


def successor_powers(successor: np.ndarray, powers: Iterable[int]) -> dict[int, np.ndarray]:
    requested = sorted(set(powers))
    if not requested:
        return {}
    current = np.arange(successor.size)
    result: dict[int, np.ndarray] = {}
    for exponent in range(1, requested[-1] + 1):
        current = successor[current]
        if exponent in requested:
            result[exponent] = current.copy()
    return result


def project_state_subspace(
    activations: np.ndarray,
    *,
    max_dimension: int,
    sample_limit: int,
) -> tuple[np.ndarray, int]:
    """Project aliases through a PCA basis fitted to alias-mean state centroids."""

    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("activations must have shape [state, alias, width]")
    state_centroids = values.mean(axis=1)
    center = state_centroids.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(state_centroids - center, full_matrices=False)
    if not singular.size or singular[0] <= 1e-12:
        raise ValueError("activation state centroids have zero rank")
    numerical_rank = int(np.sum(singular > singular[0] * 1e-6))
    dimension = min(
        max_dimension,
        numerical_rank,
        state_centroids.shape[0] - 1,
        values.shape[-1],
        max(sample_limit - 1, 1),
    )
    basis = vt[:dimension].T
    coordinates = (values - center) @ basis
    rms = float(np.sqrt(np.mean(coordinates**2)))
    if rms <= 1e-12:
        raise ValueError("projected activations have zero scale")
    return coordinates / rms, dimension


def _cartesian_examples(
    coordinates: np.ndarray,
    target_states: np.ndarray,
    states: np.ndarray,
    aliases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_state, source_alias = np.meshgrid(states, aliases, indexing="ij")
    source_state = source_state.ravel()
    source_alias = source_alias.ravel()
    return (
        coordinates[source_state, source_alias],
        coordinates[target_states[source_state], source_alias],
    )


def _split_indices(
    size: int, fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        return np.arange(size), np.asarray([], dtype=np.int64)
    count = int(round(size * fraction))
    count = min(max(count, 1), size - 1)
    order = rng.permutation(size)
    return np.sort(order[:count]), np.sort(order[count:])


def analyze_fold(
    coordinates: np.ndarray,
    *,
    successor: np.ndarray,
    powers: list[int],
    closure_exponent: int,
    state_train: np.ndarray,
    state_test: np.ndarray,
    alias_train: np.ndarray,
    alias_test: np.ndarray,
    precision: float,
) -> dict[str, object]:
    state_all = np.arange(coordinates.shape[0])
    alias_all = np.arange(coordinates.shape[1])
    shift_maps = successor_powers(
        successor, set(powers) | {1, closure_exponent}
    )

    joint_test_aliases = alias_test if len(alias_test) else alias_train
    split_specs = {
        "state": (state_train, alias_all, state_test, alias_all),
        "alias": (state_all, alias_train, state_all, alias_test),
        "joint": (state_train, alias_train, state_test, joint_test_aliases),
    }
    operators: dict[str, np.ndarray] = {}
    split_errors: dict[str, float] = {}
    for name, (train_states, train_aliases, test_states, test_aliases) in split_specs.items():
        if not len(train_aliases) or not len(test_aliases):
            split_errors[name] = float("nan")
            continue
        x_train, y_train = _cartesian_examples(
            coordinates, shift_maps[1], train_states, train_aliases
        )
        x_test, y_test = _cartesian_examples(
            coordinates, shift_maps[1], test_states, test_aliases
        )
        operator = orthogonal_fit(x_train, y_train)
        operators[name] = operator
        split_errors[name] = normalized_error(x_test @ operator, y_test)[0]

    joint = operators.get("joint")
    if joint is None:
        return {
            "state_cv_error": split_errors["state"],
            "alias_cv_error": split_errors["alias"],
            "joint_cv_error": float("nan"),
        }

    generator_power_errors: dict[str, float] = {}
    independent_power_errors: dict[str, float] = {}
    generator_sse = 0.0
    independent_sse = 0.0
    scalar_count = 0
    for exponent in powers:
        target_map = shift_maps[exponent]
        x_train, y_train = _cartesian_examples(
            coordinates, target_map, state_train, alias_train
        )
        x_test, y_test = _cartesian_examples(
            coordinates, target_map, state_test, joint_test_aliases
        )
        generator_prediction = x_test @ np.linalg.matrix_power(joint, exponent)
        independent = orthogonal_fit(x_train, y_train)
        independent_prediction = x_test @ independent
        generator_error, power_generator_sse, count = normalized_error(
            generator_prediction, y_test
        )
        independent_error, power_independent_sse, _ = normalized_error(
            independent_prediction, y_test
        )
        generator_power_errors[str(exponent)] = generator_error
        independent_power_errors[str(exponent)] = independent_error
        generator_sse += power_generator_sse
        independent_sse += power_independent_sse
        scalar_count += count

    dimension = coordinates.shape[-1]
    orthogonal_parameters = dimension * (dimension - 1) // 2
    generator_bits = bic_code_bits(
        sse=generator_sse,
        scalar_count=scalar_count,
        parameter_count=orthogonal_parameters,
        precision=precision,
    )
    independent_bits = bic_code_bits(
        sse=independent_sse,
        scalar_count=scalar_count,
        parameter_count=len(powers) * orthogonal_parameters,
        precision=precision,
    )
    closure = np.linalg.matrix_power(joint, closure_exponent)
    closure_matrix_error = float(
        np.linalg.norm(closure - np.eye(dimension)) / math.sqrt(dimension)
    )
    x_closure, y_closure = _cartesian_examples(
        coordinates,
        shift_maps[closure_exponent],
        state_test,
        joint_test_aliases,
    )
    closure_empirical_error = normalized_error(x_closure @ closure, y_closure)[0]
    return {
        "state_cv_error": split_errors["state"],
        "alias_cv_error": split_errors["alias"],
        "joint_cv_error": split_errors["joint"],
        "generator_power_errors": generator_power_errors,
        "independent_power_errors": independent_power_errors,
        "closure_matrix_error": closure_matrix_error,
        "closure_empirical_error": closure_empirical_error,
        "generator_bits": generator_bits,
        "independent_bits": independent_bits,
        "reuse_gain_bits": independent_bits - generator_bits,
    }


def _finite_median(values: Iterable[object]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(finite)) if finite else None


def analyze_activation_layer(
    activations: np.ndarray,
    *,
    successor: np.ndarray,
    powers: list[int],
    max_dimension: int,
    folds: int,
    fold_seed: int,
    state_train_fraction: float,
    alias_train_fraction: float,
    precision: float,
) -> dict[str, object]:
    state_count, alias_count, _ = activations.shape
    expected_states = max(1, round(state_count * state_train_fraction))
    expected_aliases = max(1, round(alias_count * alias_train_fraction))
    coordinates, dimension = project_state_subspace(
        activations,
        max_dimension=max_dimension,
        sample_limit=expected_states * expected_aliases,
    )
    fold_records: list[dict[str, object]] = []
    for fold in range(folds):
        rng = np.random.default_rng(fold_seed + fold)
        state_train, state_test = _split_indices(
            state_count, state_train_fraction, rng
        )
        alias_train, alias_test = _split_indices(
            alias_count, alias_train_fraction, rng
        )
        fold_records.append(
            analyze_fold(
                coordinates,
                successor=successor,
                powers=powers,
                closure_exponent=state_count,
                state_train=state_train,
                state_test=state_test,
                alias_train=alias_train,
                alias_test=alias_test,
                precision=precision,
            )
        )

    scalar_keys = (
        "state_cv_error",
        "alias_cv_error",
        "joint_cv_error",
        "closure_matrix_error",
        "closure_empirical_error",
        "generator_bits",
        "independent_bits",
        "reuse_gain_bits",
    )
    summary: dict[str, object] = {
        "dimension": dimension,
        **{
            key: _finite_median(record.get(key) for record in fold_records)
            for key in scalar_keys
        },
    }
    for exponent in powers:
        key = str(exponent)
        summary[f"generator_power_m{exponent}"] = _finite_median(
            record.get("generator_power_errors", {}).get(key)
            for record in fold_records
        )
        summary[f"independent_power_m{exponent}"] = _finite_median(
            record.get("independent_power_errors", {}).get(key)
            for record in fold_records
        )
    summary["folds"] = fold_records
    return summary


def degenerate_summary(powers: list[int], reason: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "dimension": 0,
        "status": reason,
        "state_cv_error": None,
        "alias_cv_error": None,
        "joint_cv_error": None,
        "closure_matrix_error": None,
        "closure_empirical_error": None,
        "generator_bits": None,
        "independent_bits": None,
        "reuse_gain_bits": None,
        "folds": [],
    }
    for exponent in powers:
        summary[f"generator_power_m{exponent}"] = None
        summary[f"independent_power_m{exponent}"] = None
    return summary


@torch.no_grad()
def extract_alias_activations(
    model: GeometryTransformer,
    *,
    layout: TokenLayout,
    order: int,
    context_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return layer activations as [layer, state, alias, width]."""

    model.eval()
    contexts = (
        layout.contexts
        if context_samples <= 0
        else min(context_samples, layout.contexts)
    )
    state_grid, alias_grid, context_grid = torch.meshgrid(
        torch.arange(order),
        torch.arange(layout.aliases),
        torch.arange(contexts),
        indexing="ij",
    )
    states_flat = state_grid.flatten()
    aliases_flat = alias_grid.flatten()
    contexts_flat = context_grid.flatten()
    flat_indices = states_flat * layout.aliases + aliases_flat
    layer_count = len(model.blocks) + 1
    sums = {
        "node": torch.zeros(
            (layer_count, order * layout.aliases, model.config.width),
            device=device,
        ),
        "output": torch.zeros(
            (layer_count, order * layout.aliases, model.config.width),
            device=device,
        ),
    }
    for start in range(0, states_flat.numel(), batch_size):
        stop = min(start + batch_size, states_flat.numel())
        left = states_flat[start:stop].to(device)
        left_alias = aliases_flat[start:stop].to(device)
        context = contexts_flat[start:stop].to(device)
        relation = torch.zeros_like(left)
        relation_alias = context.remainder(layout.aliases)
        tokens = build_tokens(
            layout,
            left,
            relation,
            contexts=context,
            left_alias=left_alias,
            right_alias=relation_alias,
        )
        _, layer_states = model(tokens, return_states=True)
        indices = flat_indices[start:stop].to(device)
        for layer, values in enumerate(layer_states):
            sums["node"][layer].index_add_(0, indices, values[:, 2].float())
            sums["output"][layer].index_add_(0, indices, values[:, -1].float())
    return {
        view: (
            values.div(float(contexts))
            .reshape(layer_count, order, layout.aliases, model.config.width)
            .cpu()
            .numpy()
        )
        for view, values in sums.items()
    }


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_behavior(run_dir: Path) -> dict[int, dict[str, object]]:
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


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _csv_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                key: value
                for key, value in record.items()
                if key != "folds" and not isinstance(value, (dict, list))
            }
        )
    return rows


def write_outputs(
    *,
    prefix: Path,
    metadata: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    payload = _json_safe({"metadata": metadata, "records": records})
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    rows = _csv_rows(records)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path} and {csv_path}")


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    order, aliases, width = 17, 4, 12
    angle = 2.0 * np.pi * np.arange(order) / order
    base = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    embedding, _ = np.linalg.qr(rng.normal(size=(width, 2)))
    activations = base[:, None, :] @ embedding.T
    activations = np.repeat(activations, aliases, axis=1)
    activations += 1e-4 * rng.normal(size=activations.shape)
    successor = (np.arange(order) + 1) % order
    structured = analyze_activation_layer(
        activations,
        successor=successor,
        powers=[1, 2, 4, 8],
        max_dimension=2,
        folds=3,
        fold_seed=10,
        state_train_fraction=0.5,
        alias_train_fraction=0.5,
        precision=1e-3,
    )
    scrambled = activations[rng.permutation(order)]
    control = analyze_activation_layer(
        scrambled,
        successor=successor,
        powers=[1, 2, 4, 8],
        max_dimension=2,
        folds=3,
        fold_seed=10,
        state_train_fraction=0.5,
        alias_train_fraction=0.5,
        precision=1e-3,
    )
    if float(structured["joint_cv_error"]) >= 0.01:
        raise AssertionError(f"structured generator error is too large: {structured}")
    if float(structured["closure_empirical_error"]) >= 0.01:
        raise AssertionError(f"structured closure error is too large: {structured}")
    if float(structured["reuse_gain_bits"]) <= 0.0:
        raise AssertionError(f"structured reuse gain is not positive: {structured}")
    if float(control["joint_cv_error"]) <= float(structured["joint_cv_error"]) + 0.5:
        raise AssertionError("scrambled control did not destroy generator transfer")
    print(
        "self-test passed: "
        f"joint={structured['joint_cv_error']:.6f}, "
        f"closure={structured['closure_empirical_error']:.6f}, "
        f"reuse_gain={structured['reuse_gain_bits']:.1f} bits"
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --self-test is used")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not (0.0 < args.state_train_fraction < 1.0):
        raise ValueError("--state-train-fraction must be between zero and one")
    if not (0.0 < args.alias_train_fraction < 1.0):
        raise ValueError("--alias-train-fraction must be between zero and one")
    if args.max_dimension < 1 or args.folds < 1 or args.precision <= 0.0:
        raise ValueError("dimension, folds, and precision must be positive")

    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text())
    task_seed = int(config.get("task_seed", config.get("split_seed", config["seed"])))
    task = make_task(
        config["task"],
        seed=task_seed,
        corruption=float(
            config.get(
                "task_corruption_fraction",
                config.get("corruption", 0.0),
            )
        ),
    )
    if task.generator is None:
        raise ValueError(f"{task.name} has no designated cyclic generator")
    table_path = run_dir / "operation_table.npy"
    operation_table = (
        np.asarray(np.load(table_path), dtype=np.int64)
        if table_path.exists()
        else np.asarray(task.table, dtype=np.int64)
    )
    successor = np.asarray(
        operation_table[:, task.generator], dtype=np.int64
    )
    aliases = int(config["aliases"])
    contexts = int(config["contexts"])
    model_config = ModelConfig(**config["model"])
    device = torch.device(args.device)
    token_seed = int(config.get("token_seed", config["seed"]))
    layout = TokenLayout(
        task.order,
        aliases,
        contexts,
        seed=token_seed,
        device=device,
    )
    layout_path = run_dir / "token_layout.npz"
    if layout_path.exists():
        saved_layout = np.load(layout_path)
        layout.node_tokens = torch.as_tensor(
            saved_layout["node"], dtype=torch.long, device=device
        )
        layout.relation_tokens = torch.as_tensor(
            saved_layout["relation"], dtype=torch.long, device=device
        )
    elif "token_seed" not in config:
        surface_count = task.order * aliases
        layout.node_tokens = (
            torch.arange(surface_count, device=device)
            .reshape(aliases, task.order)
            .add(layout.node_base)
        )
        layout.relation_tokens = (
            torch.arange(surface_count, device=device)
            .reshape(aliases, task.order)
            .add(layout.relation_base)
        )
    model = GeometryTransformer(
        vocab_size=layout.vocab_size,
        output_classes=task.order,
        config=model_config,
    )
    model.to(device)

    requested_steps = _parse_ints(args.steps)
    if args.checkpoint_glob == "checkpoint-*.pt":
        dense = sorted(run_dir.glob("weights-*.pt"))
        checkpoints = dense or sorted(run_dir.glob(args.checkpoint_glob))
    else:
        checkpoints = sorted(run_dir.glob(args.checkpoint_glob))
    if requested_steps is not None:
        wanted = set(requested_steps)
        checkpoints = [
            path
            for path in checkpoints
            if int(path.stem.rsplit("-", 1)[-1]) in wanted
        ]
    if not checkpoints:
        raise FileNotFoundError("no matching checkpoints found")
    powers = _parse_ints(args.powers) or default_powers(task.order)
    powers = [value for value in powers if value < task.order]
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    invalid_views = set(views) - {"node", "output"}
    if invalid_views or not views:
        raise ValueError(f"invalid activation views: {sorted(invalid_views)}")

    behavior = _load_behavior(run_dir)
    records: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        checkpoint = _load_checkpoint(checkpoint_path)
        step = int(checkpoint.get("step", checkpoint_path.stem.rsplit("-", 1)[-1]))
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        extracted = extract_alias_activations(
            model,
            layout=layout,
            order=task.order,
            context_samples=args.context_samples,
            batch_size=args.batch_size,
            device=device,
        )
        for view in views:
            for layer, activations in enumerate(extracted[view]):
                try:
                    result = analyze_activation_layer(
                        activations,
                        successor=successor,
                        powers=powers,
                        max_dimension=args.max_dimension,
                        folds=args.folds,
                        fold_seed=args.fold_seed,
                        state_train_fraction=args.state_train_fraction,
                        alias_train_fraction=args.alias_train_fraction,
                        precision=args.precision,
                    )
                except ValueError as error:
                    if "zero" not in str(error):
                        raise
                    result = degenerate_summary(powers, str(error))
                records.append(
                    {
                        "step": step,
                        "checkpoint": checkpoint_path.name,
                        "view": view,
                        "layer": layer,
                        **behavior.get(step, {}),
                        **result,
                    }
                )
        print(
            f"{config['run_name']} step={step}: "
            f"{len(views)} views x {model_config.depth + 1} layers"
        )

    prefix = args.output_prefix or (run_dir / "operator_reuse")
    metadata = {
        "run_name": config["run_name"],
        "task": task.name,
        "order": task.order,
        "aliases": aliases,
        "contexts_averaged": (
            contexts if args.context_samples <= 0 else min(args.context_samples, contexts)
        ),
        "powers": powers,
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "state_train_fraction": args.state_train_fraction,
        "alias_train_fraction": args.alias_train_fraction,
        "max_dimension": args.max_dimension,
        "precision": args.precision,
        "code": (
            "fixed-precision Gaussian residual bits plus a BIC penalty; "
            "the shared model pays for one orthogonal generator and the "
            "independent model pays for one orthogonal operator per power"
        ),
        "projection": (
            "PCA of alias-mean state centroids, fitted without transition labels"
        ),
    }
    write_outputs(prefix=prefix, metadata=metadata, records=records)


if __name__ == "__main__":
    main()
