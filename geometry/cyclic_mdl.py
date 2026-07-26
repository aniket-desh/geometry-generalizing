from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from geogen.model import GeometryTransformer, ModelConfig
from geogen.tasks import make_task
from operator_reuse import (
    _load_behavior,
    _load_checkpoint,
    _parse_ints,
    _split_indices,
    bic_code_bits,
    default_powers,
    extract_alias_activations,
)
from train import TokenLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether cyclic activations are compressed by a sparse, "
            "finite-order Fourier representation. Frequency selection and "
            "coefficients are fit without held-out states or aliases."
        )
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--checkpoint-glob", default="checkpoint-*.pt")
    parser.add_argument("--steps")
    parser.add_argument("--views", default="node,output")
    parser.add_argument("--max-dimension", type=int, default=16)
    parser.add_argument("--max-frequencies", type=int, default=16)
    parser.add_argument("--powers")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=314159)
    parser.add_argument("--state-train-fraction", type=float, default=0.6)
    parser.add_argument("--alias-train-fraction", type=float, default=0.5)
    parser.add_argument("--inner-train-fraction", type=float, default=0.75)
    parser.add_argument("--precision", type=float, default=1e-3)
    parser.add_argument("--context-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def real_fourier_features(
    states: np.ndarray,
    *,
    order: int,
    frequencies: Iterable[int],
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64).reshape(-1)
    columns = [np.ones_like(states)]
    scale = math.sqrt(2.0)
    for frequency in frequencies:
        angle = 2.0 * math.pi * int(frequency) * states / order
        columns.extend((scale * np.cos(angle), scale * np.sin(angle)))
    return np.stack(columns, axis=1)


def fit_fourier(
    coordinates: np.ndarray,
    *,
    states: np.ndarray,
    aliases: np.ndarray,
    labels: np.ndarray,
    order: int,
    frequencies: list[int],
) -> np.ndarray:
    features = real_fourier_features(
        labels[states], order=order, frequencies=frequencies
    )
    repeated_features = np.repeat(features, len(aliases), axis=0)
    targets = coordinates[np.ix_(states, aliases)].reshape(
        len(states) * len(aliases), coordinates.shape[-1]
    )
    return np.linalg.lstsq(repeated_features, targets, rcond=None)[0]


def predict_fourier(
    coefficients: np.ndarray,
    *,
    states: np.ndarray,
    labels: np.ndarray,
    order: int,
    frequencies: list[int],
) -> np.ndarray:
    return (
        real_fourier_features(
            labels[states], order=order, frequencies=frequencies
        )
        @ coefficients
    )


def normalized_error(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[float, float, int]:
    residual = np.asarray(prediction) - np.asarray(target)
    sse = float(np.sum(residual**2))
    denominator = float(np.sum(np.asarray(target) ** 2))
    return (
        math.sqrt(sse / max(denominator, 1e-30)),
        sse,
        int(residual.size),
    )


def project_fold(
    activations: np.ndarray,
    *,
    states: np.ndarray,
    aliases: np.ndarray,
    max_dimension: int,
) -> tuple[np.ndarray, int]:
    values = np.asarray(activations, dtype=np.float64)
    train_values = values[np.ix_(states, aliases)]
    center = train_values.mean(axis=(0, 1), keepdims=True)
    state_centroids = train_values.mean(axis=1)
    centered_centroids = state_centroids - center.reshape(1, -1)
    _, singular, vt = np.linalg.svd(centered_centroids, full_matrices=False)
    if not singular.size or singular[0] <= 1e-12:
        raise ValueError("training state centroids have zero rank")
    rank = int(np.sum(singular > singular[0] * 1e-6))
    dimension = min(
        max_dimension,
        rank,
        len(states) - 1,
        values.shape[-1],
    )
    basis = vt[:dimension].T
    coordinates = (values - center) @ basis
    scale = float(
        np.sqrt(np.mean(coordinates[np.ix_(states, aliases)] ** 2))
    )
    if scale <= 1e-12:
        raise ValueError("projected training activations have zero scale")
    return coordinates / scale, dimension


def frequency_subset_bits(order: int, count: int) -> float:
    available = (order - 1) // 2
    if not 0 <= count <= available:
        raise ValueError("invalid Fourier frequency count")
    log_choose = (
        math.lgamma(available + 1)
        - math.lgamma(count + 1)
        - math.lgamma(available - count + 1)
    ) / math.log(2.0)
    return math.log2(available + 1) + log_choose


def model_code_bits(
    *,
    sse: float,
    scalar_count: int,
    continuous_parameters: int,
    discrete_bits: float,
    precision: float,
) -> float:
    return (
        bic_code_bits(
            sse=sse,
            scalar_count=scalar_count,
            parameter_count=continuous_parameters,
            precision=precision,
        )
        + discrete_bits
    )


def greedy_frequency_path(
    coordinates: np.ndarray,
    *,
    fit_states: np.ndarray,
    fit_aliases: np.ndarray,
    labels: np.ndarray,
    order: int,
    max_frequencies: int,
) -> list[list[int]]:
    available = list(range(1, (order - 1) // 2 + 1))
    selected: list[int] = []
    path: list[list[int]] = [[]]
    target = coordinates[np.ix_(fit_states, fit_aliases)].reshape(
        len(fit_states) * len(fit_aliases), coordinates.shape[-1]
    )
    while len(selected) < min(max_frequencies, len(available)):
        best_frequency: int | None = None
        best_sse = float("inf")
        for frequency in available:
            candidate = sorted(selected + [frequency])
            coefficients = fit_fourier(
                coordinates,
                states=fit_states,
                aliases=fit_aliases,
                labels=labels,
                order=order,
                frequencies=candidate,
            )
            prediction = np.repeat(
                predict_fourier(
                    coefficients,
                    states=fit_states,
                    labels=labels,
                    order=order,
                    frequencies=candidate,
                ),
                len(fit_aliases),
                axis=0,
            )
            sse = float(np.sum((prediction - target) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_frequency = frequency
        if best_frequency is None:
            break
        selected.append(best_frequency)
        selected.sort()
        available.remove(best_frequency)
        path.append(selected.copy())
    return path


def choose_frequencies(
    coordinates: np.ndarray,
    *,
    fit_states: np.ndarray,
    validation_states: np.ndarray,
    aliases: np.ndarray,
    labels: np.ndarray,
    order: int,
    max_frequencies: int,
    precision: float,
) -> tuple[list[int], list[dict[str, object]]]:
    if not len(fit_states) or not len(validation_states):
        raise ValueError("frequency selection requires nonempty paired splits")
    max_identifiable = max((len(fit_states) - 1) // 2, 0)
    path = greedy_frequency_path(
        coordinates,
        fit_states=fit_states,
        fit_aliases=aliases,
        labels=labels,
        order=order,
        max_frequencies=min(max_frequencies, max_identifiable),
    )
    validation_target = coordinates[
        np.ix_(validation_states, aliases)
    ].reshape(len(validation_states) * len(aliases), coordinates.shape[-1])
    records: list[dict[str, object]] = []
    for frequencies in path:
        coefficients = fit_fourier(
            coordinates,
            states=fit_states,
            aliases=aliases,
            labels=labels,
            order=order,
            frequencies=frequencies,
        )
        prediction = np.repeat(
            predict_fourier(
                coefficients,
                states=validation_states,
                labels=labels,
                order=order,
                frequencies=frequencies,
            ),
            len(aliases),
            axis=0,
        )
        error, sse, scalar_count = normalized_error(
            prediction, validation_target
        )
        parameter_count = (1 + 2 * len(frequencies)) * coordinates.shape[-1]
        code_bits = model_code_bits(
            sse=sse,
            scalar_count=scalar_count,
            continuous_parameters=parameter_count,
            discrete_bits=frequency_subset_bits(order, len(frequencies)),
            precision=precision,
        )
        records.append(
            {
                "frequency_count": len(frequencies),
                "frequencies": frequencies,
                "validation_error": error,
                "validation_code_bits": code_bits,
            }
        )
    best = min(
        records,
        key=lambda record: (
            float(record["validation_code_bits"]),
            int(record["frequency_count"]),
        ),
    )
    return list(best["frequencies"]), records


def latent_generator(order: int, frequencies: list[int]) -> np.ndarray:
    dimension = 1 + 2 * len(frequencies)
    generator = np.eye(dimension)
    for index, frequency in enumerate(frequencies):
        angle = 2.0 * math.pi * frequency / order
        cosine, sine = math.cos(angle), math.sin(angle)
        start = 1 + 2 * index
        generator[start : start + 2, start : start + 2] = np.asarray(
            [[cosine, sine], [-sine, cosine]]
        )
    return generator


def code_comparison(
    coordinates: np.ndarray,
    *,
    alias_train: np.ndarray,
    alias_test: np.ndarray,
    labels: np.ndarray,
    order: int,
    frequencies: list[int],
    precision: float,
) -> dict[str, float]:
    states = np.arange(order)
    test_target = coordinates[np.ix_(states, alias_test)].reshape(
        order * len(alias_test), coordinates.shape[-1]
    )
    lookup_centroids = coordinates[
        np.ix_(states, alias_train)
    ].mean(axis=1)
    lookup_prediction = np.repeat(
        lookup_centroids, len(alias_test), axis=0
    )
    lookup_error, lookup_sse, scalar_count = normalized_error(
        lookup_prediction, test_target
    )
    coefficients = fit_fourier(
        coordinates,
        states=states,
        aliases=alias_train,
        labels=labels,
        order=order,
        frequencies=frequencies,
    )
    group_prediction = np.repeat(
        predict_fourier(
            coefficients,
            states=states,
            labels=labels,
            order=order,
            frequencies=frequencies,
        ),
        len(alias_test),
        axis=0,
    )
    group_error, group_sse, _ = normalized_error(
        group_prediction, test_target
    )
    null_centroid = coordinates[
        np.ix_(states, alias_train)
    ].mean(axis=(0, 1), keepdims=False).reshape(1, -1)
    null_prediction = np.broadcast_to(null_centroid, test_target.shape)
    null_error, null_sse, _ = normalized_error(null_prediction, test_target)
    dimension = coordinates.shape[-1]
    lookup_parameters = order * dimension
    group_parameters = (1 + 2 * len(frequencies)) * dimension
    lookup_bits = model_code_bits(
        sse=lookup_sse,
        scalar_count=scalar_count,
        continuous_parameters=lookup_parameters,
        discrete_bits=0.0,
        precision=precision,
    )
    group_bits = model_code_bits(
        sse=group_sse,
        scalar_count=scalar_count,
        continuous_parameters=group_parameters,
        discrete_bits=frequency_subset_bits(order, len(frequencies)),
        precision=precision,
    )
    null_bits = model_code_bits(
        sse=null_sse,
        scalar_count=scalar_count,
        continuous_parameters=dimension,
        discrete_bits=0.0,
        precision=precision,
    )
    return {
        "alias_lookup_error": lookup_error,
        "alias_group_error": group_error,
        "alias_null_error": null_error,
        "alias_lookup_bits": lookup_bits,
        "alias_group_bits": group_bits,
        "alias_null_bits": null_bits,
        "alias_mdl_gain_bits": lookup_bits - group_bits,
        "alias_shared_vs_null_gain_bits": null_bits - group_bits,
        "alias_usable_gain_bits": min(lookup_bits, null_bits) - group_bits,
        "alias_group_to_lookup_sse": group_sse / max(lookup_sse, 1e-30),
        "alias_group_r2_vs_null": 1.0 - group_sse / max(null_sse, 1e-30),
        "alias_lookup_r2_vs_null": 1.0
        - lookup_sse / max(null_sse, 1e-30),
        "lookup_parameters": float(lookup_parameters),
        "group_parameters": float(group_parameters),
        "parameter_compression": lookup_parameters / max(group_parameters, 1),
    }


def analyze_model(
    coordinates: np.ndarray,
    *,
    order: int,
    state_train: np.ndarray,
    state_test: np.ndarray,
    alias_train: np.ndarray,
    alias_test: np.ndarray,
    labels: np.ndarray,
    powers: list[int],
    max_frequencies: int,
    selection_fit_states: np.ndarray,
    selection_validation_states: np.ndarray,
    precision: float,
) -> dict[str, object]:
    frequencies, selection_path = choose_frequencies(
        coordinates,
        fit_states=selection_fit_states,
        validation_states=selection_validation_states,
        aliases=alias_train,
        labels=labels,
        order=order,
        max_frequencies=max_frequencies,
        precision=precision,
    )
    coefficients = fit_fourier(
        coordinates,
        states=state_train,
        aliases=alias_train,
        labels=labels,
        order=order,
        frequencies=frequencies,
    )
    test_target = coordinates[
        np.ix_(state_test, alias_test)
    ].reshape(len(state_test) * len(alias_test), coordinates.shape[-1])
    test_prediction = np.repeat(
        predict_fourier(
            coefficients,
            states=state_test,
            labels=labels,
            order=order,
            frequencies=frequencies,
        ),
        len(alias_test),
        axis=0,
    )
    joint_error = normalized_error(test_prediction, test_target)[0]
    generator = latent_generator(order, frequencies)
    closure_error = float(
        np.linalg.norm(
            np.linalg.matrix_power(generator, order)
            - np.eye(generator.shape[0])
        )
        / math.sqrt(generator.shape[0])
    )
    power_errors: dict[str, float] = {}
    source_features = real_fourier_features(
        labels[state_test], order=order, frequencies=frequencies
    )
    for power in powers:
        target_states = (state_test + power) % order
        target = coordinates[
            np.ix_(target_states, alias_test)
        ].reshape(len(target_states) * len(alias_test), coordinates.shape[-1])
        prediction = np.repeat(
            source_features
            @ np.linalg.matrix_power(generator, power)
            @ coefficients,
            len(alias_test),
            axis=0,
        )
        power_errors[str(power)] = normalized_error(prediction, target)[0]
    return {
        "frequency_count": len(frequencies),
        "frequencies": frequencies,
        "latent_dimension": 1 + 2 * len(frequencies),
        "joint_state_alias_error": joint_error,
        "finite_order_closure_error": closure_error,
        "power_errors": power_errors,
        "selection_path": selection_path,
        "selection_fit_states": selection_fit_states.tolist(),
        "selection_validation_states": selection_validation_states.tolist(),
        **code_comparison(
            coordinates,
            alias_train=alias_train,
            alias_test=alias_test,
            labels=labels,
            order=order,
            frequencies=frequencies,
            precision=precision,
        ),
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
    max_dimension: int,
    max_frequencies: int,
    powers: list[int],
    folds: int,
    fold_seed: int,
    state_train_fraction: float,
    alias_train_fraction: float,
    inner_train_fraction: float,
    precision: float,
) -> dict[str, object]:
    order, alias_count, _ = activations.shape
    if alias_count < 2:
        raise ValueError("cyclic MDL requires at least two aliases")
    fold_records: list[dict[str, object]] = []
    for fold in range(folds):
        rng = np.random.default_rng(fold_seed + fold)
        state_train, state_test = _split_indices(
            order, state_train_fraction, rng
        )
        alias_train, alias_test = _split_indices(
            alias_count, alias_train_fraction, rng
        )
        inner_train_index, inner_validation_index = _split_indices(
            len(state_train), inner_train_fraction, rng
        )
        selection_fit_states = state_train[inner_train_index]
        selection_validation_states = state_train[inner_validation_index]
        coordinates, dimension = project_fold(
            activations,
            states=state_train,
            aliases=alias_train,
            max_dimension=max_dimension,
        )
        true_labels = np.arange(order)
        structured = analyze_model(
            coordinates,
            order=order,
            state_train=state_train,
            state_test=state_test,
            alias_train=alias_train,
            alias_test=alias_test,
            labels=true_labels,
            powers=powers,
            max_frequencies=max_frequencies,
            selection_fit_states=selection_fit_states,
            selection_validation_states=selection_validation_states,
            precision=precision,
        )
        scrambled_labels = rng.permutation(order)
        scrambled = analyze_model(
            coordinates,
            order=order,
            state_train=state_train,
            state_test=state_test,
            alias_train=alias_train,
            alias_test=alias_test,
            labels=scrambled_labels,
            powers=powers,
            max_frequencies=max_frequencies,
            selection_fit_states=selection_fit_states,
            selection_validation_states=selection_validation_states,
            precision=precision,
        )
        fold_records.append(
            {
                "fold": fold,
                "dimension": dimension,
                "state_train": state_train.tolist(),
                "state_test": state_test.tolist(),
                "alias_train": alias_train.tolist(),
                "alias_test": alias_test.tolist(),
                "selection_fit_states": selection_fit_states.tolist(),
                "selection_validation_states": (
                    selection_validation_states.tolist()
                ),
                "structured": structured,
                "scrambled": scrambled,
            }
        )
    scalar_keys = (
        "frequency_count",
        "latent_dimension",
        "joint_state_alias_error",
        "finite_order_closure_error",
        "alias_lookup_error",
        "alias_group_error",
        "alias_null_error",
        "alias_lookup_bits",
        "alias_group_bits",
        "alias_null_bits",
        "alias_mdl_gain_bits",
        "alias_shared_vs_null_gain_bits",
        "alias_usable_gain_bits",
        "alias_group_to_lookup_sse",
        "alias_group_r2_vs_null",
        "alias_lookup_r2_vs_null",
        "lookup_parameters",
        "group_parameters",
        "parameter_compression",
    )
    summary: dict[str, object] = {
        "dimension": int(
            round(np.median([record["dimension"] for record in fold_records]))
        ),
        **{
            key: _finite_median(
                record["structured"].get(key) for record in fold_records
            )
            for key in scalar_keys
        },
        **{
            f"scrambled_{key}": _finite_median(
                record["scrambled"].get(key) for record in fold_records
            )
            for key in scalar_keys
        },
    }
    summary["state_alias_advantage"] = (
        float(summary["scrambled_joint_state_alias_error"])
        - float(summary["joint_state_alias_error"])
    )
    for power in powers:
        summary[f"power_m{power}"] = _finite_median(
            record["structured"]["power_errors"].get(str(power))
            for record in fold_records
        )
        summary[f"scrambled_power_m{power}"] = _finite_median(
            record["scrambled"]["power_errors"].get(str(power))
            for record in fold_records
        )
    summary["folds"] = fold_records
    return summary


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_outputs(
    prefix: Path,
    *,
    metadata: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            _json_safe({"metadata": metadata, "records": records}), indent=2
        )
        + "\n"
    )
    rows = [
        {
            key: value
            for key, value in record.items()
            if key != "folds" and not isinstance(value, (list, dict))
        }
        for record in records
    ]
    fieldnames = sorted({key for row in rows for key in row})
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {prefix.with_suffix('.json')} and {prefix.with_suffix('.csv')}")


def run_self_test() -> None:
    rng = np.random.default_rng(23)
    order, aliases, width = 31, 4, 10
    states = np.arange(order)
    features = real_fourier_features(
        states, order=order, frequencies=[3, 7]
    )
    loadings = rng.normal(size=(features.shape[1], width))
    exact = features[:, None, :] @ loadings[None, :, :]
    exact = np.repeat(exact, aliases, axis=1)
    noisy = exact + 0.03 * rng.normal(size=exact.shape)
    scrambled = exact[rng.permutation(order)]
    kwargs = {
        "max_dimension": 8,
        "max_frequencies": 6,
        "powers": [1, 2, 4, 8, 16],
        "folds": 3,
        "fold_seed": 9,
        "state_train_fraction": 0.65,
        "alias_train_fraction": 0.5,
        "inner_train_fraction": 0.75,
        "precision": 1e-3,
    }
    exact_result = analyze_activation_layer(exact, **kwargs)
    noisy_result = analyze_activation_layer(noisy, **kwargs)
    scrambled_result = analyze_activation_layer(scrambled, **kwargs)
    if float(exact_result["joint_state_alias_error"]) > 1e-8:
        raise AssertionError(f"exact cyclic factorization failed: {exact_result}")
    if float(exact_result["finite_order_closure_error"]) > 1e-10:
        raise AssertionError("finite-order generator did not close")
    if float(exact_result["alias_mdl_gain_bits"]) <= 0.0:
        raise AssertionError("exact cyclic code did not beat a lookup")
    if float(exact_result["alias_usable_gain_bits"]) <= 0.0:
        raise AssertionError("exact cyclic code did not beat the usable baseline")
    if float(noisy_result["joint_state_alias_error"]) >= 0.1:
        raise AssertionError(f"noisy cyclic factorization failed: {noisy_result}")
    if (
        float(scrambled_result["joint_state_alias_error"])
        <= float(noisy_result["joint_state_alias_error"]) + 0.5
    ):
        raise AssertionError("scrambled state order was not rejected")
    for fold in exact_result["folds"]:
        structured = fold["structured"]
        control = fold["scrambled"]
        if (
            structured["selection_fit_states"]
            != control["selection_fit_states"]
            or structured["selection_validation_states"]
            != control["selection_validation_states"]
        ):
            raise AssertionError(
                "structured and scrambled models used different inner splits"
            )
        if (
            fold["selection_fit_states"]
            != structured["selection_fit_states"]
            or fold["selection_validation_states"]
            != structured["selection_validation_states"]
        ):
            raise AssertionError("stored inner split plan does not match analysis")
    expected_usable_gain = _finite_median(
        fold["structured"]["alias_usable_gain_bits"]
        for fold in exact_result["folds"]
    )
    if exact_result["alias_usable_gain_bits"] != expected_usable_gain:
        raise AssertionError("usable gain was not aggregated from foldwise values")

    zero_frequency = np.repeat(
        rng.normal(size=(1, aliases, 3)), order, axis=0
    )
    zero_result = code_comparison(
        zero_frequency,
        alias_train=np.asarray([0, 1]),
        alias_test=np.asarray([2, 3]),
        labels=np.arange(order),
        order=order,
        frequencies=[],
        precision=1e-3,
    )
    if float(zero_result["alias_mdl_gain_bits"]) <= 0.0:
        raise AssertionError(
            "zero-frequency fixture did not expose the lookup-only false positive"
        )
    if float(zero_result["alias_shared_vs_null_gain_bits"]) >= 0.0:
        raise AssertionError("zero-frequency shared code falsely beat the null")
    if float(zero_result["alias_usable_gain_bits"]) >= 0.0:
        raise AssertionError("zero-frequency fixture produced usable compression")
    print(
        "self-test passed: "
        f"exact_joint={exact_result['joint_state_alias_error']:.3g}, "
        f"noisy_joint={noisy_result['joint_state_alias_error']:.3f}, "
        f"scrambled_joint={scrambled_result['joint_state_alias_error']:.3f}, "
        f"exact_usable_gain={exact_result['alias_usable_gain_bits']:.1f} bits, "
        f"zero_usable_gain={zero_result['alias_usable_gain_bits']:.1f} bits"
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
    for name, value in (
        ("state train fraction", args.state_train_fraction),
        ("alias train fraction", args.alias_train_fraction),
        ("inner train fraction", args.inner_train_fraction),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie between zero and one")

    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text())
    task_seed = int(
        config.get("task_seed", config.get("split_seed", config["seed"]))
    )
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
    aliases = int(config["aliases"])
    contexts = int(config["contexts"])
    device = torch.device(args.device)
    layout = TokenLayout(
        task.order,
        aliases,
        contexts,
        seed=int(config.get("token_seed", config["seed"])),
        device=device,
    )
    layout_path = run_dir / "token_layout.npz"
    if layout_path.exists():
        saved = np.load(layout_path)
        layout.node_tokens = torch.as_tensor(
            saved["node"], dtype=torch.long, device=device
        )
        layout.relation_tokens = torch.as_tensor(
            saved["relation"], dtype=torch.long, device=device
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
    model_config = ModelConfig(**config["model"])
    model = GeometryTransformer(
        vocab_size=layout.vocab_size,
        output_classes=task.order,
        config=model_config,
    ).to(device)

    requested_steps = _parse_ints(args.steps)
    if args.checkpoint_glob == "checkpoint-*.pt":
        dense = sorted(run_dir.glob("weights-*.pt"))
        checkpoints = dense or sorted(run_dir.glob(args.checkpoint_glob))
    else:
        checkpoints = sorted(run_dir.glob(args.checkpoint_glob))
    if requested_steps is not None:
        checkpoints = [
            path
            for path in checkpoints
            if int(path.stem.rsplit("-", 1)[-1]) in set(requested_steps)
        ]
    if not checkpoints:
        raise FileNotFoundError("no matching checkpoints found")
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    if not views or set(views) - {"node", "output"}:
        raise ValueError(f"invalid activation views: {views}")
    powers = _parse_ints(args.powers) or default_powers(task.order)
    powers = [power for power in powers if power < task.order]
    behavior = _load_behavior(run_dir)
    records: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        checkpoint = _load_checkpoint(checkpoint_path)
        step = int(
            checkpoint.get(
                "step", checkpoint_path.stem.rsplit("-", 1)[-1]
            )
        )
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
                        max_dimension=args.max_dimension,
                        max_frequencies=args.max_frequencies,
                        powers=powers,
                        folds=args.folds,
                        fold_seed=args.fold_seed,
                        state_train_fraction=args.state_train_fraction,
                        alias_train_fraction=args.alias_train_fraction,
                        inner_train_fraction=args.inner_train_fraction,
                        precision=args.precision,
                    )
                except ValueError as error:
                    if "zero rank" not in str(error):
                        raise
                    result = {"status": str(error), "folds": []}
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
    prefix = args.output_prefix or (run_dir / "cyclic_mdl")
    write_outputs(
        prefix,
        metadata={
            "run_name": config["run_name"],
            "task": task.name,
            "order": task.order,
            "aliases": aliases,
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "state_train_fraction": args.state_train_fraction,
            "alias_train_fraction": args.alias_train_fraction,
            "inner_train_fraction": args.inner_train_fraction,
            "max_dimension": args.max_dimension,
            "max_frequencies": args.max_frequencies,
            "powers": powers,
            "precision": args.precision,
            "method": (
                "Nested state and alias cross-validation of a sparse real "
                "Fourier factorization of C_p. The latent generator is block "
                "diagonal with exact allowed irrep angles, so finite order is "
                "enforced rather than assessed after unconstrained Procrustes."
            ),
            "mdl": (
                "Held-out-alias Gaussian residual code plus BIC penalties for "
                "continuous parameters and an explicit log binomial code for "
                "the selected frequency subset. The lookup transmits one "
                "centroid per state; the group code transmits one intercept "
                "and two loading vectors per selected real Fourier irrep. A "
                "separate intercept-only null is coded in every fold, and the "
                "usable gain compares the group code with the better of the "
                "lookup and null codes in that same fold."
            ),
            "interpretation_guard": (
                "A positive usable MDL gain counts as compression only when "
                "the group fit beats both the lookup and intercept-only null "
                "within folds, explains held-out-alias variance, and its "
                "jointly held-out state-and-alias error beats the paired "
                "scrambled-order control. This is a representational "
                "factorization, not a causal activation-transport result."
            ),
        },
        records=records,
    )


if __name__ == "__main__":
    main()
