from __future__ import annotations

import math

import numpy as np

from .tasks import TaskSpec


def centered(activations: np.ndarray) -> np.ndarray:
    return activations - activations.mean(axis=0, keepdims=True)


def gram(activations: np.ndarray) -> np.ndarray:
    h = centered(np.asarray(activations, dtype=np.float64))
    return h @ h.T


def effective_rank(activations: np.ndarray) -> float:
    singular = np.linalg.svd(centered(activations), compute_uv=False)
    energy = singular**2
    if energy.sum() <= 1e-20:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 1e-15]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def action_invariance_defect(
    activations: np.ndarray, task: TaskSpec
) -> float:
    g = gram(activations)
    squared_norm = float(np.sum(g**2))
    denominator = math.sqrt(squared_norm)
    if denominator <= 1e-20:
        return float("nan")
    if task.family == "cycle" and task.relation_count == task.order:
        indices = np.arange(task.order)
        diagonals = np.stack(
            [g[indices, (indices + offset) % task.order] for offset in indices]
        )
        spectrum = np.fft.fft(diagonals, axis=1)
        correlations = np.fft.ifft(
            np.abs(spectrum) ** 2, axis=1
        ).real.sum(axis=0)
        squared_defects = np.maximum(
            2.0 * (squared_norm - correlations), 0.0
        )
        return float(np.mean(np.sqrt(squared_defects) / denominator))
    defects: list[float] = []
    for relation in range(task.relation_count):
        permutation = task.table[:, relation]
        if np.unique(permutation).size != task.order:
            continue
        moved = g[np.ix_(permutation, permutation)]
        defects.append(float(np.linalg.norm(g - moved) / denominator))
    return float(np.mean(defects)) if defects else float("nan")


def cyclic_gram_defect(activations: np.ndarray) -> float:
    g = gram(activations)
    denominator = np.linalg.norm(g)
    if denominator <= 1e-20:
        return float("nan")
    order = g.shape[0]
    row, column = np.indices(g.shape)
    offset = (column - row) % order
    diagonal_means = np.bincount(
        offset.ravel(), weights=g.ravel(), minlength=order
    ) / order
    symmetrized = diagonal_means[offset]
    return float(np.linalg.norm(g - symmetrized) / denominator)


def fourier_energy(activations: np.ndarray) -> tuple[float, list[float]]:
    h = centered(np.asarray(activations, dtype=np.float64))
    spectrum = np.fft.rfft(h, axis=0)
    energy = np.sum(np.abs(spectrum) ** 2, axis=1)
    if energy.size:
        energy[0] = 0.0
    total = energy.sum()
    fractions = energy / total if total > 1e-20 else np.zeros_like(energy)
    fundamental = float(fractions[1]) if fractions.size > 1 else 0.0
    return fundamental, fractions.astype(float).tolist()


def _orthogonal_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return u @ vt


def generator_metrics(
    activations: np.ndarray, task: TaskSpec
) -> tuple[float, float]:
    if task.generator is None:
        return float("nan"), float("nan")
    h = centered(np.asarray(activations, dtype=np.float64))
    u, singular, _ = np.linalg.svd(h, full_matrices=False)
    if singular.size == 0 or singular[0] <= 1e-12:
        return float("nan"), float("nan")
    fold_size = min(
        int((np.arange(task.order) % 2 == parity).sum())
        for parity in (0, 1)
    )
    dimension = min(
        16,
        task.order - 1,
        max(fold_size - 1, 1),
        int((singular > singular[0] * 1e-6).sum()),
    )
    if dimension < 1:
        return float("nan"), float("nan")
    coordinates = u[:, :dimension] * singular[:dimension]
    target = coordinates[task.table[:, task.generator]]

    errors: list[float] = []
    operators: list[np.ndarray] = []
    for parity in (0, 1):
        train = np.arange(task.order) % 2 == parity
        test = ~train
        if train.sum() < dimension or test.sum() == 0:
            continue
        operator = _orthogonal_fit(coordinates[train], target[train])
        prediction = coordinates[test] @ operator
        scale = np.linalg.norm(target[test] - target[test].mean(axis=0))
        if scale > 1e-12:
            errors.append(float(np.linalg.norm(prediction - target[test]) / scale))
        operators.append(operator)
    heldout_error = float(np.mean(errors)) if errors else float("nan")

    if not operators:
        return heldout_error, float("nan")
    operator = _orthogonal_fit(coordinates, target)
    permutation = task.table[:, task.generator]
    moved = np.arange(task.order)
    generator_order = task.order
    for exponent in range(1, task.order + 1):
        moved = permutation[moved]
        if np.array_equal(moved, np.arange(task.order)):
            generator_order = exponent
            break
    closure = np.linalg.matrix_power(operator, generator_order)
    closure_error = float(
        np.linalg.norm(closure - np.eye(dimension)) / math.sqrt(dimension)
    )
    return heldout_error, closure_error


def table_compositionality(task: TaskSpec) -> float:
    functions = [task.table[:, relation] for relation in range(task.relation_count)]
    errors: list[float] = []
    for left in range(task.relation_count):
        for right in range(task.relation_count):
            composed = functions[right][functions[left]]
            best = min(np.mean(composed != candidate) for candidate in functions)
            errors.append(float(best))
    return float(np.mean(errors))


def geometry_summary(activations: np.ndarray, task: TaskSpec) -> dict[str, object]:
    fundamental, spectrum = fourier_energy(activations)
    mode_probabilities = np.asarray(spectrum, dtype=np.float64)
    mode_probabilities = mode_probabilities[mode_probabilities > 1e-15]
    fourier_mode_rank = (
        float(
            np.exp(
                -np.sum(mode_probabilities * np.log(mode_probabilities))
            )
        )
        if mode_probabilities.size
        else 0.0
    )
    top_fourier_energy = float(
        np.sort(mode_probabilities)[-min(3, mode_probabilities.size) :].sum()
    )
    dominant_frequency = (
        int(np.argmax(spectrum)) if len(spectrum) else 0
    )
    generator_error, closure_error = generator_metrics(activations, task)
    h = centered(activations)
    return {
        "effective_rank": effective_rank(h),
        "centroid_rms": float(np.sqrt(np.mean(np.sum(h**2, axis=1)))),
        "action_defect": action_invariance_defect(h, task),
        "cyclic_defect": (
            cyclic_gram_defect(h)
            if task.family in {"cycle", "broken_cycle"}
            else float("nan")
        ),
        "fundamental_energy": fundamental,
        "fourier_energy": spectrum,
        "fourier_mode_rank": fourier_mode_rank,
        "top_fourier_energy": top_fourier_energy,
        "dominant_frequency": dominant_frequency,
        "generator_error": generator_error,
        "closure_error": closure_error,
    }
