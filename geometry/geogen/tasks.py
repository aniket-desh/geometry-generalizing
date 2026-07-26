from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    table: np.ndarray
    generator: int | None
    description: str
    corruption_fraction: float = 0.0

    @property
    def order(self) -> int:
        return int(self.table.shape[0])

    @property
    def relation_count(self) -> int:
        return int(self.table.shape[1])

    def validate(self) -> None:
        if self.table.ndim != 2:
            raise ValueError("operation table must be two-dimensional")
        if self.table.shape[0] != self.table.shape[1]:
            raise ValueError("this benchmark expects one relation per latent state")
        if self.table.min() < 0 or self.table.max() >= self.order:
            raise ValueError("operation table contains an invalid state")


def _cycle(order: int) -> np.ndarray:
    a = np.arange(order)[:, None]
    b = np.arange(order)[None, :]
    return (a + b) % order


def _torus(side: int) -> np.ndarray:
    order = side * side
    table = np.empty((order, order), dtype=np.int64)
    for a in range(order):
        ax, ay = divmod(a, side)
        for b in range(order):
            bx, by = divmod(b, side)
            table[a, b] = ((ax + bx) % side) * side + ((ay + by) % side)
    return table


def _dihedral(order: int) -> np.ndarray:
    if order % 2:
        raise ValueError("a dihedral group must have even order")
    rotations = order // 2
    table = np.empty((order, order), dtype=np.int64)
    for a in range(order):
        ar, af = a % rotations, a // rotations
        for b in range(order):
            br, bf = b % rotations, b // rotations
            rotation = (ar + (-1 if af else 1) * br) % rotations
            reflection = af ^ bf
            table[a, b] = rotation + reflection * rotations
    return table


def _xor(order: int) -> np.ndarray:
    if order < 2 or order & (order - 1):
        raise ValueError("XOR order must be a power of two")
    a = np.arange(order)[:, None]
    b = np.arange(order)[None, :]
    return np.bitwise_xor(a, b).astype(np.int64)


def _tree_lca(order: int) -> np.ndarray:
    table = np.empty((order, order), dtype=np.int64)
    paths = [format(node + 1, "b") for node in range(order)]
    for left, left_path in enumerate(paths):
        for right, right_path in enumerate(paths):
            common = []
            for left_bit, right_bit in zip(left_path, right_path):
                if left_bit != right_bit:
                    break
                common.append(left_bit)
            table[left, right] = int("".join(common), 2) - 1
    return table


def _path(order: int) -> np.ndarray:
    shifts = np.arange(order) - order // 2
    states = np.arange(order)[:, None]
    return np.clip(states + shifts[None, :], 0, order - 1).astype(np.int64)


def _random_permutations(
    order: int, seed: int, *, preserve_identity: bool = True
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    table = np.empty((order, order), dtype=np.int64)
    first_random_relation = 0
    if preserve_identity:
        table[:, 0] = np.arange(order)
        first_random_relation = 1
    for relation in range(first_random_relation, order):
        table[:, relation] = rng.permutation(order)
    return table


def _frequency_preserving_corruption(
    table: np.ndarray, seed: int, corruption: float
) -> np.ndarray:
    if not 0.0 <= corruption <= 1.0:
        raise ValueError("corruption must lie in [0, 1]")
    corrupted = np.asarray(table, dtype=np.int64).copy()
    if corruption == 0.0:
        return corrupted

    rng = np.random.default_rng(seed)
    order, relation_count = corrupted.shape
    count = max(2, round(order * corruption))
    count = min(count, order)
    for relation in range(relation_count):
        rows = rng.choice(order, size=count, replace=False)
        values = corrupted[rows, relation].copy()
        shift = int(rng.integers(1, count))
        corrupted[rows, relation] = np.roll(values, shift)
    return corrupted


def _broken_cycle(order: int, seed: int, corruption: float = 0.15) -> np.ndarray:
    return _frequency_preserving_corruption(
        _cycle(order), seed=seed, corruption=corruption
    )


def make_task(
    name: str, seed: int = 0, *, corruption: float = 0.0
) -> TaskSpec:
    builders = {
        "cycle7": (
            "cycle",
            _cycle(7),
            1,
            "Seven-state cycle, analogous to weekdays.",
        ),
        "cycle12": (
            "cycle",
            _cycle(12),
            1,
            "Twelve-state cycle, analogous to months or clock positions.",
        ),
        "cycle24": (
            "cycle",
            _cycle(24),
            1,
            "Twenty-four-state cycle, analogous to hours in a day.",
        ),
        "cycle31": (
            "cycle",
            _cycle(31),
            1,
            "Larger prime cycle with no calendar semantics.",
        ),
        "cycle113": (
            "cycle",
            _cycle(113),
            1,
            "Classic modular-addition scale used as a grokking anchor.",
        ),
        "torus4": (
            "torus",
            _torus(4),
            1,
            "A 4 by 4 periodic grid with two latent coordinates.",
        ),
        "torus5": (
            "torus",
            _torus(5),
            1,
            "A 5 by 5 periodic grid with two latent coordinates.",
        ),
        "xor16": (
            "xor",
            _xor(16),
            1,
            "A four-bit hypercube under XOR.",
        ),
        "dihedral12": (
            "dihedral",
            _dihedral(12),
            1,
            "Rotations and reflections of a hexagon.",
        ),
        "path16": (
            "path",
            _path(16),
            None,
            "A bounded ordered path without cyclic closure.",
        ),
        "tree15": (
            "tree",
            _tree_lca(15),
            None,
            "A depth-three binary tree under the lowest-common-ancestor operation.",
        ),
        "broken12": (
            "broken_cycle",
            _broken_cycle(12, seed),
            1,
            "A twelve-state cycle with sparse, frequency-preserving exceptions.",
        ),
        "random16": (
            "random_permutation",
            _random_permutations(16, seed),
            None,
            "Independent random permutations with matched input and output frequencies.",
        ),
        "random31": (
            "random_permutation",
            _random_permutations(31, seed),
            None,
            "A larger frequency-matched random-permutation control.",
        ),
        "random113": (
            "random_permutation",
            _random_permutations(113, seed, preserve_identity=False),
            None,
            "Independent frequency-matched random operations at grokking scale.",
        ),
    }
    try:
        family, table, generator, description = builders[name]
    except KeyError as exc:
        choices = ", ".join(sorted(builders))
        raise ValueError(f"unknown task {name!r}; choose one of {choices}") from exc
    if corruption:
        if family != "cycle":
            raise ValueError("corruption is supported only for cycle tasks")
        table = _frequency_preserving_corruption(
            table, seed=seed, corruption=corruption
        )
        family = "broken_cycle"
        fraction = f"{corruption:.4f}".rstrip("0").rstrip(".")
        task_name = f"{name}-corrupt{fraction.replace('.', 'p')}"
        description = (
            f"{description} {corruption:.1%} of every relation column is "
            "permuted without changing its output frequencies."
        )
    else:
        task_name = name
    spec = TaskSpec(
        name=task_name,
        family=family,
        table=np.asarray(table, dtype=np.int64),
        generator=generator,
        description=description,
        corruption_fraction=float(corruption),
    )
    spec.validate()
    return spec


def available_tasks() -> tuple[str, ...]:
    return (
        "cycle7",
        "cycle12",
        "cycle24",
        "cycle31",
        "cycle113",
        "torus4",
        "torus5",
        "xor16",
        "dihedral12",
        "path16",
        "tree15",
        "broken12",
        "random16",
        "random31",
        "random113",
    )
