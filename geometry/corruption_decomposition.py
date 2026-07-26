from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from geogen.model import GeometryTransformer, ModelConfig
from geogen.tasks import make_task
from train import TokenLayout, build_tokens, make_split


CHECKPOINT_STEP = re.compile(r"-(\d+)\.pt$")
GROUPS = (
    "train_all",
    "train_clean_consistent",
    "train_corrupted",
    "test_all",
    "test_clean_consistent",
    "test_corrupted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose a corrupted-cycle run into clean-consistent and "
            "exception cells across saved checkpoints."
        )
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        help=(
            "Repeatable glob relative to the run directory. By default both "
            "checkpoint-*.pt and weights-*.pt are considered."
        ),
    )
    parser.add_argument(
        "--steps",
        help=(
            "Optional comma-separated exact steps or inclusive ranges, "
            "for example 1000,5000-20000."
        ),
    )
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    parser.add_argument(
        "--step-every",
        type=int,
        help="Keep checkpoints whose step is divisible by this value.",
    )
    parser.add_argument(
        "--surface-mode",
        choices=("balanced", "canonical", "exhaustive"),
        default="balanced",
        help=(
            "balanced micro-averages a deterministic design in which every "
            "context and alias pair occurs equally often; canonical uses only "
            "context/aliases zero; exhaustive uses the full Cartesian product."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_step_selection(value: str | None) -> tuple[set[int], list[tuple[int, int]]]:
    exact: set[int] = set()
    ranges: list[tuple[int, int]] = []
    if value is None:
        return exact, ranges
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" not in item:
            exact.add(int(item))
            continue
        start_text, stop_text = item.split("-", maxsplit=1)
        start, stop = int(start_text), int(stop_text)
        if start > stop:
            raise ValueError(f"descending step range {item!r}")
        ranges.append((start, stop))
    if any(step < 0 for step in exact) or any(start < 0 for start, _ in ranges):
        raise ValueError("checkpoint steps cannot be negative")
    if not exact and not ranges:
        raise ValueError("--steps selected no steps")
    return exact, ranges


def step_selected(
    step: int,
    *,
    exact: set[int],
    ranges: list[tuple[int, int]],
    min_step: int | None,
    max_step: int | None,
    step_every: int | None,
    has_explicit_selection: bool,
) -> bool:
    if has_explicit_selection and not (
        step in exact or any(start <= step <= stop for start, stop in ranges)
    ):
        return False
    if min_step is not None and step < min_step:
        return False
    if max_step is not None and step > max_step:
        return False
    if step_every is not None and step % step_every:
        return False
    return True


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_STEP.search(path.name)
    if match is None:
        raise ValueError(f"cannot infer a step from checkpoint name {path.name!r}")
    return int(match.group(1))


def discover_checkpoints(
    run_dir: Path,
    *,
    patterns: list[str] | None,
    steps: str | None,
    min_step: int | None,
    max_step: int | None,
    step_every: int | None,
) -> list[tuple[int, Path]]:
    if step_every is not None and step_every < 1:
        raise ValueError("--step-every must be positive")
    exact, ranges = parse_step_selection(steps)
    has_explicit_selection = steps is not None
    candidates: dict[int, tuple[int, Path]] = {}
    for pattern in patterns or ["checkpoint-*.pt", "weights-*.pt"]:
        for path in sorted(run_dir.glob(pattern)):
            step = checkpoint_step(path)
            if not step_selected(
                step,
                exact=exact,
                ranges=ranges,
                min_step=min_step,
                max_step=max_step,
                step_every=step_every,
                has_explicit_selection=has_explicit_selection,
            ):
                continue
            priority = 1 if path.name.startswith("checkpoint-") else 0
            previous = candidates.get(step)
            if previous is None or priority > previous[0]:
                candidates[step] = (priority, path)
    return [
        (step, priority_path[1])
        for step, priority_path in sorted(candidates.items())
    ]


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{path} is not a recognized geometry checkpoint")
    checkpoint_format = str(
        payload.get(
            "format",
            "training-checkpoint-v1"
            if "optimizer" in payload
            else "weights-only-legacy",
        )
    )
    return payload["model"], checkpoint_format


def load_layout(
    run_dir: Path,
    *,
    order: int,
    aliases: int,
    contexts: int,
    token_seed: int,
    config_has_token_seed: bool,
    device: torch.device,
) -> TokenLayout:
    layout = TokenLayout(
        order,
        aliases,
        contexts,
        seed=token_seed,
        device=device,
    )
    layout_path = run_dir / "token_layout.npz"
    if layout_path.exists():
        with np.load(layout_path) as saved:
            node = np.asarray(saved["node"], dtype=np.int64)
            relation = np.asarray(saved["relation"], dtype=np.int64)
        expected = (aliases, order)
        if node.shape != expected or relation.shape != expected:
            raise ValueError(
                f"token layout has shapes {node.shape}/{relation.shape}, "
                f"expected {expected}"
            )
        layout.node_tokens = torch.as_tensor(node, dtype=torch.long, device=device)
        layout.relation_tokens = torch.as_tensor(
            relation, dtype=torch.long, device=device
        )
    elif not config_has_token_seed:
        surface_count = order * aliases
        layout.node_tokens = (
            torch.arange(surface_count, device=device)
            .reshape(aliases, order)
            .add(layout.node_base)
        )
        layout.relation_tokens = (
            torch.arange(surface_count, device=device)
            .reshape(aliases, order)
            .add(layout.relation_base)
        )
    return layout


def evaluation_surfaces(
    *,
    contexts: int,
    aliases: int,
    mode: str,
) -> list[tuple[int, int, int]]:
    if contexts < 1 or aliases < 1:
        raise ValueError("contexts and aliases must be positive")
    if mode == "canonical":
        return [(0, 0, 0)]
    if mode == "exhaustive":
        return [
            (context, left_alias, right_alias)
            for context in range(contexts)
            for left_alias in range(aliases)
            for right_alias in range(aliases)
        ]
    alias_pairs = aliases * aliases
    count = math.lcm(contexts, alias_pairs)
    return [
        (
            index % contexts,
            (index % alias_pairs) % aliases,
            (index % alias_pairs) // aliases,
        )
        for index in range(count)
    ]


def clean_cycle_reference(
    config: dict[str, object],
    operation_table: np.ndarray,
) -> tuple[np.ndarray | None, str, str]:
    task_name = str(config.get("task", ""))
    family = str(config.get("task_family", ""))
    if operation_table.ndim != 2:
        raise ValueError("operation table must be two-dimensional")
    order, relations = operation_table.shape
    if relations != order:
        return (
            None,
            "unavailable",
            "The operation table is not square, so modular addition is undefined.",
        )
    left = np.arange(order)[:, None]
    right = np.arange(order)[None, :]
    reference = (left + right) % order
    if task_name.startswith("cycle") or family in {"cycle", "broken_cycle"}:
        return (
            reference.astype(np.int64),
            "underlying_clean_cycle",
            (
                "The reference is the uncorrupted modular-addition table; "
                "cells whose saved target differs are generated exceptions."
            ),
        )
    if task_name == "random113":
        return (
            reference.astype(np.int64),
            "diagnostic_cycle_control",
            (
                "random113 has no underlying clean rule. Modular addition is "
                "used only as a matched diagnostic baseline, so "
                "'corrupted' means cycle-inconsistent rather than a generated "
                "exception."
            ),
        )
    return (
        None,
        "unavailable",
        "This task is not a cycle corruption condition or random113 control.",
    )


def subgroup_masks(
    train_mask: np.ndarray,
    consistent_mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    test = ~train
    if consistent_mask is None:
        consistent = np.zeros_like(train)
        corrupted = np.zeros_like(train)
    else:
        consistent = np.asarray(consistent_mask, dtype=bool).reshape(-1)
        corrupted = ~consistent
    return {
        "train_all": train,
        "train_clean_consistent": train & consistent,
        "train_corrupted": train & corrupted,
        "test_all": test,
        "test_clean_consistent": test & consistent,
        "test_corrupted": test & corrupted,
    }


@dataclass
class Accumulator:
    cell_count: int
    sample_count: int = 0
    actual_loss_sum: float = 0.0
    actual_correct: int = 0
    clean_loss_sum: float = 0.0
    clean_correct: int = 0

    def result(self, *, has_clean_reference: bool) -> dict[str, object]:
        if self.sample_count == 0:
            return {
                "cell_count": self.cell_count,
                "surface_sample_count": 0,
                "actual_accuracy": None,
                "actual_loss": None,
                "clean_rule_accuracy": None,
                "clean_rule_loss": None,
            }
        return {
            "cell_count": self.cell_count,
            "surface_sample_count": self.sample_count,
            "actual_accuracy": self.actual_correct / self.sample_count,
            "actual_loss": self.actual_loss_sum / self.sample_count,
            "clean_rule_accuracy": (
                self.clean_correct / self.sample_count
                if has_clean_reference
                else None
            ),
            "clean_rule_loss": (
                self.clean_loss_sum / self.sample_count
                if has_clean_reference
                else None
            ),
        }


def systematic_ceilings(
    *,
    train_mask: np.ndarray,
    consistent_mask: np.ndarray | None,
) -> dict[str, dict[str, object]] | None:
    if consistent_mask is None:
        return None
    train = np.asarray(train_mask, dtype=bool)
    test = ~train
    consistent = np.asarray(consistent_mask, dtype=bool)
    partitions = {"train": train, "test": test, "all": np.ones_like(train)}
    result: dict[str, dict[str, object]] = {}
    for name, mask in partitions.items():
        cell_count = int(mask.sum())
        consistent_count = int((mask & consistent).sum())
        pure_clean = consistent_count / cell_count
        if name == "train":
            memorize_then_rule = 1.0
        elif name == "test":
            memorize_then_rule = pure_clean
        else:
            memorize_then_rule = (
                int(train.sum()) + int((test & consistent).sum())
            ) / cell_count
        result[name] = {
            "cell_count": cell_count,
            "clean_consistent_count": consistent_count,
            "corrupted_count": cell_count - consistent_count,
            "pure_clean_rule_accuracy": pure_clean,
            "memorize_train_exceptions_then_clean_rule_accuracy": (
                memorize_then_rule
            ),
        }
    return result


@torch.inference_mode()
def evaluate_checkpoint(
    model: GeometryTransformer,
    *,
    layout: TokenLayout,
    operation_table: np.ndarray,
    clean_reference: np.ndarray | None,
    masks: dict[str, np.ndarray],
    surfaces: list[tuple[int, int, int]],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    model.eval()
    order = operation_table.shape[0]
    left_grid, right_grid = np.meshgrid(
        np.arange(order), np.arange(order), indexing="ij"
    )
    left = torch.as_tensor(left_grid.reshape(-1), dtype=torch.long, device=device)
    right = torch.as_tensor(
        right_grid.reshape(-1), dtype=torch.long, device=device
    )
    actual = torch.as_tensor(
        operation_table.reshape(-1), dtype=torch.long, device=device
    )
    clean = (
        torch.as_tensor(
            clean_reference.reshape(-1), dtype=torch.long, device=device
        )
        if clean_reference is not None
        else None
    )
    masks_device = {
        name: torch.as_tensor(mask, dtype=torch.bool, device=device)
        for name, mask in masks.items()
    }
    accumulators = {
        name: Accumulator(cell_count=int(mask.sum()))
        for name, mask in masks.items()
    }

    for context, left_alias, right_alias in surfaces:
        for start in range(0, left.numel(), batch_size):
            stop = min(start + batch_size, left.numel())
            left_batch = left[start:stop]
            right_batch = right[start:stop]
            size = stop - start
            tokens = build_tokens(
                layout,
                left_batch,
                right_batch,
                contexts=torch.full(
                    (size,), context, dtype=torch.long, device=device
                ),
                left_alias=torch.full(
                    (size,), left_alias, dtype=torch.long, device=device
                ),
                right_alias=torch.full(
                    (size,), right_alias, dtype=torch.long, device=device
                ),
            )
            logits = model(tokens)
            actual_batch = actual[start:stop]
            predictions = logits.argmax(dim=-1)
            actual_loss = F.cross_entropy(
                logits, actual_batch, reduction="none"
            )
            if clean is None:
                clean_batch = None
                clean_loss = None
            else:
                clean_batch = clean[start:stop]
                clean_loss = F.cross_entropy(
                    logits, clean_batch, reduction="none"
                )
            for name in GROUPS:
                selected = masks_device[name][start:stop]
                count = int(selected.sum())
                if count == 0:
                    continue
                accumulator = accumulators[name]
                accumulator.sample_count += count
                accumulator.actual_loss_sum += float(actual_loss[selected].sum())
                accumulator.actual_correct += int(
                    (predictions[selected] == actual_batch[selected]).sum()
                )
                if clean_batch is not None and clean_loss is not None:
                    accumulator.clean_loss_sum += float(
                        clean_loss[selected].sum()
                    )
                    accumulator.clean_correct += int(
                        (predictions[selected] == clean_batch[selected]).sum()
                    )
    return {
        name: accumulator.result(
            has_clean_reference=clean_reference is not None
        )
        for name, accumulator in accumulators.items()
    }


def analyze_run(
    *,
    run_dir: Path,
    checkpoint_patterns: list[str] | None,
    steps: str | None,
    min_step: int | None,
    max_step: int | None,
    step_every: int | None,
    surface_mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    config = json.loads((run_dir / "config.json").read_text())
    operation_table = np.asarray(
        np.load(run_dir / "operation_table.npy"), dtype=np.int64
    )
    train_mask = np.asarray(
        np.load(run_dir / "train_mask.npy"), dtype=bool
    )
    if operation_table.shape != train_mask.shape:
        raise ValueError(
            "operation_table.npy and train_mask.npy have different shapes"
        )
    if operation_table.min() < 0 or operation_table.max() >= operation_table.shape[0]:
        raise ValueError("operation table contains an invalid output class")

    order = operation_table.shape[0]
    aliases = int(config["aliases"])
    contexts = int(config["contexts"])
    token_seed = int(config.get("token_seed", config["seed"]))
    layout = load_layout(
        run_dir,
        order=order,
        aliases=aliases,
        contexts=contexts,
        token_seed=token_seed,
        config_has_token_seed="token_seed" in config,
        device=device,
    )
    model = GeometryTransformer(
        vocab_size=layout.vocab_size,
        output_classes=order,
        config=ModelConfig(**config["model"]),
    ).to(device)
    clean_reference, reference_mode, reference_note = clean_cycle_reference(
        config, operation_table
    )
    consistent = (
        operation_table == clean_reference
        if clean_reference is not None
        else None
    )
    masks = subgroup_masks(train_mask, consistent)
    ceilings = systematic_ceilings(
        train_mask=train_mask,
        consistent_mask=consistent,
    )
    surfaces = evaluation_surfaces(
        contexts=contexts,
        aliases=aliases,
        mode=surface_mode,
    )
    checkpoints = discover_checkpoints(
        run_dir,
        patterns=checkpoint_patterns,
        steps=steps,
        min_step=min_step,
        max_step=max_step,
        step_every=step_every,
    )
    if not checkpoints:
        raise FileNotFoundError("no matching checkpoints found")

    records: list[dict[str, object]] = []
    for step, checkpoint_path in checkpoints:
        state, checkpoint_format = load_checkpoint(checkpoint_path)
        model.load_state_dict(state)
        del state
        groups = evaluate_checkpoint(
            model,
            layout=layout,
            operation_table=operation_table,
            clean_reference=clean_reference,
            masks=masks,
            surfaces=surfaces,
            batch_size=batch_size,
            device=device,
        )
        records.append(
            {
                "step": step,
                "checkpoint": checkpoint_path.name,
                "checkpoint_format": checkpoint_format,
                "groups": groups,
            }
        )
        test_all = groups["test_all"]
        test_clean = groups["test_clean_consistent"]
        test_corrupted = groups["test_corrupted"]
        print(
            f"{config.get('run_name', run_dir.name)} step={step}: "
            f"test={test_all['actual_accuracy']!s} "
            f"clean={test_clean['actual_accuracy']!s} "
            f"exceptions={test_corrupted['actual_accuracy']!s}",
            flush=True,
        )

    surface_array = np.asarray(surfaces, dtype=np.int64)
    metadata = {
        "schema_version": 1,
        "run_name": config.get("run_name", run_dir.name),
        "task": config.get("task"),
        "task_family": config.get("task_family"),
        "task_corruption_fraction": config.get(
            "task_corruption_fraction", config.get("corruption")
        ),
        "order": order,
        "aliases": aliases,
        "contexts": contexts,
        "surface_mode": surface_mode,
        "surface_count": len(surfaces),
        "surface_design": (
            "Metrics are micro-averaged over deterministic surface "
            "realizations. balanced uses lcm(contexts, aliases^2) "
            "realizations, covering each context and each ordered alias pair "
            "equally often; canonical uses (0,0,0); exhaustive uses their "
            "full Cartesian product."
        ),
        "surface_tuples_context_left_alias_right_alias": surfaces,
        "surface_tuples_sha256": hashlib.sha256(
            surface_array.tobytes()
        ).hexdigest(),
        "reference_mode": reference_mode,
        "reference_note": reference_note,
        "actual_metrics": (
            "Cross-entropy and top-1 accuracy against the saved operation "
            "table, separately for train/test and clean-consistent/exception "
            "cells."
        ),
        "clean_rule_metrics": (
            "Cross-entropy and top-1 accuracy against uncorrupted modular "
            "addition. For random113 this is a diagnostic baseline, not a "
            "ground-truth latent rule."
        ),
        "systematic_ceiling_definition": (
            "pure_clean_rule_accuracy is the exact fraction of saved labels "
            "matched by modular addition. "
            "memorize_train_exceptions_then_clean_rule_accuracy assumes every "
            "training exception is memorized and the clean rule is used on "
            "all held-out cells."
        ),
        "checkpoint_count": len(records),
    }
    return {
        "metadata": metadata,
        "systematic_ceilings": ceilings,
        "records": records,
    }


def write_outputs(prefix: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(json_safe(payload), indent=2) + "\n")

    metadata = payload["metadata"]
    ceilings = payload["systematic_ceilings"]
    rows: list[dict[str, object]] = []
    for record in payload["records"]:
        for group_name, group in record["groups"].items():
            split, subset = group_name.split("_", maxsplit=1)
            ceiling = ceilings.get(split) if ceilings is not None else None
            rows.append(
                {
                    "run_name": metadata["run_name"],
                    "task": metadata["task"],
                    "task_family": metadata["task_family"],
                    "task_corruption_fraction": metadata[
                        "task_corruption_fraction"
                    ],
                    "reference_mode": metadata["reference_mode"],
                    "surface_mode": metadata["surface_mode"],
                    "surface_count": metadata["surface_count"],
                    "step": record["step"],
                    "checkpoint": record["checkpoint"],
                    "checkpoint_format": record["checkpoint_format"],
                    "split": split,
                    "subset": subset,
                    **group,
                    "pure_clean_rule_ceiling": (
                        ceiling["pure_clean_rule_accuracy"]
                        if ceiling is not None
                        else None
                    ),
                    "memorize_then_rule_ceiling": (
                        ceiling[
                            "memorize_train_exceptions_then_clean_rule_accuracy"
                        ]
                        if ceiling is not None
                        else None
                    ),
                }
            )
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    print(f"wrote {json_path} and {csv_path}")
    return json_path, csv_path


def run_self_test() -> None:
    torch.manual_seed(3)
    np.random.seed(3)
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "cycle7-smoke"
        run_dir.mkdir()
        task = make_task("cycle7", seed=11, corruption=0.30)
        train_mask = make_split(task.order, 0.4, seed=13)
        config_model = ModelConfig(width=16, depth=1, heads=4, mlp_ratio=2)
        layout = TokenLayout(
            task.order,
            aliases=2,
            contexts=3,
            seed=17,
            device=torch.device("cpu"),
        )
        model = GeometryTransformer(
            vocab_size=layout.vocab_size,
            output_classes=task.order,
            config=config_model,
        )
        config = {
            "run_name": "cycle7-corrupt-smoke",
            "task": "cycle7",
            "task_family": "broken_cycle",
            "task_corruption_fraction": 0.30,
            "seed": 3,
            "token_seed": 17,
            "aliases": 2,
            "contexts": 3,
            "model": config_model.__dict__,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        np.save(run_dir / "operation_table.npy", task.table)
        np.save(run_dir / "train_mask.npy", train_mask)
        layout.save(run_dir / "token_layout.npz")
        torch.save(
            {
                "step": 1,
                "model": model.state_dict(),
                "optimizer": {},
            },
            run_dir / "checkpoint-000001.pt",
        )
        half_state = {
            name: (
                value.half() if value.is_floating_point() else value
            )
            for name, value in model.state_dict().items()
        }
        torch.save(
            {
                "format": "weights-only-v1",
                "step": 2,
                "model": half_state,
            },
            run_dir / "weights-000002.pt",
        )
        payload = analyze_run(
            run_dir=run_dir,
            checkpoint_patterns=None,
            steps=None,
            min_step=None,
            max_step=None,
            step_every=None,
            surface_mode="balanced",
            batch_size=32,
            device=torch.device("cpu"),
        )
        if len(payload["records"]) != 2:
            raise AssertionError("did not load both training and dense checkpoints")
        if payload["metadata"]["surface_count"] != 12:
            raise AssertionError("balanced surface design has the wrong size")
        ceilings = payload["systematic_ceilings"]
        if ceilings is None:
            raise AssertionError("cycle reference was not constructed")
        train_cells = int(train_mask.sum())
        test_cells = int((~train_mask).sum())
        for record in payload["records"]:
            train_group = record["groups"]["train_all"]
            test_group = record["groups"]["test_all"]
            if train_group["cell_count"] != train_cells:
                raise AssertionError("train cell count is wrong")
            if test_group["cell_count"] != test_cells:
                raise AssertionError("test cell count is wrong")
            if train_group["surface_sample_count"] != train_cells * 12:
                raise AssertionError("surface micro-average count is wrong")
        if (
            ceilings["all"]["clean_consistent_count"]
            + ceilings["all"]["corrupted_count"]
            != task.order**2
        ):
            raise AssertionError("corruption partition does not cover the table")
        random_reference, random_mode, _ = clean_cycle_reference(
            {
                "task": "random113",
                "task_family": "random_permutation",
            },
            np.zeros((7, 7), dtype=np.int64),
        )
        if random_reference is None or random_mode != "diagnostic_cycle_control":
            raise AssertionError("random113 diagnostic handling failed")
        json_path, csv_path = write_outputs(
            run_dir / "decomposition-smoke", payload
        )
        if not json.loads(json_path.read_text())["records"]:
            raise AssertionError("JSON output is empty")
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2 * len(GROUPS):
            raise AssertionError("CSV output has the wrong number of rows")
    print(
        "self-test passed: corruption partition, dense checkpoints, "
        "surfaces, outputs"
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --self-test is used")
    device = resolve_device(args.device)
    payload = analyze_run(
        run_dir=args.run_dir,
        checkpoint_patterns=args.checkpoint_glob,
        steps=args.steps,
        min_step=args.min_step,
        max_step=args.max_step,
        step_every=args.step_every,
        surface_mode=args.surface_mode,
        batch_size=args.batch_size,
        device=device,
    )
    prefix = args.output_prefix or (args.run_dir / "corruption_decomposition")
    write_outputs(prefix, payload)


if __name__ == "__main__":
    main()
