from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from priority_common import (
    BATCH_SIZE_BY_PRESET,
    MIN_QUALIFIED_CAUSAL_EXAMPLES,
)
from render_key60 import NORD, _style_axis
from summarize_priority_evidence import (
    EvidenceError,
    _synthetic_fixture,
    summarize,
)


MODEL_SPECS = (
    {
        "preset": "grok",
        "suite": "core",
        "width": 128,
        "depth": 1,
        "batch_size": BATCH_SIZE_BY_PRESET["grok"],
        "label": "128×1\ngrok",
    },
    {
        "preset": "micro",
        "suite": "core",
        "width": 128,
        "depth": 2,
        "batch_size": BATCH_SIZE_BY_PRESET["micro"],
        "label": "128×2\nmicro",
    },
    {
        "preset": "small",
        "suite": "scale",
        "width": 256,
        "depth": 4,
        "batch_size": BATCH_SIZE_BY_PRESET["small"],
        "label": "256×4\nsmall",
    },
    {
        "preset": "medium",
        "suite": "scale",
        "width": 512,
        "depth": 6,
        "batch_size": BATCH_SIZE_BY_PRESET["medium"],
        "label": "512×6\nmedium",
    },
    {
        "preset": "large",
        "suite": "large",
        "width": 768,
        "depth": 8,
        "batch_size": BATCH_SIZE_BY_PRESET["large"],
        "label": "768×8\nlarge",
    },
)
MODEL_BY_PRESET = {str(spec["preset"]): spec for spec in MODEL_SPECS}
SUITE_SPECS = {
    "core": {"presets": ("grok", "micro"), "run_count": 18},
    "scale": {"presets": ("small", "medium"), "run_count": 18},
    "large": {"presets": ("large",), "run_count": 9},
}
SEEDS = (0, 1, 2)
CONDITION = "corrupt15"
ENDPOINT_STEP = 30_000
TASK_ORDER = 113
CAUSAL_POLICY = {
    "minimum_qualified_examples": MIN_QUALIFIED_CAUSAL_EXAMPLES,
    "qualified_metric": "qualified_desired_accuracy",
    "fallback_metric": "desired_accuracy",
    "probability_recovery_used": False,
}
ALLOWED_CAUSAL_METRICS = {
    CAUSAL_POLICY["qualified_metric"],
    CAUSAL_POLICY["fallback_metric"],
}
COLOR = NORD["orange"]


def _valid_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one strict five-model endpoint spectrum from exact core, "
            "scale, and large suites."
        )
    )
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--scale-root", type=Path)
    parser.add_argument("--large-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate each supplied suite without rendering a partial spectrum.",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read valid JSON from {path}") from error


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError(f"cannot hash {path}") from error
    return digest.hexdigest()


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_disjoint_roots(roots: dict[str, Path]) -> None:
    resolved = {name: path.resolve() for name, path in roots.items()}
    names = list(resolved)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = resolved[left_name]
            right = resolved[right_name]
            if left == right or _within(left, right) or _within(right, left):
                raise EvidenceError(
                    f"{left_name} and {right_name} suite roots must be disjoint"
                )


def _validate_causal_metric_keys(
    run: dict[str, object],
    *,
    suite: str,
) -> None:
    causal = run.get("causal")
    sites = causal.get("sites") if isinstance(causal, dict) else None
    if not isinstance(sites, dict) or set(sites) != {"node_0", "output_final"}:
        raise EvidenceError(
            f"{suite}/{run.get('run')} does not contain both causal sites"
        )
    for site_name, site in sites.items():
        controls = site.get("controls") if isinstance(site, dict) else None
        if not isinstance(controls, dict):
            raise EvidenceError(
                f"{suite}/{run.get('run')} {site_name} lacks causal controls"
            )
        for control, record in controls.items():
            keys = (
                record.get("fold_metric_keys")
                if isinstance(record, dict)
                else None
            )
            if (
                not isinstance(keys, list)
                or len(keys) != 3
                or any(key not in ALLOWED_CAUSAL_METRICS for key in keys)
            ):
                raise EvidenceError(
                    f"{suite}/{run.get('run')} {site_name}/{control} "
                    "does not follow the bounded causal metric policy"
                )


def _validate_suite(root: Path, suite: str) -> dict[str, object]:
    spec = SUITE_SPECS[suite]
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise EvidenceError(f"{suite} results root is not a directory: {resolved}")

    summary = summarize(results_root=resolved, suite=suite)
    validation = summary.get("validation")
    if (
        summary.get("suite") != suite
        or not isinstance(validation, dict)
        or validation.get("status") != "complete"
        or int(validation.get("exact_run_count", -1)) != spec["run_count"]
        or tuple(validation.get("presets", ())) != spec["presets"]
        or tuple(validation.get("seeds", ())) != SEEDS
        or validation.get("causal_metric") != CAUSAL_POLICY
    ):
        raise EvidenceError(f"{suite} summary failed the exact protocol contract")

    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != spec["run_count"]:
        raise EvidenceError(f"{suite} summary has the wrong run count")
    expected_names = {
        str(run.get("run_name"))
        for run in runs
        if isinstance(run, dict)
    }
    config_paths = sorted(resolved.glob("*/config.json"))
    observed_names = {path.parent.name for path in config_paths}
    if (
        len(config_paths) != spec["run_count"]
        or len(expected_names) != spec["run_count"]
        or observed_names != expected_names
    ):
        raise EvidenceError(
            f"{suite} root must contain exactly its {spec['run_count']} run configs"
        )

    config_hashes: dict[str, str] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise EvidenceError(f"{suite} summary contains a non-object run")
        preset = str(run.get("preset"))
        model_spec = MODEL_BY_PRESET.get(preset)
        if model_spec is None or model_spec["suite"] != suite:
            raise EvidenceError(f"{suite} contains unexpected preset {preset}")
        config_path = resolved / str(run["run_name"]) / "config.json"
        config = _load_json(config_path)
        model = config.get("model") if isinstance(config, dict) else None
        try:
            dimensions_match = (
                isinstance(model, dict)
                and int(model["width"]) == model_spec["width"]
                and int(model["depth"]) == model_spec["depth"]
            )
        except (KeyError, TypeError, ValueError):
            dimensions_match = False
        if not dimensions_match:
            raise EvidenceError(
                f"{suite}/{run['run_name']} is not the declared "
                f"{model_spec['width']}x{model_spec['depth']} model"
            )
        _validate_causal_metric_keys(run, suite=suite)
        config_hashes[str(run["run_name"])] = _sha256(config_path)

    summary["spectrum_validation"] = {
        "exact_root": True,
        "model_dimensions": True,
        "bounded_causal_metric_keys": True,
        "config_sha256": config_hashes,
    }
    return summary


def validate_suites(roots: dict[str, Path]) -> dict[str, dict[str, object]]:
    if not roots:
        raise EvidenceError("at least one suite root is required")
    _validate_disjoint_roots(roots)
    return {
        suite: _validate_suite(root, suite)
        for suite, root in roots.items()
    }


def _corrupted_rows(
    summaries: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for suite, summary in summaries.items():
        runs = summary.get("runs")
        if not isinstance(runs, list):
            raise EvidenceError(f"{suite} summary has no runs")
        for run in runs:
            if (
                isinstance(run, dict)
                and run.get("condition") == CONDITION
            ):
                key = (str(run.get("preset")), int(run.get("seed", -1)))
                if key in indexed:
                    raise EvidenceError(f"duplicate spectrum endpoint {key}")
                indexed[key] = run

    expected = {
        (str(spec["preset"]), seed)
        for spec in MODEL_SPECS
        for seed in SEEDS
    }
    if set(indexed) != expected:
        raise EvidenceError(
            "five-model corrupted endpoint matrix is incomplete; "
            f"missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}"
        )

    rows = []
    for spec in MODEL_SPECS:
        preset = str(spec["preset"])
        for seed in SEEDS:
            run = indexed[(preset, seed)]
            try:
                behavior = run["behavior"]
                split = run["held_out_split"]
                operator = run["operator"]["endpoint"]
                node = run["causal"]["sites"]["node_0"]
                metrics = node["controls"]["learned_generator"][
                    "fold_metric_keys"
                ]
                row = {
                    "preset": preset,
                    "suite": spec["suite"],
                    "width": spec["width"],
                    "depth": spec["depth"],
                    "seed": seed,
                    "run_name": run["run_name"],
                    "endpoint_step": run["horizon"],
                    "held_out_accuracy": float(
                        behavior["final_held_out_accuracy"]
                    ),
                    "canonical_clean_rule_accuracy": float(
                        split["canonical_rule_accuracy"]
                    ),
                    "held_out_mask_sha256": str(
                        split["held_out_mask_sha256"]
                    ),
                    "operation_table_sha256": str(
                        split["operation_table_sha256"]
                    ),
                    "alias_held_out_usable_gain_kbit": (
                        float(operator["alias_held_out_usable_gain_bits"])
                        / 1000.0
                    ),
                    "node0_canonical_cycle_success": float(
                        node["canonical_cycle_median_success"]
                    ),
                    "node0_fold_metric_keys": list(metrics),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise EvidenceError(
                    f"cannot extract corrupted endpoint for {preset}/seed{seed}"
                ) from error
            if row["endpoint_step"] != ENDPOINT_STEP:
                raise EvidenceError(
                    f"{preset}/seed{seed} is not measured at {ENDPOINT_STEP}"
                )
            if (
                not 0.0 <= row["held_out_accuracy"] <= 1.0
                or not 0.0 <= row["canonical_clean_rule_accuracy"] <= 1.0
                or not 0.0 <= row["node0_canonical_cycle_success"] <= 1.0
                or not math.isfinite(row["alias_held_out_usable_gain_kbit"])
                or not _valid_sha256(row["held_out_mask_sha256"])
                or not _valid_sha256(row["operation_table_sha256"])
            ):
                raise EvidenceError(
                    f"{preset}/seed{seed} contains an invalid spectrum value"
                )
            rows.append(row)

    for seed in SEEDS:
        mask_hashes = {
            str(row["held_out_mask_sha256"])
            for row in rows
            if row["seed"] == seed
        }
        table_hashes = {
            str(row["operation_table_sha256"])
            for row in rows
            if row["seed"] == seed
        }
        if len(mask_hashes) != 1:
            raise EvidenceError(
                f"seed {seed} does not share one held-out-mask SHA-256"
            )
        if len(table_hashes) != 1:
            raise EvidenceError(
                f"seed {seed} does not share one operation-table SHA-256"
            )
        ceilings = [
            float(row["canonical_clean_rule_accuracy"])
            for row in rows
            if row["seed"] == seed
        ]
        if not np.allclose(ceilings, ceilings[0], rtol=0.0, atol=1e-12):
            raise EvidenceError(
                f"seed {seed} does not share one corrupted clean-rule ceiling"
            )
    return rows


def _series(
    rows: list[dict[str, object]],
    key: str,
) -> tuple[dict[int, list[float]], list[float]]:
    by_seed = {
        seed: [
            float(
                next(
                    row[key]
                    for row in rows
                    if row["preset"] == spec["preset"] and row["seed"] == seed
                )
            )
            for spec in MODEL_SPECS
        ]
        for seed in SEEDS
    }
    values = np.asarray([by_seed[seed] for seed in SEEDS], dtype=float)
    if values.shape != (len(SEEDS), len(MODEL_SPECS)) or not np.isfinite(
        values
    ).all():
        raise EvidenceError(f"{key} does not form a finite matched-seed matrix")
    return by_seed, np.median(values, axis=0).tolist()


def _caption() -> str:
    return (
        "All panels use the 15%-corrupted cyclic-addition runs at their measured "
        "30k endpoint. Each faint line connects the same seed across model "
        "presets, and the bold line is the pointwise seed median; straight "
        "segments connect measured endpoints only, with no smoothing or "
        "confidence interval. The clean-rule ceiling is the exact fraction of "
        "held-out corrupted-table labels that still agree with cyclic addition; "
        "the dashed line is the median of the three exact per-seed ceilings, and "
        "chance is 1/113. Alias-held-out gain compares the shared canonical-"
        "successor code with the better lookup or zero-prediction code on held-"
        "out aliases; it is not a nested state-and-alias MDL estimate. Node 0 is "
        "the input-state activation before the first transformer block. Each "
        "run's node-0 value is the median of three causal folds, using qualified "
        f"target accuracy only with at least {MIN_QUALIFIED_CAUSAL_EXAMPLES} "
        "qualified examples and absolute target accuracy otherwise. "
        "That fallback can change the evaluated population across runs, so the "
        "causal panel is descriptive rather than one homogeneous effect-size "
        "comparison; per-fold metric keys are preserved in the data. "
        "Probability-recovery ratios are excluded. The five presets change "
        "width and depth together, with batch sizes 4096, 4096, 4096, 2048, and "
        "1024 respectively, so the figure is a descriptive preset spectrum, "
        "not an isolated capacity scaling law."
    )


def _data_payload(
    rows: list[dict[str, object]],
    summaries: dict[str, dict[str, object]],
) -> dict[str, object]:
    metric_keys = (
        "held_out_accuracy",
        "alias_held_out_usable_gain_kbit",
        "node0_canonical_cycle_success",
    )
    seed_series: dict[str, object] = {}
    medians: dict[str, object] = {}
    for key in metric_keys:
        values, median = _series(rows, key)
        seed_series[key] = {str(seed): values[seed] for seed in SEEDS}
        medians[key] = median

    ceilings = {
        str(seed): float(
            next(
                row["canonical_clean_rule_accuracy"]
                for row in rows
                if row["seed"] == seed
            )
        )
        for seed in SEEDS
    }
    successor_hashes = {
        str(summary["validation"]["successor"]["sha256_little_endian_int64"])
        for summary in summaries.values()
    }
    if len(successor_hashes) != 1:
        raise EvidenceError("suite successor hashes do not agree")
    matched_inputs = {
        str(seed): {
            "held_out_mask_sha256": str(
                next(
                    row["held_out_mask_sha256"]
                    for row in rows
                    if row["seed"] == seed
                )
            ),
            "operation_table_sha256": str(
                next(
                    row["operation_table_sha256"]
                    for row in rows
                    if row["seed"] == seed
                )
            ),
        }
        for seed in SEEDS
    }
    config_hashes = {
        suite: dict(summary["spectrum_validation"]["config_sha256"])
        for suite, summary in summaries.items()
    }

    return {
        "schema_version": 1,
        "condition": CONDITION,
        "endpoint_step": ENDPOINT_STEP,
        "model_order": [dict(spec) for spec in MODEL_SPECS],
        "seed_order": list(SEEDS),
        "rows": rows,
        "seed_series": seed_series,
        "pointwise_medians": medians,
        "references": {
            "chance_accuracy": 1.0 / TASK_ORDER,
            "canonical_clean_rule_accuracy_by_seed": ceilings,
            "canonical_clean_rule_accuracy_median": float(
                np.median(list(ceilings.values()))
            ),
        },
        "provenance": {
            "suite_roots": {
                suite: str(summary["results_root"])
                for suite, summary in summaries.items()
            },
            "config_sha256_by_suite_and_run": config_hashes,
            "matched_corrupted_inputs_by_seed": matched_inputs,
        },
        "definitions": {
            "held_out_accuracy": (
                "model accuracy on unseen state-relation cells in the corrupted "
                "operation table"
            ),
            "alias_held_out_usable_gain_kbit": (
                "best lookup-or-zero code length minus shared canonical-successor "
                "code length, scored on held-out aliases, divided by 1000"
            ),
            "node0_canonical_cycle_success": (
                "median three-fold target accuracy after the preregistered "
                "canonical-cycle intervention at node 0"
            ),
            "canonical_clean_rule_accuracy": (
                "fraction of held-out corrupted-table labels equal to cyclic "
                "addition; an exact label ceiling rather than model performance"
            ),
        },
        "validation": {
            "exact_total_run_count": sum(
                int(summary["validation"]["exact_run_count"])
                for summary in summaries.values()
            ),
            "exact_corrupted_endpoint_count": len(rows),
            "suite_run_counts": {
                suite: int(summary["validation"]["exact_run_count"])
                for suite, summary in summaries.items()
            },
            "successor_sha256_little_endian_int64": successor_hashes.pop(),
            "successor_status": "preregistered",
            "causal_metric": CAUSAL_POLICY,
            "model_dimensions": "validated from every run config",
            "matched_seed_fields": (
                "split_seed=task_seed=seed and token_seed=100000+seed, "
                "validated by the suite summaries; held-out-mask and "
                "operation-table SHA-256 values match across all five presets"
            ),
        },
        "render_policy": {
            "seed_lines": "faint matched-seed spaghetti",
            "aggregate": "bold pointwise median",
            "confidence_intervals": False,
            "smoothing": False,
            "segments": "connect measured model endpoints only",
        },
        "caption": _caption(),
    }


def _save_figure(fig: plt.Figure, output: Path, caption: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    base = output / "model-spectrum"
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(
        png,
        dpi=240,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": "Five-model corrupted endpoint spectrum",
            "Description": caption,
            "Software": "geometry/render_model_spectrum.py",
        },
    )
    fig.savefig(
        pdf,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": "Five-model corrupted endpoint spectrum",
            "Subject": caption,
            "Keywords": (
                "activation geometry; matched seeds; pointwise median; "
                "causal intervention; alias-held-out code gain"
            ),
            "Creator": "geometry/render_model_spectrum.py",
        },
    )
    plt.close(fig)
    return [png, pdf]


def _render_figure(
    rows: list[dict[str, object]],
    output: Path,
) -> list[Path]:
    specs = (
        ("held-out accuracy", "accuracy", "held_out_accuracy"),
        (
            "alias-held-out usable gain",
            "gain (kbit)",
            "alias_held_out_usable_gain_kbit",
        ),
        (
            "node 0 cycle transport",
            "success",
            "node0_canonical_cycle_success",
        ),
    )
    x = np.arange(len(MODEL_SPECS), dtype=float)
    fig, axes = plt.subplots(
        1,
        len(specs),
        figsize=(10.2, 3.35),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_alpha(0)
    for column, (title, ylabel, key) in enumerate(specs):
        axis = axes[0, column]
        by_seed, median = _series(rows, key)
        for seed in SEEDS:
            values = by_seed[seed]
            axis.plot(
                x,
                values,
                color=COLOR,
                alpha=0.18,
                linewidth=0.9,
            )
            axis.scatter(
                x,
                values,
                color=COLOR,
                alpha=0.18,
                s=14,
                linewidths=0,
            )
        axis.plot(x, median, color=COLOR, linewidth=2.5)
        axis.scatter(
            x,
            median,
            color=COLOR,
            s=29,
            linewidths=0,
            zorder=3,
        )
        axis.set_title(title, fontweight="normal")
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            x,
            [str(spec["label"]) for spec in MODEL_SPECS],
        )
        axis.set_xlim(-0.18, len(MODEL_SPECS) - 0.82)
        _style_axis(axis)

        if key in {"held_out_accuracy", "node0_canonical_cycle_success"}:
            axis.set_ylim(-0.03, 1.03)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        if key == "alias_held_out_usable_gain_kbit":
            axis.axhline(
                0.0,
                color=NORD["muted"],
                alpha=0.35,
                linewidth=0.8,
            )

    ceiling = float(
        np.median(
            [
                row["canonical_clean_rule_accuracy"]
                for row in rows
                if row["preset"] == MODEL_SPECS[0]["preset"]
            ]
        )
    )
    chance = 1.0 / TASK_ORDER
    axes[0, 0].axhline(
        ceiling,
        color=NORD["muted"],
        alpha=0.6,
        linewidth=0.9,
        linestyle=(0, (4, 3)),
    )
    axes[0, 0].axhline(
        chance,
        color=NORD["muted"],
        alpha=0.45,
        linewidth=0.8,
        linestyle=(0, (1, 2)),
    )
    axes[0, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=NORD["muted"],
                alpha=0.6,
                linewidth=0.9,
                linestyle=(0, (4, 3)),
                label="median clean-rule ceiling",
            ),
            Line2D(
                [0],
                [0],
                color=NORD["muted"],
                alpha=0.45,
                linewidth=0.8,
                linestyle=(0, (1, 2)),
                label="chance",
            ),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLOR,
                alpha=0.18,
                linewidth=0.9,
                marker="o",
                markersize=3,
                label="matched seeds",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR,
                linewidth=2.5,
                marker="o",
                markersize=4,
                label="median",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    return _save_figure(fig, output, _caption())


def render(
    *,
    core_root: Path,
    scale_root: Path,
    large_root: Path,
    output: Path,
) -> dict[str, object]:
    roots = {
        "core": core_root,
        "scale": scale_root,
        "large": large_root,
    }
    summaries = validate_suites(roots)
    rows = _corrupted_rows(summaries)
    data = _data_payload(rows, summaries)
    artifacts = _render_figure(rows, output)
    data_path = output / "model-spectrum-data.json"
    caption_path = output / "model-spectrum-caption.txt"
    _atomic_json(data_path, data)
    _atomic_text(caption_path, _caption() + "\n")
    artifacts.extend((data_path, caption_path))

    manifest_path = output / "model-spectrum-manifest.json"
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "exact_run_count": 45,
        "exact_corrupted_endpoint_count": 15,
        "condition": CONDITION,
        "endpoint_step": ENDPOINT_STEP,
        "presets": [spec["preset"] for spec in MODEL_SPECS],
        "models": [dict(spec) for spec in MODEL_SPECS],
        "seeds": list(SEEDS),
        "suite_roots": {
            suite: str(path.expanduser().resolve())
            for suite, path in roots.items()
        },
        "causal_metric": CAUSAL_POLICY,
        "render_policy": data["render_policy"],
        "references": data["references"],
        "input_provenance": data["provenance"],
        "artifacts": [
            str(path)
            for path in (*artifacts, manifest_path)
        ],
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _prepare_synthetic_spectrum(root: Path) -> dict[str, Path]:
    roots = {suite: root / suite for suite in SUITE_SPECS}
    canonical = (
        np.arange(TASK_ORDER, dtype=np.int64)[:, None]
        + np.arange(TASK_ORDER, dtype=np.int64)[None, :]
    ) % TASK_ORDER
    for suite, suite_root in roots.items():
        suite_root.mkdir(parents=True)
        _synthetic_fixture(suite_root, suite=suite)
        for config_path in suite_root.glob("*/config.json"):
            config = _load_json(config_path)
            if not isinstance(config, dict):
                raise AssertionError("synthetic config is not an object")
            preset = str(config["preset"])
            config["model"]["width"] = MODEL_BY_PRESET[preset]["width"]
            config["model"]["depth"] = MODEL_BY_PRESET[preset]["depth"]
            run_dir = config_path.parent
            if (
                str(config["task"]) == "cycle113"
                and math.isclose(
                    float(config["task_corruption_fraction"]),
                    0.15,
                )
            ):
                train_mask = np.asarray(
                    np.load(run_dir / "train_mask.npy"),
                    dtype=bool,
                )
                table = canonical.copy()
                held_out = np.argwhere(~train_mask)
                changed = held_out[
                    : int(round(0.15 * held_out.shape[0]))
                ]
                table[changed[:, 0], changed[:, 1]] = (
                    table[changed[:, 0], changed[:, 1]] + 1
                ) % TASK_ORDER
                np.save(run_dir / "operation_table.npy", table)
                config["task_table_sha256"] = hashlib.sha256(
                    table.tobytes()
                ).hexdigest()
            _atomic_json(config_path, config)
    return roots


def self_test(output: Path | None = None) -> None:
    context = (
        tempfile.TemporaryDirectory(prefix="model-spectrum-")
        if output is None
        else None
    )
    root = Path(context.name) if context is not None else output
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    roots = _prepare_synthetic_spectrum(root / "inputs")
    figures = root / "figures"
    manifest = render(
        core_root=roots["core"],
        scale_root=roots["scale"],
        large_root=roots["large"],
        output=figures,
    )
    expected = {
        "model-spectrum.png",
        "model-spectrum.pdf",
        "model-spectrum-data.json",
        "model-spectrum-caption.txt",
        "model-spectrum-manifest.json",
    }
    if (
        manifest.get("status") != "complete"
        or manifest.get("exact_run_count") != 45
        or manifest.get("exact_corrupted_endpoint_count") != 15
        or expected != {path.name for path in figures.iterdir()}
    ):
        raise AssertionError("model-spectrum renderer missed an exact artifact")
    if manifest.get("causal_metric") != CAUSAL_POLICY:
        raise AssertionError("model-spectrum renderer changed the causal policy")
    provenance = manifest.get("input_provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance.get("config_sha256_by_suite_and_run", {}))
        != set(SUITE_SPECS)
        or set(provenance.get("matched_corrupted_inputs_by_seed", {}))
        != {str(seed) for seed in SEEDS}
    ):
        raise AssertionError("model-spectrum manifest lacks input provenance")
    policy = manifest.get("render_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("aggregate") != "bold pointwise median"
        or policy.get("confidence_intervals") is not False
        or policy.get("smoothing") is not False
    ):
        raise AssertionError("model-spectrum render policy is not explicit")
    data = _load_json(figures / "model-spectrum-data.json")
    if (
        not isinstance(data, dict)
        or len(data.get("rows", ())) != 15
        or data.get("seed_order") != list(SEEDS)
        or [
            model.get("preset")
            for model in data.get("model_order", ())
            if isinstance(model, dict)
        ]
        != [spec["preset"] for spec in MODEL_SPECS]
        or not math.isclose(
            float(data["references"]["chance_accuracy"]),
            1.0 / TASK_ORDER,
        )
        or any(
            float(row["node0_canonical_cycle_success"]) > 1.0
            for row in data["rows"]
        )
    ):
        raise AssertionError("model-spectrum data obscures ordering or references")

    summaries = validate_suites(roots)
    scale_corrupt = next(
        run
        for run in summaries["scale"]["runs"]
        if run["condition"] == CONDITION and run["seed"] == 0
    )
    original_mask_hash = scale_corrupt["held_out_split"][
        "held_out_mask_sha256"
    ]
    scale_corrupt["held_out_split"]["held_out_mask_sha256"] = "0" * 64
    try:
        _corrupted_rows(summaries)
    except EvidenceError as error:
        if "held-out-mask SHA-256" not in str(error):
            raise
    else:
        raise AssertionError("model-spectrum accepted mismatched held-out masks")
    finally:
        scale_corrupt["held_out_split"][
            "held_out_mask_sha256"
        ] = original_mask_hash

    original_table_hash = scale_corrupt["held_out_split"][
        "operation_table_sha256"
    ]
    scale_corrupt["held_out_split"]["operation_table_sha256"] = "f" * 64
    try:
        _corrupted_rows(summaries)
    except EvidenceError as error:
        if "operation-table SHA-256" not in str(error):
            raise
    else:
        raise AssertionError(
            "model-spectrum accepted mismatched operation tables"
        )
    finally:
        scale_corrupt["held_out_split"][
            "operation_table_sha256"
        ] = original_table_hash

    scale_config = next(roots["scale"].glob("*/config.json"))
    original = scale_config.read_text()
    tampered = json.loads(original)
    tampered["model"]["width"] = 999
    _atomic_json(scale_config, tampered)
    try:
        _validate_suite(roots["scale"], "scale")
    except EvidenceError as error:
        if "declared" not in str(error):
            raise
    else:
        raise AssertionError("model-spectrum validator accepted a wrong width")
    finally:
        scale_config.write_text(original)

    from PIL import Image

    with Image.open(figures / "model-spectrum.png") as image:
        if (
            image.getbbox() is None
            or image.width < 1_500
            or "same seed" not in image.info.get("Description", "")
        ):
            raise AssertionError("model-spectrum PNG failed visual metadata checks")
    print(f"self-test passed: {figures}")
    if context is not None:
        context.cleanup()


def _provided_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        suite: path
        for suite, path in (
            ("core", args.core_root),
            ("scale", args.scale_root),
            ("large", args.large_root),
        )
        if path is not None
    }


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.self_test_output)
        return
    if args.validate_only:
        summaries = validate_suites(_provided_roots(args))
        print(
            json.dumps(
                {
                    suite: {
                        "status": summary["validation"]["status"],
                        "exact_run_count": summary["validation"][
                            "exact_run_count"
                        ],
                        "presets": summary["validation"]["presets"],
                        "causal_metric": summary["validation"]["causal_metric"],
                    }
                    for suite, summary in summaries.items()
                },
                indent=2,
            )
        )
        return
    if (
        args.core_root is None
        or args.scale_root is None
        or args.large_root is None
        or args.output is None
    ):
        raise ValueError(
            "--core-root, --scale-root, --large-root, and --output are required"
        )
    render(
        core_root=args.core_root,
        scale_root=args.scale_root,
        large_root=args.large_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
