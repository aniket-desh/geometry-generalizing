#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ohope.config import (
    ABLATION_SAMPLE_SEEDS,
    CALIBRATION_SEEDS,
    MODEL_SPECS,
    RANDOM_PRUNING_SEEDS,
)
from ohope.data import prepare_corpus
from ohope.metrics import (
    collect_activation_rms,
    collect_evaluation_cache,
    compute_fisher_scores,
    compute_geometry,
    evaluate_pruning_curves,
    evaluate_single_neuron_ablations,
    extract_outlier_records,
    save_scores,
    select_ablation_neurons,
    summarize_ablation_correlations,
    write_json,
)
from ohope.modeling import FinalMLPAdapter
from ohope.plotting import render_model_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one observable-HOPE experiment.")
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--revision")
    parser.add_argument("--calibration-tokens", type=int, default=50_000)
    parser.add_argument("--evaluation-tokens", type=int, default=100_000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--pruning-token-batch-size", type=int, default=512)
    parser.add_argument("--ablation-sample-size", type=int, default=192)
    parser.add_argument("--ablation-tokens", type=int, default=10_000)
    parser.add_argument("--ablation-token-batch-size", type=int, default=32)
    parser.add_argument("--ablation-neuron-batch-size", type=int, default=24)
    parser.add_argument("--fisher-tokens", type=int, default=512)
    parser.add_argument("--fisher-token-batch-size", type=int, default=8)
    parser.add_argument("--fisher-neuron-batch-size", type=int, default=8)
    parser.add_argument("--skip-fisher", action="store_true")
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def stage(name: str) -> float:
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {name}", flush=True)
    return time.perf_counter()


def finish(name: str, started: float) -> None:
    print(f"[done] {name}: {time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.calibration_tokens = 512
        args.evaluation_tokens = 1_024
        args.sequence_length = 128
        args.capture_batch_size = 2
        args.pruning_token_batch_size = 64
        args.ablation_sample_size = 32
        args.ablation_tokens = 256
        args.ablation_token_batch_size = 16
        args.ablation_neuron_batch_size = 4
        args.fisher_tokens = 16
        args.fisher_token_batch_size = 4
        args.fisher_neuron_batch_size = 2

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("these experiments require a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = True

    spec = MODEL_SPECS[args.model]
    model_dir = args.output_root / spec.slug
    corpus_dir = model_dir / "corpus"
    model_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(model_dir / "run_config.json", run_config)

    started = stage(f"load {spec.hub_id}")
    adapter = FinalMLPAdapter.load(
        spec,
        device=torch.device("cuda"),
        cache_dir=str(args.cache_dir),
        revision=args.revision,
    )
    finish("model load", started)
    write_json(model_dir / "model_metadata.json", adapter.metadata())

    try:
        started = stage("prepare disjoint calibration and evaluation streams")
        corpus_paths = prepare_corpus(
            adapter.tokenizer,
            output_dir=corpus_dir,
            cache_dir=str(args.cache_dir),
            calibration_tokens=args.calibration_tokens,
            evaluation_tokens=args.evaluation_tokens,
            calibration_seeds=CALIBRATION_SEEDS,
            sequence_length=args.sequence_length,
        )
        finish("corpus", started)

        geometry_path = model_dir / "geometry.npz"
        started = stage("centered unembedding geometry")
        if geometry_path.exists() and not args.force:
            geometry = load_npz(geometry_path)
            print("  resumed geometry.npz", flush=True)
        else:
            geometry = compute_geometry(
                adapter,
                chunk_size=64,
                output_path=geometry_path,
            )
        finish("geometry", started)

        score_sets: dict[int, dict[str, np.ndarray]] = {}
        started = stage("calibration scores")
        for seed in CALIBRATION_SEEDS:
            score_path = model_dir / f"scores_seed_{seed}.npz"
            if score_path.exists() and not args.force:
                score_sets[seed] = load_npz(score_path)
                print(f"  resumed seed {seed}", flush=True)
                continue
            blocks = torch.load(
                corpus_paths[f"calibration_{seed}"],
                map_location="cpu",
            )
            activation_rms = collect_activation_rms(
                adapter,
                blocks,
                prediction_tokens=args.calibration_tokens,
                batch_size=args.capture_batch_size,
            )
            score_sets[seed] = save_scores(
                activation_rms=activation_rms,
                geometry=geometry,
                output_path=score_path,
            )
        finish("calibration", started)

        evaluation_cache_path = model_dir / "evaluation_cache.pt"
        started = stage("cache final-layer evaluation states")
        if evaluation_cache_path.exists() and not args.force:
            cache = torch.load(evaluation_cache_path, map_location="cpu")
            print("  resumed evaluation_cache.pt", flush=True)
        else:
            evaluation_blocks = torch.load(
                corpus_paths["evaluation"],
                map_location="cpu",
            )
            cache = collect_evaluation_cache(
                adapter,
                evaluation_blocks,
                prediction_tokens=args.evaluation_tokens,
                batch_size=args.capture_batch_size,
                output_path=evaluation_cache_path,
            )
        finish("evaluation cache", started)

        if not args.skip_fisher:
            fisher_path = model_dir / "fisher_scores.npz"
            started = stage("LayerNorm Fisher pullback")
            if fisher_path.exists() and not args.force:
                fisher = load_npz(fisher_path)["fisher"]
                print("  resumed fisher_scores.npz", flush=True)
            else:
                fisher = compute_fisher_scores(
                    adapter,
                    cache,
                    max_tokens=args.fisher_tokens,
                    token_batch_size=args.fisher_token_batch_size,
                    neuron_batch_size=args.fisher_neuron_batch_size,
                    output_path=fisher_path,
                )
            for scores in score_sets.values():
                scores["fisher"] = fisher
            finish("Fisher", started)

        pruning_path = model_dir / "pruning_curves.csv"
        started = stage("structured pruning curves")
        if pruning_path.exists() and not args.force:
            print("  resumed pruning_curves.csv", flush=True)
        else:
            evaluate_pruning_curves(
                adapter,
                cache,
                score_sets=score_sets,
                random_seeds=RANDOM_PRUNING_SEEDS,
                token_batch_size=args.pruning_token_batch_size,
                output_path=pruning_path,
            )
        finish("pruning", started)

        started = stage("stratified single-neuron ablations")
        force_per_group = 4 if args.smoke else 24
        union, samples, groups = select_ablation_neurons(
            score_sets[0],
            sample_size=args.ablation_sample_size,
            seeds=ABLATION_SAMPLE_SEEDS,
            force_per_group=force_per_group,
        )
        write_json(
            model_dir / "ablation_samples.json",
            {
                "union": union.tolist(),
                "samples": {str(seed): values.tolist() for seed, values in samples.items()},
                "groups": {name: values.tolist() for name, values in groups.items()},
            },
        )
        ablation_path = model_dir / "single_neuron_ablations.npz"
        if ablation_path.exists() and not args.force:
            ablations = load_npz(ablation_path)
            print("  resumed single_neuron_ablations.npz", flush=True)
        else:
            ablations = evaluate_single_neuron_ablations(
                adapter,
                cache,
                neuron_indices=union,
                token_batch_size=args.ablation_token_batch_size,
                neuron_batch_size=args.ablation_neuron_batch_size,
                max_tokens=args.ablation_tokens,
                output_path=ablation_path,
            )
        summarize_ablation_correlations(
            ablations,
            score_sets=score_sets,
            samples=samples,
            output_path=model_dir / "ablation_correlations.json",
        )
        extract_outlier_records(
            adapter,
            cache,
            scores=score_sets[0],
            groups=groups,
            ablations=ablations,
            output_path=model_dir / "outliers.json",
        )
        finish("single-neuron ablations", started)

        started = stage("Nord figures")
        if args.skip_animation:
            from ohope.plotting import (
                plot_correlation_runs,
                plot_geometry,
                plot_pruning_spaghetti,
                plot_rank_disagreement,
            )

            figure_dir = model_dir / "figures"
            plot_geometry(geometry_path, figure_dir / "geometry.png")
            plot_rank_disagreement(
                model_dir / "scores_seed_0.npz",
                figure_dir / "rank-disagreement.png",
            )
            plot_pruning_spaghetti(
                pruning_path,
                figure_dir / "pruning-spaghetti.png",
            )
            plot_correlation_runs(
                model_dir / "ablation_correlations.json",
                figure_dir / "ablation-correlations.png",
            )
        else:
            render_model_artifacts(model_dir)
        finish("figures", started)

        write_json(
            model_dir / "done.json",
            {
                "model": spec.key,
                "hub_id": spec.hub_id,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "smoke": args.smoke,
                "fisher": not args.skip_fisher,
            },
        )
        print(f"\ncomplete: {model_dir}", flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
