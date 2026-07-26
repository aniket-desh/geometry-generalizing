from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

from .config import PRUNE_FRACTIONS
from .data import iter_batches
from .modeling import FinalMLPAdapter


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def collect_activation_rms(
    adapter: FinalMLPAdapter,
    blocks: torch.Tensor,
    *,
    prediction_tokens: int,
    batch_size: int,
) -> np.ndarray:
    sum_squares = torch.zeros(adapter.d_mlp, dtype=torch.float64)
    seen = 0

    for batch_index, batch in enumerate(iter_batches(blocks, batch_size)):
        captured = adapter.capture(batch.to(adapter.device))
        acts = captured.post_activations[:, :-1].reshape(-1, adapter.d_mlp)
        remaining = prediction_tokens - seen
        acts = acts[:remaining]
        sum_squares += acts.double().square().sum(dim=0).cpu()
        seen += acts.shape[0]
        if batch_index % 10 == 0:
            print(f"  calibration tokens: {seen:,}/{prediction_tokens:,}", flush=True)
        if seen >= prediction_tokens:
            break

    if seen != prediction_tokens:
        raise RuntimeError(f"collected {seen:,} calibration tokens, expected {prediction_tokens:,}")
    return torch.sqrt(sum_squares / seen).float().numpy()


def collect_evaluation_cache(
    adapter: FinalMLPAdapter,
    blocks: torch.Tensor,
    *,
    prediction_tokens: int,
    batch_size: int,
    output_path: Path,
) -> dict[str, torch.Tensor]:
    residual_parts = []
    activation_parts = []
    label_parts = []
    seen = 0

    for batch_index, batch in enumerate(iter_batches(blocks, batch_size)):
        batch = batch.to(adapter.device)
        captured = adapter.capture(batch)
        residual = captured.pre_final_norm[:, :-1].reshape(-1, adapter.d_model)
        acts = captured.post_activations[:, :-1].reshape(-1, adapter.d_mlp)
        labels = batch[:, 1:].reshape(-1)
        remaining = prediction_tokens - seen
        residual_parts.append(residual[:remaining].to(dtype=torch.float16, device="cpu"))
        activation_parts.append(acts[:remaining].to(dtype=torch.float16, device="cpu"))
        label_parts.append(labels[:remaining].cpu())
        seen += min(residual.shape[0], remaining)
        if batch_index % 10 == 0:
            print(f"  cached evaluation tokens: {seen:,}/{prediction_tokens:,}", flush=True)
        if seen >= prediction_tokens:
            break

    if seen != prediction_tokens:
        raise RuntimeError(f"cached {seen:,} evaluation tokens, expected {prediction_tokens:,}")

    cache = {
        "residual": torch.cat(residual_parts),
        "activations": torch.cat(activation_parts),
        "labels": torch.cat(label_parts),
        "blocks": blocks.cpu(),
        "predictions_per_block": torch.tensor(blocks.shape[1] - 1),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


@torch.inference_mode()
def compute_geometry(
    adapter: FinalMLPAdapter,
    *,
    chunk_size: int,
    output_path: Path,
) -> dict[str, np.ndarray]:
    w_out = adapter.w_out.float()
    w_u = adapter.w_u.float()
    vocab_size = w_u.shape[1]

    outgoing_norm = w_out.norm(dim=1).cpu()
    logit_rms = adapter.centered_logit_rms(w_out, chunk_size=chunk_size)
    q_raw = logit_rms / outgoing_norm.clamp_min(1e-30)

    mean_column = w_u.mean(dim=1)
    metric = (w_u @ w_u.transpose(0, 1)) / vocab_size
    metric = metric - torch.outer(mean_column, mean_column)
    metric = 0.5 * (metric + metric.transpose(0, 1))
    eigenvalues = torch.linalg.eigvalsh(metric).cpu()
    mean_eigenvalue = eigenvalues.mean().clamp_min(1e-30)
    normalized_eigenvalues = eigenvalues / mean_eigenvalue
    q_normalized = q_raw / mean_eigenvalue.sqrt()

    positive = eigenvalues[eigenvalues > eigenvalues.max() * 1e-10]
    stable_condition = (
        float(eigenvalues.max() / positive.min()) if positive.numel() else float("inf")
    )
    payload = {
        "outgoing_norm": outgoing_norm.numpy(),
        "logit_rms": logit_rms.numpy(),
        "q_raw": q_raw.numpy(),
        "q_normalized": q_normalized.numpy(),
        "eigenvalues": eigenvalues.numpy(),
        "normalized_eigenvalues": normalized_eigenvalues.numpy(),
        "stable_condition_number": np.asarray(stable_condition),
        "trace": np.asarray(float(torch.trace(metric).cpu())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return payload


def save_scores(
    *,
    activation_rms: np.ndarray,
    geometry: dict[str, np.ndarray],
    output_path: Path,
) -> dict[str, np.ndarray]:
    scores = {
        "activation_rms": activation_rms,
        "outgoing_norm": geometry["outgoing_norm"],
        "hope": activation_rms * geometry["outgoing_norm"],
        "ohope": activation_rms * geometry["logit_rms"],
        "q_normalized": geometry["q_normalized"],
    }
    np.savez_compressed(output_path, **scores)
    return scores


def _variant_metrics(
    base_logp: torch.Tensor,
    variant_logp: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_probability = base_logp.exp()
    kl = (base_probability * (base_logp - variant_logp)).sum(dim=-1)
    base_nll = -base_logp.gather(-1, labels[:, None]).squeeze(-1)
    variant_nll = -variant_logp.gather(-1, labels[:, None]).squeeze(-1)
    agreement = base_logp.argmax(dim=-1).eq(variant_logp.argmax(dim=-1)).float()
    return kl, variant_nll - base_nll, agreement


def _ranking_specs(
    score_sets: dict[int, dict[str, np.ndarray]],
    random_seeds: Iterable[int],
) -> list[dict]:
    first_scores = next(iter(score_sets.values()))
    specs = [
        {
            "method": "outgoing_norm",
            "seed": -1,
            "scores": first_scores["outgoing_norm"],
        }
    ]
    for seed, scores in score_sets.items():
        for method in ("activation_rms", "hope", "ohope"):
            specs.append({"method": method, "seed": seed, "scores": scores[method]})
    if "fisher" in first_scores:
        specs.append({"method": "fisher", "seed": -1, "scores": first_scores["fisher"]})
    width = len(first_scores["hope"])
    for seed in random_seeds:
        rng = np.random.default_rng(seed)
        specs.append({"method": "random", "seed": seed, "scores": rng.random(width)})
    return specs


@torch.inference_mode()
def evaluate_pruning_curves(
    adapter: FinalMLPAdapter,
    cache: dict[str, torch.Tensor],
    *,
    score_sets: dict[int, dict[str, np.ndarray]],
    random_seeds: Iterable[int],
    token_batch_size: int,
    output_path: Path,
) -> pd.DataFrame:
    ranking_specs = _ranking_specs(score_sets, random_seeds)
    fractions = tuple(PRUNE_FRACTIONS)
    conditions = [
        (spec, fraction)
        for spec in ranking_specs
        for fraction in fractions
    ]
    totals = {
        (spec["method"], spec["seed"], fraction): np.zeros(4, dtype=np.float64)
        for spec, fraction in conditions
    }
    batch_rows = []

    residual_all = cache["residual"]
    acts_all = cache["activations"]
    labels_all = cache["labels"]
    w_out = adapter.w_out

    for start in range(0, residual_all.shape[0], token_batch_size):
        stop = min(start + token_batch_size, residual_all.shape[0])
        residual = residual_all[start:stop].to(
            device=adapter.device,
            dtype=adapter.dtype,
        )
        acts = acts_all[start:stop].to(
            device=adapter.device,
            dtype=adapter.dtype,
        )
        labels = labels_all[start:stop].to(adapter.device)

        base_logp = F.log_softmax(adapter.logits_from_residual(residual).float(), dim=-1)
        base_nll = -base_logp.gather(-1, labels[:, None]).squeeze(-1)
        base_top = base_logp.argmax(dim=-1)
        base_probability = base_logp.exp()

        for spec in ranking_specs:
            order = np.argsort(spec["scores"])
            previous = 0
            delta = torch.zeros_like(residual)
            for fraction in fractions:
                count = int(round(fraction * adapter.d_mlp))
                new_indices = torch.as_tensor(
                    order[previous:count],
                    device=adapter.device,
                    dtype=torch.long,
                )
                if new_indices.numel():
                    delta = delta + acts.index_select(1, new_indices) @ w_out.index_select(0, new_indices)
                previous = count

                logp = F.log_softmax(adapter.logits_from_residual(residual - delta).float(), dim=-1)
                kl = (base_probability * (base_logp - logp)).sum(dim=-1)
                nll = -logp.gather(-1, labels[:, None]).squeeze(-1)
                agreement = logp.argmax(dim=-1).eq(base_top).float()
                key = (spec["method"], spec["seed"], fraction)
                totals[key] += np.array(
                    [
                        float(kl.sum().cpu()),
                        float(nll.sum().cpu()),
                        float(base_nll.sum().cpu()),
                        float(agreement.sum().cpu()),
                    ]
                )
                batch_rows.append(
                    {
                        "method": spec["method"],
                        "seed": spec["seed"],
                        "fraction_pruned": fraction,
                        "batch": start // token_batch_size,
                        "tokens": stop - start,
                        "kl": float(kl.mean().cpu()),
                        "nll": float(nll.mean().cpu()),
                        "base_nll": float(base_nll.mean().cpu()),
                        "top_agreement": float(agreement.mean().cpu()),
                    }
                )
                del logp, kl, nll, agreement

        if start % (token_batch_size * 10) == 0:
            print(
                f"  pruning tokens: {stop:,}/{residual_all.shape[0]:,}",
                flush=True,
            )
        del base_logp, base_probability

    summary_rows = []
    n_tokens = residual_all.shape[0]
    for key, sums in totals.items():
        method, seed, fraction = key
        kl_sum, nll_sum, base_nll_sum, agreement_sum = sums
        mean_nll = nll_sum / n_tokens
        summary_rows.append(
            {
                "method": method,
                "seed": seed,
                "fraction_pruned": fraction,
                "fraction_retained": 1.0 - fraction,
                "kl": kl_sum / n_tokens,
                "delta_nll": (nll_sum - base_nll_sum) / n_tokens,
                "nll": mean_nll,
                "perplexity": math.exp(min(mean_nll, 30.0)),
                "top_agreement": agreement_sum / n_tokens,
                "tokens": n_tokens,
            }
        )

    frame = pd.DataFrame(summary_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    pd.DataFrame(batch_rows).to_csv(output_path.with_name("pruning_batches.csv"), index=False)
    return frame


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-12)


def select_ablation_neurons(
    scores: dict[str, np.ndarray],
    *,
    sample_size: int,
    seeds: Iterable[int],
    force_per_group: int = 32,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, np.ndarray]]:
    hope_rank = rankdata(scores["hope"], method="average") / len(scores["hope"])
    ohope_rank = rankdata(scores["ohope"], method="average") / len(scores["ohope"])
    disagreement = hope_rank - ohope_rank

    groups = {
        "high_hope_low_ohope": np.argsort(disagreement)[::-1][:force_per_group],
        "low_hope_high_ohope": np.argsort(disagreement)[:force_per_group],
        "high_both": np.argsort(np.minimum(hope_rank, ohope_rank))[::-1][:force_per_group],
        "low_both": np.argsort(np.maximum(hope_rank, ohope_rank))[:force_per_group],
    }
    forced = np.unique(np.concatenate(list(groups.values())))
    all_indices = np.arange(len(hope_rank))
    samples = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        remaining = np.setdiff1d(all_indices, forced, assume_unique=False)
        fill = max(0, min(sample_size - forced.size, remaining.size))
        sample = np.unique(
            np.concatenate([forced, rng.choice(remaining, size=fill, replace=False)])
        )
        samples[int(seed)] = np.sort(sample)
    union = np.unique(np.concatenate(list(samples.values())))
    return union, samples, groups


@torch.inference_mode()
def evaluate_single_neuron_ablations(
    adapter: FinalMLPAdapter,
    cache: dict[str, torch.Tensor],
    *,
    neuron_indices: np.ndarray,
    token_batch_size: int,
    neuron_batch_size: int,
    max_tokens: int,
    output_path: Path,
) -> dict[str, np.ndarray]:
    n_tokens = min(max_tokens, cache["residual"].shape[0])
    n_batches = (n_tokens + token_batch_size - 1) // token_batch_size
    n_neurons = len(neuron_indices)
    kl_batches = np.zeros((n_neurons, n_batches), dtype=np.float32)
    delta_nll_batches = np.zeros_like(kl_batches)
    agreement_batches = np.zeros_like(kl_batches)
    token_counts = np.zeros(n_batches, dtype=np.int32)
    row_for_neuron = {int(neuron): row for row, neuron in enumerate(neuron_indices)}

    residual_all = cache["residual"][:n_tokens]
    acts_all = cache["activations"][:n_tokens]
    labels_all = cache["labels"][:n_tokens]
    w_out = adapter.w_out

    for batch_number, start in enumerate(range(0, n_tokens, token_batch_size)):
        stop = min(start + token_batch_size, n_tokens)
        token_counts[batch_number] = stop - start
        residual = residual_all[start:stop].to(
            device=adapter.device,
            dtype=adapter.dtype,
        )
        acts = acts_all[start:stop].to(
            device=adapter.device,
            dtype=adapter.dtype,
        )
        labels = labels_all[start:stop].to(adapter.device)

        base_logp = F.log_softmax(adapter.logits_from_residual(residual).float(), dim=-1)
        base_probability = base_logp.exp()
        base_nll = -base_logp.gather(-1, labels[:, None]).squeeze(-1)
        base_top = base_logp.argmax(dim=-1)

        for neuron_start in range(0, n_neurons, neuron_batch_size):
            selected_np = neuron_indices[neuron_start : neuron_start + neuron_batch_size]
            selected = torch.as_tensor(selected_np, device=adapter.device, dtype=torch.long)
            selected_acts = acts.index_select(1, selected)
            selected_vectors = w_out.index_select(0, selected)
            modified = (
                residual[:, None, :]
                - selected_acts[:, :, None] * selected_vectors[None, :, :]
            )
            logp = F.log_softmax(adapter.logits_from_residual(modified).float(), dim=-1)
            kl = (
                base_probability[:, None, :]
                * (base_logp[:, None, :] - logp)
            ).sum(dim=-1)
            variant_nll = -logp.gather(
                -1,
                labels[:, None, None].expand(-1, selected.shape[0], 1),
            ).squeeze(-1)
            agreement = logp.argmax(dim=-1).eq(base_top[:, None]).float()

            for local, neuron in enumerate(selected_np):
                row = row_for_neuron[int(neuron)]
                kl_batches[row, batch_number] = float(kl[:, local].mean().cpu())
                delta_nll_batches[row, batch_number] = float(
                    (variant_nll[:, local] - base_nll).mean().cpu()
                )
                agreement_batches[row, batch_number] = float(
                    agreement[:, local].mean().cpu()
                )
            del modified, logp, kl, variant_nll, agreement

        if batch_number % 10 == 0:
            print(f"  ablation tokens: {stop:,}/{n_tokens:,}", flush=True)
        del base_logp, base_probability

    payload = {
        "neuron_indices": neuron_indices.astype(np.int32),
        "kl_batches": kl_batches,
        "delta_nll_batches": delta_nll_batches,
        "agreement_batches": agreement_batches,
        "token_counts": token_counts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return payload


def summarize_ablation_correlations(
    ablations: dict[str, np.ndarray],
    *,
    score_sets: dict[int, dict[str, np.ndarray]],
    samples: dict[int, np.ndarray],
    output_path: Path,
) -> list[dict]:
    union = ablations["neuron_indices"]
    union_row = {int(neuron): row for row, neuron in enumerate(union)}
    weights = ablations["token_counts"].astype(np.float64)
    weights /= weights.sum()
    damage = ablations["kl_batches"] @ weights
    rows = []

    for sample_seed, sample in samples.items():
        chosen_rows = np.asarray([union_row[int(neuron)] for neuron in sample])
        chosen_damage = damage[chosen_rows]
        for calibration_seed, scores in score_sets.items():
            methods = ["outgoing_norm", "activation_rms", "hope", "ohope"]
            if "fisher" in scores:
                methods.append("fisher")
            for method in methods:
                correlation = spearmanr(
                    scores[method][sample],
                    chosen_damage,
                    nan_policy="omit",
                ).statistic
                rows.append(
                    {
                        "sample_seed": int(sample_seed),
                        "calibration_seed": int(calibration_seed),
                        "method": method,
                        "spearman_kl": float(correlation),
                        "neurons": int(len(sample)),
                        "evaluation_tokens": int(weights.size and ablations["token_counts"].sum()),
                    }
                )
    write_json(output_path, rows)
    return rows


def extract_outlier_records(
    adapter: FinalMLPAdapter,
    cache: dict[str, torch.Tensor],
    *,
    scores: dict[str, np.ndarray],
    groups: dict[str, np.ndarray],
    ablations: dict[str, np.ndarray],
    output_path: Path,
    per_group: int = 10,
    contexts_per_neuron: int = 5,
    tokens_per_side: int = 10,
) -> list[dict]:
    union = ablations["neuron_indices"]
    union_row = {int(neuron): row for row, neuron in enumerate(union)}
    weights = ablations["token_counts"].astype(np.float64)
    weights /= weights.sum()
    mean_kl = ablations["kl_batches"] @ weights
    mean_delta_nll = ablations["delta_nll_batches"] @ weights
    mean_agreement = ablations["agreement_batches"] @ weights

    acts = cache["activations"]
    blocks = cache["blocks"]
    predictions_per_block = int(cache["predictions_per_block"])
    w_out = adapter.w_out.detach()
    w_u = adapter.w_u.detach()
    records = []

    for group_name, candidates in groups.items():
        for neuron in candidates[:per_group]:
            neuron = int(neuron)
            if neuron not in union_row:
                continue
            row = union_row[neuron]
            top_indices = torch.topk(
                acts[:, neuron],
                k=min(contexts_per_neuron, acts.shape[0]),
            ).indices.tolist()
            contexts = []
            for flat_index in top_indices:
                block_index = flat_index // predictions_per_block
                position = flat_index % predictions_per_block
                start = max(0, position - 24)
                token_slice = blocks[block_index, start : position + 1].tolist()
                contexts.append(adapter.tokenizer.decode(token_slice))

            logit_effect = w_out[neuron].to(adapter.device) @ w_u
            promoted = torch.topk(logit_effect, k=tokens_per_side)
            suppressed = torch.topk(-logit_effect, k=tokens_per_side)
            records.append(
                {
                    "group": group_name,
                    "neuron": neuron,
                    "activation_rms": float(scores["activation_rms"][neuron]),
                    "hope": float(scores["hope"][neuron]),
                    "ohope": float(scores["ohope"][neuron]),
                    "q_normalized": float(scores["q_normalized"][neuron]),
                    "ablation_kl": float(mean_kl[row]),
                    "ablation_delta_nll": float(mean_delta_nll[row]),
                    "ablation_top_agreement": float(mean_agreement[row]),
                    "top_contexts": contexts,
                    "promoted_tokens": [
                        {
                            "token": adapter.tokenizer.decode([int(index)]),
                            "effect": float(value),
                        }
                        for value, index in zip(promoted.values.cpu(), promoted.indices.cpu())
                    ],
                    "suppressed_tokens": [
                        {
                            "token": adapter.tokenizer.decode([int(index)]),
                            "effect": float(-value),
                        }
                        for value, index in zip(suppressed.values.cpu(), suppressed.indices.cpu())
                    ],
                }
            )
    write_json(output_path, records)
    return records


@torch.inference_mode()
def compute_fisher_scores(
    adapter: FinalMLPAdapter,
    cache: dict[str, torch.Tensor],
    *,
    max_tokens: int,
    token_batch_size: int,
    neuron_batch_size: int,
    output_path: Path,
) -> np.ndarray:
    """Estimate the LayerNorm-Jacobian Fisher pullback for every final-MLP unit."""

    norm = adapter.final_norm
    if not hasattr(norm, "weight"):
        raise TypeError(f"unsupported final normalization: {type(norm).__name__}")

    n_tokens = min(max_tokens, cache["residual"].shape[0])
    residual_all = cache["residual"][:n_tokens]
    acts_all = cache["activations"][:n_tokens]
    w_out = adapter.w_out
    w_u = adapter.w_u
    gamma = norm.weight.detach()
    eps = float(getattr(norm, "eps", 1e-5))
    fisher_sum = torch.zeros(adapter.d_mlp, dtype=torch.float64)

    for token_start in range(0, n_tokens, token_batch_size):
        token_stop = min(token_start + token_batch_size, n_tokens)
        residual = residual_all[token_start:token_stop].to(
            device=adapter.device,
            dtype=adapter.dtype,
        )
        acts = acts_all[token_start:token_stop].to(
            device=adapter.device,
            dtype=adapter.dtype,
        )
        centered = residual - residual.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(variance + eps)
        xhat = centered / sigma
        base_logp = F.log_softmax(adapter.logits_from_residual(residual).float(), dim=-1)
        probability = base_logp.exp()

        for neuron_start in range(0, adapter.d_mlp, neuron_batch_size):
            neuron_stop = min(neuron_start + neuron_batch_size, adapter.d_mlp)
            vectors = w_out[neuron_start:neuron_stop]
            vector_centered = vectors - vectors.mean(dim=-1, keepdim=True)
            projection = torch.einsum("td,nd->tn", xhat, vectors) / adapter.d_model
            jv = (
                vector_centered[None, :, :]
                - xhat[:, None, :] * projection[:, :, None]
            )
            jv = jv * gamma[None, None, :] / sigma[:, None, :]
            logit_delta = (jv @ w_u).float()
            expected = torch.einsum("tv,tnv->tn", probability, logit_delta)
            expected_square = torch.einsum(
                "tv,tnv->tn",
                probability,
                logit_delta.square(),
            )
            quadratic = (expected_square - expected.square()).clamp_min(0.0)
            weighted = acts[:, neuron_start:neuron_stop].square() * quadratic
            fisher_sum[neuron_start:neuron_stop] += weighted.sum(dim=0).double().cpu()
            del jv, logit_delta, expected, expected_square, quadratic, weighted

        print(f"  Fisher tokens: {token_stop:,}/{n_tokens:,}", flush=True)
        del base_logp, probability

    fisher = torch.sqrt(fisher_sum / n_tokens).float().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, fisher=fisher, tokens=np.asarray(n_tokens))
    return fisher
