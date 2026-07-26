from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from geogen.metrics import geometry_summary
from geogen.tasks import TaskSpec


@dataclass(frozen=True)
class Domain:
    name: str
    values: tuple[str, ...]
    activation_templates: tuple[str, ...]
    behavior_templates: tuple[str, ...]


DOMAINS = (
    Domain(
        "weekdays",
        ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
        (
            "Today is {value}",
            "The weekday is {value}",
            "It happened on {value}",
            "Every week includes {value}",
        ),
        (
            "The day after {value} is",
            "If today is {value}, tomorrow is",
            "The weekday following {value} is",
            "{value} is followed by",
            "In the weekly cycle, after {value} comes",
            "Starting on {value}, the next day is",
        ),
    ),
    Domain(
        "months",
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        (
            "The month is {value}",
            "It happened in {value}",
            "The calendar says {value}",
            "Every year includes {value}",
        ),
        (
            "The month after {value} is",
            "One month after {value} comes",
            "The month following {value} is",
            "{value} is followed by",
            "In the yearly cycle, after {value} comes",
            "Starting in {value}, the next month is",
        ),
    ),
    Domain(
        "seasons",
        ("spring", "summer", "autumn", "winter"),
        (
            "The season is {value}",
            "This happened in {value}",
            "Every year includes {value}",
            "The weather changed during {value}",
        ),
        (
            "The season after {value} is",
            "After {value} comes",
            "The next season following {value} is",
            "{value} is followed by",
            "In the seasonal cycle, after {value} comes",
            "Starting in {value}, the following season is",
        ),
    ),
    Domain(
        "compass",
        ("north", "east", "south", "west"),
        (
            "The direction is {value}",
            "The compass points {value}",
            "Travel toward the {value}",
            "The wind came from the {value}",
        ),
        (
            "Clockwise from {value} is",
            "A quarter turn clockwise from {value} points",
            "Turning right from {value} points",
            "The next cardinal direction clockwise from {value} is",
            "Rotate {value} ninety degrees clockwise to face",
            "Moving clockwise around a compass, after {value} comes",
        ),
    ),
    Domain(
        "clock",
        (
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ),
        (
            "The clock shows {value} o'clock",
            "The hour is {value}",
            "The time was {value} o'clock",
            "At {value} o'clock the bell rang",
        ),
        (
            "One hour after {value} o'clock is",
            "If it is {value} o'clock, the next hour is",
            "The clock hour following {value} is",
            "{value} o'clock is followed by",
            "On a clock, after {value} comes",
            "Moving one hour forward from {value} gives",
        ),
    ),
    Domain(
        "digits",
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
        (
            "The digit is {value}",
            "The number is {value}",
            "Write the digit {value}",
            "This value equals {value}",
        ),
        (
            "The digit after {value} is",
            "Counting cyclically, after {value} comes",
            "Modulo ten, the successor of {value} is",
            "{value} is followed modulo ten by",
            "Counting modulo ten, after {value} comes",
            "Increment {value} modulo ten to get",
        ),
    ),
)


def cyclic_task(order: int) -> TaskSpec:
    states = np.arange(order)[:, None]
    relations = np.arange(order)[None, :]
    return TaskSpec(
        name=f"cycle{order}",
        family="cycle",
        table=(states + relations) % order,
        generator=1,
        description=f"Synthetic cycle of order {order}.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revisions",
        nargs="+",
        default=(
            "step0",
            "step1",
            "step4",
            "step16",
            "step64",
            "step256",
            "step1000",
            "step4000",
            "step16000",
            "step64000",
            "step143000",
        ),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return value.replace("/", "--")


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@torch.no_grad()
def activation_centroids(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    domain: Domain,
    device: torch.device,
) -> np.ndarray:
    prompts: list[str] = []
    groups: list[int] = []
    for index, value in enumerate(domain.values):
        for template in domain.activation_templates:
            prompts.append(template.format(value=value))
            groups.append(index)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    outputs = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    last_positions = encoded.attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(len(prompts), device=device)
    layers = []
    group_tensor = torch.as_tensor(groups, device=device)
    for hidden in outputs.hidden_states:
        selected = hidden[batch_indices, last_positions].float()
        centroids = torch.zeros(
            (len(domain.values), hidden.shape[-1]),
            device=device,
            dtype=torch.float32,
        )
        centroids.index_add_(0, group_tensor, selected)
        centroids /= len(domain.activation_templates)
        layers.append(centroids.cpu().numpy())
    return np.stack(layers)


@torch.no_grad()
def cyclic_behavior(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    domain: Domain,
    device: torch.device,
) -> dict[str, float]:
    prompts: list[str] = []
    targets: list[int] = []
    for index, value in enumerate(domain.values):
        for template in domain.behavior_templates:
            prompts.append(template.format(value=value))
            targets.append((index + 1) % len(domain.values))
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    logits = model(**encoded, use_cache=False, return_dict=True).logits
    last_positions = encoded.attention_mask.sum(dim=1) - 1
    next_logits = logits[
        torch.arange(len(prompts), device=device),
        last_positions,
    ].float()
    candidate_ids = []
    for value in domain.values:
        token_ids = tokenizer.encode(" " + value, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"could not tokenize target {value!r}")
        candidate_ids.append(token_ids[0])
    candidate_ids_tensor = torch.as_tensor(candidate_ids, device=device)
    candidate_logits = next_logits[:, candidate_ids_tensor]
    target_tensor = torch.as_tensor(targets, device=device)
    candidate_accuracy = (
        candidate_logits.argmax(dim=-1) == target_tensor
    ).float().mean()
    candidate_probability = torch.softmax(candidate_logits, dim=-1)[
        torch.arange(len(prompts), device=device),
        target_tensor,
    ].mean()
    full_probability = torch.softmax(next_logits, dim=-1)[
        torch.arange(len(prompts), device=device),
        candidate_ids_tensor[target_tensor],
    ].mean()
    return {
        "candidate_accuracy": float(candidate_accuracy),
        "candidate_probability": float(candidate_probability),
        "full_probability": float(full_probability),
    }


def run_revision(
    *,
    model_name: str,
    revision: str,
    cache_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    done_path = output_dir / f"{revision}.done.json"
    if done_path.exists():
        print(f"{model_name} {revision}: already complete", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    revision_metrics: dict[str, object] = {}
    for domain in DOMAINS:
        centroids = activation_centroids(model, tokenizer, domain, device)
        task = cyclic_task(len(domain.values))
        layer_metrics = [
            geometry_summary(layer_centroids, task)
            for layer_centroids in centroids
        ]
        behavior = cyclic_behavior(model, tokenizer, domain, device)
        np.savez_compressed(
            output_dir / f"{revision}-{domain.name}.npz",
            centroids=centroids.astype(np.float32),
        )
        revision_metrics[domain.name] = {
            "order": len(domain.values),
            "behavior": behavior,
            "layers": layer_metrics,
        }
        print(
            f"{model_name} {revision} {domain.name}: "
            f"acc={behavior['candidate_accuracy']:.3f} "
            f"p={behavior['candidate_probability']:.3f}",
            flush=True,
        )

    metrics_path = output_dir / f"{revision}.json"
    metrics_path.write_text(
        json.dumps(json_safe(revision_metrics), indent=2) + "\n"
    )
    done_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "revision": revision,
                "metrics": str(metrics_path),
            },
            indent=2,
        )
        + "\n"
    )
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = args.output_root / safe_name(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for revision in args.revisions:
        run_revision(
            model_name=args.model,
            revision=revision,
            cache_dir=args.cache_dir,
            output_dir=output_dir,
            device=device,
        )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
