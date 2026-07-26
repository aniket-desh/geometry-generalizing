from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from geogen.metrics import geometry_summary, table_compositionality
from geogen.model import PRESETS, GeometryTransformer, ModelConfig
from geogen.tasks import TaskSpec, available_tasks, make_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=available_tasks(), required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--sequence-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--aliases", type=int, default=16)
    parser.add_argument("--eval-aliases", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument(
        "--output-root", type=Path, default=Path("hmm-geometry-results")
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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


def action_relations(task: TaskSpec) -> np.ndarray:
    order = task.order
    if task.family in {"cycle", "broken_cycle"}:
        return np.array([0, 1, order - 1, 2, order - 2], dtype=np.int64)
    if task.family == "torus":
        side = round(math.sqrt(order))
        return np.array(
            [0, 1, side - 1, side, side * (side - 1)], dtype=np.int64
        )
    if task.family == "xor":
        return np.array([0, 1, 2, 4, 8], dtype=np.int64)
    if task.family == "dihedral":
        rotations = order // 2
        return np.array([0, 1, rotations - 1, rotations], dtype=np.int64)
    if task.family == "path":
        center = order // 2
        return np.array(
            [center, center + 1, center - 1, center + 2, center - 2],
            dtype=np.int64,
        )
    return np.arange(min(5, task.relation_count), dtype=np.int64)


def make_pair_split(
    states: int, actions: int, fraction: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = max(round(states * actions * fraction), states * 2)
    target = max(target, states)
    for _ in range(10_000):
        mask = np.zeros((states, actions), dtype=bool)
        mask[:, 0] = True
        remaining = target - states
        candidates = np.arange(states * actions).reshape(states, actions)[:, 1:]
        chosen = rng.choice(candidates.flatten(), size=remaining, replace=False)
        mask.flat[chosen] = True
        if (
            np.all(mask.sum(axis=1) >= 2)
            and np.all((~mask).sum(axis=1) >= 1)
            and np.all(mask.sum(axis=0) >= 2)
            and np.all((~mask)[:, 1:].sum(axis=0) >= 1)
        ):
            return mask
    raise RuntimeError("could not construct a balanced state-action split")


class HMMTokenLayout:
    def __init__(self, states: int, aliases: int, actions: int) -> None:
        self.states = states
        self.aliases = aliases
        self.actions = actions
        self.bos = 0
        self.emission_base = 1
        self.action_base = self.emission_base + states * aliases
        self.vocab_size = self.action_base + actions
        self.output_classes = states * aliases

    def emission(self, state: torch.Tensor, alias: torch.Tensor) -> torch.Tensor:
        return self.emission_base + alias * self.states + state

    def action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_base + action

    def output(self, state: torch.Tensor, alias: torch.Tensor) -> torch.Tensor:
        return alias * self.states + state


def generate_batch(
    *,
    batch_size: int,
    sequence_steps: int,
    task_table: torch.Tensor,
    relations: torch.Tensor,
    allowed: torch.Tensor,
    layout: HMMTokenLayout,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.empty(
        (batch_size, sequence_steps + 1), dtype=torch.long, device=device
    )
    actions = torch.empty(
        (batch_size, sequence_steps), dtype=torch.long, device=device
    )
    outputs = torch.empty_like(actions)
    states[:, 0] = torch.randint(
        layout.states, (batch_size,), device=device
    )
    aliases = torch.randint(
        layout.aliases,
        (batch_size, sequence_steps + 1),
        device=device,
    )
    for step in range(sequence_steps):
        valid = allowed[states[:, step]].float()
        action = torch.multinomial(valid, num_samples=1).squeeze(1)
        relation = relations[action]
        next_state = task_table[states[:, step], relation]
        actions[:, step] = action
        states[:, step + 1] = next_state
        outputs[:, step] = layout.output(next_state, aliases[:, step + 1])

    tokens = torch.empty(
        (batch_size, 2 * sequence_steps + 2),
        dtype=torch.long,
        device=device,
    )
    tokens[:, 0] = layout.bos
    tokens[:, 1::2] = layout.emission(states, aliases)
    tokens[:, 2::2] = layout.action(actions)
    action_positions = torch.arange(
        2, 2 * sequence_steps + 1, 2, device=device
    )
    return tokens, outputs, states, action_positions


def build_evaluation_prompts(
    *,
    task_table: torch.Tensor,
    relations: torch.Tensor,
    layout: HMMTokenLayout,
    eval_aliases: int,
    sequence_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states, actions, contexts = torch.meshgrid(
        torch.arange(layout.states, device=device),
        torch.arange(layout.actions, device=device),
        torch.arange(eval_aliases, device=device),
        indexing="ij",
    )
    current = states.flatten()
    action = actions.flatten()
    context = contexts.flatten()
    target_state = task_table[current, relations[action]]

    tokens = torch.empty(
        (current.shape[0], 2 * sequence_steps + 1),
        dtype=torch.long,
        device=device,
    )
    tokens[:, 0] = layout.bos
    for step in range(sequence_steps):
        alias = (context + step) % layout.aliases
        tokens[:, 1 + 2 * step] = layout.emission(current, alias)
        if step < sequence_steps - 1:
            tokens[:, 2 + 2 * step] = layout.action(
                torch.zeros_like(action)
            )
    tokens[:, -1] = layout.action(action)
    return tokens, current, action, target_state


@torch.no_grad()
def evaluate(
    model: GeometryTransformer,
    *,
    task: TaskSpec,
    task_table: torch.Tensor,
    relations: torch.Tensor,
    train_mask: torch.Tensor,
    layout: HMMTokenLayout,
    eval_aliases: int,
    sequence_steps: int,
    device: torch.device,
) -> tuple[
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    model.eval()
    tokens, current, action, target_state = build_evaluation_prompts(
        task_table=task_table,
        relations=relations,
        layout=layout,
        eval_aliases=eval_aliases,
        sequence_steps=sequence_steps,
        device=device,
    )
    logits, states = model(tokens, return_states=True)
    state_logits = torch.logsumexp(
        logits.view(-1, layout.aliases, layout.states), dim=1
    )
    state_log_probs = state_logits - torch.logsumexp(
        state_logits, dim=-1, keepdim=True
    )
    prediction = state_logits.argmax(dim=-1)
    correct = prediction == target_state
    pair_is_train = train_mask[current, action]
    nll = -state_log_probs[
        torch.arange(target_state.shape[0], device=device), target_state
    ]

    emission_logits = logits.view(-1, layout.aliases, layout.states)
    target_alias_logits = emission_logits.gather(
        2,
        target_state[:, None, None].expand(-1, layout.aliases, 1),
    ).squeeze(2)
    target_alias_log_probs = target_alias_logits - torch.logsumexp(
        target_alias_logits, dim=1, keepdim=True
    )
    uniform_alias_kl = (
        -math.log(layout.aliases)
        - target_alias_log_probs.mean(dim=1)
    )
    behavior = {
        "train_accuracy": float(correct[pair_is_train].float().mean()),
        "test_accuracy": float(correct[~pair_is_train].float().mean()),
        "train_state_nll": float(nll[pair_is_train].mean()),
        "test_state_nll": float(nll[~pair_is_train].mean()),
        "alias_uniform_kl": float(uniform_alias_kl.mean()),
    }

    layer_count = len(states)
    width = states[0].shape[-1]
    node_centroids = torch.zeros(
        (layer_count, layout.states, width), device=device
    )
    output_centroids = torch.zeros_like(node_centroids)
    action_centroids = torch.zeros(
        (
            layer_count,
            layout.states,
            layout.actions,
            width,
        ),
        device=device,
    )
    node_counts = torch.zeros(layout.states, device=device)
    output_counts = torch.zeros(layout.states, device=device)
    emission_position = tokens.shape[1] - 2
    action_position = tokens.shape[1] - 1
    node_counts.index_add_(
        0, current, torch.ones_like(current, dtype=torch.float32)
    )
    output_counts.index_add_(
        0, target_state, torch.ones_like(target_state, dtype=torch.float32)
    )
    for layer, state in enumerate(states):
        node_centroids[layer].index_add_(
            0, current, state[:, emission_position].float()
        )
        output_centroids[layer].index_add_(
            0, target_state, state[:, action_position].float()
        )
        flat_pair = current * layout.actions + action
        action_centroids[layer].view(
            layout.states * layout.actions, width
        ).index_add_(0, flat_pair, state[:, action_position].float())
    node_centroids /= node_counts[None, :, None]
    output_centroids /= output_counts[None, :, None]
    action_centroids /= float(eval_aliases)

    final_actions = action_centroids[-1]
    final_outputs = output_centroids[-1]
    predicted_target_centroids = final_outputs[
        task_table[:, relations].T
    ].transpose(0, 1)
    within = torch.mean(
        torch.sum((final_actions - predicted_target_centroids) ** 2, dim=-1)
    )
    total = torch.mean(
        torch.sum(
            (final_actions - final_actions.mean(dim=(0, 1))) ** 2,
            dim=-1,
        )
    )
    target_collapse = float(within / total) if total > 1e-12 else float("nan")

    node_layers = [
        geometry_summary(layer, task)
        for layer in node_centroids.cpu().numpy()
    ]
    output_layers = [
        geometry_summary(layer, task)
        for layer in output_centroids.cpu().numpy()
    ]
    return (
        behavior,
        node_centroids.cpu().numpy(),
        output_centroids.cpu().numpy(),
        action_centroids.cpu().numpy(),
        {
            "node": node_layers[-1],
            "output": output_layers[-1],
            "node_layers": node_layers,
            "output_layers": output_layers,
            "target_collapse": target_collapse,
        },
    )


def run() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    split_seed = args.seed if args.split_seed is None else args.split_seed
    task = make_task(args.task, seed=split_seed)
    relations_np = action_relations(task)
    train_mask_np = make_pair_split(
        task.order, len(relations_np), args.train_fraction, split_seed
    )
    run_payload = {
        "experiment": "controlled_hmm",
        "task": args.task,
        "preset": args.preset,
        "seed": args.seed,
        "split_seed": split_seed,
        "steps": args.steps,
        "sequence_steps": args.sequence_steps,
        "train_fraction": args.train_fraction,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "aliases": args.aliases,
    }
    digest = hashlib.sha1(
        json.dumps(run_payload, sort_keys=True).encode()
    ).hexdigest()[:8]
    run_name = f"hmm-{args.task}-{args.preset}-s{args.seed}-{digest}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "done.json"
    if done_path.exists():
        print(f"{run_name}: already complete", flush=True)
        return

    device = torch.device(args.device)
    table = torch.as_tensor(task.table, device=device, dtype=torch.long)
    relations = torch.as_tensor(
        relations_np, device=device, dtype=torch.long
    )
    train_mask = torch.as_tensor(train_mask_np, device=device)
    layout = HMMTokenLayout(
        task.order, args.aliases, len(relations_np)
    )
    config: ModelConfig = PRESETS[args.preset]
    raw_model = GeometryTransformer(
        vocab_size=layout.vocab_size,
        output_classes=layout.output_classes,
        config=config,
        sequence_length=2 * args.sequence_steps + 2,
    ).to(device)
    model = (
        torch.compile(raw_model, mode="reduce-overhead")
        if args.compile
        else raw_model
    )
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    config_payload = {
        **run_payload,
        "run_name": run_name,
        "task_family": task.family,
        "task_description": task.description,
        "task_order": task.order,
        "action_relations": relations_np.tolist(),
        "model": config.__dict__,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "snapshot_every": args.snapshot_every,
        "compile": args.compile,
        "table_compositionality_error": table_compositionality(task),
        "parameter_count": sum(
            parameter.numel() for parameter in raw_model.parameters()
        ),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    (run_dir / "config.json").write_text(
        json.dumps(json_safe(config_payload), indent=2) + "\n"
    )
    np.save(run_dir / "operation_table.npy", task.table)
    np.save(run_dir / "train_mask.npy", train_mask_np)

    metrics_path = run_dir / "metrics.jsonl"
    snapshot_steps = {0, args.steps}
    snapshot_steps.update(
        range(args.snapshot_every, args.steps, args.snapshot_every)
    )
    checkpoint_steps = {args.steps}
    checkpoint_steps.update(
        range(args.checkpoint_every, args.steps, args.checkpoint_every)
    )
    start = time.time()

    for step in range(args.steps + 1):
        should_evaluate = (
            step == 0
            or step == args.steps
            or step % args.eval_every == 0
        )
        if should_evaluate:
            (
                behavior,
                node_centroids,
                output_centroids,
                action_centroids,
                geometry,
            ) = evaluate(
                raw_model,
                task=task,
                task_table=table,
                relations=relations,
                train_mask=train_mask,
                layout=layout,
                eval_aliases=min(args.eval_aliases, args.aliases),
                sequence_steps=args.sequence_steps,
                device=device,
            )
            record = {
                "step": step,
                "elapsed_seconds": time.time() - start,
                **behavior,
                "node_geometry": geometry["node"],
                "output_geometry": geometry["output"],
                "node_layers": geometry["node_layers"],
                "output_layers": geometry["output_layers"],
                "target_collapse": geometry["target_collapse"],
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(json_safe(record)) + "\n")
            print(
                f"{run_name} step={step:06d} "
                f"train={behavior['train_accuracy']:.3f} "
                f"test={behavior['test_accuracy']:.3f} "
                f"cyc={geometry['node']['cyclic_defect']:.3f} "
                f"op={geometry['node']['generator_error']:.3f}",
                flush=True,
            )
            if step in snapshot_steps:
                np.savez_compressed(
                    run_dir / f"activations-{step:06d}.npz",
                    node=node_centroids.astype(np.float32),
                    output=output_centroids.astype(np.float32),
                    action=action_centroids.astype(np.float32),
                )
        if step in checkpoint_steps and step > 0:
            torch.save(
                {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config_payload,
                },
                run_dir / f"checkpoint-{step:06d}.pt",
            )
        if step == args.steps:
            break

        raw_model.train()
        tokens, targets, _, action_positions = generate_batch(
            batch_size=args.batch_size,
            sequence_steps=args.sequence_steps,
            task_table=table,
            relations=relations,
            allowed=train_mask,
            layout=layout,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(tokens, all_positions=True)
            action_logits = logits[:, action_positions]
            loss = F.cross_entropy(
                action_logits.reshape(-1, layout.output_classes),
                targets.reshape(-1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

    done_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "elapsed_seconds": time.time() - start,
                "final_step": args.steps,
                "metrics_file": str(metrics_path),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run()
