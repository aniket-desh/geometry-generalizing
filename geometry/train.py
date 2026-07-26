from __future__ import annotations

import argparse
import fcntl
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
from geogen.tasks import available_tasks, make_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=available_tasks(), required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--task-seed", type=int)
    parser.add_argument("--token-seed", type=int)
    parser.add_argument("--corruption", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--aliases", type=int, default=4)
    parser.add_argument("--contexts", type=int, default=16)
    parser.add_argument("--eval-contexts", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--keep-checkpoints", type=int, default=0)
    parser.add_argument("--dense-checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--dense-checkpoint-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--output-root", type=Path, default=Path("geometry-results"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def save_dense_checkpoint(
    model: GeometryTransformer,
    path: Path,
    *,
    step: int,
    dtype: str,
) -> None:
    floating_dtype = torch.float16 if dtype == "float16" else torch.float32
    state = {
        name: (
            value.detach().cpu().to(floating_dtype)
            if value.is_floating_point()
            else value.detach().cpu()
        )
        for name, value in model.state_dict().items()
    }
    torch.save(
        {"format": "weights-only-v1", "step": step, "model": state},
        path,
    )


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(run_dir.glob("checkpoint-*.pt"))
    for checkpoint in checkpoints[:-keep]:
        checkpoint.unlink()


def make_split(order: int, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = round(order * order * fraction)
    target = max(target, order * 2)
    for _ in range(10_000):
        mask = np.zeros((order, order), dtype=bool)
        chosen = rng.choice(order * order, size=target, replace=False)
        mask.flat[chosen] = True
        if (
            np.all(mask.sum(axis=0) >= 2)
            and np.all(mask.sum(axis=1) >= 2)
            and np.all((~mask).sum(axis=0) >= 1)
            and np.all((~mask).sum(axis=1) >= 1)
        ):
            return mask
    raise RuntimeError("could not construct a balanced operation-table split")


class TokenLayout:
    def __init__(
        self,
        order: int,
        aliases: int,
        contexts: int,
        *,
        seed: int,
        device: torch.device,
    ) -> None:
        self.order = order
        self.aliases = aliases
        self.contexts = contexts
        self.bos = 0
        self.equals = 1
        self.context_base = 2
        self.node_base = self.context_base + contexts
        self.relation_base = self.node_base + order * aliases
        self.vocab_size = self.relation_base + order * aliases
        rng = np.random.default_rng(seed)
        surface = self.node_base + rng.permutation(2 * order * aliases)
        node_tokens = surface[: order * aliases].reshape(aliases, order)
        relation_tokens = surface[order * aliases :].reshape(aliases, order)
        self.node_tokens = torch.as_tensor(
            node_tokens, dtype=torch.long, device=device
        )
        self.relation_tokens = torch.as_tensor(
            relation_tokens, dtype=torch.long, device=device
        )

    def node(self, latent: torch.Tensor, alias: torch.Tensor) -> torch.Tensor:
        return self.node_tokens[alias, latent]

    def relation(self, latent: torch.Tensor, alias: torch.Tensor) -> torch.Tensor:
        return self.relation_tokens[alias, latent]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            node=self.node_tokens.cpu().numpy(),
            relation=self.relation_tokens.cpu().numpy(),
        )


def build_tokens(
    layout: TokenLayout,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    contexts: torch.Tensor,
    left_alias: torch.Tensor,
    right_alias: torch.Tensor,
) -> torch.Tensor:
    batch = left.shape[0]
    tokens = torch.empty((batch, 5), dtype=torch.long, device=left.device)
    tokens[:, 0] = layout.bos
    tokens[:, 1] = layout.context_base + contexts
    tokens[:, 2] = layout.node(left, left_alias)
    tokens[:, 3] = layout.relation(right, right_alias)
    tokens[:, 4] = layout.equals
    return tokens


@torch.no_grad()
def evaluate(
    model: GeometryTransformer,
    *,
    task_table: torch.Tensor,
    train_mask: torch.Tensor,
    layout: TokenLayout,
    eval_contexts: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    order = task_table.shape[0]
    left_base, right_base = torch.meshgrid(
        torch.arange(order, device=device),
        torch.arange(order, device=device),
        indexing="ij",
    )
    left_base = left_base.flatten()
    right_base = right_base.flatten()
    target_base = task_table[left_base, right_base]
    train_flat = train_mask.flatten()

    logits_across_contexts: list[torch.Tensor] = []
    node_sums: list[torch.Tensor] | None = None
    output_sums: list[torch.Tensor] | None = None
    output_counts = torch.zeros(order, device=device)

    for context in range(eval_contexts):
        context_ids = torch.full_like(left_base, context % layout.contexts)
        left_alias = torch.full_like(left_base, context % layout.aliases)
        right_alias = torch.full_like(
            right_base, (context // layout.aliases) % layout.aliases
        )
        tokens = build_tokens(
            layout,
            left_base,
            right_base,
            contexts=context_ids,
            left_alias=left_alias,
            right_alias=right_alias,
        )
        logits, states = model(tokens, return_states=True)
        logits_across_contexts.append(logits)
        if node_sums is None:
            node_sums = [
                torch.zeros((order, state.shape[-1]), device=device)
                for state in states
            ]
            output_sums = [
                torch.zeros((order, state.shape[-1]), device=device)
                for state in states
            ]
        for layer, state in enumerate(states):
            node_sums[layer].index_add_(0, left_base, state[:, 2].float())
            output_sums[layer].index_add_(0, target_base, state[:, -1].float())
        output_counts.index_add_(
            0, target_base, torch.ones_like(target_base, dtype=torch.float32)
        )

    mean_logits = torch.stack(logits_across_contexts).mean(dim=0)
    per_example_loss = F.cross_entropy(
        mean_logits, target_base, reduction="none"
    )
    predictions = mean_logits.argmax(dim=-1)
    correct = predictions == target_base
    metrics = {
        "train_loss": float(per_example_loss[train_flat].mean()),
        "test_loss": float(per_example_loss[~train_flat].mean()),
        "train_accuracy": float(correct[train_flat].float().mean()),
        "test_accuracy": float(correct[~train_flat].float().mean()),
    }
    assert node_sums is not None and output_sums is not None
    node_count = float(order * eval_contexts)
    node_centroids = torch.stack([value / node_count for value in node_sums])
    output_centroids = torch.stack(
        [value / output_counts[:, None] for value in output_sums]
    )
    return (
        metrics,
        node_centroids.cpu().numpy(),
        output_centroids.cpu().numpy(),
    )


def run() -> None:
    args = parse_args()
    if not 0.0 <= args.corruption <= 1.0:
        raise ValueError("--corruption must lie in [0, 1]")
    if args.keep_checkpoints < 0:
        raise ValueError("--keep-checkpoints cannot be negative")
    if args.dense_checkpoint_every < 0:
        raise ValueError("--dense-checkpoint-every cannot be negative")
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
    task_seed = args.seed if args.task_seed is None else args.task_seed
    token_seed = args.seed if args.token_seed is None else args.token_seed
    task = make_task(
        args.task,
        seed=task_seed,
        corruption=args.corruption,
    )
    train_mask_np = make_split(task.order, args.train_fraction, split_seed)
    identity_payload = {
        "task": args.task,
        "task_seed": task_seed,
        "corruption": args.corruption,
        "preset": args.preset,
        "seed": args.seed,
        "split_seed": split_seed,
        "token_seed": token_seed,
        "train_fraction": args.train_fraction,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "aliases": args.aliases,
        "contexts": args.contexts,
    }
    run_payload = {**identity_payload, "steps": args.steps}
    digest = hashlib.sha1(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:8]
    run_name = f"{task.name}-{args.preset}-s{args.seed}-{digest}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{run_name}: waiting for another launcher", flush=True)
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
    done_path = run_dir / "done.json"
    if done_path.exists():
        completed = json.loads(done_path.read_text())
        if int(completed["final_step"]) >= args.steps:
            print(f"{run_name}: already complete", flush=True)
            return

    device = torch.device(args.device)
    table = torch.as_tensor(task.table, device=device, dtype=torch.long)
    train_mask = torch.as_tensor(train_mask_np, device=device)
    train_pairs = torch.nonzero(train_mask, as_tuple=False)
    layout = TokenLayout(
        task.order,
        args.aliases,
        args.contexts,
        seed=token_seed,
        device=device,
    )
    config: ModelConfig = PRESETS[args.preset]
    raw_model = GeometryTransformer(
        vocab_size=layout.vocab_size,
        output_classes=task.order,
        config=config,
    ).to(device)
    model = torch.compile(raw_model, mode="reduce-overhead") if args.compile else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    start_step = 0
    if args.resume:
        checkpoints = sorted(run_dir.glob("checkpoint-*.pt"))
        if checkpoints:
            checkpoint = torch.load(
                checkpoints[-1], map_location=device, weights_only=False
            )
            raw_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint["step"]) + 1
            print(
                f"{run_name}: resuming at step {start_step}",
                flush=True,
            )

    config_payload = {
        **run_payload,
        "run_name": run_name,
        "task_family": task.family,
        "task_description": task.description,
        "task_order": task.order,
        "task_seed": task_seed,
        "task_corruption_fraction": task.corruption_fraction,
        "task_actual_corruption_fraction": (
            task.actual_corruption_fraction
        ),
        "task_table_sha256": hashlib.sha256(task.table.tobytes()).hexdigest(),
        "token_seed": token_seed,
        "model": config.__dict__,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "snapshot_every": args.snapshot_every,
        "checkpoint_every": args.checkpoint_every,
        "keep_checkpoints": args.keep_checkpoints,
        "dense_checkpoint_every": args.dense_checkpoint_every,
        "dense_checkpoint_dtype": args.dense_checkpoint_dtype,
        "compile": args.compile,
        "table_compositionality_error": table_compositionality(task),
        "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }
    (run_dir / "config.json").write_text(
        json.dumps(json_safe(config_payload), indent=2) + "\n"
    )
    np.save(run_dir / "operation_table.npy", task.table)
    np.save(run_dir / "train_mask.npy", train_mask_np)
    layout.save(run_dir / "token_layout.npz")

    metrics_path = run_dir / "metrics.jsonl"
    snapshot_steps = {0, args.steps}
    snapshot_steps.update(range(args.snapshot_every, args.steps, args.snapshot_every))
    checkpoint_steps = {args.steps}
    checkpoint_steps.update(
        range(args.checkpoint_every, args.steps, args.checkpoint_every)
    )
    dense_checkpoint_steps: set[int] = set()
    if args.dense_checkpoint_every:
        dense_checkpoint_steps = {args.steps}
        dense_checkpoint_steps.update(
            range(
                args.dense_checkpoint_every,
                args.steps,
                args.dense_checkpoint_every,
            )
        )
    start = time.time()

    for step in range(start_step, args.steps + 1):
        should_evaluate = (
            step == 0
            or step == args.steps
            or step % args.eval_every == 0
        )
        if should_evaluate:
            eval_metrics, node_centroids, output_centroids = evaluate(
                raw_model,
                task_table=table,
                train_mask=train_mask,
                layout=layout,
                eval_contexts=args.eval_contexts,
                device=device,
            )
            node_layers = [
                geometry_summary(layer, task) for layer in node_centroids
            ]
            output_layers = [
                geometry_summary(layer, task) for layer in output_centroids
            ]
            final_node = node_layers[-1]
            final_output = output_layers[-1]
            record = {
                "step": step,
                "elapsed_seconds": time.time() - start,
                **eval_metrics,
                "node_geometry": final_node,
                "output_geometry": final_output,
                "node_layers": node_layers,
                "output_layers": output_layers,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(json_safe(record)) + "\n")
            print(
                f"{run_name} step={step:06d} "
                f"train={eval_metrics['train_accuracy']:.3f} "
                f"test={eval_metrics['test_accuracy']:.3f} "
                f"cyc={final_node['cyclic_defect']:.3f} "
                f"action={final_node['action_defect']:.3f}",
                flush=True,
            )
            if step in snapshot_steps:
                np.savez_compressed(
                    run_dir / f"activations-{step:06d}.npz",
                    node=node_centroids.astype(np.float32),
                    output=output_centroids.astype(np.float32),
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
            prune_checkpoints(run_dir, args.keep_checkpoints)
        if step in dense_checkpoint_steps and step > 0:
            save_dense_checkpoint(
                raw_model,
                run_dir / f"weights-{step:06d}.pt",
                step=step,
                dtype=args.dense_checkpoint_dtype,
            )
        if step == args.steps:
            break

        raw_model.train()
        indices = torch.randint(
            train_pairs.shape[0], (args.batch_size,), device=device
        )
        pairs = train_pairs[indices]
        left, right = pairs[:, 0], pairs[:, 1]
        context_ids = torch.randint(
            args.contexts, (args.batch_size,), device=device
        )
        left_alias = torch.randint(
            args.aliases, (args.batch_size,), device=device
        )
        right_alias = torch.randint(
            args.aliases, (args.batch_size,), device=device
        )
        tokens = build_tokens(
            layout,
            left,
            right,
            contexts=context_ids,
            left_alias=left_alias,
            right_alias=right_alias,
        )
        target = table[left, right]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(tokens)
            loss = F.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

    completion = {
        "run_name": run_name,
        "elapsed_seconds": time.time() - start,
        "final_step": args.steps,
        "metrics_file": str(metrics_path),
    }
    done_path.write_text(json.dumps(completion, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run()
