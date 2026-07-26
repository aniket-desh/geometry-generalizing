from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelSpec


@dataclass
class CapturedBatch:
    pre_final_norm: torch.Tensor
    post_activations: torch.Tensor


class FinalMLPAdapter:
    """Expose the final-MLP intervention point for GPT-2 and GPT-NeoX models."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, spec: ModelSpec):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self._capture: dict[str, torch.Tensor] = {}

        if spec.family == "gpt2":
            self.backbone = model.transformer
            self.projection = model.transformer.h[-1].mlp.c_proj
            self.final_norm = model.transformer.ln_f
            weight = self.projection.weight
            if weight.shape[1] != model.config.n_embd:
                raise ValueError(f"unexpected GPT-2 c_proj shape: {tuple(weight.shape)}")
            self._w_out = weight
        elif spec.family == "pythia":
            self.backbone = model.gpt_neox
            self.projection = model.gpt_neox.layers[-1].mlp.dense_4h_to_h
            self.final_norm = model.gpt_neox.final_layer_norm
            weight = self.projection.weight
            if weight.shape[0] != model.config.hidden_size:
                raise ValueError(f"unexpected Pythia output projection shape: {tuple(weight.shape)}")
            self._w_out = weight.transpose(0, 1)
        else:
            raise ValueError(f"unsupported model family: {spec.family}")

        self.lm_head = model.get_output_embeddings()
        if self.lm_head is None or not hasattr(self.lm_head, "weight"):
            raise ValueError("model has no linear output embedding")

        self._handles = (
            self.projection.register_forward_pre_hook(self._capture_activations),
            self.final_norm.register_forward_pre_hook(self._capture_residual),
        )

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        *,
        device: torch.device,
        cache_dir: str,
        revision: str | None = None,
    ) -> "FinalMLPAdapter":
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hub_id,
            revision=revision,
            cache_dir=cache_dir,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            spec.hub_id,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device)
        return cls(model, tokenizer, spec)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def d_model(self) -> int:
        return int(self.w_out.shape[1])

    @property
    def d_mlp(self) -> int:
        return int(self.w_out.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.w_u.shape[1])

    @property
    def w_out(self) -> torch.Tensor:
        return self._w_out.detach()

    @property
    def w_u(self) -> torch.Tensor:
        return self.lm_head.weight.detach().transpose(0, 1)

    @property
    def norm_eps(self) -> float:
        return float(getattr(self.final_norm, "eps", 1e-5))

    def _capture_activations(
        self,
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        self._capture["post_activations"] = inputs[0].detach()

    def _capture_residual(
        self,
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        self._capture["pre_final_norm"] = inputs[0].detach()

    @torch.inference_mode()
    def capture(self, input_ids: torch.Tensor) -> CapturedBatch:
        self._capture.clear()
        self.backbone(input_ids=input_ids, use_cache=False, return_dict=False)
        missing = {"post_activations", "pre_final_norm"} - self._capture.keys()
        if missing:
            raise RuntimeError(f"hooks did not capture: {sorted(missing)}")
        return CapturedBatch(
            pre_final_norm=self._capture["pre_final_norm"],
            post_activations=self._capture["post_activations"],
        )

    @torch.inference_mode()
    def logits_from_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(residual))

    @torch.inference_mode()
    def centered_logit_rms(self, vectors: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        pieces = []
        w_u = self.w_u.to(dtype=vectors.dtype)
        for start in range(0, vectors.shape[0], chunk_size):
            logits = vectors[start : start + chunk_size] @ w_u
            logits = logits - logits.mean(dim=-1, keepdim=True)
            pieces.append(logits.square().mean(dim=-1).sqrt().cpu())
        return torch.cat(pieces)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_key": self.spec.key,
            "hub_id": self.spec.hub_id,
            "family": self.spec.family,
            "d_model": self.d_model,
            "d_mlp": self.d_mlp,
            "vocab_size": self.vocab_size,
            "final_norm": type(self.final_norm).__name__,
            "norm_eps": self.norm_eps,
            "dtype": str(next(self.model.parameters()).dtype),
        }
