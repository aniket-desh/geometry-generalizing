from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    width: int
    depth: int
    heads: int
    mlp_ratio: int = 4


PRESETS = {
    "grok": ModelConfig(width=128, depth=1, heads=4),
    "micro": ModelConfig(width=128, depth=2, heads=4),
    "small": ModelConfig(width=256, depth=4, heads=8),
    "medium": ModelConfig(width=512, depth=6, heads=8),
    "large": ModelConfig(width=768, depth=8, heads=12),
}


class GeometryTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        output_classes: int,
        config: ModelConfig,
        sequence_length: int = 5,
    ) -> None:
        super().__init__()
        self.config = config
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(vocab_size, config.width)
        self.position_embedding = nn.Parameter(
            torch.empty(sequence_length, config.width)
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.width,
                    nhead=config.heads,
                    dim_feedforward=config.width * config.mlp_ratio,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, output_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)
        nn.init.normal_(self.output.weight, std=0.02)
        for block in self.blocks:
            for name, parameter in block.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif name.endswith("weight"):
                    nn.init.ones_(parameter)
                else:
                    nn.init.zeros_(parameter)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_states: bool = False,
        all_positions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        sequence_length = tokens.shape[1]
        if sequence_length > self.sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.sequence_length}"
            )
        x = (
            self.token_embedding(tokens)
            + self.position_embedding[None, :sequence_length, :]
        )
        states = [x] if return_states else []
        mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=x.device,
                dtype=x.dtype,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)
            if return_states:
                states.append(x)
        x = self.final_norm(x)
        logits = self.output(x if all_positions else x[:, -1])
        if return_states:
            states[-1] = x
            return logits, states
        return logits
