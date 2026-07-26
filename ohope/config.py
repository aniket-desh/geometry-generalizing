from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hub_id: str
    family: str
    slug: str


MODEL_SPECS = {
    spec.key: spec
    for spec in (
        ModelSpec("gpt2", "openai-community/gpt2", "gpt2", "gpt2"),
        ModelSpec("gpt2-medium", "openai-community/gpt2-medium", "gpt2", "gpt2-medium"),
        ModelSpec("gpt2-large", "openai-community/gpt2-large", "gpt2", "gpt2-large"),
        ModelSpec("gpt2-xl", "openai-community/gpt2-xl", "gpt2", "gpt2-xl"),
        ModelSpec("pythia-160m", "EleutherAI/pythia-160m", "pythia", "pythia-160m"),
        ModelSpec("pythia-410m", "EleutherAI/pythia-410m", "pythia", "pythia-410m"),
        ModelSpec("pythia-1b", "EleutherAI/pythia-1b", "pythia", "pythia-1b"),
        ModelSpec("pythia-2.8b", "EleutherAI/pythia-2.8b", "pythia", "pythia-2.8b"),
        ModelSpec("pythia-6.9b", "EleutherAI/pythia-6.9b", "pythia", "pythia-6.9b"),
    )
}


CALIBRATION_SEEDS = (0, 1, 2)
RANDOM_PRUNING_SEEDS = (0, 1, 2, 3, 4)
ABLATION_SAMPLE_SEEDS = (0, 1, 2)
PRUNE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)

SCORE_LABELS = {
    "random": "random",
    "outgoing_norm": "outgoing norm",
    "activation_rms": "activation RMS",
    "hope": "HOPE",
    "ohope": "O-HOPE",
    "fisher": "Fisher O-HOPE",
}

NORD = {
    "ink": "#2e3440",
    "dark": "#3b4252",
    "muted": "#4c566a",
    "pale": "#e5e9f0",
    "blue": "#5e81ac",
    "cyan": "#88c0d0",
    "green": "#a3be8c",
    "yellow": "#ebcb8b",
    "orange": "#d08770",
    "red": "#bf616a",
    "purple": "#b48ead",
    "white": "#ffffff",
}

SCORE_COLORS = {
    "random": NORD["muted"],
    "outgoing_norm": NORD["blue"],
    "activation_rms": NORD["cyan"],
    "hope": NORD["purple"],
    "ohope": NORD["red"],
    "fisher": NORD["green"],
}
