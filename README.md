# Observable HOPE experiments

This directory contains the reproducible experiment harness for blog post VI.
It compares ordinary empirical HOPE with a centered-unembedding-weighted
variant on the final MLP of GPT-2 and Pythia models.

The expensive run is designed around a cached final-layer intervention. Each
model is passed over the calibration and evaluation corpora once. Later
ablations subtract \(a_i(x)v_i\) from the cached pre-normalization residual and
run only the final normalization and unembedding.

## RunPod layout

The remote runner expects:

```text
/workspace/ohope/
├── .venv/
├── code/
├── logs/
├── results/
└── hf-cache/
```

All long commands run in named tmux sessions and stream to files under
`/workspace/ohope/logs`.

## Stages

The default matrix runs:

- GPT-2 small, medium, large, and XL;
- Pythia 160M, 410M, 1B, 2.8B, and 6.9B;
- three disjoint 50K-token calibration windows;
- one disjoint 100K-token evaluation stream;
- five importance rankings plus random baselines;
- structured pruning at 10, 20, 30, 40, 50, and 60 percent;
- three stratified single-neuron samples, each tested against the full 100K-token
  evaluation stream;
- the optional Fisher pullback after the static readout experiments.

Every stage writes a completion artifact and can be resumed safely.

## Entry points

```bash
python run_model.py --model gpt2 --output-root /workspace/ohope/results
bash run_matrix.sh
bash run_matrix.sh gpt2-xl pythia-2.8b pythia-6.9b
python render_aggregate.py --results /workspace/ohope/results
python summarize_results.py --results /workspace/ohope/results
```

The plots use the Nord palette. Repeated calibration and random seeds appear as
faint spaghetti lines, with their pointwise median drawn more heavily. No
confidence bands are used.
