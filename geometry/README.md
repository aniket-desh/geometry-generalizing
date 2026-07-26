# Geometry of generalization

These experiments follow models across training as they replace partially
observed lookup tables with reusable relational structure. The central example
is cyclic addition, but the benchmark also includes tori, XOR groups, dihedral
groups, bounded paths, trees, broken cycles, and random permutation controls.

Each latent state and relation has several arbitrary surface tokens, and every
query is wrapped in a nuisance context. Train and test sets split the operation
table itself, so a model generalizes only when it predicts unseen combinations
of familiar states and relations.

The recorded evidence is deliberately split into three levels:

1. held-out behavioral accuracy;
2. full-space geometry, including action-invariance and cyclic Gram defects;
3. reusable operators, including cross-validated shift error and closure.

There are three complementary protocols:

1. `train.py` holds out entries from an operation table and tests whether a
   transformer learns the shared algebra rather than the observed entries.
2. `hmm_train.py` emits sequences from a latent relational system through many
   arbitrary surface aliases. It holds out state-action pairs, so the model must
   infer both which aliases name the same latent state and how the shared action
   moves between states. The main sweep predicts the canonical latent state;
   the harder `--target-mode alias` control also asks it to distribute
   probability over an unpredictable output alias.
3. `pythia_geometry.py` samples real Pythia pretraining checkpoints and tracks
   natural cyclic domains such as weekdays, months, seasons, compass directions,
   clock hours, and digits.

`pythia_weekday_movie.py` extracts the four weekday prompt contexts at a fixed
Pythia layer and projects every checkpoint through a phase decoder fitted only
at the final checkpoint. It is a developmental illustration, while the
full-space metrics remain the evidence.

`render_breadth.py --wait` validates that the nine seed-0 breadth runs have all
reached 60,000 steps before writing a behavior small-multiple and a separate
geometry companion to `/workspace/geometry-breadth-figures`. Since this pilot
has one seed per task, both figures show measured checkpoints directly and do
not draw a median.

`launch_key60_tmux.sh` runs the isolated decisive matrix without changing the
broader 64-run queue. It requires exactly 18 runs: three seeds, the `grok` and
`micro` presets, and clean cyclic addition, 15% corrupted cyclic addition, and
a random operation table. The operator analysis measures checkpoints 10k, 30k,
and 60k. Causal transport is tested at the final output residual stream with
three folds and five controls for all 18 runs at 60k; the minimal developmental
comparison adds 10k and 30k for seed 0 of the `grok` preset in all three
conditions. The final marker is withheld until behavior, generator geometry,
usable shared-rule MDL, and causal outputs validate for the full matrix.
`render_key60.py` then writes Nord spaghetti plots with faded seeds and bold
pointwise medians, and `pack_key60.py` archives the exact selected run
directories, logs, and figures.

`reproduce_engels_geometry.py` reruns the GPT-2 small layer-7 SAE feature
clusters reported by Engels et al. on the Pile, retaining only the weekday and
twentieth-century-year manifolds and rendering them in the visual style used by
the post.

`summarize.py` aggregates controlled runs, `posthoc_layers.py` reconstructs
layerwise full-space metrics from saved snapshots, `derive_trajectory.py`
computes the lightweight compression traces used in the main spaghetti plot,
and `summarize_pythia.py` extracts the best layerwise geometry and behavior from
every natural-domain checkpoint.

PCA is saved only for illustration. It is not evidence that a manifold exists.

## Quick start

```bash
python geometry/train.py \
  --task cycle7 \
  --preset small \
  --seed 0 \
  --steps 30000 \
  --output-root geometry-results

python geometry/hmm_train.py \
  --task cycle7 \
  --preset small \
  --aliases 16 \
  --output-root hmm-geometry-results

python geometry/pythia_geometry.py \
  --model EleutherAI/pythia-70m \
  --cache-dir pythia-cache \
  --output-root pythia-results
```

Long RunPod jobs are launched through `geometry/launch_sweep.py` inside a named
tmux session. The latent-sequence breadth and scale matrices use
`geometry/launch_hmm.py`. Every worker has its own log and writes completion
artifacts.
