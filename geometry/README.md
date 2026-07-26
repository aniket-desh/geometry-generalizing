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

`launch_priority_tmux.sh` is the four-GPU launcher for the decisive mixed-horizon
matrix. It assigns one disjoint training shard to each physical GPU in
`0,1,2,3`, and every tmux session receives its own `CUDA_VISIBLE_DEVICES`
restriction before Python starts. The exact matrix is three seeds of the
`grok` and `micro` presets: clean cyclic addition runs to 60k, while 15%
corruption and the random-table control run to 30k. The four shards carry the
same total number of scheduled training steps and cannot duplicate an identity.

The launcher also starts marker-gated operator analysis, four causal shards,
the causal join, final rendering, and packing in separate named tmux sessions.
Operator and causal work share a four-slot filesystem semaphore, so analysis
can use the four devices without launching an unbounded subprocess fleet.
The final marker is withheld until behavior, generator geometry, usable
shared-rule MDL, and causal outputs validate for all 18 runs.
`render_priority.py` writes Nord spaghetti plots with faded seeds and bold
pointwise medians, and `pack_priority.py` freezes and archives the exact
mixed-horizon evidence bundle.

On the four-GPU host, the defaults expect the checkout at
`/home/ubuntu/a/vi-activation-geometry` and put all generated state under
`/home/ubuntu/a/vi-activation-geometry-work`. Every filesystem path has a
`PRIORITY_*` environment override. Validate the plan before launch:

```bash
bash -n geometry/launch_priority_tmux.sh
geometry/launch_priority_tmux.sh --self-test
geometry/launch_priority_tmux.sh --dry-run
geometry/launch_priority_tmux.sh
```

`launch_key60_tmux.sh` remains the older uniform-60k protocol; it should not be
run alongside the mixed-horizon launcher against the same result root.

An optional matched-capacity phase adds the same three seeds and three
conditions for `small` (256×4) and `medium` (512×6) models. Clean runs reach
60k; 15%-corrupted and random controls end at 30k, for exactly 18 additional
runs. Evaluation is every 1k steps, activation snapshots every 5k, and dense
weights every 10k, preserving the 10k/30k/60k evidence while reducing CPU and
disk traffic. Each of the four scale tmux sessions runs one balanced, disjoint
shard and remains restricted to its assigned device. Keep it off when fastest
completion of the central matrix matters, run it concurrently when memory
permits, or gate it on all four key-training markers:

```bash
# Run the scale phase concurrently with the central matrix.
PRIORITY_LAUNCH_SCALE=1 \
  PRIORITY_SCALE_PHASE=parallel \
  geometry/launch_priority_tmux.sh

# Start its tmux sessions now, but train only after the key shards finish.
PRIORITY_LAUNCH_SCALE=1 \
  PRIORITY_SCALE_PHASE=after-key \
  geometry/launch_priority_tmux.sh

# Add only scale training when key jobs already exist.
PRIORITY_LAUNCH_KEY_TRAIN=0 \
  PRIORITY_LAUNCH_ANALYSIS=0 \
  PRIORITY_LAUNCH_SCALE=1 \
  PRIORITY_SCALE_PHASE=parallel \
  geometry/launch_priority_tmux.sh
```

Scale artifacts use their own roots, set independently with
`PRIORITY_SCALE_RESULTS_ROOT` and `PRIORITY_SCALE_LOG_ROOT`. A 5%-corruption
extension is intentionally excluded from this launch plan until the matched
three-seed evidence is complete.

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
