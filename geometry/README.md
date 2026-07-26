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

`render_breadth_replication.py` is the separate finalizer for the restricted
four-task confirmation. It reads the preserved seed-0 `.tar.gz` directly,
combines it with seeds 1 and 2 from the replication result root, and rejects
anything other than the exact 4-task × 3-seed matrix. It also verifies the
semantic protocol hash, every explicit seed field, and each operation table
against its seeded task generator. The figure shows faint measured runs and a
bold pointwise median only where all three seeds have an observed checkpoint:

```bash
python geometry/render_breadth_replication.py \
  --seed0-archive /path/to/vi-breadth60.tar.gz \
  --replication-root /path/to/breadth-replication-results \
  --output /path/to/breadth-replication-figures \
  --wait

python geometry/render_breadth_replication.py --self-test
```

`launch_priority_tmux.sh` is the four-GPU launcher for the decisive mixed-horizon
matrix. It assigns one disjoint training shard to each physical GPU in
`0,1,2,3`, and every tmux session receives its own `CUDA_VISIBLE_DEVICES`
restriction before Python starts. The exact matrix is three seeds of the
`grok` and `micro` presets: clean cyclic addition runs to 60k, while 15%
corruption and the random-table control run to 30k. The four shards carry the
same total number of scheduled training steps and cannot duplicate an identity.

The launcher also starts marker-gated operator analysis, four causal shards,
the causal join, and final rendering in separate named tmux sessions.
Operator and causal work share a four-slot filesystem semaphore, so analysis
can use the four devices without launching an unbounded subprocess fleet.
Every operator fold fits its centering, PCA basis, rank, and scale using only
the training-state × training-alias cross-product. Held-out activations are
projected afterward, and the reusable action is scored as one affine-orthogonal
map, so neither the representation basis nor the transition fit sees test
vocabulary.

The primary cross-condition probe is preregistered as the canonical latent-label
cycle `k → (k + 1) mod n` before activations are inspected. The labels are
planted by the benchmark while surface token IDs remain arbitrary, so the
orientation is never selected post hoc. Both the inductive operator fit and
the causal counterfactual use that exact vector for clean, corrupted, and
random conditions; table-relation analyses are secondary diagnostics. Causal
transport is tested at the input node before the first block and at the output
residual stream after the final block.

The final marker is withheld until behavior, canonical-cycle geometry, usable
shared-rule MDL, and causal outputs validate for all 18 runs.
`render_priority.py` writes Nord spaghetti plots with faded seeds and bold
pointwise medians. Packing is independently disabled by default. Enable it
with `PRIORITY_LAUNCH_PACK=1`; the packer then stays local by default and
finishes only after validating the archive, its SHA-256 sidecar, and every raw
chunk against a byte-for-byte reconstruction of the archive. The completion
marker and chunk manifest use `delivery: "local-only"` and contain no URLs, so
the archive directory can be copied directly from the host.

External upload requires a second, explicit switch:

```bash
# Checksummed local archive only; no network upload.
PRIORITY_LAUNCH_KEY_TRAIN=0 \
  PRIORITY_LAUNCH_ANALYSIS=0 \
  PRIORITY_LAUNCH_PACK=1 \
  geometry/launch_priority_tmux.sh

# Upload wrapped chunks only when this is intentionally requested.
PRIORITY_LAUNCH_KEY_TRAIN=0 \
  PRIORITY_LAUNCH_ANALYSIS=0 \
  PRIORITY_LAUNCH_PACK=1 \
  PRIORITY_UPLOAD=1 \
  PRIORITY_UPLOAD_ENDPOINT=https://temp.sh/upload \
  geometry/launch_priority_tmux.sh
```

The corresponding direct CLI flags are `--local-only` (the default) and
`--upload`. Merely setting `--upload-endpoint` or `PRIORITY_UPLOAD_ENDPOINT`
does not enable an upload.

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

After a core or scale suite and its endpoint analyses complete,
`summarize_priority_evidence.py` performs a stricter, renderer-independent
audit and writes both JSON and Markdown. It rejects any matrix other than the
exact 18 identities and mixed horizons, verifies the preregistered successor
vector and hash in every operator and causal output, and reports each seed's
first measured 90% crossing, peak and final held-out accuracy, post-peak
drawdown, canonical-label agreement on its exact held-out split, inductive
operator error and alias-held-out code gain, and causal success at node 0 and
output final with matched control maxima.

```bash
python geometry/summarize_priority_evidence.py \
  --results-root /path/to/core-results \
  --suite core \
  --output-json /path/to/core-evidence.json \
  --output-markdown /path/to/core-evidence.md

python geometry/summarize_priority_evidence.py --self-test
```

Analysis JSON and sidecars normally live inside each run directory. If they
were downloaded separately, pass `--analysis-root`; it must contain one
subdirectory per run, named either by the training `run_name` or by the stable
`condition-preset-seed` slug.

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
  PRIORITY_LAUNCH_SCALE_ANALYSIS=0 \
  PRIORITY_SCALE_PHASE=parallel \
  geometry/launch_priority_tmux.sh

# Add endpoint analysis after the key and scale training sessions exist.
PRIORITY_LAUNCH_KEY_TRAIN=0 \
  PRIORITY_LAUNCH_ANALYSIS=1 \
  PRIORITY_LAUNCH_PACK=0 \
  PRIORITY_LAUNCH_SCALE_TRAIN=0 \
  PRIORITY_LAUNCH_SCALE_ANALYSIS=1 \
  geometry/launch_priority_tmux.sh
```

Scale artifacts use their own roots, set independently with
`PRIORITY_SCALE_RESULTS_ROOT`, `PRIORITY_SCALE_LOG_ROOT`, and
`PRIORITY_SCALE_FIGURE_ROOT`. Scale and key operator/causal processes share the
key analysis-slot directory, so together they never exceed the configured
analysis concurrency. A 5%-corruption extension is intentionally excluded from
this launch plan until the matched three-seed evidence is complete.

The isolated H100 extension adds only the existing `large` preset (768×8):
three seeds each for clean 60k, 15%-corrupted 30k, and random-table 30k. It
uses batch size 1024, evaluates every 1k steps, saves activation snapshots every
5k, and writes float16 dense weights every 10k. Its nine runs are assigned to
four explicit, disjoint 90k-step shards:

```bash
# Audit the exact matrix and all tmux commands without starting anything.
LARGE_LAUNCH_TRAIN=1 LARGE_LAUNCH_ANALYSIS=0 \
  geometry/launch_large_tmux.sh --dry-run

# Start the four training shards, restricted to physical GPUs 0-3.
LARGE_LAUNCH_TRAIN=1 LARGE_LAUNCH_ANALYSIS=0 \
  geometry/launch_large_tmux.sh

# After training, start operator, causal, and rendering sessions.
LARGE_LAUNCH_TRAIN=0 LARGE_LAUNCH_ANALYSIS=1 \
  geometry/launch_large_tmux.sh
```

The launcher defaults to a 40 GiB free-space guard and resumable 30k-step
optimizer checkpoints. Large outputs use their own work root,
`/home/ubuntu/a/vi-activation-geometry-large-work`, and never share result or
log directories with the completed core or scale suites. Summarize the exact
nine-run suite with `--suite large`. To render or summarize a unified
small/medium/large comparison without copying raw data, run the capacity
finalizer after both endpoint analyses complete:

```bash
python geometry/finalize_capacity.py \
  --scale-results-root /path/to/exact-scale18-results \
  --large-results-root /path/to/exact-large9-results \
  --output-root /path/to/capacity-final
```

It reuses the unchanged scale and large validators, creates a collision-checked
union of 27 directory symlinks, renders the Nord capacity figures, and writes
`capacity-evidence.json` plus `capacity-evidence.md`. Rerunning the same command
is safe: it revalidates both suites, preserves matching links, and overwrites
the derived figures and summaries. It rejects incomplete endpoint analysis,
extra or duplicate run configs, stale symlinks, and input/output path overlap.

The command is tmux-friendly because it runs in the foreground and logs both
subcommands under its output root:

```bash
tmux new-session -d -s vi-capacity-finalize \
  "cd /path/to/repo && python geometry/finalize_capacity.py \
  --scale-results-root /path/to/exact-scale18-results \
  --large-results-root /path/to/exact-large9-results \
  --output-root /path/to/capacity-final"

python geometry/finalize_capacity.py --self-test
```

`render_model_spectrum.py` makes the compact endpoint comparison used in the
post. It accepts only an exact core18, scale18, and large9 suite, then plots the
15%-corrupted 30k endpoints across all five presets. Faint lines connect the
same seed across models and the bold line is the pointwise median. The three
panels separate held-out behavior, alias-held-out shared-code gain, and node-0
canonical-cycle transport. The validator checks every model width and depth,
the preregistered successor, matching operation-table and held-out-mask hashes,
the exact suite roots, and the bounded causal metric policy before it writes a
figure. The presets change width and depth together, and use batch sizes 4096,
4096, 4096, 2048, and 1024, so this is a descriptive preset comparison rather
than an isolated capacity scaling law. Causal folds use qualified accuracy when
at least 64 examples qualify and bounded absolute accuracy otherwise; the output
records each fold's metric key because those populations are not homogeneous.

```bash
python geometry/render_model_spectrum.py \
  --core-root /path/to/exact-core18-results \
  --scale-root /path/to/exact-scale18-results \
  --large-root /path/to/exact-large9-results \
  --output /path/to/model-spectrum

# Audit suites already present without producing a partial figure.
python geometry/render_model_spectrum.py \
  --core-root /path/to/exact-core18-results \
  --scale-root /path/to/exact-scale18-results \
  --validate-only

python geometry/render_model_spectrum.py --self-test
```

`render_measured_geometry_gif.py` renders only measured activation-centroid
checkpoints. It fits one unsupervised PCA basis at the final saved checkpoint,
applies that basis unchanged to every frame, uses one fixed scale, and never
interpolates point positions or aligns frames with Procrustes. With
`--view node --layer 0`, the points are learned input-state token embeddings
before the first transformer block, not context-dependent hidden states. The
manifest reports how much centered between-state centroid energy lies in the
displayed plane, explicitly excluding within-state activation variance, and
identifies the best cyclic Fourier harmonic in that plane.

```bash
python geometry/render_measured_geometry_gif.py \
  --run /path/to/clean-cycle-run \
  --output /path/to/measured-geometry.gif \
  --view node \
  --layer 0

python geometry/render_measured_geometry_gif.py --self-test
```

Automatic summary suite detection deliberately rejects the 27-link union
because it contains multiple complete sub-suites; the finalizer selects
`--suite capacity` explicitly.

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
