# Results from the July 2026 run

The production fleet covered GPT-2 small through XL and Pythia 160M through
6.9B. Every model used three disjoint 50K-token calibration windows, a
100K-token held-out stream for structured pruning, and the full same-size
held-out stream for exact single-neuron ablations. Each single-neuron sample
contained 376 to 383 unique neurons after pooling the three stratified sample
seeds.

All nine runs passed the artifact audit. Every geometry tensor, 96-condition
pruning table, full ablation tensor, outlier record, static figure, and
animation is present and finite.

## Main result

The centered-unembedding correction is useful in some settings, but it does not
uniformly improve the original HOPE score.

Ordinary HOPE is the best single-neuron KL ranker on eight of nine models.
Fisher O-HOPE wins only on Pythia-410M. Structured pruning is less uniform:
O-HOPE wins the two smallest models at 50 percent pruning, HOPE wins four
mid-sized models, outgoing norm wins GPT-2 large and XL, and Fisher O-HOPE wins
Pythia-6.9B.

| model | best 50% pruning method | KL | best single-neuron method | Spearman with KL |
|---|---:|---:|---:|---:|
| GPT-2 | O-HOPE | 0.085046 | HOPE | 0.765788 |
| GPT-2 medium | HOPE | 0.043155 | HOPE | 0.819070 |
| GPT-2 large | outgoing norm | 0.013270 | HOPE | 0.668668 |
| GPT-2 XL | outgoing norm | 0.008787 | HOPE | 0.684474 |
| Pythia-160M | O-HOPE | 0.889643 | HOPE | 0.881463 |
| Pythia-410M | HOPE | 0.111565 | Fisher O-HOPE | 0.769140 |
| Pythia-1B | HOPE | 0.120828 | HOPE | 0.728167 |
| Pythia-2.8B | HOPE | 0.046374 | HOPE | 0.725741 |
| Pythia-6.9B | Fisher O-HOPE | 0.049147 | HOPE | 0.867370 |

Across all pruning fractions, the median relative KL advantage of O-HOPE over
HOPE changes sign across models. Its association with the observed spread in
neuron visibility ratios is positive but weak and statistically unresolved
across nine models (Spearman rho 0.35, p 0.36). The larger models therefore make
the distinction between local causal importance and collective removability
more obvious, rather than making O-HOPE a universal replacement for HOPE.

## Artifacts

The lightweight result bundle excludes only corpus blocks and cached residual
states:

```text
/workspace/ohope/artifacts/ohope-light-results.tar.gz
SHA-256 201873427471fc612f6d89ad549324016e4345df51727cb06b86d56a3a015e4f
```

The full result tree and model caches remain on the persistent RunPod volume.
At completion the 100GB volume was 54 percent used.
