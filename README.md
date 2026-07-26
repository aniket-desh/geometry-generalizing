# Activation geometry and generalization

This repository contains the reproducible experiment harness for blog post VI.
It follows models through dense training checkpoints as partially observed
lookup tables become, or fail to become, reusable relational computations.

The experiment and RunPod instructions live in
[`geometry/README.md`](geometry/README.md).

All long remote runs execute in named tmux sessions, write individual logs, and
leave completion artifacts that can be audited after the pod disconnects.

Plots use the Nord palette. Multiple trials appear as faint spaghetti lines,
with their pointwise median drawn more heavily. Confidence bands are not used.
