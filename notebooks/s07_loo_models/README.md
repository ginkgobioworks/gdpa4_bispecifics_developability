# s07 — Leave-one-bispecific-out predictive models

Per-N3 leave-one-out CV view produced by `build_05_cv`
(D-2026-04-30-CV-LOO-PARENT-DISJOINT). For each held-out N3 (parents A, B),
training is restricted to N3s with neither parent in {A, B}; every one of
the 160 bispecifics receives a held-out prediction.

This section is the LOO counterpart to s05 (which renders the parent-aware
5-fold sweep). Same labels, configs, and 7 model families; different fold
geometry. The final figure (`s07_loo_vs_kfold_scatter.png`) compares the
two CV strategies head-to-head across all (label × config × model) combos.

See `manifest.yaml` for inputs and outputs.
