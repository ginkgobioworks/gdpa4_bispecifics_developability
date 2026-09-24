# s03 — Compositional baselines

For every (N3 bispecific, value_col, condition), compute the predicted value
under each operator in `prophet_ab.features.compositional.OPERATORS`
applied to the two parent monospecifics' medians, and compare to the N3's
own measured median. Report Spearman ρ and Pearson R² per
(value_col, condition, operator).

Sets the "trivial baseline" against which any sequence-aware model in s04+
must improve.

See `manifest.yaml` for inputs and outputs.
