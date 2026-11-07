module

public import Transformer.Rotate90.ComplexProduct.RecoveryWitness

/-!
# A flat-spectrum bridge to the complex-product Rotate90 source model

Source: `review/flat_spectrum.md`, *Complex-product Rotate90 bridge*, CP1–CP5;
Rust/Kernite experiment `grokking/experiments/rotate90_complex`, b70811e.

`realLogit_eq_logit` proves the independent real expansion equals the complex
formulation, including every normalization and the tied embedding/readout.
`realLogit_correct_of_feature_error` transfers separating code margins to
approximate features of that actual source model. `executed_correct_of_raw_margin`
accounts for final RMSNorm and bounded numerical logit error.

`executed_correct_of_nonunique_minimum` connects the rank-r spectral certificate
to correct executed labels, retaining source alignment as an explicit premise.
It needs preservation of score contrasts, not unique matrix recovery.
`copy_nonunique_bridge` supplies two distinct optimal matrices and a nonconstant
normalized source classifier satisfying the premises. The witness uses the
residual path; nonzero product parameters instantiate the algebraic identities.

These are exact-arithmetic and conditional decision theorems. They do not prove
that AdamW finds a spectral optimum, identify the normalized network's weight
decay with nuclear norm, or certify the learned cubic-task checkpoints.
-/
