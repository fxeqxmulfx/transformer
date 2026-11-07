module

public import Transformer.Rotate90.ComplexProduct.RecoveryWitness
public import Transformer.Rotate90.ComplexProduct.ApproximateWitness
public import Transformer.Rotate90.ComplexProduct.AnswerContrast
public import Transformer.Rotate90.ComplexProduct.TrainingTransport
public import Transformer.Rotate90.ComplexProduct.ProbabilityTransfer

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

`executed_correct_of_approximate_recovery` permits noisy fitting, a finite
objective gap and approximate contrast alignment. Its error budget retains the
actual final normalization. A nonzero perturbation of the nonunique witness
instantiates all bounds without exact optimality of the perturbed matrix.

`executed_correct_of_answer_mass` preserves each competitor's own coefficient
range and ideal gap. It uses only pooled answer error, with source alignment
and numerical execution error accounted for separately.

`executed_correct_of_training_transport` works directly with the actual tied
readout and training-state contrasts. It replaces fitted-decoder alignment with
an explicit geometric reconstruction bound, including unseen state directions.

These are exact-arithmetic and conditional decision theorems. They do not prove
that AdamW finds a spectral optimum, identify the normalized network's weight
decay with nuclear norm, or certify the learned cubic-task checkpoints.
-/
