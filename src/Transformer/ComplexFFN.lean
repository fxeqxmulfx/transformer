module

public import Transformer.ComplexFFN.Defs
public import Transformer.ComplexFFN.Basic
public import Transformer.ComplexFFN.Embedding
public import Transformer.ComplexFFN.Separation
public import Transformer.ComplexFFN.EvenPart
public import Transformer.ComplexFFN.Incomparable
public import Transformer.ComplexFFN.RMSGap
public import Transformer.ComplexFFN.RMSTwoGate
public import Transformer.ComplexFFN.RMSIncomparable
public import Transformer.ComplexFFN.RMSSurviving
public import Transformer.ComplexFFN.RMSErased

/-!
# Complex-product FFNs against the bias-free SwiGLU FFN

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`): the selected model of
`research/main_model/README.txt`, the three products of
`research/complex_transformer/README.md` (kernels `plain_product.rs`, `complex.rs`)
and the SwiGLU control of `feed_forward.py`. Review note: `review/complex_ffn.md`.

The question was whether the selected plain complex product `W_F (W_g z ⊙_ℂ W_u z)/√2`
approximates better than SiLU. As a statement about function classes it is false:

* every plain FFN with `h` complex units is a bias-free SwiGLU FFN with `6h` units
  (`plainFFN_eq_swigluFFN`) and a limit of SwiGLU FFNs with `3h` units
  (`swigluFFN_gauss_sub_plainFFN`);
* all three complex products are even (`plainFFN_neg`, `radialFFN_neg`,
  `diagonalFFN_neg`), so none of them, at any width, approximates even one SwiGLU
  unit better than half its odd part (`plainFFN_gap_swigluUnit` and its variants);
* with complex-linear projections plain and radial FFNs are negated by `z ↦ i z`
  (`plainFFN_anti`, `radialFFN_anti`); the four-unit SwiGLU block `|z₁|²` is then
  approximated no better than by zero (`plainFFN_gap_normSq`).

The gated products are not inside SwiGLU either. The even part of a bias-free SwiGLU
FFN is a quadratic form and obeys the parallelogram law (`swigluFFN_gap_even`), which
one radial or diagonal unit breaks (`swigluFFN_gap_radial`, `swigluFFN_gap_diagonal`):
those two classes and SwiGLU are incomparable. At equal parameters the plain class is
not dominated: one SwiGLU unit misses one plain unit `z₁ z₂` on the torus by as much as
the zero map (`swigluUnit_gap_plainUnit_torus`).

These are statements about one block before its branch RMS, on a set of inputs closed
under the symmetry, such as the RMS sphere that feeds it. The branch RMS is odd
(`rmsNormEps_neg`), so the symmetries survive it: after it, plain and radial FFNs with
complex-linear projections still miss the four-unit block `|φ z|² w` by as much as the
zero map (`plainFFN_rms_gap_normSq`, `radialFFN_rms_gap_normSq`). One SwiGLU unit after
it stays on a line, so it still misses one plain unit on the torus by as much as the
zero map (`swigluUnit_rms_gap_plainUnit_torus`). But with `eps = 0` the RMS erases the
one-unit witnesses of evenness and of the parallelogram law: one SwiGLU unit after it is
even (`rmsNormEps_zero_swigluUnit_neg`), and a positive gate is a positive scalar, so one
radial or diagonal unit after it is its ungated product, a SwiGLU block
(`rmsNormEps_zero_radialFFN_unit_swiglu`, `rmsNormEps_zero_diagonalFFN_unit_swiglu`); with
`eps > 0` it does not forget scale (`norm_rmsNormEps_lt_smul`). It keeps the sign of
every linear functional, and through signs the even part of SwiGLU still obeys a
four-point identity (`swigluFFN_rms_gap`). Two units on each side restore both
separations after the RMS (`RMSIncomparable`): no SwiGLU FFN approximates two radial or
diagonal units that share one product under two gates (`swigluFFN_rms_gap_radial`,
`swigluFFN_rms_gap_diagonal`), and no even map approximates two SwiGLU units whose
coefficients swap under `z ↦ -z` (`even_rms_gap_swigluPair`).

None of this says anything about training, depth or the cubic task.
-/
