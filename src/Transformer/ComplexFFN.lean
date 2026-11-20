module

public import Transformer.ComplexFFN.Defs
public import Transformer.ComplexFFN.Basic
public import Transformer.ComplexFFN.Embedding
public import Transformer.ComplexFFN.Separation

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

These are statements about one block on a set of inputs closed under the symmetry,
such as the RMS sphere that feeds it; the branch RMS keeps the symmetries
(`rmsNormEps_neg`). They say nothing about training, depth or the cubic task.
-/
