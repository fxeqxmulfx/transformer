module

public import Transformer.ComplexProjection.Defs
public import Transformer.ComplexProjection.Basic

/-!
# Complex-linear projections against real ones

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`): the complex-linear projections
`[[A,-B],[B,A]]` of the selected model (`research/main_model/README.txt`,
`phase_score_native/src/layout.rs`) and the remark of
`research/complex_transformer/README.md` that a general real projection is
`Wz + V conjugate(z)`.

The question was whether a complex projection approximates better than a real one.
At the same width it is a real projection whose conjugate-linear part vanishes: every
real projection splits uniquely into a complex-linear part and a conjugate-linear part
(`complexPart_add_antilinearPart`, `complexPart_unique`), and the storage
`[[A,-B],[B,A]]` is exactly the set of real matrices commuting with the complex
structure (`fromBlocks_comm_iff`).

The complex class loses exactly that part. A map commuting with the complex structure,
linear or not, errs on `L` over a set closed under `J` by at least the conjugate-linear
part of `L`, and the complex part of `L` errs by exactly that
(`exists_comm_approx_iff`); conjugation on the unit circle is missed by `1`, as by the
zero map (`one_le_of_approx_conj`). The branch RMS commutes with the complex structure
(`rmsNormEps_comm`), so a conjugate-linear target is missed by its full norm after it
too (`norm_rmsNormEps_le`).
-/
