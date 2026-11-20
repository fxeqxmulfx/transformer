module

public import Transformer.ComplexProjection.Defs

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
-/
