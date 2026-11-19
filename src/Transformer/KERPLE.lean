module

public import Transformer.KERPLE.Section3_Kernels
public import Transformer.KERPLE.Section3_Facts

/-!
# KERPLE

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, "KERPLE: Kernelized Relative Positional Embedding
for Length Extrapolation" (NeurIPS 2022). The TeX source is in `papers/arXiv-2205.09921v2/`.

A kernel on `X` is a function `X → X → ℝ`; PD and CPD kernels are Definitions 1 and 2.

* `Section3_Kernels`: §3.1, Definitions 1 and 2, Definition 1 as Mathlib's positive
  semidefiniteness, and "a PD kernel offset by a constant is CPD".
* `Section3_Facts`: §3.2, Fact 2 (CPD kernels are closed under nonnegative scaling and sums) and
  Fact 4 (`−‖x − x'‖²` is CPD on every real inner product space); the two-point inequalities of PD
  kernels and of CPD kernels vanishing on the diagonal.
-/
