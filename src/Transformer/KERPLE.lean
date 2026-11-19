module

public import Transformer.KERPLE.Section3_Kernels
public import Transformer.KERPLE.Section3_Embedding
public import Transformer.KERPLE.Section3_Facts
public import Transformer.KERPLE.Section3_Schoenberg
public import Transformer.KERPLE.Section3_Integral
public import Transformer.KERPLE.Section3_Powers
public import Transformer.KERPLE.SectionA_ShiftLemma

/-!
# KERPLE

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, "KERPLE: Kernelized Relative Positional Embedding
for Length Extrapolation" (NeurIPS 2022). The TeX source is in `papers/arXiv-2205.09921v2/`.

A kernel on `X` is a function `X → X → ℝ`; PD and CPD kernels are Definitions 1 and 2.

* `Section3_Kernels`: §3.1, Definitions 1 and 2, Definition 1 as Mathlib's positive
  semidefiniteness, and "a PD kernel offset by a constant is CPD".
* `Section3_Embedding`: §3.1, PD kernels are inner products of a feature map into a Hilbert space,
  and Fact 1: a CPD kernel vanishing on the diagonal is `−‖φ(x) − φ(x')‖²` for a map `φ` into a
  Hilbert space; conversely such a kernel is CPD.
* `Section3_Facts`: §3.2, Fact 2 (CPD kernels are closed under nonnegative scaling and sums) and
  Fact 4 (`−‖x − x'‖²` is CPD on every real inner product space); the two-point inequalities of PD
  kernels and of CPD kernels vanishing on the diagonal.
* `Section3_Schoenberg`: products, powers and exponentials of PD kernels are PD, and Schoenberg's
  theorem (`exp(t k̃)` is PD for a CPD `k̃` and `t ≥ 0`), on which Facts 3, 5 and 6 rest; CPD
  kernels are closed under pointwise limits.
* `Section3_Integral`: `s^α C_α = ∫_0^∞ (1 − e^{−st}) t^{−1−α} dt` for `0 < α < 1`, with
  `0 < C_α < ∞`, and integrals of CPD kernels are CPD.
* `Section3_Powers`: Fact 3 (`−(−k̃)^α` for `0 < α < 1` and `−log(1 − k̃)` are CPD when `k̃ ≤ 0` is)
  and Corollary 1 (`−a ‖x − x'‖^p` and `−b log(1 + a ‖x − x'‖^p)` are CPD for `0 < p ≤ 2`), with
  the iterations the paper mentions after it.
* `SectionA_ShiftLemma`: Lemma 1 (the CPD shift lemma, Appendix A.1) is false: `x + y` is CPD on
  `ℝ` and `c + x + y` is PD for no real `c`, nor is `c − (x − y)²`; step (iii) of its proof fails
  at `diag(1, −1)`. Its steps (iv)–(vi) prove the corrected lemma: on a finite set, a strictly CPD
  kernel plus a large enough constant is PD.
-/
