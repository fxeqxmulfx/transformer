/-
# The NVFP4 grids, round-to-nearest, and stochastic rounding

arXiv:2601.22813v2, Panferov, Schultheis, Tabesh, Alistarh, "Quartet II:
Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation"
(ICML 2026), §3.1.

NVFP4 stores a tensor at three levels: one E2M1 number per entry, one E4M3
scale per `16` entries, and one FP32 scale per tensor.  This module is the
bottom level: the two grids, the deterministic rounding `RTN` onto a grid, and
the stochastic rounding `SR` the paper relies on for unbiasedness.

Randomness is explicit.  `sr G x u` reads its coin from `u`, and *unbiased*
means that the integral over `u ∈ [0,1]` returns `x` — the paper's "`SR_FP4`
is the probabilistic rounding operation w.r.t. randomness `ω` which preserves
its argument in expectation".

`floorOn` and `ceilOn` are the supremum and the infimum of the grid points on
either side of the argument, so they are defined for every grid; outside the
range of the grid they return junk, and every statement below asks for a grid
point on either side.
-/

import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

namespace Transformer
namespace Quartet

/-- The E2M1 grid of NVFP4 elements, `±{0, 0.5, 1, 1.5, 2, 3, 4, 6}`: two
exponent bits and one mantissa bit, with `6.0` "the absolute maximum value
representable by FP4" (§3.1). -/
def fp4 : Set ℝ := {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6}

/-- The E4M3 grid of NVFP4 group scales: the numbers `±m·2^e` with `m < 16`
and `-9 ≤ e`, of absolute value at most `448`.  Three mantissa bits and an
exponent bias of `7` put the subnormal step at `2^{-9}`, and `448.0` is "the
absolute maximum value representable by FP8" (§3.1) because the all-ones
pattern is reserved for `NaN`. -/
def fp8 : Set ℝ := {x | |x| ≤ 448 ∧ ∃ (m : ℕ) (e : ℤ), m < 16 ∧ -9 ≤ e ∧ |x| = m * 2 ^ e}

/-- The largest grid point at most `x`. -/
noncomputable def floorOn (G : Set ℝ) (x : ℝ) : ℝ := sSup {y | y ∈ G ∧ y ≤ x}

/-- The smallest grid point at least `x`. -/
noncomputable def ceilOn (G : Set ℝ) (x : ℝ) : ℝ := sInf {y | y ∈ G ∧ x ≤ y}

/-- `RTN`, round-to-nearest on a grid, breaking ties downwards (§3.1, the
`RTN_FP8` and `RTN_FP4` of the quantization displays). -/
noncomputable def rtn (G : Set ℝ) (x : ℝ) : ℝ :=
  if x - floorOn G x ≤ ceilOn G x - x then floorOn G x else ceilOn G x

/-- `SR`, stochastic rounding on a grid with the coin `u`: the upper neighbour
is returned with probability `(x − ⌊x⌋_G)/(⌈x⌉_G − ⌊x⌋_G)` (§3.1, `SR_FP4`).
On a grid point both neighbours are `x`, and the coin decides nothing. -/
noncomputable def sr (G : Set ℝ) (x u : ℝ) : ℝ :=
  if u * (ceilOn G x - floorOn G x) < x - floorOn G x then ceilOn G x else floorOn G x

/-- **The two roundings bracket their argument.**  With a grid point on either
side of `x`, the supremum and the infimum defining `floorOn` and `ceilOn` are
attained, which is what §3.1 assumes of `RTN` and `SR` throughout. -/
theorem floorOn_mem_le_and_le_ceilOn_mem {G : Set ℝ} {x : ℝ} (hfin : G.Finite)
    (hlo : ∃ y ∈ G, y ≤ x) (hhi : ∃ y ∈ G, x ≤ y) :
    floorOn G x ∈ G ∧ floorOn G x ≤ x ∧ ceilOn G x ∈ G ∧ x ≤ ceilOn G x :=
  sorry

/-- The hypotheses of `floorOn_mem_le_and_le_ceilOn_mem` are satisfiable: the
E2M1 grid is finite and surrounds `0.75`. -/
example : fp4.Finite ∧ (∃ y ∈ fp4, y ≤ (0.75 : ℝ)) ∧ ∃ y ∈ fp4, (0.75 : ℝ) ≤ y :=
  ⟨by unfold fp4; apply Set.toFinite, ⟨0.5, by norm_num [fp4], by norm_num⟩,
    ⟨1, by norm_num [fp4], by norm_num⟩⟩

/-- Round-to-nearest lands on the grid. -/
theorem rtn_mem {G : Set ℝ} {x : ℝ} (hfin : G.Finite) (hlo : ∃ y ∈ G, y ≤ x)
    (hhi : ∃ y ∈ G, x ≤ y) : rtn G x ∈ G := by
  obtain ⟨hf, -, hc, -⟩ := floorOn_mem_le_and_le_ceilOn_mem hfin hlo hhi
  unfold rtn
  split
  · exact hf
  · exact hc

/-- And so does stochastic rounding, whatever the coin says. -/
theorem sr_mem {G : Set ℝ} {x u : ℝ} (hfin : G.Finite) (hlo : ∃ y ∈ G, y ≤ x)
    (hhi : ∃ y ∈ G, x ≤ y) : sr G x u ∈ G := by
  obtain ⟨hf, -, hc, -⟩ := floorOn_mem_le_and_le_ceilOn_mem hfin hlo hhi
  unfold sr
  split
  · exact hc
  · exact hf

/-- The hypotheses of `rtn_mem` and `sr_mem` are satisfiable: the E2M1 grid
also surrounds `-2.5`. -/
example : fp4.Finite ∧ (∃ y ∈ fp4, y ≤ (-2.5 : ℝ)) ∧ ∃ y ∈ fp4, (-2.5 : ℝ) ≤ y :=
  ⟨by unfold fp4; apply Set.toFinite, ⟨-3, by norm_num [fp4], by norm_num⟩,
    ⟨-2, by norm_num [fp4], by norm_num⟩⟩

/-- The E4M3 grid holds both ends of its range: the group scales `448` — its
largest element, `14·2^5` — and `0`. -/
example : (448 : ℝ) ∈ fp8 ∧ (0 : ℝ) ∈ fp8 :=
  ⟨⟨by norm_num, 14, 5, by norm_num⟩, ⟨by norm_num, 0, 0, by norm_num⟩⟩

/-- **Stochastic rounding is unbiased**: averaging over a coin uniform on
`[0,1]` returns the argument (§3.1, "`SR_FP4` … preserves its argument in
expectation").  This is the one property the whole backward pass of the paper
rests on. -/
theorem integral_sr {G : Set ℝ} {x : ℝ} (hfin : G.Finite) (hlo : ∃ y ∈ G, y ≤ x)
    (hhi : ∃ y ∈ G, x ≤ y) : ∫ u in (0 : ℝ)..1, sr G x u = x :=
  sorry

end Quartet
end Transformer
