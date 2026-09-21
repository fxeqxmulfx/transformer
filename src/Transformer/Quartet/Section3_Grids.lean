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
its argument in expectation".  That statement,
`integral_sr`, is proved in `Transformer.Quartet.Section3_Unbiased`, next to
its one use.

`floorOn` and `ceilOn` are the supremum and the infimum of the grid points on
either side of the argument.  Outside the range of the grid they saturate: an
argument above the largest point rounds to it, one below the least point to
that, which is what a floating-point cast with clipping does and what §3.3
relies on when its rescaled entries overshoot `6`.  The bracketing statements
below ask for a grid point on either side; `rtn_of_forall_le`,
`rtn_of_le_forall` and `sr_of_forall_le` are the saturation.
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

open Classical in
/-- The largest grid point at most `x`; below the grid, its least point.  The
second branch is the saturation of a floating-point cast: an argument beyond
the range is clipped to the end of the grid (§3.3, where the rescaling by
`0.93` sends entries past `6` and they "clip"). -/
noncomputable def floorOn (G : Set ℝ) (x : ℝ) : ℝ :=
  if ∃ y ∈ G, y ≤ x then sSup {y | y ∈ G ∧ y ≤ x} else sInf G

open Classical in
/-- The smallest grid point at least `x`; above the grid, its greatest point,
saturating as `floorOn` does. -/
noncomputable def ceilOn (G : Set ℝ) (x : ℝ) : ℝ :=
  if ∃ y ∈ G, x ≤ y then sInf {y | y ∈ G ∧ x ≤ y} else sSup G

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
    floorOn G x ∈ G ∧ floorOn G x ≤ x ∧ ceilOn G x ∈ G ∧ x ≤ ceilOn G x := by
  rw [floorOn, ite_eq_left hlo, ceilOn, ite_eq_left hhi]
  obtain ⟨a, haG, hax⟩ := hlo
  obtain ⟨b, hbG, hxb⟩ := hhi
  have hf := Set.Nonempty.csSup_mem (s := {y | y ∈ G ∧ y ≤ x}) ⟨a, haG, hax⟩
    (hfin.subset fun y hy => hy.1)
  have hc := Set.Nonempty.csInf_mem (s := {y | y ∈ G ∧ x ≤ y}) ⟨b, hbG, hxb⟩
    (hfin.subset fun y hy => hy.1)
  exact ⟨hf.1, hf.2, hc.1, hc.2⟩

/-- The hypotheses of `floorOn_mem_le_and_le_ceilOn_mem` are satisfiable: the
E2M1 grid is finite and surrounds `0.75`. -/
example : fp4.Finite ∧ (∃ y ∈ fp4, y ≤ (0.75 : ℝ)) ∧ ∃ y ∈ fp4, (0.75 : ℝ) ≤ y :=
  ⟨by unfold fp4; apply Set.toFinite, ⟨0.5, by norm_num [fp4], by norm_num⟩,
    ⟨1, by norm_num [fp4], by norm_num⟩⟩

/-- **Reading off the floor.**  A grid point below `x` that dominates every
grid point below `x` is `floorOn G x`: the supremum is attained there.  This is
how the `RTN` and `SR` displays of §3.1 are evaluated at a concrete argument. -/
theorem floorOn_eq {G : Set ℝ} {x f : ℝ} (hf : f ∈ G) (hfx : f ≤ x)
    (hub : ∀ y ∈ G, y ≤ x → y ≤ f) : floorOn G x = f := by
  rw [floorOn, ite_eq_left ⟨f, hf, hfx⟩]
  exact IsGreatest.csSup_eq ⟨⟨hf, hfx⟩, fun _ hy => hub _ hy.1 hy.2⟩

/-- The hypotheses of `floorOn_eq` are satisfiable: `0.5` is the E2M1 point
just below `0.75`. -/
example : (0.5 : ℝ) ∈ fp4 ∧ (0.5 : ℝ) ≤ 0.75 ∧ ∀ y ∈ fp4, y ≤ (0.75 : ℝ) → y ≤ 0.5 := by
  refine ⟨by norm_num [fp4], by norm_num, ?_⟩
  intro y hy hle
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    (revert hle; norm_num)

/-- **Reading off the ceiling**, the same statement for the infimum. -/
theorem ceilOn_eq {G : Set ℝ} {x c : ℝ} (hc : c ∈ G) (hxc : x ≤ c)
    (hlb : ∀ y ∈ G, x ≤ y → c ≤ y) : ceilOn G x = c := by
  rw [ceilOn, ite_eq_left ⟨c, hc, hxc⟩]
  exact IsLeast.csInf_eq ⟨⟨hc, hxc⟩, fun _ hy => hlb _ hy.1 hy.2⟩

/-- The hypotheses of `ceilOn_eq` are satisfiable: `1` is the E2M1 point just
above `0.75`. -/
example : (1 : ℝ) ∈ fp4 ∧ (0.75 : ℝ) ≤ 1 ∧ ∀ y ∈ fp4, (0.75 : ℝ) ≤ y → 1 ≤ y := by
  refine ⟨by norm_num [fp4], by norm_num, ?_⟩
  intro y hy hle
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    (revert hle; norm_num)

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

/-- **Both neighbours saturate above the grid.**  Past the largest grid point
`g`, the floor is `g` and there is no ceiling, so `ceilOn` falls back to `g`. -/
theorem floorOn_ceilOn_of_forall_le {G : Set ℝ} {x g : ℝ} (hg : g ∈ G)
    (hmax : ∀ y ∈ G, y ≤ g) (hgx : g < x) : floorOn G x = g ∧ ceilOn G x = g := by
  refine ⟨floorOn_eq hg hgx.le fun y hy _ => hmax y hy, ?_⟩
  rw [ceilOn, ite_eq_right fun ⟨y, hy, hxy⟩ => absurd (hmax y hy) (not_le.mpr (hgx.trans_le hxy))]
  exact IsGreatest.csSup_eq ⟨hg, hmax⟩

/-- **And below it**, onto the least grid point. -/
theorem floorOn_ceilOn_of_le_forall {G : Set ℝ} {x g : ℝ} (hg : g ∈ G)
    (hmin : ∀ y ∈ G, g ≤ y) (hxg : x < g) : floorOn G x = g ∧ ceilOn G x = g := by
  refine ⟨?_, ceilOn_eq hg hxg.le fun y hy _ => hmin y hy⟩
  rw [floorOn, ite_eq_right fun ⟨y, hy, hyx⟩ => absurd (hmin y hy) (not_le.mpr (hyx.trans_lt hxg))]
  exact IsLeast.csInf_eq ⟨hg, hmin⟩

/-- **Round-to-nearest saturates**: an E2M1 cast sends everything past `6` to
`6`, which is the clipping of §3.3's `Q_RTN`. -/
theorem rtn_of_forall_le {G : Set ℝ} {x g : ℝ} (hg : g ∈ G) (hmax : ∀ y ∈ G, y ≤ g)
    (hgx : g < x) : rtn G x = g := by
  obtain ⟨hf, hc⟩ := floorOn_ceilOn_of_forall_le hg hmax hgx
  rw [rtn, hf, hc, ite_self]

/-- The same below the grid. -/
theorem rtn_of_le_forall {G : Set ℝ} {x g : ℝ} (hg : g ∈ G) (hmin : ∀ y ∈ G, g ≤ y)
    (hxg : x < g) : rtn G x = g := by
  obtain ⟨hf, hc⟩ := floorOn_ceilOn_of_le_forall hg hmin hxg
  rw [rtn, hf, hc, ite_self]

/-- **So does stochastic rounding**, whatever the coin: both neighbours are the
end of the grid. -/
theorem sr_of_forall_le {G : Set ℝ} {x g u : ℝ} (hg : g ∈ G) (hmax : ∀ y ∈ G, y ≤ g)
    (hgx : g < x) : sr G x u = g := by
  obtain ⟨hf, hc⟩ := floorOn_ceilOn_of_forall_le hg hmax hgx
  rw [sr, hf, hc, ite_self]

/-- The E2M1 grid lies in `[-6, 6]`. -/
theorem mem_Icc_of_mem_fp4 {y : ℝ} (hy : y ∈ fp4) : -6 ≤ y ∧ y ≤ 6 := by
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> norm_num

/-- The hypotheses of the saturation lemmas are satisfiable: `±6` are the ends
of the E2M1 grid, and `±7` lie beyond them. -/
example : (6 : ℝ) ∈ fp4 ∧ (∀ y ∈ fp4, y ≤ (6 : ℝ)) ∧ (6 : ℝ) < 7 ∧
    (-6 : ℝ) ∈ fp4 ∧ (∀ y ∈ fp4, (-6 : ℝ) ≤ y) ∧ (-7 : ℝ) < -6 :=
  ⟨by norm_num [fp4], fun _ hy => (mem_Icc_of_mem_fp4 hy).2, by norm_num,
    by norm_num [fp4], fun _ hy => (mem_Icc_of_mem_fp4 hy).1, by norm_num⟩

/-- The E4M3 grid holds both ends of its range: the group scales `448` — its
largest element, `14·2^5` — and `0`. -/
example : (448 : ℝ) ∈ fp8 ∧ (0 : ℝ) ∈ fp8 :=
  ⟨⟨by norm_num, 14, 5, by norm_num⟩, ⟨by norm_num, 0, 0, by norm_num⟩⟩

end Quartet
end Transformer
