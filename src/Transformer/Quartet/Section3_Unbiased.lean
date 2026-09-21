/-
# `Q_SR` is unbiased

arXiv:2601.22813v2, §3.1: "`E_ω[x_i^FP4 × x^FP8_{i//16} × x^FP32] = x_i`",
the property every FP4 training method before the paper pays for with
element-wise stochastic rounding, and the one `Quartet II` sets out to obtain
more cheaply.

The proof is short once the scheme is known not to clip
(`Transformer.Quartet.Section3_NonClipping`): the two scales do not depend on
the rounding coin, so they come out of the integral, and what is left is
`integral_sr` on the E2M1 grid, whose two neighbours of the argument exist
because the argument lies in `[-6, 6]`.  The hypothesis
`max|x| ≤ 2^14 · max_g|x|` is the one discussed there.
-/

import Transformer.Quartet.Section3_NonClipping

namespace Transformer
namespace Quartet

variable {k : ℕ} {c : ℝ}

/-- **Stochastic rounding is unbiased**: averaging over a coin uniform on
`[0,1]` returns the argument (§3.1, "`SR_FP4` … preserves its argument in
expectation").  This is the one property the whole backward pass of the paper
rests on. -/
theorem integral_sr {G : Set ℝ} {x : ℝ} (hfin : G.Finite) (hlo : ∃ y ∈ G, y ≤ x)
    (hhi : ∃ y ∈ G, x ≤ y) : ∫ u in (0 : ℝ)..1, sr G x u = x := by
  obtain ⟨-, hfx, -, hxc⟩ := floorOn_mem_le_and_le_ceilOn_mem hfin hlo hhi
  have hsr0 : ∀ u : ℝ, sr G x u =
      if u * (ceilOn G x - floorOn G x) < x - floorOn G x then ceilOn G x else floorOn G x :=
    fun _ => rfl
  set f := floorOn G x with hfdef
  set c := ceilOn G x with hcdef
  rcases eq_or_lt_of_le (hfx.trans hxc) with hfc | hfc
  · -- the two neighbours coincide, so `x` is itself a grid point
    have hxf : x = f := le_antisymm (hfc ▸ hxc) hfx
    have hconst : ∀ u : ℝ, sr G x u = x := by
      intro u
      rw [hsr0 u, ← hfc, ite_self]
      exact hxf.symm
    simp [hconst]
  · -- the coin decides, with probability `(x - f)/(c - f)` of rounding up
    set p := (x - f) / (c - f) with hpdef
    have hcf : (0 : ℝ) < c - f := sub_pos.mpr hfc
    have hp0 : 0 ≤ p := div_nonneg (sub_nonneg.mpr hfx) hcf.le
    have hp1 : p ≤ 1 := (div_le_one hcf).mpr (by linarith)
    have hiff : ∀ u : ℝ, u * (c - f) < x - f ↔ u < p := by
      intro u
      rw [hpdef, lt_div_iff₀ hcf]
    have hsr : ∀ u : ℝ, sr G x u = if u < p then c else f := fun u =>
      (hsr0 u).trans (if_congr (hiff u) rfl rfl)
    have hanti : Antitone fun u : ℝ => if u < p then c else f := by
      intro u v huv
      by_cases hv : v < p
      · simp [hv, lt_of_le_of_lt huv hv]
      · by_cases hu : u < p <;> simp [hu, hv, hfc.le]
    have hsplit := intervalIntegral.integral_add_adjacent_intervals
      (a := (0 : ℝ)) (b := p) (c := (1 : ℝ)) (f := fun u => if u < p then c else f)
      (hanti.intervalIntegrable (μ := MeasureTheory.volume))
      (hanti.intervalIntegrable (μ := MeasureTheory.volume))
    have hlow : ∫ u in (0 : ℝ)..p, (if u < p then c else f) = p * c := by
      rw [intervalIntegral.integral_congr_ae (g := fun _ => c) ?_]
      · simp
      · filter_upwards [MeasureTheory.compl_mem_ae_iff.mpr
          (MeasureTheory.measure_singleton (μ := MeasureTheory.volume) p)] with u hu hmem
        rw [Set.uIoc_of_le hp0] at hmem
        have : u < p := lt_of_le_of_ne hmem.2 hu
        simp [this]
    have hhigh : ∫ u in p..(1 : ℝ), (if u < p then c else f) = (1 - p) * f := by
      rw [intervalIntegral.integral_congr (g := fun _ => f) ?_]
      · simp
      · intro u hu
        rw [Set.uIcc_of_le hp1] at hu
        simp [not_lt.mpr hu.1]
    simp only [hsr]
    rw [← hsplit, hlow, hhigh, hpdef]
    field_simp
    ring

/-- The group scales of `Q_SR` are positive, so the dequantization can be
divided by them: `RTN_FP8` keeps at least `16/17` of a normal argument. -/
theorem groupScaleSR_pos {x : Fin (2 ^ k) → Fin 16 → ℝ} (hc : 0 < c) (hx : 0 < absMax x)
    (i : Fin (2 ^ k)) (hg : absMax x ≤ 2 ^ (14 : ℕ) * groupAbsMax x i) :
    0 < groupScaleSR c x i := by
  have hTs : tensorScaleSR c x * c * (16 / 17) = absMax x / 448 := by
    unfold tensorScaleSR
    field_simp
  have harg : groupAbsMax x i / (tensorScaleSR c x * c * (16 / 17))
      = 448 * groupAbsMax x i / absMax x := by
    rw [hTs, div_div_eq_mul_div]
    ring
  obtain ⟨hlo, hhi⟩ := normal_groupScale_arg (b := 448) (by norm_num) le_rfl hx i hg
  have hv0 : (0 : ℝ) < 448 * groupAbsMax x i / absMax x := lt_of_lt_of_le (by positivity) hlo
  have hG : 16 / 17 * (448 * groupAbsMax x i / absMax x) ≤ groupScaleSR c x i := by
    unfold groupScaleSR
    rw [harg]
    exact le_rtn_fp8 hlo hhi
  linarith

/-- **`Q_SR` is unbiased**, entry by entry: "`E_ω[x_i^FP4 × x^FP8_{i//16} ×
x^FP32] = x_i`" (§3.1). -/
theorem integral_qSRAt {x : Fin (2 ^ k) → Fin 16 → ℝ} (hc : 0 < c) (hc' : c ≤ 6)
    (hx : 0 < absMax x) (i : Fin (2 ^ k)) (hg : absMax x ≤ 2 ^ (14 : ℕ) * groupAbsMax x i)
    (j : Fin 16) : ∫ t in (0 : ℝ)..1, qSRAt c x i j t = x i j := by
  obtain ⟨hlo6, hhi6⟩ := abs_le.mp (abs_div_groupScaleSR_le hc hc' hx i hg j)
  have hT : 0 < tensorScaleSR c x := div_pos hx (by positivity)
  have hG : 0 < groupScaleSR c x i := groupScaleSR_pos hc hx i hg
  have hsr := integral_sr (G := fp4) (x := x i j / (groupScaleSR c x i * tensorScaleSR c x))
    (by unfold fp4; exact Set.toFinite _) ⟨-6, by norm_num [fp4], hlo6⟩
    ⟨6, by norm_num [fp4], hhi6⟩
  unfold qSRAt
  rw [intervalIntegral.integral_mul_const, intervalIntegral.integral_mul_const, hsr, mul_assoc,
    div_mul_cancel₀ _ (mul_pos hG hT).ne']

/-- The hypotheses of both statements are satisfiable at the grid maximum
`6.0` of §3.1 and a tensor of ones. -/
example : 0 < (6 : ℝ) ∧ (6 : ℝ) ≤ 6 ∧
    0 < absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ∧
    absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ≤
      2 ^ (14 : ℕ) * groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 := by
  have h1 : absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ≤ 1 := by
    unfold absMax
    exact Finset.sup'_le _ _ fun p _ => by norm_num
  have h2 : (1 : ℝ) ≤ groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 := by
    simpa using abs_le_groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 0
  refine ⟨by norm_num, le_rfl, ?_, by norm_num; linarith⟩
  unfold absMax
  exact lt_of_lt_of_le (by norm_num)
    (Finset.le_sup' _ (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))

end Quartet
end Transformer
