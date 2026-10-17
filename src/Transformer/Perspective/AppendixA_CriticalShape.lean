/-
# Appendix A — the shape of the critical points of `𝖤_0`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A.

A critical point of `𝖤_0` on `(𝕊^{d-1})^n` has its barycentre `s = Σ_j z_j`
normal to the sphere at every particle, `s = ⟨z_i, s⟩ z_i`.  So either `s = 0`,
or every particle is `± s/‖s‖`: the configuration is `signedTuple e S`, the
particles of `S` at `-e` and the others at `e`, for the unit vector
`e = s/‖s‖` (`eq_signedTuple_of_critical`).  Conversely every `signedTuple`
is critical, and constant paths at critical points solve `e:gradfl`; these
are the witnesses the hypotheses of `l:nosaddleconv` and its cases are tested
on.
-/

import Transformer.Perspective.AppendixA_Beta0

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- The constant path at a critical point solves `e:gradfl`. -/
theorem E0GradientAscent_const (Z : SphereTuple d n) (hcrit : IsCriticalE0 d n Z) :
    E0GradientAscent d n fun _ => Z := by
  intro t i
  rw [proj_smul, hcrit i, smul_zero]
  exact hasDerivAt_const t _

/-- The configuration with the particles of `S` at `-x` and the others at `x`. -/
def signedTuple (x : SSphere d) (S : Finset (Idx n)) : SphereTuple d n :=
  fun k => if k ∈ S then antipode d x else x

theorem coe_signedTuple (x : SSphere d) (S : Finset (Idx n)) (k : Idx n) :
    (signedTuple x S k : EucSpace d) = if k ∈ S then -(x : EucSpace d) else x := by
  unfold signedTuple
  split_ifs <;> rfl

/-- The barycentre of `signedTuple x S` is `(n - 2 |S|) x`. -/
theorem sum_signedTuple (x : SSphere d) (S : Finset (Idx n)) :
    ∑ k : Idx n, (signedTuple x S k : EucSpace d)
      = ((n : ℝ) - 2 * S.card) • (x : EucSpace d) := by
  have h : ∀ k : Idx n, (signedTuple x S k : EucSpace d)
      = (x : EucSpace d) - (if k ∈ S then (2 : ℝ) • (x : EucSpace d) else 0) := by
    intro k
    rw [coe_signedTuple]
    split_ifs
    · module
    · module
  simp_rw [h, Finset.sum_sub_distrib, Fintype.sum_ite_mem, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin]
  module

/-- `Proj_{±x} x = 0` for a unit vector `x`. -/
theorem proj_signed_self (x : SSphere d) (b : Bool) (a : ℝ) :
    proj d (if b then -(x : EucSpace d) else x) (a • (x : EucSpace d)) = 0 := by
  cases b <;> simp [proj, real_inner_smul_right]

/-- Every `signedTuple` is a critical point of `𝖤_0`. -/
theorem isCriticalE0_signedTuple (x : SSphere d) (S : Finset (Idx n)) :
    IsCriticalE0 d n (signedTuple x S) := by
  intro i
  rw [sum_signedTuple, coe_signedTuple]
  simpa using proj_signed_self x (decide (i ∈ S)) ((n : ℝ) - 2 * S.card)

/-- At a critical point every particle is collinear with the barycentre. -/
theorem critical_coord (Z : SphereTuple d n) (hcrit : IsCriticalE0 d n Z) (i : Idx n) :
    ∑ k : Idx n, (Z k : EucSpace d)
      = inner (𝕜 := ℝ) (Z i : EucSpace d) (∑ k : Idx n, (Z k : EucSpace d))
          • (Z i : EucSpace d) :=
  sub_eq_zero.mp (hcrit i)

/-- At a critical point `|⟨z_i, s⟩| = ‖s‖`. -/
theorem abs_inner_critical (Z : SphereTuple d n) (hcrit : IsCriticalE0 d n Z) (i : Idx n) :
    |inner (𝕜 := ℝ) (Z i : EucSpace d) (∑ k : Idx n, (Z k : EucSpace d))|
      = ‖∑ k : Idx n, (Z k : EucSpace d)‖ := by
  conv_rhs => rw [critical_coord Z hcrit i]
  rw [norm_smul, Real.norm_eq_abs, mem_sphere_zero_iff_norm.mp (Z i).2, mul_one]

/-- A particle of a critical point on the far side of the barycentre is
`-s/‖s‖`. -/
theorem critical_coord_of_neg (Z : SphereTuple d n) (hcrit : IsCriticalE0 d n Z) (i : Idx n)
    (h : inner (𝕜 := ℝ) (Z i : EucSpace d) (∑ k : Idx n, (Z k : EucSpace d)) < 0) :
    (Z i : EucSpace d)
      = -(‖∑ k : Idx n, (Z k : EucSpace d)‖⁻¹ • ∑ k : Idx n, (Z k : EucSpace d)) := by
  set S := ∑ k : Idx n, (Z k : EucSpace d) with hSdef
  have hc : S = inner (𝕜 := ℝ) (Z i : EucSpace d) S • (Z i : EucSpace d) :=
    critical_coord Z hcrit i
  have habs : |inner (𝕜 := ℝ) (Z i : EucSpace d) S| = ‖S‖ := abs_inner_critical Z hcrit i
  rw [abs_of_neg h] at habs
  have hpos : 0 < ‖S‖ := by linarith
  have key : ‖S‖⁻¹ • S = ‖S‖⁻¹ • ((-‖S‖) • (Z i : EucSpace d)) :=
    congrArg (fun v => ‖S‖⁻¹ • v) (by rw [← habs, neg_neg]; exact hc)
  rw [key, smul_smul, mul_neg, inv_mul_cancel₀ hpos.ne', neg_one_smul, neg_neg]

/-- A particle of a critical point on the near side of a nonzero barycentre is
`s/‖s‖`. -/
theorem critical_coord_of_nonneg (Z : SphereTuple d n) (hcrit : IsCriticalE0 d n Z)
    (hS : ∑ k : Idx n, (Z k : EucSpace d) ≠ 0) (i : Idx n)
    (h : 0 ≤ inner (𝕜 := ℝ) (Z i : EucSpace d) (∑ k : Idx n, (Z k : EucSpace d))) :
    (Z i : EucSpace d)
      = ‖∑ k : Idx n, (Z k : EucSpace d)‖⁻¹ • ∑ k : Idx n, (Z k : EucSpace d) := by
  set S := ∑ k : Idx n, (Z k : EucSpace d) with hSdef
  have hc : S = inner (𝕜 := ℝ) (Z i : EucSpace d) S • (Z i : EucSpace d) :=
    critical_coord Z hcrit i
  have habs : |inner (𝕜 := ℝ) (Z i : EucSpace d) S| = ‖S‖ := abs_inner_critical Z hcrit i
  rw [abs_of_nonneg h] at habs
  have key : ‖S‖⁻¹ • S = ‖S‖⁻¹ • (‖S‖ • (Z i : EucSpace d)) :=
    congrArg (fun v => ‖S‖⁻¹ • v) (by rw [← habs]; exact hc)
  rw [key, smul_smul, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hS), one_smul]

/-- **The critical points with nonzero barycentre are the `signedTuple`s.**
With `e = s/‖s‖`, a critical point `Z` is `signedTuple e S` for `S` the set of
particles with `⟨z_i, s⟩ < 0`. -/
theorem eq_signedTuple_of_critical (Z : SphereTuple d n) (hcrit : IsCriticalE0 d n Z)
    (hS : ∑ k : Idx n, (Z k : EucSpace d) ≠ 0) :
    ∃ e : SSphere d,
      (e : EucSpace d)
          = ‖∑ k : Idx n, (Z k : EucSpace d)‖⁻¹ • ∑ k : Idx n, (Z k : EucSpace d) ∧
        Z = signedTuple e {i | inner (𝕜 := ℝ) (Z i : EucSpace d)
          (∑ k : Idx n, (Z k : EucSpace d)) < 0} := by
  classical
  have hnorm : ‖‖∑ k : Idx n, (Z k : EucSpace d)‖⁻¹ • ∑ k : Idx n, (Z k : EucSpace d)‖ = 1 := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hS)]
  refine ⟨⟨_, mem_sphere_zero_iff_norm.mpr hnorm⟩, rfl, funext fun i => Subtype.ext ?_⟩
  rw [coe_signedTuple]
  split_ifs with h
  · exact critical_coord_of_neg Z hcrit i (Finset.mem_filter.mp h).2
  · exact critical_coord_of_nonneg Z hcrit hS i
      (le_of_not_gt fun h' => h (Finset.mem_filter.mpr ⟨Finset.mem_univ i, h'⟩))

/-- The hypotheses of `eq_signedTuple_of_critical` hold for `(x, x, -x)`: its
barycentre is `x ≠ 0`. -/
example (x : SSphere 2) :
    IsCriticalE0 2 3 (signedTuple x {2}) ∧
      ∑ k : Idx 3, (signedTuple x {2} k : EucSpace 2) ≠ 0 := by
  refine ⟨isCriticalE0_signedTuple x {2}, ?_⟩
  rw [sum_signedTuple]
  have hx : (x : EucSpace 2) ≠ 0 := ne_zero_of_mem_unit_sphere x
  norm_num [hx]

end Perspective
end Transformer
