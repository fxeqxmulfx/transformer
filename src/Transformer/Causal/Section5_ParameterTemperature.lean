/-
# The original separated frozen-center conditions force temperature above one

ArXiv:2411.04990v2, §5, `thm:fixed_centers` and Appendix B's
interaction estimates. For at least one original moving token
and two actual frozen centers of mass at least one, the original
total mass is at least three. Their original circular separation
puts `c beta^(-1/2)` below pi.

Those original hypotheses and the original strict force inequality
already rule out beta at most one. At such a temperature the
original scale is at least one, and the actual far comparison
point lies between one and pi minus one. Its sine is greater than
five sixths. If the actual neighborhood radius is at most one
third, the original exponential lower bound `exp(-2)>2/15`
contradicts the force inequality. Otherwise the far point lies
below pi/2, and the stronger `exp(-1)>1/3` gives the contradiction.
Thus the manuscript's beta-at-least-one interaction estimates
apply in this original nontrivial separated regime without an
additional temperature hypothesis.
-/

import Transformer.Causal.Section5_ParameterGeometry
import Mathlib.Analysis.Complex.ExponentialBounds

open scoped BigOperators
open Real Set

namespace Transformer.Causal

/-- The actual original force on the first semicircle is at most its actual original angular distance.
Source: arXiv:2411.04990v2, Appendix B, the original `h` used in §5, `thm:fixed_centers`. -/
theorem h_pot_le_angle (β : ℝ) (hβ : 0 ≤ β) (x : ℝ) (hx : x ∈ Icc 0 Real.pi) : h_pot β x ≤ x := by
  have he : Real.exp (β * (Real.cos x - 1)) ≤ 1 := Real.exp_le_one_iff.mpr
    (mul_nonpos_of_nonneg_of_nonpos hβ (sub_nonpos.mpr (Real.cos_le_one x)))
  have hm := mul_le_mul_of_nonneg_right he (Real.sin_nonneg_of_mem_Icc hx)
  exact (show h_pot β x ≤ Real.sin x by simpa! only [h_pot, one_mul] using hm).trans (Real.sin_le hx.1)

/-- On the actual middle semicircle, the original sine exceeds five sixths.
Source: arXiv:2411.04990v2, Appendix B, the sine Taylor bound used in the original interaction estimates. -/
theorem sin_gt_five_sixths_on_middle (x : ℝ) (hx : x ∈ Icc 1 (Real.pi - 1)) : 5 / 6 < Real.sin x := by
  have hsin : (5 / 6 : ℝ) < Real.sin 1 := by nlinarith [Real.sin_gt_sub_cube (x := 1) one_pos]
  by_cases hh : x ≤ Real.pi / 2
  · exact hsin.trans_le (Real.sin_le_sin_of_le_of_le_pi_div_two (by linarith [Real.pi_pos]) hh hx.1)
  · have h := Real.sin_le_sin_of_le_of_le_pi_div_two
      (x := 1) (y := Real.pi - x) (by linarith [Real.pi_pos]) (by linarith) (by linarith [hx.2])
    rw [Real.sin_pi_sub] at h
    exact hsin.trans_le h

/-- The actual original exponential factor at exponent minus two is strictly larger than two fifteenths.
Source: arXiv:2411.04990v2, Appendix B, the exponential comparison for the original interaction in §5. -/
theorem exp_neg_two_gt_two_fifteenths : (2 / 15 : ℝ) < Real.exp (-2) := by
  have h1 : Real.exp 1 < 273 / 100 := by linarith [Real.exp_one_lt_d9]
  have heq : Real.exp 2 = Real.exp 1 ^ 2 := by
    simpa only [Nat.cast_ofNat, mul_one] using Real.exp_nat_mul 1 2
  have h2 : Real.exp 2 < 15 / 2 := by rw [heq]; nlinarith [Real.exp_pos 1]
  rw [Real.exp_neg, inv_eq_one_div, lt_div_iff₀ (Real.exp_pos 2)]
  linarith

/-- The actual original separated force window with original total mass at least three forces the actual original temperature above one.
Source: arXiv:2411.04990v2, §5, a derived consequence of `thm:fixed_centers`' original force and separation hypotheses. -/
theorem one_lt_temperature_of_frozen_parameters (N β ε c : ℝ) (hN : 3 ≤ N) (hβ : 0 < β)
    (hε : 0 < ε) (hc : 2 + 2 * ε < c) (hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) <
      h_pot β (ε * β ^ (-(1 / 2 : ℝ)))) : 1 < β := by
  by_contra h
  have hβ1 : β ≤ 1 := le_of_not_gt h
  let s := β ^ (-(1 / 2 : ℝ))
  let e := ε * s
  let r := (c - 1 - 2 * ε) * s
  have hs : 0 < s := Real.rpow_pos_of_pos hβ _
  have hsq : β * s ^ 2 = 1 := by rw [interactionScale_sq β hβ, mul_inv_cancel₀ hβ.ne']
  have hs1 : 1 ≤ s := by
    have hm := mul_le_mul_of_nonneg_right hβ1 (sq_nonneg s)
    nlinarith
  have he0 : 0 < e := mul_pos hε hs
  have hcr := mul_pos (show 0 < c - 2 - 2 * ε by linarith) hs
  have hr1 : 1 < r := by dsimp [r]; nlinarith
  have hrpi : r < Real.pi - 1 := by dsimp [r]; nlinarith
  have herange : e < 23 / 40 := by dsimp [e]; nlinarith [Real.pi_lt_d2]
  have hsin := sin_gt_five_sixths_on_middle r ⟨hr1.le, hrpi.le⟩
  have hupper : h_pot β e ≤ e := h_pot_le_angle β hβ.le e ⟨he0.le, by linarith [Real.pi_gt_three]⟩
  have hscore : -2 ≤ β * (Real.cos r - 1) := by
    nlinarith [mul_nonneg hβ.le (show 0 ≤ Real.cos r + 1 by linarith [Real.neg_one_le_cos r])]
  have hfactor : (2 / 15 : ℝ) < Real.exp (β * (Real.cos r - 1)) :=
    exp_neg_two_gt_two_fifteenths.trans_le (Real.exp_le_exp.mpr hscore)
  have hlow : (1 / 9 : ℝ) < h_pot β r := by
    have hm := mul_lt_mul_of_pos_right hfactor (show 0 < Real.sin r by linarith)
    change (1 / 9 : ℝ) < Real.exp (β * (Real.cos r - 1)) * Real.sin r
    nlinarith
  have hmass := mul_le_mul_of_nonneg_right hN (show 0 ≤ h_pot β r by linarith)
  change N * h_pot β r < h_pot β e at hh
  by_cases he : e ≤ 1 / 3
  · linarith
  · have hrhalf : r < Real.pi / 2 := by dsimp [r]; nlinarith [Real.pi_lt_d2]
    have hcos : 0 < Real.cos r := Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.pi_pos], hrhalf⟩
    have hscore1 : -1 ≤ β * (Real.cos r - 1) := by nlinarith [mul_nonneg hβ.le hcos.le]
    have hexp1 : (1 / 3 : ℝ) < Real.exp (-1) := by
      rw [Real.exp_neg, inv_eq_one_div, lt_div_iff₀ (Real.exp_pos 1)]
      linarith [Real.exp_one_lt_three]
    have hfactor1 := hexp1.trans_le (Real.exp_le_exp.mpr hscore1)
    have hstrong : (5 / 18 : ℝ) < h_pot β r := by
      have hm := mul_lt_mul_of_pos_right hfactor1 (show 0 < Real.sin r by linarith)
      change (5 / 18 : ℝ) < Real.exp (β * (Real.cos r - 1)) * Real.sin r
      nlinarith
    linarith

/-- The actual original nontrivial moving-and-frozen token counts make the original frozen-center force window force temperature above one.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`; the temperature bound is derived from its actual original total mass. -/
theorem one_lt_temperature_of_actual_frozen_parameters (n m : ℕ) (hn : 1 ≤ n) (hm : 2 ≤ m)
    (β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε) (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ)
    (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hh : ((n : ℝ) + ∑ j, a j) * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) <
      h_pot β (ε * β ^ (-(1 / 2 : ℝ)))) : 1 < β := by
  have hN : (3 : ℝ) ≤ (n : ℝ) + ∑ j, a j := by
    have hnreal : (1 : ℝ) ≤ n := by exact_mod_cast hn
    have hmreal : (2 : ℝ) ≤ m := by exact_mod_cast hm
    linarith [frozen_total_mass_lower n a ha]
  let i : Idx m := ⟨0, by omega⟩
  let j : Idx m := ⟨1, by omega⟩
  have hij : i ≠ j := by
    intro heq
    have hval := congrArg Fin.val heq
    dsimp [i, j] at hval
    omega
  have hcpi := frozen_separation_scale_lt_pi β c (θ i) (θ j) (hsep i j hij)
  exact one_lt_temperature_of_frozen_parameters _ β ε c hN hβ hε hc hcpi hh

/-- The original explicit frozen-center parameter example jointly realizes the derived temperature theorem's full hypotheses.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`, using the original Appendix B parameter window. -/
example : ∃ N β ε c : ℝ, 3 ≤ N ∧ 0 < β ∧ 0 < ε ∧ 2 + 2 * ε < c ∧
    c * β ^ (-(1 / 2 : ℝ)) < Real.pi ∧
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧ 1 < β := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  exact ⟨3, 11, 9 / 100, 142 / 25, le_rfl, hβ, hε, hc, hcpi, hh,
    one_lt_temperature_of_frozen_parameters 3 11 (9 / 100) (142 / 25) le_rfl hβ hε hc hcpi hh⟩

/-- Actual unit temperature and actual distance one satisfy the semicircle force and middle-sine premises jointly.
Source: arXiv:2411.04990v2, Appendix B, the actual original sine and interaction comparisons. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ∈ Icc 0 Real.pi ∧ (1 : ℝ) ∈ Icc 1 (Real.pi - 1) := by
  exact ⟨zero_le_one, ⟨zero_le_one, by linarith [Real.pi_gt_three]⟩, ⟨le_rfl, by linarith [Real.pi_gt_three]⟩⟩

/-- One actual moving token and two actual unit-mass separated frozen centers jointly realize every premise with the original total mass three.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`, with Appendix B's original explicit parameter window. -/
example : ∃ β ε c : ℝ, ∃ θ a : Idx 2 → ℝ,
    0 < β ∧ 0 < ε ∧ 2 + 2 * ε < c ∧ (∀ j, 1 ≤ a j) ∧
    (∀ i j : Idx 2, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j)) ∧
    ((1 : ℝ) + ∑ j, a j) * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) <
      h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧ 1 < β := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  let θ : Idx 2 → ℝ := ![0, Real.pi]
  let a : Idx 2 → ℝ := fun _ => 1
  have ha : ∀ j, 1 ≤ a j := fun _ => le_rfl
  have hsep : ∀ i j : Idx 2, i ≠ j → (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) <
      circDist (θ i - θ j) := by
    have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [θ, circDist_neg]
  have hmass : (1 : ℝ) + ∑ j : Idx 2, a j = 3 := by norm_num [a, Idx]
  have hactual : ((1 : ℝ) + ∑ j : Idx 2, a j) *
      h_pot 11 ((142 / 25 - 1 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) <
        h_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) := by rw [hmass]; exact hh
  exact ⟨11, 9 / 100, 142 / 25, θ, a, hβ, hε, hc, ha, hsep, hactual,
    one_lt_temperature_of_actual_frozen_parameters 1 2 (by decide) (by decide)
      11 (9 / 100) (142 / 25) hβ hε hc θ a ha hsep (by simpa! only [Nat.cast_one] using hactual)⟩

end Transformer.Causal
