/-
# Global flows of autonomous ODEs — the radial retraction and a priori bounds

Mathlib proves the Picard–Lindelöf theorem on a compact time interval, for a
vector field that is bounded and Lipschitz on a ball around the initial point.
The fields of the transformer papers are only *locally* Lipschitz, but they
grow at most linearly, `‖F x‖ ≤ C ‖x‖`; their solutions therefore exist for
all time.  This file provides the two tools that argument needs:

* the radial retraction `ballRetract R` onto `closedBall 0 R`, which is
  `2`-Lipschitz in any normed space and turns a field Lipschitz on the ball
  into one Lipschitz everywhere, without changing it on the ball;
* the a priori bound `‖x(t)‖ ≤ ‖x(0)‖ e^{C|t|}` (Grönwall), forwards and
  backwards in time.

Source: the standard proof of global existence under linear growth, as used by
arXiv:2305.05465v6, `p:wellposedparticles`.
-/

import Mathlib.Analysis.ODE.ExistUnique
import Mathlib.Analysis.ODE.Gronwall

open Set Metric
open scoped NNReal

namespace Transformer
namespace GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The radial retraction onto `closedBall 0 R`: the identity on the ball,
`x ↦ R x / ‖x‖` outside it. -/
noncomputable def ballRetract (R : ℝ) (x : E) : E := (R / max R ‖x‖) • x

theorem ballRetract_of_norm_le {R : ℝ} {x : E} (hR : 0 < R) (hx : ‖x‖ ≤ R) :
    ballRetract R x = x := by
  rw [ballRetract, max_eq_left hx, div_self hR.ne', one_smul]

theorem norm_ballRetract_le_norm {R : ℝ} (hR : 0 < R) (x : E) : ‖ballRetract R x‖ ≤ ‖x‖ := by
  rw [ballRetract, norm_smul, Real.norm_of_nonneg (div_nonneg hR.le (hR.le.trans (le_max_left _ _)))]
  exact mul_le_of_le_one_left (norm_nonneg _)
    ((div_le_one (hR.trans_le (le_max_left _ _))).2 (le_max_left _ _))

theorem norm_ballRetract_le {R : ℝ} (hR : 0 < R) (x : E) : ‖ballRetract R x‖ ≤ R := by
  have hm : 0 < max R ‖x‖ := hR.trans_le (le_max_left _ _)
  rw [ballRetract, norm_smul, Real.norm_of_nonneg (div_nonneg hR.le hm.le), div_mul_eq_mul_div,
    div_le_iff₀ hm]
  exact mul_le_mul_of_nonneg_left (le_max_right _ _) hR.le

/-- **The radial retraction is `2`-Lipschitz**, in any normed space.  With
`a = max R ‖x‖ ≥ b = max R ‖y‖`,
`‖R x / a - R y / b‖ ≤ (R / a) ‖x - y‖ + R (a - b) / a ≤ 2 ‖x - y‖`. -/
theorem lipschitzWith_ballRetract {R : ℝ} (hR : 0 < R) :
    LipschitzWith 2 (ballRetract (E := E) R) := by
  have key : ∀ x y : E, max R ‖y‖ ≤ max R ‖x‖ →
      ‖ballRetract R x - ballRetract R y‖ ≤ 2 * ‖x - y‖ := by
    intro x y hab
    set a := max R ‖x‖
    set b := max R ‖y‖
    have hb : 0 < b := hR.trans_le (le_max_left _ _)
    have ha : 0 < a := hb.trans_le hab
    have hyb : ‖y‖ ≤ b := le_max_right _ _
    have hRa : R / a ≤ 1 := (div_le_one ha).2 (le_max_left _ _)
    have hab' : a - b ≤ ‖x - y‖ := by
      have := abs_max_sub_max_le_max R ‖x‖ R ‖y‖
      simp only [sub_self, abs_zero] at this
      exact (le_abs_self _).trans (this.trans (max_le (norm_nonneg _) (abs_norm_sub_norm_le x y)))
    have hsplit : ballRetract R x - ballRetract R y =
        (R / a) • (x - y) + (R / a - R / b) • y := by
      simp only [ballRetract, smul_sub, sub_smul]; abel
    rw [hsplit]
    refine (norm_add_le _ _).trans ?_
    rw [norm_smul, norm_smul, Real.norm_of_nonneg (div_nonneg hR.le ha.le)]
    have h2 : ‖R / a - R / b‖ * ‖y‖ ≤ R / a * ‖x - y‖ := by
      have hd : R / a - R / b = -(R * (a - b) / (a * b)) := by field_simp; ring
      rw [hd, norm_neg, Real.norm_of_nonneg (by
        have := sub_nonneg.2 hab; positivity)]
      calc R * (a - b) / (a * b) * ‖y‖ ≤ R * (a - b) / (a * b) * b := by
            gcongr
        _ = R / a * (a - b) := by field_simp
        _ ≤ R / a * ‖x - y‖ := by gcongr
    have h1 : R / a * ‖x - y‖ ≤ ‖x - y‖ := mul_le_of_le_one_left (norm_nonneg _) hRa
    linarith
  refine LipschitzWith.of_dist_le_mul fun x y => ?_
  rw [dist_eq_norm, dist_eq_norm, NNReal.coe_ofNat]
  rcases le_total (max R ‖y‖) (max R ‖x‖) with h | h
  · exact key x y h
  · rw [← norm_neg, neg_sub, ← norm_neg (x - y), neg_sub]
    exact key y x h

/-- **Grönwall, forwards.**  A solution on `[-T, T]` of a field with linear
growth `‖G x‖ ≤ C ‖x‖` satisfies `‖x(t)‖ ≤ ‖x(0)‖ e^{C t}` for `t ∈ [0, T]`. -/
theorem norm_le_of_linearGrowth_fwd {G : E → E} {C T : ℝ} (hG : ∀ x, ‖G x‖ ≤ C * ‖x‖)
    {f : ℝ → E} (hf : ∀ t ∈ Icc (-T) T, HasDerivWithinAt f (G (f t)) (Icc (-T) T) t) :
    ∀ t ∈ Icc 0 T, ‖f t‖ ≤ ‖f 0‖ * Real.exp (C * t) := by
  intro t ht
  have hc : ContinuousOn f (Icc 0 T) := fun s hs =>
    ((hf s ⟨by linarith [hs.1, hs.2], hs.2⟩).continuousWithinAt).mono
      (Icc_subset_Icc_left (by linarith [hs.1, hs.2]))
  have h := norm_le_gronwallBound_of_norm_deriv_right_le (ε := 0) hc
    (fun s hs => (hf s ⟨by linarith [hs.1, hs.2], hs.2.le⟩).mono_of_mem_nhdsWithin
      (Icc_mem_nhdsGE_of_mem ⟨by linarith [hs.1, hs.2], hs.2⟩))
    le_rfl (fun s _ => by simpa using hG (f s)) t ht
  simpa [gronwallBound_ε0] using h

/-- **Grönwall, both ways.**  A solution on `[-T, T]` of a field with linear
growth satisfies `‖x(t)‖ ≤ ‖x(0)‖ e^{C T}` on the whole interval. -/
theorem norm_le_of_linearGrowth {G : E → E} {C T : ℝ} (hC : 0 ≤ C)
    (hG : ∀ x, ‖G x‖ ≤ C * ‖x‖) {f : ℝ → E}
    (hf : ∀ t ∈ Icc (-T) T, HasDerivWithinAt f (G (f t)) (Icc (-T) T) t) :
    ∀ t ∈ Icc (-T) T, ‖f t‖ ≤ ‖f 0‖ * Real.exp (C * T) := by
  have hmaps : MapsTo (fun s : ℝ => -s) (Icc (-T) T) (Icc (-T) T) := fun s hs =>
    ⟨by linarith [hs.2], by linarith [hs.1]⟩
  have hb : ∀ s ∈ Icc (-T) T, HasDerivWithinAt (fun s => f (-s)) (-G (f (-s))) (Icc (-T) T) s := by
    intro s hs
    have := (hf (-s) (hmaps hs)).scomp s (hasDerivAt_neg s).hasDerivWithinAt hmaps
    simpa [Function.comp_def] using this
  have hG' : ∀ x, ‖-G x‖ ≤ C * ‖x‖ := fun x => by rw [norm_neg]; exact hG x
  intro t ht
  have hexp : ∀ s, s ≤ T → ‖f 0‖ * Real.exp (C * s) ≤ ‖f 0‖ * Real.exp (C * T) := fun s hs =>
    mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left hs hC)) (norm_nonneg _)
  rcases le_total 0 t with h0 | h0
  · exact (norm_le_of_linearGrowth_fwd hG hf t ⟨h0, ht.2⟩).trans (hexp t ht.2)
  · have := norm_le_of_linearGrowth_fwd hG' hb (-t) ⟨by linarith, by linarith [ht.1]⟩
    simp only [neg_neg, neg_zero] at this
    exact this.trans (hexp (-t) (by linarith [ht.1]))

end GlobalFlow
end Transformer
