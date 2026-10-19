module

public import Transformer.Causal.Section5_CausalWeights
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# The inward velocity of the closest causal predecessor

ArXiv:2411.04990v2, §5, proof of `lemma:meta`, after `eq:bddotphi`.
Only interactions pointing toward the center can decrease its chord gap
to the closest predecessor. Circle geometry places each such interaction
beyond that gap, so its weighted projected norm is at most `h(r)`.

We apply that geometry to the actual normalized field, then combine the
predecessor's inward bound with the center's speed bound. The resulting
squared-chord derivative bound requires no polar-coordinate chart.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n : ℕ}

/-- **The closest predecessor cannot be pulled inward faster than `i h(r)`.**
The inner-product ordering says that `i` is a closest predecessor to `k`.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, the estimate on
`-φ̇_{r*}` following `eq:bddotphi`. -/
theorem inner_chord_causalDrift_ge (β : ℝ) (hβ : 0 < β) (r : ℝ) (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (X : SphereTuple 2 n) (k i : Idx n) (hik : i < k)
    (hclosest : ∀ j : Idx n, j < k →
      inner ℝ (X k : EucSpace 2) (X j : EucSpace 2) ≤
        inner ℝ (X k : EucSpace 2) (X i : EucSpace 2))
    (hgap : r ≤ ‖(X k : EucSpace 2) - (X i : EucSpace 2)‖) :
    -((i : ℝ) * ‖(X i : EucSpace 2) - (X k : EucSpace 2)‖ * h_pot β r) ≤
      inner ℝ ((X i : EucSpace 2) - (X k : EucSpace 2)) (causalDrift β X i) := by
  let w : EucSpace 2 := (X i : EucSpace 2) - (X k : EucSpace 2)
  have hr0 : 0 ≤ r := (Real.arccos_nonneg _).trans hrpeak
  have hH : 0 ≤ h_pot β r := h_pot_nonneg β ⟨hr0, hrπ⟩
  have hterm (j : Idx n) (hj : j < i) :
      -(‖w‖ * h_pot β r) ≤
        Real.exp (β * (inner ℝ (X i : EucSpace 2) (X j : EucSpace 2) - 1)) *
          inner ℝ w (proj 2 (X i : EucSpace 2) (X j : EucSpace 2)) := by
    by_cases hs : 0 ≤ inner ℝ w (proj 2 (X i : EucSpace 2) (X j : EucSpace 2))
    · exact (neg_nonpos.mpr (mul_nonneg (norm_nonneg w) hH)).trans
        (mul_nonneg (Real.exp_pos _).le hs)
    · have hin : inner ℝ w (proj 2 (X i : EucSpace 2) (X j : EucSpace 2)) < 0 := lt_of_not_ge hs
      dsimp [w] at hin
      rw [circle_chord_inner_proj] at hin
      have hinner := inner_le_of_circle_inward (X k) (X i) (X j)
        (hclosest j (hj.trans hik)) (by linarith)
      have hsep := hgap.trans (circle_chord_le_of_inner_le (X k) (X i) (X j) hinner)
      have hweight := circle_weighted_projection_le β hβ r hrπ hrpeak (X i) (X j) hsep
      have hcs := neg_le_of_abs_le (abs_real_inner_le_norm w
        (proj 2 (X i : EucSpace 2) (X j : EucSpace 2)))
      calc
        -(‖w‖ * h_pot β r) ≤
            -‖w‖ * (Real.exp (β * (inner ℝ (X i : EucSpace 2) (X j : EucSpace 2) - 1)) *
              ‖proj 2 (X i : EucSpace 2) (X j : EucSpace 2)‖) := by
          simpa only [neg_mul] using mul_le_mul_of_nonpos_left hweight (neg_nonpos.mpr (norm_nonneg w))
        _ = Real.exp (β * (inner ℝ (X i : EucSpace 2) (X j : EucSpace 2) - 1)) *
            -(‖w‖ * ‖proj 2 (X i : EucSpace 2) (X j : EucSpace 2)‖) := by ring
        _ ≤ _ := mul_le_mul_of_nonneg_left hcs (Real.exp_pos _).le
  have hnum : -((i : ℝ) * ‖w‖ * h_pot β r) ≤
      ∑ j ∈ Finset.Iio i,
        Real.exp (β * (inner ℝ (X i : EucSpace 2) (X j : EucSpace 2) - 1)) *
          inner ℝ w (proj 2 (X i : EucSpace 2) (X j : EucSpace 2)) := by
    calc
      _ = ∑ _j ∈ Finset.Iio i, -(‖w‖ * h_pot β r) := by
        rw [Finset.sum_const, Fin.card_Iio, nsmul_eq_mul]
        ring
      _ ≤ _ := Finset.sum_le_sum fun j hj => hterm j (Finset.mem_Iio.mp hj)
  have hZ := inv_causalNormalizer_bounds β X i
  have hL : -((i : ℝ) * ‖w‖ * h_pot β r) ≤ 0 := neg_nonpos.mpr (by positivity)
  change _ ≤ inner ℝ w (causalDrift β X i)
  rw [causalDrift, real_inner_smul_right, inner_sum]
  simp_rw [real_inner_smul_right]
  have hscaled : -((i : ℝ) * ‖w‖ * h_pot β r) ≤
      (causalNormalizer β X i)⁻¹ * -((i : ℝ) * ‖w‖ * h_pot β r) := by
    simpa only [one_mul] using mul_le_mul_of_nonpos_right hZ.2 hL
  exact hscaled.trans (mul_le_mul_of_nonneg_left hnum hZ.1)

/-- **The relative chord velocity has the source's `2k h(r)` lower bound.**
This combines the inward bound for the closest predecessor with the speed
bound for the center. Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`. -/
theorem inner_chord_relativeDrift_ge (β : ℝ) (hβ : 0 < β) (r : ℝ) (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (X : SphereTuple 2 n) (k i : Idx n) (hik : i < k)
    (hclosest : ∀ j : Idx n, j < k →
      inner ℝ (X k : EucSpace 2) (X j : EucSpace 2) ≤
        inner ℝ (X k : EucSpace 2) (X i : EucSpace 2))
    (hgap : ∀ j : Idx n, j < k → r ≤ ‖(X k : EucSpace 2) - (X j : EucSpace 2)‖) :
    -(2 * (k : ℝ) * ‖(X i : EucSpace 2) - (X k : EucSpace 2)‖ * h_pot β r) ≤
      inner ℝ ((X i : EucSpace 2) - (X k : EucSpace 2))
        (causalDrift β X i - causalDrift β X k) := by
  have hpred := inner_chord_causalDrift_ge β hβ r hrπ hrpeak X k i hik hclosest (hgap i hik)
  have hc := (real_inner_le_norm ((X i : EucSpace 2) - (X k : EucSpace 2))
    (causalDrift β X k)).trans
      (mul_le_mul_of_nonneg_left (norm_causalDrift_le β hβ r hrπ hrpeak X k hgap) (norm_nonneg _))
  have hH : 0 ≤ h_pot β r :=
    h_pot_nonneg β ⟨(Real.arccos_nonneg _).trans hrpeak, hrπ⟩
  have hikR : (i : ℝ) ≤ (k : ℝ) := by exact_mod_cast (Fin.le_def.mp hik.le)
  have hcount := mul_le_mul_of_nonneg_right hikR
    (mul_nonneg (norm_nonneg ((X i : EucSpace 2) - (X k : EucSpace 2))) hH)
  rw [inner_sub_right]
  nlinarith

/-- **Squared-chord decay for the actual causal trajectory.** At a closest
predecessor its squared distance has derivative at least `-4k D h(r)`.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, the inequality
`-Δ̇ ≤ 2s_k h(Δ)` expressed through the actual chord distance. -/
theorem squared_chord_derivative_lower (β : ℝ) (hβ : 0 < β) (r : ℝ) (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (X : ℝ → SphereTuple 2 n)
    (hX : CSA 2 n β (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) X) (t : ℝ) (k i : Idx n) (hik : i < k)
    (hclosest : ∀ j : Idx n, j < k →
      inner ℝ (X t k : EucSpace 2) (X t j : EucSpace 2) ≤
        inner ℝ (X t k : EucSpace 2) (X t i : EucSpace 2))
    (hgap : ∀ j : Idx n, j < k → r ≤ ‖(X t k : EucSpace 2) - (X t j : EucSpace 2)‖) :
    ∃ a : ℝ, HasDerivAt (fun s => ‖(X s i : EucSpace 2) - (X s k : EucSpace 2)‖ ^ 2) a t ∧
      -(4 * (k : ℝ) * ‖(X t i : EucSpace 2) - (X t k : EucSpace 2)‖ * h_pot β r) ≤ a := by
  refine ⟨_, ((hasDerivAt_causalDrift β X hX t i).sub
    (hasDerivAt_causalDrift β X hX t k)).norm_sq, ?_⟩
  have h := inner_chord_relativeDrift_ge β hβ r hrπ hrpeak (X t) k i hik hclosest hgap
  simp only [Pi.sub_apply]
  nlinarith

/-- All hypotheses of `inner_chord_causalDrift_ge`,
`inner_chord_relativeDrift_ge`, and `squared_chord_derivative_lower` are
simultaneously satisfied: two stationary antipodal tokens, `k=1`, `i=0`,
`β=1`, and `r=2`. The predecessor of the center is a genuine token, and
the actual causal equation is verified below. -/
example : ∃ X : ℝ → SphereTuple 2 2,
    (0 : ℝ) < 1 ∧ (2 : ℝ) ≤ π ∧
    Real.arccos ((-1 + Real.sqrt (4 * (1 : ℝ) ^ 2 + 1)) / (2 * 1)) ≤ 2 ∧
    (0 : Idx 2) < 1 ∧
    (∀ j : Idx 2, j < 1 → inner ℝ (X 0 1 : EucSpace 2) (X 0 j : EucSpace 2) ≤
      inner ℝ (X 0 1 : EucSpace 2) (X 0 0 : EucSpace 2)) ∧
    (∀ j : Idx 2, j < 1 → (2 : ℝ) ≤ ‖(X 0 1 : EucSpace 2) - (X 0 j : EucSpace 2)‖) ∧
    CSA 2 2 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) X := by
  let x : SSphere 2 := basePoint 1
  let y : SSphere 2 := ⟨-(x : EucSpace 2), by simp [mem_sphere_zero_iff_norm.mp x.2]⟩
  let Y : SphereTuple 2 2 := ![x, y]
  have hyx : proj 2 (y : EucSpace 2) (x : EucSpace 2) = 0 := by
    simp [y, proj, inner_neg_left, mem_sphere_zero_iff_norm.mp x.2]
  have hzero (k : Idx 2) : causalDrift 1 Y k = 0 := by
    unfold causalDrift
    have hs : ∑ j ∈ Finset.Iio k,
        Real.exp (1 * (inner ℝ (Y k : EucSpace 2) (Y j : EucSpace 2) - 1)) •
          proj 2 (Y k : EucSpace 2) (Y j : EucSpace 2) = 0 := by
      apply Finset.sum_eq_zero
      intro j hj
      have hjk := Finset.mem_Iio.mp hj
      fin_cases k <;> fin_cases j <;> norm_num at hjk
      simp [Y, hyx]
    rw [hs, smul_zero]
  obtain ⟨τ, hτ0, hτπ, hcos, _, hτ1, _⟩ := h_pot_unimodal 1 le_rfl
  refine ⟨fun _ => Y, one_pos, by linarith [pi_gt_three], ?_, by decide, ?_, ?_, ?_⟩
  · rw [← hcos, Real.arccos_cos hτ0.le hτπ.le, Real.one_rpow] at *
    linarith
  · intro j hj
    have hj0 : j = 0 := by apply Fin.ext; change (j : ℕ) < 1 at hj; omega
    rw [hj0]
  · intro j hj
    have hj0 : j = 0 := by apply Fin.ext; change (j : ℕ) < 1 at hj; omega
    rw [hj0]
    change 2 ≤ ‖-(x : EucSpace 2) - (x : EucSpace 2)‖
    rw [sub_eq_add_neg, ← neg_add, norm_neg, ← two_smul ℝ (x : EucSpace 2), norm_smul,
      mem_sphere_zero_iff_norm.mp x.2]
    norm_num
  · intro t k
    have hfield := (causalDrift_eq_original 1 Y k).trans (hzero k)
    apply (hasDerivAt_const t (Y k : EucSpace 2)).congr_deriv
    simpa only [ContinuousLinearMap.id_apply] using hfield.symm

end Transformer.Causal
