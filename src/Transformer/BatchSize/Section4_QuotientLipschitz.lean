/-
# Positive-noise normalization preserves global Lipschitz bounds

arXiv:2506.12543v1, Section 4.3, equation (3), Theorem 1.
The signed drift divides the bounded gradient by a uniformly positive,
state-dependent standard deviation. Division in the ordinary real field
requires these bounds; it is not an isometric group operation.
-/

import Transformer.BatchSize.Section4_CoefficientRegularity

noncomputable section

namespace Transformer.BatchSize

/-- A bounded Lipschitz signal divided by a uniformly positive Lipschitz
noise scale remains globally Lipschitz, as needed for Section 4.3 (3).
The constant controls both the numerator and the varying denominator. -/
theorem positive_quotient_lipschitz {E : Type*} [PseudoMetricSpace E]
    (g s : E → ℝ) (Kg Ks : NNReal) (M c : ℝ)
    (hg : LipschitzWith Kg g) (hs : LipschitzWith Ks s)
    (hM : 0 ≤ M) (hc : 0 < c) (hgb : ∀ x, |g x| ≤ M) (hsb : ∀ x, c ≤ s x) :
    LipschitzWith (Real.toNNReal ((Kg : ℝ) / c + M * Ks / c ^ 2))
      (fun x => g x / s x) := by
  have hK : 0 ≤ (Kg : ℝ) / c + M * Ks / c ^ 2 := by positivity
  apply LipschitzWith.of_dist_le_mul
  intro x y
  have hpx : 0 < s x := hc.trans_le (hsb x)
  have hpy : 0 < s y := hc.trans_le (hsb y)
  have hgxy : |g x - g y| ≤ (Kg : ℝ) * dist x y := by
    simpa only [Real.dist_eq] using hg.dist_le_mul x y
  have hsxy : |s y - s x| ≤ (Ks : ℝ) * dist x y := by
    rw [abs_sub_comm]
    simpa only [Real.dist_eq] using hs.dist_le_mul x y
  have hprod : c ^ 2 ≤ s x * s y := by
    simpa only [pow_two] using mul_le_mul (hsb x) (hsb y) hc.le hpx.le
  have heq : g x / s x - g y / s y =
      (g x - g y) / s x + g y * (s y - s x) / (s x * s y) := by
    field_simp
    ring
  rw [Real.dist_eq, Real.coe_toNNReal _ hK, heq]
  calc
    |(g x - g y) / s x + g y * (s y - s x) / (s x * s y)| ≤
        |(g x - g y) / s x| + |g y * (s y - s x) / (s x * s y)| := abs_add_le _ _
    _ = |g x - g y| / s x + |g y| * |s y - s x| / (s x * s y) := by
      rw [abs_div, abs_of_pos hpx, abs_div, abs_mul, abs_of_pos (mul_pos hpx hpy)]
    _ ≤ (Kg : ℝ) * dist x y / c + M * ((Ks : ℝ) * dist x y) / c ^ 2 := by
      apply add_le_add
      · exact (div_le_div_of_nonneg_right hgxy hpx.le).trans
          (div_le_div_of_nonneg_left (by positivity) hc (hsb x))
      · exact (div_le_div_of_nonneg_right
          (mul_le_mul (hgb y) hsxy (abs_nonneg _) hM) (mul_pos hpx hpy).le).trans
          (div_le_div_of_nonneg_left (by positivity) (sq_pos_of_pos hc) hprod)
    _ = ((Kg : ℝ) / c + M * Ks / c ^ 2) * dist x y := by ring

/-- Joint nonvacuity of the quotient hypotheses, Section 4.3 (3):
a varying bounded signal and a positive constant noise scale. -/
example : LipschitzWith 1 Real.sin ∧ LipschitzWith 0 (fun _ : ℝ => (1 : ℝ)) ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (∀ x, |Real.sin x| ≤ 1) ∧
      (∀ _ : ℝ, (1 : ℝ) ≤ 1) := by
  exact ⟨Real.lipschitzWith_sin, LipschitzWith.const 1, by norm_num, by norm_num,
    Real.abs_sin_le_one, fun _ => le_rfl⟩

/-- Finite coordinate Lipschitz bounds give a Euclidean vector bound,
Section 4.3 (2)--(3). The continuous linear equivalence accounts for the
change from the coordinate maximum norm to the Euclidean norm. -/
theorem coordinate_lipschitz_to_euclidean {E : Type*} [PseudoMetricSpace E]
    {d : ℕ} (v : E → Fin d → ℝ) (K : NNReal)
    (hv : ∀ k, LipschitzWith K (fun x => v x k)) :
    LipschitzWith
      (‖(PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.toContinuousLinearMap‖₊ * K)
      (fun x => WithLp.toLp 2 (v x)) := by
  have hp : LipschitzWith K v := by
    apply LipschitzWith.of_dist_le_mul
    intro x y
    rw [dist_eq_norm]
    exact (pi_norm_le_iff_of_nonneg (by positivity)).mpr fun k => by
      simpa only [Real.dist_eq, Real.norm_eq_abs, Pi.sub_apply] using (hv k).dist_le_mul x y
  exact (PiLp.continuousLinearEquiv 2 ℝ
    (fun _ : Fin d => ℝ)).symm.toContinuousLinearMap.lipschitzWith.comp hp

/-- Joint nonvacuity of the coordinate bounds, Section 4.3. -/
example : ∀ k : Fin 2, LipschitzWith 1 (fun x : ℝ => Real.sin (x + k)) := by
  intro k
  have ht : LipschitzWith 1 (fun x : ℝ => x + k) := by
    apply LipschitzWith.of_dist_le_mul
    intro x y
    simp only [Real.dist_eq, NNReal.coe_one, one_mul, add_sub_add_right_eq_sub, le_refl]
  simpa only [one_mul, Function.comp_def] using Real.lipschitzWith_sin.comp ht

end Transformer.BatchSize
