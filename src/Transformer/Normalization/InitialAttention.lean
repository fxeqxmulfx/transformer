/-
# Deterministic attention estimates at initialization

The softmax denominator and contracted query used in the proof of
Theorem 4.2, §4.2 and Appendix B of arXiv:2510.22026v2.
-/

import Transformer.Normalization.InitialBias
import Transformer.Normalization.Velocities
import Mathlib.Analysis.InnerProductSpace.Adjoint

open scoped BigOperators
open MeasureTheory

namespace Transformer.Normalization

variable {d n : ℕ}

/-- The query representing the bilinear attention score. Source:
arXiv:2510.22026v2, §4.2, `Q^T K` in Theorem 4.2. -/
noncomputable def contractedQuery (Q K : ParamMatrix d) (w : SSphere d) : EucSpace d :=
  K.adjoint (Q (w : EucSpace d))

/-- The contracted query reproduces the attention score. Source:
arXiv:2510.22026v2, Appendix B, definition of `X_k`. -/
theorem contractedQuery_inner (Q K : ParamMatrix d) (w : SSphere d) (x : EucSpace d) :
    inner (𝕜 := ℝ) (contractedQuery Q K w) x = inner (𝕜 := ℝ) (Q (w : EucSpace d)) (K x) :=
  K.adjoint_inner_left x (Q (w : EucSpace d))

/-- The paper's bilinear operator bound contracts a unit query. Source:
arXiv:2510.22026v2, §4.2, the operator hypothesis of Theorem 4.2. -/
theorem norm_contractedQuery_le (Q K : ParamMatrix d)
    (hQK : ∀ x y : EucSpace d, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖)
    (w : SSphere d) : ‖contractedQuery Q K w‖ ≤ 1 := by
  have hw : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.property
  have h := hQK (w : EucSpace d) (contractedQuery Q K w)
  rw [← contractedQuery_inner, real_inner_self_eq_norm_sq, abs_of_nonneg (sq_nonneg _),
    hw, one_mul] at h
  nlinarith [norm_nonneg (contractedQuery Q K w)]

/-- Zero query and key maps realize the operator hypothesis. -/
example : ∀ x y : EucSpace 1,
    |inner (𝕜 := ℝ) ((0 : ParamMatrix 1) x) ((0 : ParamMatrix 1) y)| ≤ ‖x‖ * ‖y‖ := by
  intro x y
  simp
  positivity

/-- The attention numerator before application of the value map. Source:
arXiv:2510.22026v2, Appendix B, proof of Theorem 4.2. -/
noncomputable def initialNumerator (Q K : ParamMatrix d) (Θ : SphereTuple d n) (j : Fin n) :
    EucSpace d := ∑ k, tiltedSphere (contractedQuery Q K (Θ j)) (Θ k)

/-- The numerator is continuous in the initialization. Source:
arXiv:2510.22026v2, Appendix B, numerator event in Theorem 4.2. -/
theorem continuous_initialNumerator (Q K : ParamMatrix d) (j : Fin n) :
    Continuous (fun Θ : SphereTuple d n => initialNumerator Q K Θ j) := by
  unfold initialNumerator tiltedSphere contractedQuery
  fun_prop

/-- The valid deterministic denominator bound is `Z_j ≥ n/3`.
The Appendix B bounds `n-1+e` and `n-1+e^{-1}` omit the remaining scores;
this estimate instead follows directly from every score lying in `[-1,1]`.
Source: arXiv:2510.22026v2, Appendix B, proof of Theorem 4.2. -/
theorem attention_partition_ge (Q K : ParamMatrix d)
    (hQK : ∀ x y : EucSpace d, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖)
    (Θ : SphereTuple d n) (j : Fin n) :
    (n : ℝ) / 3 ≤ ∑ k, Real.exp (inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ k))) := by
  have he : (1 : ℝ) / 3 ≤ Real.exp (-1) := by
    rw [Real.exp_neg, inv_eq_one_div]
    apply (div_le_div_iff₀ (by norm_num : (0 : ℝ) < 3) (Real.exp_pos 1)).mpr
    linarith [Real.exp_one_lt_d9]
  calc (n : ℝ) / 3 = ∑ _ : Fin n, (1 : ℝ) / 3 := by simp; ring
       _ ≤ _ := Finset.sum_le_sum fun k _ => by
         have h := hQK (Θ j) (Θ k)
         have hj : ‖(Θ j : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (Θ j).property
         have hk : ‖(Θ k : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (Θ k).property
         rw [hj, hk, one_mul] at h
         exact he.trans (Real.exp_le_exp.mpr (neg_le_of_abs_le h))

/-- Zero query and key maps realize the operator hypothesis. -/
example : ∀ x y : EucSpace 1,
    |inner (𝕜 := ℝ) ((0 : ParamMatrix 1) x) ((0 : ParamMatrix 1) y)| ≤ ‖x‖ * ‖y‖ := by
  intro x y
  simp
  positivity

/-- Controlling the unnormalized sum controls attention. Source:
arXiv:2510.22026v2, §4.2 and Appendix B, Theorem 4.2. -/
theorem norm_attentionVec_le_initialNumerator (hn : 0 < n) (Q K V : ParamMatrix d)
    (hQK : ∀ x y : EucSpace d, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖)
    (hV : ‖V‖ ≤ 1) (Θ : SphereTuple d n) (j : Fin n) :
    ‖attentionVec d n 1 Q K V (tupleCoe Θ) j‖ ≤
      3 / (n : ℝ) * ‖initialNumerator Q K Θ j‖ := by
  let Z := ∑ k : Fin n, Real.exp (inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ k)))
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hZ : (n : ℝ) / 3 ≤ Z := attention_partition_ge Q K hQK Θ j
  have hZpos : 0 < Z := lt_of_lt_of_le (by positivity) hZ
  have hnum : (∑ k, Real.exp (inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ k))) • V (Θ k)) =
      V (initialNumerator Q K Θ j) := by
    unfold initialNumerator tiltedSphere
    rw [map_sum]
    simp only [map_smul, contractedQuery_inner]
  have hVnum : ‖V (initialNumerator Q K Θ j)‖ ≤ ‖initialNumerator Q K Θ j‖ :=
    (V.le_opNorm _).trans (by nlinarith [norm_nonneg (initialNumerator Q K Θ j)])
  have hcoeff : Z⁻¹ ≤ 3 / (n : ℝ) := by
    apply (inv_le_inv₀ hZpos (by positivity : (0 : ℝ) < (n : ℝ) / 3)).mpr hZ |>.trans
    exact le_of_eq (by field_simp)
  simp only [attentionVec, tupleCoe, one_mul, hnum]
  change ‖Z⁻¹ • V (initialNumerator Q K Θ j)‖ ≤ _
  rw [norm_smul, norm_inv, Real.norm_eq_abs, abs_of_pos hZpos]
  exact (mul_le_mul_of_nonneg_left hVnum (inv_pos.mpr hZpos).le).trans
    (mul_le_mul_of_nonneg_right hcoeff (norm_nonneg _))

/-- Positive token count and zero parameter maps realize the hypotheses. -/
example : 0 < (2 : ℕ) ∧
    (∀ x y : EucSpace 1,
      |inner (𝕜 := ℝ) ((0 : ParamMatrix 1) x) ((0 : ParamMatrix 1) y)| ≤ ‖x‖ * ‖y‖) ∧
    ‖(0 : ParamMatrix 1)‖ ≤ 1 := by
  refine ⟨by decide, ?_, by simp⟩
  intro x y
  simp
  positivity

end Transformer.Normalization
