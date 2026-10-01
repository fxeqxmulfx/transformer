/-
# Exact angular and chord updates on a sphere

arXiv:2606.25971v2, §2, §3.1 and §4.1.5. For a tangential update
whose norm matches the weight norm, projection gives a cosine and a
relative chord depending only on the LR. The chord is not equal to the LR.
-/

import Transformer.MagnitudeDirection.Section3_Sphere

noncomputable section

namespace Transformer.MagnitudeDirection

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Cosine of the angle between two nonzero directions,
arXiv:2606.25971v2, §2, `angle(W,W+Delta W)`. -/
def directionCosine (w v : E) : ℝ := inner ℝ w v / (‖w‖ * ‖v‖)

/-- Exact pre-projection norm for a normalized tangential update,
arXiv:2606.25971v2, §2 and §3.1. -/
theorem tangent_candidate_norm (w u : E) (eta : ℝ)
    (horth : inner ℝ w u = 0) (hu : ‖u‖ = ‖w‖) :
    ‖w + eta • u‖ = ‖w‖ * Real.sqrt (1 + eta ^ 2) := by
  apply (sq_eq_sq₀ (norm_nonneg _) (mul_nonneg (norm_nonneg _) (Real.sqrt_nonneg _))).mp
  rw [norm_add_sq_real, inner_smul_right, horth, norm_smul, Real.norm_eq_abs,
    mul_pow, sq_abs, hu, mul_pow, Real.sq_sqrt (by positivity)]
  ring

/-- Equal-length orthogonal directions exist, arXiv:2606.25971v2, §2–3. -/
example : inner ℝ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
    (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 0 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ =
      ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ := by
  simp [EuclideanSpace.inner_single_left]

/-- Exact cosine after retracting a tangential normalized update,
arXiv:2606.25971v2, §3.1. It is independent of weight magnitude and
ambient dimension, under the stated normalization and tangency conditions. -/
theorem tangent_projected_cosine (w u : E) (eta : ℝ)
    (hw : w ≠ 0) (horth : inner ℝ w u = 0) (hu : ‖u‖ = ‖w‖) :
    directionCosine w (sphereProject ‖w‖ (w + eta • u)) =
      1 / Real.sqrt (1 + eta ^ 2) := by
  have hn := norm_pos_iff.mpr hw
  have hs : 0 < Real.sqrt (1 + eta ^ 2) := Real.sqrt_pos.mpr (by positivity)
  have hC : w + eta • u ≠ 0 := by
    apply norm_pos_iff.mp
    rw [tangent_candidate_norm w u eta horth hu]
    exact mul_pos hn hs
  unfold directionCosine
  rw [sphereProject_norm ‖w‖ _ hn.le hC]
  rw [sphereProject, inner_smul_right, inner_add_right, inner_smul_right, horth,
    real_inner_self_eq_norm_sq, tangent_candidate_norm w u eta horth hu]
  field_simp
  ring

/-- The cosine theorem's nonzero and orthogonality assumptions are satisfiable,
arXiv:2606.25971v2, §3.1. -/
example : EuclideanSpace.single (0 : Fin 2) (1 : ℝ) ≠ 0 ∧
    inner ℝ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 0 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ =
      ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ := by
  simp [EuclideanSpace.inner_single_left]

/-- Exact post-projection relative displacement for a tangential normalized
update: its square is `2 - 2/sqrt(1+eta²)`, not `eta²`.
This corrects the literal exact-step-size wording while proving the precise
LR-only relationship. Source: arXiv:2606.25971v2, §3.1 and §4.1.5. -/
theorem tangent_projected_chord_sq (w u : E) (eta : ℝ)
    (hw : w ≠ 0) (horth : inner ℝ w u = 0) (hu : ‖u‖ = ‖w‖) :
    relativeUpdate w (sphereProject ‖w‖ (w + eta • u) - w) ^ 2 =
      2 - 2 / Real.sqrt (1 + eta ^ 2) := by
  have hn := norm_pos_iff.mpr hw
  have hs : 0 < Real.sqrt (1 + eta ^ 2) := Real.sqrt_pos.mpr (by positivity)
  have hC : w + eta • u ≠ 0 := by
    apply norm_pos_iff.mp
    rw [tangent_candidate_norm w u eta horth hu]
    exact mul_pos hn hs
  have hcos := tangent_projected_cosine w u eta hw horth hu
  unfold directionCosine at hcos
  rw [sphereProject_norm ‖w‖ _ hn.le hC] at hcos
  unfold relativeUpdate
  rw [div_pow, norm_sub_sq_real, sphereProject_norm ‖w‖ _ hn.le hC,
    real_inner_comm w (sphereProject ‖w‖ (w + eta • u))]
  field_simp at hcos ⊢
  nlinarith

/-- The chord theorem's hypotheses hold on the two-dimensional unit sphere,
arXiv:2606.25971v2, §3.1. -/
example : EuclideanSpace.single (0 : Fin 2) (1 : ℝ) ≠ 0 ∧
    inner ℝ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 0 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ =
      ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ := by
  simp [EuclideanSpace.inner_single_left]

end Transformer.MagnitudeDirection
