/-
# Sphere projection and normalized updates

arXiv:2606.25971v2, §3.1, Algorithms 1–2. The candidate must be
nonzero: the displayed division does not project zero onto a positive sphere.
-/

import Transformer.MagnitudeDirection.Section2_Interference
import Transformer.MagnitudeDirection.Section3_Factorization

noncomputable section

namespace Transformer.MagnitudeDirection

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Radial projection onto the sphere, arXiv:2606.25971v2, §3.1,
Algorithm 1, line 5. At zero Lean's total division returns zero. -/
def sphereProject (c : ℝ) (x : E) : E := (c / ‖x‖) • x

/-- The projection preserves exactly the prescribed magnitude for a nonzero
candidate, arXiv:2606.25971v2, §3.1, Algorithms 1–2. -/
theorem sphereProject_norm (c : ℝ) (x : E) (hc : 0 ≤ c) (hx : x ≠ 0) :
    ‖sphereProject c x‖ = c := by
  rw [sphereProject, norm_smul_of_nonneg (div_nonneg hc (norm_nonneg x))]
  exact div_mul_cancel₀ c (norm_ne_zero_iff.mpr hx)

/-- The projection hypotheses hold, arXiv:2606.25971v2, §3.1. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 := by norm_num

/-- An already normalized direction is fixed by projection,
arXiv:2606.25971v2, §3.1. -/
theorem sphereProject_fixed (c : ℝ) (x : E) (hc : 0 < c) (hx : ‖x‖ = c) :
    sphereProject c x = x := by
  simp [sphereProject, hx, hc.ne']

/-- Fixed points exist, arXiv:2606.25971v2, §3.1. -/
example : (0 : ℝ) < 1 ∧ ‖(1 : ℝ)‖ = (1 : ℝ) := by norm_num

/-- Positive radial rescaling does not change the projected direction,
arXiv:2606.25971v2, §3.1. -/
theorem sphereProject_rescale (c a : ℝ) (x : E) (ha : 0 < a) :
    sphereProject c (a • x) = sphereProject c x := by
  unfold sphereProject
  rw [norm_smul_of_nonneg ha.le, smul_smul]
  congr 1
  field_simp

/-- Positive radial factors exist, arXiv:2606.25971v2, §3.1. -/
example : (0 : ℝ) < 2 := by norm_num

/-- A candidate normalized relative to the direction has pre-projection
relative update exactly `eta`. This is the precise scope of the LR statement
in arXiv:2606.25971v2, §3.1; the retracted displacement is different. -/
theorem normalized_relativeUpdate (w u : E) (eta : ℝ)
    (heta : 0 ≤ eta) (hw : w ≠ 0) (hu : ‖u‖ = ‖w‖) :
    relativeUpdate w (eta • u) = eta := by
  rw [relativeUpdate, norm_smul_of_nonneg heta, hu]
  exact mul_div_cancel_right₀ eta (norm_ne_zero_iff.mpr hw)

/-- Normalized nonzero updates exist, arXiv:2606.25971v2, §3.1. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ ‖(1 : ℝ)‖ = ‖(1 : ℝ)‖ := by norm_num

/-- The zero candidate refutes unconditional projection onto a positive sphere.
The missing nonzero-candidate condition is recorded explicitly.
Source: arXiv:2606.25971v2, Algorithms 1–2, projection line. -/
theorem zero_candidate_counterexample :
    ‖sphereProject 1 (0 : ℝ)‖ ≠ (1 : ℝ) := by
  norm_num [sphereProject]

/-- A nonzero radial update of normalized size can cause no directional
change at all. Thus “the relative update tracks the schedule exactly” cannot
mean the *post-projection* displacement for arbitrary normalized optimizers.
Source: arXiv:2606.25971v2, §3.1 and §4.1.5. -/
theorem radial_update_counterexample :
    relativeUpdate (1 : ℝ) (1 : ℝ) = 1 ∧
      sphereProject 1 ((1 : ℝ) + 1) = 1 ∧
      relativeUpdate (1 : ℝ) (sphereProject 1 ((1 : ℝ) + 1) - 1) = 0 := by
  norm_num [relativeUpdate, sphereProject]

/-- A radial step can have any positive, arbitrarily small relative size
and still leave the direction exactly unchanged. Thus the approximation
in §2 also requires control of the radial component, not just a small LR.
Source: arXiv:2606.25971v2, §2, “Direction change depends on magnitude”. -/
theorem radial_update_any_size (w : E) (eta : ℝ) (hw : w ≠ 0) (heta : 0 < eta) :
    relativeUpdate w (eta • w) = eta ∧ sphereProject ‖w‖ (w + eta • w) = w := by
  refine ⟨normalized_relativeUpdate w w eta heta.le hw rfl, ?_⟩
  have he : w + eta • w = (1 + eta) • w := by simp [add_smul]
  rw [he, sphereProject_rescale _ _ w (by linarith)]
  exact sphereProject_fixed ‖w‖ w (norm_pos_iff.mpr hw) rfl

/-- Nonzero weights and small positive radial steps exist,
arXiv:2606.25971v2, §2. -/
example : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 / 100 := by norm_num

/-- The actual retracted relative displacement is bounded by two,
irrespective of LR. Source: arXiv:2606.25971v2, §3.1, corrected scope
of “the effective step size is now just the LR”. -/
theorem retracted_relativeUpdate_le_two (c : ℝ) (w v : E)
    (hc : 0 < c) (hw : ‖w‖ = c) (hv : v ≠ 0) :
    relativeUpdate w (sphereProject c v - w) ≤ 2 := by
  have hn := norm_sub_le (sphereProject c v) w
  rw [sphereProject_norm c v hc.le hv, hw] at hn
  unfold relativeUpdate
  rw [hw, div_le_iff₀ hc]
  linarith

/-- Retraction-bound hypotheses hold, arXiv:2606.25971v2, §3.1. -/
example : (0 : ℝ) < 1 ∧ ‖(1 : ℝ)‖ = (1 : ℝ) ∧ (2 : ℝ) ≠ 0 := by norm_num

variable {m n : ℕ}

/-- Matrix form of the projection in Algorithms 1–2,
arXiv:2606.25971v2, §3.1 and Appendix A. -/
def matrixProject (c : ℝ) (W : Matrix (Fin m) (Fin n) ℝ) :
    Matrix (Fin m) (Fin n) ℝ := (c / frobeniusNorm W) • W

/-- The matrix projection is the Euclidean radial projection of all entries,
arXiv:2606.25971v2, §3.1. -/
theorem flatten_matrixProject (c : ℝ) (W : Matrix (Fin m) (Fin n) ℝ) :
    flatten (matrixProject c W) = sphereProject c (flatten W) := by
  exact flatten_smul _ W

/-- The matrix direction stays on its Frobenius sphere provided the candidate
is nonzero, arXiv:2606.25971v2, Algorithms 1–2. -/
theorem matrixProject_norm (c : ℝ) (W : Matrix (Fin m) (Fin n) ℝ)
    (hc : 0 ≤ c) (hW : W ≠ 0) : frobeniusNorm (matrixProject c W) = c := by
  rw [frobeniusNorm, flatten_matrixProject]
  apply sphereProject_norm c (flatten W) hc
  exact norm_pos_iff.mp ((frobeniusNorm_pos_iff W).mpr hW)

/-- Matrix projection has valid inputs, arXiv:2606.25971v2, Algorithms 1–2. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ 0 := by
  norm_num

end Transformer.MagnitudeDirection
