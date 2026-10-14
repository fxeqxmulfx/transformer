/-
# Normalizing the original orthogonal residual

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The next approximation is the actual normalized residual of the
original point against the actual previous orthonormal frame.
A projection of norm below one guarantees a nonzero residual;
its normalization is a genuine unit vector orthogonal to the frame.

Normalization changes the point by at most twice the projection norm.
This proves the original radius `sqrt(4 log(d)/d)` from projection
radius `sqrt(log(d)/d)`, including the normalization factor omitted
in the survey's displayed distance estimate. The construction is
total on ambient vectors, taking a zero residual to zero, and is
measurable everywhere. Unit norm is proved on the actual success
event, rather than assumed of failed prefixes.
-/

import Transformer.Metastability.Section4_UniformProjectionScale
import Mathlib.MeasureTheory.Group.Arithmetic
import Mathlib.Data.Fin.Tuple.Basic

open Real MeasureTheory

namespace Transformer.Metastability

/-- The actual normalized residual against the original previous frame.
Source: arXiv:2410.06833v1, §4, `prop: concentration unif`;
arXiv:2312.10794v5, Appendix D, Step 2. A zero residual maps to zero. -/
noncomputable def normalizedFrameResidual {d m : ℕ} (v : Fin m → EucSpace d)
    (x : EucSpace d) : EucSpace d :=
  ‖x - frameProjection v x‖⁻¹ • (x - frameProjection v x)

/-- The actual displacement caused by normalizing a nonzero vector
is exactly the error in its norm. Source: arXiv:2410.06833v1, §4,
the normalization step in the original orthogonal approximation. -/
theorem norm_sub_normalized_coordinate {d : ℕ} (y : EucSpace d) (hy : y ≠ 0) :
    ‖y - ‖y‖⁻¹ • y‖ = |‖y‖ - 1| := by
  have hrepr : y - ‖y‖⁻¹ • y = (‖y‖ - 1) • (‖y‖⁻¹ • y) := by
    rw [sub_smul, norm_smul_normalized_coordinate y hy, one_smul]
  rw [hrepr, norm_smul, norm_normalized_coordinate y hy, mul_one, Real.norm_eq_abs]

/-- The normalized original residual is at most twice as far from the
unit point as the residual itself. Source: arXiv:2410.06833v1, §4,
the distance radius in `eq: upto-t`. This proves the normalization
factor missing in arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem norm_sub_normalized_le_two {d : ℕ} (x y : EucSpace d)
    (hx : ‖x‖ = 1) (hy : y ≠ 0) : ‖x - ‖y‖⁻¹ • y‖ ≤ 2 * ‖x - y‖ := by
  have hnorm := abs_norm_sub_norm_le y x
  rw [hx, norm_sub_rev y x] at hnorm
  calc
    _ ≤ ‖x - y‖ + ‖y - ‖y‖⁻¹ • y‖ := norm_sub_le_norm_sub_add_norm_sub _ _ _
    _ = ‖x - y‖ + |‖y‖ - 1| := by rw [norm_sub_normalized_coordinate y hy]
    _ ≤ 2 * ‖x - y‖ := by linarith

/-- A successful original projection has a genuine nonzero residual.
Source: arXiv:2410.06833v1, §4, the original inductive approximation.
This fact needs only the actual unit point and a projection below one. -/
theorem frameResidual_ne_zero {d m : ℕ} (v : Fin m → EucSpace d) (x : EucSpace d)
    (hx : ‖x‖ = 1) (hp : ‖frameProjection v x‖ < 1) : x - frameProjection v x ≠ 0 := by
  intro hzero
  have heq := sub_eq_zero.mp hzero
  have hnorm := congrArg norm heq
  rw [hx] at hnorm
  linarith

/-- The actual successful normalized residual has unit norm.
Source: arXiv:2410.06833v1, §4, the genuine sphere point constructed
in `prop: concentration unif`. No unit hypothesis is imposed on failure. -/
theorem norm_normalizedFrameResidual {d m : ℕ} (v : Fin m → EucSpace d) (x : EucSpace d)
    (hx : ‖x‖ = 1) (hp : ‖frameProjection v x‖ < 1) : ‖normalizedFrameResidual v x‖ = 1 :=
  norm_normalized_coordinate _ (frameResidual_ne_zero v x hx hp)

/-- The actual normalized residual is orthogonal to every previous
orthonormal vector. Source: arXiv:2410.06833v1, §4, the original
inductive orthogonal approximation, including its actual normalization. -/
theorem inner_normalizedFrameResidual {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (x : EucSpace d) (i : Fin m) :
    inner (𝕜 := ℝ) (v i) (normalizedFrameResidual v x) = 0 := by
  rw [normalizedFrameResidual, inner_smul_right, inner_frameResidual v hv x i, mul_zero]

/-- The actual next sphere point is within twice the actual projection
norm of the original unit point. Source: arXiv:2410.06833v1, §4,
`eq: upto-t`, with the source's original factor four under its square root. -/
theorem norm_sub_normalizedFrameResidual_le {d m : ℕ}
    (v : Fin m → EucSpace d) (x : EucSpace d)
    (hx : ‖x‖ = 1) (hp : ‖frameProjection v x‖ < 1) :
    ‖x - normalizedFrameResidual v x‖ ≤ 2 * ‖frameProjection v x‖ := by
  have h := norm_sub_normalized_le_two x (x - frameProjection v x) hx
    (frameResidual_ne_zero v x hx hp)
  simpa only [normalizedFrameResidual, sub_sub_cancel] using h

/-- Appending an actual orthogonal unit vector preserves actual
orthonormality. Source: arXiv:2410.06833v1, §4, the original inductive
frame construction; arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem orthonormal_frame_snoc {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (w : EucSpace d) (hw : ‖w‖ = 1)
    (horth : ∀ i : Fin m, inner (𝕜 := ℝ) (v i) w = 0) : Orthonormal ℝ (Fin.snoc v w) := by
  refine ⟨?_, ?_⟩
  · intro i
    cases i using Fin.lastCases with
    | last => simpa only [Fin.snoc_last] using hw
    | cast i => simpa only [Fin.snoc_castSucc] using hv.norm_eq_one i
  · intro i j hij
    cases i using Fin.lastCases with
    | last =>
      cases j using Fin.lastCases with
      | last => exact False.elim (hij rfl)
      | cast j => simpa only [Fin.snoc_last, Fin.snoc_castSucc, real_inner_comm w (v j)] using horth j
    | cast i =>
      cases j using Fin.lastCases with
      | last => simpa only [Fin.snoc_castSucc, Fin.snoc_last] using horth i
      | cast j =>
        simpa only [Fin.snoc_castSucc] using hv.inner_eq_zero
          (fun h => hij (congrArg Fin.castSucc h))

/-- Appending the actual successful residual gives the actual next
orthonormal frame. Source: arXiv:2410.06833v1, §4, the original
inductive approximation and its genuine unit-vector normalization. -/
theorem orthonormal_snoc_normalizedFrameResidual {d m : ℕ}
    (v : Fin m → EucSpace d) (hv : Orthonormal ℝ v) (x : EucSpace d)
    (hx : ‖x‖ = 1) (hp : ‖frameProjection v x‖ < 1) :
    Orthonormal ℝ (Fin.snoc v (normalizedFrameResidual v x)) :=
  orthonormal_frame_snoc v hv _ (norm_normalizedFrameResidual v x hx hp)
    (inner_normalizedFrameResidual v hv x)

/-- The total actual normalized-residual construction is measurable
in the previous frame and original point, including zero residuals.
Source: arXiv:2410.06833v1, §4, using only the original sample prefix. -/
theorem measurable_normalizedFrameResidual {d m : ℕ} :
    Measurable (fun p : (Fin m → EucSpace d) × EucSpace d => normalizedFrameResidual p.1 p.2) := by
  have hc : Continuous (fun p : (Fin m → EucSpace d) × EucSpace d =>
      p.2 - frameProjection p.1 p.2) := continuous_snd.sub continuous_frameProjection
  exact hc.measurable.norm.inv.smul hc.measurable

/-- Two actual coordinate axes jointly satisfy the unit, orthogonality
and successful-projection hypotheses of the original normalization. -/
example : ∃ v : Fin 1 → EucSpace 2, ∃ x : EucSpace 2,
    Orthonormal ℝ v ∧ ‖x‖ = 1 ∧ ‖frameProjection v x‖ < 1 ∧
    ∀ i : Fin 1, inner (𝕜 := ℝ) (v i) x = 0 := by
  let v : Fin 1 → EucSpace 2 := fun _ => EuclideanSpace.single 0 1
  let x : EucSpace 2 := EuclideanSpace.single 1 1
  refine ⟨v, x, ?_, by simp [x], ?_, ?_⟩
  · rw [orthonormal_iff_ite]
    intro i j
    simp [v, Subsingleton.elim i j]
  · simp [frameProjection, frameSynthesis, frameCoordinates, v, x, EuclideanSpace.inner_single_left]
  · intro i
    simp [v, x, EuclideanSpace.inner_single_left]

end Transformer.Metastability
