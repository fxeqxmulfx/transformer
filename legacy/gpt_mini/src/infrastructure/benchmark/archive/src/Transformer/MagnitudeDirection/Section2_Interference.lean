/-
# Magnitude--direction interference

arXiv:2606.25971v2, §2. Matrices carry the Frobenius inner product, so
the geometric statements apply to every real inner product space.
-/

import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic

noncomputable section

namespace Transformer.MagnitudeDirection

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The relative size of an additive update, arXiv:2606.25971v2, §2.
It is an approximation to an angular change only for small tangential steps. -/
def relativeUpdate (w delta : E) : ℝ := ‖delta‖ / ‖w‖

/-- Exact radial/tangential accounting, arXiv:2606.25971v2, §2,
“Magnitude grows despite no radial gradient”. -/
theorem update_norm_sq (w delta : E) :
    ‖w + delta‖ ^ 2 = ‖w‖ ^ 2 + 2 * inner ℝ w delta + ‖delta‖ ^ 2 :=
  norm_add_sq_real w delta

/-- A perpendicular update satisfies Pythagoras, arXiv:2606.25971v2, §2. -/
theorem perpendicular_update_norm_sq (w delta : E) (h : inner ℝ w delta = 0) :
    ‖w + delta‖ ^ 2 = ‖w‖ ^ 2 + ‖delta‖ ^ 2 := by
  rw [update_norm_sq, h]
  ring

/-- The perpendicularity hypothesis is satisfiable, arXiv:2606.25971v2, §2. -/
example : inner ℝ (1 : ℝ) (0 : ℝ) = 0 := by simp

/-- A nonzero perpendicular update strictly increases magnitude.
The source's “always increases” needs the nonzero-update condition.
Source: arXiv:2606.25971v2, §2, “Magnitude grows despite no radial gradient”. -/
theorem perpendicular_update_grows (w delta : E)
    (h : inner ℝ w delta = 0) (hd : delta ≠ 0) : ‖w‖ < ‖w + delta‖ := by
  have hs := perpendicular_update_norm_sq w delta h
  have hp := norm_pos_iff.mpr hd
  have hn := norm_nonneg w
  have hn' := norm_nonneg (w + delta)
  nlinarith

/-- Nonzero perpendicular updates exist, arXiv:2606.25971v2, §2. -/
example : inner ℝ (0 : ℝ) (1 : ℝ) = 0 ∧ (1 : ℝ) ≠ 0 := by norm_num

/-- Exact size of the negative radial signal needed to avoid norm growth,
arXiv:2606.25971v2, §2. -/
theorem norm_not_growing_iff (w delta : E) :
    ‖w + delta‖ ≤ ‖w‖ ↔ 2 * inner ℝ w delta + ‖delta‖ ^ 2 ≤ 0 := by
  have hs := update_norm_sq w delta
  have hn := norm_nonneg w
  have hn' := norm_nonneg (w + delta)
  constructor <;> intro h <;> nlinarith

/-- For a fixed update, rescaling the weight inversely rescales the relative
update. Source: arXiv:2606.25971v2, §2, “Direction change depends on magnitude”. -/
theorem relativeUpdate_rescale (w delta : E) (a : ℝ) (ha : 0 < a) :
    relativeUpdate (a • w) delta = relativeUpdate w delta / a := by
  unfold relativeUpdate
  rw [norm_smul_of_nonneg ha.le]
  ring

/-- Positive rescalings exist, arXiv:2606.25971v2, §2. -/
example : (0 : ℝ) < 2 := by norm_num

/-- Differentiable scale-invariant losses have zero radial derivative.
Scale invariance is required only along the positive ray through `w`.
Source: arXiv:2606.25971v2, §2, “a scale-invariant function has no radial gradient”. -/
theorem scaleInvariant_radial_derivative (L : E → ℝ) (w : E) (D : E →L[ℝ] ℝ)
    (hD : HasFDerivAt L D w) (hscale : ∀ a : ℝ, 0 < a → L (a • w) = L w) :
    D w = 0 := by
  have hr : HasDerivAt (fun a : ℝ => a • w) w 1 := by
    simpa using (hasDerivAt_id (1 : ℝ)).smul_const w
  have hc : HasDerivAt (fun a : ℝ => L (a • w)) (D w) 1 := by
    simpa [Function.comp_def] using
      hD.comp_hasDerivAt_of_eq 1 hr (by simp : w = (1 : ℝ) • w)
  have he : (fun a : ℝ => L (a • w)) =ᶠ[nhds (1 : ℝ)] (fun _ => L w) := by
    filter_upwards [eventually_gt_nhds (show (0 : ℝ) < 1 by norm_num)] with a ha
    exact hscale a ha
  exact (hc.congr_of_eventuallyEq he.symm).unique (hasDerivAt_const 1 (L w))

/-- A differentiable scale-invariant loss satisfies the hypotheses,
arXiv:2606.25971v2, §2. -/
example : HasFDerivAt (fun _ : ℝ => (3 : ℝ)) (0 : ℝ →L[ℝ] ℝ) 1 ∧
    (∀ a : ℝ, 0 < a → (fun _ : ℝ => (3 : ℝ)) (a • (1 : ℝ)) = 3) := by
  exact ⟨hasFDerivAt_const 3 1, fun _ _ => rfl⟩

end Transformer.MagnitudeDirection
