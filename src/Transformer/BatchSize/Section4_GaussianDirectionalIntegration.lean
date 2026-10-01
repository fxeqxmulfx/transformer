/-
# Gaussian integration by parts for Euclidean directional derivatives

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The affine-state derivative and coordinate Fubini identity give the
actual diagonal generator contraction for bounded derivative fields.
-/

import Transformer.BatchSize.Section4_DiagonalGaussianState

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Gaussian integration by parts for a Euclidean derivative field,
Section 4.3 (2)--(3). G is the genuine Frechet derivative of g;
evaluating a coordinate innovation contracts G with the corresponding
diagonal noise amplitude and Euclidean basis vector. -/
theorem gaussianAffineState_directional_integration_by_parts {n : ℕ}
    (x : EucSpace (n + 1)) (A : Fin (n + 1) → ℝ) (k : Fin (n + 1)) (v : EucSpace (n + 1))
    (g : EucSpace (n + 1) → EucSpace (n + 1) →L[ℝ] ℝ)
    (G : EucSpace (n + 1) → EucSpace (n + 1) →L[ℝ] (EucSpace (n + 1) →L[ℝ] ℝ))
    (hg : ∀ y, HasFDerivAt g (G y) y) (hG : Continuous G)
    (C D : ℝ) (hC : ∀ y, ‖g y‖ ≤ C) (hD : ∀ y, ‖G y‖ ≤ D) :
    (∫ z, z k * g (gaussianAffineState x A z) v ∂standardGaussianVectorLaw (n + 1)) =
      A k * ∫ z, G (gaussianAffineState x A z) (EuclideanSpace.single k 1) v
        ∂standardGaussianVectorLaw (n + 1) := by
  let e := EuclideanSpace.single k (1 : ℝ)
  let ev := ContinuousLinearMap.apply ℝ ℝ v
  let L := ev.comp (ContinuousLinearMap.apply ℝ (EucSpace (n + 1) →L[ℝ] ℝ) e)
  have he : ‖e‖ = 1 := by simp [e, EuclideanSpace.single, PiLp.norm_single]
  have hgc : Continuous g := continuous_iff_continuousAt.mpr fun y => (hg y).continuousAt
  have hcont : Continuous (fun z => g (gaussianAffineState x A z) v) :=
    ev.continuous.comp (hgc.comp (gaussianAffineState_continuous x A))
  have hcont' : Continuous (fun z => A k * G (gaussianAffineState x A z) e v) :=
    continuous_const.mul (L.continuous.comp (hG.comp (gaussianAffineState_continuous x A)))
  have hslice (z : Fin n → ℝ) (u : ℝ) :
      HasDerivAt (fun w => g (gaussianAffineState x A (k.insertNth w z)) v)
        (A k * G (gaussianAffineState x A (k.insertNth u z)) e v) u := by
    have hs := gaussianAffineState_slice_hasDerivAt x A k z u
    have hd := (hg (gaussianAffineState x A (k.insertNth u z))).comp_hasDerivAt u hs
    have h := ev.hasFDerivAt.comp_hasDerivAt u hd
    change HasDerivAt (fun w => g (gaussianAffineState x A (k.insertNth w z)) v)
      (G (gaussianAffineState x A (k.insertNth u z)) (A k • e) v) u at h
    simpa only [map_smul, smul_apply, smul_eq_mul] using h
  have hbound (z : Fin (n + 1) → ℝ) : |g (gaussianAffineState x A z) v| ≤ C * ‖v‖ := by
    have hpoint : |g (gaussianAffineState x A z) v| ≤ ‖g (gaussianAffineState x A z)‖ * ‖v‖ := by
      simpa only [Real.norm_eq_abs] using (g (gaussianAffineState x A z)).le_opNorm v
    exact hpoint.trans (mul_le_mul_of_nonneg_right (hC _) (norm_nonneg v))
  have hGbound (z : Fin (n + 1) → ℝ) : |G (gaussianAffineState x A z) e v| ≤ D * ‖v‖ := by
    calc
      _ ≤ ‖G (gaussianAffineState x A z) e‖ * ‖v‖ := by
        simpa only [Real.norm_eq_abs] using (G (gaussianAffineState x A z) e).le_opNorm v
      _ ≤ (‖G (gaussianAffineState x A z)‖ * ‖e‖) * ‖v‖ :=
        mul_le_mul_of_nonneg_right ((G (gaussianAffineState x A z)).le_opNorm e) (norm_nonneg v)
      _ = ‖G (gaussianAffineState x A z)‖ * ‖v‖ := by rw [he, mul_one]
      _ ≤ _ := mul_le_mul_of_nonneg_right (hD _) (norm_nonneg v)
  have hbound' (z : Fin (n + 1) → ℝ) : |A k * G (gaussianAffineState x A z) e v| ≤ |A k| * D * ‖v‖ := by
    rw [abs_mul]
    exact (mul_le_mul_of_nonneg_left (hGbound z) (abs_nonneg _)).trans_eq (by ring)
  have h := standardGaussianVector_integration_by_parts k _ _ hcont hcont' hslice
    (C * ‖v‖) (|A k| * D * ‖v‖) hbound hbound'
  rw [integral_const_mul] at h
  exact h

/-- Joint nonvacuity of the directional derivative-field hypotheses,
Section 4.3: a nonzero constant coordinate functional and its zero derivative. -/
example : (∀ y : EucSpace 1, HasFDerivAt
    (fun _ : EucSpace 1 => PiLp.proj (𝕜 := ℝ) (p := 2) (β := fun _ : Fin 1 => ℝ) 0)
      (0 : EucSpace 1 →L[ℝ] (EucSpace 1 →L[ℝ] ℝ)) y) ∧
    Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1 →L[ℝ] (EucSpace 1 →L[ℝ] ℝ))) ∧
    (∀ y : EucSpace 1,
      ‖(fun _ : EucSpace 1 => PiLp.proj (𝕜 := ℝ) (p := 2) (β := fun _ : Fin 1 => ℝ) 0) y‖ ≤
        ‖PiLp.proj (𝕜 := ℝ) (p := 2) (β := fun _ : Fin 1 => ℝ) 0‖) ∧
    (∀ y : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1 →L[ℝ] (EucSpace 1 →L[ℝ] ℝ))) y‖ ≤ 0) := by
  exact ⟨fun y => hasFDerivAt_const _ y, continuous_const, fun _ => le_rfl,
    fun _ => le_of_eq (norm_zero (E := EucSpace 1 →L[ℝ] (EucSpace 1 →L[ℝ] ℝ)))⟩

end Transformer.BatchSize
