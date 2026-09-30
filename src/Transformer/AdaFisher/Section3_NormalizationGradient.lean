/-
# AdaFisher: the actual scale and shift derivatives

arXiv:2405.16397v3, Proposition 3.1, Appendix A.2.
The loss may depend jointly on all normalization sites. Its given
preactivation derivative is composed with the affine normalization map.
-/

import Transformer.AdaFisher.Section3_Normalization
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {c t : ℕ}

/-- Preactivation loss derivative represented by the sensitivities s,
Appendix A.2, Proposition 3.1. -/
def normalizationSensitivity (s : Fin t → Fin c → ℝ) :
    (Fin t → Fin c → ℝ) →L[ℝ] ℝ :=
  ∑ x, ∑ i, s x i •
    ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin c => ℝ) i).comp
      (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin t => Fin c → ℝ) x))

/-- The sensitivity functional is the finite inner product with s,
Appendix A.2, the definition of preactivation gradients. -/
theorem normalizationSensitivity_apply (s z : Fin t → Fin c → ℝ) :
    normalizationSensitivity s z = ∑ x, ∑ i, s x i * z x i := by
  simp [normalizationSensitivity]

/-- Pairing an arbitrary scale/shift perturbation with the preactivation
derivative yields exactly the source's scale and shift gradients,
Appendix A.2. No separability of the loss across sites is assumed. -/
theorem normalization_gradient_pairing (h s : Fin t → Fin c → ℝ)
    (dν dβ : Fin c → ℝ) :
    normalizationSensitivity s (fun x i => dν i * h x i + dβ i) =
      ∑ i, (scaleGradient h s i * dν i + shiftGradient s i * dβ i) := by
  rw [normalizationSensitivity_apply]
  simp only [mul_add, Finset.sum_add_distrib, scaleGradient, shiftGradient,
    Finset.sum_mul]
  congr 1
  · rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i hi
    apply Finset.sum_congr rfl
    intro x hx
    ring
  · exact Finset.sum_comm

/-- Actual derivative of the loss through normalization scale and shift,
Appendix A.2, Proposition 3.1. The Frechet derivative hypothesis concerns
the full loss at all sites, and the conclusion proves the summed gradient
formulas along every parameter direction. -/
theorem normalization_loss_directional_derivative
    (loss : (Fin t → Fin c → ℝ) → ℝ) (ν β dν dβ : Fin c → ℝ)
    (h s : Fin t → Fin c → ℝ)
    (hloss : HasFDerivAt loss (normalizationSensitivity s) (normalizationAffine ν β h)) :
    HasDerivAt (fun u : ℝ => loss (normalizationAffine (ν + u • dν) (β + u • dβ) h))
      (∑ i, (scaleGradient h s i * dν i + shiftGradient s i * dβ i)) 0 := by
  have ha : HasDerivAt
      (fun u : ℝ => normalizationAffine (ν + u • dν) (β + u • dβ) h)
      (fun x i => dν i * h x i + dβ i) 0 := by
    apply hasDerivAt_pi.mpr
    intro x
    apply hasDerivAt_pi.mpr
    intro i
    simpa [normalizationAffine, Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.add_def] using
      ((((hasDerivAt_id (0 : ℝ)).mul_const (dν i)).const_add (ν i)).mul_const (h x i)).add
        (((hasDerivAt_id (0 : ℝ)).mul_const (dβ i)).const_add (β i))
  have hloss' : HasFDerivAt loss (normalizationSensitivity s)
      (normalizationAffine (ν + (0 : ℝ) • dν) (β + (0 : ℝ) • dβ) h) := by
    simpa only [zero_smul, add_zero] using hloss
  have hc := hloss'.comp_hasDerivAt 0 ha
  simpa only [normalization_gradient_pairing, Function.comp_def] using hc

example :
    HasFDerivAt (normalizationSensitivity (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)))
      (normalizationSensitivity (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)))
      (normalizationAffine (fun _ : Fin 1 => 1) (fun _ => 0) (fun _ _ => 1)) :=
  (normalizationSensitivity _).hasFDerivAt

end Transformer.AdaFisher
