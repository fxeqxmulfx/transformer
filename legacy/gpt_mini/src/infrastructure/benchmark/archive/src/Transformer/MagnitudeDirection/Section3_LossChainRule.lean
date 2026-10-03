/-
# Loss chain rule for magnitude--direction decoupling

arXiv:2606.25971v2, §3.1 and Appendix A, Algorithms 1–2.
The input is the actual Fréchet derivative of the loss at the fused
weight, not an assumed gradient-splitting identity.
-/

import Transformer.MagnitudeDirection.Section3_Gradients
import Transformer.MagnitudeDirection.Section4_PositiveGains
import Mathlib.Analysis.Calculus.Deriv.Comp

open scoped BigOperators Matrix.Norms.Elementwise

noncomputable section

namespace Transformer.MagnitudeDirection

variable {m n : ℕ}

/-- The direction/row/column formulas give the derivative of a differentiable
loss along every simultaneous perturbation of the factors.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, lines 3–6. -/
theorem decoupled_loss_hasDerivAt (L : Matrix (Fin m) (Fin n) ℝ → ℝ)
    (row dr : Fin m → ℝ) (col dc : Fin n → ℝ)
    (D H G : Matrix (Fin m) (Fin n) ℝ)
    (hL : HasFDerivAt L (gradientFunctional G) (fuse row col D)) :
    HasDerivAt (fun t : ℝ => L (fuse (row + t • dr) (col + t • dc) (D + t • H)))
      (frobeniusPairing (directionGradient row col G) H +
        (∑ i, rowGradient col D G i * dr i) + (∑ j, colGradient row D G j * dc j)) 0 := by
  have h := hL.comp_hasDerivAt_of_eq (0 : ℝ) (fuse_curve_hasDerivAt row dr col dc D H)
    (by simp : fuse row col D = fuse (row + (0 : ℝ) • dr) (col + (0 : ℝ) • dc)
      (D + (0 : ℝ) • H))
  change HasDerivAt _ (frobeniusPairing G (fusionVariation row dr col dc D H)) (0 : ℝ) at h
  simpa only [Function.comp_def, fusionVariation_pairing] using h

/-- A nonconstant linear matrix loss has the required differential,
arXiv:2606.25971v2, Appendix A. -/
example : HasFDerivAt (gradientFunctional (1 : Matrix (Fin 1) (Fin 1) ℝ))
    (gradientFunctional (1 : Matrix (Fin 1) (Fin 1) ℝ))
    (fuse (fun _ => 1) (fun _ => 1) (1 : Matrix (Fin 1) (Fin 1) ℝ)) :=
  ContinuousLinearMap.hasFDerivAt _

/-- Backpropagation through a smooth raw gain is the ordinary chain rule,
arXiv:2606.25971v2, Appendix A, Algorithm 2, line 5. -/
theorem raw_gain_hasDerivAt (phi L : ℝ → ℝ) (raw dphi g : ℝ)
    (hp : HasDerivAt phi dphi raw) (hL : HasDerivAt L g (phi raw)) :
    HasDerivAt (fun x => L (phi x)) (g * dphi) raw := by
  simpa [Function.comp_def] using hL.comp raw hp

/-- Smooth raw-gain chain-rule hypotheses hold, arXiv:2606.25971v2, Appendix A. -/
example : HasDerivAt softplus (softplusDerivative 0) 0 ∧
    HasDerivAt (fun x : ℝ => 2 * x) 2 (softplus 0) := by
  exact ⟨softplus_hasDerivAt 0, by simpa using (hasDerivAt_id (softplus 0)).const_mul 2⟩

/-- Actual weight derivative with separately perturbed raw softplus gains,
arXiv:2606.25971v2, Appendix A, Algorithm 2, line 5. -/
theorem softplus_fuse_curve_hasDerivAt (row dr : Fin m → ℝ) (col dc : Fin n → ℝ)
    (D H : Matrix (Fin m) (Fin n) ℝ) :
    HasDerivAt (fun t : ℝ => fuse (fun i => softplus (row i + t * dr i))
      (fun j => softplus (col j + t * dc j)) (D + t • H))
      (fusionVariation (fun i => softplus (row i))
        (fun i => softplusDerivative (row i) * dr i) (fun j => softplus (col j))
        (fun j => softplusDerivative (col j) * dc j) D H) 0 := by
  have hd (a b : ℝ) : HasDerivAt (fun t : ℝ => a + t * b) b 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const b).const_add a
  have hp (a b : ℝ) : HasDerivAt (fun t : ℝ => softplus (a + t * b))
      (softplusDerivative a * b) 0 := by
    simpa [Function.comp_def] using
      (softplus_hasDerivAt a).comp_of_eq 0 (hd a b) (by simp)
  apply hasDerivAt_pi.mpr
  intro i
  apply hasDerivAt_pi.mpr
  intro j
  convert ((hp (row i) (dr i)).mul (hd (D i j) (H i j))).mul (hp (col j) (dc j)) using 1
  · rfl
  · dsimp [fusionVariation, fuse]
    simp only [zero_mul, add_zero]
    ring

/-- Raw-gain gradients in Algorithm 2 preserve the complete loss derivative,
including the separate direction perturbation.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, lines 3–6. -/
theorem softplus_decoupled_loss_hasDerivAt (L : Matrix (Fin m) (Fin n) ℝ → ℝ)
    (row dr : Fin m → ℝ) (col dc : Fin n → ℝ) (D H G : Matrix (Fin m) (Fin n) ℝ)
    (hL : HasFDerivAt L (gradientFunctional G)
      (fuse (fun i => softplus (row i)) (fun j => softplus (col j)) D)) :
    HasDerivAt (fun t : ℝ => L (fuse (fun i => softplus (row i + t * dr i))
      (fun j => softplus (col j + t * dc j)) (D + t • H)))
      (frobeniusPairing (directionGradient (fun i => softplus (row i))
        (fun j => softplus (col j)) G) H +
        (∑ i, (rowGradient (fun j => softplus (col j)) D G i *
          softplusDerivative (row i)) * dr i) +
        (∑ j, (colGradient (fun i => softplus (row i)) D G j *
          softplusDerivative (col j)) * dc j)) 0 := by
  have h := hL.comp_hasDerivAt_of_eq (0 : ℝ) (softplus_fuse_curve_hasDerivAt row dr col dc D H)
    (by simp)
  change HasDerivAt _ (frobeniusPairing G (fusionVariation (fun i => softplus (row i))
    (fun i => softplusDerivative (row i) * dr i) (fun j => softplus (col j))
    (fun j => softplusDerivative (col j) * dc j) D H)) (0 : ℝ) at h
  simpa only [Function.comp_def, fusionVariation_pairing,
    mul_assoc] using h

/-- The raw-gain loss assumptions hold for a nonconstant linear loss,
arXiv:2606.25971v2, Appendix A. -/
example : HasFDerivAt (gradientFunctional (1 : Matrix (Fin 1) (Fin 1) ℝ))
    (gradientFunctional (1 : Matrix (Fin 1) (Fin 1) ℝ))
    (fuse (fun _ => softplus 0) (fun _ => softplus 0)
      (1 : Matrix (Fin 1) (Fin 1) ℝ)) := ContinuousLinearMap.hasFDerivAt _

end Transformer.MagnitudeDirection
