/-
# DASH — convergence from fitted samples to optimized matrix powers

arXiv:2602.02016v2, Appendix A. The actual cosine coefficient fit and
the optimized matrix Clenshaw recurrence converge in Frobenius norm.
The original-scale version includes the documented regularization
and output-scaling corrections.
-/

import Transformer.DASH.SectionA_FitConvergence
import Transformer.DASH.SectionA_Approximation

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- For every positive tolerance, all sufficiently large fitted degrees
and every larger sample count give an accurate optimized matrix result.
The coefficient array is computed by `chebFit` for the source's interval;
the actual normalized target is `(A/c+εI)^exponent`.
Source: arXiv:2602.02016v2, Appendix A, fitting and optimized matrix Clenshaw. -/
theorem fittedClenshaw_matrix_uniform_accuracy
    (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (c ε exponent δ : ℝ)
    (hQ : Orthogonal Q) (hc : 0 < c) (hε : 0 < ε)
    (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ c) (hδ : 0 < δ) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d → ∀ N : ℕ, d < N → frobeniusNorm
      ((optimizedClenshaw (chebMatrixArgument (spectralMatrix Q s) c)
        (chebFit (fun x : ℝ => x ^ exponent) ε (1 + ε) N d)).1 -
          spectralPower Q (fun i => s i / c + ε) exponent) < δ := by
  let η : ℝ := δ / (Real.sqrt (n : ℝ) + 1)
  have hden : 0 < Real.sqrt (n : ℝ) + 1 := by positivity
  have hη : 0 < η := div_pos hδ hden
  obtain ⟨D, hD⟩ := chebFit_inverse_power_uniform_accuracy ε exponent η hε hη
  refine ⟨D, ?_⟩
  intro d hd N hN
  have happrox : ∀ x : ℝ, ε ≤ x → x ≤ 1 + ε →
      |clenshaw (2 * (x - ε) - 1)
        (chebFit (fun x : ℝ => x ^ exponent) ε (1 + ε) N d) - x ^ exponent| ≤ η := by
    intro x hx hx'
    have hcoord : chebFromCoordinate ε (1 + ε) (2 * (x - ε) - 1) = x := by
      unfold chebFromCoordinate
      ring
    simpa only [hcoord] using
      (hD d hd N hN (2 * (x - ε) - 1) (by linarith) (by linarith)).le
  have hmatrix := optimizedClenshaw_approximation Q s _ c ε exponent η
    hQ hc hs hupper hη.le happrox
  have hstrict : Real.sqrt (n : ℝ) * η < δ := by
    unfold η
    rw [← mul_div_assoc]
    apply (div_lt_iff₀ hden).2
    nlinarith
  exact hmatrix.trans_lt hstrict

/-- Actual fitted matrix-accuracy assumptions are satisfiable,
arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) ∧ (0 : ℝ) < 1 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- Fitting at `ε/c` and multiplying the computed output by `c^exponent`
gives uniform-in-degree accuracy for the original-scale regularized power
`(A+εI)^exponent`. These corrections repair the target printed in the
source's matrix algorithms; the coefficient array remains the actual fit.
Source: arXiv:2602.02016v2, Appendix A, matrix output and normalization. -/
theorem rescaledFittedClenshaw_regularized_uniform_accuracy
    (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (c ε exponent δ : ℝ)
    (hQ : Orthogonal Q) (hc : 0 < c) (hε : 0 < ε)
    (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ c) (hδ : 0 < δ) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d → ∀ N : ℕ, d < N → frobeniusNorm
      (c ^ exponent • (optimizedClenshaw (chebMatrixArgument (spectralMatrix Q s) c)
        (chebFit (fun x : ℝ => x ^ exponent) (ε / c) (1 + ε / c) N d)).1 -
          spectralPower Q (fun i => s i + ε) exponent) < δ := by
  have hpower : 0 < c ^ exponent := Real.rpow_pos_of_pos hc exponent
  obtain ⟨D, hD⟩ := fittedClenshaw_matrix_uniform_accuracy Q s c (ε / c) exponent
    (δ / c ^ exponent) hQ hc (div_pos hε hc) hs hupper (div_pos hδ hpower)
  refine ⟨D, ?_⟩
  intro d hd N hN
  have htarget : c ^ exponent • spectralPower Q (fun i => s i / c + ε / c) exponent =
      spectralPower Q (fun i => s i + ε) exponent := by
    rw [regularized_spectralPower_rescale Q s c (ε / c) exponent hc hs (div_pos hε hc).le]
    congr 1
    funext i
    field_simp
  rw [← htarget, ← smul_sub, frobeniusNorm_smul, abs_of_pos hpower]
  have hbound := mul_lt_mul_of_pos_left (hD d hd N hN) hpower
  have heq : c ^ exponent * (δ / c ^ exponent) = δ := by field_simp
  exact hbound.trans_eq heq

/-- Original-scale fitted accuracy has a positive-definite instance,
arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 2) ∧ (0 : ℝ) < 1 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
