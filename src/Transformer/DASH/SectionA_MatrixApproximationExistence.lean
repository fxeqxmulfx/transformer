/-
# DASH — uniform accuracy of an actual optimized matrix evaluator

arXiv:2602.02016v2, Appendix A. Finite coefficient arrays exist for any
positive accuracy target. The corrected original-scale algorithm fits at
`ε/c` and multiplies its output by `c^exponent`.
-/

import Transformer.DASH.SectionA_DenseCoefficients
import Transformer.DASH.SectionA_Approximation

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The actual optimized matrix Clenshaw output can approximate its normalized
regularized power to any positive Frobenius tolerance. The finite array is
supplied by the proved approximation theorem, rather than by an unproved
accuracy assumption. Source: arXiv:2602.02016v2, Appendix A, matrix approximation. -/
theorem exists_optimizedClenshaw_matrix_approximation
    (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (c ε exponent δ : ℝ)
    (hQ : Orthogonal Q) (hc : 0 < c) (hε : 0 < ε)
    (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ c) (hδ : 0 < δ) :
    ∃ cs : List ℝ, frobeniusNorm
      ((optimizedClenshaw (chebMatrixArgument (spectralMatrix Q s) c) cs).1 -
        spectralPower Q (fun i => s i / c + ε) exponent) < δ := by
  let η : ℝ := δ / (Real.sqrt (n : ℝ) + 1)
  have hden : 0 < Real.sqrt (n : ℝ) + 1 := by positivity
  have hη : 0 < η := div_pos hδ hden
  obtain ⟨cs, hcs⟩ := exists_clenshaw_inverse_power_approximation ε exponent η hε hη
  refine ⟨cs, ?_⟩
  have happrox : ∀ x : ℝ, ε ≤ x → x ≤ 1 + ε →
      |clenshaw (2 * (x - ε) - 1) cs - x ^ exponent| ≤ η := by
    intro x hx hx'
    have ht : 2 * (x - ε) - 1 ∈ Set.Icc (-1 : ℝ) 1 := ⟨by linarith, by linarith⟩
    have hcoord : chebFromCoordinate ε (1 + ε) (2 * (x - ε) - 1) = x := by
      unfold chebFromCoordinate
      ring
    simpa only [hcoord] using (hcs _ ht).le
  have hmatrix := optimizedClenshaw_approximation Q s cs c ε exponent η hQ hc hs hupper hη.le happrox
  have hstrict : Real.sqrt (n : ℝ) * η < δ := by
    unfold η
    rw [← mul_div_assoc]
    apply (div_lt_iff₀ hden).2
    nlinarith
  exact hmatrix.trans_lt hstrict

/-- Normalized matrix-approximation assumptions are satisfiable,
arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) ∧ (0 : ℝ) < 1 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- The corrected original-scale algorithm fits with regularization `ε/c`
and rescales the output. Its optimized Clenshaw evaluation approximates
`(A+εI)^exponent` to any requested Frobenius tolerance. The source instead
fits at `ε` and omits output scaling; that printed target is refuted by the
earlier counterexample. Source: arXiv:2602.02016v2, Appendix A, matrix output claim. -/
theorem exists_rescaledClenshaw_regularized_approximation
    (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (c ε exponent δ : ℝ)
    (hQ : Orthogonal Q) (hc : 0 < c) (hε : 0 < ε)
    (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ c) (hδ : 0 < δ) :
    ∃ cs : List ℝ, frobeniusNorm
      (c ^ exponent • (optimizedClenshaw (chebMatrixArgument (spectralMatrix Q s) c) cs).1 -
        spectralPower Q (fun i => s i + ε) exponent) < δ := by
  have hpower : 0 < c ^ exponent := Real.rpow_pos_of_pos hc exponent
  obtain ⟨cs, hcs⟩ := exists_optimizedClenshaw_matrix_approximation Q s c (ε / c) exponent
    (δ / c ^ exponent) hQ hc (div_pos hε hc) hs hupper (div_pos hδ hpower)
  refine ⟨cs, ?_⟩
  have htarget : c ^ exponent • spectralPower Q (fun i => s i / c + ε / c) exponent =
      spectralPower Q (fun i => s i + ε) exponent := by
    rw [regularized_spectralPower_rescale Q s c (ε / c) exponent hc hs (div_pos hε hc).le]
    congr 1
    funext i
    field_simp
  rw [← htarget, ← smul_sub, frobeniusNorm_smul, abs_of_pos hpower]
  have hbound := mul_lt_mul_of_pos_left hcs hpower
  have heq : c ^ exponent * (δ / c ^ exponent) = δ := by field_simp
  exact hbound.trans_eq heq

/-- Original-scale correction assumptions are satisfiable,
arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) ∧ (0 : ℝ) < 1 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
