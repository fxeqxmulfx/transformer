/-
# DASH — matrix accuracy of the CN stopping criterion

arXiv:2602.02016v2, §3.2. The Euclidean operator-norm residual
of `M_k` bounds the relative inverse-root error in both operator
and Frobenius norms. All bounds concern the actual finite iterate.
-/

import Transformer.DASH.Section3_StoppingAccuracy
import Transformer.DASH.Section3_OperatorNorm
import Transformer.DASH.Section3_NewtonSpectra

open scoped BigOperators Matrix Matrix.Norms.L2Operator

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The matrix stopping criterion bounds every eigenvalue's actual
inverse-root error relative to its exact inverse root.
Source: arXiv:2602.02016v2, §3.2, stopping when `M_k` is close to `I`. -/
theorem cnIterate_residual_coordinate_accuracy (p : ℕ)
    (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (c δ : ℝ)
    (hp : 0 < p) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) (hc : 0 < c)
    (hupper : ∀ i, s i < ((p : ℝ) + 1) * c ^ p) (k : ℕ)
    (hstop : ‖1 - (cnIterate p (spectralMatrix Q s) c (k + 1)).2‖ ≤ δ) :
    ∀ i, |cnScalarX p (s i) c (k + 1) - s i ^ (-(1 / (p : ℝ)))| ≤
      s i ^ (-(1 / (p : ℝ))) * δ := by
  have hδ : 0 ≤ δ := (norm_nonneg _).trans hstop
  have hmatrix : (1 : Matrix (Fin n) (Fin n) ℝ) -
      spectralMatrix Q (fun i => cnScalarM p (s i) c (k + 1)) =
      spectralMatrix Q (fun i => 1 - cnScalarM p (s i) c (k + 1)) := by
    rw [← spectralMatrix_one Q hQ, spectralMatrix_sub]
  rw [cnIterate_spectrum p Q s c hQ (k + 1), hmatrix,
    spectralMatrix_opNorm_le_iff Q _ δ hQ hδ] at hstop
  intro i
  exact cnScalarX_residual_accuracy p (s i) c δ hp (hs i) hc (hupper i) k
    ((le_abs_self _).trans (hstop i))

/-- The finite matrix stopping assumptions have a positive-definite instance,
arXiv:2602.02016v2, §3.2. -/
example : 0 < (2 : ℕ) ∧ Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ)) ∧
    ‖1 - (cnIterate 2 (spectralMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ)
      (fun _ => 1)) 1 1).2‖ ≤ (1 / 10 : ℝ) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns, spectralMatrix, Muon.singularMatrix,
    cnIterate, cnStep, cnCorrection, ← sub_smul]

/-- An operator-norm identity residual at most `δ` guarantees relative
inverse-root accuracy at most `δ` in both Euclidean operator norm and
Frobenius norm. Absolute tolerance must be multiplied by the norm of the
true inverse root, so it is sensitive to small input eigenvalues.
Source: arXiv:2602.02016v2, §3.2, the early-stopping rule, and §3.4, scaling. -/
theorem cnIterate_residual_norm_accuracy (p : ℕ)
    (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (c δ : ℝ)
    (hp : 0 < p) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) (hc : 0 < c)
    (hupper : ∀ i, s i < ((p : ℝ) + 1) * c ^ p) (k : ℕ)
    (hstop : ‖1 - (cnIterate p (spectralMatrix Q s) c (k + 1)).2‖ ≤ δ) :
    ‖(cnIterate p (spectralMatrix Q s) c (k + 1)).1 -
        spectralPower Q s (-(1 / (p : ℝ)))‖ ≤
      δ * ‖spectralPower Q s (-(1 / (p : ℝ)))‖ ∧
    frobeniusNorm ((cnIterate p (spectralMatrix Q s) c (k + 1)).1 -
        spectralPower Q s (-(1 / (p : ℝ)))) ≤
      δ * frobeniusNorm (spectralPower Q s (-(1 / (p : ℝ)))) := by
  have hδ : 0 ≤ δ := (norm_nonneg _).trans hstop
  have he := cnIterate_residual_coordinate_accuracy p Q s c δ hp hQ hs hc hupper k hstop
  have htarget (i : Fin n) : 0 < s i ^ (-(1 / (p : ℝ))) :=
    Real.rpow_pos_of_pos (hs i) _
  rw [cnIterate_spectrum p Q s c hQ (k + 1)]
  change ‖spectralMatrix Q (fun i => cnScalarX p (s i) c (k + 1)) -
      spectralMatrix Q (fun i => s i ^ (-(1 / (p : ℝ))))‖ ≤ _ ∧
    frobeniusNorm (spectralMatrix Q (fun i => cnScalarX p (s i) c (k + 1)) -
      spectralMatrix Q (fun i => s i ^ (-(1 / (p : ℝ))))) ≤ _
  rw [spectralMatrix_sub]
  constructor
  · apply (spectralMatrix_opNorm_le_iff Q _ _ hQ
      (mul_nonneg hδ (norm_nonneg _))).mpr
    intro i
    have hb : s i ^ (-(1 / (p : ℝ))) ≤
        ‖spectralPower Q s (-(1 / (p : ℝ)))‖ := by
      rw [spectralPower, spectralMatrix_opNorm Q _ hQ]
      simpa only [Real.norm_eq_abs, abs_of_pos (htarget i)] using
        norm_le_pi_norm (fun j => s j ^ (-(1 / (p : ℝ)))) i
    exact (he i).trans ((mul_le_mul_of_nonneg_right hb hδ).trans_eq (mul_comm _ _))
  · have hsum : (∑ i, (cnScalarX p (s i) c (k + 1) -
        s i ^ (-(1 / (p : ℝ)))) ^ 2) ≤
        (∑ i, (s i ^ (-(1 / (p : ℝ)))) ^ 2) * δ ^ 2 := by
      rw [Finset.sum_mul]
      apply Finset.sum_le_sum
      intro i hi
      simpa only [sq_abs, mul_pow] using pow_le_pow_left₀ (abs_nonneg _) (he i) 2
    rw [frobeniusNorm, spectralMatrix_energy Q _ hQ]
    calc
      _ ≤ Real.sqrt ((∑ i, (s i ^ (-(1 / (p : ℝ)))) ^ 2) * δ ^ 2) :=
        Real.sqrt_le_sqrt hsum
      _ = δ * frobeniusNorm (spectralPower Q s (-(1 / (p : ℝ)))) := by
        rw [Real.sqrt_mul (Finset.sum_nonneg fun i _ => sq_nonneg _), Real.sqrt_sq hδ,
          frobeniusNorm, spectralPower, spectralMatrix_energy Q _ hQ, mul_comm]

/-- The norm certificate is attainable by a finite iterate,
arXiv:2602.02016v2, §3.2–3.4. -/
example : 0 < (2 : ℕ) ∧ Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ)) ∧
    ‖1 - (cnIterate 2 (spectralMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ)
      (fun _ => 1)) 1 1).2‖ ≤ (1 / 10 : ℝ) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns, spectralMatrix, Muon.singularMatrix,
    cnIterate, cnStep, cnCorrection, ← sub_smul]

end Transformer.DASH
