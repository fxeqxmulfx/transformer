/-
# DASH — solver convergence from the printed operator-norm conditions

arXiv:2602.02016v2, §3.2–3.4. The Euclidean operator-norm conditions
are converted to the verified spectral domains, so the CN and NDB
convergence theorems apply directly to the source's mathematical criteria.
-/

import Transformer.DASH.Section3_OperatorNorm
import Transformer.DASH.Section3_MatrixConvergence

open scoped Matrix Matrix.Norms.L2Operator

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The exact NDB condition `‖I-A‖₂<1` is equivalent to every eigenvalue
lying strictly between zero and two in an orthogonal eigenframe.
Source: arXiv:2602.02016v2, §3.3, the condition before `equation:NDB-init`. -/
theorem ndb_operator_domain_iff (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) :
    ‖1 - spectralMatrix Q s‖ < 1 ↔ ∀ i, 0 < s i ∧ s i < 2 := by
  rw [← spectralMatrix_one Q hQ, spectralMatrix_sub,
    spectralMatrix_opNorm_lt_iff Q _ 1 hQ (by norm_num)]
  apply forall_congr'
  intro i
  rw [abs_lt]
  constructor <;> rintro ⟨h, h'⟩ <;> constructor <;> linarith

/-- NDB norm-domain equivalence has an orthogonal example,
arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- The actual NDB matrix iteration converges under the source's
Euclidean operator-norm hypothesis; positive eigenvalues follow from that
hypothesis and are not assumed separately.
Source: arXiv:2602.02016v2, §3.3, `‖I-A‖₂<1` and `equation:NDB-Y-Z`. -/
theorem ndb_matrix_convergence_of_opNorm (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (hNorm : ‖1 - spectralMatrix Q s‖ < 1) :
    Filter.Tendsto (ndbIterate (spectralMatrix Q s)) Filter.atTop
      (nhds (spectralPower Q s (1 / 2), spectralPower Q s (-(1 / 2)))) := by
  have hdomain := (ndb_operator_domain_iff Q s hQ).1 hNorm
  exact ndb_matrix_convergence Q s hQ (fun i => (hdomain i).1) (fun i => (hdomain i).2)

/-- The exact NDB operator-domain assumptions are satisfiable,
arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ‖1 - spectralMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1)‖ < 1 := by
  have hQ : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
    simp [Orthogonal, Muon.OrthonormalColumns]
  refine ⟨hQ, ?_⟩
  rw [spectralMatrix_one _ hQ, sub_self, norm_zero]
  norm_num

/-- Positive spectra under the source's strict operator-norm bound satisfy
the corrected open CN domain, and its actual matrix iterates converge.
Unlike NDB's residual norm, an input operator-norm upper bound does not
itself imply positivity, so that required hypothesis is explicit.
Source: arXiv:2602.02016v2, §3.2 and §3.4, `‖A‖₂<(p+1)c^p`. -/
theorem cn_matrix_convergence_of_opNorm (p : ℕ) (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (c : ℝ) (hQ : Orthogonal Q) (hp : 0 < p) (hc : 0 < c)
    (hs : ∀ i, 0 < s i) (hNorm : ‖spectralMatrix Q s‖ < ((p : ℝ) + 1) * c ^ p) :
    Filter.Tendsto (cnIterate p (spectralMatrix Q s) c) Filter.atTop
      (nhds (spectralPower Q s (-(1 / (p : ℝ))), 1)) := by
  have hbound : (0 : ℝ) < ((p : ℝ) + 1) * c ^ p := by positivity
  have hdomain := (spectralMatrix_opNorm_lt_iff Q s _ hQ hbound).1 hNorm
  apply cn_matrix_convergence p Q s c hQ hp hc hs
  intro i
  exact (le_abs_self (s i)).trans_lt (hdomain i)

/-- CN's positivity and strict operator-norm bound coexist,
arXiv:2602.02016v2, §3.2–3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧
    ‖spectralMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1)‖ <
      ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) := by
  have hQ : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
    simp [Orthogonal, Muon.OrthonormalColumns]
  refine ⟨hQ, by norm_num, by norm_num, by norm_num, ?_⟩
  rw [spectralMatrix_one _ hQ]
  norm_num

end Transformer.DASH
