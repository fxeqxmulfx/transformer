/-
# DASH — matrix iterates in eigenvalue coordinates

arXiv:2602.02016v2, §3.2–3.4. Every iteration retains the input
eigenvectors, and each eigenvalue follows the scalar recurrence.
-/

import Transformer.DASH.Section3_NewtonModels
import Mathlib.Topology.Instances.Matrix

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- CN acts independently on every eigenvalue, including its initialization.
Source: arXiv:2602.02016v2, §3.2, `equation:CN-init`–`equation:CN-X-M`. -/
theorem cnIterate_spectrum (p : ℕ) (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (c : ℝ) (hQ : Orthogonal Q) (k : ℕ) :
    cnIterate p (spectralMatrix Q s) c k =
      (spectralMatrix Q (fun i => cnScalarX p (s i) c k),
        spectralMatrix Q (fun i => cnScalarM p (s i) c k)) := by
  induction k with
  | zero =>
    simp only [cnIterate, cnScalarX, cnScalarM]
    rw [← spectralMatrix_one Q hQ, spectralMatrix_smul, spectralMatrix_smul]
    simp only [mul_one, div_eq_mul_inv, mul_comm]
  | succ k ih =>
    simp only [cnIterate, ih, cnStep, cnCorrection_spectrum p Q _ hQ,
      spectralMatrix_pow Q _ hQ, spectralMatrix_mul Q _ _ hQ,
      cnScalarX, cnScalarM, cnMap]
    simp only [mul_comm]

/-- Spectral recurrence hypotheses are satisfiable,
arXiv:2602.02016v2, §3.2. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- NDB acts independently on every eigenvalue for both output sequences.
Source: arXiv:2602.02016v2, §3.3, `equation:NDB-init`–`equation:NDB-Y-Z`. -/
theorem ndbIterate_spectrum (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hQ : Orthogonal Q) (k : ℕ) :
    ndbIterate (spectralMatrix Q s) k =
      (spectralMatrix Q (fun i => (ndbScalar (s i) k).1),
        spectralMatrix Q (fun i => (ndbScalar (s i) k).2)) := by
  induction k with
  | zero => simp only [ndbIterate, ndbScalar, spectralMatrix_one Q hQ]
  | succ k ih =>
    simp only [ndbIterate, ih, ndbStep, ndbCorrection_spectrum Q _ _ hQ,
      spectralMatrix_mul Q _ _ hQ, ndbScalar]

/-- NDB spectral hypotheses are satisfiable, arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- NDB and CN of order two with `c=1` have identical inverse-root iterates.
Their different behavior in the source's scaling experiments comes from
different initial scalings, rather than a different scalar polynomial.
Source: arXiv:2602.02016v2, §3.2–3.4. -/
theorem ndb_cn_same_scaling (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hQ : Orthogonal Q) (k : ℕ) :
    (ndbIterate (spectralMatrix Q s) k).2 =
      (cnIterate 2 (spectralMatrix Q s) 1 k).1 := by
  rw [ndbIterate_spectrum Q s hQ, cnIterate_spectrum 2 Q s 1 hQ]
  simp only [ndbScalar_eq_cn]

/-- Equal-scaling comparison hypotheses are satisfiable,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Convergent finite eigenvalue coordinates give a convergent matrix.
Source: arXiv:2602.02016v2, §3.1–3.3, the spectral interpretation of the solvers. -/
theorem spectralMatrix_tendsto (Q : Matrix (Fin n) (Fin n) ℝ)
    (u : ℕ → Fin n → ℝ) (s : Fin n → ℝ)
    (hs : ∀ i, Filter.Tendsto (fun k => u k i) Filter.atTop (nhds (s i))) :
    Filter.Tendsto (fun k => spectralMatrix Q (u k)) Filter.atTop
      (nhds (spectralMatrix Q s)) := by
  have hcont : Continuous (spectralMatrix Q) := by
    unfold spectralMatrix Muon.singularMatrix
    fun_prop
  exact (hcont.tendsto s).comp (tendsto_pi_nhds.mpr hs)

/-- Coordinate convergence assumptions are satisfiable,
arXiv:2602.02016v2, §3.1. -/
example : ∀ i : Fin 1, Filter.Tendsto (fun _ : ℕ => (1 : ℝ))
    Filter.atTop (nhds ((fun _ : Fin 1 => (1 : ℝ)) i)) := fun _ => tendsto_const_nhds

end Transformer.DASH
