/-
# DASH — matrix root iteration models

arXiv:2602.02016v2, §3.2–3.3. The exact recurrences are kept separate
from the spectral and convergence proofs. The first NDB step is reduced
to one matrix multiplication, as described in §3.3.
-/

import Transformer.DASH.Section3_Spectral
import Transformer.DASH.Section3_CoupledNewton

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The CN correction matrix, arXiv:2602.02016v2, §3.2, `equation:CN-C`. -/
def cnCorrection (p : ℕ) (M : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  (1 + 1 / (p : ℝ)) • 1 - (1 / (p : ℝ)) • M

/-- One CN step for the pair `(X,M)`,
arXiv:2602.02016v2, §3.2, `equation:CN-X-M`. -/
def cnStep (p : ℕ) (state : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ :=
  let C := cnCorrection p state.2
  (state.1 * C, C ^ p * state.2)

/-- Coupled Newton from the printed initialization,
arXiv:2602.02016v2, §3.2, `equation:CN-init`–`equation:CN-X-M`. -/
def cnIterate (p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (c : ℝ) :
    ℕ → Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ
  | 0 => (c⁻¹ • 1, (c ^ p)⁻¹ • A)
  | k + 1 => cnStep p (cnIterate p A c k)

/-- The NDB correction, arXiv:2602.02016v2, §3.3, `equation:NDB-E`. -/
def ndbCorrection (Y Z : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  (1 / 2 : ℝ) • ((3 : ℝ) • 1 - Z * Y)

/-- One NDB step for `(Y,Z)`, arXiv:2602.02016v2, §3.3, `equation:NDB-Y-Z`. -/
def ndbStep (state : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ :=
  let E := ndbCorrection state.1 state.2
  (state.1 * E, E * state.2)

/-- Newton–Denman–Beavers with `Y₀=A,Z₀=I`,
arXiv:2602.02016v2, §3.3, `equation:NDB-init`–`equation:NDB-Y-Z`. -/
def ndbIterate (A : Matrix (Fin n) (Fin n) ℝ) :
    ℕ → Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ
  | 0 => (A, 1)
  | k + 1 => ndbStep (ndbIterate A k)

/-- Scalar NDB, with the multiplication order printed in the matrix rule,
arXiv:2602.02016v2, §3.3, `equation:NDB-Y-Z`. -/
def ndbScalar (a : ℝ) : ℕ → ℝ × ℝ
  | 0 => (a, 1)
  | k + 1 =>
      let yz := ndbScalar a k
      let E := (3 - yz.2 * yz.1) / 2
      (yz.1 * E, E * yz.2)

/-- The first NDB step has two redundant identity multiplications.
The source calls the first correction `E₁` in prose, though its recurrence
indexes it `E₀`; the resulting first state is the same.
Source: arXiv:2602.02016v2, §3.3, first-iteration optimization. -/
theorem ndb_first_step (A : Matrix (Fin n) (Fin n) ℝ) :
    ndbIterate A 1 =
      (A * ((3 / 2 : ℝ) • 1 - (1 / 2 : ℝ) • A),
        (3 / 2 : ℝ) • 1 - (1 / 2 : ℝ) • A) := by
  have hE : ndbCorrection A 1 = (3 / 2 : ℝ) • 1 - (1 / 2 : ℝ) • A := by
    simp only [ndbCorrection, Matrix.one_mul, smul_sub, smul_smul]
    norm_num
  simp only [ndbIterate, ndbStep, hE, Matrix.mul_one]

/-- Scalar NDB and square-root CN with `c=1` have the same inverse-root
sequence; the NDB square-root sequence is `a X_k`.
Source: arXiv:2602.02016v2, §3.2–3.3 and §3.4. -/
theorem ndbScalar_eq_cn (a : ℝ) (k : ℕ) :
    ndbScalar a k = (a * cnScalarX 2 a 1 k, cnScalarX 2 a 1 k) := by
  induction k with
  | zero => simp [ndbScalar, cnScalarX]
  | succ k ih =>
    have hE : (3 - cnScalarX 2 a 1 k * (a * cnScalarX 2 a 1 k)) / 2 =
        cnFactor 2 (cnScalarM 2 a 1 k) := by
      rw [cnScalar_invariant]
      norm_num [cnFactor]
      ring
    simp only [ndbScalar, ih, hE, cnScalarX]
    ext <;> ring

/-- CN correction respects the eigendecomposition,
arXiv:2602.02016v2, §3.2, `equation:CN-C`. -/
theorem cnCorrection_spectrum (p : ℕ) (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) :
    cnCorrection p (spectralMatrix Q s) = spectralMatrix Q (fun i => cnFactor p (s i)) := by
  rw [cnCorrection, ← spectralMatrix_one Q hQ, spectralMatrix_smul,
    spectralMatrix_smul, spectralMatrix_sub]
  congr 1
  funext i
  unfold cnFactor
  ring

/-- Correction hypotheses are satisfiable, arXiv:2602.02016v2, §3.2. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- NDB correction respects the eigendecomposition,
arXiv:2602.02016v2, §3.3, `equation:NDB-E`. -/
theorem ndbCorrection_spectrum (Q : Matrix (Fin n) (Fin n) ℝ)
    (y z : Fin n → ℝ) (hQ : Orthogonal Q) :
    ndbCorrection (spectralMatrix Q y) (spectralMatrix Q z) =
      spectralMatrix Q (fun i => (3 - z i * y i) / 2) := by
  rw [ndbCorrection, ← spectralMatrix_one Q hQ, spectralMatrix_mul Q z y hQ,
    spectralMatrix_smul, spectralMatrix_sub, spectralMatrix_smul]
  congr 1
  funext i
  ring

/-- NDB correction hypotheses are satisfiable, arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
