/-
# DASH — convergent finite matrix chains for dyadic powers

arXiv:2602.02016v2, §3.3. Every intermediate matrix is computed by
a finite NDB solve. For fixed chain depth, increasing each solve's
iteration count converges to the corresponding positive or negative
dyadic matrix power.
-/

import Transformer.DASH.Section3_DyadicChaining

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Repeated actual finite matrix square-root calls.
Source: arXiv:2602.02016v2, §3.3, chaining the NDB procedure. -/
def ndbFiniteMatrixChain (A : Matrix (Fin n) (Fin n) ℝ) (iterations : ℕ) :
    ℕ → Matrix (Fin n) (Fin n) ℝ
  | 0 => A
  | m + 1 => (ndbIterate (ndbFiniteMatrixChain A iterations m) iterations).1

/-- Each computed matrix chain has the original eigenframe and the
corresponding finite scalar chain on its diagonal.
Source: arXiv:2602.02016v2, §3.3, the spectral meaning of chained solves. -/
theorem ndbFiniteMatrixChain_spectrum (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (iterations m : ℕ) :
    ndbFiniteMatrixChain (spectralMatrix Q s) iterations m =
      spectralMatrix Q (fun i => ndbFiniteSquareChain (s i) iterations m) := by
  induction m with
  | zero => rfl
  | succ m ih =>
    rw [ndbFiniteMatrixChain, ih, ndbIterate_spectrum Q _ hQ]
    rfl

/-- A finite-chain orthogonal frame exists,
arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- A fixed chain of finite matrix square-root solves converges to
`A^(1/2^m)` as the common finite iteration count grows.
Source: arXiv:2602.02016v2, §3.3, the positive dyadic powers. -/
theorem ndbFiniteMatrixChain_convergence (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i)
    (hupper : ∀ i, s i < 2) (m : ℕ) :
    Tendsto (fun k : ℕ => ndbFiniteMatrixChain (spectralMatrix Q s) (k + 1) m)
      atTop (𝓝 (spectralPower Q s ((1 / 2 : ℝ) ^ m))) := by
  have h := spectralMatrix_tendsto Q
    (fun k i => ndbFiniteSquareChain (s i) (k + 1) m)
    (fun i => s i ^ ((1 / 2 : ℝ) ^ m))
    (fun i => ndbFiniteSquareChain_convergence (s i) (hs i) (hupper i) m)
  simpa only [← ndbFiniteMatrixChain_spectrum Q s hQ, spectralPower] using h

/-- A nonempty positive-definite chain domain exists,
arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i < 2) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- A final finite inverse-square-root solve after a fixed finite matrix
square-root chain converges to `A^(-1/2^(m+1))`. The case `m=1` is the
source's inverse-fourth-root method, with all intermediate errors included.
Source: arXiv:2602.02016v2, §3.3, negative dyadic powers from NDB calls. -/
theorem ndbFiniteMatrixInverseChain_convergence (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i)
    (hupper : ∀ i, s i < 2) (m : ℕ) :
    Tendsto (fun k : ℕ =>
      (ndbIterate (ndbFiniteMatrixChain (spectralMatrix Q s) (k + 1) m) (k + 1)).2)
      atTop (𝓝 (spectralPower Q s (-((1 / 2 : ℝ) ^ (m + 1))))) := by
  have h := spectralMatrix_tendsto Q
    (fun k i => (ndbScalar (ndbFiniteSquareChain (s i) (k + 1) m) (k + 1)).2)
    (fun i => s i ^ (-((1 / 2 : ℝ) ^ (m + 1))))
    (fun i => ndbFiniteInverseChain_convergence (s i) (hs i) (hupper i) m)
  have hsequence : (fun k : ℕ =>
      (ndbIterate (ndbFiniteMatrixChain (spectralMatrix Q s) (k + 1) m) (k + 1)).2) =
      fun k : ℕ => spectralMatrix Q
        (fun i => (ndbScalar (ndbFiniteSquareChain (s i) (k + 1) m) (k + 1)).2) := by
    funext k
    rw [ndbFiniteMatrixChain_spectrum Q s hQ, ndbIterate_spectrum Q _ hQ]
  rw [hsequence]
  exact h

/-- Final inverse-chain assumptions are satisfiable,
arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i < 2) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
