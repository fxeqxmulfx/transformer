/-
# DASH — guarded inverse roots for the original positive-definite matrix

arXiv:2602.02016v2, §3.2–3.5. No eigenbasis, positive scale or
reliable PI estimate is supplied by the caller. The EVD is used in
the proof and target specification, not in these iterative algorithms.
-/

import Transformer.DASH.Section3_GuardedNewton
import Transformer.DASH.Section2_History

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The corrected CN algorithm converges to the original matrix's exact
EVD inverse root. Positive definiteness and positive order suffice.
Source: arXiv:2602.02016v2, §3.2–3.4, corrected CN normalization. -/
theorem guardedCnIterate_posDef (p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)
    (r : ℝ) (hA : A.PosDef) (hp : 0 < p) :
    Tendsto (guardedCnIterate p A r) atTop (𝓝 (evdInverseRoot A hA p)) := by
  have h := guardedCnIterate_convergence p (evdFrame A hA.1) hA.1.eigenvalues r
    (evdFrame_orthogonal A hA.1) hA.eigenvalues_pos hp
  simpa only [← evd_reconstruction A hA.1, evdInverseRoot] using h

/-- The original-matrix CN assumptions are satisfiable,
arXiv:2602.02016v2, §3.2–3.4. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef ∧ 0 < (4 : ℕ) := by
  refine ⟨?_, by norm_num⟩
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

/-- The corrected NDB algorithm converges to both roots of the original
positive-definite input for every candidate estimate.
Source: arXiv:2602.02016v2, §3.3–3.4, corrected NDB normalization. -/
theorem guardedNdbIterate_posDef (A : Matrix (Fin n) (Fin n) ℝ)
    (r : ℝ) (hA : A.PosDef) :
    Tendsto (guardedNdbIterate A r) atTop
      (𝓝 (spectralPower (evdFrame A hA.1) hA.1.eigenvalues (1 / 2),
        evdInverseRoot A hA 2)) := by
  have h := guardedNdbIterate_convergence (evdFrame A hA.1) hA.1.eigenvalues r
    (evdFrame_orthogonal A hA.1) hA.eigenvalues_pos
  simpa only [← evd_reconstruction A hA.1, evdInverseRoot, Nat.cast_ofNat] using h

/-- The original-matrix NDB assumption is satisfiable,
arXiv:2602.02016v2, §3.3–3.4. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef := by
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

/-- Both finite NDB calls, together with automatic normalization and
output rescaling, converge to the original input's inverse fourth root.
Source: arXiv:2602.02016v2, §3.3–3.4, corrected chained solver. -/
theorem guardedInverseFourth_posDef (A : Matrix (Fin n) (Fin n) ℝ)
    (r : ℝ) (hA : A.PosDef) :
    Tendsto (fun k : ℕ => guardedInverseFourth A r (k + 1)) atTop
      (𝓝 (evdInverseRoot A hA 4)) := by
  have h := guardedInverseFourth_convergence (evdFrame A hA.1) hA.1.eigenvalues r
    (evdFrame_orthogonal A hA.1) hA.eigenvalues_pos
  simpa only [← evd_reconstruction A hA.1, evdInverseRoot, Nat.cast_ofNat] using h

/-- The original-matrix chained-root assumption is satisfiable,
arXiv:2602.02016v2, §3.3–3.4. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef := by
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

/-- Actual multi-PI followed by the corrected CN solver. The independent
guard is inside `guardedCnIterate`; start vectors need not be nonzero.
Source: arXiv:2602.02016v2, §3.2 and §3.4–3.5, repaired PI pipeline. -/
def pooledGuardedCnIterate {κ : Type} [Fintype κ] [Nonempty κ]
    (p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (starts : κ → Fin n → ℝ)
    (piSteps rootSteps : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  guardedCnIterate p A (pooledRayleigh A starts piSteps) rootSteps

/-- The complete multi-PI/CN pipeline converges for arbitrary starts
and any fixed PI budget, including zero iterations.
Source: arXiv:2602.02016v2, §3.2 and §3.4–3.5, corrected scaling guarantee. -/
theorem pooledGuardedCnIterate_posDef {κ : Type} [Fintype κ] [Nonempty κ]
    (p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (starts : κ → Fin n → ℝ)
    (piSteps : ℕ) (hA : A.PosDef) (hp : 0 < p) :
    Tendsto (pooledGuardedCnIterate p A starts piSteps) atTop
      (𝓝 (evdInverseRoot A hA p)) :=
  guardedCnIterate_posDef p A (pooledRayleigh A starts piSteps) hA hp

/-- The complete PI pipeline's assumptions are satisfiable with zero
start vectors, arXiv:2602.02016v2, §3.2 and §3.4–3.5. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef ∧ 0 < (4 : ℕ) := by
  refine ⟨?_, by norm_num⟩
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

end Transformer.DASH
