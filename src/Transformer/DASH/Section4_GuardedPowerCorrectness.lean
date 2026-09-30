/-
# DASH — guarded fitted powers specified on the original PSD input

arXiv:2602.02016v2, §2.2, §3.4, §4 and Appendix A. The EVD supplies
the mathematical target and proves its inverse-root identity. The actual
fitted algorithm uses no eigendecomposition and receives no eigenframe
or spectral upper bound as an input.
-/

import Transformer.DASH.Section4_GuardedChebyshev
import Transformer.DASH.Section3_InverseRootCorrectness

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {n : ℕ}

/-- Power of `A+εI` in the canonical EVD frame of the original symmetric
input. This is a target specification, not the fitted numerical algorithm.
Source: arXiv:2602.02016v2, §3.1 and Appendix A, corrected regularized target. -/
def regularizedEvdPower (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian)
    (ε exponent : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  spectralPower (evdFrame A hA) (fun i => hA.eigenvalues i + ε) exponent

/-- The target of the corrected Chebyshev inverse-root solver is an
actual inverse root of the original regularized matrix on both sides.
Source: arXiv:2602.02016v2, §2.2 and Appendix A, `A^(-1/p)` with regularization;
the printed normalized output requires the documented scale corrections. -/
theorem regularizedEvdPower_inverseRoot (A : Matrix (Fin n) (Fin n) ℝ)
    (hA : A.PosSemidef) (ε : ℝ) (hε : 0 < ε) (p : ℕ) (hp : 0 < p) :
    (A + ε • 1) * regularizedEvdPower A hA.1 ε (-(1 / (p : ℝ))) ^ p = 1 ∧
      regularizedEvdPower A hA.1 ε (-(1 / (p : ℝ))) ^ p * (A + ε • 1) = 1 := by
  have hshift : A + ε • 1 =
      spectralMatrix (evdFrame A hA.1) (fun i => hA.1.eigenvalues i + ε) := by
    calc
      _ = spectralMatrix (evdFrame A hA.1) hA.1.eigenvalues + ε • 1 :=
        congrArg (fun M => M + ε • 1) (evd_reconstruction A hA.1)
      _ = _ := spectralMatrix_shift _ _ ε (evdFrame_orthogonal A hA.1)
  rw [hshift]
  exact spectral_inverseRoot_identity _ _ p (evdFrame_orthogonal A hA.1)
    (fun i => add_pos_of_nonneg_of_pos (hA.eigenvalues_nonneg i) hε) hp

/-- The original-input inverse-root assumptions are satisfiable,
arXiv:2602.02016v2, §2.2 and Appendix A. -/
example : (0 : Matrix (Fin 1) (Fin 1) ℝ).PosSemidef ∧ (0 : ℝ) < 1 ∧ 0 < (4 : ℕ) :=
  ⟨Matrix.PosSemidef.zero, by norm_num, by norm_num⟩

/-- The complete guarded fitted algorithm approximates powers of the
original regularized PSD matrices. The caller supplies PSD, positive
regularization and tolerance; scale bounds and sample restrictions are
discharged by the algorithm itself. The degree threshold is common to
the entire finite bucket and works for every requested sample count.
Source: arXiv:2602.02016v2, §3.4, §4 and Appendix A, corrected fitted powers. -/
theorem countedBatchGuardedPower_posSemidef_accuracy [Finite ι]
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (r : ι → ℝ) (hA : ∀ j, (A j).PosSemidef)
    (ε exponent δ : ℝ) (hε : 0 < ε) (hδ : 0 < δ) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d → ∀ N : ℕ, ∀ j : ι, frobeniusNorm
      (((countedBatchGuardedPower A r ε exponent N d).run 0).1 j -
        regularizedEvdPower (A j) (hA j).1 ε exponent) < δ := by
  have h := countedBatchGuardedPower_uniform_accuracy
    (fun j => evdFrame (A j) (hA j).1) (fun j => (hA j).1.eigenvalues) r ε exponent δ
    (fun j => evdFrame_orthogonal (A j) (hA j).1) hε
    (fun j i => (hA j).eigenvalues_nonneg i) hδ
  simpa only [← evd_reconstruction, regularizedEvdPower] using h

/-- The original-PSD accuracy assumptions are satisfiable in a nonempty
bucket, arXiv:2602.02016v2, §3.4, §4 and Appendix A. -/
example : (∀ j : Fin 2, ((fun _ : Fin 2 => (0 : Matrix (Fin 1) (Fin 1) ℝ)) j).PosSemidef) ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 :=
  ⟨fun _ => Matrix.PosSemidef.zero, by norm_num, by norm_num⟩

/-- Actual multi-PI preprocessing preserves the same uniform fitted
accuracy, even with zero or badly aligned starts and any finite PI budget.
Source: arXiv:2602.02016v2, §3.4–3.5, §4 and Appendix A, repaired PI pipeline. -/
theorem countedBatchPiPower_posSemidef_accuracy {κ : Type} [Fintype κ] [Nonempty κ]
    [Finite ι] (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (starts : ι → κ → Fin n → ℝ) (piSteps : ℕ) (hA : ∀ j, (A j).PosSemidef)
    (ε exponent δ : ℝ) (hε : 0 < ε) (hδ : 0 < δ) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d → ∀ N : ℕ, ∀ j : ι, frobeniusNorm
      (((countedBatchPiPower A starts piSteps ε exponent N d).run 0).1 j -
        regularizedEvdPower (A j) (hA j).1 ε exponent) < δ :=
  countedBatchGuardedPower_posSemidef_accuracy A
    (fun j => pooledRayleigh (A j) (starts j) piSteps) hA ε exponent δ hε hδ

/-- The multi-PI fitted-accuracy contract is satisfiable,
arXiv:2602.02016v2, §3.4–3.5, §4 and Appendix A. -/
example : (∀ j : Fin 2, ((fun _ : Fin 2 => (0 : Matrix (Fin 1) (Fin 1) ℝ)) j).PosSemidef) ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 :=
  ⟨fun _ => Matrix.PosSemidef.zero, by norm_num, by norm_num⟩

end Transformer.DASH
