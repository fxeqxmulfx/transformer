/-
# DASH — automatically normalized batched NDB inverse fourth roots

arXiv:2602.02016v2, §3.3–3.5 and §4. The batch solver includes the
guard and output rescaling. Its input contract is positive definiteness,
not a presumed bound on a PI estimate. Counts refer only to root-solver
batched matrix products, as in the source, not preprocessing or PI work.
-/

import Transformer.DASH.Section4_BatchedFourthRoot
import Transformer.DASH.Section3_GuardedRootCorrectness

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {n : ℕ}

/-- Guard each block independently, perform both finite batched NDB
calls, and restore the original inverse-fourth-root scale.
Source: arXiv:2602.02016v2, §3.3–3.4 and §4, corrected batched root pipeline. -/
def countedBatchGuardedInverseFourth (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (r : ι → ℝ) (k : ℕ) : StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ) := do
  let root ← countedBatchInverseFourth (fun j => guardedNormalize (A j) (r j)) k
  pure (fun j => guardedScale (A j) (r j) ^ (-(1 / 4 : ℝ)) • root j)

/-- The complete guarded batch call preserves the single-matrix solver
and the `2*max(3k-2,0)` root-product count for every iteration budget.
Source: arXiv:2602.02016v2, §3.3 and §4, counted stacked inverse fourth roots. -/
theorem countedBatchGuardedInverseFourth_correct
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (r : ι → ℝ) (k count : ℕ) :
    (countedBatchGuardedInverseFourth A r k).run count =
      (fun j => guardedInverseFourth (A j) (r j) k, count + 2 * (3 * k - 2)) := by
  unfold countedBatchGuardedInverseFourth
  rw [StateT.run_bind, countedBatchInverseFourth_correct]
  rfl

/-- For positive-definite inputs the actual batched pipeline converges
to their canonical inverse fourth roots. Scale and spectral-domain conditions
are supplied by the algorithm, for arbitrary estimates on every block.
Source: arXiv:2602.02016v2, §3.3–3.4 and §4, corrected batched NDB convergence. -/
theorem countedBatchGuardedInverseFourth_convergence
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (r : ι → ℝ) (hA : ∀ j, (A j).PosDef) :
    Tendsto (fun k : ℕ => ((countedBatchGuardedInverseFourth A r (k + 1)).run 0).1)
      atTop (𝓝 (fun j => evdInverseRoot (A j) (hA j) 4)) := by
  apply tendsto_pi_nhds.mpr
  intro j
  simpa only [countedBatchGuardedInverseFourth_correct] using
    guardedInverseFourth_posDef (A j) (r j) (hA j)

/-- A nonempty bucket meets the complete solver's assumptions,
arXiv:2602.02016v2, §3.3–3.4 and §4. -/
example : ∀ j : Fin 2, ((fun _ : Fin 2 => (1 : Matrix (Fin 1) (Fin 1) ℝ)) j).PosDef := by
  intro j
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

/-- Complete multi-PI and batched NDB pipeline with the safety guard
inside the root solver. Arbitrary starts and PI budgets are accepted.
Source: arXiv:2602.02016v2, §3.3–3.5 and §4, corrected multi-PI preprocessing. -/
def countedBatchPiInverseFourth {κ : Type} [Fintype κ] [Nonempty κ]
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (starts : ι → κ → Fin n → ℝ)
    (piSteps rootSteps : ℕ) : StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ) :=
  countedBatchGuardedInverseFourth A (fun j => pooledRayleigh (A j) (starts j) piSteps)
    rootSteps

/-- Fixed finite PI preprocessing needs no accuracy assumption: the
complete algorithm converges even for starts orthogonal to the top eigenspace.
Source: arXiv:2602.02016v2, §3.3–3.5 and §4, repaired PI safety guarantee. -/
theorem countedBatchPiInverseFourth_convergence {κ : Type} [Fintype κ] [Nonempty κ]
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (starts : ι → κ → Fin n → ℝ)
    (piSteps : ℕ) (hA : ∀ j, (A j).PosDef) :
    Tendsto (fun k : ℕ => ((countedBatchPiInverseFourth A starts piSteps (k + 1)).run 0).1)
      atTop (𝓝 (fun j => evdInverseRoot (A j) (hA j) 4)) :=
  countedBatchGuardedInverseFourth_convergence A
    (fun j => pooledRayleigh (A j) (starts j) piSteps) hA

/-- The PI pipeline's input contract is satisfiable,
arXiv:2602.02016v2, §3.3–3.5 and §4. -/
example : ∀ j : Fin 2, ((fun _ : Fin 2 => (1 : Matrix (Fin 1) (Fin 1) ℝ)) j).PosDef := by
  intro j
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

end Transformer.DASH
