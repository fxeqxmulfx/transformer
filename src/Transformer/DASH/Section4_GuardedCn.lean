/-
# DASH — complete guarded batched CN inverse fourth roots

arXiv:2602.02016v2, §3.2–3.5 and §4. Each recurrence step uses
the four-product CN implementation. The guard repairs both the
unit-domain endpoint and unreliable PI estimates before iteration.
-/

import Transformer.DASH.Section4_BatchedCounts
import Transformer.DASH.Section3_GuardedRootCorrectness

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {n : ℕ}

/-- Full fourth-order CN recurrence using the counted batched primitive.
Source: arXiv:2602.02016v2, §3.2 and §4, four products per iteration. -/
def countedBatchCnFourIterate (A : ι → Matrix (Fin n) (Fin n) ℝ) (c : ι → ℝ) :
    ℕ → StateM ℕ ((ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ))
  | 0 => pure (batchCnIterate 4 A c 0)
  | k + 1 => do
      let state ← countedBatchCnFourIterate A c k
      countedBatchCnFour state

/-- Every counted iterate equals the batched CN recurrence and costs
exactly `4k` root-solver products, regardless of bucket cardinality.
Source: arXiv:2602.02016v2, §3.2 and §4, counted batched CN. -/
theorem countedBatchCnFourIterate_correct (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (c : ι → ℝ) (k count : ℕ) :
    (countedBatchCnFourIterate A c k).run count =
      (batchCnIterate 4 A c k, count + 4 * k) := by
  induction k generalizing count with
  | zero => rfl
  | succ k ih =>
    change ((countedBatchCnFourIterate A c k >>= countedBatchCnFour).run count) = _
    rw [StateT.run_bind, ih]
    change (countedBatchCnFour (batchCnIterate 4 A c k)).run (count + 4 * k) = _
    rw [countedBatchCnFour_correct]
    change (batchCnIterate 4 A c (k + 1), count + 4 * k + 4) = _
    congr 1

/-- Automatically normalized CN in the paper's comparison regime,
followed by inverse-fourth-root output rescaling on each block.
Source: arXiv:2602.02016v2, §3.2–3.4 and §4, repaired batched CN pipeline. -/
def countedBatchGuardedCnFour (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (r : ι → ℝ) (k : ℕ) : StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ) := do
  let state ← countedBatchCnFourIterate (fun j => guardedNormalize (A j) (r j))
    (fun _ => cnUnitScale 4) k
  pure (fun j => guardedScale (A j) (r j) ^ (-(1 / 4 : ℝ)) • state.1 j)

/-- The full guarded batched pipeline equals corrected CN for every
block and uses `4k` products. Certification and PI are preprocessing;
these counts make no assertion about their cost or wall-clock speed.
Source: arXiv:2602.02016v2, §3.2 and §4, CN product count. -/
theorem countedBatchGuardedCnFour_correct (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (r : ι → ℝ) (k count : ℕ) :
    (countedBatchGuardedCnFour A r k).run count =
      (fun j => guardedCnIterate 4 (A j) (r j) k, count + 4 * k) := by
  unfold countedBatchGuardedCnFour
  rw [StateT.run_bind, countedBatchCnFourIterate_correct]
  simp only [batchCnIterate_eq]
  rfl

/-- The actual guarded CN batch converges on arbitrary positive-definite
matrices without assumptions on the candidate PI estimates.
Source: arXiv:2602.02016v2, §3.2–3.4 and §4, corrected CN solver domain. -/
theorem countedBatchGuardedCnFour_convergence
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (r : ι → ℝ) (hA : ∀ j, (A j).PosDef) :
    Tendsto (fun k : ℕ => ((countedBatchGuardedCnFour A r k).run 0).1)
      atTop (𝓝 (fun j => evdInverseRoot (A j) (hA j) 4)) := by
  apply tendsto_pi_nhds.mpr
  intro j
  simpa only [countedBatchGuardedCnFour_correct] using
    guardedCnIterate_posDef 4 (A j) (r j) (hA j) (by norm_num)

/-- A nonempty bucket satisfies the guarded CN input contract,
arXiv:2602.02016v2, §3.2–3.4 and §4. -/
example : ∀ j : Fin 2, ((fun _ : Fin 2 => (1 : Matrix (Fin 1) (Fin 1) ℝ)) j).PosDef := by
  intro j
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

/-- Complete multi-PI preprocessing followed by guarded batched CN.
Source: arXiv:2602.02016v2, §3.2 and §3.4–3.5 and §4, corrected PI pipeline. -/
def countedBatchPiCnFour {κ : Type} [Fintype κ] [Nonempty κ]
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (starts : ι → κ → Fin n → ℝ)
    (piSteps rootSteps : ℕ) : StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ) :=
  countedBatchGuardedCnFour A (fun j => pooledRayleigh (A j) (starts j) piSteps) rootSteps

/-- The complete multi-PI/CN batch converges for arbitrary starts and
any fixed PI budget; the safety check is part of the algorithm.
Source: arXiv:2602.02016v2, §3.2 and §3.4–3.5 and §4. -/
theorem countedBatchPiCnFour_convergence {κ : Type} [Fintype κ] [Nonempty κ]
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (starts : ι → κ → Fin n → ℝ)
    (piSteps : ℕ) (hA : ∀ j, (A j).PosDef) :
    Tendsto (fun k : ℕ => ((countedBatchPiCnFour A starts piSteps k).run 0).1)
      atTop (𝓝 (fun j => evdInverseRoot (A j) (hA j) 4)) :=
  countedBatchGuardedCnFour_convergence A
    (fun j => pooledRayleigh (A j) (starts j) piSteps) hA

/-- The full PI/CN pipeline's input contract is satisfiable,
arXiv:2602.02016v2, §3.2 and §3.4–3.5 and §4. -/
example : ∀ j : Fin 2, ((fun _ : Fin 2 => (1 : Matrix (Fin 1) (Fin 1) ℝ)) j).PosDef := by
  intro j
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

end Transformer.DASH
