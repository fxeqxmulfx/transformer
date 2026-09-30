/-
# DASH — the complete finite batched NDB inverse-fourth-root procedure

arXiv:2602.02016v2, §3.3–3.4 and §4. Both NDB calls are finite
and counted. Input normalization and output rescaling are explicit;
the numerical error of the first square-root call is included.
-/

import Transformer.DASH.Section4_BatchedCounts
import Transformer.DASH.Section3_FiniteChaining
import Transformer.DASH.Section3_Scaling

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {n : ℕ}

/-- The source's two-call inverse-fourth-root solver for an entire bucket:
first compute the square root buffer, then compute its inverse square root.
Source: arXiv:2602.02016v2, §3.3 and §4, stacked NDB chaining. -/
def countedBatchInverseFourth (A : ι → Matrix (Fin n) (Fin n) ℝ) (k : ℕ) :
    StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ) := do
  let first ← countedBatchNdbIterate A k
  let second ← countedBatchNdbIterate first.1 k
  pure second.2

/-- The complete two-call procedure equals two finite NDB solves per
matrix and uses `2*max(3k-2,0)` batched products. No exact square root is
substituted for the first numerical call.
Source: arXiv:2602.02016v2, §3.3 and §4, inverse-fourth-root chaining. -/
theorem countedBatchInverseFourth_correct (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (k count : ℕ) :
    (countedBatchInverseFourth A k).run count =
      (fun j => (ndbIterate (ndbIterate (A j) k).1 k).2,
        count + 2 * (3 * k - 2)) := by
  unfold countedBatchInverseFourth
  rw [StateT.run_bind, countedBatchNdbIterate_correct]
  change ((countedBatchNdbIterate (batchNdbIterate A k).1 k >>= fun state =>
    pure state.2).run (count + (3 * k - 2))) = _
  rw [StateT.run_bind, countedBatchNdbIterate_correct]
  change ((batchNdbIterate (batchNdbIterate A k).1 k).2,
    count + (3 * k - 2) + (3 * k - 2)) = _
  simp only [batchNdbIterate_eq, two_mul, Nat.add_assoc]

/-- With a positive spectral upper bound for each input, the actual
finite batched solver on `A/c`, followed by `c^(-1/4)` output rescaling,
converges to the inverse fourth roots of the original matrices. This is
convergence of the whole function of batch indices, and therefore of
every finite stacked bucket. Scaling is independent for each block.
Source: arXiv:2602.02016v2, §3.3–3.4 and §4, normalization and stacked roots. -/
theorem countedBatchInverseFourth_rescaled_convergence
    (Q : ι → Matrix (Fin n) (Fin n) ℝ) (s : ι → Fin n → ℝ) (c : ι → ℝ)
    (hQ : ∀ j, Orthogonal (Q j)) (hs : ∀ j i, 0 < s j i)
    (hc : ∀ j, 0 < c j) (hupper : ∀ j i, s j i ≤ c j) :
    Tendsto (fun k : ℕ => fun j => c j ^ (-(1 / 4 : ℝ)) •
      ((countedBatchInverseFourth
        (fun j => (c j)⁻¹ • spectralMatrix (Q j) (s j)) (k + 1)).run 0).1 j)
      atTop (𝓝 (fun j => spectralPower (Q j) (s j) (-(1 / 4 : ℝ)))) := by
  apply tendsto_pi_nhds.mpr
  intro j
  have hpositive : ∀ i, 0 < s j i / c j := fun i => div_pos (hs j i) (hc j)
  have hdomain : ∀ i, s j i / c j < 2 := fun i =>
    lt_of_le_of_lt ((div_le_one (hc j)).mpr (hupper j i)) (by norm_num)
  have hnormalized : (c j)⁻¹ • spectralMatrix (Q j) (s j) =
      spectralMatrix (Q j) (fun i => s j i / c j) := by
    rw [spectralMatrix_smul]
    congr 1
    funext i
    simp [div_eq_mul_inv, mul_comm]
  have h := (ndb_matrix_finite_chain_convergence (Q j)
    (fun i => s j i / c j) (hQ j) hpositive hdomain).const_smul (c j ^ (-(1 / 4 : ℝ)))
  rw [spectralPower_rescale (Q j) (s j) (c j) (-(1 / 4 : ℝ))
    (hc j) (fun i => (hs j i).le)] at h
  simpa only [countedBatchInverseFourth_correct, hnormalized] using h

/-- A nonempty bucket satisfies the complete normalized-chain assumptions,
arXiv:2602.02016v2, §3.3–3.4 and §4. -/
example : (∀ j : Fin 2, Orthogonal ((fun _ : Fin 2 =>
    (1 : Matrix (Fin 1) (Fin 1) ℝ)) j)) ∧
    (∀ j : Fin 2, ∀ i : Fin 1, (0 : ℝ) < (fun (_ : Fin 2) (_ : Fin 1) => (1 : ℝ)) j i) ∧
    (∀ j : Fin 2, (0 : ℝ) < (fun _ : Fin 2 => (2 : ℝ)) j) ∧
    (∀ j : Fin 2, ∀ i : Fin 1,
      (fun (_ : Fin 2) (_ : Fin 1) => (1 : ℝ)) j i ≤ (fun _ : Fin 2 => (2 : ℝ)) j) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
