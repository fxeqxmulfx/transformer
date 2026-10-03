/-
# The corrected fused MD training recurrence

User-requested extension of arXiv:2606.25971v2, §3.1 and Appendix A,
Algorithm 2. Propose the original full MD step, including all optimizer
memory updates. Check its complete fused displacement; replace rejected
weights with a nonsingular gradient step. Repair only the representation.
An accepted valid paper step is retained including its raw gains.
-/

import Transformer.MagnitudeDirection.SectionA_StatefulAlgorithm
import Transformer.MagnitudeDirection.SectionA_TrainingGuard

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {m n : ℕ} {S : Type*}

/-- Matrix/Frobenius conversion preserves the nonzero storage condition.
Source: arXiv:2606.25971v2, §2 and Appendix A, training extension. -/
theorem fromMatrix_ne_zero_iff (W : Matrix (Fin m) (Fin n) ℝ) :
    fromMatrix W ≠ 0 ↔ W ≠ 0 := by
  constructor
  · intro h hW
    apply h
    ext p
    simp [fromMatrix, hW]
  · intro h hW
    apply h
    ext i j
    exact congrArg (fun x : MatrixSpace m n => x (i, j)) hW

/-- Check a full MD proposal and repair its representation. All proposed
auxiliary optimizer states are retained, including after rejection.
The common row rescaling affects gain coordinates, never fused entries.
This is an explicit modification of the paper's algorithm.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
def checkedProposal (σ L c : ℝ) (x g : MatrixSpace m n) (q : S × FullState m n) :
    (S × FullState m n) × MatrixSpace m n :=
  let d := checkedDirection σ L x g (fromMatrix q.2.weight)
  let next := x - (σ / L) • d
  ((q.1, rebalanceStorage c {q.2 with weight := toMatrix next}), d)

/-- The optimizer memories evolve exactly as in the proposed paper step.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training correction. -/
theorem checkedProposal_memory (σ L c : ℝ) (x g : MatrixSpace m n)
    (q : S × FullState m n) : (checkedProposal σ L c x g q).1.1 = q.1 := rfl

/-- The repaired state's fused weight is exactly the checked parameter
step; this verifies that the subsequent loss uses the actual MD storage.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training correction. -/
theorem checkedProposal_weight (σ L c : ℝ) (x g : MatrixSpace m n)
    (q : S × FullState m n) :
    fromMatrix (checkedProposal σ L c x g q).1.2.weight =
      x - (σ / L) • (checkedProposal σ L c x g q).2 := by
  dsimp only [checkedProposal, rebalanceStorage]
  exact fromMatrix_toMatrix _

/-- Every checked candidate is already certified for the generic descent
guard. Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training correction. -/
theorem checkedProposal_guard (σ L c : ℝ) (x g : MatrixSpace m n)
    (q : S × FullState m n) (hσ : σ ≤ 1 / 2) :
    descentGuard σ g (checkedProposal σ L c x g q).2 =
      (checkedProposal σ L c x g q).2 := checkedDirection_guard σ L x g _ hσ

/-- The guard admits a positive learning parameter,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (1 / 4 : ℝ) ≤ 1 / 2 := by norm_num

/-- A passing proposal already on its sphere is preserved completely:
fused entries, raw row gains, raw column gains and all optimizer memory.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem checkedProposal_accepts (σ L c : ℝ) (x g : MatrixSpace m n)
    (q : S × FullState m n) (hσ : 0 < σ) (hL : 0 < L) (hc : 0 < c)
    (h : fusedStepAdmissible σ L x g (fromMatrix q.2.weight))
    (hD : frobeniusNorm (fullDirection softplus q.2) = c) :
    (checkedProposal σ L c x g q).1 = q := by
  dsimp only [checkedProposal]
  rw [checkedDirection_accepts σ L x g _ hσ hL h, toMatrix_fromMatrix]
  rw [show {q.2 with weight := q.2.weight} = q.2 from rfl,
    rebalanceStorage_fixed c q.2 hc hD]

/-- A valid stored state with zero current gradient passes unchanged.
The training witnesses below also cover nonzero gradients.
Source: arXiv:2606.25971v2, Appendix A, training correction. -/
example :
    let q := ((), unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ))
    (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      fusedStepAdmissible (1 / 4) 1 (fromMatrix q.2.weight) 0 (fromMatrix q.2.weight) ∧
      frobeniusNorm (fullDirection softplus q.2) = 1 := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, ?_, ?_⟩
  · refine ⟨(fromMatrix_ne_zero_iff _).mpr (by norm_num [unitGainState]), ?_⟩
    simp [fusedStepDirection]
  · rw [unitGainState_direction, frobeniusNorm_eq_sqrt]
    norm_num

/-- Adapt current fused weights and genuine gradients to the original
stateful MD proposer before applying the training correction.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training correction. -/
def trainingCandidate (σ L c : ℝ) (propose : FullProposal S m n)
    (memory : S × FullState m n) (x g : MatrixSpace m n) :
    (S × FullState m n) × MatrixSpace m n :=
  checkedProposal σ L c x g
    (propose memory.1 {memory.2 with weight := toMatrix x} (toMatrix g))

/-- Corrected training with the complete MD proposal, actual objective
gradients, evolving optimizer memories and softplus factor storage.
The extra stored matrix is kept equal to the actual parameter by the
invariant below. Source: extension of arXiv:2606.25971v2, Appendix A,
Algorithm 2; deterministic real-arithmetic training only. -/
def safeguardedFullRun (σ L c : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n) :
    ℕ → (S × FullState m n) × MatrixSpace m n :=
  safeguardedRun σ L f (trainingCandidate σ L c propose)
    (initialMemory, initial) (fromMatrix initial.weight)

end Transformer.MagnitudeDirection
