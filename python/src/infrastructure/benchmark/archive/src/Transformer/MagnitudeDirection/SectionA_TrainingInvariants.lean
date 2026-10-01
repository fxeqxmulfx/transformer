/-
# Representation invariants throughout corrected training

User-requested extension of arXiv:2606.25971v2, Appendix A, Algorithm 2.
The parameter whose loss is differentiated equals the stored fused
weight. Every finite iterate is nonzero, every raw gain has a positive
effective value, and the recovered direction remains on its sphere.
The limit weight is allowed to be zero; no bounded-raw-gain premise is used.
-/

import Transformer.MagnitudeDirection.SectionA_TrainingModels

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {m n : ℕ} {S : Type*}

/-- The corrected state's fused weight equals the parameter returned by
the generic guarded step. Source: arXiv:2606.25971v2, Appendix A,
Algorithm 2, training correction. -/
theorem checkedProposal_parameter (σ L c : ℝ) (x g : MatrixSpace m n)
    (q : S × FullState m n) (hσ : σ ≤ 1 / 2) :
    fromMatrix (checkedProposal σ L c x g q).1.2.weight =
      safeguardedStep σ L x g (checkedProposal σ L c x g q).2 := by
  rw [checkedProposal_weight, safeguardedStep, checkedProposal_guard σ L c x g q hσ]

/-- The step/storage bridge has valid parameters,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (1 / 4 : ℝ) ≤ 1 / 2 := by norm_num

/-- Repair produces valid fixed-sphere storage for every nonzero current
weight and every proposed optimizer state. Source: training correction
to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem checkedProposal_norm (σ L c : ℝ) (x g : MatrixSpace m n)
    (q : S × FullState m n) (hσ : 0 < σ) (hL : 0 < L) (hc : 0 < c) (hx : x ≠ 0) :
    frobeniusNorm (fullDirection softplus (checkedProposal σ L c x g q).1.2) = c := by
  apply rebalanceStorage_norm c _ hc
  apply (fromMatrix_ne_zero_iff _).mp
  rw [fromMatrix_toMatrix]
  exact checkedDirection_step_ne_zero σ L x g _ hσ hL hx

/-- Positive parameters and nonzero matrix weights exist,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    fromMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ 0 := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  exact (fromMatrix_ne_zero_iff _).mpr (by norm_num)

/-- Every actual training parameter agrees with the MD state used for
the next gradient and proposal. Source: training correction to
arXiv:2606.25971v2, Appendix A, Algorithm 2, lines 1–2, 10. -/
theorem safeguardedFullRun_storage (σ L c : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hσ : σ ≤ 1 / 2) (t : ℕ) :
    fromMatrix (safeguardedFullRun σ L c f propose initialMemory initial t).1.2.weight =
      (safeguardedFullRun σ L c f propose initialMemory initial t).2 := by
  cases t with
  | zero => rfl
  | succ t =>
    exact checkedProposal_parameter σ L c _ _ _ hσ

/-- The all-time storage invariant has admissible parameters,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (1 / 4 : ℝ) ≤ 1 / 2 := by norm_num

/-- The nonsingular fallback guarantees a nonzero fused parameter at
every finite time. A limit of zero is permitted, including for a strongly
convex objective minimized at zero. Source: training correction to
arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedFullRun_ne_zero (σ L c : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2) (hL : 0 < L) (hW : initial.weight ≠ 0) (t : ℕ) :
    (safeguardedFullRun σ L c f propose initialMemory initial t).2 ≠ 0 := by
  induction t with
  | zero => exact (fromMatrix_ne_zero_iff _).mpr hW
  | succ t ih =>
    change safeguardedStep σ L _ _ (checkedProposal σ L c _ _ _).2 ≠ 0
    rw [safeguardedStep, checkedProposal_guard σ L c _ _ _ hσ']
    exact checkedDirection_step_ne_zero σ L _ _ _ hσ hL ih

/-- Nonzero initialization and positive guard parameters are compatible,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧
    (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight ≠ 0 := by norm_num [unitGainState]

/-- Positive-sphere initialization is necessarily nonzero, with no
additional nonzero premise. Source: arXiv:2606.25971v2, §3.1,
Appendix A, Algorithm 2, positive-gain training extension. -/
theorem fullState_ne_zero_of_norm (c : ℝ) (s : FullState m n)
    (hc : 0 < c) (hD : frobeniusNorm (fullDirection softplus s) = c) : s.weight ≠ 0 := by
  intro hW
  have hz : fullDirection softplus s = 0 := by simp [fullDirection, hW]
  rw [hz] at hD
  have hn : frobeniusNorm (0 : Matrix (Fin m) (Fin n) ℝ) = 0 := by
    rw [frobeniusNorm_eq_sqrt]
    simp
  rw [hn] at hD
  linarith

/-- A genuine matrix state has positive sphere radius,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : (0 : ℝ) < 1 ∧
    frobeniusNorm (fullDirection softplus (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 1 := by
  rw [unitGainState_direction, frobeniusNorm_eq_sqrt]
  norm_num

/-- Starting on the sphere, the actual MD representation remains on it
throughout corrected training, including rejected original updates.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedFullRun_direction_norm (σ L c : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2) (hL : 0 < L) (hc : 0 < c)
    (hD : frobeniusNorm (fullDirection softplus initial) = c) (t : ℕ) :
    frobeniusNorm (fullDirection softplus
      (safeguardedFullRun σ L c f propose initialMemory initial t).1.2) = c := by
  cases t with
  | zero => exact hD
  | succ t =>
    apply checkedProposal_norm σ L c _ _ _ hσ hL hc
    exact safeguardedFullRun_ne_zero σ L c f propose initialMemory initial hσ hσ' hL
      (fullState_ne_zero_of_norm c initial hc hD) t

/-- Positive learning parameters and valid initialization jointly exist,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    frobeniusNorm (fullDirection softplus (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 1 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  rw [unitGainState_direction, frobeniusNorm_eq_sqrt]
  norm_num

end Transformer.MagnitudeDirection
