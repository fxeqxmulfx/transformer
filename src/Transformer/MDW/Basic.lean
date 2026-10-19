module

public import Transformer.MDW.Defs

/-!
# Physical MDW storage and native Adam correspondence

Properties of the local AdamW extension to arXiv:2606.25971v2, Appendix A,
Algorithm 2. The physical correction changes neither gradient histories nor
the chosen physical weight during repair. Sphere claims require positive
radius and nonzero corrected storage; the zero-weight branch is explicit.
-/

@[expose] public section

noncomputable section

namespace Transformer.MDW

open MagnitudeDirection

variable {ι S : Type*} {m n : ℕ}

/-- Storage repair leaves the literal AdamW-style physical update unchanged.
Source: local decay extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem correctWeight_weight (radius eta decay : ℝ) (old : Matrix (Fin m) (Fin n) ℝ)
    (proposal : FullState m n) :
    (correctWeight radius eta decay old proposal).weight =
      proposal.weight - (eta * decay) • old := by
  dsimp only [correctWeight]
  split
  · rfl
  · exact (rebalanceStorage_preserves _ _).1

/-- A positive-radius, nonzero corrected weight retains the direction sphere.
Source: local physical-decay extension of MD Appendix A, Algorithm 2. -/
theorem correctWeight_norm (radius eta decay : ℝ) (old : Matrix (Fin m) (Fin n) ℝ)
    (proposal : FullState m n) (hr : 0 < radius)
    (hw : proposal.weight - (eta * decay) • old ≠ 0) :
    frobeniusNorm (fullDirection softplus (correctWeight radius eta decay old proposal)) =
      radius := by
  dsimp only [correctWeight]
  rw [ite_eq_right hw]
  exact rebalanceStorage_norm _ _ hr hw

example : 0 < (1 : ℝ) ∧
    (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight -
      ((1/4 : ℝ) * (1/2)) • (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ 0 := by
  constructor
  · norm_num
  · intro h
    have := congrFun (congrFun h 0) 0
    norm_num [unitGainState] at this

/-- The zero-result branch keeps the proposed finite raw gains rather than log zero.
Source: explicit zero-storage extension of arXiv:2606.25971v2, Appendix A. -/
theorem correctWeight_zero (radius eta decay : ℝ) (old : Matrix (Fin m) (Fin n) ℝ)
    (proposal : FullState m n) (hw : proposal.weight - (eta * decay) • old = 0) :
    correctWeight radius eta decay old proposal = { proposal with weight := 0 } := by
  simp [correctWeight, hw]

example : (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight -
    ((1 : ℝ) * 1) • (1 : Matrix (Fin 1) (Fin 1) ℝ) = 0 := by
  simp [unitGainState]

/-- Decay vanishes at zero coefficient; an already valid proposal is unchanged.
Source: local physical-decay extension of arXiv:2606.25971v2, Appendix A. -/
theorem correctWeight_zero_decay (radius eta : ℝ) (old : Matrix (Fin m) (Fin n) ℝ)
    (proposal : FullState m n) (hr : 0 < radius)
    (hn : frobeniusNorm (fullDirection softplus proposal) = radius) :
    correctWeight radius eta 0 old proposal = proposal := by
  simp only [correctWeight, mul_zero, zero_smul, sub_zero]
  split
  · rfl
  · exact rebalanceStorage_fixed _ _ hr hn

example : 0 < (1 : ℝ) ∧
    frobeniusNorm (fullDirection softplus
      (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 1 := by
  rw [unitGainState_direction, frobeniusNorm_eq_sqrt]
  norm_num

/-- The wrapper leaves all actual MD optimizer memories as returned by its callback.
Source: decoupled physical-decay extension of MD Appendix A, Algorithm 2. -/
theorem stepWith_memory (propose : FullProposal S m n) (radius eta decay : ℝ)
    (memory : S) (state : FullState m n) (gradient : Matrix (Fin m) (Fin n) ℝ) :
    (stepWith propose radius eta decay memory state gradient).1 =
      (propose memory state gradient).1 := rfl

/-- The complete wrapped physical weight uses the old weight for independent decay.
Source: local AdamW-style extension of arXiv:2606.25971v2, Appendix A. -/
theorem stepWith_weight (propose : FullProposal S m n) (radius eta decay : ℝ)
    (memory : S) (state : FullState m n) (gradient : Matrix (Fin m) (Fin n) ℝ) :
    (stepWith propose radius eta decay memory state gradient).2.weight =
      (propose memory state gradient).2.weight - (eta * decay) • state.weight := by
  exact correctWeight_weight _ _ _ _ _

/-- Every factor uses the same native Adam completed-update bias clock.
Source: `Grokking.AdamW.scalarNativeStep`, substituted into MD Appendix A. -/
theorem adamVectorCallback_native (epsilon betaOne betaTwo eta : ℝ)
    (state : AdamBuffers ι) (point gradient : ι → ℝ) (i : ι) :
    (adamVectorCallback epsilon betaOne betaTwo state point gradient eta).2 i =
      (Grokking.AdamW.scalarNativeStep betaOne betaTwo epsilon 0 eta
        ⟨point i, state.momentum i, state.second i, state.clock⟩ (gradient i)).parameter := by
  dsimp [adamVectorCallback, Grokking.AdamW.scalarNativeStep, Grokking.AdamW.nextBufferDirection]
  ring

/-- Physical decay gives exactly native AdamW on an unprojected auxiliary coordinate.
Source: `Grokking.AdamW.scalarNativeStep`; auxiliary extension of MD Appendix A. -/
theorem adam_auxiliary_native (epsilon betaOne betaTwo eta decay : ℝ)
    (state : AdamBuffers ι) (point gradient : ι → ℝ) (i : ι) :
    (adamVectorCallback epsilon betaOne betaTwo state point gradient eta).2 i -
        eta * decay * point i =
      (Grokking.AdamW.scalarNativeStep betaOne betaTwo epsilon decay eta
        ⟨point i, state.momentum i, state.second i, state.clock⟩ (gradient i)).parameter := by
  dsimp [adamVectorCallback, Grokking.AdamW.scalarNativeStep, Grokking.AdamW.nextBufferDirection]
  ring

end Transformer.MDW
