import Transformer.GPTMini.Convex.Structured.MixedTraining
import Transformer.GPTMini.Convex.Structured.RecallGap

/-!
# Derived finite whole-model confidence for the learned mixture

Source: actual shared finite state witnesses at d72d3b3, complete raw
recall confidence at e5e8823 and the genuine mixed model at d436526.
Data/reference rules choose given ordinary finite raw weights only.
Inference still reads every learned row, all visible pairs and both
positive branch weights; it receives no reference state or parsed route.

The complete state configuration loses at most 796*exp(-gain) through
context 128, including learned head selection. Complete raw recall
loses at most 1073741826*exp(-gain) through cap 64. These are derived
true mixed probabilities, not assumed correct encoders or logits.
Finite logarithmic gains discharge both whole-distribution decoder
budgets. Strict greedy/task coupling, tensor/prenorm/residual/tied
realization and successful ordinary AdamW training remain separate.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.Semantics
open scoped Classical
noncomputable section

variable {V C : ℕ}

/-- One actual finite common mixed state assignment, with freely trainable relative fields set to zero in this witness only.
Source: referenceSharedParameters and its genuine learned initial/transition/emission/head fields; no rule enters inference. -/
def mixedReferenceParameters (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) : BindingParameters V C :=
  (referenceSharedParameters rule labels start gain, fun _ => 0)

/-- Complete state/path/channel training data for the finite witness, absent from actual learned inference.
Source: physical chronological referencePath and independently checked reference labels. -/
def mixedReferenceTarget (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (tokens : List (Fin V)) : SharedStateConfiguration tokens :=
  (start, (referencePath rule start tokens, outputDigit (labels (referenceRun rule start tokens))))

/-- The actual mixed state witness reads the same finite freely learned head row as its shared assignment.
Source: genuine head coordinate projection and the concrete shared global weights. -/
theorem mixedReference_head (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (head : Fin 2) :
    mixedHeadRead head (mixedReferenceParameters (C := C) rule labels start gain) = sharpRowLogits (0 : Fin 2) gain head := by
  change referenceSharedParameters (C := C) rule labels start gain
    (.inr (.inr (.inr (.inr (.inr head))))) = _
  exact referenceShared_head rule labels start gain head

/-- The actual mixed pointer witness reads its genuine finite learned branch-one preference.
Source: the complete shared recall parameter assignment and the true linear head-logit accessor. -/
theorem mixedRecall_head (P : ℕ) (gain : ℝ) (head : Fin 2) :
    mixedHeadRead head (recallBindingParameters P gain) = sharpRowLogits (1 : Fin 2) gain head := by
  change (recallBindingParameters P gain).1 (.inr (.inr (.inr (.inr (.inr head))))) = _
  exact recallBinding_head P gain head

/-- The genuine learned state probability at given raw mixed weights equals the finite stochastic model already bounded.
Source: actual initial/transition/value coordinate identities, without replacing the inference recurrence by a hard data scan. -/
theorem mixedReference_state (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (tokens : List (Fin V)) (z : SharedStateConfiguration tokens) :
    sharedStateProbability (mixedReferenceParameters (C := C) rule labels start gain).1 tokens z =
      markovJoint (sharpRowLogits start gain) (sharpReferenceTable rule gain) tokens
        (sharpReferenceEmission (fun state => outputDigit (labels state)) gain) z := by
  unfold sharedStateProbability mixedReferenceParameters
  simp_rw [referenceShared_initial, referenceShared_transition, referenceShared_emission]

/-- No true complete shared state-branch configuration exceeds one at arbitrary actual raw parameters.
Source: the same genuine normalized state/path/value model used by inference, not a supplied stochastic invariant. -/
theorem sharedStateProbability_le_one (θ : SharedParameters V C) (tokens : List (Fin V))
    (z : SharedStateConfiguration tokens) : sharedStateProbability θ tokens z ≤ 1 :=
  markovJoint_le_one _ _ _ _ _

/-- Every actual finite mixed state witness has derived full correct-configuration confidence through the entire Basis context cap.
Source: genuine 794 state/path/value tail plus the actual two-way learned head tail, using both real normalized probabilities. -/
theorem mixedReference_probability (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (hT : tokens.length ≤ 128) : 1 - 796 * Real.exp (-gain) ≤
      mixedProbability (mixedReferenceParameters rule labels start gain) tokens hcap query
        (.inl (mixedReferenceTarget rule labels start tokens)) := by
  let θ := mixedReferenceParameters (C := C) rule labels start gain
  let z := mixedReferenceTarget rule labels start tokens
  have hs : 1 - 794 * Real.exp (-gain) ≤ sharedStateProbability θ.1 tokens z := by
    rw [mixedReference_state]
    exact sharpReference_context_bound rule (fun state => outputDigit (labels state)) start tokens gain hT
  have hh : 1 - 2 * Real.exp (-gain) ≤ mixedHeadWeight θ 0 := by
    unfold mixedHeadWeight
    dsimp only [θ]
    simp_rw [mixedReference_head]
    simpa only [Fintype.card_fin, Nat.cast_ofNat] using stateRow_sharp (0 : Fin 2) gain
  have hhu : mixedHeadWeight θ 0 ≤ 1 := stateRow_le_one _ _
  have hsu := sharedStateProbability_le_one θ.1 tokens z
  have hc := mul_nonneg (sub_nonneg.mpr hhu) (sub_nonneg.mpr hsu)
  change 1 - 796 * Real.exp (-gain) ≤ mixedHeadWeight θ 0 * sharedStateProbability θ.1 tokens z
  nlinarith

example : ([0, 1] : List (Fin 2)).length ≤ 4 ∧ ([0, 1] : List (Fin 2)).length ≤ 128 := ⟨by decide, by decide⟩

/-- Full raw recall parsing derives a true complete mixed configuration and its entire learned-model confidence budget.
Source: unchanged all-pair raw confidence plus genuine learned branch selection, without a correct route/probability premise. -/
theorem mixedRecall_probability (P : ℕ) (rewrites : Bool) (gain : ℝ) (hgain : 0 ≤ gain)
    (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ query previous selected : Fin tokens.length,
      query.val + 1 = tokens.length ∧ ((tokens.get selected).val : ℤ) = answer ∧
      1 - 1073741826 * Real.exp (-gain) ≤ mixedProbability (recallBindingParameters P gain) tokens hcap query
        (.inr (recallBindingTarget tokens previous selected)) := by
  obtain ⟨query, previous, selected, hquery, ha, hp⟩ := recallBinding_raw_probability P rewrites gain hgain tokens hcap answer hanswer
  let : Nonempty (Fin tokens.length) := ⟨query⟩
  let θ := recallBindingParameters P gain
  let z := recallBindingTarget tokens previous selected
  have hh : 1 - 2 * Real.exp (-gain) ≤ mixedHeadWeight θ 1 := by
    unfold mixedHeadWeight
    dsimp only [θ]
    simp_rw [mixedRecall_head]
    simpa only [Fintype.card_fin, Nat.cast_ofNat] using stateRow_sharp (1 : Fin 2) gain
  have hhu : mixedHeadWeight θ 1 ≤ 1 := stateRow_le_one _ _
  have hpu : rawBindingProbability θ tokens hcap query z ≤ 1 := by
    rw [rawBinding_probability]
    exact probability_le_one _ _ _ _
  have hc := mul_nonneg (sub_nonneg.mpr hhu) (sub_nonneg.mpr hpu)
  refine ⟨query, previous, selected, hquery, ha, ?_⟩
  change 1 - 1073741826 * Real.exp (-gain) ≤ mixedHeadWeight θ 1 * rawBindingProbability θ tokens hcap query z
  nlinarith

example : (0 : ℝ) ≤ 1 ∧ recallOverwriteTokens.length ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by norm_num, by decide, by decide⟩

/-- The same ordinary finite state gain also covers genuine learned head selection in the complete mixed decoder budget.
Source: exact positive exp/log inverse at 100000 and the derived 796 whole-model tail constant. -/
theorem mixedReferenceGain_confidence : 796 * Real.exp (-referenceGain) < (1 / 11 : ℝ) := by
  unfold referenceGain
  rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 100000)]
  norm_num

/-- Both actual finite witness gains discharge the complete mixture's whole-distribution readout budgets simultaneously.
Source: the exact state and raw recall tail calculations; this is a real-arithmetic capacity witness, not AdamW convergence. -/
theorem mixedGains_confidence : 796 * Real.exp (-referenceGain) < (1 / 11 : ℝ) ∧
    1073741826 * Real.exp (-recallBindingGain) < (1 / 11 : ℝ) :=
  ⟨mixedReferenceGain_confidence, recallBindingGain_tail⟩

/-- Every true learned branch weight is at most one throughout the unrestricted mixed parameter space.
Source: the actual normalized row softmax, without a fixed task gate or supplied branch simplex. -/
theorem mixedHeadWeight_le_one (θ : BindingParameters V C) (head : Fin 2) : mixedHeadWeight θ head ≤ 1 :=
  stateRow_le_one _ _

/-- A genuine all-zero free mixed head uses both branches equally before training.
Source: actual shared head coordinate reads and the real two-way exponential normalizer. -/
theorem mixedHeadWeight_zero : mixedHeadWeight (0 : BindingParameters 2 4) 0 = 1 / 2 ∧
    mixedHeadWeight (0 : BindingParameters 2 4) 1 = 1 / 2 := by
  norm_num [mixedHeadWeight, stateRow, mixedHeadRead, Fin.sum_univ_two,
    LinearMap.comp_apply, LinearMap.proj_apply, LinearMap.fst_apply]

end
end Transformer.GPTMini.Convex.Structured
