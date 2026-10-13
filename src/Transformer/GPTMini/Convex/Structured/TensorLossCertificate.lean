import Transformer.GPTMini.Convex.Structured.TensorGain

/-!
# Complete loss certifies actual semantic predictions

Source: plan.md §4's falling-loss/incorrect-depth control, MixedHeads'
true 11*p-10 whole-vocabulary margin, and TensorBasisStackTraining's
actual full-stack NLL/probability coupling. This is a sufficient property
of arbitrary actual weights, not only the finite capacity witness.

A correct complete configuration with NLL < log(11/10) has probability
above 10/11. The true mixed scores then have a strict margin against
every other vocabulary token. Independently correct raw data labels
transfer that margin to the real tensor stack's checked integer answer.
The same certificate applies to fixed-gain coordinates. The finite
state witness satisfies the loss hypothesis, so the premises are real.
A small minibatch mean is not a uniform bound over all raw prefixes;
optimizer convergence and floating-point certification remain separate.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

variable {V C : ℕ}

/-- Actual complete loss below the decoder threshold is exactly a quantitative true-joint confidence condition.
Source: MixedTraining.mixedNLL_eq and real exp/log inverse, with genuine positive probabilities at every unrestricted weight. -/
theorem mixedNLL_small_iff (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : MixedConfiguration tokens) (θ : BindingParameters V C) :
    mixedNLL tokens hcap query observed θ < Real.log (11 / 10 : ℝ) ↔
      (10 / 11 : ℝ) < mixedProbability θ tokens hcap query observed := by
  rw [mixedNLL_eq]
  have hp := mixedProbability_pos θ tokens hcap query observed
  have he : Real.exp (-Real.log (11 / 10 : ℝ)) = (10 / 11 : ℝ) := by
    rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 11 / 10)]
    norm_num
  constructor
  · intro h
    have hl : -Real.log (11 / 10 : ℝ) < Real.log (mixedProbability θ tokens hcap query observed) := by linarith
    have hx := Real.exp_lt_exp.mpr hl
    rw [he, Real.exp_log hp] at hx
    exact hx
  · intro h
    have hx : Real.exp (-Real.log (11 / 10 : ℝ)) < Real.exp (Real.log (mixedProbability θ tokens hcap query observed)) := by
      rw [he, Real.exp_log hp]
      exact h
    have hl := Real.exp_lt_exp.mp hx
    linarith

example : ([(0 : Fin 2)] : List (Fin 2)).length ≤ 4 := by decide

/-- The actual finite shared state witness uniformly satisfies the sufficient complete-loss threshold.
Source: MixedConfidence's derived 796 tail through context 128, including the genuine learned two-head mixture. -/
theorem mixedReference_NLL_small (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024) (start : Fin 6)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) (hT : tokens.length ≤ 128) :
    mixedNLL tokens hcap query (.inl (mixedReferenceTarget rule labels start tokens))
      (mixedReferenceParameters rule labels start referenceGain) < Real.log (11 / 10 : ℝ) := by
  apply (mixedNLL_small_iff tokens hcap query _ _).2
  have hp := mixedReference_probability rule labels start referenceGain tokens hcap query hT
  linarith [mixedReferenceGain_confidence]

example : ([(0 : Fin 2)] : List (Fin 2)).length ≤ 4 ∧ ([(0 : Fin 2)] : List (Fin 2)).length ≤ 128 := by decide

/-- Low actual complete loss at a correctly labeled configuration derives the genuine greedy answer at arbitrary free weights.
Source: exact true-joint confidence and MixedHeads' whole-vocabulary margin, without assumed correct logits or a supplied encoder. -/
theorem mixedGreedy_of_small_NLL (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : MixedConfiguration tokens) (target : Fin V)
    (hlabel : mixedChannels observed = outputDigit (vocabularyCode hsize target))
    (hloss : mixedNLL tokens hcap query observed θ < Real.log (11 / 10 : ℝ)) :
    mixedGreedy hV hsize θ tokens hcap query = target := by
  have hp := (mixedNLL_small_iff tokens hcap query observed θ).1 hloss
  apply bestToken_of_strict
  intro rival hrival
  have hne : vocabularyCode hsize target ≠ vocabularyCode hsize rival := by
    intro he
    exact hrival ((vocabularyCode_injective hsize he).symm)
  have hm := mixedScore_margin θ tokens hcap query observed (vocabularyCode hsize target)
    (vocabularyCode hsize rival) hlabel hne
  linarith

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ ([(0 : Fin 2)] : List (Fin 2)).length ≤ 4 ∧
    mixedChannels (.inl (mixedReferenceTarget (fun _ state => state) (fun _ => (1 : Fin 1024)) 0 [(0 : Fin 2)])) =
      outputDigit (vocabularyCode (by decide : (2 : ℕ) ≤ 1024) (1 : Fin 2)) ∧
    mixedNLL [(0 : Fin 2)] (by decide : ([(0 : Fin 2)] : List (Fin 2)).length ≤ 4) 0
      (.inl (mixedReferenceTarget (fun _ state => state) (fun _ => (1 : Fin 1024)) 0 [(0 : Fin 2)]))
      (mixedReferenceParameters (fun _ state => state) (fun _ => (1 : Fin 1024)) 0 referenceGain) < Real.log (11 / 10 : ℝ) := by
  refine ⟨by decide, by decide, by decide, ?_, ?_⟩
  · change outputDigit (1 : Fin 1024) = outputDigit (vocabularyCode (by decide) (1 : Fin 2))
    congr 1
  · exact mixedReference_NLL_small _ _ _ _ _ _ (by decide)

/-- Actual correctly labeled full-stack parity training at the finite witness satisfies the sufficient loss threshold on every capped raw list.
Source: real full-stack/raw-loss equality and the uniform true state witness; the loss property is not assumed from correct predictions. -/
theorem basisParity_NLL_small (mode : Mode) (eps : ℝ) (heps : 0 < eps) (head : Fin 68) (tail : List (Fin 68))
    (hcap : (head :: tail).length ≤ contextSize .parity) :
    basisStackNLL mode .parity eps head tail hcap (basisMixedParameters mode .parity) < Real.log (11 / 10 : ℝ) := by
  rw [basisStackNLL_sequence mode .parity eps heps head tail hcap (basisMixedParameters mode .parity)]
  unfold basisDataNLL basisDataTarget basisMixedParameters paritySharedTargets
  exact mixedReference_NLL_small _ _ _ _ hcap _ (by change (head :: tail).length ≤ 19 at hcap; omega)

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity :=
  ⟨by norm_num, by decide⟩

/-- At arbitrary actual weights, a valid raw prefix and sufficiently low real complete loss derive the independent Basis next answer.
Source: independently correct generated data channels, genuine full-stack/raw NLL and actual checked full-model greedy transfer. -/
theorem basisStack_next_of_small_NLL (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task))
    (hprefix : TaskPrefix mode task (decodeTokens (head :: tail)))
    (hloss : basisStackNLL mode task eps head tail hcap θ < Real.log (11 / 10 : ℝ)) :
    tensorNext (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1 (basisTensorConfig_fits mode task).2
      eps θ (decodeTokens (head :: tail)) = taskNext mode task (decodeTokens (head :: tail)) := by
  obtain ⟨target, htarget, hlabel⟩ := basisDataTarget_answer mode task head tail hprefix
  rw [basisStackNLL_sequence mode task eps heps] at hloss
  have hb := mixedGreedy_of_small_NLL (basisTensorConfig mode task).vocab_pos (basisTensorConfig_fits mode task).1
    θ (head :: tail) hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail) target hlabel hloss
  rw [tensorNext_of_encode (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1
    (basisTensorConfig_fits mode task).2 eps heps θ _ head tail (encode_decode _)
    (by simpa only [List.length_cons] using hcap), hb]
  exact htarget

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity ∧
    TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18]) ∧
    basisStackNLL .easy .parity (1 / 100000) (1 : Fin 68) ([(22 : Fin 68), 18] : List (Fin 68)) (by decide)
      (basisMixedParameters .easy .parity) < Real.log (11 / 10 : ℝ) := by
  exact ⟨by norm_num, by decide, ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩,
    basisParity_NLL_small _ _ (by norm_num) _ _ _⟩

/-- The same arbitrary-weight certificate applies to the actual public fixed-gain List Int to List Int function.
Source: genuine gain on every free field and unchanged raw complete labels, without any assertion that AdamW attains the bound. -/
theorem basisGain_function_of_small_NLL (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps) (gain : ℝ)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task))
    (hprefix : TaskPrefix mode task (decodeTokens (head :: tail)))
    (hloss : basisGainNLL mode task eps gain head tail hcap θ < Real.log (11 / 10 : ℝ)) :
    tensorGainFunction (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1 (basisTensorConfig_fits mode task).2
      eps gain θ (decodeTokens (head :: tail)) =
      decodeTokens (head :: tail) ++ [taskNext mode task (decodeTokens (head :: tail))] := by
  unfold tensorGainFunction tensorFunction extend
  rw [basisStack_next_of_small_NLL mode task eps heps head tail hcap (gain • θ) hprefix hloss]

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity ∧
    TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18]) ∧
    basisGainNLL .easy .parity (1 / 100000) 8 (1 : Fin 68) ([(22 : Fin 68), 18] : List (Fin 68)) (by decide)
      (basisGainWitness .easy .parity 8) < Real.log (11 / 10 : ℝ) := by
  refine ⟨by norm_num, by decide, ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩, ?_⟩
  unfold basisGainWitness
  rw [basisGainNLL_recover .easy .parity (1 / 100000) 8 (by norm_num : (8 : ℝ) ≠ 0)
    (1 : Fin 68) ([(22 : Fin 68), 18] : List (Fin 68)) (by decide) (basisMixedParameters .easy .parity)]
  exact basisParity_NLL_small _ _ (by norm_num) _ _ _

end
end Transformer.GPTMini.Convex.Structured
