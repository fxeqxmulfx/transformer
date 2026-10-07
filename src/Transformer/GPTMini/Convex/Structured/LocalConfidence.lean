import Transformer.GPTMini.Convex.Structured.TensorLossCertificate

/-!
# Finite row certificates at arbitrary learned parameters

Source: plan.md §4, MarkovTeacher.referencePath_probability's genuine
chronological product, MarkovConfidence's true channel contraction, and
MixedHeads' actual normalized learned branch. Local conditions below
read the learned initial, transition, value and branch probabilities;
they do not assume a correct encoder or correct output logits.

Four finite row deficit bounds imply complete configuration confidence
with error deltaHead + deltaInitial + T*deltaTransition + 5*deltaValue.
The reference rule and labels specify the conditions to check against
data semantics only; neither enters the learned forward computation.
The given finite sharp assignment satisfies all conditions, but arbitrary
learned weights may fail them. Tensor task correctness and a numerical
certificate for a learned checkpoint need separate transfers/checks.
Transition checks range over the actual finite token/state tables,
and output checks over the actual finite state/channel tables. Their
number does not grow with the number of possible input prefixes.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

/-- True factors bounded above by one lose at most the sum of their certified deficits.
Source: the actual product inequality (1-p)*(1-q) >= 0 used by MarkovTeacher, with every condition explicit. -/
theorem probability_mul_lower (p q a b : ℝ) (hp : p ≤ 1) (hq : q ≤ 1)
    (ha : 1 - a ≤ p) (hb : 1 - b ≤ q) : 1 - (a + b) ≤ p * q := by
  have hc := mul_nonneg (sub_nonneg.mpr hp) (sub_nonneg.mpr hq)
  nlinarith

example : (1 / 2 : ℝ) ≤ 1 ∧ (1 / 3 : ℝ) ≤ 1 ∧ 1 - (1 / 2 : ℝ) ≤ 1 / 2 ∧
    1 - (2 / 3 : ℝ) ≤ 1 / 3 := by norm_num

/-- Finite confidence of the actual freely learned value rows bounds their true complete output-channel assignment.
Source: exact channelProbability_rows and the actual normalized product's finite deficit inequality, not replacement values. -/
theorem channelProbability_of_rows {H D : Type*} [Fintype H] [Fintype D] [Nonempty D]
    (potential : H → D → ℝ) (labels : H → D) (delta : ℝ)
    (hrow : ∀ h, 1 - delta ≤ stateRow (potential h) (labels h)) :
    1 - (Fintype.card H : ℝ) * delta ≤ channelProbability potential labels := by
  rw [channelProbability_rows]
  have hp := probability_product_lower Finset.univ (fun h => stateRow (potential h) (labels h))
    (fun h _ => ⟨(stateRow_pos _ _).le, stateRow_le_one _ _⟩)
  have hd := Finset.sum_le_sum (fun h (_ : h ∈ Finset.univ) =>
    show 1 - stateRow (potential h) (labels h) ≤ delta from by linarith [hrow h])
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at hd
  linarith

example : ∀ h : Fin 5, 1 - (1 : ℝ) ≤ stateRow (fun _ : Fin 4 => (0 : ℝ)) (if h = 0 then 0 else 1) := by
  intro h
  simpa only [sub_self] using (stateRow_pos (fun _ : Fin 4 => (0 : ℝ)) (if h = 0 then 0 else 1)).le

variable {V C : ℕ}

/-- Actual complete path confidence and the needed endpoint rows bound the genuine learned mixed joint.
Source: the true initial/path/channel/branch factorization; task theorems must derive path confidence from their finite required transitions. -/
theorem mixedState_probability_from_path (θ : BindingParameters V C) (rule : Fin V → Fin 6 → Fin 6)
    (labels : Fin 6 → Fin 1024) (start : Fin 6) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (deltaInitial deltaTransition deltaValue deltaHead : ℝ)
    (hi : 1 - deltaInitial ≤ stateRow (fun s => sharedInitialRead s θ.1) start)
    (hp : 1 - (tokens.length : ℝ) * deltaTransition ≤ conditionalStatePath
      (fun token state next => sharedTransitionRead token state next θ.1) start tokens (referencePath rule start tokens))
    (hv : ∀ h, 1 - deltaValue ≤ stateRow (fun d =>
      sharedEmissionRead (referenceRun rule start tokens) h d θ.1) (outputDigit (labels (referenceRun rule start tokens)) h))
    (hh : 1 - deltaHead ≤ mixedHeadWeight θ 0) :
    1 - (deltaHead + deltaInitial + (tokens.length : ℝ) * deltaTransition + 5 * deltaValue) ≤
      mixedProbability θ tokens hcap query (.inl (mixedReferenceTarget rule labels start tokens)) := by
  let initial := fun s => sharedInitialRead s θ.1
  let transition := fun token state next => sharedTransitionRead token state next θ.1
  let emission := fun state h d => sharedEmissionRead state h d θ.1
  let path := referencePath rule start tokens
  let endpoint := referenceRun rule start tokens
  have hpu := conditionalStatePath_le_one transition start tokens path
  have hip := probability_mul_lower (stateRow initial start) (conditionalStatePath transition start tokens path)
    deltaInitial ((tokens.length : ℝ) * deltaTransition) (stateRow_le_one _ _) hpu hi hp
  have hipu : stateRow initial start * conditionalStatePath transition start tokens path ≤ 1 := by
    have h := mul_le_mul_of_nonneg_left hpu (stateRow_pos initial start).le
    rw [mul_one] at h
    exact h.trans (stateRow_le_one _ _)
  have he := channelProbability_of_rows (emission endpoint) (outputDigit (labels endpoint)) deltaValue hv
  norm_num only [Fintype.card_fin] at he
  have heu : channelProbability (emission endpoint) (outputDigit (labels endpoint)) ≤ 1 := by
    rw [channelProbability_rows]
    exact Finset.prod_le_one₀ (fun _ _ => (stateRow_pos _ _).le) (fun _ _ => stateRow_le_one _ _)
  have hj := probability_mul_lower (stateRow initial start * conditionalStatePath transition start tokens path)
    (channelProbability (emission endpoint) (outputDigit (labels endpoint)))
    (deltaInitial + (tokens.length : ℝ) * deltaTransition) (5 * deltaValue) hipu heu hip he
  have hs : 1 - (deltaInitial + (tokens.length : ℝ) * deltaTransition + 5 * deltaValue) ≤
      sharedStateProbability θ.1 tokens (mixedReferenceTarget rule labels start tokens) := by
    unfold sharedStateProbability mixedReferenceTarget markovJoint
    rw [referencePath_end]
    exact hj
  have hm := probability_mul_lower (mixedHeadWeight θ 0)
    (sharedStateProbability θ.1 tokens (mixedReferenceTarget rule labels start tokens)) deltaHead
    (deltaInitial + (tokens.length : ℝ) * deltaTransition + 5 * deltaValue)
    (mixedHeadWeight_le_one _ _) (sharedStateProbability_le_one _ _ _) hh hs
  change 1 - (deltaHead + deltaInitial + (tokens.length : ℝ) * deltaTransition + 5 * deltaValue) ≤ _
  change 1 - (deltaHead + deltaInitial + (tokens.length : ℝ) * deltaTransition + 5 * deltaValue) ≤
    mixedHeadWeight θ 0 * sharedStateProbability θ.1 tokens (mixedReferenceTarget rule labels start tokens)
  linarith

example : ([(0 : Fin 2)] : List (Fin 2)).length ≤ 4 ∧
    1 - (1 : ℝ) ≤ stateRow (fun s => sharedInitialRead s (0 : BindingParameters 2 4).1) 0 ∧
    1 - (1 : ℝ) * 1 ≤ conditionalStatePath (fun token state next => sharedTransitionRead token state next
      (0 : BindingParameters 2 4).1) 0 [(0 : Fin 2)] (referencePath (fun _ state => state) 0 [(0 : Fin 2)]) ∧
    (∀ h : Fin 5, 1 - (1 : ℝ) ≤ stateRow (fun d => sharedEmissionRead (0 : Fin 6) h d
      (0 : BindingParameters 2 4).1) (outputDigit 1 h)) ∧
    1 - (1 : ℝ) ≤ mixedHeadWeight (0 : BindingParameters 2 4) 0 := by
  refine ⟨by decide, ?_, ?_, ?_, ?_⟩
  · simpa only [sub_self] using (stateRow_pos _ _).le
  · simpa only [mul_one, sub_self] using (conditionalStatePath_pos _ _ _ _).le
  · intro h
    simpa only [sub_self] using (stateRow_pos _ _).le
  · simpa only [sub_self] using (mixedHeadWeight_pos _ _).le

/-- Local finite confidence checks derive the true whole complete state/path/value mass at arbitrary free weights.
Source: real learned row probabilities, referencePath_probability, exact channel contraction and the actual learned branch. -/
theorem mixedState_probability_of_rows (θ : BindingParameters V C) (rule : Fin V → Fin 6 → Fin 6)
    (labels : Fin 6 → Fin 1024) (start : Fin 6) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (deltaInitial deltaTransition deltaValue deltaHead : ℝ)
    (hi : 1 - deltaInitial ≤ stateRow (fun s => sharedInitialRead s θ.1) start)
    (ht : ∀ token state, 1 - deltaTransition ≤
      stateRow (fun next => sharedTransitionRead token state next θ.1) (rule token state))
    (hv : ∀ state h, 1 - deltaValue ≤
      stateRow (fun d => sharedEmissionRead state h d θ.1) (outputDigit (labels state) h))
    (hh : 1 - deltaHead ≤ mixedHeadWeight θ 0) :
    1 - (deltaHead + deltaInitial + (tokens.length : ℝ) * deltaTransition + 5 * deltaValue) ≤
      mixedProbability θ tokens hcap query (.inl (mixedReferenceTarget rule labels start tokens)) := by
  exact mixedState_probability_from_path θ rule labels start tokens hcap query _ _ _ _ hi
    (referencePath_probability _ rule start tokens deltaTransition ht) (hv _) hh

example : ([(0 : Fin 2)] : List (Fin 2)).length ≤ 4 ∧
    1 - (1 : ℝ) ≤ stateRow (fun s => sharedInitialRead s (0 : BindingParameters 2 4).1) 0 ∧
    (∀ token : Fin 2, ∀ state : Fin 6, 1 - (1 : ℝ) ≤
      stateRow (fun next => sharedTransitionRead token state next (0 : BindingParameters 2 4).1) state) ∧
    (∀ state : Fin 6, ∀ h : Fin 5, 1 - (1 : ℝ) ≤
      stateRow (fun d => sharedEmissionRead state h d (0 : BindingParameters 2 4).1) (outputDigit 1 h)) ∧
    1 - (1 : ℝ) ≤ mixedHeadWeight (0 : BindingParameters 2 4) 0 := by
  refine ⟨by decide, ?_, ?_, ?_, ?_⟩
  · simpa only [sub_self] using (stateRow_pos _ _).le
  · intro token state
    simpa only [sub_self] using (stateRow_pos _ _).le
  · intro state h
    simpa only [sub_self] using (stateRow_pos _ _).le
  · simpa only [sub_self] using (mixedHeadWeight_pos _ _).le

/-- Genuine finite shared parameters satisfy every local confidence condition with explicit row-size tails.
Source: actual shared coordinate reads at SharedReference and learned branch reads at MixedConfidence, evaluated at finite sharp logits. -/
theorem mixedReference_local_rows (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) :
    1 - 6 * Real.exp (-gain) ≤ stateRow (fun s => sharedInitialRead s
      (mixedReferenceParameters (C := C) rule labels start gain).1) start ∧
    (∀ token state, 1 - 6 * Real.exp (-gain) ≤ stateRow (fun next => sharedTransitionRead token state next
      (mixedReferenceParameters (C := C) rule labels start gain).1) (rule token state)) ∧
    (∀ state h, 1 - 4 * Real.exp (-gain) ≤ stateRow (fun d => sharedEmissionRead state h d
      (mixedReferenceParameters (C := C) rule labels start gain).1) (outputDigit (labels state) h)) ∧
    1 - 2 * Real.exp (-gain) ≤ mixedHeadWeight (mixedReferenceParameters (C := C) rule labels start gain) 0 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold mixedReferenceParameters
    simp_rw [referenceShared_initial]
    simpa only [Fintype.card_fin, Nat.cast_ofNat] using stateRow_sharp start gain
  · intro token state
    unfold mixedReferenceParameters
    simp_rw [referenceShared_transition]
    exact stateRow_sharp (rule token state) gain
  · intro state h
    unfold mixedReferenceParameters
    simp_rw [referenceShared_emission]
    exact stateRow_sharp (outputDigit (labels state) h) gain
  · unfold mixedHeadWeight
    simp_rw [mixedReference_head]
    simpa only [Fintype.card_fin, Nat.cast_ofNat] using stateRow_sharp (0 : Fin 2) gain

end
end Transformer.GPTMini.Convex.Structured
