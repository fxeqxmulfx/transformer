import Transformer.GPTMini.Convex.Structured.ParitySemanticRows

/-!
# Finite learned parity checks certify the entire actual tensor model

Source: plan.md §4, true mixed-joint margin and the actual full tensor-stack
complete-loss/inference coupling. All weights are arbitrary unrestricted
parameters. The needed finite rows imply correctness for every valid raw
input, rather than supplying correct output logits or an encoder premise.

The sufficient nineteen-position budget is
  deltaHead + deltaInitial + 19*deltaTransition + 5*deltaValue < 1/11.
Nine transition checks and fifteen output-channel checks derive the actual
correct complete path probability; initial and learned branch checks also
enter. The strict mixed margin then transfers through genuine RMSNorm,
residual layers, tied readout and checked raw integer decoding.

Both modes and both autoregressive parity calls are covered by the same
finite weight checks. The actual finite capacity witness satisfies them.
This is an arbitrary-weight semantic certificate, not a convergence claim
for AdamW or a proof that IEEE arithmetic preserves the real margins.
Output-only loss and freely trained FFNs remain outside this prototype.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

/-- The finite learned row conditions derive sufficiently low actual full-stack complete loss on every valid raw parity prefix.
Source: actual path factorization, true reachable output states, nineteen-position deficit bound and the exact log(11/10) decoder threshold. -/
theorem basisParity_NLL_of_rows (mode : Mode) (eps : ℝ) (heps : 0 < eps) (head : Fin 68)
    (tail : List (Fin 68)) (hcap : (head :: tail).length ≤ contextSize .parity)
    (θ : BindingParameters 68 19) (hprefix : TaskPrefix mode .parity (decodeTokens (head :: tail)))
    (deltaInitial deltaTransition deltaValue deltaHead : ℝ) (hdelta : 0 ≤ deltaTransition)
    (hbudget : deltaHead + deltaInitial + 19 * deltaTransition + 5 * deltaValue < (1 / 11 : ℝ))
    (hrows : ParityModelRows θ deltaInitial deltaTransition deltaValue deltaHead) :
    basisStackNLL mode .parity eps head tail hcap θ < Real.log (11 / 10 : ℝ) := by
  have hc : (head :: tail).length ≤ 19 := hcap
  have hpath := parityFinite_valid_path mode (head :: tail) hprefix
    (fun token state next => sharedTransitionRead token state next θ.1) deltaTransition hrows.2.1
  have hend := parityFinite_valid_endpoint mode (head :: tail) hprefix
  have hp := mixedState_probability_from_path θ paritySharedRule
    (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 (head :: tail) hc
    (bindingFinalPosition tail) deltaInitial deltaTransition deltaValue deltaHead hrows.1 hpath
    (fun h => hrows.2.2.1 _ hend h) hrows.2.2.2
  have hT : ((head :: tail).length : ℝ) ≤ 19 := by
    exact_mod_cast hc
  have ht := mul_le_mul_of_nonneg_right hT hdelta
  have hl : mixedNLL (C := 19) (head :: tail) hc (bindingFinalPosition tail)
      (.inl (mixedReferenceTarget paritySharedRule
        (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 (head :: tail))) θ <
      Real.log (11 / 10 : ℝ) := by
    apply (mixedNLL_small_iff (C := 19) (head :: tail) hc (bindingFinalPosition tail) _ θ).2
    linarith
  rw [basisStackNLL_sequence mode .parity eps heps head tail hcap θ]
  exact hl

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity ∧
    TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18]) ∧
    0 ≤ 6 * Real.exp (-referenceGain) ∧
    2 * Real.exp (-referenceGain) + 6 * Real.exp (-referenceGain) +
      19 * (6 * Real.exp (-referenceGain)) + 5 * (4 * Real.exp (-referenceGain)) < (1 / 11 : ℝ) ∧
    ParityModelRows (basisMixedParameters .easy .parity) (6 * Real.exp (-referenceGain))
      (6 * Real.exp (-referenceGain)) (4 * Real.exp (-referenceGain)) (2 * Real.exp (-referenceGain)) := by
  exact ⟨by norm_num, by decide, ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩,
    basisParity_rows_budget.1, basisParity_rows_budget.2, basisParity_model_rows .easy⟩

/-- Finite checks on arbitrary actual learned weights certify all original raw parity inputs for the literal tensor List Int to List Int model.
Source: derived small actual complete loss, whole-vocabulary strict margin and checked full-stack/raw answer transfer; no assumed correct outputs. -/
theorem tensorParity_solves_of_rows (mode : Mode) (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters 68 19)
    (deltaInitial deltaTransition deltaValue deltaHead : ℝ) (hdelta : 0 ≤ deltaTransition)
    (hbudget : deltaHead + deltaInitial + 19 * deltaTransition + 5 * deltaValue < (1 / 11 : ℝ))
    (hrows : ParityModelRows θ deltaInitial deltaTransition deltaValue deltaHead) :
    SolvesTask (tensorFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps θ) mode .parity := by
  apply (solvesTask_extend_iff _ mode .parity).2
  intro tokens hprefix
  obtain ⟨head, tail, he, hlen⟩ := Encoding.parity_inputs mode tokens hprefix
  have hd := decode_encode he
  have hc : (head :: tail).length ≤ contextSize .parity := by
    change (head :: tail).length ≤ 19
    simpa only [List.length_cons] using hlen
  have hf : TaskPrefix mode .parity (decodeTokens (head :: tail)) := by
    rw [hd]
    exact hprefix
  have hl := basisParity_NLL_of_rows mode eps heps head tail hc θ hf
    deltaInitial deltaTransition deltaValue deltaHead hdelta hbudget hrows
  have hn := basisStack_next_of_small_NLL mode .parity eps heps head tail hc θ hf hl
  change tensorNext (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
    (basisTensorConfig_fits mode .parity).2 eps θ (decodeTokens (head :: tail)) =
    taskNext mode .parity (decodeTokens (head :: tail)) at hn
  rw [hd] at hn
  exact hn

example : (0 : ℝ) < 1 / 100000 ∧ 0 ≤ 6 * Real.exp (-referenceGain) ∧
    2 * Real.exp (-referenceGain) + 6 * Real.exp (-referenceGain) +
      19 * (6 * Real.exp (-referenceGain)) + 5 * (4 * Real.exp (-referenceGain)) < (1 / 11 : ℝ) ∧
    ParityModelRows (basisMixedParameters .hard .parity) (6 * Real.exp (-referenceGain))
      (6 * Real.exp (-referenceGain)) (4 * Real.exp (-referenceGain)) (2 * Real.exp (-referenceGain)) :=
  ⟨by norm_num, basisParity_rows_budget.1, basisParity_rows_budget.2, basisParity_model_rows .hard⟩

/-- The same finite checks apply to all actual gained free fields in the public tensor adapter.
Source: fixed linear gain at the unrestricted real parameter assignment and the independently derived whole-model parity certificate. -/
theorem tensorGainParity_solves_of_rows (mode : Mode) (eps : ℝ) (heps : 0 < eps) (gain : ℝ)
    (θ : BindingParameters 68 19) (deltaInitial deltaTransition deltaValue deltaHead : ℝ)
    (hdelta : 0 ≤ deltaTransition)
    (hbudget : deltaHead + deltaInitial + 19 * deltaTransition + 5 * deltaValue < (1 / 11 : ℝ))
    (hrows : ParityModelRows (gain • θ) deltaInitial deltaTransition deltaValue deltaHead) :
    SolvesTask (tensorGainFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps gain θ) mode .parity := by
  unfold tensorGainFunction
  exact tensorParity_solves_of_rows mode eps heps (gain • θ)
    deltaInitial deltaTransition deltaValue deltaHead hdelta hbudget hrows

example : (0 : ℝ) < 1 / 100000 ∧ 0 ≤ 6 * Real.exp (-referenceGain) ∧
    2 * Real.exp (-referenceGain) + 6 * Real.exp (-referenceGain) +
      19 * (6 * Real.exp (-referenceGain)) + 5 * (4 * Real.exp (-referenceGain)) < (1 / 11 : ℝ) ∧
    ParityModelRows ((8 : ℝ) • ((8 : ℝ)⁻¹ • mixedReferenceParameters (C := 19) paritySharedRule
      (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 referenceGain)) (6 * Real.exp (-referenceGain))
      (6 * Real.exp (-referenceGain)) (4 * Real.exp (-referenceGain)) (2 * Real.exp (-referenceGain)) := by
  refine ⟨by norm_num, basisParity_rows_budget.1, basisParity_rows_budget.2, ?_⟩
  rw [smul_inv_smul₀ (by norm_num : (8 : ℝ) ≠ 0)]
  exact basisParity_model_rows .hard

/-- The same finite arbitrary-weight checks derive the complete freely generated parity label followed by EOS.
Source: actual all-prefix tensor correctness and Basis's original two-call generation theorem, using the first predicted answer as the second input. -/
theorem tensorParity_twice_of_rows (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters 68 19)
    (deltaInitial deltaTransition deltaValue deltaHead : ℝ) (hdelta : 0 ≤ deltaTransition)
    (hbudget : deltaHead + deltaInitial + 19 * deltaTransition + 5 * deltaValue < (1 / 11 : ℝ))
    (hrows : ParityModelRows θ deltaInitial deltaTransition deltaValue deltaHead)
    (bits : List Bool) (hmin : 1 ≤ bits.length) (hmax : bits.length ≤ 16) :
    let f := tensorFunction (basisTensorConfig .easy .parity) (basisTensorConfig_fits .easy .parity).1
      (basisTensorConfig_fits .easy .parity).2 eps θ
    f (f (parityPrompt bits)) = parityPrompt bits ++ [parityLabel bits, eos] := by
  exact solvesTask_parity_twice _ .easy
    (tensorParity_solves_of_rows .easy eps heps θ deltaInitial deltaTransition deltaValue deltaHead hdelta hbudget hrows)
    bits hmin hmax

example : (0 : ℝ) < 1 / 100000 ∧ 0 ≤ 6 * Real.exp (-referenceGain) ∧
    2 * Real.exp (-referenceGain) + 6 * Real.exp (-referenceGain) +
      19 * (6 * Real.exp (-referenceGain)) + 5 * (4 * Real.exp (-referenceGain)) < (1 / 11 : ℝ) ∧
    ParityModelRows (basisMixedParameters .easy .parity) (6 * Real.exp (-referenceGain))
      (6 * Real.exp (-referenceGain)) (4 * Real.exp (-referenceGain)) (2 * Real.exp (-referenceGain)) ∧
    1 ≤ ([true, false, true] : List Bool).length ∧ ([true, false, true] : List Bool).length ≤ 16 :=
  ⟨by norm_num, basisParity_rows_budget.1, basisParity_rows_budget.2,
    basisParity_model_rows .easy, by decide, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
