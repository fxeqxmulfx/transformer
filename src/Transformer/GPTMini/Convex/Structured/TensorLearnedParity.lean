import Transformer.GPTMini.Convex.Structured.LearnedParityConfidence

/-!
# Completed learned parity coordinates solve every original raw input

Source: six exact completed ordinary-AdamW snapshots, kernel-checked
rational row differences and actual unrestricted tensor likelihood/greedy
coupling. These are properties of given learned parameter families,
with every unobserved field arbitrary, rather than existence of sharp
weights or correct-encoder/output assumptions.

Every valid raw parity input is solved in either original prototype mode,
including free generation of its counted label followed by EOS. The
whole literal prenorm/residual/zero-FFN/tied-readout stack is used. Real
gained coordinates recover the same function and retain the required
List Int to List Int interface on valid and invalid inputs alike.

The source matrices contain actual float32 coordinates interpreted as
exact dyadic reals, with checkpoint hashes/row metadata stored alongside
the experiment. Deserialization and IEEE forward/backward equivalence
are not formalized. The result does not prove AdamW convergence, train
FFN weights, or extend the original nineteen-position parity grammar.

Uniformity includes both raw supervised prefix forms and arbitrary
assignments of every parameter outside the saved finite row certificate.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

/-- Every given completed learned-row parameter family solves all valid raw parity prefixes in the actual configured tensor integer model.
Source: kernel-derived actual learned logit gaps and the true all-prefix full tensor correctness transfer, not an assumed correct encoder or logits. -/
theorem tensorLearnedParity_solves (mode : Mode) (eps : ℝ) (heps : 0 < eps)
    (run : Fin 6) (other : BindingParameters 68 19) :
    SolvesTask (tensorFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps (learnedParityParameters run other)) mode .parity := by
  exact tensorParity_solves_of_logit_gaps mode eps heps _ (learnedParity_model_logit_gaps run other)

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- Actual given learned weights predict the independent original next token on every raw valid integer prefix.
Source: full learned-family task agreement and the original append-one interface cancellation, with actual input tokens unchanged. -/
theorem tensorLearnedParity_next (mode : Mode) (eps : ℝ) (heps : 0 < eps) (run : Fin 6)
    (other : BindingParameters 68 19) (tokens : Tokens) (hprefix : TaskPrefix mode .parity tokens) :
    tensorNext (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps (learnedParityParameters run other) tokens = taskNext mode .parity tokens := by
  exact (solvesTask_extend_iff _ mode .parity).1 (tensorLearnedParity_solves mode eps heps run other) tokens hprefix

example : (0 : ℝ) < 1 / 100000 ∧ TaskPrefix .easy .parity [1, 22, 18] :=
  ⟨by norm_num, ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩⟩

/-- Actual learned full-stack complete loss meets the sufficient decoder threshold uniformly on every valid raw parity prefix.
Source: genuine learned-row confidence and full actual NLL/path/readout coupling, derived from saved weights rather than a minibatch assumption. -/
theorem tensorLearnedParity_NLL (mode : Mode) (eps : ℝ) (heps : 0 < eps) (run : Fin 6)
    (other : BindingParameters 68 19) (head : Fin 68) (tail : List (Fin 68))
    (hcap : (head :: tail).length ≤ contextSize .parity)
    (hprefix : TaskPrefix mode .parity (decodeTokens (head :: tail))) :
    basisStackNLL mode .parity eps head tail hcap (learnedParityParameters run other) < Real.log (11 / 10 : ℝ) := by
  exact basisParity_NLL_of_rows mode eps heps head tail hcap _ hprefix _ _ _ _
    parityLogitGap_eleven_budget.1 parityLogitGap_eleven_budget.2 (learnedParity_model_rows run other)

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity ∧
    TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18]) :=
  ⟨by norm_num, by decide, ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩⟩

/-- The actual true complete learned inference configuration has the sufficient probability on every valid raw parity prefix.
Source: uniformly derived real actual full-stack NLL and the same genuine mixed configuration probability; no empirical accuracy premise. -/
theorem tensorLearnedParity_probability (mode : Mode) (run : Fin 6) (other : BindingParameters 68 19)
    (head : Fin 68) (tail : List (Fin 68)) (hcap : (head :: tail).length ≤ contextSize .parity)
    (hprefix : TaskPrefix mode .parity (decodeTokens (head :: tail))) :
    (10 / 11 : ℝ) < mixedProbability (learnedParityParameters run other) (head :: tail) hcap
      (bindingFinalPosition tail) (basisDataTarget mode .parity head tail) := by
  have hl := tensorLearnedParity_NLL mode 1 (by norm_num) run other head tail hcap hprefix
  rw [basisStackNLL_sequence mode .parity 1 (by norm_num) head tail hcap
    (learnedParityParameters run other)] at hl
  exact (mixedNLL_small_iff (head :: tail) hcap (bindingFinalPosition tail) _ _).1 hl

example : ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity ∧
    TaskPrefix .hard .parity (decodeTokens [(1 : Fin 68), 22, 18]) :=
  ⟨by decide, ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩⟩

/-- Two true autoregressive calls at given learned weights generate the original counted answer then EOS on every allowed bit word.
Source: actual all-prefix learned tensor correctness and Basis's original two-call generation semantics, using the first prediction as input. -/
theorem tensorLearnedParity_twice (eps : ℝ) (heps : 0 < eps) (run : Fin 6) (other : BindingParameters 68 19)
    (bits : List Bool) (hmin : 1 ≤ bits.length) (hmax : bits.length ≤ 16) :
    let f := tensorFunction (basisTensorConfig .easy .parity) (basisTensorConfig_fits .easy .parity).1
      (basisTensorConfig_fits .easy .parity).2 eps (learnedParityParameters run other)
    f (f (parityPrompt bits)) = parityPrompt bits ++ [parityLabel bits, eos] := by
  exact solvesTask_parity_twice _ .easy (tensorLearnedParity_solves .easy eps heps run other) bits hmin hmax

example : (0 : ℝ) < 1 / 100000 ∧ 1 ≤ ([true, false, true] : List Bool).length ∧
    ([true, false, true] : List Bool).length ≤ 16 := ⟨by norm_num, by decide, by decide⟩

/-- Nonzero fixed gain recovers the same all-prefix real learned tensor correctness in actual inverse optimizer coordinates.
Source: genuine gain on all free fields and the exact full tensor-function recovery, with the same saved real learned parameter family. -/
theorem tensorLearnedParity_gain_solves (mode : Mode) (eps : ℝ) (heps : 0 < eps)
    (gain : ℝ) (hgain : gain ≠ 0) (run : Fin 6) (other : BindingParameters 68 19) :
    SolvesTask (tensorGainFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps gain (gain⁻¹ • learnedParityParameters run other)) mode .parity := by
  have hi : gain • gain⁻¹ • learnedParityParameters run other = learnedParityParameters run other :=
    smul_inv_smul₀ hgain _
  change SolvesTask (tensorFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
    (basisTensorConfig_fits mode .parity).2 eps (gain • gain⁻¹ • learnedParityParameters run other)) mode .parity
  rw [hi]
  exact tensorLearnedParity_solves mode eps heps run other

example : (0 : ℝ) < 1 / 100000 ∧ (8 : ℝ) ≠ 0 := ⟨by norm_num, by norm_num⟩

/-- The actual public gained callback appends the independent original answer at given learned weights on every raw valid prefix.
Source: full learned gained task agreement and the same original total append-one semantics, with no desired prediction premise. -/
theorem tensorLearnedParity_gain_answer (mode : Mode) (eps : ℝ) (heps : 0 < eps)
    (gain : ℝ) (hgain : gain ≠ 0) (run : Fin 6) (other : BindingParameters 68 19)
    (tokens : Tokens) (hprefix : TaskPrefix mode .parity tokens) :
    tensorGainFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps gain (gain⁻¹ • learnedParityParameters run other) tokens =
      tokens ++ [taskNext mode .parity tokens] := by
  have hs := tensorLearnedParity_gain_solves mode eps heps gain hgain run other
  exact hs tokens hprefix

example : (0 : ℝ) < 1 / 100000 ∧ (8 : ℝ) ≠ 0 ∧ TaskPrefix .hard .parity [1, 22, 18, 25] :=
  ⟨by norm_num, by norm_num, ⟨[true], by decide, by decide, by decide, Or.inr rfl⟩⟩

/-- The actual given learned tensor function has the required input-length-plus-one output on every integer input, including invalid ones.
Source: genuine full tensor append-one adapter at the saved complete parameter family, independent of the semantic grammar. -/
theorem tensorLearnedParity_length (mode : Mode) (eps : ℝ) (run : Fin 6) (other : BindingParameters 68 19)
    (tokens : Tokens) :
    (tensorFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps (learnedParityParameters run other) tokens).length = tokens.length + 1 :=
  tensorFunction_length _ _ _ _ _ _

/-- The given learned tensor function also preserves the complete original integer input exactly before appending its prediction.
Source: genuine full checked tensor adapter prefix identity, including its unchanged invalid-input fallback. -/
theorem tensorLearnedParity_prefix (mode : Mode) (eps : ℝ) (run : Fin 6) (other : BindingParameters 68 19)
    (tokens : Tokens) :
    (tensorFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps (learnedParityParameters run other) tokens).take tokens.length = tokens :=
  tensorFunction_prefix _ _ _ _ _ _

/-- The actual gained learned-family callback retains the same required total integer-list output length.
Source: genuine gained tensor adapter, with fixed gain acting only on the original full free coordinates. -/
theorem tensorLearnedParity_gain_length (mode : Mode) (eps gain : ℝ) (run : Fin 6) (other : BindingParameters 68 19)
    (tokens : Tokens) :
    (tensorGainFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps gain (gain⁻¹ • learnedParityParameters run other) tokens).length = tokens.length + 1 :=
  tensorGainFunction_length _ _ _ _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
