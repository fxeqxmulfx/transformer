import Transformer.GPTMini.Semantics.ParityState
import Transformer.Basis.Tasks

/-!
# A variable-length collision in the first count-and-phase block

Source: Parity.solve in synthetic/bits.py at cbafbe9 and the original
Block.forward at f11b6e2. The phase control retains only the normalized
ONE count at a final SEP. One ONE in four raw tokens and two ONEs in
eight raw tokens therefore have exactly the same first hidden state,
although both are legal Basis prompts with different parity answers.

This is a counterexample to decoding parity from this particular first
hidden state. It is not an impossibility theorem for all GPTMini weights
or for a later attention block, which can still inspect earlier states.
In particular, the externally known length used in count_block_recovers
does not provide a length feature to the actual FFN or tied readout.
The next construction must retain an independent denominator signal.
All equalities concern the actual prenorm, softmax, XSA and residual block.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- The entire final first-block state consists of the retained embedding and one count coordinate.
Source: the phase control's actual rank-one W_o and zero FFN, not merely a projected feature. -/
theorem phase_block_state (eps : ℝ) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22) :
    hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ =
      phaseParams.embedding last +
        (((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) /
          (body.length + 2 : ℕ) * countScale eps) • controlUnit := by
  have hlastToken : (head :: (body ++ [last])).get ⟨body.length + 1, by simp⟩ = last := by
    simp only [List.get_eq_getElem, List.getElem_cons_succ,
      List.getElem_append_right (le_refl body.length), Nat.sub_self, List.getElem_cons_zero]
  have hstate : hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ - phaseParams.embedding last =
      phaseOutput (headMerge countConfig
        (fun h => headAt countConfig (phaseParams.blocks ⟨0, by decide⟩).attn eps
          (fun j => (j.val : ℝ)) (embed countConfig phaseParams (head :: (body ++ [last])).get)
          h ⟨body.length + 1, by simp⟩)) := by
    rw [hidden, dite_eq_left (by decide : 0 < countConfig.n_layers)]
    simp only [hidden]
    unfold blockForward
    simp only [phaseParams, ffnSubLayer, relu2FFN, zero_apply, add_zero]
    rw [embed, hlastToken, attnSubLayer_eq_heads]
    change phaseParams.embedding last + phaseOutput _ - phaseParams.embedding last = _
    module
  have himage : hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ - phaseParams.embedding last =
      inner (𝕜 := ℝ) controlUnit
        (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ -
          phaseParams.embedding last) • controlUnit := by
    rw [hstate, phaseOutput_probe]
    rfl
  rw [phase_block_one_signal eps head last body hlast] at himage
  calc
    _ = phaseParams.embedding last +
        (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ -
          phaseParams.embedding last) := by module
    _ = _ := by rw [himage]

example : (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 := by decide

/-- A legal two-bit prompt with one ONE, whose answer is ODD.
Source: Basis's raw BOS, ONE, ZERO, SEP vocabulary at cbafbe9. -/
abbrev oddFractionTokens : List (Fin countConfig.vocab_size) :=
  [⟨1, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩, ⟨18, by decide⟩]

/-- A legal six-bit prompt with two ONEs, whose answer is EVEN.
Source: Basis parity allows every bit length from one through sixteen. -/
abbrev evenFractionTokens : List (Fin countConfig.vocab_size) :=
  [⟨1, by decide⟩, ⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
    ⟨21, by decide⟩, ⟨21, by decide⟩, ⟨21, by decide⟩, ⟨18, by decide⟩]

/-- Both collision inputs belong to the actual supervised grammar and have different oracle labels.
Source: ParityPrefix 19 and Parity.solve; no invalid or over-context input witnesses the collision. -/
theorem fraction_collision_legal :
    TaskPrefix .easy .parity (decodeTokens oddFractionTokens) ∧
      TaskPrefix .easy .parity (decodeTokens evenFractionTokens) ∧
      parityNext (decodeTokens oddFractionTokens) = 25 ∧
      parityNext (decodeTokens evenFractionTokens) = 24 := by
  refine ⟨?_, ?_, by decide, by decide⟩
  · refine ⟨[true, false], by decide, by decide, by decide, Or.inl ?_⟩
    rfl
  · refine ⟨[true, true, false, false, false, false], by decide, by decide,
      by decide, Or.inl ?_⟩
    rfl

/-- Different lengths erase an odd/even count distinction in this actual first-block state.
Source: one/four equals two/eight in the phase control; this compares full 64-dimensional states. -/
theorem phase_variable_length_collision (eps : ℝ) :
    hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        oddFractionTokens.get 1 ⟨3, by decide⟩ =
      hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        evenFractionTokens.get 1 ⟨7, by decide⟩ := by
  have hs := phase_block_state eps ⟨1, by decide⟩ ⟨18, by decide⟩
    [⟨22, by decide⟩, ⟨21, by decide⟩] (by decide)
  have hl := phase_block_state eps ⟨1, by decide⟩ ⟨18, by decide⟩
    [⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
      ⟨21, by decide⟩, ⟨21, by decide⟩, ⟨21, by decide⟩] (by decide)
  norm_num [decodeTokens, oneBit] at hs hl
  simp only [List.cons_append, List.nil_append, List.length_cons, List.length_nil] at hs hl
  exact hs.trans hl.symm

/-- No postprocessor of this one first-block state can answer both legal collision prompts correctly.
Source: the variable-length collision above; the statement allows an arbitrary, even nonlinear decoder. -/
theorem phase_first_state_decoder_obstruction (eps : ℝ) (decoder : EucSpace 64 → ℤ) :
    ¬ (decoder (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          oddFractionTokens.get 1 ⟨3, by decide⟩) = parityNext (decodeTokens oddFractionTokens) ∧
      decoder (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          evenFractionTokens.get 1 ⟨7, by decide⟩) = parityNext (decodeTokens evenFractionTokens)) := by
  intro hcorrect
  have heq := congrArg decoder (phase_variable_length_collision eps)
  have hlabels := hcorrect.1.symm.trans (heq.trans hcorrect.2)
  rw [fraction_collision_legal.2.2.1, fraction_collision_legal.2.2.2] at hlabels
  norm_num at hlabels

/-- The earlier count recovery succeeds because it also receives the external context length.
Source: count_block_recovers versus the actual state-only FFN interface at f11b6e2. -/
theorem normalized_count_needs_denominator :
    (1 / 4 : ℝ) = 2 / 8 ∧ (1 : ℕ) % 2 ≠ 2 % 2 := by
  constructor
  · norm_num
  · decide

/-- Applying the actual final normalization and tied readout directly to this state retains the collision.
Source: RMSNorm and tied unembedding at f11b6e2; this does not skip a layer in a claimed full-model solver. -/
theorem phase_first_state_readout_collision (eps : ℝ) (token : Fin countConfig.vocab_size) :
    inner (𝕜 := ℝ)
        (rmsNormEps eps (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          oddFractionTokens.get 1 ⟨3, by decide⟩)) (phaseParams.embedding token) =
      inner (𝕜 := ℝ)
        (rmsNormEps eps (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          evenFractionTokens.get 1 ⟨7, by decide⟩)) (phaseParams.embedding token) := by
  have hstates := phase_variable_length_collision eps
  exact congrArg (fun state : EucSpace 64 =>
    inner (𝕜 := ℝ) (rmsNormEps eps state) (phaseParams.embedding token)) hstates

/-!
The obstruction concerns a local representation, not gradient convergence.
Retaining phase repaired the prompt/answer collision at equal length, but
it did not retain the denominator across different lengths. Proving the
two repaired coordinates together is therefore necessary before assigning
a parity decoder to the actual FFN.
-/

end Transformer.GPTMini.Semantics
