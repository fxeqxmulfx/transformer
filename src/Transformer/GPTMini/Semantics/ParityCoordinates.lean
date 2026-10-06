import Transformer.GPTMini.Semantics.ParityInputs

/-!
# Exact coordinates and tied scores of the original parity decoder

Source: ParityConstruction's explicit ordinary parameter matrices, the
raw-state identities in ParityInputs, and GPTMini.unembed at f11b6e2.
The five-coordinate representation below is a vector, not a correctness
predicate or an oracle. Every scalar controls a distinct actual residual
coordinate. Its norm and all 68 tied scores are computed explicitly.
These identities will support uniform finite-weight readout margins.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- Five actual residual coordinates, including the two independent output channels.
Source: the concrete unit vectors of ParityConstruction and CompletionFFN. -/
noncomputable def decoderState (p a b q u : ℝ) : EucSpace 64 :=
  p • phaseUnit + a • controlUnit + b • bosUnit + q • parityUnit + u • eosUnit

/-- The exact ONE, BOS and completion coordinates of this ordinary vector.
Source: the independent embedding coordinates, with no semantic assumptions. -/
theorem decoderState_probes (p a b q u : ℝ) :
    inner (𝕜 := ℝ) controlUnit (decoderState p a b q u) = a ∧
      inner (𝕜 := ℝ) bosUnit (decoderState p a b q u) = b ∧
      phaseProbe (decoderState p a b q u) = p := by
  simp [decoderState, phaseProbe, innerSL_apply_apply, inner_add_right,
    controlUnit, phaseUnit, bosUnit, parityUnit, eosUnit,
    EuclideanSpace.inner_single_left]

/-- Its squared norm is the sum of five squares, with every cross term zero.
Source: the actual orthogonal residual basis, not a norm bound supplied by an encoder. -/
theorem decoderState_norm_sq (p a b q u : ℝ) :
    ‖decoderState p a b q u‖ ^ 2 = p ^ 2 + a ^ 2 + b ^ 2 + q ^ 2 + u ^ 2 := by
  rw [← real_inner_self_eq_norm_sq]
  simp only [decoderState, inner_add_left, inner_add_right,
    real_inner_smul_left, real_inner_smul_right]
  norm_num [controlUnit, phaseUnit, bosUnit, parityUnit, eosUnit,
    EuclideanSpace.inner_single_left]
  ring

/-- The actual tied embedding table has these scores on all vocabulary entries.
Source: GPTMini.unembed and the complete decoderEmbedding table, including zero-code distractors. -/
theorem decoderState_score (p a b q u : ℝ) (token : Fin countConfig.vocab_size) :
    inner (𝕜 := ℝ) (decoderState p a b q u) (decoderEmbedding token) =
      if token.val = 24 then -p + q else
        if token.val = 25 then -p - q else
          if token.val = 17 then u else
            if token.val = 1 then b else
              if token.val = 22 then a else
                if token.val = 18 then p else 0 := by
  unfold decoderEmbedding ratioEmbedding phaseEmbedding
  split_ifs <;>
    simp [decoderState, inner_add_left, inner_add_right, inner_sub_right,
      real_inner_smul_left, controlUnit, phaseUnit, bosUnit, parityUnit, eosUnit,
      EuclideanSpace.inner_single_left]
  all_goals omega

/-- The parity sign has unit magnitude at every integer count.
Source: CountSpline's EVEN/ODD sign definition, including the zero-count endpoint. -/
theorem countSign_sq (n : ℕ) : countSign n ^ 2 = 1 := by
  unfold countSign
  split_ifs <;> norm_num

/-- Positive prompt and negative supplied-answer phase coexist with the exact count channels.
Source: ParityInputs' universally proved raw first-FFN states. -/
theorem decoderInput_prompt_coordinates (eps : ℝ) (bits : List Bool) :
    decoderInput eps (fun j => (j.val : ℝ)) (finiteParityPrompt bits).get
        ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩ =
      decoderState 1 ((bits.count true : ℝ) / (bits.length + 2 : ℕ) * countScale eps)
        ((1 : ℝ) / (bits.length + 2 : ℕ) * countScale eps) 0 0 := by
  rw [decoderInput_prompt]
  simp only [decoderState, one_smul, zero_smul, add_zero]
  module

/-- The actual supplied-answer residual has a unit parity code and negative completion phase.
Source: the raw teacher-forcing continuation grammar, with no change to the encoder matrices. -/
theorem decoderInput_answer_coordinates (eps : ℝ) (bits : List Bool) :
    decoderInput eps (fun j => (j.val : ℝ))
        (finiteParityPrompt bits ++ [finiteParityLabel bits]).get
        ⟨bits.length + 2, by simp [finiteParityPrompt, finiteBitTokens]⟩ =
      decoderState (-1) ((bits.count true : ℝ) / (bits.length + 3 : ℕ) * countScale eps)
        ((1 : ℝ) / (bits.length + 3 : ℕ) * countScale eps) (countSign (bits.count true)) 0 := by
  rw [decoderInput_answer]
  simp only [decoderState, neg_one_smul, zero_smul, add_zero]
  module

/-- The normalized integer-count channels produce the exact FFN update at every bounded count.
Source: the actual completionFFN matrices, RMSNorm and homogeneous parity spline. -/
theorem completionFFN_count_state (parityScale eosScale eps p b q u : ℝ)
    (hb : 0 ≤ b) (n : ℕ) (hn : n ≤ 16) :
    let x := decoderState p (b * n) b q u
    let r := Real.sqrt 64 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)
    relu2FFN completionFFNIn (completionFFNOut parityScale eosScale) (rmsNormEps eps x) =
      (parityScale * (r * b) ^ 2 * countSign n) • parityUnit +
        (eosScale * r ^ 2 * relu2 (-p)) • eosUnit := by
  dsimp only
  rw [completionFFN_apply, completion_phase_rms]
  simp only [rmsNormEps, real_inner_smul_right, (decoderState_probes p (b * n) b q u).1,
    (decoderState_probes p (b * n) b q u).2.1, (decoderState_probes p (b * n) b q u).2.2]
  have hs : 0 ≤ Real.sqrt 64 /
      Real.sqrt (‖decoderState p (b * n) b q u‖ ^ 2 + 64 * eps) * b := by positivity
  norm_num only [Nat.cast_ofNat] at hs ⊢
  rw [← mul_assoc, paritySpline_exact _ hs n hn]
  congr 1 <;> congr 1 <;> ring

example : (0 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 16 := ⟨by norm_num, by decide⟩

/-- Adding both real FFN output channels retains the exact residual coordinates.
Source: the original block's second residual addition, with independently computed output scalars. -/
theorem decoderState_add_outputs (p a b q u v w : ℝ) :
    decoderState p a b q u + (v • parityUnit + w • eosUnit) =
      decoderState p a b (q + v) (u + w) := by
  unfold decoderState
  module

/-- Prenormalization multiplies every scalar coordinate by the same actual RMS coefficient.
Source: GPTMini.rmsNormEps, including the positive-epsilon denominator used by the reference. -/
theorem decoderState_rms (eps p a b q u : ℝ) :
    let r := Real.sqrt 64 / Real.sqrt (‖decoderState p a b q u‖ ^ 2 + 64 * eps)
    rmsNormEps eps (decoderState p a b q u) =
      decoderState (r * p) (r * a) (r * b) (r * q) (r * u) := by
  dsimp only
  simp only [rmsNormEps, decoderState, smul_add, smul_smul]
  norm_num only [Nat.cast_ofNat]

/-- Every bounded raw prompt reaches the exact signed label state in the full original model.
Source: the universal raw attention calculation, actual FFN and inactive second original block. -/
theorem decoder_prompt_hidden (parityScale eosScale eps : ℝ) (bits : List Bool)
    (hn : bits.count true ≤ 16) :
    let b := (1 : ℝ) / (bits.length + 2 : ℕ) * countScale eps
    let x := decoderState 1 (b * bits.count true) b 0 0
    let r := Real.sqrt 64 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)
    hidden countConfig (decoderParams parityScale eosScale) eps (fun j => (j.val : ℝ))
        (finiteParityPrompt bits).get 2
        ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩ =
      decoderState 1 (b * bits.count true) b
        (parityScale * (r * b) ^ 2 * countSign (bits.count true)) 0 := by
  dsimp only
  rw [decoder_full_state, decoderInput_prompt_coordinates]
  have ha : (bits.count true : ℝ) / (bits.length + 2 : ℕ) * countScale eps =
      ((1 : ℝ) / (bits.length + 2 : ℕ) * countScale eps) * bits.count true := by ring
  rw [ha, completionFFN_count_state parityScale eosScale eps 1 _ 0 0
    (by unfold countScale; positivity) _ hn]
  norm_num [relu2]
  unfold decoderState
  module

example : [true, false].count true ≤ 16 := by decide

/-- Every bounded supplied-answer continuation reaches both real parity and EOS channels.
Source: the same original matrices at the actual second supervised position, including its label residual. -/
theorem decoder_answer_hidden (parityScale eosScale eps : ℝ) (bits : List Bool)
    (hn : bits.count true ≤ 16) :
    let b := (1 : ℝ) / (bits.length + 3 : ℕ) * countScale eps
    let x := decoderState (-1) (b * bits.count true) b (countSign (bits.count true)) 0
    let r := Real.sqrt 64 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)
    hidden countConfig (decoderParams parityScale eosScale) eps (fun j => (j.val : ℝ))
        (finiteParityPrompt bits ++ [finiteParityLabel bits]).get 2
        ⟨bits.length + 2, by simp [finiteParityPrompt, finiteBitTokens]⟩ =
      decoderState (-1) (b * bits.count true) b
        ((1 + parityScale * (r * b) ^ 2) * countSign (bits.count true)) (eosScale * r ^ 2) := by
  dsimp only
  rw [decoder_full_state, decoderInput_answer_coordinates]
  have ha : (bits.count true : ℝ) / (bits.length + 3 : ℕ) * countScale eps =
      ((1 : ℝ) / (bits.length + 3 : ℕ) * countScale eps) * bits.count true := by ring
  rw [ha, completionFFN_count_state parityScale eosScale eps (-1) _ _ 0
    (by unfold countScale; positivity) _ hn]
  norm_num [relu2]
  unfold decoderState
  module

example : [false].count true ≤ 16 := by decide

/-!
The preceding formulas evaluate the genuine five-coordinate hidden state
and the actual finite-vocabulary table. A readout proof must still bound
the normalization multiplier from the raw context-length/count grammar;
none of these identities asserts that a trained model is already correct.
-/

end Transformer.GPTMini.Semantics
