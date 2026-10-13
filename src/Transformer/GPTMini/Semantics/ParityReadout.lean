import Transformer.GPTMini.Semantics.ParityBounds

/-!
# Strict tied readout for the complete original parity parameter family

Source: ParityCoordinates' exact full hidden states and tied scores,
ParityBounds' uniform finite-weight inequalities, and GPTMini.forward at
f11b6e2. Final RMSNorm has a positive common coefficient, so every proved
strict residual-score advantage survives the actual final logits. No
correct output logits are premises of the task-specific theorems below.
Prompt and completion comparisons cover every vocabulary competitor.
The integer interface is connected separately in ParityCorrectness.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- The computed sign channel wins against every one of the 68 tied codes.
Source: the explicit embedding table, with the sufficient finite signal bound derived in ParityBounds. -/
theorem decoderState_parity_margin (n : ℕ) (a b m : ℝ)
    (ha : a ≤ 8) (hb : b ≤ 8) (hm : 9 < m) :
    ∀ token : Fin countConfig.vocab_size,
      token ≠ (if n % 2 = 1 then ⟨25, by decide⟩ else ⟨24, by decide⟩) →
      inner (𝕜 := ℝ) (decoderState 1 a b (m * countSign n) 0) (decoderEmbedding token) <
        inner (𝕜 := ℝ) (decoderState 1 a b (m * countSign n) 0)
          (decoderEmbedding (if n % 2 = 1 then ⟨25, by decide⟩ else ⟨24, by decide⟩)) := by
  intro token hne
  by_cases hodd : n % 2 = 1
  · have hval : token.val ≠ 25 := by
      intro h
      apply hne
      simp only [ite_eq_left hodd]
      exact Fin.ext h
    simp only [decoderState_score, countSign, ite_eq_left hodd]
    split_ifs <;> simp_all <;> linarith
  · have hval : token.val ≠ 24 := by
      intro h
      apply hne
      simp only [ite_eq_right hodd]
      exact Fin.ext h
    simp only [decoderState_score, countSign, ite_eq_right hodd]
    split_ifs <;> simp_all <;> linarith

example : (2 : ℝ) ≤ 8 ∧ (1 : ℝ) ≤ 8 ∧ (9 : ℝ) < 16 := by norm_num

/-- The computed completion channel wins against all input and parity codes after a supplied answer.
Source: the exact answer residual and simultaneous actual FFN outputs, with explicit scalar margins. -/
theorem decoderState_eos_margin (n : ℕ) (a b m u : ℝ)
    (ha : a ≤ 8) (hb : b ≤ 8) (hm : 0 ≤ m) (hu : 8 < u) (hlabel : 2 + m < u) :
    ∀ token : Fin countConfig.vocab_size, token ≠ ⟨17, by decide⟩ →
      inner (𝕜 := ℝ) (decoderState (-1) a b ((1 + m) * countSign n) u)
          (decoderEmbedding token) <
        inner (𝕜 := ℝ) (decoderState (-1) a b ((1 + m) * countSign n) u)
          (decoderEmbedding ⟨17, by decide⟩) := by
  intro token hne
  have hval : token.val ≠ 17 := by
    intro h
    exact hne (Fin.ext h)
  simp only [decoderState_score, countSign]
  split_ifs <;> simp_all <;> linarith

example : (2 : ℝ) ≤ 8 ∧ (1 : ℝ) ≤ 8 ∧ (0 : ℝ) ≤ 16 ∧
    (8 : ℝ) < 32 ∧ (2 + 16 : ℝ) < 32 := by norm_num

/-- Positive final RMSNorm preserves strict tied comparisons of any actual residual state.
Source: GPTMini's final normalization and tied unembedding, valid even when the final state has large norm. -/
theorem decoder_rms_margin (eps : ℝ) (heps : 0 < eps) (x : EucSpace 64)
    (answer competitor : Fin countConfig.vocab_size)
    (hmargin : inner (𝕜 := ℝ) x (decoderEmbedding competitor) <
      inner (𝕜 := ℝ) x (decoderEmbedding answer)) :
    inner (𝕜 := ℝ) (rmsNormEps eps x) (decoderEmbedding competitor) <
      inner (𝕜 := ℝ) (rmsNormEps eps x) (decoderEmbedding answer) := by
  rw [rmsNormEps, real_inner_smul_left, real_inner_smul_left]
  exact mul_lt_mul_of_pos_left hmargin (by positivity)

example : (0 : ℝ) < 1 / 100000 ∧
    inner (𝕜 := ℝ) phaseUnit (decoderEmbedding ⟨21, by decide⟩) <
      inner (𝕜 := ℝ) phaseUnit (decoderEmbedding ⟨18, by decide⟩) := by
  norm_num [decoderEmbedding, ratioEmbedding, phaseEmbedding, controlUnit, phaseUnit,
    EuclideanSpace.inner_single_left]

/-- On every legal-length raw prompt, the full original model strictly prefers its correct parity label.
Source: the actual hidden-state theorem and all derived raw-coordinate bounds; no correct-logit premise. -/
theorem decoder_prompt_logits (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (bits : List Bool) (hlen : bits.length ≤ 16) :
    ∀ token : Fin countConfig.vocab_size, token ≠ finiteParityLabel bits →
      forward countConfig (decoderParams 1024 131072) eps (fun j => (j.val : ℝ))
          (finiteParityPrompt bits).get
          ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩ token <
        forward countConfig (decoderParams 1024 131072) eps (fun j => (j.val : ℝ))
          (finiteParityPrompt bits).get
          ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩ (finiteParityLabel bits) := by
  intro token hne
  have hn : bits.count true ≤ 16 := List.count_le_length.trans hlen
  have hbounds := parity_raw_state_bounds eps heps hsmall 1 0 (by norm_num)
    (by norm_num) bits hlen 2 ⟨by decide, by decide⟩
  dsimp only at hbounds
  have ha : (bits.count true : ℝ) / (bits.length + 2 : ℕ) * countScale eps =
      ((1 : ℝ) / (bits.length + 2 : ℕ) * countScale eps) * bits.count true := by ring
  rw [ha] at hbounds
  change inner (𝕜 := ℝ) (rmsNormEps eps
    (hidden countConfig (decoderParams 1024 131072) eps _ _ 2 _)) (decoderEmbedding token) <
      inner (𝕜 := ℝ) (rmsNormEps eps
        (hidden countConfig (decoderParams 1024 131072) eps _ _ 2 _))
        (decoderEmbedding (finiteParityLabel bits))
  rw [decoder_prompt_hidden _ _ _ bits hn]
  apply decoder_rms_margin eps heps
  apply decoderState_parity_margin (bits.count true) _ _ _ hbounds.1 hbounds.2.2.1
    (parity_signal_margin _ _ hbounds.2.2.2 hbounds.2.1) token
  exact hne

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    [true, false].length ≤ 16 := by norm_num

/-- On every legal-length supplied-answer prefix, the actual full model strictly prefers EOS.
Source: the actual simultaneous parity/completion FFN, including its tied label residual and final RMSNorm. -/
theorem decoder_answer_logits (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (bits : List Bool) (hlen : bits.length ≤ 16) :
    ∀ token : Fin countConfig.vocab_size, token ≠ ⟨17, by decide⟩ →
      forward countConfig (decoderParams 1024 131072) eps (fun j => (j.val : ℝ))
          (finiteParityPrompt bits ++ [finiteParityLabel bits]).get
          ⟨bits.length + 2, by simp [finiteParityPrompt, finiteBitTokens]⟩ token <
        forward countConfig (decoderParams 1024 131072) eps (fun j => (j.val : ℝ))
          (finiteParityPrompt bits ++ [finiteParityLabel bits]).get
          ⟨bits.length + 2, by simp [finiteParityPrompt, finiteBitTokens]⟩ ⟨17, by decide⟩ := by
  intro token hne
  have hn : bits.count true ≤ 16 := List.count_le_length.trans hlen
  have hbounds := parity_raw_state_bounds eps heps hsmall (-1) (countSign (bits.count true))
    (by norm_num) (by rw [countSign_sq]) bits hlen 3 ⟨by decide, by decide⟩
  dsimp only at hbounds
  have ha : (bits.count true : ℝ) / (bits.length + 3 : ℕ) * countScale eps =
      ((1 : ℝ) / (bits.length + 3 : ℕ) * countScale eps) * bits.count true := by ring
  rw [ha] at hbounds
  change inner (𝕜 := ℝ) (rmsNormEps eps
    (hidden countConfig (decoderParams 1024 131072) eps _ _ 2 _)) (decoderEmbedding token) <
      inner (𝕜 := ℝ) (rmsNormEps eps
        (hidden countConfig (decoderParams 1024 131072) eps _ _ 2 _)) (decoderEmbedding ⟨17, by decide⟩)
  rw [decoder_answer_hidden _ _ _ bits hn]
  apply decoder_rms_margin eps heps
  have hb0 : 0 ≤ (1 : ℝ) / (bits.length + 3 : ℕ) * countScale eps := by
    linarith [hbounds.2.1]
  have hm := parity_eos_margin _ _ hbounds.2.2.2 ⟨hb0, hbounds.2.2.1⟩
  exact decoderState_eos_margin (bits.count true) _ _ _ _ hbounds.1 hbounds.2.2.1
    (by positivity) hm.2 hm.1 token hne

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    [false].length ≤ 16 := by norm_num

end Transformer.GPTMini.Semantics
