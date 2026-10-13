import Transformer.GPTMini.Semantics.RecallReadout

/-!
# Complete original recall parameters and the actual integer adapter

Source: GPTMini.forward's embedding/two-block/final-RMS/tied loop at
f11b6e2 and the small Basis width/head/FFN/context recipe at cbafbe9.
The shared given matrices below are the verified original encoder
and retrieval blocks; neither receives a selected record or answer.
They fit width 64, four heads of dimension 16, FFN width 256 and
the existing two-layer model with its full 548-token tied table.

The complete hidden recurrence is connected to the already proved
actual states at every array position. Genuine final logits have
the strict whole-vocabulary margin. The checked List Int adapter
then appends the selected raw value, using its actual last row and
context check. A concrete raw overwrite control is included.

These correctness results assume only the raw alphabet/adjacency
and chronological-last-write predicates, not an encoded semantic
state or a prepared logit. Full Basis parser discharge remains.
Real arithmetic and the documented shared epsilon family are used;
floating-point equivalence and AdamW success do not follow here.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- Given complete ordinary two-layer parameters with the unchanged original tied raw embedding.
Source: the real model record, using fixed shared finite encoder/matching/output coefficients for the recipe. -/
noncomputable def recallModelParams (P : ℕ) (eps : ℝ) : ModelParams recallConfig where
  embedding := recallRawEmbedding
  blocks := fun b => if b.val = 0 then
    recallEncoderBlock P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P)
    else recallRetrievalBlock P eps

/-- The actual model loop's first layer is precisely the genuine raw encoder already analyzed.
Source: hidden's initial embedding lookup and first original residual block, with no alternate prefix encoder. -/
theorem recallModel_hidden_one (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    hidden recallConfig (recallModelParams P eps) eps (fun r => (r.val : ℝ)) tokens 1 i =
      recallMatchState P eps tokens i := by
  rw [hidden, dite_eq_left (by decide : 0 < recallConfig.n_layers)]
  rfl

/-- The actual model loop's complete second layer is exactly the genuine final residual, at every position.
Source: hidden's next original block application and the whole first-layer equality. -/
theorem recallModel_hidden_two (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    hidden recallConfig (recallModelParams P eps) eps (fun r => (r.val : ℝ)) tokens 2 i =
      recallFinalState P eps tokens i := by
  rw [hidden, dite_eq_left (by decide : 1 < recallConfig.n_layers)]
  change blockForward recallConfig (recallRetrievalBlock P eps) eps (fun r => (r.val : ℝ))
    (hidden recallConfig (recallModelParams P eps) eps (fun r => (r.val : ℝ)) tokens 1) i = _
  rw [funext (recallModel_hidden_one P eps tokens)]
  rfl

/-- The actual full forward call evaluates the real final normalization and the unchanged raw input embedding as its tied readout.
Source: GPTMini.forward and both genuine hidden-state recurrences, retaining all original operators. -/
theorem recallModel_forward (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (token : Fin recallConfig.vocab_size) :
    forward recallConfig (recallModelParams P eps) eps (fun r => (r.val : ℝ)) tokens i token =
      inner (𝕜 := ℝ) (rmsNormEps eps (recallFinalState P eps tokens i)) (recallRawEmbedding token) := by
  change inner (𝕜 := ℝ) (rmsNormEps eps
    (hidden recallConfig (recallModelParams P eps) eps (fun r => (r.val : ℝ)) tokens 2 i)) (recallRawEmbedding token) = _
  rw [recallModel_hidden_two]

/-- Every genuine final logit of the complete original model strictly loses to the actual selected raw value except that value itself.
Source: full model/operator coupling and the derived actual finite-softmax/RMS/tied margin, without a correct-logit premise. -/
theorem recallModel_logits (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ∀ token : Fin recallConfig.vocab_size, token ≠ recallValueId value →
      forward recallConfig (recallModelParams P eps) eps (fun r => (r.val : ℝ)) tokens query token <
        forward recallConfig (recallModelParams P eps) eps (fun r => (r.val : ℝ)) tokens query (recallValueId value) := by
  intro token hne
  rw [recallModel_forward, recallModel_forward]
  exact recallFinalState_rms_margin P eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest token hne

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

/-- The original raw overwrite control as a finite list, without any prepared key/value pairing.
Source: MQAR's literal integer IDs for two chronological writes and the final query. -/
def recallOverwriteTokens : List (Fin recallConfig.vocab_size) :=
  [⟨1, by decide⟩, recallKeyId 0, recallValueId 0, recallKeyId 0, recallValueId 1, recallKeyId 0]

/-- Indexing the genuine finite overwrite input agrees with the raw array witness used throughout the routing proof.
Source: the six literal token positions; no transformer state or answer occurs in this equality. -/
theorem recallOverwriteTokens_array : recallOverwriteTokens.get = recallScoreGapWitness := by
  funext i
  fin_cases i <;> rfl

/-- The checked actual integer model function appends the selected value from its true final-row greedy decoder.
Source: the complete original model's derived logits and the real encode/context/probability/greedy adapter. -/
theorem recallModel_tokens (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    (head : Fin recallConfig.vocab_size) (tail : List (Fin recallConfig.vocab_size))
    (hlen : tail.length + 1 ≤ 64) (first selected previous : Fin (head :: tail).length)
    (hlayout : RecallRawLayout P (head :: tail).get first) (key value : Fin 256)
    (hquery : (head :: tail).get ⟨tail.length, by simp⟩ = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : (head :: tail).get previous = recallKeyId key)
    (hvalue : (head :: tail).get selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ tail.length) (hlatest : RecallRawLatestWrite P (head :: tail).get selected key) :
    tokenFunction recallConfig (recallModelParams P eps) eps (decodeTokens (head :: tail)) =
      decodeTokens (head :: tail) ++ [((292 + value.val : ℕ) : ℤ)] := by
  have hg : bestToken recallConfig.vocab_pos (lastLogits recallConfig (recallModelParams P eps) eps head tail) =
      recallValueId value := by
    apply bestToken_of_strict
    exact recallModel_logits P eps heps hclip (by simpa only [List.length_cons] using hlen) (head :: tail).get
      first ⟨tail.length, by simp⟩ selected previous hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest
  have h := tokenFunction_decode_cons recallConfig (recallModelParams P eps) eps head tail hlen
  rw [hg] at h
  exact h

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallOverwriteTokens.get (0 : Fin 6) ∧ recallOverwriteTokens.get (5 : Fin 6) = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallOverwriteTokens.get (3 : Fin 6) = recallKeyId 0 ∧
    recallOverwriteTokens.get (4 : Fin 6) = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallOverwriteTokens.get (4 : Fin 6) 0 := by
  rw [recallOverwriteTokens_array]
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

/-- An ordinary call to the complete original GPTMini on the raw overwrite control preserves its input and emits value one.
Source: all original embedding/block/readout operators plus the checked public List Int interface, with finite shared parameters. -/
theorem recallModel_overwrite_control (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) :
    tokenFunction recallConfig (recallModelParams 2 eps) eps [1, 36, 292, 36, 293, 36] =
      [1, 36, 292, 36, 293, 36, 293] := by
  have h := recallModel_tokens 2 eps heps hclip ⟨1, by decide⟩
    [recallKeyId 0, recallValueId 0, recallKeyId 0, recallValueId 1, recallKeyId 0]
    (by decide) 0 4 3 (by exact recallOverwriteTokens_array ▸ recallScoreGapWitness_layout) 0 1 rfl (by decide)
    rfl rfl (by decide) (by decide) (by exact recallOverwriteTokens_array ▸ recallScoreGapWitness_latest)
  exact h

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- The given complete recall model obeys the requested append-one integer-list length contract on every input.
Source: the actual checked public adapter, including its specified empty/invalid/overlong fallback behavior. -/
theorem recallModel_tokens_length (P : ℕ) (eps : ℝ) (tokens : Tokens) :
    (tokenFunction recallConfig (recallModelParams P eps) eps tokens).length = tokens.length + 1 :=
  tokenFunction_length recallConfig (recallModelParams P eps) eps tokens

/-- The given recall model retains every raw integer in order through the public interface.
Source: the actual model tokenFunction wrapper, independently of whether its prefix is supervised. -/
theorem recallModel_tokens_prefix (P : ℕ) (eps : ℝ) (tokens : Tokens) :
    (tokenFunction recallConfig (recallModelParams P eps) eps tokens).take tokens.length = tokens :=
  tokenFunction_prefix recallConfig (recallModelParams P eps) eps tokens

end Transformer.GPTMini.Semantics
