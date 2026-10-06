import Transformer.Basis.Encoding.Recall

/-!
# The entire Basis input domain fits the actual model interface

Source: the task vocabulary/context sizes in domain/basis.py at cbafbe9.
For every validated supervised prefix, successful nonempty encoding and
the final-row context bound are proved from the independent raw grammar.
They are not extra correctness assumptions supplied by the user or by
an unknown transformer-weight witness.
-/

namespace Transformer.Basis.Encoding

/-- Every supervised task prefix uses its actual task vocabulary, in both modes.
Source: the independently checked depth, raw MQAR and parity input parsers. -/
theorem taskPrefix_legal (mode : Mode) (task : Task) (tokens : Tokens)
    (h : TaskPrefix mode task tokens) : Legal (vocabularySize task) tokens := by
  cases task with
  | depth => exact depthPrefix_legal h
  | «recall» =>
      cases mode with
      | easy => exact recallPrefix_legal easyRecall h
      | hard => exact recallPrefix_legal hardRecall h
  | parity => exact parityPrefix_legal h

example : TaskPrefix .easy .depth [1, 9, 10] :=
  ⟨[some false, some true], by decide, by decide, rfl⟩

/-- Every supervised task prefix fits the actual raw-token context cap.
Source: DepthPrefix, RecallPrefix and ParityPrefix include the model cap explicitly. -/
theorem taskPrefix_length (mode : Mode) (task : Task) (tokens : Tokens)
    (h : TaskPrefix mode task tokens) : tokens.length ≤ contextSize task := by
  cases task with
  | depth =>
      obtain ⟨w, _, hlen, rfl⟩ := h
      simp only [List.length_cons, depthBody, List.length_map]
      exact hlen
  | «recall» => exact h.1
  | parity => obtain ⟨bits, _, _, hlen, _⟩ := h; exact hlen

example : TaskPrefix .hard .recall
    [1, 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299,
      36, 300, 37, 301, 38, 302, 39, 303, 40, 304, 41, 305, 42, 306, 43, 307, 36] :=
  hardRecall_prefix

/-- Every supervised prefix supplies a genuine final position for the model's readout.
Source: the nonempty depth/MQAR grammar and parity's BOS/bits/SEP prompt. -/
theorem taskPrefix_nonempty (mode : Mode) (task : Task) (tokens : Tokens)
    (h : TaskPrefix mode task tokens) : tokens ≠ [] := by
  cases task with
  | depth => exact depthPrefix_nonempty h
  | «recall» =>
      cases mode with
      | easy => exact recallPrefix_nonempty easyRecall h
      | hard => exact recallPrefix_nonempty hardRecall h
  | parity =>
      obtain ⟨bits, _, _, _, hform⟩ := h
      rcases hform with rfl | rfl
      · intro hempty; cases hempty
      · simp only [parityPrompt, List.cons_append]
        intro hempty
        cases hempty

example : TaskPrefix .easy .parity [1, 22, 18, 25] :=
  ⟨[true], by decide, by decide, by decide, Or.inr rfl⟩

/-- Every supervised prefix encodes successfully into its actual task vocabulary.
Source: raw per-token legality and encodeTokens's independently checked integer bounds. -/
theorem taskPrefix_encodes (mode : Mode) (task : Task) (tokens : Tokens)
    (h : TaskPrefix mode task tokens) : ∃ encoded,
      encodeTokens (vocabularySize task) tokens = some encoded :=
  encode_exists tokens (taskPrefix_legal mode task tokens h)

example : TaskPrefix .easy .recall
    [1, 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] :=
  easyRecall_prefix

/-- Vocabulary size and context size alone ensure an actual nonempty model input for every prefix.
Source: Basis's raw task grammar, not an assumption about task labels or trained parameters. -/
theorem model_input (mode : Mode) (task : Task) (tokens : Tokens) (V cap : ℕ)
    (hV : vocabularySize task ≤ V) (hcap : contextSize task ≤ cap)
    (h : TaskPrefix mode task tokens) :
    ∃ (head : Fin V) (tail : List (Fin V)),
      encodeTokens V tokens = some (head :: tail) ∧ tail.length + 1 ≤ cap := by
  have hlegal := legal_mono hV tokens (taskPrefix_legal mode task tokens h)
  obtain ⟨encoded, he⟩ := encode_exists tokens hlegal
  have hn := taskPrefix_nonempty mode task tokens h
  cases encoded with
  | nil =>
      have hnil : tokens = [] := by simpa [decodeTokens] using (decode_encode he).symm
      exact False.elim (hn hnil)
  | cons head tail =>
      refine ⟨head, tail, he, ?_⟩
      have hlen := encode_length he
      have ht := taskPrefix_length mode task tokens h
      simp only [List.length_cons] at hlen
      omega

example : vocabularySize .depth ≤ 36 ∧ contextSize .depth ≤ 128 ∧
    TaskPrefix .easy .depth [1, 9] :=
  ⟨by decide, by decide, [some false], by decide, by decide, rfl⟩

/-- The smaller GPTMini depth input shape covers all validated depth prefixes.
Source: experiments/basis's width-64 model context; width/depth do not affect token legality. -/
theorem depth_inputs (mode : Mode) (tokens : Tokens) (h : TaskPrefix mode .depth tokens) :
    ∃ (head : Fin 36) (tail : List (Fin 36)),
      encodeTokens 36 tokens = some (head :: tail) ∧ tail.length + 1 ≤ 128 :=
  model_input mode .depth tokens 36 128 (by decide) (by decide) h

example : TaskPrefix .hard .depth [1, 9, 10, 9, 10] :=
  ⟨[some false, some true, some false, some true], by decide, by decide, rfl⟩

/-- Both recall recipes fit the real 548-token vocabulary and 64-token context.
Source: domain/basis.py's 256-key/256-value MQAR recipes. -/
theorem recall_inputs (mode : Mode) (tokens : Tokens) (h : TaskPrefix mode .recall tokens) :
    ∃ (head : Fin 548) (tail : List (Fin 548)),
      encodeTokens 548 tokens = some (head :: tail) ∧ tail.length + 1 ≤ 64 :=
  model_input mode .recall tokens 548 64 (by decide) (by decide) h

example : TaskPrefix .easy .recall
    [1, 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] :=
  easyRecall_prefix

/-- All parity prompts and answer-before-EOS prefixes fit its actual 19-position model input.
Source: the 16-bit recipe plus BOS, SEP and one generated label at cbafbe9. -/
theorem parity_inputs (mode : Mode) (tokens : Tokens) (h : TaskPrefix mode .parity tokens) :
    ∃ (head : Fin 68) (tail : List (Fin 68)),
      encodeTokens 68 tokens = some (head :: tail) ∧ tail.length + 1 ≤ 19 :=
  model_input mode .parity tokens 68 19 (by decide) (by decide) h

example : TaskPrefix .hard .parity [1, 22, 21, 22, 18, 24] :=
  ⟨[true, false, true], by decide, by decide, by decide, Or.inr rfl⟩

/-- Task inputs retain every raw token and position through the finite model representation.
Source: the validated task grammar and the checked encoder's exact round-trip property. -/
theorem taskPrefix_roundtrip (mode : Mode) (task : Task) (tokens : Tokens)
    (h : TaskPrefix mode task tokens) :
    ∃ (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task))),
      encodeTokens (vocabularySize task) tokens = some (head :: tail) ∧
      decodeTokens (head :: tail) = tokens ∧
      (head :: tail).length = tokens.length ∧ tail.length + 1 ≤ contextSize task := by
  obtain ⟨head, tail, he, hlen⟩ := model_input mode task tokens
    (vocabularySize task) (contextSize task) (by omega) (by omega) h
  exact ⟨head, tail, he, decode_encode he, encode_length he, hlen⟩

example : TaskPrefix .easy .parity [1, 21, 18] :=
  ⟨[false], by decide, by decide, by decide, Or.inl rfl⟩

end Transformer.Basis.Encoding
