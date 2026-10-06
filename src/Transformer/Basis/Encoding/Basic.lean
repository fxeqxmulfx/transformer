import Transformer.Basis.Tasks

/-!
# Legal integer tokens and checked model encoding

Source: synthetic/vocabulary.py, Bits.prompt and AlternatingBlocks.targets
at cbafbe9. Valid depth and parity prefixes consist of genuine model token
IDs. The encoding theorem checks both integer bounds and reconstructs the
original input; validity is not a premise about the model's answers.
-/

namespace Transformer.Basis.Encoding

/-- Every token is a nonnegative integer strictly below the model vocabulary size.
Source: the embedding-index contract in infrastructure/nn/transformer.py at cbafbe9. -/
def Legal (V : ℕ) (tokens : Tokens) : Prop := ∀ t ∈ tokens, 0 ≤ t ∧ t < (V : ℤ)

/-- Legality of a cons checks the head and every token in its tail.
Source: the checked encodeTokens definition, with neither bound omitted. -/
theorem legal_cons (V : ℕ) (t : ℤ) (tokens : Tokens) :
    Legal V (t :: tokens) ↔ (0 ≤ t ∧ t < (V : ℤ)) ∧ Legal V tokens := by
  exact List.forall_mem_cons

/-- Legality is preserved exactly by splitting an appended input.
Source: the raw-token prompt/answer layout used in Basis. -/
theorem legal_append (V : ℕ) (left right : Tokens) :
    Legal V (left ++ right) ↔ Legal V left ∧ Legal V right := by
  exact List.forall_mem_append

/-- Every legal integer list successfully encodes into the finite model vocabulary.
Source: the actual checked encoder, proved recursively rather than assumed for task inputs. -/
theorem encode_exists {V : ℕ} (tokens : Tokens) (h : Legal V tokens) :
    ∃ encoded, encodeTokens V tokens = some encoded := by
  induction tokens with
  | nil => exact ⟨[], rfl⟩
  | cons t rest ih =>
      rcases (legal_cons V t rest).mp h with ⟨ht, hr⟩
      obtain ⟨encoded, he⟩ := ih hr
      refine ⟨⟨t.toNat, by omega⟩ :: encoded, ?_⟩
      simp only [encodeTokens, ht, true_and, ↓reduceDIte, he]

example : Legal 36 [1, 9, 10] := by norm_num [Legal]

/-- Successful finite encoding implies legality of every original integer ID.
Source: the encoder's round-trip theorem and the bounds in Fin V. -/
theorem legal_of_encode {V : ℕ} {tokens : Tokens} {encoded : List (Fin V)}
    (h : encodeTokens V tokens = some encoded) : Legal V tokens := by
  rw [← decode_encode h]
  clear h
  induction encoded with
  | nil => intro t ht; contradiction
  | cons t rest ih =>
      rw [decodeTokens, List.map_cons, legal_cons]
      refine ⟨?_, ih⟩
      constructor
      · omega
      · exact_mod_cast t.isLt

example : encodeTokens 36 [1, 9] = some [⟨1, by decide⟩, ⟨9, by decide⟩] := by decide

/-- The model vocabulary check is equivalent to an independent per-token integer property.
Source: both directions of the checked encoder, including its lower-bound rejection. -/
theorem encode_exists_iff (V : ℕ) (tokens : Tokens) :
    (∃ encoded, encodeTokens V tokens = some encoded) ↔ Legal V tokens := by
  constructor
  · rintro ⟨encoded, h⟩
    exact legal_of_encode h
  · exact encode_exists tokens

/-- A larger vocabulary preserves the validity of every existing input token.
Source: the strict-upper-bound vocabulary convention of nn.Embedding. -/
theorem legal_mono {V W : ℕ} (hVW : V ≤ W) (tokens : Tokens) (h : Legal V tokens) :
    Legal W tokens := by
  intro t ht
  have hb := h t ht
  constructor
  · exact hb.1
  · have hcast : (V : ℤ) ≤ (W : ℤ) := by exact_mod_cast hVW
    omega

example : (26 : ℕ) ≤ 36 ∧ Legal 26 [1, 22, 18] := by norm_num [Legal]

/-- Every neutral or active depth symbol serializes to an actual reserved ID below 36.
Source: the depth alphabet at cbafbe9, including all neutral positions. -/
theorem depthBody_legal (w : List (Option Bool)) : Legal 36 (depthBody w) := by
  induction w with
  | nil => intro t ht; contradiction
  | cons bit w ih =>
      cases bit with
      | none => exact (legal_cons 36 neutral _).mpr ⟨by decide, ih⟩
      | some b => cases b <;> exact (legal_cons 36 _ _).mpr ⟨by decide, ih⟩

/-- All depth supervised prefixes are legal inputs to the actual depth vocabulary.
Source: DepthPrefix's raw grammar, checked independently of its answer language. -/
theorem depthPrefix_legal {bound : ℕ} {tokens : Tokens} (h : DepthPrefix bound tokens) :
    Legal 36 tokens := by
  obtain ⟨w, _, _, rfl⟩ := h
  exact (legal_cons 36 bos _).mpr ⟨by decide, depthBody_legal w⟩

example : DepthPrefix 128 [1, 9, 11, 10] :=
  ⟨[some false, none, some true], by decide, by decide, rfl⟩

/-- Raw bit serialization uses IDs 21 and 22, both inside parity's actual 68-token vocabulary.
Source: Bits.prompt and Generator.vocab at cbafbe9. -/
theorem bitTokens_legal (bits : List Bool) : Legal 68 (bitTokens bits) := by
  induction bits with
  | nil => intro t ht; contradiction
  | cons bit bits ih =>
      cases bit <;> exact (legal_cons 68 _ _).mpr ⟨by decide, ih⟩

/-- The parity label is always a legal vocabulary ID.
Source: Parity.solve's EVEN/ODD alternatives, without scratchpad tokens. -/
theorem parityLabel_legal (bits : List Bool) : Legal 68 [parityLabel bits] := by
  unfold parityLabel
  split_ifs <;> norm_num [Legal, oddToken, evenToken]

/-- Every complete BOS/bits/SEP prompt is a legal model input.
Source: Bits.prompt's actual raw layout; delimiter IDs are checked explicitly. -/
theorem parityPrompt_legal (bits : List Bool) : Legal 68 (parityPrompt bits) := by
  apply (legal_cons 68 bos _).mpr
  refine ⟨by decide, ?_⟩
  exact (legal_append 68 _ _).mpr ⟨bitTokens_legal bits, by norm_num [Legal, sep]⟩

/-- Both parity supervision positions are legal, including the supplied correct answer before EOS.
Source: ParityPrefix and Parity.solve's two-step generation protocol. -/
theorem parityPrefix_legal {bound : ℕ} {tokens : Tokens} (h : ParityPrefix bound tokens) :
    Legal 68 tokens := by
  obtain ⟨bits, _, _, _, hform⟩ := h
  rcases hform with rfl | rfl
  · exact parityPrompt_legal bits
  · exact (legal_append 68 _ _).mpr ⟨parityPrompt_legal bits, parityLabel_legal bits⟩

example : ParityPrefix 19 [1, 22, 18, 25] :=
  ⟨[true], by decide, by decide, by decide, Or.inr rfl⟩

/-- A valid depth prefix has an actual final row for greedy decoding.
Source: DepthPrefix requires BOS and a nonempty body. -/
theorem depthPrefix_nonempty {bound : ℕ} {tokens : Tokens} (h : DepthPrefix bound tokens) :
    tokens ≠ [] := by
  obtain ⟨w, _, _, rfl⟩ := h
  intro hnil
  cases hnil

example : DepthPrefix 128 [1, 9] := ⟨[some false], by decide, by decide, rfl⟩

/-- No nonempty raw input is legal for an empty vocabulary.
Source: the strict interval 0 <= token < V in the actual checked model encoder. -/
theorem legal_zero_iff (tokens : Tokens) : Legal 0 tokens ↔ tokens = [] := by
  cases tokens with
  | nil =>
      constructor
      · intro _
        rfl
      · intro _ t ht
        contradiction
  | cons t rest =>
      constructor
      · intro h
        have ht := ((legal_cons 0 t rest).mp h).1
        omega
      · intro h
        cases h

end Transformer.Basis.Encoding
