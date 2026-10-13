import Transformer.GPTMini.Semantics.CompletionFFN

/-!
# A complete original-model parameter family for the parity construction

Source: GPTMini.forward/Block.forward at f11b6e2 and the small Basis
dimensions at cbafbe9. The new concrete embedding table adds independent
EVEN/ODD/EOS readout codes to the proved count/BOS/phase encoding. Its
first original block uses the proved attention and the simultaneous
69-unit parity/completion FFN. The second original block is inactive.

Changing the tied label embeddings does not change the actual first QKV
features: their ONE/BOS projections remain zero after their true RMSNorm.
The first attention update is therefore exactly the proved raw counting
calculation, while the new embedding codes remain in the residual. The
full two-layer hidden state is connected below to the actual first block.
Readout margins and universal integer task correctness are later results;
neither is assumed as a hypothesis here. There is no optimizer assertion.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface

/-- Tied output codes distinguish EVEN/ODD and EOS without altering raw ONE/BOS coordinates.
Source: vocabulary.py at cbafbe9, with an explicit new ordinary embedding assignment. -/
noncomputable def decoderEmbedding (token : Fin countConfig.vocab_size) : EucSpace 64 :=
  if token.val = 24 then -phaseUnit + parityUnit else
    if token.val = 25 then -phaseUnit - parityUnit else
      if token.val = 17 then eosUnit else ratioEmbedding token

/-- The original first block, with the proved raw counting attention and real completion FFN.
Source: BlockParams at f11b6e2; both output scales are genuine ordinary FFN weights. -/
noncomputable def decoderBlock (parityScale eosScale : ℝ) : BlockParams countConfig where
  attn := (ratioParams.blocks ⟨0, by decide⟩).attn
  ffn := completionFFN parityScale eosScale

/-- An inactive original residual block, retaining its input exactly.
Source: the original bias-free block, with zero output matrices. -/
noncomputable def decoderIdle : BlockParams countConfig where
  attn := { W_qkv := 0, W_o := 0, log_alpha := fun _ => 0 }
  ffn := { W_in := 0, W_out := 0 }

/-- Given two-layer model parameters, not merely an existential weight witness.
Source: the unchanged width 64 / two layers / four heads / FFN width 256 / vocabulary 68 architecture. -/
noncomputable def decoderParams (parityScale eosScale : ℝ) : ModelParams countConfig where
  embedding := decoderEmbedding
  blocks := fun b => if b.val = 0 then decoderBlock parityScale eosScale else decoderIdle

/-- The actual projected embeddings are unchanged even when the tied label codes have different norms.
Source: the orthogonal parity/EOS coordinates and the true prenorm QKV, checked on all vocabulary entries. -/
theorem decoder_projected_eq_ratio (eps : ℝ) (token : Fin countConfig.vocab_size) :
    ratioQKV (rmsNormEps eps (decoderEmbedding token)) =
      ratioQKV (rmsNormEps eps (ratioEmbedding token)) := by
  unfold decoderEmbedding
  split_ifs with heven hodd heos
  · simp [ratioQKV, countQKV, bosQKV, ratioEmbedding, phaseEmbedding, heven,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, rmsNormEps,
      controlUnit, bosUnit, phaseUnit, parityUnit, EuclideanSpace.inner_single_left]
  · simp [ratioQKV, countQKV, bosQKV, ratioEmbedding, phaseEmbedding, hodd,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, rmsNormEps,
      controlUnit, bosUnit, phaseUnit, parityUnit, EuclideanSpace.inner_single_left]
  · simp [ratioQKV, countQKV, bosQKV, ratioEmbedding, phaseEmbedding, heos,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, rmsNormEps,
      controlUnit, bosUnit, eosUnit, EuclideanSpace.inner_single_left]
  · rfl

/-- Every actual first head still computes the same raw feature despite the new tied embedding codes.
Source: equality of true projected inputs, followed by the unchanged RoPE/QKNorm/softmax/XSA operators. -/
theorem decoder_first_head (parityScale eosScale eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (h : Fin countConfig.n_heads) (i : Fin T) :
    headAt countConfig (decoderBlock parityScale eosScale).attn eps positions
        (embed countConfig (decoderParams parityScale eosScale) tokens) h i =
      headAt countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps positions
        (embed countConfig ratioParams tokens) h i := by
  dsimp only [headAt, embed, decoderBlock, decoderParams, ratioParams]
  have hp := funext fun j => decoder_projected_eq_ratio eps (tokens j)
  exact congrArg (fun projected : Fin T → EucSpace 192 =>
    attentionHead countConfig 0 eps
      (fun j => headSlice countConfig (qkvSlice countConfig (qkvQ countConfig) (projected j)) h)
      (fun j => headSlice countConfig (qkvSlice countConfig (qkvK countConfig) (projected j)) h)
      (fun j => headSlice countConfig (qkvSlice countConfig (qkvV countConfig) (projected j)) h)
      positions i) hp

/-- The actual first attention update is precisely the earlier simultaneous ONE/BOS computation.
Source: equality of every head and the unchanged nonzero rank-two W_o. -/
theorem decoder_first_attention (parityScale eosScale eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) :
    attnSubLayer countConfig (decoderBlock parityScale eosScale).attn eps positions
        (embed countConfig (decoderParams parityScale eosScale) tokens) i =
      attnSubLayer countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps positions
        (embed countConfig ratioParams tokens) i := by
  rw [attnSubLayer_eq_heads, attnSubLayer_eq_heads]
  have he := funext fun h => decoder_first_head parityScale eosScale eps positions tokens h i
  exact congrArg (fun heads => ratioOutput (headMerge countConfig heads)) he

/-- The reference feature control's first state is its ordinary embedding plus its actual attention update.
Source: ratioParams' zero FFN; this identity holds at every finite position, not only selected parity inputs. -/
theorem ratio_first_state (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) :
    hidden countConfig ratioParams eps positions tokens 1 i =
      ratioEmbedding (tokens i) +
        attnSubLayer countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps positions
          (embed countConfig ratioParams tokens) i := by
  rw [hidden, dite_eq_left (by decide : 0 < countConfig.n_layers)]
  simp only [hidden]
  unfold blockForward
  simp [ratioParams, ffnSubLayer, relu2FFN, embed]

/-- The complete active block retains its actual residual input and applies the real decoder FFN to its prenorm.
Source: original blockForward, with the proved attention update and both residual additions retained. -/
theorem decoder_hidden_one (parityScale eosScale eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) :
    hidden countConfig (decoderParams parityScale eosScale) eps positions tokens 1 i =
      let x := decoderEmbedding (tokens i) +
        attnSubLayer countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps positions
          (embed countConfig ratioParams tokens) i
      x + relu2FFN completionFFNIn (completionFFNOut parityScale eosScale) (rmsNormEps eps x) := by
  rw [hidden, dite_eq_left (by decide : 0 < countConfig.n_layers)]
  simp only [hidden]
  change blockForward countConfig (decoderBlock parityScale eosScale) eps positions
    (fun j => decoderEmbedding (tokens j)) i = _
  dsimp only [blockForward, ffnSubLayer, decoderBlock, completionFFN, embed,
    decoderParams, ratioParams]
  have ha := decoder_first_attention parityScale eosScale eps positions tokens i
  dsimp only [decoderBlock, embed, decoderParams, ratioParams] at ha
  unfold embed at ha ⊢
  rw [ha]

/-- The second original layer passes its input through exactly.
Source: decoderIdle's two zero output matrices; no attention approximation or oracle state is used. -/
theorem decoderIdle_forward (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace countConfig.d_model) (i : Fin T) :
    blockForward countConfig decoderIdle eps positions x i = x i := by
  simp [decoderIdle, blockForward, attnSubLayer, ffnSubLayer, relu2FFN]

/-- The verified active first-block state is also the complete two-layer hidden state read by the model.
Source: GPTMini.hidden's actual configured loop and the exact inactive second residual block. -/
theorem decoder_hidden_two (parityScale eosScale eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) :
    hidden countConfig (decoderParams parityScale eosScale) eps positions tokens 2 i =
      hidden countConfig (decoderParams parityScale eosScale) eps positions tokens 1 i := by
  rw [hidden, dite_eq_left (by decide : 1 < countConfig.n_layers)]
  change blockForward countConfig decoderIdle eps positions
    (hidden countConfig (decoderParams parityScale eosScale) eps positions tokens 1) i = _
  exact decoderIdle_forward eps positions _ i

/-- The supplied correct label's tied embedding retains the independently proved parity sign.
Source: EVEN=24, ODD=25 in the raw continuation grammar, with one fixed embedding per token. -/
theorem decoder_label_embedding (bits : List Bool) :
    decoderEmbedding (finiteParityLabel bits) =
      -phaseUnit + countSign (bits.count true) • parityUnit := by
  unfold finiteParityLabel countSign
  split_ifs
  · simp [decoderEmbedding]
    module
  · simp [decoderEmbedding]

end Transformer.GPTMini.Semantics
