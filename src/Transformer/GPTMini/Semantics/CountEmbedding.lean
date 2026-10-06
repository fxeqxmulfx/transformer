import Transformer.GPTMini.Semantics.Block
import Transformer.GPTMini.Semantics.Counting
import Transformer.GPTMini.TokenInterface.Control
import Mathlib.Analysis.InnerProductSpace.LinearMap

/-!
# A raw ONE feature in the original embedding and fused QKV

Source: the 64-wide, two-layer, four-head Basis GPTMini at cbafbe9 and
CausalMHA.forward/RMSNorm.forward at f11b6e2. This explicit parameter
assignment encodes ONE in one embedding coordinate and reads that
coordinate into the first head's value channel. Its query and key
channels are zero. The normalization multiplier is retained exactly.

The output projection is the identity and the FFN is zero. Thus the
feature can enter the actual residual stream; it is not a synthetic
attention operator or an oracle supplied with the parity answer.
Other token embeddings are zero. These parameters implement a counting
feature, not a complete parity model or its EOS decoder. The context cap
is parity's 19 tokens; real arithmetic uses the existing shared-epsilon
formalization, rather than certifying floating-point rounding.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface

/-- The actual small Basis parity dimensions; Source: experiments/basis and Bits.context at cbafbe9. -/
abbrev countConfig : Config where
  vocab_size := 68
  n_layers := 2
  n_heads := 4
  d_model := 64
  d_ff := 256
  max_seq_len := 19
  rope_theta := 10000
  divides := by decide
  head_even := by decide
  n_heads_pos := by decide
  n_layers_pos := by decide
  d_model_pos := by decide
  d_ff_pos := by decide
  vocab_pos := by decide
  max_seq_len_pos := by decide
  theta_pos := by norm_num

/-- Only the raw ONE ID activates the embedding coordinate; Source: vocabulary.py's ONE=22. -/
noncomputable def countEmbedding (token : Fin countConfig.vocab_size) : EucSpace 64 :=
  if token.val = 22 then controlUnit else 0

/-- Read the first model coordinate into the first value coordinate of the fused QKV matrix.
Source: the original bias-free projection; index 128 begins V after two blocks of 64. -/
noncomputable def countQKV : EucSpace 64 →L[ℝ] EucSpace 192 :=
  (innerSL ℝ controlUnit).smulRight (EuclideanSpace.single ⟨128, by decide⟩ 1)

/-- Original attention parameters with a genuine nonzero value projection and identity output projection.
Source: CausalMHA.forward at f11b6e2; no alternate head or optimizer is introduced. -/
noncomputable def countAttn : AttnParams countConfig where
  W_qkv := countQKV
  W_o := ContinuousLinearMap.id ℝ (EucSpace 64)
  log_alpha := fun _ => 0

/-- The unchanged original residual block with inactive FFN; Source: Block.forward at f11b6e2. -/
noncomputable def countBlock : BlockParams countConfig where
  attn := countAttn
  ffn := { W_in := 0, W_out := 0 }

/-- A concrete model parameter assignment, used to verify the first block's count feature.
Source: the new internal-feature control in the original small Basis architecture. -/
noncomputable def countParams : ModelParams countConfig where
  embedding := countEmbedding
  blocks := fun _ => countBlock

/-- RMSNorm's exact multiplier for the active unit embedding; Source: RMSNorm.forward at f11b6e2. -/
noncomputable def countScale (eps : ℝ) : ℝ :=
  Real.sqrt 64 / Real.sqrt (1 + 64 * eps)

/-- The active counting direction inside the first original head; Source: the reshape of coordinate 128. -/
noncomputable def countDirection : EucSpace countConfig.head_dim :=
  EuclideanSpace.single ⟨0, by decide⟩ 1

/-- The active embedding's prenorm projection retains the exact RMS multiplier.
Source: the real RMSNorm followed by the explicit fused linear map, not a desired-value premise. -/
theorem count_projected (eps : ℝ) (token : Fin countConfig.vocab_size) :
    countQKV (rmsNormEps eps (countEmbedding token)) =
      if token.val = 22 then countScale eps • EuclideanSpace.single ⟨128, by decide⟩ 1 else 0 := by
  by_cases htoken : token.val = 22
  · simp only [countEmbedding, htoken, ↓reduceIte, countQKV,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, rmsNormEps, controlUnit_norm,
      real_inner_smul_right, real_inner_self_eq_norm_sq]
    simp [countScale]
  · simp [countEmbedding, htoken, countQKV, rmsNormEps]

/-- Splitting the actual projected embedding gives zero Q/K and the raw ONE indicator in V.
Source: chunk/view in CausalMHA.forward; all 68 embedding entries are covered uniformly. -/
theorem count_qkv_features (eps : ℝ) (token : Fin countConfig.vocab_size) :
    headSlice countConfig (qkvSlice countConfig (qkvQ countConfig)
      (countAttn.W_qkv (rmsNormEps eps (countParams.embedding token)))) 0 = 0 ∧
    headSlice countConfig (qkvSlice countConfig (qkvK countConfig)
      (countAttn.W_qkv (rmsNormEps eps (countParams.embedding token)))) 0 = 0 ∧
    headSlice countConfig (qkvSlice countConfig (qkvV countConfig)
      (countAttn.W_qkv (rmsNormEps eps (countParams.embedding token)))) 0 =
        oneValue (countScale eps • countDirection) (token.val : ℤ) := by
  have hq : ∀ j : Fin 64, (qkvQ countConfig j).val = j.val := by decide
  have hk : ∀ j : Fin 64, (qkvK countConfig j).val = 64 + j.val := by decide
  have hv : ∀ j : Fin 64, (qkvV countConfig j).val = 128 + j.val := by decide
  have hh : ∀ j : Fin countConfig.head_dim,
      ((headSplit countConfig).symm (0, j)).val = j.val := by decide
  change headSlice countConfig (qkvSlice countConfig (qkvQ countConfig)
      (countQKV (rmsNormEps eps (countEmbedding token)))) 0 = 0 ∧
    headSlice countConfig (qkvSlice countConfig (qkvK countConfig)
      (countQKV (rmsNormEps eps (countEmbedding token)))) 0 = 0 ∧
    headSlice countConfig (qkvSlice countConfig (qkvV countConfig)
      (countQKV (rmsNormEps eps (countEmbedding token)))) 0 = _
  rw [count_projected]
  by_cases htoken : token.val = 22
  · simp only [ite_eq_left htoken]
    have hraw : (token.val : ℤ) = Transformer.Basis.oneBit := by
      change (token.val : ℤ) = 22
      exact_mod_cast htoken
    simp only [oneValue, ite_eq_left hraw]
    refine ⟨?_, ?_, ?_⟩ <;> ext j
    · have hj : j.val < 16 := j.isLt
      simp only [headSlice_apply, qkvSlice_apply, PiLp.smul_apply, PiLp.single_apply,
        PiLp.zero_apply]
      rw [ite_eq_right (by
        intro h
        have hi := congrArg Fin.val h
        rw [hq, hh] at hi
        change j.val = 128 at hi
        omega)]
      simp
    · have hj : j.val < 16 := j.isLt
      simp only [headSlice_apply, qkvSlice_apply, PiLp.smul_apply, PiLp.single_apply,
        PiLp.zero_apply]
      rw [ite_eq_right (by
        intro h
        have hi := congrArg Fin.val h
        rw [hk, hh] at hi
        change 64 + j.val = 128 at hi
        omega)]
      simp
    · simp only [headSlice_apply, qkvSlice_apply, PiLp.smul_apply, countDirection,
        PiLp.single_apply]
      have heq : qkvV countConfig ((headSplit countConfig).symm (0, j)) =
          (⟨128, by decide⟩ : Fin 192) ↔ j = ⟨0, by decide⟩ := by
        rw [Fin.ext_iff, hv, hh, Fin.ext_iff]
        change 128 + j.val = 128 ↔ j.val = 0
        omega
      simp only [heq]
  · have hraw : (token.val : ℤ) ≠ Transformer.Basis.oneBit := by
      intro h
      apply htoken
      change (token.val : ℤ) = 22 at h
      exact_mod_cast h
    simp only [ite_eq_right htoken, oneValue, ite_eq_right hraw]
    refine ⟨?_, ?_, ?_⟩ <;> ext j <;> simp

/-- The prenorm multiplier is strictly positive at the actual positive epsilon.
Source: RMSNorm.forward's denominator; this ensures that the count feature is nonzero. -/
theorem countScale_pos (eps : ℝ) (heps : 0 < eps) : 0 < countScale eps := by
  unfold countScale
  positivity

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- The feature direction has unit norm; Source: the first coordinate of the original 16-wide head. -/
theorem countDirection_norm : ‖countDirection‖ = 1 := by
  simp [countDirection, PiLp.norm_single]

end Transformer.GPTMini.Semantics
