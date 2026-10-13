import Transformer.GPTMini.Semantics.ParityCollision

/-!
# Two raw indicator channels in the original fused QKV

Source: the small 64-wide Basis GPTMini at cbafbe9 and the bias-free
embedding/QKV/reshape operators at f11b6e2. ONE and BOS activate distinct
unit embedding coordinates. The first head's two value coordinates read
these indicators after the real RMSNorm; Q and K remain zero. SEP and
answer tokens retain an independent phase coordinate and have zero values.

The new BOS channel is a denominator marker, not an externally supplied
length or a parity label. All vocabulary entries are checked, and both
channels coexist in a single original head. The output projection moves
the BOS signal back to its own residual coordinate, protecting phase.
The shared-epsilon real-arithmetic deviation of GPTMini is unchanged.
These weights implement features; a complete parity FFN/readout remains
an independent proof obligation.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- BOS occupies a third residual coordinate, independently of ONE and completion phase.
Source: the new denominator marker in the unchanged 64-dimensional embedding table. -/
noncomputable def bosUnit : EucSpace 64 := EuclideanSpace.single 2 1

/-- The second coordinate of the original first head carries the BOS indicator.
Source: the contiguous V chunk and head layout at f11b6e2. -/
noncomputable def bosDirection : EucSpace countConfig.head_dim :=
  EuclideanSpace.single ⟨1, by decide⟩ 1

/-- The original raw embedding table with simultaneous ONE, BOS and phase channels.
Source: cbafbe9 token IDs; no context statistic is inserted into an embedding entry. -/
noncomputable def ratioEmbedding (token : Fin countConfig.vocab_size) : EucSpace 64 :=
  if token.val = 1 then bosUnit else phaseEmbedding token

/-- Read the BOS residual coordinate into V's second coordinate, without a query/key component.
Source: the original fused projection; coordinate 129 is V coordinate one. -/
noncomputable def bosQKV : EucSpace 64 →L[ℝ] EucSpace 192 :=
  (innerSL ℝ bosUnit).smulRight (EuclideanSpace.single ⟨129, by decide⟩ 1)

/-- Two token indicators share the same original fused projection.
Source: the sum of the ONE and BOS rank-one matrices, with no added attention operator. -/
noncomputable def ratioQKV : EucSpace 64 →L[ℝ] EucSpace 192 := countQKV + bosQKV

/-- Put the head's count and denominator into independent residual coordinates.
Source: an ordinary nonzero W_o; the head's coordinate one is a value, not the residual phase. -/
noncomputable def ratioOutput : EucSpace 64 →L[ℝ] EucSpace 64 :=
  phaseOutput + (innerSL ℝ phaseUnit).smulRight bosUnit

/-- A concrete original model carrying the two channels and retained phase.
Source: Basis small parity dimensions, with zero FFN only for the feature calculation. -/
noncomputable def ratioParams : ModelParams countConfig where
  embedding := ratioEmbedding
  blocks := fun _ =>
    { attn := { W_qkv := ratioQKV, W_o := ratioOutput, log_alpha := fun _ => 0 }
      ffn := { W_in := 0, W_out := 0 } }

/-- The actual prenorm/QKV gives the same multiplier to the two unit token embeddings.
Source: RMSNorm.forward at f11b6e2; orthogonal phase tokens project to zero in both value channels. -/
theorem ratio_projected (eps : ℝ) (token : Fin countConfig.vocab_size) :
    ratioQKV (rmsNormEps eps (ratioEmbedding token)) =
      if token.val = 22 then countScale eps • EuclideanSpace.single ⟨128, by decide⟩ 1
      else if token.val = 1 then countScale eps • EuclideanSpace.single ⟨129, by decide⟩ 1
      else 0 := by
  by_cases hbos : token.val = 1
  · simp [ratioEmbedding, hbos, ratioQKV, bosQKV, countQKV,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, rmsNormEps,
      controlUnit, bosUnit, EuclideanSpace.inner_single_left, PiLp.norm_single, countScale]
  · simp only [ratioEmbedding, ite_eq_right hbos]
    by_cases hone : token.val = 22
    · simp [phaseEmbedding, hone, ratioQKV, bosQKV, countQKV,
        ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, rmsNormEps,
        controlUnit, bosUnit, EuclideanSpace.inner_single_left, PiLp.norm_single, countScale]
    · simp only [ite_eq_right hone, phaseEmbedding]
      split_ifs <;>
        simp [ratioQKV, bosQKV, countQKV, ContinuousLinearMap.smulRight_apply,
          innerSL_apply_apply, rmsNormEps, controlUnit, bosUnit, phaseUnit,
          EuclideanSpace.inner_single_left]

/-- Every value-coordinate unit in the first head has zero Q/K and the expected V coordinate.
Source: the original chunk/view, uniformly for its sixteen within-head coordinates. -/
theorem first_value_slices (c : Fin countConfig.head_dim) (a : ℝ) :
    headSlice countConfig (qkvSlice countConfig (qkvQ countConfig)
      (a • EuclideanSpace.single ⟨128 + c.val, by change 128 + c.val < 192; have hc := c.isLt; change c.val < 16 at hc; omega⟩ 1)) 0 = 0 ∧
    headSlice countConfig (qkvSlice countConfig (qkvK countConfig)
      (a • EuclideanSpace.single ⟨128 + c.val, by change 128 + c.val < 192; have hc := c.isLt; change c.val < 16 at hc; omega⟩ 1)) 0 = 0 ∧
    headSlice countConfig (qkvSlice countConfig (qkvV countConfig)
      (a • EuclideanSpace.single ⟨128 + c.val, by change 128 + c.val < 192; have hc := c.isLt; change c.val < 16 at hc; omega⟩ 1)) 0 =
        a • EuclideanSpace.single c 1 := by
  have hq : ∀ j : Fin 64, (qkvQ countConfig j).val = j.val := by decide
  have hk : ∀ j : Fin 64, (qkvK countConfig j).val = 64 + j.val := by decide
  have hv : ∀ j : Fin 64, (qkvV countConfig j).val = 128 + j.val := by decide
  have hh : ∀ j : Fin countConfig.head_dim,
      ((headSplit countConfig).symm (0, j)).val = j.val := by decide
  refine ⟨?_, ?_, ?_⟩ <;> ext j
  · have hj : j.val < 16 := j.isLt
    simp only [headSlice_apply, qkvSlice_apply, PiLp.smul_apply, PiLp.single_apply, PiLp.zero_apply]
    rw [ite_eq_right (by
      intro h
      have hi := congrArg Fin.val h
      rw [hq, hh] at hi
      change j.val = 128 + c.val at hi
      omega)]
    simp
  · have hj : j.val < 16 := j.isLt
    simp only [headSlice_apply, qkvSlice_apply, PiLp.smul_apply, PiLp.single_apply, PiLp.zero_apply]
    rw [ite_eq_right (by
      intro h
      have hi := congrArg Fin.val h
      rw [hk, hh] at hi
      change 64 + j.val = 128 + c.val at hi
      omega)]
    simp
  · simp only [headSlice_apply, qkvSlice_apply, PiLp.smul_apply, PiLp.single_apply]
    have heq : qkvV countConfig ((headSplit countConfig).symm (0, j)) =
        (⟨128 + c.val, by have hc := c.isLt; change c.val < 16 at hc; omega⟩ : Fin 192) ↔ j = c := by
      rw [Fin.ext_iff, hv, hh, Fin.ext_iff]
      change 128 + j.val = 128 + c.val ↔ j.val = c.val
      omega
    simp only [heq]

/-- Raw ONE and BOS indicators in distinct actual value coordinates.
Source: the token embedding assignment and cbafbe9's integer vocabulary. -/
noncomputable def ratioValue (scale : ℝ) (token : ℤ) : EucSpace countConfig.head_dim :=
  if token = oneBit then scale • countDirection
  else if token = bos then scale • bosDirection else 0

/-- Splitting the actual projection yields both faithful indicators, with zero Q/K.
Source: the proved embedding/QKV calculation above; no prepared semantic values are assumed. -/
theorem ratio_qkv_features (eps : ℝ) (token : Fin countConfig.vocab_size) :
    headSlice countConfig (qkvSlice countConfig (qkvQ countConfig)
      (ratioQKV (rmsNormEps eps (ratioEmbedding token)))) 0 = 0 ∧
    headSlice countConfig (qkvSlice countConfig (qkvK countConfig)
      (ratioQKV (rmsNormEps eps (ratioEmbedding token)))) 0 = 0 ∧
    headSlice countConfig (qkvSlice countConfig (qkvV countConfig)
      (ratioQKV (rmsNormEps eps (ratioEmbedding token)))) 0 =
        ratioValue (countScale eps) (token.val : ℤ) := by
  rw [ratio_projected]
  by_cases hone : token.val = 22
  · have hraw : (token.val : ℤ) = oneBit := by
      change (token.val : ℤ) = 22
      exact_mod_cast hone
    simp only [ite_eq_left hone, ratioValue, ite_eq_left hraw]
    exact first_value_slices ⟨0, by decide⟩ (countScale eps)
  · have hraw : (token.val : ℤ) ≠ oneBit := by
      change (token.val : ℤ) ≠ 22
      exact_mod_cast hone
    simp only [ite_eq_right hone, ratioValue, ite_eq_right hraw]
    by_cases hbos : token.val = 1
    · have hrawbos : (token.val : ℤ) = bos := by
        change (token.val : ℤ) = 1
        exact_mod_cast hbos
      simp only [ite_eq_left hbos, ite_eq_left hrawbos]
      exact first_value_slices ⟨1, by decide⟩ (countScale eps)
    · have hrawbos : (token.val : ℤ) ≠ bos := by
        change (token.val : ℤ) ≠ 1
        exact_mod_cast hbos
      simp only [ite_eq_right hbos, ite_eq_right hrawbos]
      refine ⟨?_, ?_, ?_⟩ <;> ext j <;> simp

end Transformer.GPTMini.Semantics
