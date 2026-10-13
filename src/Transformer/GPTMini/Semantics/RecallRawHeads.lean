import Transformer.GPTMini.Semantics.RecallHeadValues
import Transformer.GPTMini.Semantics.Block

/-!
# Evaluate both original recall heads from raw tokens

Source: headAt/attnSubLayer at f11b6e2 and the simultaneous new raw
embedding/QKV. The common raw RMS multiplier is canceled by a single
fixed ordinary matrix scale. The actual first head therefore has the
proved predecessor query/key and faithful compact raw-key values.
The actual second head has zero Q/K and reads the protected reserved
type. On legal raw recall inputs this type marks only BOS.

The nonzero original W_o writes the copied eight coordinates to 18..25
and the marker to 26. It preserves raw symbol/type coordinates. No
prepared paired feature, positional score gap, correct output logit or
task-answer hypothesis is used to derive these actual head formulas.
Deriving the causal marker mass, the table gate and the second-block
retrieval/readout remains necessary for full raw recall correctness.
-/

namespace Transformer.GPTMini.Semantics

/-- Original W_o writes both head outputs into separate, initially empty residual channels.
Source: first-head columns 0..7 and head-one column 16 of the true merged head array. -/
noncomputable def recallFirstOutput : EucSpace 64 →L[ℝ] EucSpace 64 :=
  (recallSlotWrite 18).comp (recallSlotRead 0) +
    (innerSL ℝ (recallUnit 16)).smulRight (recallUnit 26)

/-- A simultaneous original attention parameter record, with finite learned predecessor temperature.
Source: unchanged AttnParams and original four heads; marker head uses zero scores. -/
noncomputable def recallFirstAttention (eps alpha : ℝ) : AttnParams recallConfig where
  W_qkv := recallFirstQKV eps
  W_o := recallFirstOutput
  log_alpha := fun h => if h = 0 then alpha else 0

/-- The ordinary output matrix truly merges the predecessor and marker channels in their intended slots.
Source: original headMerge's exact contiguous view and the actual two-channel W_o. -/
theorem recallFirstOutput_merge (heads : Fin recallConfig.n_heads → EucSpace recallConfig.head_dim) :
    recallFirstOutput (headMerge recallConfig heads) =
      recallSlotWrite 18 (recallHeadRead (heads 0)) +
        (heads 1 ⟨0, by decide⟩) • recallUnit 26 := by
  have hs : ∀ c : Fin 8, headSplit recallConfig (recallSlotIndex 0 c) =
      (0, ⟨c.val, by change c.val < 16; have hc := c.isLt; omega⟩) := by decide
  have hm : headSplit recallConfig 16 = (1, ⟨0, by decide⟩) := by decide
  have hr : recallSlotRead 0 (headMerge recallConfig heads) = recallHeadRead (heads 0) := by
    ext c
    rw [recallSlotRead_at, headMerge_apply, hs]
    rfl
  simp only [recallFirstOutput, add_apply, ContinuousLinearMap.comp_apply, hr,
    ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, recallUnit,
    EuclideanSpace.inner_single_left, map_one, one_mul, headMerge_apply, hm]

/-- Every actual raw input yields the specified first query/key after real prenorm and fused QKV.
Source: the complete raw constant/RMS identities and fixed matrix compensation. -/
theorem recall_raw_qk (eps : ℝ) (heps : 0 ≤ eps) (token : Fin recallConfig.vocab_size) :
    headSlice recallConfig (qkvSlice recallConfig (qkvQ recallConfig)
        (recallFirstQKV eps (rmsNormEps eps (recallRawEmbedding token)))) 0 = adjacentQuery ∧
      headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig)
        (recallFirstQKV eps (rmsNormEps eps (recallRawEmbedding token)))) 0 = adjacentDirection := by
  have hn := ne_of_gt (recallRawScale_pos eps heps)
  rw [recallFirstQKV_query, recallFirstQKV_key, recallRawEmbedding_rms_constant,
    inv_mul_cancel₀ hn, one_smul, one_smul]
  exact ⟨rfl, rfl⟩

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The actual first value slice is the faithful raw compact key projection with all prenorm factors discharged.
Source: true raw RMS scaling, real QKV rows and proved linear head insertion. -/
theorem recall_raw_value (eps : ℝ) (heps : 0 ≤ eps) (token : Fin recallConfig.vocab_size) :
    headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig)
        (recallFirstQKV eps (rmsNormEps eps (recallRawEmbedding token)))) 0 =
      recallHeadValue (recallSlotRead 1 (recallRawEmbedding token)) := by
  rw [recallFirstQKV_value, (recallRawEmbedding_rms_reads eps token).1, recallHeadValue_smul,
    inv_smul_smul₀ (ne_of_gt (recallRawScale_pos eps heps))]

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The second actual head simultaneously has zero Q/K and exactly the raw reserved/BOS flag as its value.
Source: true marker rows, evaluated raw normalization and fixed inverse scale. -/
theorem recall_raw_marker (eps : ℝ) (heps : 0 ≤ eps) (token : Fin recallConfig.vocab_size) :
    headSlice recallConfig (qkvSlice recallConfig (qkvQ recallConfig)
        (recallFirstQKV eps (rmsNormEps eps (recallRawEmbedding token)))) 1 = 0 ∧
      headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig)
        (recallFirstQKV eps (rmsNormEps eps (recallRawEmbedding token)))) 1 = 0 ∧
      headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig)
        (recallFirstQKV eps (rmsNormEps eps (recallRawEmbedding token)))) 1 =
        recallRawEmbedding token 37 • (EuclideanSpace.single 0 1 : EucSpace 16) := by
  have hqk := recallFirstQKV_marker_qk eps (rmsNormEps eps (recallRawEmbedding token))
  refine ⟨hqk.1, hqk.2, ?_⟩
  rw [recallFirstQKV_marker_value, recallRawEmbedding_rms]
  simp only [PiLp.smul_apply, smul_eq_mul, ← mul_assoc,
    inv_mul_cancel₀ (ne_of_gt (recallRawScale_pos eps heps)), one_mul]
  rfl

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The first actual original head is exactly the predecessor head evaluated on faithful raw-token values.
Source: headAt's true prenorm/fused QKV, with every projected query, key and value derived above. -/
theorem recall_predecessor_head (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    headAt recallConfig (recallFirstAttention eps alpha) eps positions
        (fun j => recallRawEmbedding (tokens j)) 0 i =
      attentionHead recallConfig alpha eps (fun _ => adjacentQuery) (fun _ => adjacentDirection)
        (fun j => recallHeadValue (recallSlotRead 1 (recallRawEmbedding (tokens j)))) positions i := by
  have hq := funext fun j => (recall_raw_qk eps heps (tokens j)).1
  have hk := funext fun j => (recall_raw_qk eps heps (tokens j)).2
  have hv := funext fun j => recall_raw_value eps heps (tokens j)
  unfold headAt
  dsimp only [recallFirstAttention]
  rw [hq, hk, hv, ite_eq_left (rfl : (0 : Fin 4) = 0)]

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The second original head is the true uniform causal marker head on the same raw embeddings and QKV.
Source: simultaneous headAt projections, so positional marking does not replace or widen the predecessor head. -/
theorem recall_marker_head (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    headAt recallConfig (recallFirstAttention eps alpha) eps positions
        (fun j => recallRawEmbedding (tokens j)) 1 i =
      attentionHead recallConfig 0 eps (fun _ => 0) (fun _ => 0)
        (fun j => recallRawEmbedding (tokens j) 37 • (EuclideanSpace.single 0 1 : EucSpace 16)) positions i := by
  have hq := funext fun j => (recall_raw_marker eps heps (tokens j)).1
  have hk := funext fun j => (recall_raw_marker eps heps (tokens j)).2.1
  have hv := funext fun j => (recall_raw_marker eps heps (tokens j)).2.2
  unfold headAt
  dsimp only [recallFirstAttention]
  rw [hq, hk, hv, ite_eq_right (by decide : (1 : Fin 4) ≠ 0)]

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The actual first output projection is zero on every protected raw code/type coordinate.
Source: its exact copy/marker support, independently of head values or temperature. -/
theorem recallFirstOutput_protected (x : EucSpace 64) (i : Fin 64)
    (hi : i.val < 18 ∨ 27 ≤ i.val) : recallFirstOutput x i = 0 := by
  have hn : i ≠ 26 := by intro he; have hv := congrArg Fin.val he; change i.val = 26 at hv; omega
  simp only [recallFirstOutput, add_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, PiLp.add_apply,
    recallUnit, PiLp.smul_apply, PiLp.single_apply, ite_eq_right hn, smul_eq_mul, mul_zero, add_zero]
  apply recallSlotWrite_outside
  change i.val < 18 ∨ 18 + 8 ≤ i.val
  omega

example : (0 : Fin 64).val < 18 ∨ 27 ≤ (0 : Fin 64).val := by decide

/-- The actual attention residual preserves every raw symbol/type coordinate for any context.
Source: original attnSubLayer/headMerge/W_o, with its derived output support above. -/
theorem recall_first_attn_protected (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace 64) (i : Fin T) (c : Fin 64) (hc : c.val < 18 ∨ 27 ≤ c.val) :
    (x i + attnSubLayer recallConfig (recallFirstAttention eps alpha) eps positions x i) c = x i c := by
  rw [PiLp.add_apply, attnSubLayer_eq_heads]
  change x i c + recallFirstOutput _ c = x i c
  rw [recallFirstOutput_protected _ c hc, add_zero]

example : (36 : Fin 64).val < 18 ∨ 27 ≤ (36 : Fin 64).val := by decide

end Transformer.GPTMini.Semantics
