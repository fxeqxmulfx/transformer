import Transformer.GPTMini.Semantics.RecallProjectionScale

/-!
# The genuine fused second QKV matrix for compact raw recall matching

Source: the unchanged 64-to-192 ordinary QKV/chunk/view at f11b6e2.
Head zero reads raw query slot 1 into the actual slow RoPE layout,
and reads genuine gated table-key slot 27 into that same layout.
Its unrotated values read the independent raw value slot 9. Every
unused row is zero. One ordinary shared gain scales both Q and K;
it does not rescale values or use the actual RMS multiplier as an
input-dependent matrix coefficient.

The coordinate slices are evaluated on arbitrary real residuals,
including imperfect copied keys. The next-prenorm formulas retain
their genuine position-dependent multiplier. The finite uniform
gain derived from raw encoder bounds is supplied to this matrix in
the next coupling step. Actual normalized key errors, robust routing,
tied readout and complete validated-parser correctness remain.
-/

namespace Transformer.GPTMini.Semantics

/-- Every row is an ordinary linear functional on the true residual; no symbol-only decoding is used.
Source: actual Q/K/V offsets 0/64/128, faithful eight-to-sixteen insertion and independent value interval. -/
noncomputable def recallSecondRow (gain : ℝ) (i : Fin 192) : EucSpace 64 →L[ℝ] ℝ :=
  if h : i.val < 16 then gain •
    ((innerSL ℝ (EuclideanSpace.single ⟨i.val, h⟩ 1 : EucSpace 16)).comp
      (recallRotaryInsert.comp (recallSlotRead 1)))
  else if h : 64 ≤ i.val ∧ i.val < 80 then gain •
    ((innerSL ℝ (EuclideanSpace.single ⟨i.val - 64, by omega⟩ 1 : EucSpace 16)).comp
      (recallRotaryInsert.comp (recallSlotRead 27)))
  else if h : 128 ≤ i.val ∧ i.val < 136 then recallProbe ⟨i.val - 119, by omega⟩ else 0

/-- The complete shared fused matrix fits the original width, four heads and three ordinary chunks.
Source: genuine finite row assembly and Euclidean coordinate equivalence, without a prefix or position argument. -/
noncomputable def recallSecondQKV (gain : ℝ) : EucSpace 64 →L[ℝ] EucSpace 192 :=
  (EuclideanSpace.equiv (Fin 192) ℝ).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi (recallSecondRow gain))

/-- Every coordinate of the actual fused projection equals its specified real linear row.
Source: evaluated finite pi assembly and actual Euclidean coordinate equivalence. -/
theorem recallSecondQKV_coordinate (gain : ℝ) (x : EucSpace 64) (i : Fin 192) :
    recallSecondQKV gain x i = recallSecondRow gain i x := by rfl

/-- Every actual query coordinate is the shared gain times the real compact insertion of the raw query interval.
Source: all sixteen true query chunk/view indices and ordinary matrix rows, including unused fast pairs. -/
theorem recallSecondQKV_query (gain : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvQ recallConfig) (recallSecondQKV gain x)) 0 =
      gain • recallRotaryInsert (recallSlotRead 1 x) := by
  have hq : ∀ c : Fin 16, qkvQ recallConfig ((headSplit recallConfig).symm (0, c)) =
      (⟨c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallSecondQKV gain x (qkvQ recallConfig ((headSplit recallConfig).symm (0, c))) =
    (gain • recallRotaryInsert (recallSlotRead 1 x)) c
  rw [hq, recallSecondQKV_coordinate, recallSecondRow, dite_eq_left c.isLt]
  simp only [smul_apply, ContinuousLinearMap.comp_apply, innerSL_apply_apply,
    EuclideanSpace.inner_single_left, map_one, one_mul, PiLp.smul_apply, smul_eq_mul]

/-- Every actual key coordinate reads the genuine gated copied-key slot with the same shared ordinary gain.
Source: all sixteen true key chunk/view rows, simultaneously with query and independent value rows. -/
theorem recallSecondQKV_key (gain : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig) (recallSecondQKV gain x)) 0 =
      gain • recallRotaryInsert (recallSlotRead 27 x) := by
  have hk : ∀ c : Fin 16, qkvK recallConfig ((headSplit recallConfig).symm (0, c)) =
      (⟨64 + c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallSecondQKV gain x (qkvK recallConfig ((headSplit recallConfig).symm (0, c))) =
    (gain • recallRotaryInsert (recallSlotRead 27 x)) c
  rw [hk, recallSecondQKV_coordinate]
  have hc : c.val < 16 := c.isLt
  have hn : ¬64 + c.val < 16 := by omega
  have hr : 64 ≤ 64 + c.val ∧ 64 + c.val < 80 := by omega
  rw [recallSecondRow, dite_eq_right hn, dite_eq_left hr]
  simp only [smul_apply, ContinuousLinearMap.comp_apply, innerSL_apply_apply,
    EuclideanSpace.inner_single_left, map_one, one_mul, PiLp.smul_apply, smul_eq_mul,
    Nat.add_sub_cancel_left]

/-- The simultaneous genuine value slice reads the complete independent compact value code and zeros the other half.
Source: all sixteen true V chunk/view coordinates, with no query/key gain applied to values. -/
theorem recallSecondQKV_value (gain : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig) (recallSecondQKV gain x)) 0 =
      recallHeadValue (recallSlotRead 9 x) := by
  have hv : ∀ c : Fin 16, qkvV recallConfig ((headSplit recallConfig).symm (0, c)) =
      (⟨128 + c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallSecondQKV gain x (qkvV recallConfig ((headSplit recallConfig).symm (0, c))) =
    recallHeadValue (recallSlotRead 9 x) c
  rw [hv, recallSecondQKV_coordinate]
  have hn : ¬128 + c.val < 16 := by omega
  have hk : ¬(64 ≤ 128 + c.val ∧ 128 + c.val < 80) := by omega
  rw [recallSecondRow, dite_eq_right hn, dite_eq_right hk]
  by_cases hs : c.val < 8
  · have hr : 128 ≤ 128 + c.val ∧ 128 + c.val < 136 := by omega
    rw [dite_eq_left hr, recallProbe_apply, recallHeadValue_coordinate, dite_eq_left hs, recallSlotRead_at]
    have he : (⟨128 + c.val - 119, by omega⟩ : Fin 64) = recallSlotIndex 9 ⟨c.val, hs⟩ := by
      apply Fin.ext
      change 128 + c.val - 119 = 9 + c.val
      omega
    rw [he]
  · have hr : ¬(128 ≤ 128 + c.val ∧ 128 + c.val < 136) := by omega
    simp only [dite_eq_right hr, zero_apply, recallHeadValue_coordinate, dite_eq_right hs]

/-- Every unused row is exactly zero on every real residual, independent of the matching gain.
Source: the complete ordinary fused matrix's original chunk/head intervals. -/
theorem recallSecondQKV_unused (gain : ℝ) (x : EucSpace 64) (i : Fin 192)
    (hi : (16 ≤ i.val ∧ i.val < 64) ∨ (80 ≤ i.val ∧ i.val < 128) ∨ 136 ≤ i.val) :
    recallSecondQKV gain x i = 0 := by
  have hq : ¬i.val < 16 := by omega
  have hk : ¬(64 ≤ i.val ∧ i.val < 80) := by omega
  have hv : ¬(128 ≤ i.val ∧ i.val < 136) := by omega
  rw [recallSecondQKV_coordinate, recallSecondRow, dite_eq_right hq, dite_eq_right hk, dite_eq_right hv, zero_apply]

example : (16 ≤ (16 : Fin 192).val ∧ (16 : Fin 192).val < 64) ∨
    (80 ≤ (16 : Fin 192).val ∧ (16 : Fin 192).val < 128) ∨ 136 ≤ (16 : Fin 192).val := by decide

/-- Real second prenorm leaves the genuine position-dependent multiplier in the actual query projection.
Source: original RMSNorm followed by the proved shared linear query rows, rather than per-input inverse compensation. -/
theorem recallSecondQKV_rms_query (gain eps : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvQ recallConfig)
      (recallSecondQKV gain (rmsNormEps eps x))) 0 =
        (gain * recallResidualScale eps x) • recallRotaryInsert (recallSlotRead 1 x) := by
  rw [recallSecondQKV_query, recallResidualScale_rms, map_smul, map_smul, smul_smul]

/-- Real second prenorm gives the same genuine multiplier on the ordinary gated-key projection.
Source: actual shared matrix and exact linear insertion of all imperfect real copied-key coordinates. -/
theorem recallSecondQKV_rms_key (gain eps : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig)
      (recallSecondQKV gain (rmsNormEps eps x))) 0 =
        (gain * recallResidualScale eps x) • recallRotaryInsert (recallSlotRead 27 x) := by
  rw [recallSecondQKV_key, recallResidualScale_rms, map_smul, map_smul, smul_smul]

/-- The true second value projection retains its actual prenorm multiplier without the Q/K matching gain.
Source: independent ordinary V rows and genuine linear compact-head insertion. -/
theorem recallSecondQKV_rms_value (gain eps : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig)
      (recallSecondQKV gain (rmsNormEps eps x))) 0 =
        recallResidualScale eps x • recallHeadValue (recallSlotRead 9 x) := by
  rw [recallSecondQKV_value, recallResidualScale_rms, map_smul, recallHeadValue_smul]

/-- An excluded genuine zero table-key slot yields an exactly zero original matching key.
Source: the full ordinary matrix and faithful linear insertion, needed for raw keys/BOS/post-table false writes. -/
theorem recallSecondQKV_key_zero (gain : ℝ) (x : EucSpace 64) (hkey : recallSlotRead 27 x = 0) :
    headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig) (recallSecondQKV gain x)) 0 = 0 := by
  rw [recallSecondQKV_key, hkey, map_zero, smul_zero]
  rfl

example : recallSlotRead 27 (0 : EucSpace 64) = 0 := by exact map_zero _

/-- A real query's zero own-value slot remains zero through actual prenorm and the simultaneous fused value projection.
Source: genuine RMS scaling and ordinary V rows, supplying original XSA with an actual zero self-value. -/
theorem recallSecondQKV_rms_self_zero (gain eps : ℝ) (x : EucSpace 64) (hvalue : recallSlotRead 9 x = 0) :
    headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig)
      (recallSecondQKV gain (rmsNormEps eps x))) 0 = 0 := by
  rw [recallSecondQKV_rms_value, hvalue, recallHeadValue_zero, smul_zero]

example : recallSlotRead 9 (recallRawEmbedding (recallKeyId 0)) = 0 := by
  rw [recallRawEmbedding_key]
  exact (recallKeyEmbedding_reads 0).2

end Transformer.GPTMini.Semantics
