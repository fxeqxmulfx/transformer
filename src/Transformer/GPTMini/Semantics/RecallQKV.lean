import Transformer.GPTMini.Semantics.RecallEmbeddingFeatures
import Transformer.GPTMini.Reshape

/-!
# Simultaneous predecessor and BOS rows in the original fused QKV

Source: CausalMHA.forward's 64-to-192 bias-free projection/chunk/view
at f11b6e2. Head zero reads the protected constant into the actual
shifted predecessor query and key, and reads the compact raw key slot
into its first eight value coordinates. Head one has zero Q/K and
reads the reserved type into its first value coordinate. Its uniform
causal mass will be a BOS position marker on the validated grammar.

Each row below is an ordinary linear functional. The compensating
multiplier is fixed from the model epsilon and the proved common raw
embedding norm; it is not an input-dependent normalization or oracle.
Both heads occupy the original four-head projection simultaneously.
This module evaluates the actual chunk/view on arbitrary real inputs;
the next module connects the rows to raw tokens and headAt.
-/

namespace Transformer.GPTMini.Semantics

/-- Values occupy the first eight coordinates of a genuine original head.
Source: the ordinary V reshape, which does not apply RoPE to values. -/
noncomputable def recallHeadValue (code : EucSpace 8) : EucSpace recallConfig.head_dim :=
  (EuclideanSpace.equiv (Fin 16) ℝ).symm fun c =>
    if h : c.val < 8 then code ⟨c.val, h⟩ else 0

/-- The true head-value coordinate retains each of the eight input scalars, then zeros.
Source: the explicit V layout, independent of any semantic routing assumption. -/
theorem recallHeadValue_coordinate (code : EucSpace 8) (c : Fin 16) :
    recallHeadValue code c = if h : c.val < 8 then code ⟨c.val, h⟩ else 0 := by
  rfl

/-- Every row of a simultaneous ordinary QKV matrix, with all unused rows explicitly zero.
Source: original chunk offsets 0/64/128 and head-one value offset 144. -/
noncomputable def recallFirstRow (i : Fin 192) : EucSpace 64 →L[ℝ] ℝ :=
  if h : i.val < 16 then
    adjacentQuery ⟨i.val, h⟩ • innerSL ℝ (recallUnit 0)
  else if h : 64 ≤ i.val ∧ i.val < 80 then
    adjacentDirection ⟨i.val - 64, by omega⟩ • innerSL ℝ (recallUnit 0)
  else if h : 128 ≤ i.val ∧ i.val < 136 then
    innerSL ℝ (recallUnit ⟨i.val - 127, by omega⟩)
  else if i.val = 144 then innerSL ℝ (recallUnit 37) else 0

/-- The actual 64-to-192 fused matrix has these linear rows and one fixed prenorm compensation.
Source: original W_qkv type; no paired input or prefix function is called by a row. -/
noncomputable def recallFirstQKV (eps : ℝ) : EucSpace 64 →L[ℝ] EucSpace 192 :=
  (recallRawScale eps)⁻¹ •
    ((EuclideanSpace.equiv (Fin 192) ℝ).symm.toContinuousLinearMap.comp
      (ContinuousLinearMap.pi recallFirstRow))

/-- A true fused-matrix coordinate equals its specified ordinary linear row.
Source: pi assembly and the actual Euclidean coordinate equivalence. -/
theorem recallFirstQKV_coordinate (eps : ℝ) (x : EucSpace 64) (i : Fin 192) :
    recallFirstQKV eps x i = (recallRawScale eps)⁻¹ * recallFirstRow i x := by
  rfl

/-- The genuine first-head query slice reads only the protected constant into the shifted rotary direction.
Source: all sixteen actual query rows and the original chunk/view indices. -/
theorem recallFirstQKV_query (eps : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvQ recallConfig) (recallFirstQKV eps x)) 0 =
      ((recallRawScale eps)⁻¹ * x 0) • adjacentQuery := by
  have hq : ∀ c : Fin 16,
      qkvQ recallConfig ((headSplit recallConfig).symm (0, c)) =
        (⟨c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallFirstQKV eps x (qkvQ recallConfig ((headSplit recallConfig).symm (0, c))) =
    (((recallRawScale eps)⁻¹ * x 0) • adjacentQuery) c
  rw [hq, recallFirstQKV_coordinate]
  have hc : c.val < 16 := c.isLt
  rw [recallFirstRow, dite_eq_left hc]
  simp only [smul_apply, innerSL_apply_apply, recallUnit, EuclideanSpace.inner_single_left,
    map_one, one_mul, PiLp.smul_apply, smul_eq_mul]
  ring

/-- The genuine first-head key slice reads the same constant into the actual unshifted rotary direction.
Source: the key chunk's sixteen original rows, simultaneously with the query rows. -/
theorem recallFirstQKV_key (eps : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig) (recallFirstQKV eps x)) 0 =
      ((recallRawScale eps)⁻¹ * x 0) • adjacentDirection := by
  have hk : ∀ c : Fin 16,
      qkvK recallConfig ((headSplit recallConfig).symm (0, c)) =
        (⟨64 + c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallFirstQKV eps x (qkvK recallConfig ((headSplit recallConfig).symm (0, c))) =
    (((recallRawScale eps)⁻¹ * x 0) • adjacentDirection) c
  rw [hk, recallFirstQKV_coordinate]
  have hc : c.val < 16 := c.isLt
  have hn : ¬64 + c.val < 16 := by omega
  have hr : 64 ≤ 64 + c.val ∧ 64 + c.val < 80 := by omega
  rw [recallFirstRow, dite_eq_right hn, dite_eq_left hr]
  simp only [smul_apply, innerSL_apply_apply, recallUnit, EuclideanSpace.inner_single_left,
    map_one, one_mul, PiLp.smul_apply, smul_eq_mul, Nat.add_sub_cancel_left]
  ring

/-- The genuine first-head value slice reads exactly the compact raw key slot and no type/value slot.
Source: all sixteen original V coordinates, including the unused zero half. -/
theorem recallFirstQKV_value (eps : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig) (recallFirstQKV eps x)) 0 =
      (recallRawScale eps)⁻¹ • recallHeadValue (recallSlotRead 1 x) := by
  have hv : ∀ c : Fin 16,
      qkvV recallConfig ((headSplit recallConfig).symm (0, c)) =
        (⟨128 + c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallFirstQKV eps x (qkvV recallConfig ((headSplit recallConfig).symm (0, c))) =
    (recallRawScale eps)⁻¹ * recallHeadValue (recallSlotRead 1 x) c
  rw [hv, recallFirstQKV_coordinate]
  have hc : c.val < 16 := c.isLt
  have hn : ¬128 + c.val < 16 := by omega
  have hk : ¬(64 ≤ 128 + c.val ∧ 128 + c.val < 80) := by omega
  rw [recallFirstRow, dite_eq_right hn, dite_eq_right hk]
  by_cases hs : c.val < 8
  · have hr : 128 ≤ 128 + c.val ∧ 128 + c.val < 136 := by omega
    rw [dite_eq_left hr]
    simp only [innerSL_apply_apply, recallUnit, EuclideanSpace.inner_single_left, map_one, one_mul,
      recallHeadValue_coordinate, dite_eq_left hs, recallSlotRead_at]
    have he : (⟨128 + c.val - 127, by omega⟩ : Fin 64) = recallSlotIndex 1 ⟨c.val, hs⟩ := by
      apply Fin.ext
      change 128 + c.val - 127 = 1 + c.val
      omega
    rw [he]
  · have hr : ¬(128 ≤ 128 + c.val ∧ 128 + c.val < 136) := by omega
    have hm : ¬128 + c.val = 144 := by omega
    simp only [dite_eq_right hr, ite_eq_right hm, zero_apply, mul_zero,
      recallHeadValue_coordinate, dite_eq_right hs]

/-- The original second head has no query or key contribution in this simultaneous matrix.
Source: its actual Q/K chunk/view rows, all outside head zero's assigned intervals. -/
theorem recallFirstQKV_marker_qk (eps : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvQ recallConfig) (recallFirstQKV eps x)) 1 = 0 ∧
      headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig) (recallFirstQKV eps x)) 1 = 0 := by
  have hq : ∀ c : Fin 16, qkvQ recallConfig ((headSplit recallConfig).symm (1, c)) =
      (⟨16 + c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  have hk : ∀ c : Fin 16, qkvK recallConfig ((headSplit recallConfig).symm (1, c)) =
      (⟨80 + c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  constructor
  · apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
    funext c
    change recallFirstQKV eps x (qkvQ recallConfig ((headSplit recallConfig).symm (1, c))) = (0 : EucSpace 16) c
    rw [hq, recallFirstQKV_coordinate]
    have hc : c.val < 16 := c.isLt
    have ha : ¬16 + c.val < 16 := by omega
    have hb : ¬(64 ≤ 16 + c.val ∧ 16 + c.val < 80) := by omega
    have hd : ¬(128 ≤ 16 + c.val ∧ 16 + c.val < 136) := by omega
    have hm : ¬16 + c.val = 144 := by omega
    simp only [recallFirstRow, dite_eq_right ha, dite_eq_right hb, dite_eq_right hd,
      ite_eq_right hm, zero_apply, mul_zero, PiLp.zero_apply]
  · apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
    funext c
    change recallFirstQKV eps x (qkvK recallConfig ((headSplit recallConfig).symm (1, c))) = (0 : EucSpace 16) c
    rw [hk, recallFirstQKV_coordinate]
    have hc : c.val < 16 := c.isLt
    have ha : ¬80 + c.val < 16 := by omega
    have hb : ¬(64 ≤ 80 + c.val ∧ 80 + c.val < 80) := by omega
    have hd : ¬(128 ≤ 80 + c.val ∧ 80 + c.val < 136) := by omega
    have hm : ¬80 + c.val = 144 := by omega
    simp only [recallFirstRow, dite_eq_right ha, dite_eq_right hb, dite_eq_right hd,
      ite_eq_right hm, zero_apply, mul_zero, PiLp.zero_apply]

/-- Head one's true value slice reads the protected reserved/BOS flag in its first coordinate.
Source: row 144 and the remaining fifteen zero V rows, simultaneously with the predecessor head. -/
theorem recallFirstQKV_marker_value (eps : ℝ) (x : EucSpace 64) :
    headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig) (recallFirstQKV eps x)) 1 =
      ((recallRawScale eps)⁻¹ * x 37) • (EuclideanSpace.single 0 1 : EucSpace 16) := by
  have hv : ∀ c : Fin 16, qkvV recallConfig ((headSplit recallConfig).symm (1, c)) =
      (⟨144 + c.val, by have hc := c.isLt; omega⟩ : Fin 192) := by decide
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallFirstQKV eps x (qkvV recallConfig ((headSplit recallConfig).symm (1, c))) =
    (((recallRawScale eps)⁻¹ * x 37) • (EuclideanSpace.single 0 1 : EucSpace 16)) c
  rw [hv, recallFirstQKV_coordinate]
  have hc : c.val < 16 := c.isLt
  have ha : ¬144 + c.val < 16 := by omega
  have hb : ¬(64 ≤ 144 + c.val ∧ 144 + c.val < 80) := by omega
  have hd : ¬(128 ≤ 144 + c.val ∧ 144 + c.val < 136) := by omega
  rw [recallFirstRow, dite_eq_right ha, dite_eq_right hb, dite_eq_right hd]
  by_cases hz : c = 0
  · subst c
    norm_num [recallUnit, innerSL_apply_apply, EuclideanSpace.inner_single_left, PiLp.smul_apply]
  · have hm : ¬144 + c.val = 144 := by intro he; apply hz; apply Fin.ext; change c.val = 0; omega
    simp only [ite_eq_right hm, zero_apply, PiLp.smul_apply, PiLp.single_apply,
      ite_eq_right hz, smul_eq_mul, mul_zero]

end Transformer.GPTMini.Semantics
