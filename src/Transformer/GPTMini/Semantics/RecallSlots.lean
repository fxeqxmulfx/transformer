import Transformer.GPTMini.Semantics.RecallMatching

/-!
# Disjoint compact code slots in the original residual stream

Source: the small 64-wide Basis GPTMini at cbafbe9 and the bias-free
linear projections at f11b6e2. Eight-coordinate symbol codes are moved
by ordinary finite rank-one matrix sums. Each slot is a contiguous
eight-coordinate interval of the original residual stream. The exact
coordinate, readback, inner-product and disjointness calculations below
will support simultaneous raw key/value embeddings and copied keys.

No statistic of a whole prefix is inserted by these maps. The maps only
read and write token-local real coordinates. There is no widening or
alternate attention operator; raw token interpretation follows next.
Separate slots therefore preserve the key/value association problem:
the binding must still be computed by the actual causal head and FFN.
The insertion alone does not claim to solve that task.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- A compact eight-coordinate slot fits entirely inside the original width.
Source: the explicit new residual layout, with start at most 56. -/
def recallSlotIndex (offset : Fin 57) (c : Fin 8) : Fin 64 :=
  ⟨offset.val + c.val, by have ho := offset.isLt; have hc := c.isLt; omega⟩

/-- The ordinary matrix inserts eight coordinates into one residual slot.
Source: a finite sum of actual rank-one continuous linear operators. -/
noncomputable def recallSlotWrite (offset : Fin 57) : EucSpace 8 →L[ℝ] EucSpace 64 :=
  ∑ c : Fin 8, (innerSL ℝ (EuclideanSpace.single c 1)).smulRight
    (EuclideanSpace.single (recallSlotIndex offset c) 1)

/-- The reverse ordinary matrix reads exactly the same residual slot.
Source: the transposed rank-one coordinate matrices, without a semantic oracle. -/
noncomputable def recallSlotRead (offset : Fin 57) : EucSpace 64 →L[ℝ] EucSpace 8 :=
  ∑ c : Fin 8, (innerSL ℝ (EuclideanSpace.single (recallSlotIndex offset c) 1)).smulRight
    (EuclideanSpace.single c 1)

/-- Slot coordinates are distinct, so simultaneous code coordinates do not interfere.
Source: addition of the fixed offset is injective on Fin 8. -/
theorem recallSlotIndex_injective (offset : Fin 57) : Function.Injective (recallSlotIndex offset) := by
  intro a b h
  apply Fin.ext
  have he := congrArg Fin.val h
  change offset.val + a.val = offset.val + b.val at he
  omega

/-- A coordinate of the actual insertion matrix is its explicit finite rank-one sum.
Source: the stated matrix, before using any assumptions about a token or prefix. -/
theorem recallSlotWrite_coordinate (offset : Fin 57) (x : EucSpace 8) (i : Fin 64) :
    recallSlotWrite offset x i =
      ∑ c : Fin 8, x c * (if i = recallSlotIndex offset c then 1 else 0) := by
  simp [recallSlotWrite, sum_apply, ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
    EuclideanSpace.inner_single_left, WithLp.ofLp_sum, Finset.sum_apply,
    WithLp.ofLp_smul, Pi.smul_apply, Pi.single_apply]

/-- The actual insertion matrix evaluates to a sum of eight scaled residual axes.
Source: the explicit rank-one operator, with every input coordinate retained. -/
theorem recallSlotWrite_apply (offset : Fin 57) (x : EucSpace 8) :
    recallSlotWrite offset x =
      ∑ c : Fin 8, x c • (EuclideanSpace.single (recallSlotIndex offset c) 1 : EucSpace 64) := by
  simp only [recallSlotWrite, sum_apply, ContinuousLinearMap.smulRight_apply,
    innerSL_apply_apply, EuclideanSpace.inner_single_left, map_one, one_mul]

/-- Each assigned coordinate retains the exact input scalar in the genuine matrix output.
Source: the finite insertion sum and proved coordinate injectivity. -/
theorem recallSlotWrite_at (offset : Fin 57) (x : EucSpace 8) (c : Fin 8) :
    recallSlotWrite offset x (recallSlotIndex offset c) = x c := by
  rw [recallSlotWrite_coordinate, Fintype.sum_eq_single c]
  · simp
  · intro j hj
    have hn : recallSlotIndex offset c ≠ recallSlotIndex offset j :=
      fun he => hj ((recallSlotIndex_injective offset he).symm)
    simp only [ite_eq_right hn, mul_zero]

/-- Every coordinate outside the actual assigned interval is zero.
Source: the ordinary finite insertion matrix, with its exact support. -/
theorem recallSlotWrite_outside (offset : Fin 57) (x : EucSpace 8) (i : Fin 64)
    (hout : i.val < offset.val ∨ offset.val + 8 ≤ i.val) : recallSlotWrite offset x i = 0 := by
  rw [recallSlotWrite_coordinate]
  apply Finset.sum_eq_zero
  intro c hc
  have hn : i ≠ recallSlotIndex offset c := by
    intro he
    have hv := congrArg Fin.val he
    change i.val = offset.val + c.val at hv
    have hbound := c.isLt
    omega
  simp only [ite_eq_right hn, mul_zero]

example : (0 : Fin 64).val < (1 : Fin 57).val ∨
    (1 : Fin 57).val + 8 ≤ (0 : Fin 64).val := by decide

/-- Reading the slot matrix selects its actual residual coordinate.
Source: the transposed matrix and all eight distinct output unit coordinates. -/
theorem recallSlotRead_at (offset : Fin 57) (x : EucSpace 64) (c : Fin 8) :
    recallSlotRead offset x c = x (recallSlotIndex offset c) := by
  simp only [recallSlotRead, sum_apply, ContinuousLinearMap.smulRight_apply,
    innerSL_apply_apply, EuclideanSpace.inner_single_left, map_one, one_mul,
    WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_smul, Pi.smul_apply, PiLp.single_apply,
    smul_eq_mul]
  rw [Fintype.sum_eq_single c]
  · simp
  · intro j hj
    simp only [ite_eq_right (Ne.symm hj), mul_zero]

/-- The real coordinate matrices round-trip a compact code exactly.
Source: evaluated insertion and extraction coordinates, with no dropped input coordinate. -/
theorem recallSlotRead_write (offset : Fin 57) (x : EucSpace 8) :
    recallSlotRead offset (recallSlotWrite offset x) = x := by
  ext c
  rw [recallSlotRead_at, recallSlotWrite_at]

/-- Inner products of inserted codes are unchanged in the full original residual dimension.
Source: the actual rank-one insertion matrix, not an assumed abstract isometry. -/
theorem recallSlotWrite_inner (offset : Fin 57) (x y : EucSpace 8) :
    inner (𝕜 := ℝ) (recallSlotWrite offset x) (recallSlotWrite offset y) = inner (𝕜 := ℝ) x y := by
  rw [recallSlotWrite_apply offset x]
  simp only [sum_inner, real_inner_smul_left]
  simp only [EuclideanSpace.inner_single_left, map_one, one_mul, recallSlotWrite_at]
  rw [PiLp.inner_apply]
  simp only [RCLike.inner_apply, conj_trivial, mul_comm]

/-- The insertion has exact norm preservation, including the zero code.
Source: its evaluated real inner product and nonnegative actual Euclidean norms. -/
theorem recallSlotWrite_norm (offset : Fin 57) (x : EucSpace 8) : ‖recallSlotWrite offset x‖ = ‖x‖ := by
  have hs := recallSlotWrite_inner offset x x
  rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq] at hs
  nlinarith [norm_nonneg (recallSlotWrite offset x), norm_nonneg x]

/-- Disjoint actual intervals have zero inner product for every pair of input codes.
Source: the explicit insertion support, so key and value slots can coexist without cross-talk. -/
theorem recallSlotWrite_disjoint (a b : Fin 57) (horder : a.val + 8 ≤ b.val) (x y : EucSpace 8) :
    inner (𝕜 := ℝ) (recallSlotWrite a x) (recallSlotWrite b y) = 0 := by
  rw [recallSlotWrite_apply a x]
  simp only [sum_inner, real_inner_smul_left]
  apply Finset.sum_eq_zero
  intro c hc
  rw [EuclideanSpace.inner_single_left, recallSlotWrite_outside b y _ (Or.inl (by
    change a.val + c.val < b.val
    have hh := c.isLt
    omega))]
  simp

example : (1 : Fin 57).val + 8 ≤ (9 : Fin 57).val := by decide

/-- A slot read annihilates an isolated residual axis outside the interval.
Source: the actual transposed coordinate matrix, not a promised projection property. -/
theorem recallSlotRead_unit_outside (offset : Fin 57) (i : Fin 64)
    (hout : i.val < offset.val ∨ offset.val + 8 ≤ i.val) :
    recallSlotRead offset (EuclideanSpace.single i 1) = 0 := by
  ext c
  rw [recallSlotRead_at]
  have hn : recallSlotIndex offset c ≠ i := by
    intro he
    have hv := congrArg Fin.val he
    change offset.val + c.val = i.val at hv
    have hb := c.isLt
    omega
  simp only [PiLp.single_apply, ite_eq_right hn, PiLp.zero_apply]

example : (0 : Fin 64).val < (1 : Fin 57).val ∨
    (1 : Fin 57).val + 8 ≤ (0 : Fin 64).val := by decide

/-- Two disjoint actual code slots read as zero across one another in either order.
Source: evaluated extraction and insertion coordinates, including adjacent slot boundaries. -/
theorem recallSlotRead_disjoint (a b : Fin 57)
    (hsep : a.val + 8 ≤ b.val ∨ b.val + 8 ≤ a.val) (x : EucSpace 8) :
    recallSlotRead a (recallSlotWrite b x) = 0 := by
  ext c
  rw [recallSlotRead_at, recallSlotWrite_outside b x _ (by
    change a.val + c.val < b.val ∨ b.val + 8 ≤ a.val + c.val
    have hc := c.isLt
    omega), PiLp.zero_apply]

example : (1 : Fin 57).val + 8 ≤ (9 : Fin 57).val ∨
    (9 : Fin 57).val + 8 ≤ (1 : Fin 57).val := by decide

end Transformer.GPTMini.Semantics
