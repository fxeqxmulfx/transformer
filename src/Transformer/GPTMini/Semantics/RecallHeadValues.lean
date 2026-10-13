import Transformer.GPTMini.Semantics.RecallQKV

/-!
# Genuine head values retain the compact raw code and its norm

Source: the original sixteen-coordinate V head at f11b6e2 and the
simultaneous raw MQAR embedding. The first eight coordinates retain
all code information; the second half is zero. These formulas concern
values, so no rotary transformation is inserted into this layout.

The actual raw key projections have norm two. Values and reserved tokens
have zero first-head self-values. The diameter of every raw value array
is therefore bounded by four without a prepared semantic-value premise.
Reading the copied eight coordinates cannot increase the head's error.
These are the quantitative input facts needed by the original finite
softmax/XSA predecessor estimate, not a new exact shift operation.
The bounds cover both sign choices in every categorical digit and all
reserved entries. They apply before any table gating or latest-write
retrieval, so those subsequent computations remain separate obligations.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- Extract the actual first eight value coordinates without introducing any task operation.
Source: the original V layout used by recallFirstQKV_value. -/
noncomputable def recallHeadRead (x : EucSpace 16) : EucSpace 8 :=
  (EuclideanSpace.equiv (Fin 8) ℝ).symm fun c => x ⟨c.val, by have hc := c.isLt; omega⟩

/-- The actual value layout commutes with every ordinary scalar multiplication.
Source: its eight retained coordinates and eight zero coordinates. -/
theorem recallHeadValue_smul (a : ℝ) (code : EucSpace 8) :
    recallHeadValue (a • code) = a • recallHeadValue code := by
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallHeadValue (a • code) c = a * recallHeadValue code c
  simp only [recallHeadValue_coordinate, PiLp.smul_apply, smul_eq_mul]
  split_ifs <;> simp only [mul_zero]

/-- A zero compact code gives the zero actual head value, including the unused half.
Source: the evaluated sixteen-coordinate V layout. -/
theorem recallHeadValue_zero : recallHeadValue 0 = 0 := by
  apply (EuclideanSpace.equiv (Fin 16) ℝ).injective
  funext c
  change recallHeadValue 0 c = 0
  simp only [recallHeadValue_coordinate, PiLp.zero_apply, dite_eq_ite, ite_self]

/-- The compact head value occupies exactly the original first half-pair coordinates.
Source: all eight ropeFst indices equal zero through seven; values themselves are not rotated. -/
theorem recallHeadValue_fst (code : EucSpace 8) (p : Fin 8) :
    recallHeadValue code (ropeFst 16 p) = code p := by
  have hf : ∀ p : Fin 8, (ropeFst 16 p).val = p.val := by decide
  rw [recallHeadValue_coordinate]
  have hp : (ropeFst 16 p).val < 8 := by rw [hf]; exact p.isLt
  rw [dite_eq_left hp]
  apply congrArg (fun c : Fin 8 => code c)
  apply Fin.ext
  exact hf p

/-- Every actual coordinate in the second half of this V layout is zero.
Source: ropeSnd adds eight to the coordinate, outside the compact insertion interval. -/
theorem recallHeadValue_snd (code : EucSpace 8) (p : Fin 8) :
    recallHeadValue code (ropeSnd 16 p) = 0 := by
  have hs : ∀ p : Fin 8, (ropeSnd 16 p).val = p.val + 8 := by decide
  rw [recallHeadValue_coordinate, dite_eq_right (by rw [hs]; omega)]

/-- The genuine head insertion preserves squared norm, with all sixteen coordinates accounted for.
Source: the original half-pair partition of Fin 16 and the true Euclidean sums. -/
theorem recallHeadValue_norm_sq (code : EucSpace 8) : ‖recallHeadValue code‖ ^ 2 = ‖code‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq]
  change (∑ i : Fin 16, ‖recallHeadValue code i‖ ^ 2) = _
  rw [sum_split 16]
  change ((∑ p : Fin 8, ‖recallHeadValue code (ropeFst 16 p)‖ ^ 2) +
    ∑ p : Fin 8, ‖recallHeadValue code (ropeSnd 16 p)‖ ^ 2) +
    (∑ p : Fin 0, ‖recallHeadValue code ((ropeSplit 16).symm (Sum.inr p))‖ ^ 2) = _
  simp only [recallHeadValue_fst, recallHeadValue_snd, norm_zero, zero_pow (by decide : 2 ≠ 0),
    Finset.sum_const_zero, Finset.univ_eq_empty, Finset.sum_empty, add_zero]
  exact (EuclideanSpace.norm_sq_eq code).symm

/-- The actual sixteen-coordinate value has exactly the original compact-code norm.
Source: its evaluated squared norm and genuine nonnegative norms. -/
theorem recallHeadValue_norm (code : EucSpace 8) : ‖recallHeadValue code‖ = ‖code‖ := by
  nlinarith [recallHeadValue_norm_sq code, norm_nonneg (recallHeadValue code), norm_nonneg code]

/-- Reading an inserted value retains all eight compact coordinates exactly.
Source: the actual code insertion and coordinate extraction. -/
theorem recallHeadRead_value (code : EucSpace 8) : recallHeadRead (recallHeadValue code) = code := by
  ext c
  change recallHeadValue code (⟨c.val, by have hc := c.isLt; omega⟩ : Fin 16) = code c
  rw [recallHeadValue_coordinate, dite_eq_left c.isLt]

/-- Extracting the eight copied coordinates never amplifies a genuine head vector.
Source: the original norm_comp_injective_le coordinate-reading theorem. -/
theorem recallHeadRead_norm_le (x : EucSpace 16) : ‖recallHeadRead x‖ ≤ ‖x‖ := by
  apply norm_comp_injective_le
  intro a b he
  apply Fin.ext
  exact congrArg (fun i : Fin 16 => i.val) he

/-- The ordinary extraction also cannot amplify an error between two actual head outputs.
Source: its evaluated linear subtraction and coordinate norm bound. -/
theorem recallHeadRead_error (x y : EucSpace 16) :
    ‖recallHeadRead x - recallHeadRead y‖ ≤ ‖x - y‖ := by
  have hs : recallHeadRead (x - y) = recallHeadRead x - recallHeadRead y := by
    ext c
    rfl
  rw [← hs]
  exact recallHeadRead_norm_le _

/-- A true raw key ID supplies the complete norm-two head value.
Source: the actual raw table and exact compact-code readback, with no paired-array hypothesis. -/
theorem recall_raw_key_value (symbol : Fin 256) :
    recallHeadValue (recallSlotRead 1 (recallRawEmbedding (recallKeyId symbol))) =
      recallHeadValue (recallSymbolCode symbol) := by
  rw [recallRawEmbedding_key, (recallKeyEmbedding_reads symbol).1]

/-- Every true raw value token has zero first-head self-value, so XSA does not erase its predecessor copy.
Source: the simultaneous raw table's disjoint key/value projections. -/
theorem recall_raw_value_self_zero (symbol : Fin 256) :
    recallHeadValue (recallSlotRead 1 (recallRawEmbedding (recallValueId symbol))) = 0 := by
  rw [recallRawEmbedding_value, (recallValueEmbedding_reads symbol).1, recallHeadValue_zero]

/-- The actual projected value of each raw key has exact norm two.
Source: the original raw-ID projection, evaluated head insertion and compact four-pair norm. -/
theorem recall_raw_key_value_norm (symbol : Fin 256) :
    ‖recallHeadValue (recallSlotRead 1 (recallRawEmbedding (recallKeyId symbol)))‖ = 2 := by
  rw [recall_raw_key_value, recallHeadValue_norm]
  exact recallCode_norm _

/-- The true BOS entry also has zero first-head value; it cannot contribute a spurious raw key.
Source: the independently placed reserved/BOS embedding and actual raw key read. -/
theorem recall_raw_bos_self_zero :
    recallHeadValue (recallSlotRead 1 (recallRawEmbedding ⟨1, by decide⟩)) = 0 := by
  rw [recallRawEmbedding_bos, recallReservedEmbedding_reads.1, recallHeadValue_zero]

/-- Every checked raw token's actual first-head value has norm at most two.
Source: the complete three-branch raw table, with norm-two keys and zero other entries. -/
theorem recall_raw_head_value_bound (token : Fin recallConfig.vocab_size) :
    ‖recallHeadValue (recallSlotRead 1 (recallRawEmbedding token))‖ ≤ 2 := by
  unfold recallRawEmbedding
  split_ifs
  · rw [(recallKeyEmbedding_reads _).1, recallHeadValue_norm, recallSymbolCode, recallCode_norm]
  · rw [(recallValueEmbedding_reads _).1, recallHeadValue_zero, norm_zero]; norm_num
  · rw [recallReservedEmbedding_reads.1, recallHeadValue_zero, norm_zero]; norm_num

/-- The actual first-head values of any two raw vocabulary entries have diameter at most four.
Source: the derived complete raw value bound and the real norm triangle inequality. -/
theorem recall_raw_head_value_diameter (a b : Fin recallConfig.vocab_size) :
    ‖recallHeadValue (recallSlotRead 1 (recallRawEmbedding a)) -
      recallHeadValue (recallSlotRead 1 (recallRawEmbedding b))‖ ≤ 4 := by
  have hn := norm_sub_le
    (recallHeadValue (recallSlotRead 1 (recallRawEmbedding a)))
    (recallHeadValue (recallSlotRead 1 (recallRawEmbedding b)))
  linarith [recall_raw_head_value_bound a, recall_raw_head_value_bound b]

end Transformer.GPTMini.Semantics
