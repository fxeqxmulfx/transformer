import Transformer.GPTMini.Semantics.RecallEmbedding

/-!
# Read all raw recall channels with simultaneous ordinary projections

Source: the explicit raw embedding table in RecallEmbedding and the
bias-free QKV input coordinates at f11b6e2. A key-slot projection reads
all eight compact key coordinates and annihilates values and BOS. A
value-slot projection does the converse. The protected constant and
three type axes are evaluated directly in the actual residual vectors.

These identities are proved for every symbol in each original alphabet,
not assumed as a paired semantic array. They will feed the actual first
head and table gate. The table boundary is not present in these raw
token-local entries; it must still be derived from a causal BOS head.
Original RMSNorm scales every projection by the already proved common
multiplier. No correct logits or desired recall answer is a premise.
-/

namespace Transformer.GPTMini.Semantics

/-- The actual raw key projection retains its complete compact symbol and has no value component.
Source: the evaluated two disjoint eight-coordinate residual matrices. -/
theorem recallKeyEmbedding_reads (symbol : Fin 256) :
    recallSlotRead 1 (recallKeyEmbedding symbol) = recallSymbolCode symbol ∧
      recallSlotRead 9 (recallKeyEmbedding symbol) = 0 := by
  have h0a := recallSlotRead_unit_outside 1 0 (by decide)
  have h0b := recallSlotRead_unit_outside 9 0 (by decide)
  have h35a := recallSlotRead_unit_outside 1 35 (by decide)
  have h35b := recallSlotRead_unit_outside 9 35 (by decide)
  simp only [recallKeyEmbedding, recallUnit, map_add, h0a, h0b, h35a, h35b,
    recallSlotRead_write, recallSlotRead_disjoint 9 1 (by decide), zero_add, add_zero]
  simp

/-- Every raw value retains its own code and has exactly zero projected key value for first-head XSA.
Source: the simultaneous value slot, disjoint from the key slot and protected axes. -/
theorem recallValueEmbedding_reads (symbol : Fin 256) :
    recallSlotRead 1 (recallValueEmbedding symbol) = 0 ∧
      recallSlotRead 9 (recallValueEmbedding symbol) = recallSymbolCode symbol := by
  have h0a := recallSlotRead_unit_outside 1 0 (by decide)
  have h0b := recallSlotRead_unit_outside 9 0 (by decide)
  have h36a := recallSlotRead_unit_outside 1 36 (by decide)
  have h36b := recallSlotRead_unit_outside 9 36 (by decide)
  simp only [recallValueEmbedding, recallUnit, map_add, h0a, h0b, h36a, h36b,
    recallSlotRead_write, recallSlotRead_disjoint 1 9 (by decide), zero_add, add_zero]
  simp

/-- BOS/reserved entries have no projected key or value code.
Source: the fallback's independent coordinate 17 and protected constant/type axes. -/
theorem recallReservedEmbedding_reads :
    recallSlotRead 1 recallReservedEmbedding = 0 ∧ recallSlotRead 9 recallReservedEmbedding = 0 := by
  have h0a := recallSlotRead_unit_outside 1 0 (by decide)
  have h0b := recallSlotRead_unit_outside 9 0 (by decide)
  have h17a := recallSlotRead_unit_outside 1 17 (by decide)
  have h17b := recallSlotRead_unit_outside 9 17 (by decide)
  have h37a := recallSlotRead_unit_outside 1 37 (by decide)
  have h37b := recallSlotRead_unit_outside 9 37 (by decide)
  simp only [recallReservedEmbedding, recallUnit, map_add, map_smul, h0a, h0b, h17a, h17b,
    h37a, h37b, smul_zero, add_zero]
  simp

/-- At protected indices, an embedding's code contributes exactly zero.
Source: the actual slot support; the constant and type coordinates remain separate. -/
theorem recall_embedding_protected_coordinate (offset : Fin 57) (kind i : Fin 64)
    (code : EucSpace 8) (hout : i.val < offset.val ∨ offset.val + 8 ≤ i.val) :
    (recallUnit 0 + recallSlotWrite offset code + recallUnit kind) i =
      (if i = 0 then 1 else 0) + (if i = kind then 1 else 0) := by
  rw [PiLp.add_apply, PiLp.add_apply, recallSlotWrite_outside _ _ _ hout]
  simp only [recallUnit, PiLp.single_apply, add_zero]

example : (0 : Fin 64).val < (1 : Fin 57).val ∨
    (1 : Fin 57).val + 8 ≤ (0 : Fin 64).val := by decide

/-- Every key has the protected constant one and exactly the key-type indicator.
Source: actual coordinates 0 and 35..37, evaluated without a supplied semantic classification. -/
theorem recallKeyEmbedding_flags (symbol : Fin 256) :
    recallKeyEmbedding symbol 0 = 1 ∧ recallKeyEmbedding symbol 35 = 1 ∧
      recallKeyEmbedding symbol 36 = 0 ∧ recallKeyEmbedding symbol 37 = 0 := by
  unfold recallKeyEmbedding
  rw [recall_embedding_protected_coordinate 1 35 0 _ (by decide),
    recall_embedding_protected_coordinate 1 35 35 _ (by decide),
    recall_embedding_protected_coordinate 1 35 36 _ (by decide),
    recall_embedding_protected_coordinate 1 35 37 _ (by decide)]
  norm_num

/-- Every value has constant one and exactly its own protected type indicator.
Source: actual disjoint raw value embedding, including both positive and negative digit axes. -/
theorem recallValueEmbedding_flags (symbol : Fin 256) :
    recallValueEmbedding symbol 0 = 1 ∧ recallValueEmbedding symbol 35 = 0 ∧
      recallValueEmbedding symbol 36 = 1 ∧ recallValueEmbedding symbol 37 = 0 := by
  unfold recallValueEmbedding
  rw [recall_embedding_protected_coordinate 9 36 0 _ (by decide),
    recall_embedding_protected_coordinate 9 36 35 _ (by decide),
    recall_embedding_protected_coordinate 9 36 36 _ (by decide),
    recall_embedding_protected_coordinate 9 36 37 _ (by decide)]
  norm_num

/-- BOS/reserved entries activate only the third protected type.
Source: the explicit fallback table entry, not a whole-prefix BOS-count premise. -/
theorem recallReservedEmbedding_flags :
    recallReservedEmbedding 0 = 1 ∧ recallReservedEmbedding 35 = 0 ∧
      recallReservedEmbedding 36 = 0 ∧ recallReservedEmbedding 37 = 1 := by
  norm_num [recallReservedEmbedding, recallUnit, PiLp.add_apply, PiLp.smul_apply, PiLp.single_apply]

/-- All checked raw tokens provide a genuine protected constant channel.
Source: exhaustive raw vocabulary branches and their actual coordinate calculations. -/
theorem recallRawEmbedding_constant (token : Fin recallConfig.vocab_size) : recallRawEmbedding token 0 = 1 := by
  unfold recallRawEmbedding
  split_ifs
  · exact (recallKeyEmbedding_flags _).1
  · exact (recallValueEmbedding_flags _).1
  · exact recallReservedEmbedding_flags.1

/-- The genuine first prenorm's constant channel gives exactly its derived shared multiplier.
Source: the actual normalized raw table and evaluated protected coordinate zero. -/
theorem recallRawEmbedding_rms_constant (eps : ℝ) (token : Fin recallConfig.vocab_size) :
    rmsNormEps eps (recallRawEmbedding token) 0 = recallRawScale eps := by
  rw [recallRawEmbedding_rms, PiLp.smul_apply, recallRawEmbedding_constant]
  exact mul_one _

/-- Reading key and value slots after the true first RMSNorm scales both faithful projections equally.
Source: the complete raw norm proof and ordinary matrix linearity, for every vocabulary entry. -/
theorem recallRawEmbedding_rms_reads (eps : ℝ) (token : Fin recallConfig.vocab_size) :
    recallSlotRead 1 (rmsNormEps eps (recallRawEmbedding token)) =
        recallRawScale eps • recallSlotRead 1 (recallRawEmbedding token) ∧
      recallSlotRead 9 (rmsNormEps eps (recallRawEmbedding token)) =
        recallRawScale eps • recallSlotRead 9 (recallRawEmbedding token) := by
  rw [recallRawEmbedding_rms, map_smul, map_smul]
  exact ⟨rfl, rfl⟩

/-- The actual raw table has no precomputed copied key or table-position feature.
Source: unused coordinates 18..34 lie outside both raw code slots and all protected axes. -/
theorem recallRawEmbedding_fresh (token : Fin recallConfig.vocab_size) (i : Fin 64)
    (hi : 18 ≤ i.val ∧ i.val < 35) : recallRawEmbedding token i = 0 := by
  have hn0 : i ≠ 0 := by intro he; have hv := congrArg Fin.val he; change i.val = 0 at hv; omega
  have hn35 : i ≠ 35 := by intro he; have hv := congrArg Fin.val he; change i.val = 35 at hv; omega
  have hn36 : i ≠ 36 := by intro he; have hv := congrArg Fin.val he; change i.val = 36 at hv; omega
  have hn37 : i ≠ 37 := by intro he; have hv := congrArg Fin.val he; change i.val = 37 at hv; omega
  have hn17 : i ≠ 17 := by intro he; have hv := congrArg Fin.val he; change i.val = 17 at hv; omega
  unfold recallRawEmbedding
  split_ifs
  · rw [recallKeyEmbedding, recall_embedding_protected_coordinate 1 35 i _ (Or.inr (by simp; omega))]
    simp only [ite_eq_right hn0, ite_eq_right hn35, add_zero]
  · rw [recallValueEmbedding, recall_embedding_protected_coordinate 9 36 i _ (Or.inr (by simp; omega))]
    simp only [ite_eq_right hn0, ite_eq_right hn36, add_zero]
  · simp only [recallReservedEmbedding, recallUnit, PiLp.add_apply, PiLp.smul_apply, PiLp.single_apply,
      ite_eq_right hn0, ite_eq_right hn17, ite_eq_right hn37, smul_eq_mul, mul_zero, add_zero]

example : 18 ≤ (26 : Fin 64).val ∧ (26 : Fin 64).val < 35 := by decide

end Transformer.GPTMini.Semantics
