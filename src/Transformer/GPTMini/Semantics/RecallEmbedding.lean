import Transformer.GPTMini.Semantics.RecallSlots

/-!
# Simultaneous raw key, value and type embeddings for recall

Source: MQAR's original integer intervals at cbafbe9 and GPTMini's
64-coordinate embedding table/RMSNorm at f11b6e2. Key IDs 36..291
and value IDs 292..547 have distinct eight-coordinate code slots.
A protected constant and three type axes distinguish keys, values
and reserved tokens. Reserved entries share a fixed fallback code;
the validated MQAR grammar uses only BOS from that group.

Every vocabulary entry has exact squared norm six, so the true first
prenorm has a single derived multiplier. No token entry reads a prefix,
paired record, table boundary or desired answer. Copying neighboring
keys and gating the table are subsequent actual block computations.
These are given ordinary baseline weights, not a convexity assertion.
-/

namespace Transformer.GPTMini.Semantics

/-- An actual residual coordinate axis in the original width.
Source: the explicit recall layout; all axes remain ordinary model coordinates. -/
noncomputable def recallUnit (i : Fin 64) : EucSpace 64 := EuclideanSpace.single i 1

/-- Serialize an actual key symbol into its original checked vocabulary ID.
Source: MQAR's identityBase=36, symbols=256. -/
def recallKeyId (symbol : Fin 256) : Fin recallConfig.vocab_size :=
  ⟨36 + symbol.val, by have hs := symbol.isLt; change 36 + symbol.val < 548; omega⟩

/-- Serialize its corresponding value alphabet ID, without binding it to any key.
Source: the disjoint original MQAR value interval beginning at 292. -/
def recallValueId (symbol : Fin 256) : Fin recallConfig.vocab_size :=
  ⟨292 + symbol.val, by have hs := symbol.isLt; change 292 + symbol.val < 548; omega⟩

/-- A token-local key embedding: compact code, protected constant, and key type.
Source: the simultaneous new coordinates 0, 1..8 and 35 in the existing table. -/
noncomputable def recallKeyEmbedding (symbol : Fin 256) : EucSpace 64 :=
  recallUnit 0 + recallSlotWrite 1 (recallSymbolCode symbol) + recallUnit 35

/-- A token-local value embedding has its own code slot and its own type.
Source: coordinates 0, 9..16 and 36, disjoint from every key coordinate. -/
noncomputable def recallValueEmbedding (symbol : Fin 256) : EucSpace 64 :=
  recallUnit 0 + recallSlotWrite 9 (recallSymbolCode symbol) + recallUnit 36

/-- The reserved token group has the same norm and a separate type/marker.
Source: BOS is the only reserved token in a legal MQAR input; fallback occupies 17 and 37. -/
noncomputable def recallReservedEmbedding : EucSpace 64 :=
  recallUnit 0 + (2 : ℝ) • recallUnit 17 + recallUnit 37

/-- The complete ordinary embedding table interprets the actual raw integer IDs.
Source: cbafbe9's MQAR vocabulary, retaining all 548 checked entries and no context argument. -/
noncomputable def recallRawEmbedding (token : Fin recallConfig.vocab_size) : EucSpace 64 :=
  if h : 36 ≤ token.val ∧ token.val < 292 then
    recallKeyEmbedding ⟨token.val - 36, by omega⟩
  else if h : 292 ≤ token.val then
    recallValueEmbedding ⟨token.val - 292, by have ht := token.isLt; change token.val < 548 at ht; omega⟩
  else recallReservedEmbedding

/-- Inserting a code between the protected constant and a later type axis gives their exact total norm.
Source: the actual disjoint coordinate layout and evaluated insertion matrix. -/
theorem recall_code_type_norm_sq (offset : Fin 57) (kind : Fin 64)
    (hstart : 0 < offset.val) (hkind : offset.val + 8 ≤ kind.val) (code : EucSpace 8) :
    ‖recallUnit 0 + recallSlotWrite offset code + recallUnit kind‖ ^ 2 = 2 + ‖code‖ ^ 2 := by
  have hk : kind ≠ 0 := by
    intro he
    have hv := congrArg Fin.val he
    change kind.val = 0 at hv
    omega
  have hc : recallSlotWrite offset code 0 = 0 :=
    recallSlotWrite_outside _ _ _ (Or.inl hstart)
  have ht : recallSlotWrite offset code kind = 0 :=
    recallSlotWrite_outside _ _ _ (Or.inr hkind)
  rw [norm_add_sq_real, norm_add_sq_real, recallSlotWrite_norm, inner_add_left]
  simp [recallUnit, PiLp.norm_single, EuclideanSpace.inner_single_left,
    EuclideanSpace.inner_single_right, hc, ht, hk]
  ring

example : 0 < (1 : Fin 57).val ∧ (1 : Fin 57).val + 8 ≤ (35 : Fin 64).val := by decide

/-- Every actual key symbol has exact squared embedding norm six.
Source: four unit digit pairs, the protected constant and independent key type. -/
theorem recallKeyEmbedding_norm_sq (symbol : Fin 256) : ‖recallKeyEmbedding symbol‖ ^ 2 = 6 := by
  rw [recallKeyEmbedding, recall_code_type_norm_sq 1 35 (by decide) (by decide)]
  unfold recallSymbolCode
  rw [recallCode_norm_sq]
  norm_num

/-- Every actual value symbol has the same exact norm, simultaneously with the key alphabet.
Source: its independent slot and type, with the same compact four-pair code. -/
theorem recallValueEmbedding_norm_sq (symbol : Fin 256) : ‖recallValueEmbedding symbol‖ ^ 2 = 6 := by
  rw [recallValueEmbedding, recall_code_type_norm_sq 9 36 (by decide) (by decide)]
  unfold recallSymbolCode
  rw [recallCode_norm_sq]
  norm_num

/-- The reserved/BOS embedding has the same squared norm rather than an exceptional prenorm multiplier.
Source: three distinct real axes of amplitudes one, two and one. -/
theorem recallReservedEmbedding_norm_sq : ‖recallReservedEmbedding‖ ^ 2 = 6 := by
  rw [recallReservedEmbedding, norm_add_sq_real, norm_add_sq_real, inner_add_left]
  norm_num [recallUnit, norm_smul, PiLp.norm_single, real_inner_smul_left, real_inner_smul_right,
    EuclideanSpace.inner_single_left, PiLp.single_apply]

/-- The full checked raw table has the same norm at every entry, including reserved IDs.
Source: the three actual vocabulary branches, each proved separately above. -/
theorem recallRawEmbedding_norm_sq (token : Fin recallConfig.vocab_size) :
    ‖recallRawEmbedding token‖ ^ 2 = 6 := by
  unfold recallRawEmbedding
  split_ifs
  · exact recallKeyEmbedding_norm_sq _
  · exact recallValueEmbedding_norm_sq _
  · exact recallReservedEmbedding_norm_sq

/-- A raw key ID reads precisely its compact symbol embedding in the actual table.
Source: the original integer vocabulary interval, not a supplied semantic feature. -/
theorem recallRawEmbedding_key (symbol : Fin 256) :
    recallRawEmbedding (recallKeyId symbol) = recallKeyEmbedding symbol := by
  have hs := symbol.isLt
  have hr : 36 ≤ (recallKeyId symbol).val ∧ (recallKeyId symbol).val < 292 := by
    change 36 ≤ 36 + symbol.val ∧ 36 + symbol.val < 292
    omega
  rw [recallRawEmbedding, dite_eq_left hr]
  apply congrArg recallKeyEmbedding
  apply Fin.ext
  change 36 + symbol.val - 36 = symbol.val
  omega

/-- A raw value ID selects the distinct value slot for exactly its own symbol.
Source: the original value interval, simultaneously with all key entries. -/
theorem recallRawEmbedding_value (symbol : Fin 256) :
    recallRawEmbedding (recallValueId symbol) = recallValueEmbedding symbol := by
  have hr : ¬(36 ≤ (recallValueId symbol).val ∧ (recallValueId symbol).val < 292) := by
    change ¬(36 ≤ 292 + symbol.val ∧ 292 + symbol.val < 292)
    omega
  have hv : 292 ≤ (recallValueId symbol).val := by change 292 ≤ 292 + symbol.val; omega
  rw [recallRawEmbedding, dite_eq_right hr, dite_eq_left hv]
  apply congrArg recallValueEmbedding
  apply Fin.ext
  change 292 + symbol.val - 292 = symbol.val
  omega

/-- The actual BOS entry is the norm-matched reserved marker, independently of any prefix.
Source: the validated raw grammar starts with BOS=1. -/
theorem recallRawEmbedding_bos : recallRawEmbedding ⟨1, by decide⟩ = recallReservedEmbedding := by
  norm_num [recallRawEmbedding]

/-- The common first prenorm multiplier is derived from the true raw embedding norm.
Source: original RMSNorm.forward, with the unchanged shared-epsilon real convention. -/
noncomputable def recallRawScale (eps : ℝ) : ℝ := 8 / Real.sqrt (6 + 64 * eps)

/-- Every raw entry's actual prenorm equals the same multiplier times that entry.
Source: the evaluated RMSNorm and complete vocabulary norm proof, not an assumed scale. -/
theorem recallRawEmbedding_rms (eps : ℝ) (token : Fin recallConfig.vocab_size) :
    rmsNormEps eps (recallRawEmbedding token) = recallRawScale eps • recallRawEmbedding token := by
  simp only [rmsNormEps, recallRawEmbedding_norm_sq]
  have hsqrt : Real.sqrt 64 = 8 := by
    rw [show (64 : ℝ) = 8 ^ 2 from by norm_num]
    exact Real.sqrt_sq (by norm_num)
  change (Real.sqrt 64 / Real.sqrt (6 + 64 * eps)) • recallRawEmbedding token = _
  rw [hsqrt]
  rfl

/-- This actual prenorm multiplier is strictly positive for all nonnegative epsilons.
Source: the positive norm-six denominator, needed for an ordinary compensating QKV matrix. -/
theorem recallRawScale_pos (eps : ℝ) (heps : 0 ≤ eps) : 0 < recallRawScale eps := by
  unfold recallRawScale
  apply div_pos (by norm_num)
  apply Real.sqrt_pos.mpr
  linarith

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

end Transformer.GPTMini.Semantics
