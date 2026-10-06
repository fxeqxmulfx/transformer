import Transformer.GPTMini.Semantics.RecallMarkerWeights

/-!
# Derive the table-position signal from actual raw BOS tokens

Source: MQAR's validated raw alphabet at cbafbe9 and the simultaneous
original fused QKV/headAt/W_o at f11b6e2. Only initial BOS has the
reserved-type flag on a legal recall prefix. Every later key or value
has flag zero. The true marker head therefore contains reciprocal-
prefix mass at every later position, including intermediate positions
of a larger array. The genuine attention residual writes this mass
to coordinate 26, initially zero in all raw embedding entries.

The hypotheses below specify only raw IDs and their position in the
input. They do not supply a marker array, a prefix length to the model,
paired key/value features or desired output scores. A table cutoff can
subsequently be an ordinary fixed linear form in coordinates 0/26/36.
The actual FFN cutoff and complete retrieval/readout remain separate.
-/

namespace Transformer.GPTMini.Semantics

/-- The actual reserved/BOS value direction has exact unit norm.
Source: the marker head's genuine first V coordinate in the original sixteen-coordinate layout. -/
theorem recallMarkerDirection_norm : ‖recallMarkerDirection‖ = 1 := by
  unfold recallMarkerDirection
  rw [PiLp.norm_single]
  norm_num

/-- The actual initial BOS entry activates the raw marker flag exactly once.
Source: original BOS=1 and the evaluated reserved embedding's type axis. -/
theorem recall_raw_bos_marker : recallRawEmbedding ⟨1, by decide⟩ 37 = 1 := by
  rw [recallRawEmbedding_bos]
  exact recallReservedEmbedding_flags.2.2.2

/-- Every actual key and value symbol has zero raw marker flag.
Source: disjoint original vocabulary intervals and evaluated protected type axes. -/
theorem recall_raw_symbol_marker (symbol : Fin 256) :
    recallRawEmbedding (recallKeyId symbol) 37 = 0 ∧
      recallRawEmbedding (recallValueId symbol) 37 = 0 := by
  rw [recallRawEmbedding_key, recallRawEmbedding_value]
  exact ⟨(recallKeyEmbedding_flags symbol).2.2.2, (recallValueEmbedding_flags symbol).2.2.2⟩

/-- The true raw marker values are the single initial indicator, derived entirely from checked integer IDs.
Source: the validated recall alphabet excludes every other reserved token after BOS. -/
theorem recall_raw_marker_array {T : ℕ} (tokens : Fin T → Fin recallConfig.vocab_size) (first : Fin T)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol) :
    (fun j => recallRawEmbedding (tokens j) 37 • recallMarkerDirection) =
      (fun j => if j = first then recallMarkerDirection else 0) := by
  funext j
  by_cases hj : j = first
  · subst j
    rw [hbos, recall_raw_bos_marker, one_smul, ite_eq_left (rfl : first = first)]
  · obtain ⟨symbol, hs⟩ := hrange j hj
    rcases hs with hs | hs
    · rw [hs, (recall_raw_symbol_marker symbol).1, zero_smul, ite_eq_right hj]
    · rw [hs, (recall_raw_symbol_marker symbol).2, zero_smul, ite_eq_right hj]

example : (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallKeyId 0) 0 =
    ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
      (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallKeyId 0) = recallKeyId symbol ∨
      (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallKeyId 0) = recallValueId symbol) := by
  refine ⟨by simp, ?_⟩
  intro j hj
  exact ⟨0, Or.inl (by simp only [ite_eq_right hj])⟩

/-- The actual original marker head has exact reciprocal-prefix mass on every legal raw-alphabet position after BOS.
Source: true prenorm/fused QKV/XSA head and the derived raw marker array, even in the presence of future tokens. -/
theorem recall_raw_marker_head_exact (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hne : i ≠ first)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol) :
    headAt recallConfig (recallFirstAttention eps alpha) eps positions
        (fun j => recallRawEmbedding (tokens j)) 1 i =
      (1 / ((i.val + 1 : ℕ) : ℝ)) • recallMarkerDirection := by
  rw [recall_marker_head eps alpha heps]
  change attentionHead recallConfig 0 eps (fun _ => 0) (fun _ => 0)
    (fun j => recallRawEmbedding (tokens j) 37 • recallMarkerDirection) positions i = _
  rw [recall_raw_marker_array tokens first hbos hrange]
  exact recall_initial_marker_head recallConfig eps positions first i hfirst hne _

example : (0 : ℝ) ≤ 1 / 100000 ∧ (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) := by
  refine ⟨by norm_num, by decide, by decide, by simp, ?_⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

/-- Original W_o reads exactly the marker head's first coordinate into the true position channel.
Source: evaluated head merge/output matrix and disjoint copy slot support. -/
theorem recallFirstOutput_marker (heads : Fin recallConfig.n_heads → EucSpace recallConfig.head_dim) :
    recallFirstOutput (headMerge recallConfig heads) 26 = heads 1 ⟨0, by decide⟩ := by
  rw [recallFirstOutput_merge, PiLp.add_apply, recallSlotWrite_outside 18 _ 26 (by decide), zero_add]
  simp only [PiLp.smul_apply, recallUnit, PiLp.single_apply, ite_true, smul_eq_mul, mul_one]

/-- The exact internally computed position marker is present in the actual first pre-FFN residual.
Source: original full attention sublayer/head merge/W_o and the initially empty raw coordinate 26. -/
theorem recall_first_residual_marker (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hne : i ≠ first)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol) :
    (recallRawEmbedding (tokens i) +
      attnSubLayer recallConfig (recallFirstAttention eps alpha) eps positions
        (fun j => recallRawEmbedding (tokens j)) i) 26 = 1 / ((i.val + 1 : ℕ) : ℝ) := by
  rw [PiLp.add_apply, recallRawEmbedding_fresh (tokens i) 26 (by decide), zero_add, attnSubLayer_eq_heads]
  change recallFirstOutput (headMerge recallConfig _) 26 = _
  rw [recallFirstOutput_marker, recall_raw_marker_head_exact eps alpha heps positions tokens first i hfirst hne hbos hrange]
  simp only [PiLp.smul_apply, recallMarkerDirection, PiLp.single_apply, ite_true, smul_eq_mul, mul_one]

example : (0 : ℝ) ≤ 1 / 100000 ∧ (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) := by
  refine ⟨by norm_num, by decide, by decide, by simp, ?_⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

/-- This true residual marker is positive and uniformly bounded throughout the complete recall context.
Source: its actual computation and the real cap 64, without an assumed position feature. -/
theorem recall_first_residual_marker_bounds (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ} (hT : T ≤ 64)
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hne : i ≠ first)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol) :
    let x := recallRawEmbedding (tokens i) +
      attnSubLayer recallConfig (recallFirstAttention eps alpha) eps positions
        (fun j => recallRawEmbedding (tokens j)) i
    1 / 64 ≤ x 26 ∧ x 26 ≤ (1 / 2 : ℝ) := by
  dsimp only
  rw [recall_first_residual_marker eps alpha heps positions tokens first i hfirst hne hbos hrange]
  exact ⟨recall_marker_mass_ge_context hT i, recall_marker_mass_le_half first i hfirst hne⟩

example : (0 : ℝ) ≤ 1 / 100000 ∧ (2 : ℕ) ≤ 64 ∧ (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) := by
  refine ⟨by norm_num, by decide, by decide, by decide, by simp, ?_⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

end Transformer.GPTMini.Semantics
