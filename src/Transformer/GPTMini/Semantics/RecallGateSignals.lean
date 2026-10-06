import Transformer.GPTMini.Semantics.RecallGateBounds

/-!
# Derive the gate margins from the actual raw first residual

Source: MQAR's table boundary at cbafbe9 and the original shared
embedding/attention/FFN at f11b6e2. The table contains 2P positions
after initial BOS. Values inside that boundary have a positive gate
margin, while values after it have a negative one. The position
signal used here is computed by the real simultaneous BOS head.

All raw keys and BOS are excluded by their protected type flags;
this also handles the initial position without assuming its XSA
marker equals reciprocal-prefix mass. The hypotheses describe raw
IDs and indices, not an encoded table indicator or desired gate.
Coupling this raw alphabet and boundary to the complete Basis parser
and proving the second-block selection remain later obligations.
-/

namespace Transformer.GPTMini.Semantics

/-- No checked actual value ID can be the initial reserved BOS ID.
Source: the original disjoint intervals, with values beginning at 292 and BOS=1. -/
theorem recallValueId_ne_bos (symbol : Fin 256) :
    recallValueId symbol ≠ (⟨1, by decide⟩ : Fin recallConfig.vocab_size) := by
  intro he
  have hv := congrArg Fin.val he
  change 292 + symbol.val = 1 at hv
  omega

/-- A raw value position is distinct from the raw BOS position without an assumed encoded phase.
Source: checked actual token IDs and the proved disjoint value/BOS vocabulary intervals. -/
theorem recall_raw_value_ne_first {T : ℕ} (tokens : Fin T → Fin recallConfig.vocab_size)
    (first i : Fin T) (value : Fin 256)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hvalue : tokens i = recallValueId value) : i ≠ first := by
  intro he
  rw [he, hbos] at hvalue
  exact recallValueId_ne_bos value hvalue.symm

example : (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
    ⟨1, by decide⟩ ∧ (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 1 =
    recallValueId 0 := by simp

/-- Any actual nonvalue-type residual has a sufficient negative gate margin.
Source: derived real constant/type channels and the complete raw marker bound, including initial BOS. -/
theorem recallGateForm_nonvalue (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (hflag : recallRawEmbedding (tokens i) 36 = 0) :
    recallGateForm P (recallFirstState eps alpha positions tokens i) ≤ -recallTableMargin P := by
  rw [recallGateForm_apply, recallFirstState_constant,
    recallFirstState_protected eps alpha positions tokens i 36 (by decide), hflag]
  have hm := recallFirstState_marker_le eps alpha heps positions tokens i
  linarith [recallTableMargin_le_threshold P]

example : (0 : ℝ) ≤ 1 / 100000 ∧ recallRawEmbedding (recallKeyId 0) 36 = 0 := by
  refine ⟨by norm_num, ?_⟩
  rw [recallRawEmbedding_key]
  exact (recallKeyEmbedding_flags 0).2.2.1

/-- Every actual query/key is excluded by the real gate, independently of copied-key contents.
Source: the original key ID interval and its derived protected value-type flag. -/
theorem recallGateForm_key (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (symbol : Fin 256) (htoken : tokens i = recallKeyId symbol) :
    recallGateForm P (recallFirstState eps alpha positions tokens i) ≤ -recallTableMargin P := by
  apply recallGateForm_nonvalue P eps alpha heps positions tokens i
  rw [htoken, recallRawEmbedding_key]
  exact (recallKeyEmbedding_flags symbol).2.2.1

example : (0 : ℝ) ≤ 0 ∧ (fun _ : Fin 1 => recallKeyId 0) 0 = recallKeyId 0 := by
  exact ⟨by norm_num, rfl⟩

/-- Actual BOS is excluded even at its own initial position.
Source: the true reserved embedding and marker-head bound, without an assumed later-position formula. -/
theorem recallGateForm_bos (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (htoken : tokens i = (⟨1, by decide⟩ : Fin recallConfig.vocab_size)) :
    recallGateForm P (recallFirstState eps alpha positions tokens i) ≤ -recallTableMargin P := by
  apply recallGateForm_nonvalue P eps alpha heps positions tokens i
  rw [htoken, recallRawEmbedding_bos]
  exact recallReservedEmbedding_flags.2.2.1

example : (0 : ℝ) ≤ 0 ∧
    (fun _ : Fin 1 => (⟨1, by decide⟩ : Fin recallConfig.vocab_size)) 0 = ⟨1, by decide⟩ := by
  exact ⟨by norm_num, rfl⟩

/-- At a later actual value, the ordinary matrix row is exactly the internally computed marker minus cutoff.
Source: raw BOS/alphabet IDs, true causal marker head and preserved actual value/constant coordinates. -/
theorem recallGateForm_value (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hne : i ≠ first)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens i = recallValueId value) :
    recallGateForm P (recallFirstState eps alpha positions tokens i) =
      1 / ((i.val + 1 : ℕ) : ℝ) - recallTableThreshold P := by
  rw [recallGateForm_apply, recallFirstState_constant,
    recallFirstState_value_flag eps alpha positions tokens i value hvalue]
  unfold recallFirstState
  rw [recall_first_residual_marker eps alpha heps positions tokens first i hfirst hne hbos hrange]
  ring

example : (0 : ℝ) ≤ 0 ∧ (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) ∧
    (if (1 : Fin 2) = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId 0 := by
  refine ⟨by norm_num, by decide, by decide, by simp, ?_, by simp⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

/-- An actual value inside the table has a strictly separated positive gate margin.
Source: true raw marker formula and the derived midpoint separation at the fixed table boundary. -/
theorem recallGateForm_inside (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hne : i ≠ first)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    recallTableMargin P ≤ recallGateForm P (recallFirstState eps alpha positions tokens i) := by
  rw [recallGateForm_value P eps alpha heps positions tokens first i hfirst hne hbos hrange value hvalue]
  exact recallTableThreshold_inside P i.val hinside

example : (0 : ℝ) ≤ 0 ∧ (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) ∧
    (if (1 : Fin 2) = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId 0 ∧
    (1 : Fin 2).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by decide, by decide, by simp, ?_, by simp, by decide⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

/-- A post-table raw value has a strictly separated negative gate margin, excluding false writes.
Source: true causal BOS mass and the first-position-after-table midpoint separation. -/
theorem recallGateForm_after (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hne : i ≠ first)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens i = recallValueId value) (hafter : 2 * P < i.val) :
    recallGateForm P (recallFirstState eps alpha positions tokens i) ≤ -recallTableMargin P := by
  rw [recallGateForm_value P eps alpha heps positions tokens first i hfirst hne hbos hrange value hvalue]
  exact recallTableThreshold_after P i.val hafter

example : (0 : ℝ) ≤ 0 ∧ (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) ∧
    (if (1 : Fin 2) = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId 0 ∧
    2 * (0 : ℕ) < (1 : Fin 2).val := by
  refine ⟨by norm_num, by decide, by decide, by simp, ?_, by simp, by decide⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

end Transformer.GPTMini.Semantics
