/-
# Attention retrieves an earlier unique key

Arora et al., arXiv:2312.04927v1, §4, Proposition `prop: attention-ar`,
and Appendix Proposition `prop: app-attention`.  The appendix combines an
adjacent key and value in its first layer and uses one-hot inner products in
its second layer.  Its displayed second layer has no causal mask and the
proof assumes there is just one matching key.  Here the mask and unique-key
condition are explicit.  This module proves the retrieval step after the
fixed local key/value alignment; it does not assert the paper's full
two-layer cost bound.
-/

import Transformer.Zoology.Section3_MQAR

open scoped BigOperators

namespace Transformer.Zoology

/-- Standard one-hot encoding of a token.  Source: Appendix
`rem: attn-input`. -/
def oneHot {c : ℕ} (a : Fin c) (t : Fin c) : ℝ :=
  if t = a then 1 else 0

/-- One-hot query-key products are exact equality indicators.
Source: Appendix Proposition `prop: app-attention`, computation following
equation `eq: shifted-value`. -/
theorem oneHot_inner {c : ℕ} (q k : Fin c) :
    (∑ t : Fin c, oneHot q t * oneHot k t) =
      if q = k then 1 else 0 := by
  classical
  by_cases h : q = k
  · subst k
    simp [oneHot]
  · simp [oneHot, h, eq_comm]

/-- All keys in the sequence are distinct.  This is the assumption used by
the synthetic generator when it subselects key-value pairs and is needed by
the appendix's equality-score lookup.  Source: Appendix `app:synthetic`,
Procedure `alg:synthetic`, and Appendix Proposition `prop: app-attention`. -/
def UniqueKeys {n c : ℕ} (x : MQARInstance n c) : Prop :=
  Function.Injective x.key

/-- Unique keys are possible when the vocabulary has at least as many
elements as the sequence of key-value pairs. -/
example : UniqueKeys ({
    key := id
    value := id
    query := id
  } : MQARInstance 2 2) := by
  intro i j h
  exact h

/-- The unnormalized attention output after a fixed local shift has placed
each value at its key's position.  The causal indicator restricts the sum to
earlier keys.  The source's displayed `QKᵀV` omits this mask; without one,
future matches would produce answers before their key-value pair appeared.
Source: Appendix Proposition `prop: app-attention`, second-layer computation,
with that causal correction. -/
def pairedAttentionOutput {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) : ℝ :=
  ∑ j : Fin n,
    (if j < i then oneHot (x.query i) (x.key j) else 0) *
      oneHot (x.value j) v

/-- The actual second-layer expression `QKᵀV` printed in the appendix,
after values have been aligned with their keys.  It contains no causal mask.
Source: Appendix Proposition `prop: app-attention`, final computation. -/
def unmaskedPairedAttentionOutput {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) : ℝ :=
  ∑ j : Fin n, oneHot (x.query i) (x.key j) * oneHot (x.value j) v

/-- Every coordinate of the output agrees with the value at the unique
matching earlier key.  Source: Appendix Proposition `prop: app-attention`,
corrected with explicit unique keys and a causal mask. -/
theorem paired_attention_exact {n c : ℕ} (x : MQARInstance n c)
    (hx : UniqueKeys x) (i j : Fin n) (hj : j < i)
    (hmatch : x.key j = x.query i) (v : Fin c) :
    pairedAttentionOutput x i v = oneHot (x.value j) v := by
  classical
  unfold pairedAttentionOutput
  rw [Finset.sum_eq_single j]
  · simp [oneHot, hj, hmatch]
  · intro k _ hk
    have hne : x.key k ≠ x.query i := by
      intro h
      exact hk (hx (h.trans hmatch.symm))
    simp [oneHot, hne]
  · intro h
    exact (h (Finset.mem_univ j)).elim

/-- When no earlier key matches, causal equality-score attention returns the
zero vector.  Source: Appendix `sec: intro-general-ar`, Setup, and Appendix
Proposition `prop: app-attention`, with an explicit causal mask. -/
theorem paired_attention_no_match {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (h : ∀ j : Fin n, j < i → x.key j ≠ x.query i)
    (v : Fin c) : pairedAttentionOutput x i v = 0 := by
  classical
  simp only [pairedAttentionOutput]
  apply Finset.sum_eq_zero
  intro j _
  by_cases hj : j < i
  · simp [hj, oneHot, h j hj]
  · simp [hj]

/-- The appendix's unmasked `QKᵀV` expression can read a future key.  At
the first query no earlier key exists, yet the output has a nonzero value
coordinate.  Thus the printed construction does not meet the unrestricted
causal MQAR definition; the mask in `pairedAttentionOutput` corrects this.
Source: Appendix Proposition `prop: app-attention`, second layer, compared
with §3 Definition `def: general-AR`. -/
theorem unmasked_attention_reads_future :
    ∃ x : MQARInstance 2 2,
      (¬ ∃ v : Fin 2, PriorAnswer x 0 v) ∧
      unmaskedPairedAttentionOutput x 0 1 = 1 := by
  let x : MQARInstance 2 2 := {
    key := id
    value := id
    query := fun _ => 1
  }
  refine ⟨x, ?_, ?_⟩
  · intro h
    obtain ⟨v, j, hj, _, _⟩ := h
    exact (Fin.not_lt_zero j) hj
  · norm_num [unmaskedPairedAttentionOutput, oneHot, Fin.sum_univ_two, x]

/-- Even a one-triple input makes the printed `QKᵀV` construction read the
current pair: `q₀ = k₀` returns `v₀` although MQAR asks for `j < 0` and
there is no prior pair.  This is why the corrected mask is strict at the
pair index, not merely causal at the raw token index.
Source: Appendix Proposition `prop: app-attention`, second layer. -/
theorem unmasked_attention_reads_current :
    ∃ x : MQARInstance 1 1,
      (¬ ∃ v : Fin 1, PriorAnswer x 0 v) ∧
      unmaskedPairedAttentionOutput x 0 0 = 1 := by
  let x : MQARInstance 1 1 := {
    key := fun _ => 0
    value := fun _ => 0
    query := fun _ => 0
  }
  refine ⟨x, ?_, ?_⟩
  · rintro ⟨v, j, hj, _, _⟩
    exact (Fin.not_lt_zero j) hj
  · simp [unmaskedPairedAttentionOutput, oneHot, x]

/-- The hypotheses of the corrected lookup theorem are simultaneously
satisfiable: a query at the second position finds the first key. -/
example : ∃ x : MQARInstance 2 2, UniqueKeys x ∧
    (0 : Fin 2) < 1 ∧ x.key 0 = x.query 1 := by
  refine ⟨{key := id, value := id, query := fun _ => 0}, ?_, by decide, rfl⟩
  intro i j h
  exact h

end Transformer.Zoology
