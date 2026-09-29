/-
# Content-dependent convex routing on a causal MQAR prefix

Extension of arXiv:2211.11052v1, §3.1: a separate simplex program for
each query, using a shared learned metric. This is not its shared positional
matrix. The MQAR key/value alignment and unique-key condition are those of
arXiv:2312.04927v1, Appendix `prop: app-attention` and `app:synthetic`.
The null slot gives a zero output when no earlier key matches.
-/

import Transformer.ConvexRecall.Simplex
import Transformer.Zoology.Section4_PairedRecall

open scoped BigOperators

noncomputable section

namespace Transformer.ConvexRecall

open Transformer.Zoology

/-- Candidate key slots plus one null slot. -/
abbrev RecallSlot (n : ℕ) := Option (Fin n)

/-- Only strictly earlier pairs and the null slot can receive mass. -/
def allowedSlots {n : ℕ} (i : Fin n) : Set (RecallSlot n) :=
  fun slot => match slot with
  | none => True
  | some j => j < i

/-- Null cost `1/2`; key costs use the trainable content metric. -/
def recallCosts {n c b : ℕ} (code : Fin c → Code b)
    (w : MetricWeights b) (x : MQARInstance n c) (i : Fin n) : RecallSlot n → ℝ
  | none => 1 / 2
  | some j => metricCost w (code (x.query i)) (code (x.key j))

/-- The actual optimization problem for a query, with the causal mask. -/
def IsRecallOptimum {n c b : ℕ} (code : Fin c → Code b)
    (w : MetricWeights b) (x : MQARInstance n c) (i : Fin n)
    (a : RecallSlot n → ℝ) : Prop :=
  a ∈ simplexOn (allowedSlots i) ∧
    ∀ z ∈ simplexOn (allowedSlots i),
      routingObjective (recallCosts code w x i) a ≤
        routingObjective (recallCosts code w x i) z

/-- A unique matching earlier key beats every allowed alternative by at
least `1/2`, including the null slot. Extension of arXiv:2211.11052v1,
§3.1, for arXiv:2312.04927v1, Appendix `prop: app-attention`. -/
theorem matching_slot_gap {n c b : ℕ} (code : Fin c → Code b)
    (hc : Function.Injective code) (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (x : MQARInstance n c) (hx : UniqueKeys x)
    (i j : Fin n) (hmatch : x.key j = x.query i) :
    ∀ slot, slot ∈ allowedSlots i → slot ≠ some j →
      recallCosts code w x i (some j) + 1 / 2 ≤ recallCosts code w x i slot := by
  have he : recallCosts code w x i (some j) = 0 := by
    simp [recallCosts, ← hmatch, metricCost_self]
  intro slot _ hslot
  rw [he]
  cases slot with
  | none => simp [recallCosts]
  | some k =>
    have hk : k ≠ j := fun h => hslot (congrArg some h)
    have hne : code (x.query i) ≠ code (x.key k) := by
      intro h
      exact hk (hx ((hc h).symm.trans hmatch.symm))
    have hg := metricCost_gap w hw _ _ hne
    change (0 : ℝ) + 1 / 2 ≤ metricCost w _ _
    linarith

/-- With no prior match, the null slot beats every allowed key by `1/2`.
This avoids a false positive at the first occurrence of a key. Extension
of arXiv:2211.11052v1, §3.1, using the no-match semantics in
arXiv:2312.04927v1, Appendix `sec: intro-general-ar`. -/
theorem null_slot_gap {n c b : ℕ} (code : Fin c → Code b)
    (hc : Function.Injective code) (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (x : MQARInstance n c) (i : Fin n)
    (hno : ∀ j, j < i → x.key j ≠ x.query i) :
    ∀ slot, slot ∈ allowedSlots i → slot ≠ none →
      recallCosts code w x i none + 1 / 2 ≤ recallCosts code w x i slot := by
  intro slot hallowed hslot
  cases slot with
  | none => exact (hslot rfl).elim
  | some j =>
    have hne : code (x.query i) ≠ code (x.key j) := by
      intro h
      exact hno j hallowed (hc h).symm
    have hg := metricCost_gap w hw _ _ hne
    change (1 / 2 : ℝ) + 1 / 2 ≤ metricCost w _ _
    norm_num
    exact hg

/-- Every query has a global optimum of its convex routing program.
The proof exhibits a matching or null basis vector; existence is not an
assumed guarantee of a numerical solver. Extension of arXiv:2211.11052v1,
§3.1, on the MQAR inputs of arXiv:2312.04927v1, Appendix `app:synthetic`. -/
theorem exists_recall_optimum {n c b : ℕ} (code : Fin c → Code b)
    (hc : Function.Injective code) (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (x : MQARInstance n c) (hx : UniqueKeys x)
    (i : Fin n) : ∃ a, IsRecallOptimum code w x i a := by
  classical
  by_cases h : ∃ j, j < i ∧ x.key j = x.query i
  · obtain ⟨j, hj, hmatch⟩ := h
    exact ⟨basis (some j), basis_minimizes_routing (allowedSlots i)
      (recallCosts code w x i) (some j) hj
      (matching_slot_gap code hc w hw x hx i j hmatch)⟩
  · have hno : ∀ j, j < i → x.key j ≠ x.query i := by
      intro j hj he
      exact h ⟨j, hj, he⟩
    exact ⟨basis none, basis_minimizes_routing (allowedSlots i)
      (recallCosts code w x i) none (show True from True.intro)
      (null_slot_gap code hc w hw x i hno)⟩

/-- All hypotheses are jointly attainable, including a prior matching key.
Source context: arXiv:2312.04927v1, Appendix `app:synthetic`. -/
example : ∃ code : Fin 2 → Code 1, ∃ w : MetricWeights 1,
    ∃ x : MQARInstance 2 2, Function.Injective code ∧
      (∀ r, 1 ≤ w r) ∧ UniqueKeys x ∧ PriorAnswer x 1 0 := by
  refine ⟨binaryCode (by norm_num), fun _ => 1,
    {key := id, value := id, query := fun _ => 0},
    binaryCode_injective _, fun _ => le_rfl, ?_, ?_⟩
  · intro i j h
    exact h
  · exact ⟨0, by decide, rfl, rfl⟩

/-- No-match hypotheses are jointly attainable at the initial query.
Source context: arXiv:2312.04927v1, §3, causal recall definition. -/
example : ∃ code : Fin 2 → Code 1, ∃ w : MetricWeights 1,
    ∃ x : MQARInstance 2 2, Function.Injective code ∧
      (∀ r, 1 ≤ w r) ∧ (∀ j, j < (0 : Fin 2) → x.key j ≠ x.query 0) := by
  refine ⟨binaryCode (by norm_num), fun _ => 1,
    {key := id, value := id, query := id},
    binaryCode_injective _, fun _ => le_rfl, ?_⟩
  intro j hj
  exact (Fin.not_lt_zero j hj).elim

end Transformer.ConvexRecall
