/-
# A trained convex mechanism performs exact multi-query associative recall

New extension of arXiv:2211.11052v1, §3.1–§3.2, evaluated against the
MQAR specification of arXiv:2312.04927v1, §3 and Appendix `app:synthetic`.
It trains a metric on fixed binary codes and solves a content-dependent
simplex quadratic program for each query. Values are copied from the input
dictionary, not remembered in training parameters. The theorem includes
all query positions, arbitrary value assignments, and no-match outputs.
-/

import Transformer.ConvexRecall.Training
import Transformer.ConvexRecall.Routing

open scoped BigOperators

noncomputable section

namespace Transformer.ConvexRecall

open Transformer.Zoology

/-- Attention-style value mixing; the null slot carries the zero vector. -/
def readSlots {n c : ℕ} (x : MQARInstance n c)
    (a : RecallSlot n → ℝ) (v : Fin c) : ℝ :=
  ∑ j, a (some j) * oneHot (x.value j) v

/-- A basis weight copies its selected value. Extension of
arXiv:2211.11052v1, §3.1, with the value mixing of arXiv:2312.04927v1,
Appendix `prop: app-attention`. -/
theorem readSlots_basis_some {n c : ℕ} (x : MQARInstance n c)
    (j : Fin n) (v : Fin c) :
    readSlots x (basis (some j)) v = oneHot (x.value j) v := by
  classical
  simp [readSlots, basis]

/-- The null basis outputs zero in every value coordinate. Extension of
arXiv:2211.11052v1, §3.1; no-match specification from
arXiv:2312.04927v1, Appendix `sec: intro-general-ar`. -/
theorem readSlots_basis_none {n c : ℕ} (x : MQARInstance n c) (v : Fin c) :
    readSlots x (basis none) v = 0 := by
  simp [readSlots, basis]

/-- Every solution of the new convex inference program produces exactly
the same vector as the paper's corrected causal equality-score attention.
This compares functional recall, not optimization speed or parameter count.
Extension of arXiv:2211.11052v1, §3.1; reference construction:
arXiv:2312.04927v1, Appendix `prop: app-attention`. -/
theorem recall_optimum_agrees_attention {n c b : ℕ} (code : Fin c → Code b)
    (hc : Function.Injective code) (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (x : MQARInstance n c) (hx : UniqueKeys x)
    (i : Fin n) (a : RecallSlot n → ℝ) (ha : IsRecallOptimum code w x i a)
    (v : Fin c) : readSlots x a v = pairedAttentionOutput x i v := by
  classical
  by_cases h : ∃ j, j < i ∧ x.key j = x.query i
  · obtain ⟨j, hj, hmatch⟩ := h
    have he := routing_minimizer_unique (allowedSlots i) (recallCosts code w x i)
      (some j) (matching_slot_gap code hc w hw x hx i j hmatch) a ha.1
      (ha.2 _ (basis_mem_simplex (allowedSlots i) (some j) hj))
    rw [he, readSlots_basis_some]
    exact (paired_attention_exact x hx i j hj hmatch v).symm
  · have hno : ∀ j, j < i → x.key j ≠ x.query i := by
      intro j hj he
      exact h ⟨j, hj, he⟩
    have he := routing_minimizer_unique (allowedSlots i) (recallCosts code w x i)
      none (null_slot_gap code hc w hw x i hno) a ha.1
      (ha.2 _ (basis_mem_simplex (allowedSlots i) none (show True from True.intro)))
    rw [he, readSlots_basis_none]
    exact (paired_attention_no_match x i hno v).symm

/-- Choose an optimum of the actual convex program. Exact recall is a
proved property of this optimizer, not the definition of its output. -/
def convexRecallWeights {n c b : ℕ} (code : Fin c → Code b)
    (hc : Function.Injective code) (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (x : MQARInstance n c) (hx : UniqueKeys x)
    (i : Fin n) : RecallSlot n → ℝ :=
  Classical.choose (exists_recall_optimum code hc w hw x hx i)

/-- Chosen weights solve the content-dependent masked simplex program.
Extension of arXiv:2211.11052v1, §3.1, with the MQAR mask from
arXiv:2312.04927v1, §3. -/
theorem convexRecallWeights_optimal {n c b : ℕ} (code : Fin c → Code b)
    (hc : Function.Injective code) (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (x : MQARInstance n c) (hx : UniqueKeys x)
    (i : Fin n) :
    IsRecallOptimum code w x i (convexRecallWeights code hc w hw x hx i) :=
  Classical.choose_spec (exists_recall_optimum code hc w hw x hx i)

/-- Output of the convex content-routing mechanism. -/
def convexRecallOutput {n c b : ℕ} (code : Fin c → Code b)
    (hc : Function.Injective code) (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (x : MQARInstance n c) (hx : UniqueKeys x)
    (i : Fin n) (v : Fin c) : ℝ :=
  readSlots x (convexRecallWeights code hc w hw x hx i) v

/-- All positions and all value coordinates satisfy the causal MQAR
specification, including queries without a match. Keys must be distinct;
values need not be distinct. Extension of arXiv:2211.11052v1, §3.1;
task specification: arXiv:2312.04927v1, §3 and Appendix `app:synthetic`. -/
theorem convexRecall_solves_mqar {n c b : ℕ} (code : Fin c → Code b)
    (hc : Function.Injective code) (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (x : MQARInstance n c) (hx : UniqueKeys x)
    (i : Fin n) (v : Fin c) :
    convexRecallOutput code hc w hw x hx i v = expectedPairedAnswer x i v := by
  unfold convexRecallOutput
  rw [recall_optimum_agrees_attention code hc w hw x hx i _
    (convexRecallWeights_optimal code hc w hw x hx i)]
  exact paired_attention_solves_mqar x hx i v

/-- A trained global optimum supplies every required metric margin.
Extension of arXiv:2211.11052v1, §3.2, with the explicit calibration loss. -/
theorem trained_metric_margins {b : ℕ} (w : MetricWeights b)
    (hmin : ∀ z : MetricWeights b,
      calibrationObjective w ≤ calibrationObjective z) : ∀ r, 1 ≤ w r := by
  rw [calibration_minimizer_unique w hmin]
  exact fun _ => le_rfl

/-- **Convex training followed by exact MQAR.** For any vocabulary fitting
in `b` bits, every global calibration optimizer, every distinct-key input,
and every query, the learned convex mechanism returns the exact answer.
No training dictionary or sequence-length parameter is used by the metric.
The fixed binary encoder and extra matching calibration are changes to the
experimental setup, not claims of parity with end-to-end Transformer training.
Extension of arXiv:2211.11052v1, §3.1–§3.2; task:
arXiv:2312.04927v1, §3 and Appendix `app:synthetic`. -/
theorem trained_convexRecall_solves_mqar {n c b : ℕ} (hsize : c ≤ 2 ^ b)
    (w : MetricWeights b) (hmin : ∀ z : MetricWeights b,
      calibrationObjective w ≤ calibrationObjective z)
    (x : MQARInstance n c) (hx : UniqueKeys x) (i : Fin n) (v : Fin c) :
    convexRecallOutput (binaryCode hsize) (binaryCode_injective hsize)
      w (trained_metric_margins w hmin) x hx i v = expectedPairedAnswer x i v :=
  convexRecall_solves_mqar _ _ w (trained_metric_margins w hmin) x hx i v

/-- The complete trained-recall hypotheses hold together in a nonempty
instance with a genuine prior answer. Source context: arXiv:2312.04927v1,
Appendix `app:synthetic`. -/
example : ∃ w : MetricWeights 1, ∃ x : MQARInstance 2 2,
    (2 : ℕ) ≤ 2 ^ 1 ∧
    (∀ z : MetricWeights 1, calibrationObjective w ≤ calibrationObjective z) ∧
    UniqueKeys x ∧ PriorAnswer x 1 0 := by
  refine ⟨unitMetric 1, {key := id, value := id, query := fun _ => 0},
    by norm_num, unitMetric_minimizes, ?_, ⟨0, by decide, rfl, rfl⟩⟩
  intro i j h
  exact h

/-- The optimizer premise is also attained by a genuine prior-query
instance. Source context: arXiv:2312.04927v1, Appendix `app:synthetic`. -/
example : ∃ a : RecallSlot 2 → ℝ,
    IsRecallOptimum (binaryCode (by norm_num : 2 ≤ 2 ^ 1)) (unitMetric 1)
      ({key := id, value := id, query := fun _ => 0} : MQARInstance 2 2) 1 a := by
  apply exists_recall_optimum _ (binaryCode_injective _) _ (fun _ => le_rfl)
  intro i j h
  exact h

end Transformer.ConvexRecall
