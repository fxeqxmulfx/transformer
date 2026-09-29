/-
# A trainable content metric for convex associative recall

This is a new extension, not a theorem or model claimed by either paper.
It keeps the convex-training aim of arXiv:2211.11052v1, §3.1–§3.4, but
replaces sample-independent positional weights by content-dependent routing.
The task is arXiv:2312.04927v1, §3 and Appendix `app:synthetic`.
Token codes are fixed binary encodings; all metric coordinates are trainable.
-/

import Mathlib

open scoped BigOperators

namespace Transformer.ConvexRecall

/-- A binary token embedding; unlike the learned embeddings of the MQAR
experiment, this encoding is fixed. Source context: arXiv:2312.04927v1,
§4, the binary-encoding hypothesis of `thm: indep-genar`. -/
abbrev Code (b : ℕ) := Fin b → Fin 2

/-- Trainable diagonal metric coordinates. -/
abbrev MetricWeights (b : ℕ) := Fin b → ℝ

/-- An injective binary encoding for a vocabulary that fits in `b` bits. -/
def binaryCode {b c : ℕ} (hc : c ≤ 2 ^ b) (t : Fin c) : Code b :=
  finFunctionFinEquiv.symm (Fin.castLE hc t)

/-- Fixed binary codes distinguish all vocabulary tokens. This supplies
the encoding assumption used below, rather than assuming a learned embedding
is injective. Extension of arXiv:2312.04927v1, §4, binary input encoding. -/
theorem binaryCode_injective {b c : ℕ} (hc : c ≤ 2 ^ b) :
    Function.Injective (binaryCode hc) :=
  finFunctionFinEquiv.symm.injective.comp (Fin.castLE_injective hc)

/-- The binary vocabulary-size hypothesis is attainable. Source context:
arXiv:2312.04927v1, §4, binary input encoding. -/
example : (8192 : ℕ) ≤ 2 ^ 13 := by norm_num

/-- The weighted Hamming cost is linear in the learned coordinates.
Negative coordinates are permitted during unconstrained convex training. -/
def metricCost {b : ℕ} (w : MetricWeights b) (q k : Code b) : ℝ :=
  ∑ r, if q r = k r then 0 else w r

/-- Equal codes have zero cost for every trainable metric. Extension of
arXiv:2211.11052v1, §3.1; recall context: arXiv:2312.04927v1, §4. -/
theorem metricCost_self {b : ℕ} (w : MetricWeights b) (q : Code b) :
    metricCost w q q = 0 := by simp [metricCost]

/-- Costs preserve convex combinations of the metric coordinates. This is
the linear feature lift used for convex training, not joint learning of two
query/key matrices. Extension of arXiv:2211.11052v1, §3.2. -/
theorem metricCost_mix {b : ℕ} (w z : MetricWeights b) (q k : Code b)
    (a t : ℝ) :
    metricCost (a • w + t • z) q k =
      a * metricCost w q k + t * metricCost z q k := by
  simp only [metricCost, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro r _
  by_cases h : q r = k r <;> simp [h]

/-- Unit lower bounds on the learned coordinates imply nonnegative costs.
Extension of arXiv:2211.11052v1, §3.1, for the MQAR task of
arXiv:2312.04927v1, §4. -/
theorem metricCost_nonneg {b : ℕ} (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (q k : Code b) : 0 ≤ metricCost w q k := by
  apply Finset.sum_nonneg
  intro r _
  split_ifs
  · exact le_rfl
  · exact (by norm_num : (0 : ℝ) ≤ 1).trans (hw r)

/-- Every distinct code has cost at least one, independently of sequence
length. The condition concerns the trained metric, not a bound on the
number or distance of queries. Extension of arXiv:2211.11052v1, §3.1;
task context: arXiv:2312.04927v1, §4, `prop: attention-ar`. -/
theorem metricCost_gap {b : ℕ} (w : MetricWeights b)
    (hw : ∀ r, 1 ≤ w r) (q k : Code b) (h : q ≠ k) :
    1 ≤ metricCost w q k := by
  classical
  obtain ⟨r, hr⟩ : ∃ r, q r ≠ k r := by
    by_contra hno
    apply h
    funext r
    by_contra hr
    exact hno ⟨r, hr⟩
  have hsum := Finset.single_le_sum
    (s := Finset.univ) (f := fun r => if q r = k r then (0 : ℝ) else w r)
    (fun r _ => by split_ifs; exact le_rfl; exact le_trans (by norm_num) (hw r))
    (Finset.mem_univ r)
  simpa [metricCost, hr] using (hw r).trans (by simpa [hr] using hsum)

/-- Distinct codes and admissible trainable weights exist. Source context:
arXiv:2312.04927v1, §4, binary input encoding. -/
example : ∃ w : MetricWeights 1, ∃ q k : Code 1,
    (∀ r, 1 ≤ w r) ∧ q ≠ k := by
  refine ⟨fun _ => 1, fun _ => 0, fun _ => 1, fun _ => le_rfl, ?_⟩
  intro h
  have he := congrFun h 0
  norm_num at he

end Transformer.ConvexRecall
