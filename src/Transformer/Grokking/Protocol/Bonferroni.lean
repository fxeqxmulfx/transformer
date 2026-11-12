module

public import Transformer.Grokking.Protocol.Target

/-!
# Bonferroni's inequality: a lower bound on the probability of a union

Source: `review/grok_division_convergence.md`, section 2, where the union is the event that a
split of arXiv:2201.02177v1 (Power et al.), Figure 1, leaves some residue unlabelled. The
inequality is classical and is not in the paper, which states no probability.

The probability of a union of finitely many events is at most the sum of their probabilities.
Bonferroni's inequality is the bound from the other side: the sum exceeds the probability of
the union by at most the probabilities that two of the events occur together. It is stated
here without a subtraction and doubled, so that each pair is counted in both orders
(`two_mul_sum_measure_le`). Under the uniform law on the shuffles every set is an event
(`two_mul_sum_uniformPerm_le`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- **Bonferroni's inequality**: the probabilities of finitely many events add up to at most
the probability that one of them occurs plus the probabilities that two of them occur. Both
sides are doubled, so that the pairs are counted in both orders. -/
theorem two_mul_sum_measure_le {α ι : Type*} [DecidableEq ι] [MeasurableSpace α] (μ : Measure α)
    {A : ι → Set α} (hA : ∀ i, MeasurableSet (A i)) (T : Finset ι) :
    2 * ∑ i ∈ T, μ (A i) ≤
      2 * μ (⋃ i ∈ T, A i) + ∑ i ∈ T, ∑ j ∈ T.erase i, μ (A i ∩ A j) := by
  induction T using Finset.induction_on with
  | empty => simp
  | insert a T ha ih =>
    have hU : MeasurableSet (⋃ i ∈ T, A i) := Finset.measurableSet_biUnion T fun i _ => hA i
    have hunion := measure_union_add_inter (μ := μ) (A a) hU
    have hinter : μ (A a ∩ ⋃ i ∈ T, A i) ≤ ∑ i ∈ T, μ (A a ∩ A i) := by
      rw [Set.inter_iUnion₂]
      exact measure_biUnion_finset_le T _
    have hpairs : ∑ i ∈ insert a T, ∑ j ∈ (insert a T).erase i, μ (A i ∩ A j) =
        2 * ∑ i ∈ T, μ (A a ∩ A i) + ∑ i ∈ T, ∑ j ∈ T.erase i, μ (A i ∩ A j) := by
      rw [Finset.sum_insert ha, Finset.erase_insert ha, two_mul, add_assoc]
      congr 1
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i hi => ?_
      have hai : a ≠ i := fun h => ha (h ▸ hi)
      rw [Finset.erase_insert_of_ne hai,
        Finset.sum_insert (fun h => ha (Finset.mem_of_mem_erase h)), Set.inter_comm (A i) (A a)]
    rw [Finset.sum_insert ha, Finset.set_biUnion_insert, hpairs]
    calc 2 * (μ (A a) + ∑ i ∈ T, μ (A i))
        = 2 * μ (A a) + 2 * ∑ i ∈ T, μ (A i) := mul_add _ _ _
      _ ≤ 2 * μ (A a) + (2 * μ (⋃ i ∈ T, A i) + ∑ i ∈ T, ∑ j ∈ T.erase i, μ (A i ∩ A j)) :=
          add_le_add le_rfl ih
      _ = 2 * (μ (A a ∪ ⋃ i ∈ T, A i) + μ (A a ∩ ⋃ i ∈ T, A i)) +
            ∑ i ∈ T, ∑ j ∈ T.erase i, μ (A i ∩ A j) := by
          rw [hunion]
          ring
      _ ≤ 2 * (μ (A a ∪ ⋃ i ∈ T, A i) + ∑ i ∈ T, μ (A a ∩ A i)) +
            ∑ i ∈ T, ∑ j ∈ T.erase i, μ (A i ∩ A j) := by
          gcongr
      _ = 2 * μ (A a ∪ ⋃ i ∈ T, A i) +
            (2 * ∑ i ∈ T, μ (A a ∩ A i) + ∑ i ∈ T, ∑ j ∈ T.erase i, μ (A i ∩ A j)) := by
          ring

/-- **Bonferroni's inequality for the uniform law on the shuffles**, where every set is an
event. -/
theorem two_mul_sum_uniformPerm_le {N : ℕ} {ι : Type*} [DecidableEq ι]
    (A : ι → Set (Equiv.Perm (Fin N))) (T : Finset ι) :
    2 * ∑ i ∈ T, uniformPerm N (A i) ≤
      2 * uniformPerm N (⋃ i ∈ T, A i) + ∑ i ∈ T, ∑ j ∈ T.erase i, uniformPerm N (A i ∩ A j) :=
  two_mul_sum_measure_le _ (fun _ => MeasurableSpace.measurableSet_top) T

/-! The witness: the shuffles of three items, and the three events "the item `0` is in the
slot `i`", which are measurable. -/

example : 2 * ∑ i : Fin 3, uniformPerm 3 {π | π i = 0} ≤
    2 * uniformPerm 3 (⋃ i ∈ (Finset.univ : Finset (Fin 3)), {π | π i = 0}) +
      ∑ i : Fin 3, ∑ j ∈ Finset.univ.erase i, uniformPerm 3 ({π | π i = 0} ∩ {π | π j = 0}) :=
  two_mul_sum_measure_le (uniformPerm 3) (fun _ => MeasurableSpace.measurableSet_top) Finset.univ

end Transformer.Grokking.Protocol
