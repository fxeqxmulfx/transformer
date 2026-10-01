/-
# Finite signed sums on split real intervals

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.ChainZeros

noncomputable section
open scoped BigOperators

namespace Transformer.Sturm

/-- Splitting a half-open interval partitions a finite signed sum
exactly. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem sum_filter_Ioc_split (S : Finset ℝ) (w : ℝ → ℤ) {a c b : ℝ}
    (hac : a ≤ c) (hcb : c ≤ b) :
    (∑ x ∈ S.filter (fun r => r ∈ Set.Ioc a b), w x) =
      (∑ x ∈ S.filter (fun r => r ∈ Set.Ioc a c), w x) +
        ∑ x ∈ S.filter (fun r => r ∈ Set.Ioc c b), w x := by
  rw [Finset.sum_filter, Finset.sum_filter, Finset.sum_filter,
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro x _
  by_cases hxc : x ≤ c
  · have hxb : x ≤ b := hxc.trans hcb
    simp only [Set.mem_Ioc, hxc, hxb, and_true, not_lt_of_ge hxc,
      ite_false, add_zero]
  · have hcx : c < x := lt_of_not_ge hxc
    have hax : a < x := hac.trans_lt hcx
    simp only [Set.mem_Ioc, hxc, hcx, hax, true_and, and_false, ite_false, zero_add]

/-- A finite sum across an actual interior split satisfies both ordered
endpoint assumptions. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : (-1 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

end Transformer.Sturm
