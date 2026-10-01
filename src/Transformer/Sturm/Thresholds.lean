/-
# Common objective thresholds across polynomial fibers

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.ConstrainedMinima

noncomputable section
open Polynomial Transformer.Normalization

namespace Transformer.Sturm

/-- Actual signs for a root with objective below an arbitrary common
threshold, together with the interval and all original conditions.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def belowThresholdConditions (V : Polynomial ℝ) (level a b : ℝ)
    (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    List (Polynomial ℝ × AnalyticSignRequirement) :=
  (C level - V, .positive) :: (intervalSignConditions a b ++ conditions)

/-- Threshold signs preserve the objective, interval, and every
admissibility condition exactly. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem belowThresholdConditions_iff (V : Polynomial ℝ) (level a b x : ℝ)
    (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    (∀ c ∈ belowThresholdConditions V level a b conditions, c.2.Holds (c.1.eval x)) ↔
      V.eval x < level ∧ x ∈ Set.Ioo a b ∧
        ∀ c ∈ conditions, c.2.Holds (c.1.eval x) := by
  simp only [belowThresholdConditions, List.forall_mem_cons, List.forall_mem_append,
    intervalSignConditions_iff]
  rw [eval_sub, eval_C]
  change (0 < level - V.eval x ∧ x ∈ Set.Ioo a b ∧
    ∀ c ∈ conditions, c.2.Holds (c.1.eval x)) ↔ _
  rw [sub_pos]

/-- A single real threshold is a lower bound for every admissible root
exactly when its strict-below-threshold query is zero. The threshold
may come from a candidate in a different fiber. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomial_root_lower_bound_iff {p : Polynomial ℝ} (hp : p ≠ 0)
    (V : Polynomial ℝ) (conditions : List (Polynomial ℝ × AnalyticSignRequirement))
    (level a b : ℝ) :
    (∀ x ∈ Set.Ioo a b, p.eval x = 0 →
      (∀ c ∈ conditions, c.2.Holds (c.1.eval x)) → level ≤ V.eval x) ↔
      signConstraintQuery p 1 (belowThresholdConditions V level a b conditions) = 0 := by
  rw [← no_root_with_signs_iff hp]
  constructor
  · intro hbound hex
    obtain ⟨x, hpx, hsigns⟩ := hex
    obtain ⟨hbelow, hx, hconditions⟩ :=
      (belowThresholdConditions_iff V level a b x conditions).mp hsigns
    exact not_lt_of_ge (hbound x hx hpx hconditions) hbelow
  · intro hno x hx hpx hconditions
    apply le_of_not_gt
    intro hbelow
    exact hno ⟨x, hpx, (belowThresholdConditions_iff V level a b x conditions).mpr
      ⟨hbelow, hx, hconditions⟩⟩

/-- A repeated-root polynomial is a nonzero input for arbitrary
common thresholds. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : (X ^ 4 : Polynomial ℝ) ≠ 0 ∧ (X ^ 4 : Polynomial ℝ).eval 0 = 0 := by
  exact ⟨pow_ne_zero _ X_ne_zero, by simp⟩

end Transformer.Sturm
