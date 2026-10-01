/-
# Exact arithmetic tests for constrained polynomial root minima

The universal comparison is eliminated with a strict-better-root query.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.SignConstraints
import Transformer.Sturm.RootExistence

noncomputable section
open Polynomial Transformer.Normalization
open scoped Classical

namespace Transformer.Sturm

/-- The constrained root count expression is nonnegative because it
is an actual positive multiple of a finite cardinality. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signConstraintQuery_nonneg {p : Polynomial ℝ} (hp : p ≠ 0)
    (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    0 ≤ signConstraintQuery p 1 conditions := by
  rw [signConstraintQuery_eq_count hp]
  exact mul_nonneg (pow_nonneg (by norm_num) _) (Nat.cast_nonneg _)

/-- Absence of a root satisfying the finite signs is exactly the
vanishing of the finite arithmetic query. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem no_root_with_signs_iff {p : Polynomial ℝ} (hp : p ≠ 0)
    (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    ¬(∃ x : ℝ, p.eval x = 0 ∧ ∀ c ∈ conditions, c.2.Holds (c.1.eval x)) ↔
      signConstraintQuery p 1 conditions = 0 := by
  constructor
  · intro hno
    apply le_antisymm
    · exact le_of_not_gt (fun h => hno ((exists_root_with_signs_iff hp conditions).mpr h))
    · exact signConstraintQuery_nonneg hp conditions
  · intro hzero hex
    have hpos := (exists_root_with_signs_iff hp conditions).mp hex
    rw [hzero] at hpos
    exact (lt_irrefl 0) hpos

/-- Strict polynomial signs expressing an open interval exactly.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def intervalSignConditions (a b : ℝ) : List (Polynomial ℝ × AnalyticSignRequirement) :=
  [(X - C a, .positive), (C b - X, .positive)]

/-- The two interval signs are equivalent to actual membership in
`(a, b)`, without any endpoint ordering assumption. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem intervalSignConditions_iff (a b x : ℝ) :
    (∀ c ∈ intervalSignConditions a b, c.2.Holds (c.1.eval x)) ↔ x ∈ Set.Ioo a b := by
  simp [intervalSignConditions, AnalyticSignRequirement.Holds, sub_pos, Set.mem_Ioo]

/-- The signs expressing strict improvement over a candidate root,
interval membership, and all original constraints. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def betterRootConditions (V : Polynomial ℝ) (y a b : ℝ)
    (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    List (Polynomial ℝ × AnalyticSignRequirement) :=
  (C (V.eval y) - V, .positive) :: (intervalSignConditions a b ++ conditions)

/-- The better-root signs retain the actual objective comparison and
every original admissibility condition. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem betterRootConditions_iff (V : Polynomial ℝ) (y a b x : ℝ)
    (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    (∀ c ∈ betterRootConditions V y a b conditions, c.2.Holds (c.1.eval x)) ↔
      V.eval x < V.eval y ∧ x ∈ Set.Ioo a b ∧
        ∀ c ∈ conditions, c.2.Holds (c.1.eval x) := by
  simp only [betterRootConditions, List.forall_mem_cons, List.forall_mem_append,
    intervalSignConditions_iff]
  rw [eval_sub, eval_C]
  change (0 < V.eval y - V.eval x ∧ x ∈ Set.Ioo a b ∧
    ∀ c ∈ conditions, c.2.Holds (c.1.eval x)) ↔ _
  rw [sub_pos]

/-- A candidate's objective value is no worse than every admissible
root exactly when the arithmetic query for a strictly better root is
zero. This eliminates the entire scalar universal comparison, for any
degree, repeated roots, and any finite mixture of zero, positive, and
nonnegative conditions. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem polynomial_root_minimum_iff {p : Polynomial ℝ} (hp : p ≠ 0)
    (V : Polynomial ℝ) (conditions : List (Polynomial ℝ × AnalyticSignRequirement))
    (y a b : ℝ) :
    (∀ x ∈ Set.Ioo a b, p.eval x = 0 →
      (∀ c ∈ conditions, c.2.Holds (c.1.eval x)) → V.eval y ≤ V.eval x) ↔
      signConstraintQuery p 1 (betterRootConditions V y a b conditions) = 0 := by
  rw [← no_root_with_signs_iff hp]
  constructor
  · intro hmin hex
    obtain ⟨x, hpx, hsigns⟩ := hex
    obtain ⟨hbetter, hx, hconditions⟩ :=
      (betterRootConditions_iff V y a b x conditions).mp hsigns
    exact not_lt_of_ge (hmin x hx hpx hconditions) hbetter
  · intro hno x hx hpx hconditions
    apply le_of_not_gt
    intro hbetter
    exact hno ⟨x, hpx, (betterRootConditions_iff V y a b x conditions).mpr
      ⟨hbetter, hx, hconditions⟩⟩

/-- A repeated-root polynomial with an actual feasible central root
and all three kinds of constraints witnesses the nonzero input.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (X ^ 4 : Polynomial ℝ) ≠ 0 ∧
    ∃ x ∈ Set.Ioo (-1 : ℝ) 1, (X ^ 4 : Polynomial ℝ).eval x = 0 := by
  exact ⟨pow_ne_zero _ X_ne_zero, 0, by constructor <;> norm_num, by simp⟩

end Transformer.Sturm
