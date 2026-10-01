/-
# Joint witnesses for the Sturm and signed-query hypotheses

These examples include actual roots, an interior sign collapse,
nonzero query values, generic endpoints, and repeated roots. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.ConstrainedMinima

noncomputable section
open Filter Topology Polynomial
open Transformer.Normalization

namespace Transformer.Sturm

/-- The actual chain `[X, 1]` is a Sturm chain for the linear polynomial.
This is the joint witness for the geometric chain assumptions in the
copied root-count API. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem linear_isSturmChain : IsSturmChain (X : Polynomial ℝ) [X, 1] := by
  refine ⟨rfl, ?_, ?_, ?_, ?_⟩
  · intro r hr
    have hr0 : r = 0 := by simpa only [Polynomial.IsRoot, eval_X] using hr
    subst r
    refine ⟨1, rfl, by simp, ?_, ?_⟩
    · filter_upwards [self_mem_nhdsWithin] with x hx
      simpa only [Set.mem_Iio, mul_one, eval_X] using hx
    · filter_upwards [self_mem_nhdsWithin] with x hx
      simpa only [Set.mem_Ioi, mul_one, eval_X] using hx
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl
    · exact X_ne_zero
    · exact one_ne_zero
  · intro i x a b c h0 h1 h2 hb
    rw [List.getElem?_eq_none (by simp)] at h2
    contradiction
  · intro q hq x
    have hq1 : q = 1 := by simpa using hq.symm
    subst q
    simp

/-- An interval with an actual head root and generic endpoints
satisfies the root-crossing, bounded-query, and root-count hypotheses
simultaneously. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : IsSturmChain (X : Polynomial ℝ) [X, 1] ∧
    (X : Polynomial ℝ).roots.Nodup ∧ (-1 : ℝ) < 0 ∧ (0 : ℝ) < 1 ∧
    (X : Polynomial ℝ).eval 0 = 0 ∧
    (∀ q ∈ ([X, 1] : List (Polynomial ℝ)), ∀ x ∈ Set.Icc (-1 : ℝ) 1,
      x ≠ 0 → q.eval x ≠ 0) ∧
    (∀ q ∈ ([X, 1] : List (Polynomial ℝ)), q.eval (-1) ≠ 0 ∧ q.eval 1 ≠ 0) := by
  refine ⟨linear_isSturmChain, by simp, by norm_num, by norm_num, by simp, ?_, ?_⟩
  · intro q hq x hx hne
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl
    · simpa only [eval_X] using hne
    · simp
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl <;> norm_num

/-- A nonzero constant chain witnesses the interior-crossing API on
an interval where its head has no zero. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : IsSturmChain (1 : Polynomial ℝ) [1] ∧
    ¬(1 : Polynomial ℝ).IsRoot 0 ∧ (-1 : ℝ) < 0 ∧ (0 : ℝ) < 1 ∧
    (∀ q ∈ ([1] : List (Polynomial ℝ)), ∀ x ∈ Set.Icc (-1 : ℝ) 1, q.eval x ≠ 0) := by
  refine ⟨?_, by simp, by norm_num, by norm_num, ?_⟩
  · refine ⟨rfl, ?_, ?_, ?_, ?_⟩
    · intro r hr
      simp [Polynomial.IsRoot] at hr
    · intro q hq
      have hq1 : q = 1 := by simpa only [List.mem_singleton] using hq
      subst q
      exact one_ne_zero
    · intro i x a b c h0 h1 h2 hb
      rw [List.getElem?_eq_none (by simp)] at h2
      contradiction
    · intro q hq x
      have hq1 : q = 1 := by simpa using hq.symm
      subst q
      simp
  · intro q hq x hx
    have hq1 : q = 1 := by simpa only [List.mem_singleton] using hq
    subst q
    simp

/-- A real sign collapse is variation-neutral and has nonzero
opposite-sign neighbors. This witnesses the finite sign-relation
hypotheses. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : SignRelation [1, 2, -1] [1, 0, -1] := by
  apply SignRelation.collapse (by norm_num) (by norm_num) (by norm_num) rfl rfl
  · norm_num [sign_pos, sign_neg]
  · exact SignRelation.same (by norm_num) (by norm_num) rfl SignRelation.nil

/-- A nonzero continuous function on a preconnected interval witnesses
the general sign-continuity hypotheses. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : IsPreconnected (Set.Icc (-1 : ℝ) 1) ∧
    ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Set.Icc (-1 : ℝ) 1) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, (fun _ : ℝ => (1 : ℝ)) x ≠ 0) ∧
    (-1 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧ (1 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 := by
  exact ⟨isPreconnected_Icc, continuousOn_const, fun _ _ => one_ne_zero,
    by constructor <;> norm_num, by constructor <;> norm_num⟩

end Transformer.Sturm
