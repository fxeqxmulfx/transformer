/-
# Moving original concentrated quadrant laws to their final target caps

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, §3.2,
selective isolation, and §5, matching after concentration. Start with
the actual laws supported near a generic intermediate atom family.
Move them one at a time to the prescribed positive target atoms.

Every unprocessed law is exactly its original intermediate input;
every processed law is supported near its final target. The selected
law's normalized barycenter stays in its supporting intermediate cap.
The common robust plane conditions therefore protect both kinds of
other laws despite the changes in their actual barycenters.

At most six original phases per processed index suffice. The final
concentration tolerance may be smaller than the geometric radius,
which was chosen before the final tolerance.
-/

import Transformer.Interpolation.Section4_RobustQuadrantFamily

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- A finite subset of intermediate original laws can be concentrated
at their final targets while the other complete laws remain unchanged.
Source auxiliary: arXiv:2411.04551v3, §1, §3.2 and §5. -/
theorem exists_partial_laws_to_targets_phases (hd : 3 ≤ d) (i : Fin d)
    (ν : Idx N → Perspective.ProbSphere d) (Z Y : Idx N → SSphere d)
    (r : ℝ) (hr : 0 < r) (hr1 : r ≤ 1)
    (hν : ∀ k, (ν k : Measure (SSphere d)).support ⊆ Metric.closedBall (Z k) r)
    (hZ : ∀ k, Metric.closedBall (Z k) r ⊆ positiveQuadrant d)
    (hY : ∀ k, Metric.closedBall (Y k) r ⊆ positiveQuadrant d)
    (hnew : ∀ j k, k ≠ j → ∀ p m : EucSpace d,
      dist p (Z j : EucSpace d) ≤ r → dist m (Z k : EucSpace d) ≤ r →
      ∀ α β : ℝ, m ≠ α • p + β • (Y j : EucSpace d))
    (htarget : ∀ j k, k ≠ j → ∀ p m : EucSpace d,
      dist p (Z j : EucSpace d) ≤ r → dist m (Y k : EucSpace d) ≤ r →
      ∀ α β : ℝ, m ≠ α • p + β • (Y j : EucSpace d))
    (ε : ℝ) (hε : 0 < ε) (hεr : ε ≤ r) (S : Finset (Idx N)) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 6 * S.card ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      (∀ k ∈ S, (phaseSequenceLaw L (ν k) : Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) ε) ∧
      ∀ k, k ∉ S → phaseSequenceLaw L (ν k) = ν k := by
  classical
  induction S using Finset.induction_on with
  | empty =>
    refine ⟨[], Nat.zero_le _, ?_, ?_, ?_⟩
    · intro P hP
      cases hP
    · intro k hk
      cases hk
    · intro k _
      exact phaseSequenceLaw_nil (ν k)
  | @insert j S hj ih =>
    obtain ⟨L, hlen, hL, hselected, hrest⟩ := ih
    let C (k : Idx N) := phaseSequenceLaw L (ν k)
    have hprocessed (k : Idx N) (hk : k ∈ S) :
        (C k : Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) r := by
      intro x hx
      exact (Metric.mem_closedBall.mp (hselected k hk hx)).trans hεr
    have hunprocessed (k : Idx N) (hk : k ∉ S) :
        (C k : Measure (SSphere d)).support ⊆ Metric.closedBall (Z k) r := by
      rw [show C k = ν k from hrest k hk]
      exact hν k
    have hCQ (k : Idx N) : (C k : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
      by_cases hk : k ∈ S
      · exact (hprocessed k hk).trans (hY k)
      · exact (hunprocessed k hk).trans (hZ k)
    obtain ⟨p, hp, _⟩ := exists_positive_barycenter_pole (by omega : 1 ≤ d) (C j) (hCQ j)
    have hplane (k : Idx N) (hkj : k ≠ j) :
        ∀ α β : ℝ, barycenter d (C k) ≠ α • (p : EucSpace d) + β • (Y j : EucSpace d) := by
      by_cases hk : k ∈ S
      · exact plane_exclusion_for_concentrated_laws (Z j) (Y k) p (Y j) (C j) (C k) r
          hr.le hr1 hp (hunprocessed j hj) (hprocessed k hk) (htarget j k hkj)
      · exact plane_exclusion_for_concentrated_laws (Z j) (Z k) p (Y j) (C j) (C k) r
          hr.le hr1 hp (hunprocessed j hj) (hunprocessed k hk) (hnew j k hkj)
    have hYQ : Y j ∈ positiveQuadrant d :=
      hY j (by simpa only [Metric.mem_closedBall, dist_self] using hr.le)
    obtain ⟨M, hMlen, hM, hMselected, hMrest⟩ :=
      exists_quadrant_law_replacement_phases hd i C hCQ j p (Y j) hp hYQ hplane ε hε
    refine ⟨L ++ M, ?_, ?_, ?_, ?_⟩
    · rw [List.length_append, Finset.card_insert_of_notMem hj]
      omega
    · intro P hP
      rcases List.mem_append.mp hP with hP | hP
      · exact hL P hP
      · exact hM P hP
    · intro k hk
      rw [phaseSequenceLaw_append]
      by_cases hkj : k = j
      · subst k
        exact hMselected
      · rw [hMrest k hkj]
        exact hselected k ((Finset.mem_insert.mp hk).resolve_left hkj)
    · intro k hk
      have hkj : k ≠ j := fun he => hk (Finset.mem_insert.mpr (Or.inl he))
      have hkS : k ∉ S := fun h => hk (Finset.mem_insert.mpr (Or.inr h))
      rw [phaseSequenceLaw_append, hMrest k hkj]
      exact hrest k hkS

/-- The entire intermediate original family reaches the final target
caps in at most six phases per law. Source auxiliary:
arXiv:2411.04551v3, §1, §3.2 and §5, the second matching stage. -/
theorem exists_laws_to_targets_phases (hd : 3 ≤ d) (i : Fin d)
    (ν : Idx N → Perspective.ProbSphere d) (Z Y : Idx N → SSphere d)
    (r : ℝ) (hr : 0 < r) (hr1 : r ≤ 1)
    (hν : ∀ k, (ν k : Measure (SSphere d)).support ⊆ Metric.closedBall (Z k) r)
    (hZ : ∀ k, Metric.closedBall (Z k) r ⊆ positiveQuadrant d)
    (hY : ∀ k, Metric.closedBall (Y k) r ⊆ positiveQuadrant d)
    (hnew : ∀ j k, k ≠ j → ∀ p m : EucSpace d,
      dist p (Z j : EucSpace d) ≤ r → dist m (Z k : EucSpace d) ≤ r →
      ∀ α β : ℝ, m ≠ α • p + β • (Y j : EucSpace d))
    (htarget : ∀ j k, k ≠ j → ∀ p m : EucSpace d,
      dist p (Z j : EucSpace d) ≤ r → dist m (Y k : EucSpace d) ≤ r →
      ∀ α β : ℝ, m ≠ α • p + β • (Y j : EucSpace d))
    (ε : ℝ) (hε : 0 < ε) (hεr : ε ≤ r) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 6 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, (phaseSequenceLaw L (ν k) : Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) ε := by
  classical
  obtain ⟨L, hlen, hL, hselected, _⟩ := exists_partial_laws_to_targets_phases hd i ν Z Y
    r hr hr1 hν hZ hY hnew htarget ε hε hεr Finset.univ
  refine ⟨L, ?_, hL, fun k => hselected k (Finset.mem_univ k)⟩
  simpa only [Finset.card_univ, Fintype.card_fin] using hlen

/-- Actual generic-point Diracs and robustly chosen intermediate and
final caps jointly realize all hypotheses, including a positive final
tolerance smaller than the geometric radius and a processed index. -/
example : ∃ (ν : Idx 2 → Perspective.ProbSphere 3) (Z Y : Idx 2 → SSphere 3)
    (r ε : ℝ) (S : Finset (Idx 2)), 3 ≤ 3 ∧ 0 < r ∧ r ≤ 1 ∧ 0 < ε ∧ ε ≤ r ∧ S.Nonempty ∧
      (∀ k, (ν k : Measure (SSphere 3)).support ⊆ Metric.closedBall (Z k) r) ∧
      (∀ k, Metric.closedBall (Z k) r ⊆ positiveQuadrant 3) ∧
      (∀ k, Metric.closedBall (Y k) r ⊆ positiveQuadrant 3) ∧
      (∀ j k, k ≠ j → ∀ p m : EucSpace 3,
        dist p (Z j : EucSpace 3) ≤ r → dist m (Z k : EucSpace 3) ≤ r →
        ∀ α β : ℝ, m ≠ α • p + β • (Y j : EucSpace 3)) ∧
      ∀ j k, k ≠ j → ∀ p m : EucSpace 3,
        dist p (Z j : EucSpace 3) ≤ r → dist m (Y k : EucSpace 3) ≤ r →
        ∀ α β : ℝ, m ≠ α • p + β • (Y j : EucSpace 3) := by
  let Y (k : Idx 2) := quadrantThreePoint k.castSucc
  have hinj : Function.Injective Y := by
    intro k l he
    exact Fin.castSucc_injective 2 (quadrantThreePoint_injective he)
  have hQ (k : Idx 2) : Y k ∈ positiveQuadrant 3 := quadrantThreePoint_mem k.castSucc
  obtain ⟨Z, r, _, hr, hr1, hZ, hY, _, _, hnew, htarget⟩ :=
    exists_robust_quadrant_matching_family (d := 3) le_rfl (by decide) Y Y hinj hinj hQ hQ
  refine ⟨(fun k => Perspective.diracProb 3 (Z k)), Z, Y, r, r / 2, {0}, le_rfl,
    hr, hr1, half_pos hr, (half_lt_self hr).le, by simp, ?_, hZ, hY, hnew, htarget⟩
  intro k x hx
  rw [eq_of_mem_support_dirac hx]
  exact Metric.mem_closedBall.mpr (by simpa only [dist_self] using hr.le)

end Transformer.Interpolation
