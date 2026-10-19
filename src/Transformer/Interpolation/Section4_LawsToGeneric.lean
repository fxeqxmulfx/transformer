module

public import Transformer.Interpolation.Section4_RobustQuadrantFamily

/-!
# Moving finite original quadrant laws to generic concentration caps

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, §3.2,
selective contraction, and §5, concentration before point matching.
Induct over the finite set of indices already processed. Such laws
are supported near their generic intermediate centers; every other
law is still exactly its original input probability.

The common robust plane radius controls the actual means of the
processed laws. The old-plane exclusions protect the unprocessed
laws after normalizing their original barycenters. The selected law
is therefore moved by the six-phase concentration-and-transfer
construction while every other complete law is restored exactly.

At most six phases per processed law suffice. The induction retains
strictly positive-quadrant supports at each completed replacement,
without assuming that contraction preserves a selected mean ray.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- A finite subset of original quadrant laws can be put into generic
caps while all unprocessed original laws remain exactly unchanged.
Source auxiliary: arXiv:2411.04551v3, §1, §3.2 and §5. -/
theorem exists_partial_laws_to_generic_phases (hd : 3 ≤ d) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (X Z : Idx N → SSphere d) (hp : ∀ k, (X k : EucSpace d) = barycenterDirection (μ k))
    (r : ℝ) (hr : 0 < r) (hcap : ∀ k, Metric.closedBall (Z k) r ⊆ positiveQuadrant d)
    (hold : ∀ j k, k ≠ j → ∀ α β : ℝ,
      (X k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (Z j : EucSpace d))
    (hnew : ∀ j k, k ≠ j → ∀ m : EucSpace d, dist m (Z k : EucSpace d) ≤ r →
      ∀ α β : ℝ, m ≠ α • (X j : EucSpace d) + β • (Z j : EucSpace d))
    (S : Finset (Idx N)) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 6 * S.card ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      (∀ k ∈ S, (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ Metric.closedBall (Z k) r) ∧
      ∀ k, k ∉ S → phaseSequenceLaw L (μ k) = μ k := by
  classical
  induction S using Finset.induction_on with
  | empty =>
    refine ⟨[], Nat.zero_le _, ?_, ?_, ?_⟩
    · intro P hP
      cases hP
    · intro k hk
      cases hk
    · intro k _
      exact phaseSequenceLaw_nil (μ k)
  | @insert j S hj ih =>
    obtain ⟨L, hlen, hL, hselected, hrest⟩ := ih
    let C (k : Idx N) := phaseSequenceLaw L (μ k)
    have hCj : C j = μ j := hrest j hj
    have hCQ (k : Idx N) : (C k : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
      by_cases hk : k ∈ S
      · exact (hselected k hk).trans (hcap k)
      · rw [show C k = μ k from hrest k hk]
        exact hμ k
    have hmean : (X j : EucSpace d) = barycenterDirection (C j) := by
      rw [hCj]
      exact hp j
    have hplane (k : Idx N) (hkj : k ≠ j) (α β : ℝ) :
        barycenter d (C k) ≠ α • (X j : EucSpace d) + β • (Z j : EucSpace d) := by
      by_cases hk : k ∈ S
      · have hb : dist (barycenter d (C k)) (Z k : EucSpace d) ≤ r := by
          simpa only [dist_eq_norm] using norm_barycenter_sub_le_support_radius (C k) (Z k) r (hselected k hk)
        exact hnew j k hkj _ hb α β
      · rw [show C k = μ k from hrest k hk]
        intro he
        apply hold j k hkj (‖barycenter d (μ k)‖⁻¹ * α) (‖barycenter d (μ k)‖⁻¹ * β)
        rw [hp k, barycenterDirection, he, smul_add, smul_smul, smul_smul]
    have hZQ : Z j ∈ positiveQuadrant d :=
      hcap j (by simpa only [Metric.mem_closedBall, dist_self] using hr.le)
    obtain ⟨M, hMlen, hM, hMselected, hMrest⟩ :=
      exists_quadrant_law_replacement_phases hd i C hCQ j (X j) (Z j) hmean hZQ hplane r hr
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

/-- All original laws reach their generic positive concentration caps
using at most six phases per law. Source auxiliary:
arXiv:2411.04551v3, §1, §3.2 and §5, the first matching stage. -/
theorem exists_laws_to_generic_phases (hd : 3 ≤ d) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (X Z : Idx N → SSphere d) (hp : ∀ k, (X k : EucSpace d) = barycenterDirection (μ k))
    (r : ℝ) (hr : 0 < r) (hcap : ∀ k, Metric.closedBall (Z k) r ⊆ positiveQuadrant d)
    (hold : ∀ j k, k ≠ j → ∀ α β : ℝ,
      (X k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (Z j : EucSpace d))
    (hnew : ∀ j k, k ≠ j → ∀ m : EucSpace d, dist m (Z k : EucSpace d) ≤ r →
      ∀ α β : ℝ, m ≠ α • (X j : EucSpace d) + β • (Z j : EucSpace d)) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 6 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ Metric.closedBall (Z k) r := by
  classical
  obtain ⟨L, hlen, hL, hselected, _⟩ :=
    exists_partial_laws_to_generic_phases hd i μ hμ X Z hp r hr hcap hold hnew Finset.univ
  refine ⟨L, ?_, hL, fun k => hselected k (Finset.mem_univ k)⟩
  simpa only [Finset.card_univ, Fintype.card_fin] using hlen

/-- Two actual positive original Diracs and the robustly constructed
generic family jointly inhabit all first-stage geometric hypotheses,
with a genuinely nonempty set of processed indices. -/
example : ∃ (μ : Idx 2 → Perspective.ProbSphere 3) (X Z : Idx 2 → SSphere 3)
    (r : ℝ) (S : Finset (Idx 2)), 3 ≤ 3 ∧ 0 < r ∧ S.Nonempty ∧
      (∀ k, (μ k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
      (∀ k, (X k : EucSpace 3) = barycenterDirection (μ k)) ∧
      (∀ k, Metric.closedBall (Z k) r ⊆ positiveQuadrant 3) ∧
      (∀ j k, k ≠ j → ∀ α β : ℝ,
        (X k : EucSpace 3) ≠ α • (X j : EucSpace 3) + β • (Z j : EucSpace 3)) ∧
      ∀ j k, k ≠ j → ∀ m : EucSpace 3, dist m (Z k : EucSpace 3) ≤ r →
        ∀ α β : ℝ, m ≠ α • (X j : EucSpace 3) + β • (Z j : EucSpace 3) := by
  let X (k : Idx 2) := quadrantThreePoint k.castSucc
  let μ (k : Idx 2) := Perspective.diracProb 3 (X k)
  have hinj : Function.Injective X := by
    intro k l he
    exact Fin.castSucc_injective 2 (quadrantThreePoint_injective he)
  have hQ (k : Idx 2) : X k ∈ positiveQuadrant 3 := quadrantThreePoint_mem k.castSucc
  obtain ⟨Z, r, _, hr, _, hcap, _, hold, hnew, _, _⟩ :=
    exists_robust_quadrant_matching_family (d := 3) le_rfl (by decide) X X hinj hinj hQ hQ
  refine ⟨μ, X, Z, r, {0}, le_rfl, hr, by simp, ?_, ?_, hcap, hold, hnew⟩
  · intro k x hx
    rw [eq_of_mem_support_dirac hx]
    exact hQ k
  · intro k
    simp [μ, barycenterDirection, barycenter, Perspective.diracProb, mem_sphere_zero_iff_norm.mp (X k).2]

end Transformer.Interpolation
