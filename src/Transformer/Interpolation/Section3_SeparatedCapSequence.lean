module

public import Transformer.Interpolation.Section3_SelectivePhaseClocks
public import Transformer.Interpolation.Section3_SeparatedCaps

/-!
# Finite contraction into prescribed disjoint quadrant caps

ArXiv:2411.04551v3, §3.2, `prop: separation`.
Process a finite set of indices using the actual three-phase maps.
Processed laws remain in their prescribed caps; unprocessed laws are
restored exactly. The normalized mean of each processed law remains
in its own cap. Thus the original means stay non-collinear throughout.

This supplies the finite selective contraction without assuming that a
gated perceptron preserves a mean ray. Every concrete phase retains
its original weak law and zero scores. Three phases per input suffice.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Disjoint small chordal caps give disjoint original geodesic hulls.
Source: arXiv:2411.04551v3, §3.2, the final support separation. -/
theorem disjoint_convG_of_separated_caps (p q : SSphere d) (A B : Set (SSphere d))
    (r : ℝ) (hr : 0 ≤ r) (hr1 : r ≤ 1)
    (hA : A ⊆ Metric.closedBall p r) (hB : B ⊆ Metric.closedBall q r)
    (hdisj : Disjoint (Metric.closedBall p r) (Metric.closedBall q r)) :
    Disjoint (convG d A) (convG d B) := by
  apply Set.disjoint_left.mpr
  intro x hx hy
  have hxcap : x ∈ Metric.closedBall p r :=
    convG_chordal_radius p A r hr hr1 (fun _ ha => hA ha) x hx
  have hycap : x ∈ Metric.closedBall q r :=
    convG_chordal_radius q B r hr hr1 (fun _ hb => hB hb) x hy
  exact Set.disjoint_left.mp hdisj hxcap hycap

/-- Actual selective phases place every input in its prescribed disjoint
quadrant cap, using three phases per law. Source: arXiv:2411.04551v3,
§3.2, the support contraction in `prop: separation`. -/
theorem exists_separated_cap_phase_sequence (hd : 2 ≤ d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (p : Idx N → SSphere d) (hp : ∀ k, (p k : EucSpace d) = barycenterDirection (μ k))
    (r : ℝ) (hr : 0 < r) (hr1 : r ≤ 1)
    (hcap : ∀ k, Metric.closedBall (p k) r ⊆ positiveQuadrant d)
    (hdisj : ∀ k l, k ≠ l → Disjoint (Metric.closedBall (p k) r) (Metric.closedBall (p l) r)) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 3 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ Metric.closedBall (p k) r := by
  classical
  have haux (S : Finset (Idx N)) : ∃ L : List (AutonomousPhase d),
      L.length ≤ 3 * S.card ∧ (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      (∀ k ∈ S, (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ Metric.closedBall (p k) r) ∧
      (∀ k, k ∉ S → phaseSequenceLaw L (μ k) = μ k) := by
    induction S using Finset.induction_on with
    | empty =>
      exact ⟨[], by simp, (by simp), (by simp), fun _ _ => rfl⟩
    | @insert j S hj ih =>
      obtain ⟨L, hlen, hL, hdone, hrest⟩ := ih
      let ν (k : Idx N) := phaseSequenceLaw L (μ k)
      have hQ (k : Idx N) : (ν k : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
        by_cases hk : k ∈ S
        · exact fun _ hx => hcap k (hdone k hk hx)
        · change (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ _
          rw [hrest k hk]
          exact hμ k
      have hpole (k : Idx N) : ∃ q : SSphere d,
          (q : EucSpace d) = barycenterDirection (ν k) ∧ q ∈ Metric.closedBall (p k) r := by
        obtain ⟨q, hq, -⟩ := exists_positive_barycenter_pole (by omega : 1 ≤ d) (ν k) (hQ k)
        refine ⟨q, hq, ?_⟩
        by_cases hk : k ∈ S
        · exact barycenter_pole_mem_closedBall (p k) q (ν k) r hr.le hr1 (hdone k hk) hq
        · have he : q = p k := Subtype.ext (hq.trans (by
            change barycenterDirection (phaseSequenceLaw L (μ k)) = (p k : EucSpace d)
            rw [hrest k hk, hp k]))
          rw [he]
          simpa only [Metric.mem_closedBall, dist_self] using hr.le
      have hsep (k : Idx N) (hk : k ≠ j) (γ : ℝ) : barycenter d (ν k) ≠ γ • barycenter d (ν j) := by
        apply (barycenterDirection_ne_iff_noncolinear ⟨0, by omega⟩ (ν k) (ν j) (hQ k) (hQ j)).mp ?_ γ
        intro he
        obtain ⟨q, hq, hqcap⟩ := hpole k
        obtain ⟨z, hz, hzcap⟩ := hpole j
        have hqz : q = z := Subtype.ext (hq.trans (he.trans hz.symm))
        rw [← hqz] at hzcap
        exact (Set.disjoint_left.mp (hdisj k j hk)) hqcap hzcap
      have hpj : (p j : EucSpace d) = barycenterDirection (ν j) := by
        change (p j : EucSpace d) = barycenterDirection (phaseSequenceLaw L (μ j))
        rw [hrest j hj, hp j]
      obtain ⟨M, hMlen, hM, hselected, hrestore⟩ :=
        exists_target_contraction_phases hd j ν hQ hsep (p j) hpj r hr
      refine ⟨L ++ M, ?_, ?_, ?_, ?_⟩
      · rw [List.length_append, hMlen, Finset.card_insert_of_notMem hj]
        omega
      · intro P hP
        rcases List.mem_append.mp hP with hP | hP
        · exact hL P hP
        · exact hM P hP
      · intro k hk
        rw [phaseSequenceLaw_append]
        rcases Finset.mem_insert.mp hk with rfl | hk
        · exact hselected
        · by_cases hkj : k = j
          · subst k
            exact hselected
          · change (phaseSequenceLaw M (ν k) : Measure (SSphere d)).support ⊆ _
            rw [hrestore k hkj]
            exact hdone k hk
      · intro k hk
        have hkj : k ≠ j := fun he => hk (Finset.mem_insert.mpr (Or.inl he))
        have hkS : k ∉ S := fun he => hk (Finset.mem_insert.mpr (Or.inr he))
        rw [phaseSequenceLaw_append]
        change phaseSequenceLaw M (ν k) = μ k
        rw [hrestore k hkj]
        exact hrest k hkS
  obtain ⟨L, hlen, hL, hdone, -⟩ := haux Finset.univ
  refine ⟨L, ?_, hL, fun k => hdone k (Finset.mem_univ k)⟩
  simpa only [Finset.card_univ, Fintype.card_fin] using hlen

/-- Two distinct positive circle atoms and the proved common cap radius
jointly satisfy every hypothesis, with both indices present. -/
example : ∃ (μ : Idx 2 → Perspective.ProbSphere 2) (p : Idx 2 → SSphere 2) (r : ℝ),
    (2 : ℕ) ≤ 2 ∧
    (∀ k, (μ k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2) ∧
    (∀ k, (p k : EucSpace 2) = barycenterDirection (μ k)) ∧ 0 < r ∧ r ≤ 1 ∧
    (∀ k, Metric.closedBall (p k) r ⊆ positiveQuadrant 2) ∧
    (∀ k, (μ k : Measure (SSphere 2)).support ⊆ Metric.closedBall (p k) r) ∧
    ∀ k l, k ≠ l → Disjoint (Metric.closedBall (p k) r) (Metric.closedBall (p l) r) := by
  let a : SSphere 2 := circlePt (3 / 5) (4 / 5) (by norm_num)
  let b : SSphere 2 := circlePt (4 / 5) (3 / 5) (by norm_num)
  let p : Idx 2 → SSphere 2 := ![a, b]
  let μ (k : Idx 2) := Perspective.diracProb 2 (p k)
  have hp (k : Idx 2) : p k ∈ positiveQuadrant 2 := by
    intro i
    fin_cases k <;> fin_cases i <;> norm_num [p, a, b, circlePt]
  have hinj : Function.Injective p := by
    intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · have hh := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 0) he
      norm_num [p, a, b, circlePt] at hh
    · have hh := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 0) he
      norm_num [p, a, b, circlePt] at hh
    · rfl
  have hQ (k : Idx 2) : (μ k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    intro x hx
    rw [eq_of_mem_support_dirac hx]
    exact hp k
  have hdir (k : Idx 2) : (p k : EucSpace 2) = barycenterDirection (μ k) := by
    simp [barycenterDirection, μ, barycenter, Perspective.diracProb,
      mem_sphere_zero_iff_norm.mp (p k).2]
  obtain ⟨r, hr, hr1, hcap, hdisj⟩ := exists_separated_quadrant_caps p hp hinj
  have hsame (k : Idx 2) : (μ k : Measure (SSphere 2)).support ⊆ Metric.closedBall (p k) r := by
    intro x hx
    rw [eq_of_mem_support_dirac hx]
    simpa only [Metric.mem_closedBall, dist_self] using hr.le
  exact ⟨μ, p, r, le_rfl, hQ, hdir, hr, hr1, hcap, hsame, hdisj⟩

end Transformer.Interpolation
