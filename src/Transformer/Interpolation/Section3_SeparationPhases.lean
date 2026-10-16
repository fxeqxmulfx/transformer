/-
# Actual phase sequences separating all original geodesic hulls

ArXiv:2411.04551v3, §3.2, `prop: separation`.
For non-collinear original means, choose their positive unit poles and
one common radius giving pairwise disjoint quadrant caps. The finite
selective induction puts each actual final support in its own cap.
The original geodesic hulls lie in those same disjoint caps.

For arbitrary distinct quadrant laws, first refine the finite classes
of original mean directions. At most N actual ridge phases suffice.
The selective stage uses at most 3N further actual phases. Every phase
has its original weak law and zero scores; the final phase list has
length at most 4N, independently of dimension and the particular data.
-/

import Transformer.Interpolation.Section3_SeparatedCapSequence

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Non-collinear original means admit actual selective phases with
disjoint positive-quadrant caps supporting every final law.
Source: arXiv:2411.04551v3, §3.2, `prop: separation`. -/
theorem exists_noncolinear_separated_cap_phases (hd : 2 ≤ d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hsep : ∀ k l, k ≠ l → ∀ γ : ℝ, barycenter d (μ k) ≠ γ • barycenter d (μ l)) :
    ∃ (L : List (AutonomousPhase d)) (p : Idx N → SSphere d) (r : ℝ),
      L.length ≤ 3 * N ∧ (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      0 < r ∧ r ≤ 1 ∧ (∀ k, Metric.closedBall (p k) r ⊆ positiveQuadrant d) ∧
      (∀ k l, k ≠ l → Disjoint (Metric.closedBall (p k) r) (Metric.closedBall (p l) r)) ∧
      ∀ k, (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ Metric.closedBall (p k) r := by
  classical
  choose p hp hppos using fun k => exists_positive_barycenter_pole (by omega : 1 ≤ d) (μ k) (hμ k)
  have hinj : Function.Injective p := by
    intro k l he
    by_contra hkl
    have hdir : barycenterDirection (μ k) = barycenterDirection (μ l) :=
      (hp k).symm.trans ((congrArg (fun q : SSphere d => (q : EucSpace d)) he).trans (hp l))
    obtain ⟨γ, hγ⟩ := barycenter_colinear_of_direction_eq (μ k) (μ l) hdir
    exact hsep k l hkl γ hγ
  obtain ⟨r, hr, hr1, hcap, hdisj⟩ := exists_separated_quadrant_caps p hppos hinj
  obtain ⟨L, hlen, hL, hfinal⟩ := exists_separated_cap_phase_sequence hd μ hμ p hp r hr hr1 hcap hdisj
  exact ⟨L, p, r, hlen, hL, hr, hr1, hcap, hdisj, hfinal⟩

/-- Non-collinear original means can be separated in at most 3N actual
autonomous phases. Source: arXiv:2411.04551v3, §3.2, `prop: separation`. -/
theorem exists_noncolinear_separation_phases (hd : 2 ≤ d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hsep : ∀ k l, k ≠ l → ∀ γ : ℝ, barycenter d (μ k) ≠ γ • barycenter d (μ l)) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 3 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k l, k ≠ l → Disjoint
        (convG d (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support)
        (convG d (phaseSequenceLaw L (μ l) : Measure (SSphere d)).support) := by
  obtain ⟨L, p, r, hlen, hL, hr, hr1, -, hdisj, hfinal⟩ :=
    exists_noncolinear_separated_cap_phases hd μ hμ hsep
  exact ⟨L, hlen, hL, fun k l hkl => disjoint_convG_of_separated_caps (p k) (p l)
    _ _ r hr.le hr1 (hfinal k) (hfinal l) (hdisj k l hkl)⟩

/-- Distinct quadrant laws admit at most 4N actual zero-score phases
separating their original geodesic support hulls. Source:
arXiv:2411.04551v3, §3.2, `prop: separation`; finite small ridge
refinement replaces the source's false fixed-outside-hulls clause. -/
theorem exists_separation_phases (hd : 2 ≤ d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective μ) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 4 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k l, k ≠ l → Disjoint
        (convG d (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support)
        (convG d (phaseSequenceLaw L (μ l) : Measure (SSphere d)).support) := by
  let i : Fin d := ⟨0, by omega⟩
  obtain ⟨L, hlen, hQ, -, hsep⟩ := exists_ridge_sequence_noncolinear i μ hμ hinj 1 zero_lt_one
  have hlen' : L.length ≤ N := by simpa only [Fintype.card_fin] using hlen
  let ν (k : Idx N) := ridgeSequenceLaw L 1 (μ k)
  obtain ⟨M, hMlen, hM, hMsep⟩ := exists_noncolinear_separation_phases hd ν hQ hsep
  have hend (k : Idx N) : phaseSequenceLaw (ridgePhaseList i L ++ M) (μ k) =
      phaseSequenceLaw M (ν k) := by
    rw [phaseSequenceLaw_append, phaseSequenceLaw_ridges]
  refine ⟨ridgePhaseList i L ++ M, ?_, ?_, ?_⟩
  · rw [List.length_append, ridgePhaseList_length]
    omega
  · intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · exact ridgePhaseList_valid i L P hP
    · exact hM P hP
  · intro k l hkl
    rw [hend k, hend l]
    exact hMsep k l hkl

/-- Two distinct positive circle atoms have non-collinear original
means. They satisfy both the selective-stage and full preprocessing
hypotheses with two genuine indices. -/
example : ∃ μ : Idx 2 → Perspective.ProbSphere 2,
    (2 : ℕ) ≤ 2 ∧
    (∀ k, (μ k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2) ∧
    (∀ k l, k ≠ l → ∀ γ : ℝ, barycenter 2 (μ k) ≠ γ • barycenter 2 (μ l)) ∧
    Function.Injective μ := by
  let a : SSphere 2 := circlePt (3 / 5) (4 / 5) (by norm_num)
  let b : SSphere 2 := circlePt (4 / 5) (3 / 5) (by norm_num)
  let μ : Idx 2 → Perspective.ProbSphere 2 :=
    ![Perspective.diracProb 2 a, Perspective.diracProb 2 b]
  have hQ (k : Idx 2) : (μ k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    intro x hx
    fin_cases k
    · rw [eq_of_mem_support_dirac hx]
      exact circlePt_mem_positiveQuadrant _ (by norm_num) (by norm_num)
    · rw [eq_of_mem_support_dirac hx]
      exact circlePt_mem_positiveQuadrant _ (by norm_num) (by norm_num)
  have hsep (k l : Idx 2) (hkl : k ≠ l) (γ : ℝ) :
      barycenter 2 (μ k) ≠ γ • barycenter 2 (μ l) := by
    intro he
    have h0 := congrArg (fun x : EucSpace 2 => x 0) he
    have h1 := congrArg (fun x : EucSpace 2 => x 1) he
    fin_cases k <;> fin_cases l
    · exact hkl rfl
    · norm_num [μ, barycenter, Perspective.diracProb, a, b, circlePt] at h0 h1
      linarith
    · norm_num [μ, barycenter, Perspective.diracProb, a, b, circlePt] at h0 h1
      linarith
    · exact hkl rfl
  have hinj : Function.Injective μ := by
    intro k l he
    by_contra hkl
    exact hsep k l hkl 1 (he ▸ (one_smul ℝ (barycenter 2 (μ l))).symm)
  exact ⟨μ, le_rfl, hQ, hsep, hinj⟩

/-- The source's distinct equal-mean atomic counterexample also
satisfies the full phase theorem's hypotheses. Its equal directions
require the genuine preprocessing stage before selective contraction. -/
example : ∃ μ : Idx 2 → Perspective.ProbSphere 3,
    (2 : ℕ) ≤ 3 ∧
    (∀ k, (μ k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
    Function.Injective μ ∧ barycenter 3 (μ 0) = barycenter 3 (μ 1) := by
  obtain ⟨ν, η, hne, hν, hη, hmean, -⟩ := perturbation_equal_barycenter_counterexample
  let μ : Idx 2 → Perspective.ProbSphere 3 := ![ν, η]
  have hQ (k : Idx 2) : (μ k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 := by
    fin_cases k
    · exact hν
    · exact hη
  have hinj : Function.Injective μ := by
    intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · exact (hne he).elim
    · exact (hne he.symm).elim
    · rfl
  exact ⟨μ, by norm_num, hQ, hinj, hmean⟩

end Transformer.Interpolation
