/-
# Original quadrant law matching with a linear phase count

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, §3.2,
separation and selective contraction, and §5, concentration before
matching. A family with non-collinear original means first reaches
robust generic intermediate caps, then reaches its final target caps.
Both finite inductions use at most six phases per original law.

For arbitrary distinct original quadrant laws, the proved small-ridge
refinement makes their means non-collinear in at most N additional
phases, retaining every positive-quadrant support. The resulting
full-network construction has at most thirteen phases per law.
Every phase has its actual original weak law and zero score matrix.

The target support radius is arbitrary and positive. These are
existence and phase-count statements; the quantitative parameter-norm
estimate is a further part of the source's main theorem.
-/

import Transformer.Interpolation.Section4_LawsToGeneric
import Transformer.Interpolation.Section4_LawsToTargets
import Transformer.Interpolation.Section3_RidgeClassSequence

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Original quadrant laws with non-collinear means can all be
concentrated near their distinct positive targets in twelve phases
per law. Source auxiliary: arXiv:2411.04551v3, §1, §3.2 and §5. -/
theorem exists_noncolinear_quadrant_target_phases (hd : 3 ≤ d) (hN : 1 ≤ N) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hsep : ∀ k l, k ≠ l → ∀ γ : ℝ, barycenter d (μ k) ≠ γ • barycenter d (μ l))
    (Y : Idx N → SSphere d) (hYinj : Function.Injective Y) (hY : ∀ k, Y k ∈ positiveQuadrant d)
    (ε : ℝ) (hε : 0 < ε) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 12 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) ε := by
  classical
  choose X hp hX using fun k => exists_positive_barycenter_pole (by omega : 1 ≤ d) (μ k) (hμ k)
  have hXinj : Function.Injective X := by
    intro k l he
    by_contra hkl
    have hdir : barycenterDirection (μ k) = barycenterDirection (μ l) :=
      (hp k).symm.trans ((congrArg (fun p : SSphere d => (p : EucSpace d)) he).trans (hp l))
    obtain ⟨γ, hγ⟩ := barycenter_colinear_of_direction_eq (μ k) (μ l) hdir
    exact hsep k l hkl γ hγ
  obtain ⟨Z, r, _, hr, hr1, hZ, hYcap, hold, hprocessed, hnew, htarget⟩ :=
    exists_robust_quadrant_matching_family hd hN X Y hXinj hYinj hX hY
  obtain ⟨A, hAlen, hA, hAfinal⟩ :=
    exists_laws_to_generic_phases hd i μ hμ X Z hp r hr hZ hold hprocessed
  let ν (k : Idx N) := phaseSequenceLaw A (μ k)
  have hδ : 0 < min r ε := by positivity
  obtain ⟨M, hMlen, hM, hMfinal⟩ := exists_laws_to_targets_phases hd i ν Z Y r hr hr1
    hAfinal hZ hYcap hnew htarget (min r ε) hδ (min_le_left _ _)
  refine ⟨A ++ M, ?_, ?_, ?_⟩
  · rw [List.length_append]
    omega
  · intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · exact hA P hP
    · exact hM P hP
  · intro k x hx
    rw [phaseSequenceLaw_append] at hx
    exact (Metric.mem_closedBall.mp (hMfinal k hx)).trans (min_le_right r ε)

/-- Distinct original quadrant laws admit actual phases placing all
supports near their distinct positive targets in at most thirteen
phases per law. Source auxiliary: arXiv:2411.04551v3, §1 and §3.2;
the finite ridge refinement handles equal original means. -/
theorem exists_quadrant_target_phases (hd : 3 ≤ d) (hN : 1 ≤ N) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective μ) (Y : Idx N → SSphere d)
    (hYinj : Function.Injective Y) (hY : ∀ k, Y k ∈ positiveQuadrant d) (ε : ℝ) (hε : 0 < ε) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 13 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) ε := by
  obtain ⟨R, hRlen, hQ, _, hsep⟩ := exists_ridge_sequence_noncolinear i μ hμ hinj 1 zero_lt_one
  have hRlen' : R.length ≤ N := by simpa only [Fintype.card_fin] using hRlen
  let ν (k : Idx N) := ridgeSequenceLaw R 1 (μ k)
  obtain ⟨L, hlen, hL, hfinal⟩ := exists_noncolinear_quadrant_target_phases hd hN i ν hQ hsep Y hYinj hY ε hε
  refine ⟨ridgePhaseList i R ++ L, ?_, ?_, ?_⟩
  · rw [List.length_append, ridgePhaseList_length]
    omega
  · intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · exact ridgePhaseList_valid i R P hP
    · exact hL P hP
  · intro k
    rw [phaseSequenceLaw_append, phaseSequenceLaw_ridges]
    exact hfinal k

/-- One original control curve realizes the whole quadrant-to-target
construction at every positive physical horizon. Original global
weak solutions exist from every law, and every weak endpoint has
the prescribed target support bound. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, and §5. -/
theorem exists_quadrant_target_controls (hd : 3 ≤ d) (hN : 1 ≤ N) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective μ) (Y : Idx N → SSphere d)
    (hYinj : Function.Injective Y) (hY : ∀ k, Y k ∈ positiveQuadrant d)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 13 * N + 1 ∧ PiecewiseConstant d θ T K ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ ν : ℝ → Perspective.ProbSphere d,
        ν 0 = ρ ∧ cauchyPB d θ ν) ∧
      ∀ k, ∀ ν : ℝ → Perspective.ProbSphere d, ν 0 = μ k → cauchyPB d θ ν →
        (ν T : Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) ε := by
  obtain ⟨L, hlen, hL, hfinal⟩ := exists_quadrant_target_phases hd hN i μ hμ hinj Y hYinj hY ε hε
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, L.length + 1, by omega, hpc, hB, hex, ?_⟩
  intro k ν h0 hν
  rw [hend ν hν, h0]
  exact hfinal k

/-- Two genuine distinct positive original atom laws and positive
targets satisfy all refinement, non-collinearity and prescribed-time
hypotheses together. -/
example : ∃ (μ : Idx 2 → Perspective.ProbSphere 3) (Y : Idx 2 → SSphere 3),
    3 ≤ 3 ∧ 1 ≤ 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ Function.Injective μ ∧
      (∀ k, (μ k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
      Function.Injective Y ∧ (∀ k, Y k ∈ positiveQuadrant 3) ∧
      ∀ k l, k ≠ l → ∀ γ : ℝ, barycenter 3 (μ k) ≠ γ • barycenter 3 (μ l) := by
  let Y (k : Idx 2) := quadrantThreePoint k.castSucc
  let μ (k : Idx 2) := Perspective.diracProb 3 (Y k)
  have hYinj : Function.Injective Y := by
    intro k l he
    exact Fin.castSucc_injective 2 (quadrantThreePoint_injective he)
  have hY (k : Idx 2) : Y k ∈ positiveQuadrant 3 := quadrantThreePoint_mem k.castSucc
  have hb (k : Idx 2) : barycenter 3 (μ k) = (Y k : EucSpace 3) := by
    simp [μ, barycenter, Perspective.diracProb]
  refine ⟨μ, Y, le_rfl, by decide, one_pos, one_pos, ?_, ?_, hYinj, hY, ?_⟩
  · intro k l he
    have hh := congrArg (barycenter 3) he
    rw [hb k, hb l] at hh
    exact hYinj (Subtype.ext hh)
  · intro k x hx
    rw [eq_of_mem_support_dirac hx]
    exact hY k
  · intro k l hkl γ he
    rw [hb k, hb l] at he
    exact quadrant_unit_noncollinear 0 (Y l) (Y k) (hY l) (hY k) (fun h => hkl (hYinj h)) γ he

end Transformer.Interpolation
