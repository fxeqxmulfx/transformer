/-
# Existence and linear piece count for original atomic-target interpolation

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, with its
construction in §3 and §5. First send the original input laws into
the positive quadrant using their common support hole. Independently
send the finite target atoms into that quadrant by two actual flows.
Match the quadrant laws to sufficiently small target neighborhoods,
then reverse the target reduction.

The entire original phase sequence has at most thirteen N plus four
phases, all with proved actual weak laws and zero score matrices.
Its realization constructs global weak solutions at every positive
physical horizon and gives every weak solution the desired genuine
W2 target accuracy. Six dN bounds the piece count for nonempty data.

These are the existence and switch conclusions of the source theorem.
The parameter-norm conclusion requires its own quantitative estimates;
the overstrong uniform-data norm formalization is separately refuted.
-/

import Transformer.Interpolation.Section1_InverseQuadrantSupports
import Transformer.Interpolation.Section4_AtomicQuadrantReduction

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Original distinct input laws with a support hole can all be
concentrated near arbitrary distinct unit targets by actual phases.
Source: arXiv:2411.04551v3, §1, the existence and switch part of
`thm: targets.atoms`, using the actual construction of §3 and §5. -/
theorem exists_atomic_target_support_phases (hd : 3 ≤ d) (hN : 1 ≤ N) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d) (hinj : Function.Injective μ)
    (hhole : (⋃ k, (μ k : Measure (SSphere d)).support) ≠ Set.univ)
    (Y : Idx N → SSphere d) (hYinj : Function.Injective Y) (ε : ℝ) (hε : 0 < ε) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 13 * N + 4 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, (phaseSequenceLaw L (μ k) : Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) ε := by
  obtain ⟨A, hAlen, hA, hAQ⟩ := exists_quadrant_preprocessing_phases (by omega : 2 ≤ d) i μ hhole
  let ν (k : Idx N) := phaseSequenceLaw A (μ k)
  have hνinj : Function.Injective ν := by
    intro k l he
    exact hinj ((bijective_phaseSequenceLaw A (fun P hP => (hA P hP).1)).1 he)
  obtain ⟨w, _, hw⟩ := exists_finite_atomic_missing_point hd Y
  have hYproper : Set.range Y ≠ Set.univ := by
    intro he
    exact hw (he.symm ▸ Set.mem_univ _)
  obtain ⟨w₁, w₂, τ₁, τ₂, _, _, hYQ⟩ := exists_two_projection_quadrant (by omega : 2 ≤ d)
    (Set.range Y) (Set.finite_range Y).isCompact hYproper
  let Y' (k : Idx N) := sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ (Y k))
  have hY'inj : Function.Injective Y' := by
    intro k l he
    exact hYinj (injective_twoProjection_map w₁ w₂ τ₁ τ₂ he)
  have hY'Q (k : Idx N) : Y' k ∈ positiveQuadrant d := hYQ (Y k) ⟨k, rfl⟩
  obtain ⟨δ, hδ, hback⟩ := exists_inverse_twoProjection_radius i w₁ w₂ τ₁ τ₂ Y ε hε
  obtain ⟨M, hMlen, hM, hMfinal⟩ :=
    exists_quadrant_target_phases hd hN i ν hAQ hνinj Y' hY'inj hY'Q δ hδ
  let B := twoProjectionPhaseList i w₁ w₂ τ₁ τ₂
  have hB : ∀ P ∈ B, IsAutonomousPhase P ∧ P.1.B = 0 :=
    twoProjectionPhaseList_valid i w₁ w₂ τ₁ τ₂
  refine ⟨A ++ (M ++ inversePhaseList B), ?_, ?_, ?_⟩
  · rw [List.length_append, List.length_append, inversePhaseList_length, hAlen]
    have hBlen : B.length = 2 := rfl
    omega
  · intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · exact hA P hP
    · rcases List.mem_append.mp hP with hP | hP
      · exact hM P hP
      · exact inversePhaseList_valid B hB P hP
  · intro k
    rw [phaseSequenceLaw_append, phaseSequenceLaw_append]
    exact hback k _ (hMfinal k)

/-- Common actual original controls provide all target-support
conclusions at any positive prescribed horizon. Global weak solutions
exist from every law and every weak endpoint is controlled. Source:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, existence and switches. -/
theorem exists_atomic_target_support_controls (hd : 3 ≤ d) (hN : 1 ≤ N) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d) (hinj : Function.Injective μ)
    (hhole : (⋃ k, (μ k : Measure (SSphere d)).support) ≠ Set.univ)
    (Y : Idx N → SSphere d) (hYinj : Function.Injective Y)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 13 * N + 5 ∧ PiecewiseConstant d θ T K ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ ν : ℝ → Perspective.ProbSphere d,
        ν 0 = ρ ∧ cauchyPB d θ ν) ∧
      ∀ k, ∀ ν : ℝ → Perspective.ProbSphere d, ν 0 = μ k → cauchyPB d θ ν →
        (ν T : Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) ε := by
  obtain ⟨L, hlen, hL, hfinal⟩ := exists_atomic_target_support_phases hd hN i μ hinj hhole Y hYinj ε hε
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, L.length + 1, by omega, hpc, hB, hex, ?_⟩
  intro k ν h0 hν
  rw [hend ν hν, h0]
  exact hfinal k

/-- The original atomic-target theorem's existence and switch
conclusions hold with a uniform six-dN piece bound. The target
accuracy uses the genuine original W2 distance and constructed
global weak solutions. Source: arXiv:2411.04551v3, §1,
`thm: targets.atoms`, excluding its separate norm estimate. -/
theorem exists_atomic_target_accuracy_linear_pieces (hd : 3 ≤ d) (hN : 1 ≤ N)
    (μ : Idx N → Perspective.ProbSphere d) (hinj : Function.Injective μ)
    (hhole : (⋃ k, (μ k : Measure (SSphere d)).support) ≠ Set.univ)
    (Y : Idx N → SSphere d) (hYinj : Function.Injective Y)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ) (ν : Idx N → ℝ → Perspective.ProbSphere d),
      K ≤ 6 * d * N ∧ PiecewiseConstant d θ T K ∧ (∀ s, (θ s).B = 0) ∧
      (∀ k, ν k 0 = μ k ∧ cauchyPB d θ (ν k)) ∧
      ∀ k, W2 d (ν k T : Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε := by
  let i : Fin d := ⟨0, by omega⟩
  obtain ⟨θ, K, hK, hpc, hB, hex, hfinal⟩ :=
    exists_atomic_target_support_controls hd hN i μ hinj hhole Y hYinj ε T hε hT
  choose ν h0 hν using fun k => hex (μ k)
  refine ⟨θ, K, ν, hK.trans ?_, hpc, hB, (fun k => ⟨h0 k, hν k⟩), ?_⟩
  · nlinarith
  · intro k
    exact W2_le_radius_of_support (ν k T) (Y k) ε hε.le (hfinal k (ν k) (h0 k) (hν k))

/-- Two antipodal original Diracs and swapped antipodal targets
jointly satisfy all hypotheses with genuine nonempty input and
positive horizon and accuracy. No original quadrant condition holds
for these inputs, so the preprocessing case is inhabited. -/
example : ∃ (μ : Idx 2 → Perspective.ProbSphere 3) (Y : Idx 2 → SSphere 3),
    3 ≤ 3 ∧ 1 ≤ 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ Function.Injective μ ∧
      (⋃ k, (μ k : Measure (SSphere 3)).support) ≠ Set.univ ∧ Function.Injective Y := by
  let p := basePoint 2
  let X : Idx 2 → SSphere 3 := ![p, antipode 3 p]
  let μ (k : Idx 2) := Perspective.diracProb 3 (X k)
  let Y (k : Idx 2) := X (Fin.rev k)
  have hXinj : Function.Injective X := by
    intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · exact False.elim (antipode_ne 3 p he.symm)
    · exact False.elim (antipode_ne 3 p he)
    · rfl
  refine ⟨μ, Y, le_rfl, by decide, one_pos, one_pos, ?_,
    finite_atomic_support_union_ne_univ (d := 3) le_rfl X, hXinj.comp Fin.rev_injective⟩
  intro k l he
  have hb (k : Idx 2) : barycenter 3 (μ k) = (X k : EucSpace 3) := by
    simp [μ, barycenter, Perspective.diracProb]
  have hh := congrArg (barycenter 3) he
  rw [hb k, hb l] at hh
  exact hXinj (Subtype.ext hh)

end Transformer.Interpolation
