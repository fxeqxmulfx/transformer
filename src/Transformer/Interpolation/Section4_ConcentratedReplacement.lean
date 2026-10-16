/-
# Moving one original quadrant law while restoring every other law

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, §3.2,
selective contraction, and §5, the concentration-then-matching step.
Choose stable protected transfer phases using only the selected
original mean direction, the target and the other original laws.
Their support-error constant is fixed before the selected law is
contracted. Contract it to a sufficiently small neighborhood of its
original mean direction and then perform that protected transfer.

At most six actual original phases place the selected support in
any prescribed target ball and restore all other entire laws exactly.
The selected contracted law need not preserve its original mean ray;
the proved transfer estimate permits its actual changed mean.
Equal selected and target poles need only the contraction stage.

A genuine dummy law outside the selected-target plane supplies the
nonempty protected family when the actual selected family is a singleton.
-/

import Transformer.Interpolation.Section4_StableProtectedTransfers
import Transformer.Interpolation.Section4_GenericQuadrantPoints
import Transformer.Interpolation.Section3_SelectivePhaseClocks

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- One original quadrant law can be concentrated near a prescribed
positive target while every other original law is restored exactly.
Source auxiliary: arXiv:2411.04551v3, §1, §3.2 and §5; the
other original means must avoid the selected-mean-target plane. -/
theorem exists_quadrant_law_replacement_phases (hd : 3 ≤ d) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (j : Idx N) (p q : SSphere d) (hp : (p : EucSpace d) = barycenterDirection (μ j))
    (hq : q ∈ positiveQuadrant d)
    (hplane : ∀ k, k ≠ j → ∀ α β : ℝ,
      barycenter d (μ k) ≠ α • (p : EucSpace d) + β • (q : EucSpace d))
    (ε : ℝ) (hε : 0 < ε) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 6 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      (phaseSequenceLaw L (μ j) : Measure (SSphere d)).support ⊆ Metric.closedBall q ε ∧
      ∀ k, k ≠ j → phaseSequenceLaw L (μ k) = μ k := by
  classical
  have hsep (k : Idx N) (hkj : k ≠ j) (γ : ℝ) :
      barycenter d (μ k) ≠ γ • barycenter d (μ j) := by
    intro he
    apply hplane k hkj (γ * ‖barycenter d (μ j)‖) 0
    rw [zero_smul, add_zero, ← smul_smul, hp, norm_smul_barycenterDirection]
    exact he
  have hpQ : p ∈ positiveQuadrant d := by
    intro k
    rw [hp]
    exact barycenterDirection_coordinate_pos (μ j) (hμ j) k
  by_cases hpq : p = q
  · obtain ⟨L, hlen, hL, hselected, hrest⟩ := exists_target_contraction_phases
      (by omega : 2 ≤ d) j μ hμ hsep p hp ε hε
    refine ⟨L, by omega, hL, ?_, hrest⟩
    simpa only [hpq] using hselected
  obtain ⟨r, hrQ, hr⟩ := exists_quadrant_point_avoiding_planes hd
    (fun _ : Fin 1 => (p : EucSpace d)) (fun _ : Fin 1 => (q : EucSpace d))
  let J := Option {k : Idx N // k ≠ j}
  let η : J → Perspective.ProbSphere d := fun k => match k with
    | none => Perspective.diracProb d r
    | some k => μ k.1
  have hηQ (k : J) : (η k : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
    cases k with
    | none =>
      intro z hz
      rw [eq_of_mem_support_dirac hz]
      exact hrQ
    | some k => exact hμ k.1
  have hηplane (k : J) (α β : ℝ) :
      barycenter d (η k) ≠ α • (p : EucSpace d) + β • (q : EucSpace d) := by
    cases k with
    | none =>
      have hb : barycenter d (η none) = (r : EucSpace d) := by
        simp [η, barycenter, Perspective.diracProb]
      rw [hb]
      exact hr 0 α β
    | some k => exact hplane k.1 k.2 α β
  obtain ⟨M, C, hMlen, hM, hC, _, hrestM, hbound⟩ :=
    exists_stable_quadrant_transfer_phases i p q hpQ hq hpq η hηQ hηplane
  let δ := ε / C
  have hδ : 0 < δ := div_pos hε hC
  obtain ⟨L, hlen, hL, hselected, hrestL⟩ := exists_target_contraction_phases
    (by omega : 2 ≤ d) j μ hμ hsep p hp δ hδ
  refine ⟨L ++ M, ?_, ?_, ?_, ?_⟩
  · rw [List.length_append, hlen, hMlen]
  · intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · exact hL P hP
    · exact hM P hP
  · rw [phaseSequenceLaw_append]
    have he : C * δ = ε := by dsimp only [δ]; field_simp
    simpa only [he] using hbound δ hδ.le (phaseSequenceLaw L (μ j)) hselected
  · intro k hkj
    rw [phaseSequenceLaw_append, hrestL k hkj]
    exact hrestM (some ⟨k, hkj⟩)

/-- A positive prescribed horizon admits the same original one-law
replacement with at most seven constant pieces and weak existence
from every input. Every original weak endpoint obeys the support
and restoration conclusions. Source auxiliary: arXiv:2411.04551v3,
§1, §3.2 and §5, physical-time realization. -/
theorem exists_quadrant_law_replacement_controls (hd : 3 ≤ d) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (j : Idx N) (p q : SSphere d) (hp : (p : EucSpace d) = barycenterDirection (μ j))
    (hq : q ∈ positiveQuadrant d)
    (hplane : ∀ k, k ≠ j → ∀ α β : ℝ,
      barycenter d (μ k) ≠ α • (p : EucSpace d) + β • (q : EucSpace d))
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 7 ∧ PiecewiseConstant d θ T K ∧ (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ ν : ℝ → Perspective.ProbSphere d,
        ν 0 = ρ ∧ cauchyPB d θ ν) ∧
      (∀ ν : ℝ → Perspective.ProbSphere d, ν 0 = μ j → cauchyPB d θ ν →
        (ν T : Measure (SSphere d)).support ⊆ Metric.closedBall q ε) ∧
      ∀ k, k ≠ j → ∀ ν : ℝ → Perspective.ProbSphere d, ν 0 = μ k → cauchyPB d θ ν → ν T = μ k := by
  obtain ⟨L, hlen, hL, hselected, hrest⟩ :=
    exists_quadrant_law_replacement_phases hd i μ hμ j p q hp hq hplane ε hε
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, L.length + 1, by omega, hpc, hB, hex, ?_, ?_⟩
  · intro ν h0 hν
    rw [hend ν hν, h0]
    exact hselected
  · intro k hkj ν h0 hν
    rw [hend ν hν, h0, hrest k hkj]

/-- Genuine two-law positive data and a third positive target meet
all mean-direction, plane, concentration and time hypotheses. -/
example : ∃ (μ : Idx 2 → Perspective.ProbSphere 3) (p q : SSphere 3),
    3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (∀ k, (μ k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
      (p : EucSpace 3) = barycenterDirection (μ 0) ∧ q ∈ positiveQuadrant 3 ∧
      ∀ k, k ≠ 0 → ∀ α β : ℝ,
        barycenter 3 (μ k) ≠ α • (p : EucSpace 3) + β • (q : EucSpace 3) := by
  let p := quadrantThreePoint 0
  let q := quadrantThreePoint 1
  let μ : Idx 2 → Perspective.ProbSphere 3 :=
    ![Perspective.diracProb 3 p, Perspective.diracProb 3 (quadrantThreePoint 2)]
  refine ⟨μ, p, q, le_rfl, one_pos, one_pos, ?_, ?_, quadrantThreePoint_mem 1, ?_⟩
  · intro k x hx
    fin_cases k
    · rw [eq_of_mem_support_dirac hx]
      exact quadrantThreePoint_mem 0
    · rw [eq_of_mem_support_dirac hx]
      exact quadrantThreePoint_mem 2
  · simp [μ, barycenterDirection, barycenter, Perspective.diracProb, mem_sphere_zero_iff_norm.mp p.2]
  · intro k hk α β
    fin_cases k
    · exact False.elim (hk rfl)
    · have hb : barycenter 3 (μ 1) = (quadrantThreePoint 2 : EucSpace 3) := by
        simp [μ, barycenter, Perspective.diracProb]
      change barycenter 3 (μ 1) ≠ α • (p : EucSpace 3) + β • (q : EucSpace 3)
      rw [hb]
      exact quadrantThreePoint_not_plane 0 1 2 (by decide) (by decide) (by decide) α β

end Transformer.Interpolation
