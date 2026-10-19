module

public import Transformer.Interpolation.Section4_QuadrantTransfers
public import Transformer.Interpolation.Section4_TransferStability

/-!
# Protected original transfers with uniform concentration error

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, its proof in §5,
and §3.2, the attention isolation construction. Choose the exact
gated transfer and gather all unselected laws near the two inactive
poles. The resulting three actual phases restore those entire laws
exactly and have a radius constant for every selected input law.

The same phases and constant are chosen before the selected input
law or its concentration radius. Forward attention can therefore
change that input's mean: the support estimate already accounts for
the actual nonlinear clock. Concentration can be performed afterward
while keeping the unselected laws fixed.

Positive-quadrant endpoints and unselected laws satisfy all geometric
premises whenever their original means avoid the selected plane.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ} {ι : Type*}

/-- Actual protected phases simultaneously give exact restoration
and a uniform support error for every concentrated selected law.
Source auxiliary: arXiv:2411.04551v3, §1, §3.2 and §5,
combining protected interpolation with its concentration estimate. -/
theorem exists_stable_protected_transfer_phases [Fintype ι]
    (i : Fin d) (p q w : SSphere d) (hpq : p ≠ q) (hanti : q ≠ antipode d p)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (η : ι → Perspective.ProbSphere d)
    (hη : ∀ k, meanAxisMoment (w : EucSpace d) (η k) ≠ 0)
    (hmiss : ∀ k, w ∉ (η k : Measure (SSphere d)).support ∧
      antipode d w ∉ (η k : Measure (SSphere d)).support) :
    ∃ (L : List (AutonomousPhase d)) (C : ℝ), L.length = 3 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧ 0 < C ∧
      phaseSequenceLaw L (Perspective.diracProb d p) = Perspective.diracProb d q ∧
      (∀ k, phaseSequenceLaw L (η k) = η k) ∧
      ∀ δ : ℝ, 0 ≤ δ → ∀ ν : Perspective.ProbSphere d,
        (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p δ →
        (phaseSequenceLaw L ν : Measure (SSphere d)).support ⊆ Metric.closedBall q (C * δ) := by
  obtain ⟨a, c, T, hc, _, hend, _⟩ := exists_exact_gated_transfer p q hpq hanti
  let m : EucSpace d := (p : EucSpace d) + (q : EucSpace d)
  have hwm : inner ℝ (w : EucSpace d) m = 0 := by
    simp only [m, inner_add_right, hwp, hwq, add_zero]
  let r := c / (2 * (‖m‖ + 1))
  have hr : 0 < r := by dsimp only [r]; positivity
  have hrad : r * ‖m‖ ≤ c := by
    have he : r * (2 * (‖m‖ + 1)) = c := div_mul_cancel₀ c (by positivity)
    nlinarith [mul_nonneg hr.le (norm_nonneg m)]
  have he : ∀ᶠ t : ℝ in Filter.atTop, ∀ k,
      (meanAxisLaw w 1 (η k) t : Measure (SSphere d)).support ⊆
        Metric.ball w r ∪ Metric.ball (antipode d w) r :=
    Filter.eventually_all.mpr (fun k => eventually_meanAxisLaw_two_poles w 1 (η k)
      one_pos (hη k) (hmiss k).1 (hmiss k).2 r hr)
  obtain ⟨t₁, hpre, ht₁⟩ := (he.and (Filter.eventually_gt_atTop (0 : ℝ))).exists
  obtain ⟨C, hC, hbound⟩ := exists_transfer_phase_radius_scale i (a : EucSpace d) m w p q
    c t₁ T ht₁.le hwp hwq hend
  refine ⟨transferPhaseList i (a : EucSpace d) m w c t₁ T, C,
    transferPhaseList_length i _ _ _ _ _ _, transferPhaseList_valid i _ _ _ _ _ _, hC,
    transferPhaseList_dirac i _ _ w p q c t₁ T hwp hwq hend, ?_, hbound⟩
  intro k
  apply transferPhaseList_restore_fixed
  exact ridgeLaw_fixed_of_two_pole_support (a : EucSpace d) m w c r T hwm hrad _ (hpre k)

/-- Actual positive-quadrant data admit exact restoration and uniform
selected-law error with three phases, under the original mean-plane
condition. Source auxiliary: arXiv:2411.04551v3, §1, §3.2 and §5. -/
theorem exists_stable_quadrant_transfer_phases [Fintype ι] [Nonempty ι]
    (i : Fin d) (p q : SSphere d) (hp : p ∈ positiveQuadrant d)
    (hq : q ∈ positiveQuadrant d) (hpq : p ≠ q) (η : ι → Perspective.ProbSphere d)
    (hQ : ∀ k, (η k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hplane : ∀ k, ∀ α β : ℝ, barycenter d (η k) ≠ α • (p : EucSpace d) + β • (q : EucSpace d)) :
    ∃ (L : List (AutonomousPhase d)) (C : ℝ), L.length = 3 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧ 0 < C ∧
      phaseSequenceLaw L (Perspective.diracProb d p) = Perspective.diracProb d q ∧
      (∀ k, phaseSequenceLaw L (η k) = η k) ∧
      ∀ δ : ℝ, 0 ≤ δ → ∀ ν : Perspective.ProbSphere d,
        (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p δ →
        (phaseSequenceLaw L ν : Measure (SSphere d)).support ⊆ Metric.closedBall q (C * δ) := by
  obtain ⟨w, hwp, hwq, hw⟩ := exists_common_unit_mean_plane_normal p (q : EucSpace d) η hplane
  exact exists_stable_protected_transfer_phases i p q w hpq (quadrant_ne_antipode i p q hp hq)
    hwp hwq η hw (fun k => orthogonal_poles_not_support i (p : EucSpace d) hp w hwp (η k) (hQ k))

/-- The same fixed stable transfer has actual common controls at
every positive physical horizon. Every original weak solution
satisfies the restoration or selected-support conclusion. Source
auxiliary: arXiv:2411.04551v3, §1, §3.2 and §5. -/
theorem exists_stable_protected_transfer_controls [Fintype ι]
    (i : Fin d) (p q w : SSphere d) (hpq : p ≠ q) (hanti : q ≠ antipode d p)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (η : ι → Perspective.ProbSphere d)
    (hη : ∀ k, meanAxisMoment (w : EucSpace d) (η k) ≠ 0)
    (hmiss : ∀ k, w ∉ (η k : Measure (SSphere d)).support ∧
      antipode d w ∉ (η k : Measure (SSphere d)).support) (H : ℝ) (hH : 0 < H) :
    ∃ (θ : TimeParams d) (C : ℝ), PiecewiseConstant d θ H 4 ∧ (∀ s, (θ s).B = 0) ∧ 0 < C ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      (∀ k, ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = η k → cauchyPB d θ μ → μ H = η k) ∧
      ∀ δ : ℝ, 0 ≤ δ → ∀ ν : Perspective.ProbSphere d,
        (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p δ →
        ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d θ μ →
          (μ H : Measure (SSphere d)).support ⊆ Metric.closedBall q (C * δ) := by
  obtain ⟨L, C, hlen, hL, hC, _, hrest, hbound⟩ :=
    exists_stable_protected_transfer_phases i p q w hpq hanti hwp hwq η hη hmiss
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL H hH
  refine ⟨θ, C, by simpa only [hlen] using hpc, hB, hC, hex, ?_, ?_⟩
  · intro k μ h0 hμ
    rw [hend μ hμ, h0, hrest k]
  · intro δ hδ ν hν μ h0 hμ
    rw [hend μ hμ, h0]
    exact hbound δ hδ ν hν

/-- Three genuine positive rational unit points jointly realize all
plane, quadrant and protected-law hypotheses. A constructed transverse
unit axis gives the nonzero axial mean and misses both support poles. -/
example : ∃ (p q w : SSphere 3) (η : Fin 1 → Perspective.ProbSphere 3),
    (0 : ℝ) < 1 ∧ p ∈ positiveQuadrant 3 ∧ q ∈ positiveQuadrant 3 ∧ p ≠ q ∧
      q ≠ antipode 3 p ∧ inner ℝ (w : EucSpace 3) (p : EucSpace 3) = 0 ∧
      inner ℝ (w : EucSpace 3) (q : EucSpace 3) = 0 ∧
      (∀ k, (η k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
      (∀ k, ∀ α β : ℝ, barycenter 3 (η k) ≠ α • (p : EucSpace 3) + β • (q : EucSpace 3)) ∧
      (∀ k, meanAxisMoment (w : EucSpace 3) (η k) ≠ 0) ∧
      ∀ k, w ∉ (η k : Measure (SSphere 3)).support ∧
        antipode 3 w ∉ (η k : Measure (SSphere 3)).support := by
  let p := quadrantThreePoint 0
  let q := quadrantThreePoint 1
  let η : Fin 1 → Perspective.ProbSphere 3 := fun _ => Perspective.diracProb 3 (quadrantThreePoint 2)
  have hQ (k : Fin 1) : (η k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 := by
    intro z hz
    rw [eq_of_mem_support_dirac hz]
    exact quadrantThreePoint_mem 2
  have hplane (k : Fin 1) (α β : ℝ) :
      barycenter 3 (η k) ≠ α • (p : EucSpace 3) + β • (q : EucSpace 3) := by
    have hb : barycenter 3 (η k) = (quadrantThreePoint 2 : EucSpace 3) := by
      simp [η, barycenter, Perspective.diracProb]
    rw [hb]
    exact quadrantThreePoint_not_plane 0 1 2 (by decide) (by decide) (by decide) α β
  obtain ⟨w, hwp, hwq, hw⟩ := exists_common_unit_mean_plane_normal p (q : EucSpace 3) η hplane
  refine ⟨p, q, w, η, one_pos, quadrantThreePoint_mem 0, quadrantThreePoint_mem 1,
    ?_, quadrant_ne_antipode 0 p q (quadrantThreePoint_mem 0) (quadrantThreePoint_mem 1),
    hwp, hwq, hQ, hplane, hw, ?_⟩
  · exact fun he => (by decide : (0 : Fin 3) ≠ 1) (quadrantThreePoint_injective he)
  · intro k
    exact orthogonal_poles_not_support 0 (p : EucSpace 3) (quadrantThreePoint_mem 0) w hwp (η k) (hQ k)

end Transformer.Interpolation
