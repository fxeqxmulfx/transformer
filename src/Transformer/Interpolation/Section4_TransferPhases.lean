/-
# Original attention, gated transfer and inverse-attention phases

ArXiv:2411.04551v3, §1, `eq: vf`, and the gather, move and
restore construction for point interpolation in §4.
Three actual autonomous original phases are assembled using a
complete gated neuron between opposite rank-one attention clocks.
Every phase has its proved original weak law and zero scores.

If both atomic endpoints are transverse to the attention axis, the
first and last phases fix them and the actual middle transfer maps
one Dirac exactly to the other. More generally a zero-axial-mean law
is unchanged by the first phase; the proved reverse-attention radius
bound controls its final support. Every middle-fixed unselected law
is restored exactly by the original nonlinear inverse.
-/

import Transformer.Interpolation.Section4_GatedTransfers
import Transformer.Interpolation.Section3_PhaseLawMaps

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Three actual original phases with independent intrinsic clocks.
Source auxiliary: arXiv:2411.04551v3, §4, gather, transfer and restore. -/
noncomputable def transferPhaseList (i : Fin d) (a m : EucSpace d) (w : SSphere d)
    (c t₁ T : ℝ) : List (AutonomousPhase d) :=
  [meanAxisPhase w t₁, ridgePhase i (T • a) (-m) (-c), meanAxisPhase w (-t₁)]

/-- An actual protected transfer uses three original phases.
Source auxiliary: arXiv:2411.04551v3, §4, finite switch accounting. -/
theorem transferPhaseList_length (i : Fin d) (a m : EucSpace d) (w : SSphere d) (c t₁ T : ℝ) :
    (transferPhaseList i a m w c t₁ T).length = 3 := rfl

/-- Every transfer phase has its proved original weak law and zero
score matrix. Source auxiliary: arXiv:2411.04551v3, §1 and §4. -/
theorem transferPhaseList_valid (i : Fin d) (a m : EucSpace d) (w : SSphere d) (c t₁ T : ℝ) :
    ∀ P ∈ transferPhaseList i a m w c t₁ T, IsAutonomousPhase P ∧ P.1.B = 0 := by
  intro P hP
  simp only [transferPhaseList, List.mem_cons, List.not_mem_nil, or_false] at hP
  rcases hP with rfl | rfl | rfl
  · exact ⟨isAutonomousPhase_meanAxis w t₁, meanAxisPhase_B_zero w t₁⟩
  · exact ⟨isAutonomousPhase_ridge i _ _ _, ridgePhase_B_zero i _ _ _⟩
  · exact ⟨isAutonomousPhase_meanAxis w (-t₁), meanAxisPhase_B_zero w (-t₁)⟩

/-- The actual transfer sequence has the exact three original law maps
at their elapsed clocks. Source auxiliary: arXiv:2411.04551v3, §4,
composition of the gather, transfer and inverse maps. -/
theorem phaseSequenceLaw_transfer (i : Fin d) (a m : EucSpace d) (w : SSphere d)
    (c t₁ T : ℝ) (ν : Perspective.ProbSphere d) :
    phaseSequenceLaw (transferPhaseList i a m w c t₁ T) ν =
      meanAxisLaw w (-1) (ridgeLaw a (-m) (-c) (meanAxisLaw w 1 ν t₁) T) t₁ := by
  change meanAxisLaw w (-t₁) (ridgeLaw (T • a) (-m) (-c) (meanAxisLaw w t₁ ν 1) 1) 1 = _
  have hf (ρ : Perspective.ProbSphere d) : meanAxisLaw w t₁ ρ 1 = meanAxisLaw w 1 ρ t₁ := by
    rw [meanAxisLaw_scale, mul_one]
  have hm (ρ : Perspective.ProbSphere d) : ridgeLaw (T • a) (-m) (-c) ρ 1 =
      ridgeLaw a (-m) (-c) ρ T := by rw [ridgeLaw_smul_output, mul_one]
  have hl (ρ : Perspective.ProbSphere d) : meanAxisLaw w (-t₁) ρ 1 = meanAxisLaw w (-1) ρ t₁ := by
    rw [meanAxisLaw_scale w (-t₁) ρ 1, mul_one, meanAxisLaw_scale w (-1) ρ t₁, neg_one_mul]
  rw [hf, hm, hl]

/-- Transverse atomic endpoints undergo exactly the middle original
neuron's transfer. Source auxiliary: arXiv:2411.04551v3, §4,
the selected point in the gather, transfer and restore construction. -/
theorem transferPhaseList_dirac (i : Fin d) (a m : EucSpace d) (w p q : SSphere d)
    (c t₁ T : ℝ) (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (hend : ridgeFlow a (-m) (-c) p T = q) :
    phaseSequenceLaw (transferPhaseList i a m w c t₁ T) (Perspective.diracProb d p) =
      Perspective.diracProb d q := by
  have hp : meanAxisMoment (w : EucSpace d) (Perspective.diracProb d p) = 0 := by
    simpa [meanAxisMoment, barycenter, Perspective.diracProb] using hwp
  have hq : meanAxisMoment (w : EucSpace d) (Perspective.diracProb d q) = 0 := by
    simpa [meanAxisMoment, barycenter, Perspective.diracProb] using hwq
  rw [phaseSequenceLaw_transfer, meanAxisLaw_of_moment_eq_zero w 1 _ hp,
    ridgeLaw_diracProb, hend, meanAxisLaw_of_moment_eq_zero w (-1) _ hq]

/-- Actual reverse attention controls the final radius around a
transverse target after a middle gated transfer. Source auxiliary:
arXiv:2411.04551v3, §1 and §4, stability of the selected transfer. -/
theorem transferPhaseList_selected_radius (i : Fin d) (a m : EucSpace d) (w q : SSphere d)
    (c t₁ T δ : ℝ) (ht₁ : 0 ≤ t₁)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (ν : Perspective.ProbSphere d) (hν : meanAxisMoment (w : EucSpace d) ν = 0)
    (hmid : (ridgeLaw a (-m) (-c) ν T : Measure (SSphere d)).support ⊆ Metric.closedBall q δ) :
    (phaseSequenceLaw (transferPhaseList i a m w c t₁ T) ν : Measure (SSphere d)).support ⊆
      Metric.closedBall q (δ * exp t₁) := by
  rw [phaseSequenceLaw_transfer, meanAxisLaw_of_moment_eq_zero w 1 ν hν]
  simpa only [abs_neg, abs_one, one_mul] using support_meanAxisLaw_radius w q (-1)
    (ridgeLaw a (-m) (-c) ν T) δ hwq (fun _ hx => hmid hx) t₁ ht₁

/-- Every unselected actual law fixed by the middle neuron is restored
exactly by the original attention inverse. Source auxiliary:
arXiv:2411.04551v3, §4, exact restoration in point interpolation. -/
theorem transferPhaseList_restore_fixed (i : Fin d) (a m : EucSpace d) (w : SSphere d)
    (c t₁ T : ℝ) (ν : Perspective.ProbSphere d)
    (hmid : ridgeLaw a (-m) (-c) (meanAxisLaw w 1 ν t₁) T = meanAxisLaw w 1 ν t₁) :
    phaseSequenceLaw (transferPhaseList i a m w c t₁ T) ν = ν := by
  rw [phaseSequenceLaw_transfer, hmid, meanAxisLaw_inverse]

/-- A nonzero exact transfer between orthogonal atoms, a transverse
unit axis, and a genuinely fixed unselected law jointly realize all
selected and restoration hypotheses with positive time and radius. -/
example : ∃ (a m : EucSpace 3) (w p q : SSphere 3) (c T : ℝ)
    (ν₀ ν₁ : Perspective.ProbSphere 3),
    (0 : ℝ) ≤ 1 ∧ 0 < c ∧ 0 < T ∧ inner ℝ (w : EucSpace 3) (p : EucSpace 3) = 0 ∧
    inner ℝ (w : EucSpace 3) (q : EucSpace 3) = 0 ∧ ridgeFlow a (-m) (-c) p T = q ∧
    meanAxisMoment (w : EucSpace 3) ν₀ = 0 ∧
    (ridgeLaw a (-m) (-c) ν₀ T : Measure (SSphere 3)).support ⊆ Metric.closedBall q 1 ∧
    ridgeLaw a (-m) (-c) (meanAxisLaw w 1 ν₁ 1) T = meanAxisLaw w 1 ν₁ 1 := by
  let p : SSphere 3 := ⟨PiLp.single 2 (0 : Fin 3) (1 : ℝ), by simp⟩
  let q : SSphere 3 := ⟨PiLp.single 2 (1 : Fin 3) (1 : ℝ), by simp⟩
  let w : SSphere 3 := ⟨PiLp.single 2 (2 : Fin 3) (1 : ℝ), by simp⟩
  have hpq : p ≠ q := by
    intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (0 : Fin 3)) he
    norm_num [p, q] at hh
  have hanti : q ≠ antipode 3 p := by
    intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (1 : Fin 3)) he
    norm_num [p, q, antipode] at hh
  obtain ⟨a, c, T, hc, hT, hend, _⟩ := exists_exact_gated_transfer p q hpq hanti
  let m : EucSpace 3 := (p : EucSpace 3) + (q : EucSpace 3)
  change ridgeFlow (a : EucSpace 3) (-m) (-c) p T = q at hend
  let ν₀ := Perspective.diracProb 3 p
  let ν₁ := Perspective.diracProb 3 w
  have hwp : inner ℝ (w : EucSpace 3) (p : EucSpace 3) = 0 := by
    simp [w, p, EuclideanSpace.inner_single_left]
  have hwq : inner ℝ (w : EucSpace 3) (q : EucSpace 3) = 0 := by
    simp [w, q, EuclideanSpace.inner_single_left]
  have hpre : meanAxisLaw w 1 ν₁ 1 = ν₁ := by
    rw [meanAxisLaw, show ν₁ = Perspective.diracProb 3 w from rfl, projectionLaw_dirac_axis]
  refine ⟨a, m, w, p, q, c, T, ν₀, ν₁, zero_le_one, hc, hT, hwp, hwq, hend, ?_, ?_, ?_⟩
  · simpa [ν₀, meanAxisMoment, barycenter, Perspective.diracProb] using hwp
  · change (ridgeLaw (a : EucSpace 3) (-m) (-c) (Perspective.diracProb 3 p) T :
      Measure (SSphere 3)).support ⊆ _
    rw [ridgeLaw_diracProb, hend]
    intro z hz
    rw [eq_of_mem_support_dirac hz]
    simpa only [Metric.mem_closedBall, dist_self] using (zero_le_one : (0 : ℝ) ≤ 1)
  · rw [hpre, show ν₁ = Perspective.diracProb 3 w from rfl, ridgeLaw_diracProb]
    have hf : ridgeFlow (a : EucSpace 3) (-m) (-c) w T = w := by
      apply ridgeFlow_inactive
      rw [inner_neg_right]
      have hm : inner ℝ (w : EucSpace 3) m = 0 := by simp only [m, inner_add_right, hwp, hwq, add_zero]
      rw [hm]
      linarith
    rw [hf]

end Transformer.Interpolation
