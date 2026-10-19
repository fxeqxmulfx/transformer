module

public import Transformer.Interpolation.Section4_ProtectedTransfers
public import Transformer.Interpolation.Section4_TransferStability

/-!
# Original contraction and transfer using the same protected attention axis

ArXiv:2411.04551v3, §3.2, selective contraction, and §5,
`thm: targets.atoms`, matching the concentrated laws to their targets.
One common forward attention phase isolates the unselected laws.
Two actual gated neurons then contract the selected law and move its
center to the final target. The common inverse attention restores
every unselected law fixed by both neurons.

The selected law is fixed by the first attention phase because its
original axial mean is zero. No zero-mean condition is imposed after
contraction. The genuine ridge Lipschitz estimate and the nonlinear
reverse-attention support estimate control its final concentration.
Using one gathering time for both neurons keeps that time independent
of the requested accuracy. Equal selected endpoints allow a zero
transfer clock while the other phases retain their actual weak laws.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original gated transfer also includes equal endpoints with
zero elapsed clock. Source auxiliary: arXiv:2411.04551v3, §4 and §5,
the local neuron following concentration. -/
theorem exists_gated_transfer_nonnegative (p q : SSphere d) (hanti : q ≠ antipode d p) :
    ∃ (a : SSphere d) (c T : ℝ), 0 < c ∧ 0 ≤ T ∧
      ridgeFlow (a : EucSpace d) (-((p : EucSpace d) + (q : EucSpace d))) (-c) p T = q := by
  by_cases hpq : p = q
  · subst q
    exact ⟨p, 1, 0, one_pos, le_rfl, ridgeFlow_zero _ _ _ _⟩
  · obtain ⟨a, c, T, hc, hT, hend, _⟩ := exists_exact_gated_transfer p q hpq hanti
    exact ⟨a, c, T, hc, hT.le, hend⟩

/-- Four actual original phases sharing one forward and inverse
attention pair. Source auxiliary: arXiv:2411.04551v3, §3.2 and §5,
isolation, concentration, exact atomic transfer and restoration. -/
noncomputable def contractionTransferPhaseList (i : Fin d) (p w : SSphere d)
    (a m : EucSpace d) (c₀ c t₁ t₂ T : ℝ) : List (AutonomousPhase d) :=
  [meanAxisPhase w t₁, ridgePhase i (t₂ • (p : EucSpace d)) (-(p : EucSpace d)) (-c₀),
    ridgePhase i (T • a) (-m) (-c), meanAxisPhase w (-t₁)]

/-- The combined contraction and transfer has four original phases.
Source auxiliary: arXiv:2411.04551v3, §5, finite switch accounting. -/
theorem contractionTransferPhaseList_length (i : Fin d) (p w : SSphere d)
    (a m : EucSpace d) (c₀ c t₁ t₂ T : ℝ) :
    (contractionTransferPhaseList i p w a m c₀ c t₁ t₂ T).length = 4 := rfl

/-- Every combined phase has its proved original law and zero scores.
Source auxiliary: arXiv:2411.04551v3, §1, §3.2 and §5. -/
theorem contractionTransferPhaseList_valid (i : Fin d) (p w : SSphere d)
    (a m : EucSpace d) (c₀ c t₁ t₂ T : ℝ) :
    ∀ P ∈ contractionTransferPhaseList i p w a m c₀ c t₁ t₂ T,
      IsAutonomousPhase P ∧ P.1.B = 0 := by
  intro P hP
  simp only [contractionTransferPhaseList, List.mem_cons, List.not_mem_nil, or_false] at hP
  rcases hP with rfl | rfl | rfl | rfl
  · exact ⟨isAutonomousPhase_meanAxis w t₁, meanAxisPhase_B_zero w t₁⟩
  · exact ⟨isAutonomousPhase_ridge i _ _ _, ridgePhase_B_zero i _ _ _⟩
  · exact ⟨isAutonomousPhase_ridge i _ _ _, ridgePhase_B_zero i _ _ _⟩
  · exact ⟨isAutonomousPhase_meanAxis w (-t₁), meanAxisPhase_B_zero w (-t₁)⟩

/-- The four original phases have precisely their stated elapsed-law
composition. Source auxiliary: arXiv:2411.04551v3, §3.2 and §5. -/
theorem phaseSequenceLaw_contractionTransfer (i : Fin d) (p w : SSphere d)
    (a m : EucSpace d) (c₀ c t₁ t₂ T : ℝ) (ν : Perspective.ProbSphere d) :
    phaseSequenceLaw (contractionTransferPhaseList i p w a m c₀ c t₁ t₂ T) ν =
      meanAxisLaw w (-1) (ridgeLaw a (-m) (-c)
        (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c₀) (meanAxisLaw w 1 ν t₁) t₂) T) t₁ := by
  change meanAxisLaw w (-t₁)
    (ridgeLaw (T • a) (-m) (-c) (ridgeLaw (t₂ • (p : EucSpace d))
      (-(p : EucSpace d)) (-c₀) (meanAxisLaw w t₁ ν 1) 1) 1) 1 = _
  have hf (ρ : Perspective.ProbSphere d) : meanAxisLaw w t₁ ρ 1 = meanAxisLaw w 1 ρ t₁ := by
    rw [meanAxisLaw_scale, mul_one]
  have hl (ρ : Perspective.ProbSphere d) : meanAxisLaw w (-t₁) ρ 1 = meanAxisLaw w (-1) ρ t₁ := by
    rw [meanAxisLaw_scale w (-t₁) ρ 1, mul_one, meanAxisLaw_scale w (-1) ρ t₁, neg_one_mul]
  rw [hf, ridgeLaw_smul_output, mul_one, ridgeLaw_smul_output, mul_one, hl]

/-- The selected final radius includes the actual transfer Lipschitz
constant and reverse-attention expansion. The contracted law's mean
may change. Source auxiliary: arXiv:2411.04551v3, §3.2 and §5. -/
theorem contractionTransferPhaseList_selected_radius (i : Fin d) (p q w : SSphere d)
    (a m : EucSpace d) (c₀ c t₁ t₂ T δ : ℝ) (ht₁ : 0 ≤ t₁)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (hend : ridgeFlow a (-m) (-c) p T = q)
    (K : ℝ≥0) (hK : LipschitzWith K (fun x : SSphere d => ridgeFlow a (-m) (-c) x T))
    (ν : Perspective.ProbSphere d) (hν : meanAxisMoment (w : EucSpace d) ν = 0)
    (hcenter : (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c₀) ν t₂ :
      Measure (SSphere d)).support ⊆ Metric.closedBall p δ) :
    (phaseSequenceLaw (contractionTransferPhaseList i p w a m c₀ c t₁ t₂ T) ν :
      Measure (SSphere d)).support ⊆ Metric.closedBall q (((K : ℝ) * δ) * exp t₁) := by
  have hmid := support_ridgeLaw_closedBall a (-m) (-c) T K hK p
    (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c₀) ν t₂) δ hcenter
  rw [hend] at hmid
  rw [phaseSequenceLaw_contractionTransfer, meanAxisLaw_of_moment_eq_zero w 1 ν hν]
  simpa only [abs_neg, abs_one, one_mul] using support_meanAxisLaw_radius w q (-1)
    (ridgeLaw a (-m) (-c) (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c₀) ν t₂) T)
    ((K : ℝ) * δ) hwq (fun _ hx => hmid hx) t₁ ht₁

/-- Both middle neurons fixing a gathered law implies exact recovery
of that entire original law. Source auxiliary: arXiv:2411.04551v3,
§3.2, Step 3, and §5, protected matching. -/
theorem contractionTransferPhaseList_restore_fixed (i : Fin d) (p w : SSphere d)
    (a m : EucSpace d) (c₀ c t₁ t₂ T : ℝ) (ν : Perspective.ProbSphere d)
    (hcenter : ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c₀)
      (meanAxisLaw w 1 ν t₁) t₂ = meanAxisLaw w 1 ν t₁)
    (htransfer : ridgeLaw a (-m) (-c) (meanAxisLaw w 1 ν t₁) T = meanAxisLaw w 1 ν t₁) :
    phaseSequenceLaw (contractionTransferPhaseList i p w a m c₀ c t₁ t₂ T) ν = ν := by
  rw [phaseSequenceLaw_contractionTransfer, hcenter, htransfer, meanAxisLaw_inverse]

/-- Transverse coordinate endpoints, an actual exact neuron and a
pole-supported protected law realize all combined phase hypotheses.
The selected input has zero axial mean and a genuine contracted cap. -/
example : ∃ (p q w a : SSphere 3) (m : EucSpace 3) (c T : ℝ)
    (K : ℝ≥0) (ν η : Perspective.ProbSphere 3),
    q ≠ antipode 3 p ∧ 0 < c ∧ 0 ≤ T ∧ (0 : ℝ) ≤ 1 ∧
      inner ℝ (w : EucSpace 3) (q : EucSpace 3) = 0 ∧
      ridgeFlow (a : EucSpace 3) (-m) (-c) p T = q ∧
      LipschitzWith K (fun x : SSphere 3 => ridgeFlow (a : EucSpace 3) (-m) (-c) x T) ∧
      meanAxisMoment (w : EucSpace 3) ν = 0 ∧
      (ridgeLaw (p : EucSpace 3) (-(p : EucSpace 3)) (-(1 / 2)) ν 1 :
        Measure (SSphere 3)).support ⊆ Metric.closedBall p 1 ∧
      ridgeLaw (p : EucSpace 3) (-(p : EucSpace 3)) (-(1 / 2)) (meanAxisLaw w 1 η 1) 1 =
        meanAxisLaw w 1 η 1 ∧
      ridgeLaw (a : EucSpace 3) (-m) (-c) (meanAxisLaw w 1 η 1) T = meanAxisLaw w 1 η 1 := by
  let p := basePoint 2
  let q : SSphere 3 := ⟨PiLp.single 2 (1 : Fin 3) (1 : ℝ), by simp⟩
  let w : SSphere 3 := ⟨PiLp.single 2 (2 : Fin 3) (1 : ℝ), by simp⟩
  have hanti : q ≠ antipode 3 p := by
    intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (1 : Fin 3)) he
    norm_num [q, p, basePoint, antipode] at hh
  obtain ⟨a, c, T, hc, hT, hend⟩ := exists_gated_transfer_nonnegative p q hanti
  let m : EucSpace 3 := (p : EucSpace 3) + (q : EucSpace 3)
  obtain ⟨K, hK⟩ := exists_lipschitzWith_ridgeFlow (a : EucSpace 3) (-m) (-c) T
  let ν := Perspective.diracProb 3 p
  let η := Perspective.diracProb 3 w
  have hwp : inner ℝ (w : EucSpace 3) (p : EucSpace 3) = 0 := by
    simp [w, p, basePoint, EuclideanSpace.inner_single_left]
  have hwq : inner ℝ (w : EucSpace 3) (q : EucSpace 3) = 0 := by
    simp [w, q, EuclideanSpace.inner_single_left]
  have hη : meanAxisLaw w 1 η 1 = η := by
    change projectionLaw w η _ = η
    exact projectionLaw_dirac_axis w _
  refine ⟨p, q, w, a, m, c, T, K, ν, η, hanti, hc, hT, zero_le_one, hwq, hend, hK, ?_, ?_, ?_, ?_⟩
  · simpa [ν, meanAxisMoment, barycenter, Perspective.diracProb] using hwp
  · rw [ridgeLaw_diracProb]
    have he : ridgeFlow (p : EucSpace 3) (-(p : EucSpace 3)) (-(1 / 2)) p 1 = p := by
      have ha : 0 < inner ℝ (p : EucSpace 3) (p : EucSpace 3) - 1 / 2 := by
        rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
        norm_num
      have hb := norm_ridge_cap_center_sub_le p p (1 / 2) ha 1 zero_le_one
      rw [sub_self, norm_zero, zero_mul] at hb
      apply Subtype.ext
      exact sub_eq_zero.mp (norm_eq_zero.mp (le_antisymm hb (norm_nonneg _)))
    rw [he]
    intro x hx
    rw [eq_of_mem_support_dirac hx]
    exact Metric.mem_closedBall.mpr (by simp)
  · rw [hη]
    apply ridgeLaw_inactive_support
    intro x hx
    rw [eq_of_mem_support_dirac hx, inner_neg_right, hwp]
    norm_num
  · rw [hη]
    apply ridgeLaw_inactive_support
    intro x hx
    rw [eq_of_mem_support_dirac hx, inner_neg_right]
    have hm : inner ℝ (w : EucSpace 3) m = 0 := by simp only [m, inner_add_right, hwp, hwq, add_zero]
    rw [hm]
    linarith

end Transformer.Interpolation
