module

public import Transformer.Interpolation.Section1_PhaseSequenceControls
public import Transformer.Interpolation.Section3_RidgeClassSequence
public import Transformer.Interpolation.Section3_MeanAxisStability

/-!
# Actual probability maps of preprocessing and selective phases

ArXiv:2411.04551v3, §3.2, `prop: separation` and Steps 1–3
of `lem: induction.barycenter`. The finite ridge preprocessing list is
embedded in the actual autonomous phase data without changing its law
or length. Every concrete phase has the original weak law and zero scores.

Three further actual phases realize the selective nonlinear map:
forward rank-one attention, a gated ridge and reversed attention.
Exact clock identities give its three-law composition. A selected
zero-mean law contracts through the middle field and obeys the proved
reverse-attention radius bound. If the middle law is fixed, the complete
nonlinear attention inverse restores the original law exactly.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Embed actual ridge data as original autonomous network phases.
Source: arXiv:2411.04551v3, §3.2, perceptron preprocessing. -/
noncomputable def ridgePhaseList (i : Fin d) (L : List (EucSpace d × EucSpace d × ℝ)) :
    List (AutonomousPhase d) := L.map (fun q => ridgePhase i q.1 q.2.1 q.2.2)

/-- Embedding the original ridge data preserves its finite phase count.
Source: arXiv:2411.04551v3, §3.2, switch counting. -/
theorem ridgePhaseList_length (i : Fin d) (L : List (EucSpace d × EucSpace d × ℝ)) :
    (ridgePhaseList i L).length = L.length := List.length_map _

/-- Every embedded ridge has its actual original weak law and zero scores.
Source: arXiv:2411.04551v3, §1 and §3.2. -/
theorem ridgePhaseList_valid (i : Fin d) (L : List (EucSpace d × EucSpace d × ℝ)) :
    ∀ P ∈ ridgePhaseList i L, IsAutonomousPhase P ∧ P.1.B = 0 := by
  intro P hP
  obtain ⟨q, -, rfl⟩ := List.mem_map.mp hP
  exact ⟨isAutonomousPhase_ridge i q.1 q.2.1 q.2.2, ridgePhase_B_zero i q.1 q.2.1 q.2.2⟩

/-- The embedded preprocessing list has the same actual probability map.
Source: arXiv:2411.04551v3, §3.2, composing perceptron phases. -/
theorem phaseSequenceLaw_ridges (i : Fin d) (L : List (EucSpace d × EucSpace d × ℝ))
    (ν : Perspective.ProbSphere d) : phaseSequenceLaw (ridgePhaseList i L) ν = ridgeSequenceLaw L 1 ν := by
  induction L generalizing ν with
  | nil => rfl
  | cons q L ih => exact ih (ridgeLaw q.1 q.2.1 q.2.2 ν 1)

/-- Three actual network phases with the independently selected
intrinsic clocks. Source: arXiv:2411.04551v3, §3.2, Steps 1–3. -/
noncomputable def selectivePhaseList (i : Fin d) (p w : SSphere d) (c t₁ t₂ : ℝ) :
    List (AutonomousPhase d) :=
  [meanAxisPhase w t₁, ridgePhase i (t₂ • (p : EucSpace d)) (-(p : EucSpace d)) (-c),
    meanAxisPhase w (-t₁)]

/-- Selective transport uses three original constant phases.
Source: arXiv:2411.04551v3, §3.2, switch counting. -/
theorem selectivePhaseList_length (i : Fin d) (p w : SSphere d) (c t₁ t₂ : ℝ) :
    (selectivePhaseList i p w c t₁ t₂).length = 3 := rfl

/-- All three concrete fields have their proved original weak laws
and zero scores. Source: arXiv:2411.04551v3, §1 and §3.2. -/
theorem selectivePhaseList_valid (i : Fin d) (p w : SSphere d) (c t₁ t₂ : ℝ) :
    ∀ P ∈ selectivePhaseList i p w c t₁ t₂, IsAutonomousPhase P ∧ P.1.B = 0 := by
  intro P hP
  simp only [selectivePhaseList, List.mem_cons, List.not_mem_nil, or_false] at hP
  rcases hP with rfl | rfl | rfl
  · exact ⟨isAutonomousPhase_meanAxis w t₁, meanAxisPhase_B_zero w t₁⟩
  · exact ⟨isAutonomousPhase_ridge i _ _ _, ridgePhase_B_zero i _ _ _⟩
  · exact ⟨isAutonomousPhase_meanAxis w (-t₁), meanAxisPhase_B_zero w (-t₁)⟩

/-- The actual finite-phase law is the three original solution maps
at their selected clocks. Source: arXiv:2411.04551v3, §3.2. -/
theorem phaseSequenceLaw_selective (i : Fin d) (p w : SSphere d) (c t₁ t₂ : ℝ)
    (ν : Perspective.ProbSphere d) : phaseSequenceLaw (selectivePhaseList i p w c t₁ t₂) ν =
      meanAxisLaw w (-1) (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c)
        (meanAxisLaw w 1 ν t₁) t₂) t₁ := by
  change meanAxisLaw w (-t₁) (ridgeLaw (t₂ • (p : EucSpace d)) (-(p : EucSpace d)) (-c)
    (meanAxisLaw w t₁ ν 1) 1) 1 = _
  have hfirst (ρ : Perspective.ProbSphere d) : meanAxisLaw w t₁ ρ 1 = meanAxisLaw w 1 ρ t₁ := by
    rw [meanAxisLaw_scale, mul_one]
  have hmid (ρ : Perspective.ProbSphere d) : ridgeLaw (t₂ • (p : EucSpace d))
      (-(p : EucSpace d)) (-c) ρ 1 = ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ρ t₂ := by
    rw [ridgeLaw_smul_output, mul_one]
  have hlast (ρ : Perspective.ProbSphere d) : meanAxisLaw w (-t₁) ρ 1 = meanAxisLaw w (-1) ρ t₁ := by
    rw [meanAxisLaw_scale w (-t₁) ρ 1, mul_one, meanAxisLaw_scale w (-1) ρ t₁, neg_one_mul]
  rw [hfirst, hmid, hlast]

/-- The actual reverse attention enlarges the middle radius by at most
the proved factor. Source: arXiv:2411.04551v3, §3.2, Step 3. -/
theorem selectivePhaseList_selected_radius (i : Fin d) (p w : SSphere d)
    (c t₁ t₂ δ : ℝ) (ht₁ : 0 ≤ t₁)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (ν : Perspective.ProbSphere d) (hν : meanAxisMoment (w : EucSpace d) ν = 0)
    (hmid : (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ν t₂ : Measure (SSphere d)).support ⊆
      Metric.closedBall p δ) :
    (phaseSequenceLaw (selectivePhaseList i p w c t₁ t₂) ν : Measure (SSphere d)).support ⊆
      Metric.closedBall p (δ * exp t₁) := by
  rw [phaseSequenceLaw_selective, meanAxisLaw_of_moment_eq_zero w 1 ν hν]
  simpa only [abs_neg, abs_one, one_mul] using support_meanAxisLaw_radius w p (-1)
    (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ν t₂) δ hwp (fun x hx => hmid hx) t₁ ht₁

/-- A fixed middle law is restored by the actual nonlinear inverse.
Source: arXiv:2411.04551v3, §3.2, flow reversal. -/
theorem selectivePhaseList_restore_fixed (i : Fin d) (p w : SSphere d) (c t₁ t₂ : ℝ)
    (ν : Perspective.ProbSphere d)
    (hmid : ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) (meanAxisLaw w 1 ν t₁) t₂ =
      meanAxisLaw w 1 ν t₁) : phaseSequenceLaw (selectivePhaseList i p w c t₁ t₂) ν = ν := by
  rw [phaseSequenceLaw_selective, hmid, meanAxisLaw_inverse]

/-- An active two-atom hemisphere law has zero transverse mean and a
common positive gate margin. A transverse pole is genuinely inactive.
Their actual middle laws realize the contraction and exact-fixing inputs
for the same three-phase map, with positive observation time and radius. -/
example : ∃ (p w : SSphere 2) (ν₀ ν₁ : Perspective.ProbSphere 2) (δ : ℝ),
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 16 ∧ 0 < δ ∧
    inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 ∧ meanAxisMoment (w : EucSpace 2) ν₀ = 0 ∧
    (ridgeLaw (p : EucSpace 2) (-(p : EucSpace 2)) (-(1 / 4 : ℝ)) ν₀ 16 :
      Measure (SSphere 2)).support ⊆ Metric.closedBall p δ ∧
    ridgeLaw (p : EucSpace 2) (-(p : EucSpace 2)) (-(1 / 4 : ℝ))
      (meanAxisLaw w 1 ν₁ 1) 16 = meanAxisLaw w 1 ν₁ 1 := by
  let p : SSphere 2 := basePoint 1
  let w : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let a : SSphere 2 := circlePt (3 / 5) (4 / 5) (by norm_num)
  let b : SSphere 2 := circlePt (3 / 5) (-4 / 5) (by norm_num)
  let ν₀ := halfDirac 2 a b
  let ν₁ := Perspective.diracProb 2 w
  let δ : ℝ := 2 * exp (-(((1 / 2 : ℝ) - 1 / 4) / 2 * 16))
  have hwp : inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 := by
    simp [w, p, basePoint, EuclideanSpace.inner_single_left]
  have hm : meanAxisMoment (w : EucSpace 2) ν₀ = 0 := by
    rw [meanAxisMoment, barycenter_halfDirac]
    norm_num [w, a, b, circlePt, inner_add_right, real_inner_smul_right,
      EuclideanSpace.inner_single_left]
  have hheight : ∀ x ∈ (ν₀ : Measure (SSphere 2)).support,
      (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2) := by
    intro x hx
    rcases mem_of_mem_support_halfDirac hx with rfl | rfl <;>
      norm_num [a, b, p, circlePt, basePoint, EuclideanSpace.inner_single_right]
  have hmid := support_ridgeLaw_center_radius p ν₀ (1 / 2) (1 / 4)
    (by norm_num) (by norm_num) hheight 16 (by norm_num)
  have hstate : meanAxisLaw w 1 ν₁ 1 = ν₁ := by
    rw [meanAxisLaw, show ν₁ = Perspective.diracProb 2 w from rfl, projectionLaw_dirac_axis]
  have hfixed : ridgeLaw (p : EucSpace 2) (-(p : EucSpace 2)) (-(1 / 4 : ℝ))
      (meanAxisLaw w 1 ν₁ 1) 16 = meanAxisLaw w 1 ν₁ 1 := by
    rw [hstate]
    apply ridgeLaw_inactive_support
    intro x hx
    rw [eq_of_mem_support_dirac hx, inner_neg_right, hwp]
    norm_num
  exact ⟨p, w, ν₀, ν₁, δ, zero_le_one, by norm_num,
    mul_pos (by norm_num) (exp_pos _), hwp, hm, hmid, hfixed⟩

end Transformer.Interpolation
