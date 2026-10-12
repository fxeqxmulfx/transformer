/-
# Nonzero localized gradient directions with zero measure mean

ArXiv:2601.21366v2, §5.2, the construction of `ξ₀` in the
proof of `thm: circle.gelu`. Two distinct actual support points
in an open neighborhood supply two smooth gradient directions.
The first is chosen to vanish at the second point. If its measure
mean is already zero, it suffices; otherwise subtracting its
suitable scalar multiple from the second direction gives mean zero.

The selected direction still has speed one at a support point.
Continuity and the actual definition of measure support therefore
give a strictly positive squared `L²` norm. This proves all the
nonvanishing and admissibility properties needed for the strict
Hessian test without assuming any atomic presentation.
-/

import Transformer.Perceptron.Section5_LocalGradients
import Transformer.Perceptron.Section3_TwoAtomEnergy
import Mathlib.MeasureTheory.Function.LocallyIntegrable

open Real MeasureTheory Set Filter
open scoped Topology ENNReal

namespace Transformer.Perceptron

/-- The source's probability Cauchy--Schwarz bound for actual scalar speeds.
Source: arXiv:2601.21366v2, §5.2, the last bound in the proof of
`thm: circle.gelu`. Integrating the nonnegative variance of `|s|`
proves the squared `L¹` bound without a density assumption. -/
theorem sq_integral_abs_le_integral_sq (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) :
    (∫ x, |s x| ∂(μ : Measure (SSphere 2))) ^ 2 ≤
      ∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2)) := by
  let m : ℝ := ∫ x, |s x| ∂(μ : Measure (SSphere 2))
  have hIabs : Integrable (fun x => |s x|) (μ : Measure (SSphere 2)) :=
    hs.abs.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hIsq : Integrable (fun x => s x ^ 2) (μ : Measure (SSphere 2)) :=
    (hs.pow 2).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hIscale := hIabs.const_mul (2 * m)
  have hIdiff : Integrable (fun x => s x ^ 2 - (2 * m) * |s x|)
      (μ : Measure (SSphere 2)) := hIsq.sub hIscale
  have hexpand : (fun x => (|s x| - m) ^ 2) =
      fun x => s x ^ 2 - (2 * m) * |s x| + m ^ 2 := by
    funext x
    nlinarith [sq_abs (s x)]
  have hvar : (∫ x, (|s x| - m) ^ 2 ∂(μ : Measure (SSphere 2))) =
      (∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2))) - m ^ 2 := by
    rw [hexpand, integral_add hIdiff (integrable_const _), integral_sub hIsq hIscale,
      integral_const_mul, integral_const]
    simp only [probReal_univ, smul_eq_mul, one_mul]
    change (∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2))) - (2 * m) * m + m ^ 2 = _
    ring
  have hnonneg : 0 ≤ ∫ x, (|s x| - m) ^ 2 ∂(μ : Measure (SSphere 2)) :=
    integral_nonneg (fun x => sq_nonneg (|s x| - m))
  rw [hvar] at hnonneg
  change m ^ 2 ≤ ∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2))
  linarith

/-- A continuous speed nonzero at an actual support point has positive
squared `L²` norm. Source: arXiv:2601.21366v2, §5.2, ensuring
`‖ηᵢ‖_{L²(μ)}>0` in the proof of `thm: circle.gelu`. -/
theorem integral_sq_pos_of_support_point (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (y : SSphere 2)
    (hy : y ∈ (μ : Measure (SSphere 2)).support) (hsy : s y ≠ 0) :
    0 < ∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2)) := by
  have hI : Integrable (fun x => s x ^ 2) (μ : Measure (SSphere 2)) :=
    (hs.pow 2).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hU : IsOpen {x : SSphere 2 | 0 < s x ^ 2} := isOpen_lt continuous_const (hs.pow 2)
  have hyU : y ∈ {x : SSphere 2 | 0 < s x ^ 2} := sq_pos_of_ne_zero hsy
  have hmass : 0 < (μ : Measure (SSphere 2)) {x : SSphere 2 | 0 < s x ^ 2} :=
    (Measure.mem_support_iff_forall y).mp hy _ (hU.mem_nhds hyU)
  apply (integral_pos_iff_support_of_nonneg (fun x => sq_nonneg (s x)) hI).mpr
  exact hmass.trans_le (measure_mono (fun x hx => Function.mem_support.mpr (ne_of_gt hx)))

/-- The positivity hypotheses are jointly realized by constant unit
speed at the support point of an actual Dirac probability. -/
example : Continuous (fun _ : SSphere 2 => (1 : ℝ)) ∧
    (basePoint 1 : SSphere 2) ∈
      (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support ∧
    (fun _ : SSphere 2 => (1 : ℝ)) (basePoint 1) ≠ 0 := by
  refine ⟨continuous_const, ?_, one_ne_zero⟩
  apply (Measure.mem_support_iff_forall _).mpr
  intro U hU
  have hx : (basePoint 1 : SSphere 2) ∈ U := mem_of_mem_nhds hU
  change 0 < Measure.dirac (basePoint 1) U
  simp only [Measure.dirac_apply_of_mem hx, zero_lt_one]

/-- Two distinct actual support points in an open neighborhood give a
nonzero smooth gradient direction, supported there, with measure mean zero.
Source: arXiv:2601.21366v2, §5.2, the construction of `ξ₀` in
the proof of `thm: circle.gelu`. Dividing by the first nonzero mean
replaces the source's common scalar factor without changing admissibility. -/
theorem exists_zero_mean_gradient_in_open (μ : Perspective.ProbSphere 2)
    (U : Set (SSphere 2)) (hU : IsOpen U) (y₁ y₂ : SSphere 2) (hne : y₁ ≠ y₂)
    (hy₁ : y₁ ∈ (μ : Measure (SSphere 2)).support)
    (hy₂ : y₂ ∈ (μ : Measure (SSphere 2)).support) (hU₁ : y₁ ∈ U) (hU₂ : y₂ ∈ U) :
    ∃ s : SSphere 2 → ℝ, Continuous s ∧
      IsGradientField (fun x => s x • circleTangent (x : EucSpace 2)) ∧
      Function.support s ⊆ U ∧ (∫ x, s x ∂(μ : Measure (SSphere 2))) = 0 ∧
      0 < ∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2)) := by
  have hU' : IsOpen (U ∩ ({y₂}ᶜ : Set (SSphere 2))) := hU.inter isOpen_compl_singleton
  have hy' : y₁ ∈ U ∩ ({y₂}ᶜ : Set (SSphere 2)) :=
    ⟨hU₁, by simpa only [mem_compl_iff, mem_singleton_iff] using hne⟩
  obtain ⟨s₁, hs₁, hgrad₁, hval₁, hsupp₁⟩ := exists_local_circle_gradient_in_open _ hU' y₁ hy'
  obtain ⟨s₂, hs₂, hgrad₂, hval₂, hsupp₂⟩ := exists_local_circle_gradient_in_open U hU y₂ hU₂
  have hz : s₁ y₂ = 0 := by
    by_contra h
    have hbad := (hsupp₁ (Function.mem_support.mpr h)).2
    simp at hbad
  by_cases hmean : (∫ x, s₁ x ∂(μ : Measure (SSphere 2))) = 0
  · exact ⟨s₁, hs₁, hgrad₁, hsupp₁.trans inter_subset_left, hmean,
      integral_sq_pos_of_support_point μ s₁ hs₁ y₁ hy₁ (by rw [hval₁]; exact one_ne_zero)⟩
  · let c : ℝ := (∫ x, s₂ x ∂(μ : Measure (SSphere 2))) /
      (∫ x, s₁ x ∂(μ : Measure (SSphere 2)))
    let s : SSphere 2 → ℝ := fun x => s₂ x - c * s₁ x
    have hs : Continuous s := hs₂.sub (continuous_const.mul hs₁)
    have hgrad := isGradientField_speed_sub_const_mul s₂ s₁ hgrad₂ hgrad₁ c
    have hsupp : Function.support s ⊆ U := by
      intro x hx
      by_contra hxU
      have hx₁ : s₁ x = 0 := by
        by_contra hn
        exact hxU (hsupp₁ (Function.mem_support.mpr hn)).1
      have hx₂ : s₂ x = 0 := by
        by_contra hn
        exact hxU (hsupp₂ (Function.mem_support.mpr hn))
      apply (Function.mem_support.mp hx)
      simp only [s, hx₁, hx₂, mul_zero, sub_zero]
    have hI₁ : Integrable s₁ (μ : Measure (SSphere 2)) :=
      hs₁.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
    have hI₂ : Integrable s₂ (μ : Measure (SSphere 2)) :=
      hs₂.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
    have hmeanS : (∫ x, s x ∂(μ : Measure (SSphere 2))) = 0 := by
      change (∫ x, s₂ x - c * s₁ x ∂(μ : Measure (SSphere 2))) = 0
      rw [integral_sub hI₂ (hI₁.const_mul c), integral_const_mul]
      dsimp only [c]
      field_simp
      ring
    have hval : s y₂ = 1 := by simp only [s, hval₂, hz, mul_zero, sub_zero]
    exact ⟨s, hs, hgrad, hsupp, hmeanS,
      integral_sq_pos_of_support_point μ s hs y₂ hy₂ (by rw [hval]; exact one_ne_zero)⟩

/-- All open-set, distinctness and actual-support hypotheses are jointly
realized by the two equally weighted antipodal atoms, in the full circle.
Both atoms have actual positive mass, independently of any stationarity claim. -/
example : IsOpen (univ : Set (SSphere 2)) ∧ circlePoint 0 ≠ circlePoint π ∧
    circlePoint 0 ∈ (twoAtomProb (circlePoint 0) (circlePoint π) : Measure (SSphere 2)).support ∧
    circlePoint π ∈ (twoAtomProb (circlePoint 0) (circlePoint π) : Measure (SSphere 2)).support ∧
    circlePoint 0 ∈ (univ : Set (SSphere 2)) ∧ circlePoint π ∈ (univ : Set (SSphere 2)) := by
  refine ⟨isOpen_univ, ?_, ?_, ?_, mem_univ _, mem_univ _⟩
  · intro h
    have hcoord := congrArg (fun z : SSphere 2 => (z : EucSpace 2) 0) h
    norm_num only [circlePoint_coord_zero, Real.cos_zero, Real.cos_pi] at hcoord
  · apply (Measure.mem_support_iff_forall _).mpr
    intro U hU
    have hx : circlePoint 0 ∈ U := mem_of_mem_nhds hU
    simp [Measure.add_apply, Measure.smul_apply, hx]
  · apply (Measure.mem_support_iff_forall _).mpr
    intro U hU
    have hx : circlePoint π ∈ U := mem_of_mem_nhds hU
    simp [Measure.add_apply, Measure.smul_apply, hx]

end Transformer.Perceptron
