module

public import Transformer.Perceptron.Section5_SphericalLocalGradients
public import Transformer.Perceptron.Section6_CircleNullSupport
public import Transformer.Perceptron.Section5_UniformSupport
public import Mathlib.MeasureTheory.Measure.Regular

/-!
# Nonzero smooth gradients with arbitrarily small support mass

ArXiv:2601.21366v2, §5.3, the localization argument for
`thm: any.d` (i). Positive uniform measure of the original support
supplies a support point with zero original singleton mass:
a finite measure has only countably many positive-mass singletons,
while the original uniform law assigns zero mass to countable sets.

Regularity gives arbitrarily small open neighborhoods of such a
point. The actual localized smooth gradient has norm one there,
so continuity and the original support definition make its squared
L² norm strictly positive. Its support lies in that small open set.
This replaces the circle proof's zero-mean kernel subtraction by
the small-mass Cauchy–Schwarz estimate, without changing its goal.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology ENNReal

namespace Transformer.Perceptron

/-- Positive original uniform support measure yields an original support
point with no atom. Source: arXiv:2601.21366v2, §5.3, the localization
step in the analytic alternative of `thm: any.d` (i). -/
theorem exists_nonatom_support_point_of_positive_uniform_measure {d : ℕ} (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∃ x : SSphere d, x ∈ (μ : Measure (SSphere d)).support ∧
      (μ : Measure (SSphere d)) {x} = 0 := by
  classical
  let A : Set (SSphere d) := {x | 0 < (μ : Measure (SSphere d)) {x}}
  have hcount : A.Countable := by
    simpa only [id_eq, Set.ofPred_eq_eq_singleton] using
      (Measure.countable_meas_level_set_pos (μ := (μ : Measure (SSphere d))) measurable_id)
  let : NullSingletonClass ν := nullSingletonClass_of_isUniformOn hd ν hν
  have hA : ν A = 0 := hcount.measure_zero ν
  by_contra h
  have hsub : (μ : Measure (SSphere d)).support ⊆ A := by
    intro x hx
    by_cases hz : (μ : Measure (SSphere d)) {x} = 0
    · exact (h ⟨x, hx, hz⟩).elim
    · exact bot_lt_iff_ne_bot.mpr hz
  have hzero : ν (μ : Measure (SSphere d)).support = 0 :=
    le_antisymm ((measure_mono hsub).trans hA.le) bot_le
  exact hpos.ne' hzero

/-- A zero-mass singleton has open neighborhoods with arbitrarily small
original real mass. Source: arXiv:2601.21366v2, §5.3, the localized
Hessian directions, using regularity of the original sphere probability. -/
theorem exists_small_mass_open_of_nonatom {d : ℕ} (μ : Perspective.ProbSphere d)
    (y : SSphere d) (hy : (μ : Measure (SSphere d)) {y} = 0)
    (ε : ℝ) (hε : 0 < ε) :
    ∃ U : Set (SSphere d), IsOpen U ∧ y ∈ U ∧ (μ : Measure (SSphere d)).real U < ε := by
  obtain ⟨U, hsub, hU, hmass⟩ := ({y} : Set (SSphere d)).exists_isOpen_lt_of_lt
    (μ := (μ : Measure (SSphere d))) (ENNReal.ofReal ε) (by
      rw [hy]
      exact ENNReal.ofReal_pos.mpr hε)
  have hreal := (ENNReal.toReal_lt_toReal (measure_lt_top (μ : Measure (SSphere d)) U).ne
    ENNReal.ofReal_ne_top).mpr hmass
  rw [ENNReal.toReal_ofReal hε.le] at hreal
  exact ⟨U, hU, hsub (mem_singleton y), hreal⟩

/-- A continuous original field nonzero at an actual support point has
strictly positive squared L² norm. Source: arXiv:2601.21366v2,
§5.3, the nonzero admissible directions in `thm: any.d` (i). -/
theorem integral_norm_sq_pos_of_support_point {d : ℕ} (μ : Perspective.ProbSphere d)
    (ξ : SSphere d → EucSpace d) (hcξ : Continuous ξ) (y : SSphere d)
    (hy : y ∈ (μ : Measure (SSphere d)).support) (hξy : ξ y ≠ 0) :
    0 < ∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d)) := by
  have hI : Integrable (fun x => ‖ξ x‖ ^ 2) (μ : Measure (SSphere d)) :=
    (hcξ.norm.pow 2).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hU : IsOpen {x : SSphere d | 0 < ‖ξ x‖ ^ 2} :=
    isOpen_lt continuous_const (hcξ.norm.pow 2)
  have hyU : y ∈ {x : SSphere d | 0 < ‖ξ x‖ ^ 2} :=
    sq_pos_of_ne_zero (norm_ne_zero_iff.mpr hξy)
  have hmass : 0 < (μ : Measure (SSphere d)) {x : SSphere d | 0 < ‖ξ x‖ ^ 2} :=
    (Measure.mem_support_iff_forall y).mp hy _ (hU.mem_nhds hyU)
  apply (integral_pos_iff_support_of_nonneg (fun x => sq_nonneg (‖ξ x‖)) hI).mpr
  exact hmass.trans_le (measure_mono (fun x hx => Function.mem_support.mpr (ne_of_gt hx)))

/-- At an original non-atomic support point, every positive mass scale
admits an actual nonzero smooth gradient supported in a smaller-mass
open set. Source: arXiv:2601.21366v2, §5.3, the localization step
contradicting `eq:strict2order` in `thm: any.d` (i). -/
theorem exists_small_mass_nonzero_gradient {d : ℕ} (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (y : SSphere d)
    (hy : y ∈ (μ : Measure (SSphere d)).support)
    (hynull : (μ : Measure (SSphere d)) {y} = 0) (ε : ℝ) (hε : 0 < ε) :
    ∃ U : Set (SSphere d), IsOpen U ∧ 0 < (μ : Measure (SSphere d)).real U ∧
      (μ : Measure (SSphere d)).real U < ε ∧
      ∃ ξ : SSphere d → EucSpace d, IsGradientField ξ ∧ Continuous ξ ∧
        Function.support ξ ⊆ U ∧ 0 < ∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d)) := by
  obtain ⟨U, hU, hyU, hmass⟩ := exists_small_mass_open_of_nonatom μ y hynull ε hε
  obtain ⟨ξ, hgrad, hcξ, hξy, hsupp⟩ := exists_local_unit_sphere_gradient_in_open hd U hU y hyU
  have hpos : 0 < (μ : Measure (SSphere d)) U :=
    (Measure.mem_support_iff_forall y).mp hy U (hU.mem_nhds hyU)
  refine ⟨U, hU, ENNReal.toReal_pos hpos.ne' (measure_lt_top (μ : Measure (SSphere d)) U).ne,
    hmass, ξ, hgrad, hcξ, hsupp, integral_norm_sq_pos_of_support_point μ ξ hcξ y hy ?_⟩
  apply norm_ne_zero_iff.mp
  rw [hξy]
  exact one_ne_zero

/-- Positive original uniform support measure directly supplies a nonzero
smooth gradient in an arbitrarily small-mass open neighborhood.
Source: arXiv:2601.21366v2, §5.3, the admissible localized directions
used in the analytic alternative of `thm: any.d` (i). -/
theorem exists_small_mass_gradient_of_positive_uniform_support {d : ℕ} (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (hpos : 0 < ν (μ : Measure (SSphere d)).support)
    (ε : ℝ) (hε : 0 < ε) :
    ∃ U : Set (SSphere d), IsOpen U ∧ 0 < (μ : Measure (SSphere d)).real U ∧
      (μ : Measure (SSphere d)).real U < ε ∧
      ∃ ξ : SSphere d → EucSpace d, IsGradientField ξ ∧ Continuous ξ ∧
        Function.support ξ ⊆ U ∧ 0 < ∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d)) := by
  obtain ⟨y, hy, hynull⟩ := exists_nonatom_support_point_of_positive_uniform_measure hd μ ν hν hpos
  exact exists_small_mass_nonzero_gradient hd μ y hy hynull ε hε

/-- The original uniform probability realizes all dimension, uniformity,
positive-support and non-atomic support-point inputs. The positive mass
scale is one, and a continuous nonzero tangent witnesses the norm input.
Source: arXiv:2601.21366v2, §5.3, the original uniform support test. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    0 < (MeanField.uniformLaw 2) (μ : Measure (SSphere 2)).support ∧
    (basePoint 1 : SSphere 2) ∈ (μ : Measure (SSphere 2)).support ∧
    (μ : Measure (SSphere 2)) {(basePoint 1 : SSphere 2)} = 0 ∧
    Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    circleTangent ((basePoint 1 : SSphere 2) : EucSpace 2) ≠ 0 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hprob : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  let : NullSingletonClass (MeanField.uniformLaw 2) :=
    nullSingletonClass_of_isUniformOn le_rfl _ hν
  refine ⟨μ, le_rfl, one_pos, hν, ?_, ?_, by simp [μ],
    continuous_circleTangent.comp continuous_subtype_val, ?_⟩
  · rw [support_eq_univ_of_uniform μ hν]
    simp
  · rw [support_eq_univ_of_uniform μ hν]
    exact mem_univ _
  · apply norm_ne_zero_iff.mp
    rw [norm_circleTangent, norm_basePoint_one]
    exact one_ne_zero

end Transformer.Perceptron
