module

public import Transformer.Perceptron.Section5_SignedTransform

/-!
# Identifying the original measure from a continuous signed candidate

ArXiv:2601.21366v2, §5.1, proof of `lem: quadpol` (i).
The source constructs a signed candidate density, then identifies its
signed measure with the original probability. The candidate is not
assumed nonnegative before that identification.

The original nonnegative finite measure plus the candidate's actual
negative part has the same original exponential transform as its actual
positive part. Finite-measure injectivity identifies these two measures.
Testing their equality against the continuous negative part gives a
sum of two nonnegative integrals equal to zero: the second is its actual
square integral. Thus the negative part vanishes almost everywhere,
and the original measure is the original `withDensity` measure.

For the actual uniform law, its proved full support makes the original
continuous candidate nonnegative at every sphere point. This supplies
the missing positivity step in a polynomial inverse argument without
putting the inverse, measure equality or nonnegativity into a definition.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Perceptron

/-- A continuous signed candidate with the original measure's transform
is its actual density, and is nonnegative almost everywhere.
Source: arXiv:2601.21366v2, §5.1, proof of `lem: quadpol` (i).
This finite-measure extension proves positivity from equality of actual
measures; it does not assume it during the signed inverse construction. -/
theorem eq_withDensity_of_integral_exp_inner_eq_integral_mul {d : ℕ}
    (β : ℝ) (hβ : 0 < β) (μ ν : Measure (SSphere d))
    [IsFiniteMeasure μ] [IsFiniteMeasure ν] (g : SSphere d → ℝ) (hg : Continuous g)
    (h : ∀ x : SSphere d,
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ) =
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν) :
    (∀ᵐ y ∂ν, 0 ≤ g y) ∧ μ = ν.withDensity (fun y => ENNReal.ofReal (g y)) := by
  let P := ν.withDensity fun y => ENNReal.ofReal (g y)
  let N := ν.withDensity fun y => ENNReal.ofReal (-g y)
  let q := fun y : SSphere d => max (-g y) 0
  have hq : Continuous q := hg.neg.max continuous_const
  let _ : IsFiniteMeasure P := isFiniteMeasure_withDensity_ofReal
    (Perspective.integrable_of_continuous_compact hg ν).hasFiniteIntegral
  let _ : IsFiniteMeasure N := isFiniteMeasure_withDensity_ofReal
    (Perspective.integrable_of_continuous_compact hg.neg ν).hasFiniteIntegral
  have hparts : μ + N = P := by
    apply eq_of_integral_exp_inner_eq_finite β hβ
    intro x
    have hK : Continuous (fun y : SSphere d =>
        exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) :=
      Real.continuous_exp.comp (continuous_const.mul
        (continuous_const.inner continuous_subtype_val))
    rw [integral_add_measure (Perspective.integrable_of_continuous_compact hK μ)
      (Perspective.integrable_of_continuous_compact hK N), h x]
    have hp := integral_exp_inner_mul_eq_positive_parts β ν g hg x
    change (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν) =
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂P) -
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂N at hp
    linarith
  have htest := congrArg (fun ρ : Measure (SSphere d) => ∫ y, q y ∂ρ) hparts
  rw [integral_add_measure (Perspective.integrable_of_continuous_compact hq μ)
    (Perspective.integrable_of_continuous_compact hq N)] at htest
  change (∫ y, q y ∂μ) +
    (∫ y, q y ∂ν.withDensity (fun y => ENNReal.ofReal (-g y))) =
    ∫ y, q y ∂ν.withDensity (fun y => ENNReal.ofReal (g y)) at htest
  rw [integral_withDensity_ofReal_eq_posPart ν (fun y => -g y) hg.neg,
    integral_withDensity_ofReal_eq_posPart ν g hg] at htest
  have hprod : ∀ y : SSphere d, max (g y) 0 * q y = 0 := by
    intro y
    change max (g y) 0 * max (-g y) 0 = 0
    by_cases hy : 0 ≤ g y
    · rw [max_eq_right (neg_nonpos.mpr hy), mul_zero]
    · rw [max_eq_right (le_of_not_ge hy), zero_mul]
  simp_rw [hprod] at htest
  rw [integral_zero] at htest
  have hsq : (∫ y, q y * q y ∂ν) = 0 := by
    have hμq : 0 ≤ ∫ y, q y ∂μ := integral_nonneg fun y => le_max_right _ _
    have hνq : 0 ≤ ∫ y, q y * q y ∂ν := integral_nonneg fun y =>
      mul_self_nonneg (q y)
    change (∫ y, q y ∂μ) + (∫ y, q y * q y ∂ν) = 0 at htest
    linarith
  have hae := (integral_eq_zero_iff_of_nonneg
    (f := fun y => q y * q y) (fun y => mul_self_nonneg (q y))
    (Perspective.integrable_of_continuous_compact (hq.mul hq) ν)).mp hsq
  have hnn : ∀ᵐ y ∂ν, 0 ≤ g y := by
    filter_upwards [hae] with y hy
    change q y * q y = 0 at hy
    have hqy : q y = 0 := (mul_eq_zero.mp hy).elim id id
    have hle : -g y ≤ q y := le_max_left _ _
    rw [hqy] at hle
    linarith
  have hN : N = 0 := by
    change ν.withDensity (fun y => ENNReal.ofReal (-g y)) = 0
    rw [← withDensity_zero (μ := ν)]
    apply withDensity_congr_ae
    filter_upwards [hnn] with y hy
    exact ENNReal.ofReal_eq_zero.mpr (neg_nonpos.mpr hy)
  exact ⟨hnn, by simpa only [hN, add_zero] using hparts⟩

/-- The original continuous signed inverse candidate is nonnegative at
every sphere point against the actual uniform law, and gives the original
measure. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The pointwise positivity follows from that law's proved full support. -/
theorem nonneg_and_eq_withDensity_of_uniform_transform_eq {d : ℕ}
    (β : ℝ) (hβ : 0 < β) (μ ν : Measure (SSphere d)) [IsFiniteMeasure μ]
    (hν : Metastability.IsUniformOn d ν) (g : SSphere d → ℝ) (hg : Continuous g)
    (h : ∀ x : SSphere d,
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ) =
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν) :
    (∀ y, 0 ≤ g y) ∧ μ = ν.withDensity (fun y => ENNReal.ofReal (g y)) := by
  let _ : IsProbabilityMeasure ν := hν.1
  let _ : ν.IsOpenPosMeasure := isOpenPosMeasure_of_uniform ν hν
  obtain ⟨hnn, heq⟩ := eq_withDensity_of_integral_exp_inner_eq_integral_mul β hβ μ ν g hg h
  have hclip : g =ᵐ[ν] fun y => max (g y) 0 := by
    filter_upwards [hnn] with y hy
    exact (max_eq_left hy).symm
  have hpoint := Measure.eq_of_ae_eq hclip hg (hg.max continuous_const)
  refine ⟨fun y => ?_, heq⟩
  have hy := congrFun hpoint y
  rw [hy]
  exact le_max_right _ _

/-- An actual continuous nonnegative candidate uses its original value
in the original exponential integral, with no clipping discrepancy.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), the converse.
The statement keeps the actual `withDensity` law and original kernel. -/
theorem integral_exp_inner_withDensity_of_nonneg {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (g : SSphere d → ℝ) (hg : Continuous g)
    (hnn : ∀ y, 0 ≤ g y) (x : SSphere d) :
    (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
      ∂ν.withDensity (fun y => ENNReal.ofReal (g y))) =
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν := by
  rw [integral_withDensity_ofReal_eq_posPart ν g hg]
  apply integral_congr_ae
  exact ae_of_all ν fun y => by
    change max (g y) 0 * exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) =
      exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y
    rw [max_eq_left (hnn y), mul_comm]

/-- Every identification input holds jointly for the original uniform
circle probability and the actual density one, at positive temperature.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The witness has the actual nonzero original probability mass. -/
example : ∃ ν : Measure (SSphere 2), (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 ν ∧ IsFiniteMeasure ν ∧
    Continuous (fun _ : SSphere 2 => (1 : ℝ)) ∧
    (∀ _y : SSphere 2, (0 : ℝ) ≤ 1) ∧
    ∀ x : SSphere 2,
      (∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) ∂ν) =
      ∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) * (1 : ℝ) ∂ν := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let _ : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  exact ⟨MeanField.uniformLaw 2, one_pos, hν, inferInstance,
    continuous_const, fun _ => zero_le_one, fun _ => by simp⟩

end Transformer.Perceptron
