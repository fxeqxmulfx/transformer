/-
# Chernoff optimization with the original quadratic rate

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
A quadratic MGF estimate on a fixed coefficient interval gives the
minimum of the Gaussian and exponential tail rates. Both signs use
the same original random variable and probability measure. The union
bound contributes exactly the factor two in the manuscript.
Zero denominators are treated using the probability bound, consistently
with real division by zero in the formal statement.
-/

import Transformer.Homogenized.Section6_CenteredQuadraticMGF

open Real MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Transformer.Homogenized

/-- The actual coefficient minimizing the quadratic Chernoff bound
within the permitted interval gives the original two-rate minimum.
Source: arXiv:2604.01978v1, §6.3, the optimization in `lem:Hanson_wright`. -/
theorem quadratic_chernoff_exponent_le (K v w r s a : ℝ)
    (hK : 0 < K) (hv : 0 < v) (hw : 0 < w) (hr : 0 < r) (hs : 0 ≤ s)
    (ha : 0 ≤ a) (hKa : a ≤ 1 / (4 * K)) (hra : a ≤ r / 2) :
    let u := min (s / (2 * K * v)) (r / w)
    K * u ^ 2 * v - u * s ≤ -(a * min (s ^ 2 / v) (s / w)) := by
  let u := min (s / (2 * K * v)) (r / w)
  have hu : 0 ≤ u := le_min (by positivity) (by positivity)
  have hb : u * (2 * K * v) ≤ s :=
    (le_div_iff₀ (show 0 < 2 * K * v by positivity)).mp (min_le_left _ _)
  have hq := mul_le_mul_of_nonneg_right hb hu
  have hdecay : K * u ^ 2 * v ≤ u * s / 2 := by nlinarith only [hq]
  have hrate : a * min (s ^ 2 / v) (s / w) ≤ u * s / 2 := by
    calc
      _ ≤ min ((1 / (4 * K)) * (s ^ 2 / v)) ((r / 2) * (s / w)) := by
        apply le_min
        · exact (mul_le_mul_of_nonneg_left (min_le_left _ _) ha).trans
            (mul_le_mul_of_nonneg_right hKa (by positivity))
        · exact (mul_le_mul_of_nonneg_left (min_le_right _ _) ha).trans
            (mul_le_mul_of_nonneg_right hra (by positivity))
      _ = u * s / 2 := by
        have h₁ : (1 / (4 * K)) * (s ^ 2 / v) = (s / (2 * K * v)) * (s / 2) := by
          field_simp
          ring
        have h₂ : (r / 2) * (s / w) = (r / w) * (s / 2) := by ring
        rw [h₁, h₂, ← min_mul_of_nonneg _ _ (show 0 ≤ s / 2 by positivity)]
        dsimp only [u]
        ring
  change K * u ^ 2 * v - u * s ≤ _
  linarith

/-- A true local quadratic MGF bound gives an absolute tail bound
with exactly two tails and the original variance and scale denominators.
This is a general Chernoff implication; the independent-coordinate MGF
used by Hanson–Wright is proved separately, not assumed in that theorem.
Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem abs_tail_of_local_quadratic_mgf {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (K v w r : ℝ)
    (hK : 0 < K) (hv : 0 ≤ v) (hw : 0 ≤ w) (hr : 0 < r)
    (hmgf : ∀ u : ℝ, |u| * w ≤ r →
      Integrable (fun ω => Real.exp (u * X ω)) P ∧
      (∫ ω, Real.exp (u * X ω) ∂P) ≤ Real.exp (K * u ^ 2 * v))
    (s : ℝ) (hs : 0 ≤ s) :
    P {ω | s ≤ |X ω|} ≤ ENNReal.ofReal (2 * Real.exp (-(min (1 / (4 * K)) (r / 2) *
      min (s ^ 2 / v) (s / w)))) := by
  have hp : P {ω | s ≤ |X ω|} ≤ 1 := by
    simpa only [measure_univ] using measure_mono (μ := P) (Set.subset_univ {ω | s ≤ |X ω|})
  have htwo : (1 : ℝ≥0∞) ≤ ENNReal.ofReal 2 := by norm_num
  by_cases hv0 : v = 0
  · simpa only [hv0, div_zero, min_eq_left (div_nonneg hs hw), mul_zero, neg_zero,
      Real.exp_zero, mul_one] using hp.trans htwo
  by_cases hw0 : w = 0
  · simpa only [hw0, div_zero, min_eq_right (div_nonneg (sq_nonneg s) hv), mul_zero,
      neg_zero, Real.exp_zero, mul_one] using hp.trans htwo
  have hvp : 0 < v := lt_of_le_of_ne hv (Ne.symm hv0)
  have hwp : 0 < w := lt_of_le_of_ne hw (Ne.symm hw0)
  let u := min (s / (2 * K * v)) (r / w)
  let a := min (1 / (4 * K)) (r / 2)
  let L := Real.exp (-(a * min (s ^ 2 / v) (s / w)))
  have hu : 0 ≤ u := le_min (by positivity) (by positivity)
  have hus : |u| * w ≤ r := by
    rw [abs_of_nonneg hu]
    exact (le_div_iff₀ hwp).mp (min_le_right _ _)
  obtain ⟨hi, hm⟩ := hmgf u hus
  have hnus : |-u| * w ≤ r := by simpa only [abs_neg] using hus
  obtain ⟨hni, hnm⟩ := hmgf (-u) hnus
  have hopt : K * u ^ 2 * v - u * s ≤ -(a * min (s ^ 2 / v) (s / w)) :=
    quadratic_chernoff_exponent_le K v w r s a hK hvp hwp hr hs
      (le_min (by positivity) (by positivity)) (min_le_left _ _) (min_le_right _ _)
  have hupper : P.real {ω | s ≤ X ω} ≤ L := by
    calc
      _ ≤ Real.exp (-u * s) * (∫ ω, Real.exp (u * X ω) ∂P) :=
        measure_ge_le_exp_mul_mgf s hu hi
      _ ≤ Real.exp (-u * s) * Real.exp (K * u ^ 2 * v) :=
        mul_le_mul_of_nonneg_left hm (Real.exp_nonneg _)
      _ = Real.exp (K * u ^ 2 * v - u * s) := by rw [← Real.exp_add]; congr 1; ring
      _ ≤ L := Real.exp_le_exp.mpr hopt
  have hnegint : Integrable (fun ω => Real.exp (u * (-X ω))) P := by
    simpa only [neg_mul, mul_neg] using hni
  have hnegmgf : (∫ ω, Real.exp (u * (-X ω)) ∂P) ≤ Real.exp (K * u ^ 2 * v) := by
    simpa only [neg_mul, mul_neg, neg_sq] using hnm
  have hlower : P.real {ω | s ≤ -X ω} ≤ L := by
    calc
      _ ≤ Real.exp (-u * s) * (∫ ω, Real.exp (u * (-X ω)) ∂P) :=
        measure_ge_le_exp_mul_mgf s hu hnegint
      _ ≤ Real.exp (-u * s) * Real.exp (K * u ^ 2 * v) :=
        mul_le_mul_of_nonneg_left hnegmgf (Real.exp_nonneg _)
      _ = Real.exp (K * u ^ 2 * v - u * s) := by rw [← Real.exp_add]; congr 1; ring
      _ ≤ L := Real.exp_le_exp.mpr hopt
  have hupp : P {ω | s ≤ X ω} ≤ ENNReal.ofReal L := by
    rw [← ofReal_measureReal (μ := P) (s := {ω | s ≤ X ω})]
    exact ENNReal.ofReal_le_ofReal hupper
  have hlow : P {ω | s ≤ -X ω} ≤ ENNReal.ofReal L := by
    rw [← ofReal_measureReal (μ := P) (s := {ω | s ≤ -X ω})]
    exact ENNReal.ofReal_le_ofReal hlower
  have hevent : {ω | s ≤ |X ω|} = {ω | s ≤ X ω} ∪ {ω | s ≤ -X ω} := by
    ext ω
    change s ≤ |X ω| ↔ s ≤ X ω ∨ s ≤ -X ω
    exact le_abs
  rw [hevent]
  calc
    _ ≤ P {ω | s ≤ X ω} + P {ω | s ≤ -X ω} := measure_union_le _ _
    _ ≤ ENNReal.ofReal L + ENNReal.ofReal L := add_le_add hupp hlow
    _ = ENNReal.ofReal (2 * L) := by rw [← ENNReal.ofReal_add (Real.exp_nonneg _) (Real.exp_nonneg _)]; congr 1; ring

/-- The coefficient furnished by the actual Chernoff optimization is
strictly positive and depends only on the two universal MGF constants.
Source: arXiv:2604.01978v1, §6.3, the universal `c` in `lem:Hanson_wright`. -/
theorem quadratic_tail_constant_pos (K r : ℝ) (hK : 0 < K) (hr : 0 < r) :
    0 < min (1 / (4 * K)) (r / 2) := by
  exact lt_min (by positivity) (by positivity)

/-- Positive denominators and an interior universal coefficient witness
all hypotheses of the purely numerical Chernoff optimization. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 4 ∧
    (1 / 4 : ℝ) ≤ 1 / (4 * 1) ∧ (1 / 4 : ℝ) ≤ 1 / 2 := by norm_num

/-- The original zero variable on a probability space genuinely satisfies
the quadratic MGF bound for every coefficient, hence on every local interval. -/
example : ∃ P : Measure Unit, IsProbabilityMeasure P ∧
    ∀ u : ℝ, Integrable (fun _ : Unit => Real.exp (u * (0 : ℝ))) P ∧
      (∫ _ : Unit, Real.exp (u * (0 : ℝ)) ∂P) ≤ Real.exp (1 * u ^ 2 * 1) := by
  refine ⟨Measure.dirac (), inferInstance, fun u => ?_⟩
  constructor
  · simpa only [mul_zero, Real.exp_zero] using integrable_const (1 : ℝ)
  · simpa only [mul_zero, Real.exp_zero, integral_const, measureReal_def,
      measure_univ, ENNReal.toReal_one, smul_eq_mul, one_mul, mul_one]
      using Real.one_le_exp (sq_nonneg u)

end Transformer.Homogenized
