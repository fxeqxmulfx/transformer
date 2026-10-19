module

public import Transformer.Homogenized.Section6_SubgaussianQuadraticMoments
public import Mathlib.Analysis.Real.Sqrt

/-!
# Actual subgaussian scalar and squared tails

ArXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. The actual
two-sided scalar tail follows from the original MGF and its genuine
negative-variable MGF. The original square event is the same actual
event after taking the nonnegative square root of its threshold.

For original linear forms, the denominator retains the exact original
coefficient-square variance proxy already proved from independence.
These actual squared-tail estimates will control exponential moments
of the original coordinate squares before their centered estimates.
The genuine zero-proxy case is handled by actual almost-everywhere
vanishing, including its original exponential integrals.
All statements keep the original probability space, events and scales;
no quadratic-form concentration estimate is assumed.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The original subgaussian variable has the actual two-sided
tail bound. Source: arXiv:2604.01978v1, §6.3,
the scalar exponential estimates in the Hanson–Wright argument. -/
theorem subgaussian_abs_tail_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (t : ℝ) (ht : 0 ≤ t) :
    P.real {ω | t ≤ |X ω|} ≤ 2 * Real.exp (-t ^ 2 / (2 * (c : ℝ))) := by
  have hset : {ω | t ≤ |X ω|} = {ω | t ≤ X ω} ∪ {ω | t ≤ -X ω} := by
    ext ω
    change t ≤ |X ω| ↔ t ≤ X ω ∨ t ≤ -X ω
    exact le_abs (a := t) (b := X ω)
  rw [hset]
  apply (measureReal_union_le _ _).trans
  have hneg : P.real {ω | t ≤ -X ω} ≤ Real.exp (-t ^ 2 / (2 * (c : ℝ))) := by
    simpa only [Pi.neg_apply] using hX.neg.measure_ge_le ht
  have h := add_le_add (hX.measure_ge_le ht) hneg
  simpa only [two_mul] using h

/-- The actual original square event has its original exponential
tail bound. Source: arXiv:2604.01978v1, §6.3,
the squared coordinate estimates for `lem:Hanson_wright`. -/
theorem subgaussian_sq_tail_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (s : ℝ) (hs : 0 ≤ s) :
    P.real {ω | s ≤ (X ω) ^ 2} ≤ 2 * Real.exp (-s / (2 * (c : ℝ))) := by
  have hset : {ω | s ≤ (X ω) ^ 2} = {ω | Real.sqrt s ≤ |X ω|} := by
    ext ω
    simp only [Real.sqrt_le_iff, abs_nonneg, true_and, sq_abs]
  rw [hset]
  simpa only [Real.sq_sqrt hs] using subgaussian_abs_tail_le P X c hX (Real.sqrt s) (Real.sqrt_nonneg s)

/-- The same actual original two-sided tail bounds its original
extended-real probability. Source: arXiv:2604.01978v1, §6.3,
the original probability convention of `lem:Hanson_wright`. -/
theorem subgaussian_abs_tail_ennreal_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (t : ℝ) (ht : 0 ≤ t) :
    P {ω | t ≤ |X ω|} ≤ ENNReal.ofReal (2 * Real.exp (-t ^ 2 / (2 * (c : ℝ)))) := by
  have h := ENNReal.ofReal_le_ofReal (subgaussian_abs_tail_le P X c hX t ht)
  simpa only [Measure.real, ENNReal.ofReal_toReal (measure_ne_top P _)] using h

/-- The actual original square tail also bounds its genuine original
extended-real probability. Source: arXiv:2604.01978v1, §6.3,
the tail probability input to the Hanson–Wright exponential-square estimate. -/
theorem subgaussian_sq_tail_ennreal_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (s : ℝ) (hs : 0 ≤ s) :
    P {ω | s ≤ (X ω) ^ 2} ≤ ENNReal.ofReal (2 * Real.exp (-s / (2 * (c : ℝ)))) := by
  have h := ENNReal.ofReal_le_ofReal (subgaussian_sq_tail_le P X c hX s hs)
  simpa only [Measure.real, ENNReal.ofReal_toReal (measure_ne_top P _)] using h

/-- The original linear-form tail retains the exact original
coefficient-square proxy. Source: arXiv:2604.01978v1, §6.3,
the conditional original linear exponential estimate for Hanson–Wright. -/
theorem subgaussian_linearForm_abs_tail_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (a : Fin n → ℝ) (t : ℝ) (ht : 0 ≤ t) :
    P.real {ω | t ≤ |∑ i : Fin n, a i * Z ω i|} ≤
      2 * Real.exp (-t ^ 2 / (2 * (c : ℝ) * ∑ i : Fin n, a i ^ 2)) := by
  have hc : ((∑ i : Fin n, NNReal.mk (a i ^ 2) (sq_nonneg (a i)) * c : ℝ≥0) : ℝ) =
      (c : ℝ) * ∑ i : Fin n, a i ^ 2 := by
    simp only [NNReal.coe_sum, NNReal.coe_mul, NNReal.coe_mk]
    rw [← Finset.sum_mul, mul_comm]
  simpa only [hc, mul_assoc] using subgaussian_abs_tail_le P (fun ω => ∑ i : Fin n, a i * Z ω i)
    _ (hasSubgaussianMGF_linearForm P Z c hindep hZ a) t ht

/-- The actual original squared linear-form tail has the same exact
original coefficient-square denominator. Source: arXiv:2604.01978v1,
§6.3, the squared conditional linear estimate for `lem:Hanson_wright`. -/
theorem subgaussian_linearForm_sq_tail_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (a : Fin n → ℝ) (s : ℝ) (hs : 0 ≤ s) :
    P.real {ω | s ≤ (∑ i : Fin n, a i * Z ω i) ^ 2} ≤
      2 * Real.exp (-s / (2 * (c : ℝ) * ∑ i : Fin n, a i ^ 2)) := by
  have hc : ((∑ i : Fin n, NNReal.mk (a i ^ 2) (sq_nonneg (a i)) * c : ℝ≥0) : ℝ) =
      (c : ℝ) * ∑ i : Fin n, a i ^ 2 := by
    simp only [NNReal.coe_sum, NNReal.coe_mul, NNReal.coe_mk]
    rw [← Finset.sum_mul, mul_comm]
  simpa only [hc, mul_assoc] using subgaussian_sq_tail_le P (fun ω => ∑ i : Fin n, a i * Z ω i)
    _ (hasSubgaussianMGF_linearForm P Z c hindep hZ a) s hs

/-- A genuine zero variance proxy gives the actual original square
exponential integral one, including every real scalar coefficient.
Source: arXiv:2604.01978v1, §6.3, the degenerate original
coordinate case in the Hanson–Wright exponential estimates. -/
theorem subgaussian_zero_proxy_square_exp {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ)
    (hX : HasSubgaussianMGF X 0 P) (r : ℝ) :
    Integrable (fun ω => Real.exp (r * (X ω) ^ 2)) P ∧
      (∫ ω, Real.exp (r * (X ω) ^ 2) ∂P) = 1 := by
  have hz := hX.ae_eq_zero_of_hasSubgaussianMGF_zero
  have he : (fun ω => Real.exp (r * (X ω) ^ 2)) =ᵐ[P] (fun _ => (1 : ℝ)) := by
    filter_upwards [hz] with ω hω
    change X ω = 0 at hω
    simp only [hω, zero_pow (by decide : 2 ≠ 0), mul_zero, Real.exp_zero]
  refine ⟨(integrable_const (1 : ℝ)).congr he.symm, ?_⟩
  rw [integral_congr_ae he]
  simp

/-- The genuine scalar standard law and a positive actual threshold
jointly witness the original nondegenerate scalar-tail hypotheses. -/
example : HasSubgaussianMGF (id : ℝ → ℝ) 1 (gaussianReal 0 1) ∧
    IsProbabilityMeasure (gaussianReal 0 1) ∧ (0 : ℝ) ≤ 1 :=
  ⟨hasSubgaussianMGF_standardGaussianReal, inferInstance, by norm_num⟩

/-- Genuine independent standard coordinates and a positive original
threshold jointly witness the weighted tail hypotheses. -/
example : iIndepFun (fun i : Fin 2 => fun ω : Fin 2 → ℝ => ω i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun ω : Fin 2 → ℝ => ω i) 1
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨?_, hasSubgaussianMGF_standardProduct_coordinate, inferInstance, by norm_num⟩
  simpa only [id_eq] using iIndepFun_pi (μ := fun _ : Fin 2 => gaussianReal 0 1)
    (X := fun _ : Fin 2 => id) (fun i => measurable_id.aemeasurable)

/-- The genuine zero scalar law witnesses the degenerate exponential
hypothesis on an original probability space. -/
example : HasSubgaussianMGF (fun _ : Unit => (0 : ℝ)) 0 (Measure.dirac ()) ∧
    IsProbabilityMeasure (Measure.dirac ()) := ⟨HasSubgaussianMGF.fun_zero, inferInstance⟩

end Transformer.Homogenized
