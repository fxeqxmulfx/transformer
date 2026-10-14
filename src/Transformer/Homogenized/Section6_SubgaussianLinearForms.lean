/-
# Original subgaussian means and exact linear-form variance proxies

ArXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. The source's
independent mean-zero subgaussian coordinates are encoded by their
actual moment-generating-function bounds. These bounds force the
original means to vanish; zero mean is not supplied as a new input.

At zero, the original MGF equals one and its actual derivative is
the original mean. Its subgaussian upper bound touches it there
and has derivative zero. Fermat's theorem applied to their actual
difference proves the original centering.

The original weighted sums have finite expectations, and independence
gives their exact variance proxies: the coordinate proxy times the sum
of the original squared coefficients. This is the conditional linear
MGF input in the general quadratic-form proof, with the original
coordinates and coefficients retained. Genuine standard Gaussian
coordinates witness the hypotheses with positive variance proxies.
-/

import Transformer.Homogenized.RandomChain
import Mathlib.Probability.Moments.MGFAnalytic
import Mathlib.Analysis.Calculus.LocalExtr.Basic
import Mathlib.Probability.Distributions.Gaussian.Real

open Real MeasureTheory ProbabilityTheory Filter
open scoped BigOperators NNReal Topology

namespace Transformer.Homogenized

/-- The original subgaussian MGF hypothesis forces the original
mean to be zero. Source: arXiv:2604.01978v1, §6.3,
the mean-zero coordinates in `lem:Hanson_wright`.
The actual MGF derivative and its touching upper bound supply the proof. -/
theorem subgaussian_integral_eq_zero {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) : (∫ ω, X ω ∂P) = 0 := by
  have hm : HasDerivAt (mgf X P) (∫ ω, X ω ∂P) 0 := by
    have h := hasDerivAt_mgf (X := X) (μ := P) (t := 0) (by
      rw [hX.integrableExpSet_eq_univ, interior_univ]
      exact Set.mem_univ _)
    simpa only [zero_mul, Real.exp_zero, mul_one] using h
  have hu : HasDerivAt (fun r : ℝ => Real.exp ((c : ℝ) * r ^ 2 / 2)) 0 0 := by
    simpa using (((hasDerivAt_id (0 : ℝ)).pow 2).const_mul (c : ℝ)).div_const 2 |>.exp
  have hdiff : HasDerivAt (fun r : ℝ => Real.exp ((c : ℝ) * r ^ 2 / 2) - mgf X P r)
      (-(∫ ω, X ω ∂P)) 0 := by
    have h := hu.sub hm
    change HasDerivAt (fun r : ℝ => Real.exp ((c : ℝ) * r ^ 2 / 2) - mgf X P r)
      (0 - ∫ ω, X ω ∂P) 0 at h
    simpa only [zero_sub] using h
  have hmin : IsLocalMin (fun r : ℝ => Real.exp ((c : ℝ) * r ^ 2 / 2) - mgf X P r) 0 := by
    change ∀ᶠ r in 𝓝 (0 : ℝ), Real.exp ((c : ℝ) * 0 ^ 2 / 2) - mgf X P 0 ≤
      Real.exp ((c : ℝ) * r ^ 2 / 2) - mgf X P r
    apply Eventually.of_forall
    intro r
    simp only [zero_pow (by decide : 2 ≠ 0), mul_zero, zero_div, Real.exp_zero, mgf_zero, sub_self]
    exact sub_nonneg.mpr (hX.mgf_le r)
  exact neg_eq_zero.mp (hmin.hasDerivAt_eq_zero hdiff)

/-- The original linear form of subgaussian coordinates has a finite
expectation. Source: arXiv:2604.01978v1, §6.3,
the actual weighted coordinates in the Hanson–Wright argument. -/
theorem integrable_subgaussian_linearForm {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (a : Fin n → ℝ) :
    Integrable (fun ω => ∑ i : Fin n, a i * Z ω i) P :=
  integrable_finsetSum _ (fun i _ => (hZ i).integrable.const_mul (a i))

/-- The actual original linear-form mean is zero, without requiring
independence to commute a finite sum with the original expectation.
Source: arXiv:2604.01978v1, §6.3, the original centering
used in `lem:Hanson_wright`. -/
theorem integral_subgaussian_linearForm_eq_zero {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (a : Fin n → ℝ) :
    (∫ ω, ∑ i : Fin n, a i * Z ω i ∂P) = 0 := by
  rw [integral_finsetSum _ (fun i hi => (hZ i).integrable.const_mul (a i))]
  simp only [integral_const_mul, subgaussian_integral_eq_zero P _ c (hZ _), mul_zero, Finset.sum_const_zero]

/-- Independence gives the actual weighted sum its exact original
subgaussian variance proxy. Source: arXiv:2604.01978v1, §6.3,
the linear MGF conditioning in `lem:Hanson_wright`.
The original coefficient squares are retained without a norm substitute. -/
theorem hasSubgaussianMGF_linearForm {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (a : Fin n → ℝ) :
    HasSubgaussianMGF (fun ω => ∑ i : Fin n, a i * Z ω i)
      (∑ i : Fin n, NNReal.mk (a i ^ 2) (sq_nonneg (a i)) * c) P := by
  have hscaled : iIndepFun (fun i : Fin n => fun ω => a i * Z ω i) P := by
    simpa only [Function.comp_def] using
      hindep.comp (fun (i : Fin n) (r : ℝ) => a i * r) (fun i => by fun_prop)
  exact HasSubgaussianMGF.sum_of_iIndepFun hscaled (fun i hi => (hZ i).const_mul (a i))

/-- Every actual original linear exponential test is integrable.
Source: arXiv:2604.01978v1, §6.3, the finite conditional
exponential expectations in the Hanson–Wright argument. -/
theorem integrable_exp_subgaussian_linearForm {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (a : Fin n → ℝ) (t : ℝ) :
    Integrable (fun ω => Real.exp (t * ∑ i : Fin n, a i * Z ω i)) P :=
  (hasSubgaussianMGF_linearForm P Z c hindep hZ a).integrable_exp_mul t

/-- The exact original coefficient-square proxy gives the actual
linear exponential expectation bound. Source: arXiv:2604.01978v1,
§6.3, the conditional linear MGF estimate for `lem:Hanson_wright`. -/
theorem subgaussian_linearForm_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (a : Fin n → ℝ) (t : ℝ) :
    (∫ ω, Real.exp (t * ∑ i : Fin n, a i * Z ω i) ∂P) ≤
      Real.exp ((c : ℝ) * (∑ i : Fin n, a i ^ 2) * t ^ 2 / 2) := by
  have hproxy : ((∑ i : Fin n, NNReal.mk (a i ^ 2) (sq_nonneg (a i)) * c : ℝ≥0) : ℝ) =
      (c : ℝ) * ∑ i : Fin n, a i ^ 2 := by
    simp only [NNReal.coe_sum, NNReal.coe_mul, NNReal.coe_mk]
    rw [← Finset.sum_mul, mul_comm]
  simpa only [mgf, hproxy] using (hasSubgaussianMGF_linearForm P Z c hindep hZ a).mgf_le t

/-- The genuine scalar standard Gaussian has the original positive
variance proxy one. Source: arXiv:2604.01978v1, §6.3,
the standard Gaussian comparison in the Hanson–Wright argument. -/
theorem hasSubgaussianMGF_standardGaussianReal :
    HasSubgaussianMGF (id : ℝ → ℝ) 1 (gaussianReal 0 1) := by
  refine ⟨fun t => ?_, fun t => ?_⟩
  · simpa only [id_eq] using integrable_exp_mul_gaussianReal (μ := 0) (v := 1) t
  · rw [mgf_gaussianReal (HasLaw.id (μ := gaussianReal 0 1)) t]
    simp only [zero_mul, zero_add, NNReal.coe_one, one_mul]
    exact le_rfl

/-- Each genuine original standard product coordinate has positive
subgaussian proxy one. Source: arXiv:2604.01978v1, §6.3,
the actual independent Gaussian comparison coordinates. -/
theorem hasSubgaussianMGF_standardProduct_coordinate {n : ℕ} (i : Fin n) :
    HasSubgaussianMGF (fun ω : Fin n → ℝ => ω i) 1
      (Measure.pi (fun _ : Fin n => gaussianReal 0 1)) := by
  have hmap := (measurePreserving_eval (fun _ : Fin n => gaussianReal 0 1) i).map_eq
  have h : HasSubgaussianMGF (id : ℝ → ℝ) 1
      ((Measure.pi (fun _ : Fin n => gaussianReal 0 1)).map (fun ω => ω i)) := by
    rw [hmap]
    exact hasSubgaussianMGF_standardGaussianReal
  have heval : AEMeasurable (fun ω : Fin n → ℝ => ω i)
      (Measure.pi (fun _ : Fin n => gaussianReal 0 1)) := (measurable_pi_apply i).aemeasurable
  simpa only [Function.comp_def, id_eq] using HasSubgaussianMGF.of_map heval h

/-- Two genuine independent coordinates with positive proxies and a
nonzero original linear form jointly witness all linear-form inputs. -/
example : iIndepFun (fun i : Fin 2 => fun ω : Fin 2 → ℝ => ω i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun ω : Fin 2 → ℝ => ω i) 1
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)) (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  refine ⟨?_, hasSubgaussianMGF_standardProduct_coordinate, inferInstance, rfl⟩
  simpa only [id_eq] using iIndepFun_pi (μ := fun _ : Fin 2 => gaussianReal 0 1)
    (X := fun _ : Fin 2 => id) (fun i => measurable_id.aemeasurable)

end Transformer.Homogenized
