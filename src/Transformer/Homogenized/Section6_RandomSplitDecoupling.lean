/-
# Actual random-split quadratic decoupling

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The expectation of the actual random cross matrix equals one quarter
of the original off-diagonal matrix form. Jensen's inequality for the
real exponential therefore transfers its original exponential to the
random cross exponential with coefficient four times as large.
The original fixed-split cross form has the genuine bilinear MGF bound
already proved for independent coordinate blocks. All matrix entries,
Frobenius squares and Euclidean operator norms remain the original ones.
-/

import Transformer.Homogenized.Section6_FairCoordinateSplits
import Mathlib.Analysis.Convex.Integral
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

open Real MeasureTheory ProbabilityTheory Set
open scoped BigOperators NNReal ENNReal

namespace Transformer.Homogenized

/-- The original off-diagonal form is four times the genuine mean
of its random cross form. Source: arXiv:2604.01978v1, §6.3,
the actual averaging identity for `lem:Hanson_wright`. -/
theorem integral_random_cross_quadratic {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (z : EucSpace n) :
    (∫ σ : Fin n → Bool, hansonWrightQuadraticForm
      (coordinateMask (randomCoordinateSubset σ) * B * coordinateMask (randomCoordinateSubset σ)ᶜ) z
      ∂randomSplitLaw n) = (1 / 4 : ℝ) * hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) z := by
  simp_rw [random_cross_quadratic_eq_coin_sum]
  rw [integral_finsetSum _ (fun i _ => Integrable.of_finite)]
  simp_rw [integral_finsetSum _ (fun j _ => Integrable.of_finite), integral_const_mul, integral_split_pair]
  simp only [hansonWrightQuadraticForm, Matrix.sub_apply, Matrix.diagonal_apply]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hij : i = j
  · subst j
    simp
  · simp only [ite_eq_right hij, sub_zero]
    ring

/-- The actual off-diagonal exponential is bounded by the genuine
random-split expectation with its exact factor-four coefficient.
Source: arXiv:2604.01978v1, §6.3, the Jensen decoupling step
for `lem:Hanson_wright`; both coefficient signs are included. -/
theorem random_split_exp_jensen {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (z : EucSpace n) (t : ℝ) :
    Real.exp (t * hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) z) ≤
      ∫ σ : Fin n → Bool, Real.exp (4 * t * hansonWrightQuadraticForm
        (coordinateMask (randomCoordinateSubset σ) * B * coordinateMask (randomCoordinateSubset σ)ᶜ) z)
        ∂randomSplitLaw n := by
  let f : (Fin n → Bool) → ℝ := fun σ => 4 * t * hansonWrightQuadraticForm
    (coordinateMask (randomCoordinateSubset σ) * B * coordinateMask (randomCoordinateSubset σ)ᶜ) z
  have hm : (∫ σ, f σ ∂randomSplitLaw n) =
      t * hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) z := by
    dsimp only [f]
    rw [integral_const_mul, integral_random_cross_quadratic]
    ring
  have h := convexOn_exp.map_integral_le Real.continuous_exp.continuousOn isClosed_univ
    (ae_of_all (randomSplitLaw n) (fun σ => mem_univ (f σ)))
    (integrable_randomSplit_function n f) (integrable_randomSplit_function n (Real.exp ∘ f))
  simpa only [hm, f, Function.comp_def] using h

/-- The actual Jensen comparison also bounds its nonnegative integrals
before original-vector integrability is established. Source:
arXiv:2604.01978v1, §6.3, the Tonelli decoupling form of `lem:Hanson_wright`. -/
theorem random_split_exp_lintegral_jensen {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (z : EucSpace n) (t : ℝ) :
    ENNReal.ofReal (Real.exp (t * hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) z)) ≤
      ∫⁻ σ : Fin n → Bool, ENNReal.ofReal (Real.exp (4 * t * hansonWrightQuadraticForm
        (coordinateMask (randomCoordinateSubset σ) * B * coordinateMask (randomCoordinateSubset σ)ᶜ) z))
        ∂randomSplitLaw n := by
  have h := ENNReal.ofReal_le_ofReal (random_split_exp_jensen B z t)
  rw [ofReal_integral_eq_lintegral_ofReal (integrable_randomSplit_function n _)
    (ae_of_all _ (fun σ => Real.exp_nonneg _))] at h
  exact h

/-- The genuine random cross-exponential kernel is jointly measurable
in its finite split and original Euclidean vector. Source:
arXiv:2604.01978v1, §6.3, the Tonelli decoupling kernel in `lem:Hanson_wright`. -/
theorem measurable_random_cross_exp {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    Measurable (fun p : (Fin n → Bool) × EucSpace n => ENNReal.ofReal (Real.exp (4 * t *
      hansonWrightQuadraticForm (coordinateMask (randomCoordinateSubset p.1) * B *
        coordinateMask (randomCoordinateSubset p.1)ᶜ) p.2))) := by
  apply measurable_from_prod_countable_right
  intro σ
  simp_rw [hansonWrightQuadraticForm_eq_inner]
  have hm : Measurable (fun z : EucSpace n => inner ℝ z
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask (randomCoordinateSubset σ) * B *
        coordinateMask (randomCoordinateSubset σ)ᶜ) z)) :=
    measurable_id.inner (Matrix.toEuclideanCLM (𝕜 := ℝ) _).continuous.measurable
  exact ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp (measurable_const.fun_mul hm))

/-- The original fixed-split cross exponential has the actual bilinear
integral bound with the exact original matrix norms. Source:
arXiv:2604.01978v1, §6.3, the fixed-split MGF step for `lem:Hanson_wright`. -/
theorem subgaussian_cross_mask_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n)) (t : ℝ)
    (hsmall : t ^ 2 * (c : ℝ) ^ 2 * matOpNorm B ^ 2 ≤ 1 / 2) :
    Integrable (fun ω => Real.exp (t * hansonWrightQuadraticForm (coordinateMask s * B * coordinateMask sᶜ) (Z ω))) P ∧
    (∫ ω, Real.exp (t * hansonWrightQuadraticForm (coordinateMask s * B * coordinateMask sᶜ) (Z ω)) ∂P) ≤
      Real.exp (4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by
  have hs : t ^ 2 * (c : ℝ) * (c : ℝ) * matOpNorm B ^ 2 ≤ 1 / 2 := by nlinarith [hsmall]
  have h := independent_subgaussian_bilinear_exp_integral_le P
    (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) (Z ω))
    (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask sᶜ) (Z ω)) c c
    (indepFun_coordinateMasks_compl P Z hindep (fun i => (hZ i).aemeasurable) s)
    (iIndepFun_coordinateMask P Z hindep s) (iIndepFun_coordinateMask P Z hindep sᶜ)
    (hasSubgaussianMGF_coordinateMask P Z c hZ s) (hasSubgaussianMGF_coordinateMask P Z c hZ sᶜ) B t hs
  have he : 4 * t ^ 2 * (c : ℝ) * (c : ℝ) * (∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) =
      4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2 := by ring
  simpa only [cross_mask_quadratic_eq_inner, he] using h

/-- The same fixed-split estimate bounds the original extended-real
integral before exchanging the original and random-split expectations.
Source: arXiv:2604.01978v1, §6.3, the decoupling Tonelli input
for `lem:Hanson_wright`. -/
theorem subgaussian_cross_mask_lintegral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n)) (t : ℝ)
    (hsmall : t ^ 2 * (c : ℝ) ^ 2 * matOpNorm B ^ 2 ≤ 1 / 2) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (t * hansonWrightQuadraticForm
      (coordinateMask s * B * coordinateMask sᶜ) (Z ω))) ∂P) ≤
      ENNReal.ofReal (Real.exp (4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2)) := by
  obtain ⟨hi, hb⟩ := subgaussian_cross_mask_exp_integral_le P Z c hindep hZ B s t hsmall
  have h := ENNReal.ofReal_le_ofReal hb
  rw [ofReal_integral_eq_lintegral_ofReal hi (ae_of_all _ (fun ω => Real.exp_nonneg _))] at h
  exact h

/-- Genuine independent standard coordinates and a positive admissible
coefficient witness every fixed-split hypothesis on a probability space. -/
example : ∃ Z : (Fin 2 → ℝ) → EucSpace 2,
    iIndepFun (fun i : Fin 2 => fun z => Z z i) (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun z => Z z i) 1 (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (1 / 8 : ℝ) ^ 2 * ((1 : ℝ≥0) : ℝ) ^ 2 * matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) ^ 2 ≤ 1 / 2 := by
  refine ⟨WithLp.toLp 2, ?_, hasSubgaussianMGF_standardProduct_coordinate, inferInstance, ?_⟩
  · simpa only [WithLp.ofLp_toLp, id_eq] using iIndepFun_pi
      (μ := fun _ : Fin 2 => gaussianReal 0 1) (X := fun _ => id) (fun i => measurable_id.aemeasurable)
  · have hn : matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) = 1 := by rw [← Matrix.diagonal_one, matOpNorm_diagonal]; simp
    rw [hn]
    norm_num

end Transformer.Homogenized
