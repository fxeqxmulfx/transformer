module

public import Transformer.Homogenized.Section6_MatrixNorms
public import Mathlib.Probability.Distributions.Gaussian.Multivariate
public import Mathlib.Analysis.Normed.Lp.MeasurableSpace

/-!
# Positive diagonal square-exponential estimates

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The off-diagonal argument uses an auxiliary standard Gaussian.
Its positive quadratic form is reduced to a positive diagonal
form in an orthonormal eigenbasis. This module supplies the
diagonal exponential estimate with the actual coefficient sum.
First it is proved for arbitrary independent subgaussian coordinates;
then it is transferred to the genuine product and Euclidean Gaussian
laws, with measurability and integrability retained explicitly.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Actual weighted coordinate squares preserve independence. Source:
arXiv:2604.01978v1, §6.3, the auxiliary square-exponential argument
for `lem:Hanson_wright`. -/
theorem iIndepFun_weighted_squares {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P) (a : Fin n → ℝ) :
    iIndepFun (fun i : Fin n => fun ω => a i * (Z ω i) ^ 2) P := by
  simpa only [Function.comp_def] using hindep.comp
    (fun (i : Fin n) (x : ℝ) => a i * x ^ 2)
    (fun i => (continuous_const.mul (continuous_id.fun_pow 2)).measurable)

/-- Actual independent positive coordinate squares have an integrable
sum exponential bounded by the exact coefficient sum. Source:
arXiv:2604.01978v1, §6.3, the positive Gaussian-comparison step
for `lem:Hanson_wright`. The original subgaussian proxy is retained. -/
theorem subgaussian_positive_diagonal_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (a : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i) (r : ℝ) (hr : 0 ≤ r)
    (hsmall : ∀ i, r * (c : ℝ) * a i ≤ 1 / 4) :
    Integrable (fun ω => Real.exp (r * ∑ i : Fin n, a i * (Z ω i) ^ 2)) P ∧
      (∫ ω, Real.exp (r * ∑ i : Fin n, a i * (Z ω i) ^ 2) ∂P) ≤
        Real.exp (8 * r * (c : ℝ) * ∑ i : Fin n, a i) := by
  let F : Fin n → Ω → ℝ := fun i ω => a i * (Z ω i) ^ 2
  have hf : iIndepFun F P := iIndepFun_weighted_squares P Z hindep a
  have hm : ∀ i, AEMeasurable (F i) P := fun i =>
    ((integrable_subgaussian_square P (fun ω => Z ω i) c (hZ i)).const_mul (a i)).aemeasurable
  have hscalar : ∀ i, Integrable (fun ω => Real.exp (r * F i ω)) P ∧
      (∫ ω, Real.exp (r * F i ω) ∂P) ≤ Real.exp (8 * (r * a i) * (c : ℝ)) := by
    intro i
    have hs : (r * a i) * (c : ℝ) ≤ 1 / 4 := by
      simpa only [mul_assoc, mul_comm, mul_left_comm] using hsmall i
    simpa only [F, mul_assoc] using subgaussian_square_exp_integral_le P (fun ω => Z ω i)
      c (hZ i) (r * a i) (mul_nonneg hr (ha i)) hs
  obtain ⟨hi, hb⟩ := independent_exp_sum_integral_le P F hf hm r Finset.univ
    (fun i => 8 * (r * a i) * (c : ℝ)) (fun i _ => (hscalar i).1) (fun i _ => (hscalar i).2)
  have he : (∑ i : Fin n, 8 * (r * a i) * (c : ℝ)) = 8 * r * (c : ℝ) * ∑ i : Fin n, a i := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  simpa only [F, Finset.sum_apply, he] using And.intro hi hb

/-- The same exact positive estimate applies to the original diagonal
matrix form. Source: arXiv:2604.01978v1, §6.3, the auxiliary
positive quadratic form in `lem:Hanson_wright`. -/
theorem subgaussian_positive_diagonal_quadratic_exp {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (a : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i) (r : ℝ) (hr : 0 ≤ r)
    (hsmall : ∀ i, r * (c : ℝ) * a i ≤ 1 / 4) :
    Integrable (fun ω => Real.exp (r * hansonWrightQuadraticForm (Matrix.diagonal a) (Z ω))) P ∧
      (∫ ω, Real.exp (r * hansonWrightQuadraticForm (Matrix.diagonal a) (Z ω)) ∂P) ≤
        Real.exp (8 * r * (c : ℝ) * ∑ i : Fin n, a i) := by
  simpa only [hansonWrightQuadraticForm_diagonal] using
    subgaussian_positive_diagonal_exp_integral_le P Z c hindep hZ a ha r hr hsmall

/-- Genuine independent standard Gaussian coordinates satisfy the
positive diagonal estimate. Source: arXiv:2604.01978v1, §6.3,
the actual auxiliary Gaussian in `lem:Hanson_wright`. -/
theorem standardProduct_positive_diagonal_exp {n : ℕ}
    (a : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i) (r : ℝ) (hr : 0 ≤ r)
    (hsmall : ∀ i, r * a i ≤ 1 / 4) :
    Integrable (fun z : Fin n → ℝ => Real.exp (r * ∑ i : Fin n, a i * (z i) ^ 2))
      (Measure.pi (fun _ : Fin n => gaussianReal 0 1)) ∧
    (∫ z : Fin n → ℝ, Real.exp (r * ∑ i : Fin n, a i * (z i) ^ 2)
      ∂Measure.pi (fun _ : Fin n => gaussianReal 0 1)) ≤ Real.exp (8 * r * ∑ i : Fin n, a i) := by
  let P := Measure.pi (fun _ : Fin n => gaussianReal 0 1)
  let Z : (Fin n → ℝ) → EucSpace n := WithLp.toLp 2
  have hf : iIndepFun (fun i : Fin n => fun z : Fin n → ℝ => Z z i) P := by
    simpa only [Z, WithLp.ofLp_toLp, id_eq] using iIndepFun_pi
      (μ := fun _ : Fin n => gaussianReal 0 1) (X := fun _ => id) (fun i => measurable_id.aemeasurable)
  have hZ : ∀ i : Fin n, HasSubgaussianMGF (fun z : Fin n → ℝ => Z z i) 1 P := by
    simpa only [Z, WithLp.ofLp_toLp] using (hasSubgaussianMGF_standardProduct_coordinate (n := n))
  simpa only [Z, WithLp.ofLp_toLp, NNReal.coe_one, mul_one] using
    subgaussian_positive_diagonal_exp_integral_le P Z 1 hf hZ a ha r hr
      (by simpa only [NNReal.coe_one, mul_one] using hsmall)

/-- The exact positive diagonal estimate holds for the original
Euclidean standard Gaussian law, with true integrability transferred
through its measurable product representation. Source: arXiv:2604.01978v1,
§6.3, the auxiliary quadratic-exponential step in `lem:Hanson_wright`. -/
theorem stdGaussian_positive_diagonal_exp {n : ℕ}
    (a : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i) (r : ℝ) (hr : 0 ≤ r)
    (hsmall : ∀ i, r * a i ≤ 1 / 4) :
    Integrable (fun z : EucSpace n => Real.exp (r * ∑ i : Fin n, a i * (z i) ^ 2))
      (stdGaussian (EucSpace n)) ∧
    (∫ z : EucSpace n, Real.exp (r * ∑ i : Fin n, a i * (z i) ^ 2) ∂stdGaussian (EucSpace n)) ≤
      Real.exp (8 * r * ∑ i : Fin n, a i) := by
  obtain ⟨hi, hb⟩ := standardProduct_positive_diagonal_exp a ha r hr hsmall
  let f : EucSpace n → ℝ := fun z => Real.exp (r * ∑ i : Fin n, a i * (z i) ^ 2)
  have hm : StronglyMeasurable f := by
    apply Measurable.stronglyMeasurable
    dsimp only [f]
    fun_prop
  have hg : AEMeasurable (WithLp.toLp 2 : (Fin n → ℝ) → EucSpace n)
      (Measure.pi (fun _ : Fin n => gaussianReal 0 1)) :=
    (WithLp.measurable_toLp 2 (Fin n → ℝ)).aemeasurable
  rw [← map_pi_eq_stdGaussian]
  refine ⟨(integrable_map_measure hm.aestronglyMeasurable hg).mpr ?_, ?_⟩
  · simpa only [f, Function.comp_def, WithLp.ofLp_toLp] using hi
  · rw [integral_map hg hm.aestronglyMeasurable]
    simpa only [f, WithLp.ofLp_toLp] using hb

/-- The actual Euclidean Gaussian norm-square exponential is integrable
with its dimension-dependent trace bound. Source: arXiv:2604.01978v1,
§6.3, the identity-matrix auxiliary Gaussian case for `lem:Hanson_wright`. -/
theorem stdGaussian_norm_sq_exp_integral_le (n : ℕ) (r : ℝ) (hr : 0 ≤ r) (hsmall : r ≤ 1 / 4) :
    Integrable (fun z : EucSpace n => Real.exp (r * ‖z‖ ^ 2)) (stdGaussian (EucSpace n)) ∧
      (∫ z : EucSpace n, Real.exp (r * ‖z‖ ^ 2) ∂stdGaussian (EucSpace n)) ≤
        Real.exp (8 * r * (n : ℝ)) := by
  have h := stdGaussian_positive_diagonal_exp (fun _ : Fin n => (1 : ℝ))
    (fun i => by norm_num) r hr (fun i => by simpa only [mul_one] using hsmall)
  simpa only [EuclideanSpace.real_norm_sq_eq, one_mul, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, mul_one] using h

/-- Genuine standard coordinates, a positive identity diagonal and a
positive coefficient jointly witness the positive diagonal hypotheses. -/
example : iIndepFun (fun i : Fin 2 => fun z : Fin 2 → ℝ => z i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun z : Fin 2 → ℝ => z i) 1
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, 0 ≤ (fun _ : Fin 2 => (1 : ℝ)) i) ∧ (0 : ℝ) ≤ 1 / 8 ∧
    (∀ i : Fin 2, (1 / 8 : ℝ) * (fun _ : Fin 2 => (1 : ℝ)) i ≤ 1 / 4) := by
  refine ⟨?_, hasSubgaussianMGF_standardProduct_coordinate, inferInstance,
    fun i => by norm_num, by norm_num, fun i => by norm_num⟩
  simpa only [id_eq] using iIndepFun_pi (μ := fun _ : Fin 2 => gaussianReal 0 1)
    (X := fun _ => id) (fun i => measurable_id.aemeasurable)

end Transformer.Homogenized
