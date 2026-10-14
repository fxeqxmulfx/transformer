/-
# Independent diagonal quadratic-form MGFs

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The diagonal part is a weighted sum of the original coordinate
squares centered by their original integrals. Independence survives
these actual coordinate transformations. Finite MGF factorization
then retains the exact sum of squared diagonal coefficients.
The auxiliary sum-integrability argument extends the induction in
Mathlib.Probability.Moments.Basic to almost-everywhere measurable
coordinates, as required by the original subgaussian hypotheses.
-/

import Transformer.Homogenized.Section6_CenteredSubgaussianSquare

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Almost-everywhere measurable independent coordinates with integrable
scalar exponentials have an integrable sum exponential. Source:
arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. The induction follows
Mathlib's `iIndepFun.integrable_exp_mul_sum`, using its AE independence
lemma instead of requiring everywhere measurable coordinates. -/
theorem integrable_exp_sum_of_iIndepFun_ae {Ω ι : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : ι → Ω → ℝ)
    (hindep : iIndepFun X P) (hX : ∀ i, AEMeasurable (X i) P)
    (t : ℝ) (s : Finset ι)
    (hi : ∀ i ∈ s, Integrable (fun ω => Real.exp (t * X i ω)) P) :
    Integrable (fun ω => Real.exp (t * (∑ i ∈ s, X i) ω)) P := by
  classical
  induction s using Finset.induction_on with
  | empty =>
      simp only [Finset.sum_empty, Pi.zero_apply, mul_zero, Real.exp_zero]
      exact integrable_const _
  | @insert i s hnot ih =>
      have hs : ∀ j ∈ s, Integrable (fun ω => Real.exp (t * X j ω)) P :=
        fun j hj => hi j (Finset.mem_insert_of_mem hj)
      rw [Finset.sum_insert hnot]
      exact (hindep.indepFun_finsetSum_of_notMem₀ hX hnot).symm.integrable_exp_mul_add
        (hi i (Finset.mem_insert_self _ _)) (ih hs)

/-- Finite independent scalar MGF bounds multiply on the original
probability space, including genuine sum integrability. Source:
arXiv:2604.01978v1, §6.3, the diagonal exponential argument for
`lem:Hanson_wright`. -/
theorem independent_exp_sum_integral_le {Ω ι : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : ι → Ω → ℝ)
    (hindep : iIndepFun X P) (hX : ∀ i, AEMeasurable (X i) P)
    (t : ℝ) (s : Finset ι) (b : ι → ℝ)
    (hi : ∀ i ∈ s, Integrable (fun ω => Real.exp (t * X i ω)) P)
    (hb : ∀ i ∈ s, (∫ ω, Real.exp (t * X i ω) ∂P) ≤ Real.exp (b i)) :
    Integrable (fun ω => Real.exp (t * (∑ i ∈ s, X i) ω)) P ∧
      (∫ ω, Real.exp (t * (∑ i ∈ s, X i) ω) ∂P) ≤ Real.exp (∑ i ∈ s, b i) := by
  refine ⟨integrable_exp_sum_of_iIndepFun_ae P X hindep hX t s hi, ?_⟩
  change mgf (∑ i ∈ s, X i) P t ≤ _
  rw [hindep.mgf_sum₀ hX s, Real.exp_sum]
  apply Finset.prod_le_prod₀
  · intro i _
    exact integral_nonneg (fun ω => Real.exp_nonneg _)
  · exact hb

/-- The original weighted mean-centered squares remain independent.
Source: arXiv:2604.01978v1, §6.3, the diagonal part of
`lem:Hanson_wright`; no replacement of the original centering is made. -/
theorem iIndepFun_weighted_centered_squares {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P) (a : Fin n → ℝ) :
    iIndepFun (fun i : Fin n => fun ω => a i * ((Z ω i) ^ 2 - ∫ η, (Z η i) ^ 2 ∂P)) P := by
  have hm : ∀ i : Fin n, Measurable (fun x : ℝ => a i * (x ^ 2 - ∫ η, (Z η i) ^ 2 ∂P)) := by
    intro i
    exact (continuous_const.mul ((continuous_id.fun_pow 2).fun_sub continuous_const)).measurable
  simpa only [Function.comp_def] using hindep.comp
    (fun (i : Fin n) (x : ℝ) => a i * (x ^ 2 - ∫ η, (Z η i) ^ 2 ∂P)) hm

/-- The original independent diagonal quadratic term has its universal
quadratic MGF bound with the exact coefficient-square sum. Source:
arXiv:2604.01978v1, §6.3, the diagonal estimate for `lem:Hanson_wright`. -/
theorem subgaussian_centered_diagonal_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (a : Fin n → ℝ) (t : ℝ)
    (ht : ∀ i : Fin n, |t| * (c : ℝ) * |a i| ≤ 1 / (32 * Real.exp 1)) :
    Integrable (fun ω => Real.exp (t * ∑ i : Fin n,
      a i * ((Z ω i) ^ 2 - ∫ η, (Z η i) ^ 2 ∂P))) P ∧
    (∫ ω, Real.exp (t * ∑ i : Fin n,
      a i * ((Z ω i) ^ 2 - ∫ η, (Z η i) ^ 2 ∂P)) ∂P) ≤
      Real.exp (2048 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, a i ^ 2) := by
  let F : Fin n → Ω → ℝ := fun i ω => a i * ((Z ω i) ^ 2 - ∫ η, (Z η i) ^ 2 ∂P)
  have hf : iIndepFun F P := iIndepFun_weighted_centered_squares P Z hindep a
  have hm : ∀ i : Fin n, AEMeasurable (F i) P := fun i =>
    ((integrable_centered_subgaussian_square P (fun ω => Z ω i) c (hZ i)).const_mul (a i)).aemeasurable
  have hscalar : ∀ i : Fin n, Integrable (fun ω => Real.exp (t * F i ω)) P ∧
      (∫ ω, Real.exp (t * F i ω) ∂P) ≤
        Real.exp (2048 * (Real.exp 1) ^ 4 * (t * a i) ^ 2 * (c : ℝ) ^ 2) := by
    intro i
    have hh : |t * a i| * (c : ℝ) ≤ 1 / (32 * Real.exp 1) := by
      simpa only [abs_mul, mul_assoc, mul_left_comm, mul_comm] using ht i
    simpa only [F, mul_assoc] using
      subgaussian_centered_square_exp_integral_le P (fun ω => Z ω i) c (hZ i) (t * a i) hh
  obtain ⟨hi, hb⟩ := independent_exp_sum_integral_le P F hf hm t Finset.univ
    (fun i => 2048 * (Real.exp 1) ^ 4 * (t * a i) ^ 2 * (c : ℝ) ^ 2)
    (fun i _ => (hscalar i).1) (fun i _ => (hscalar i).2)
  have he : (∑ i : Fin n, 2048 * (Real.exp 1) ^ 4 * (t * a i) ^ 2 * (c : ℝ) ^ 2) =
      2048 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, a i ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  simpa only [F, Finset.sum_apply, he] using And.intro hi hb

/-- The diagonal estimate applies to the actual diagonal matrix form
centered by its own integral. Source: arXiv:2604.01978v1, §6.3,
the original diagonal quadratic part of `lem:Hanson_wright`. -/
theorem subgaussian_diagonal_quadratic_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (a : Fin n → ℝ) (t : ℝ)
    (ht : ∀ i : Fin n, |t| * (c : ℝ) * |a i| ≤ 1 / (32 * Real.exp 1)) :
    Integrable (fun ω => Real.exp (t * (hansonWrightQuadraticForm (Matrix.diagonal a) (Z ω) -
      ∫ η, hansonWrightQuadraticForm (Matrix.diagonal a) (Z η) ∂P))) P ∧
    (∫ ω, Real.exp (t * (hansonWrightQuadraticForm (Matrix.diagonal a) (Z ω) -
      ∫ η, hansonWrightQuadraticForm (Matrix.diagonal a) (Z η) ∂P)) ∂P) ≤
      Real.exp (2048 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, a i ^ 2) := by
  have he : ∀ ω, hansonWrightQuadraticForm (Matrix.diagonal a) (Z ω) -
      (∫ η, hansonWrightQuadraticForm (Matrix.diagonal a) (Z η) ∂P) =
      ∑ i : Fin n, a i * ((Z ω i) ^ 2 - ∫ η, (Z η i) ^ 2 ∂P) := by
    intro ω
    rw [integral_hansonWrightQuadraticForm P Z c hindep hZ, hansonWrightQuadraticForm_diagonal]
    simp only [Matrix.diagonal_apply_eq, mul_sub, Finset.sum_sub_distrib]
  simpa only [he] using subgaussian_centered_diagonal_exp_integral_le P Z c hindep hZ a t ht

/-- Independent genuine standard coordinates, the identity diagonal and
a nonzero coefficient witness every diagonal hypothesis together. -/
example : iIndepFun (fun i : Fin 2 => fun ω : Fin 2 → ℝ => ω i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun ω : Fin 2 → ℝ => ω i) 1
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, |(1 / (64 * Real.exp 1) : ℝ)| * (1 : ℝ≥0) * |(fun _ : Fin 2 => (1 : ℝ)) i| ≤
      1 / (32 * Real.exp 1)) := by
  refine ⟨?_, hasSubgaussianMGF_standardProduct_coordinate, inferInstance, ?_⟩
  · simpa only [id_eq] using iIndepFun_pi (μ := fun _ : Fin 2 => gaussianReal 0 1)
      (X := fun _ : Fin 2 => id) (fun i => measurable_id.aemeasurable)
  · intro i
    norm_num only [NNReal.coe_one, abs_one, mul_one]
    rw [abs_of_nonneg (by positivity)]
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith [Real.exp_pos (1 : ℝ)]

/-- Genuine independent zero coordinates witness the finite-sum
integrability and scalar-bound assumptions on a probability space. -/
example : ∃ X : Fin 2 → Unit → ℝ, iIndepFun X (Measure.dirac ()) ∧
    (∀ i : Fin 2, AEMeasurable (X i) (Measure.dirac ())) ∧
    (∀ i : Fin 2, Integrable (fun ω => Real.exp ((1 : ℝ) * X i ω)) (Measure.dirac ())) ∧
    (∀ i : Fin 2, (∫ ω, Real.exp ((1 : ℝ) * X i ω) ∂Measure.dirac ()) ≤ Real.exp 0) := by
  refine ⟨fun _ _ => 0, iIndepFun_of_unit _, fun i => aemeasurable_const,
    fun i => integrable_const _, ?_⟩
  intro i
  simp

end Transformer.Homogenized
