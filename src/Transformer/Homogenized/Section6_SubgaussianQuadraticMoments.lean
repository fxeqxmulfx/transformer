module

public import Transformer.Homogenized.Section6_SubgaussianLinearForms
public import Mathlib.Probability.Independence.Integration
public import Mathlib.Data.Matrix.Diagonal

/-!
# The original quadratic form and its actual subgaussian expectation

ArXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. The original
quadratic form retains every original matrix entry and coordinate.
All coordinate products have finite expectations by the genuine
subgaussian second moments and Hölder's inequality.

For distinct original coordinates, independence identifies the product
expectation with the product of their actual means. Those means are
already proved zero from their original MGF bounds. The expectation of
the original quadratic form is therefore its original diagonal entries
weighted by the original coordinate second moments.
The variance proxy is never substituted for these actual second moments.
Finite sums are commuted only after each original product is proved
integrable. These facts justify the genuine centering in the original
quadratic-form tail inequality, before its exponential estimates.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual original quadratic form `zᵀBz`.
Source: arXiv:2604.01978v1, §6.3, the original random
quadratic form in `lem:Hanson_wright`. -/
noncomputable def hansonWrightQuadraticForm {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (z : EucSpace n) : ℝ :=
  ∑ i : Fin n, ∑ j : Fin n, B i j * z i * z j

/-- The actual diagonal matrix gives the genuine original weighted
coordinate squares. Source: arXiv:2604.01978v1, §6.3,
the diagonal part of the original Hanson–Wright quadratic form. -/
theorem hansonWrightQuadraticForm_diagonal {n : ℕ} (b : Fin n → ℝ) (z : EucSpace n) :
    hansonWrightQuadraticForm (Matrix.diagonal b) z = ∑ i : Fin n, b i * (z i) ^ 2 := by
  simp [hansonWrightQuadraticForm, Matrix.diagonal_apply, pow_two, mul_assoc]

/-- Addition of the original matrices adds their actual original
quadratic forms. Source: arXiv:2604.01978v1, §6.3,
the original diagonal and remaining terms of `lem:Hanson_wright`. -/
theorem hansonWrightQuadraticForm_add {n : ℕ} (B C : Matrix (Fin n) (Fin n) ℝ) (z : EucSpace n) :
    hansonWrightQuadraticForm (B + C) z = hansonWrightQuadraticForm B z + hansonWrightQuadraticForm C z := by
  simp only [hansonWrightQuadraticForm, Matrix.add_apply, add_mul, Finset.sum_add_distrib]

/-- The actual original quadratic form also retains subtraction of
the original matrices. Source: arXiv:2604.01978v1, §6.3,
removing the original diagonal in the Hanson–Wright argument. -/
theorem hansonWrightQuadraticForm_sub {n : ℕ} (B C : Matrix (Fin n) (Fin n) ℝ) (z : EucSpace n) :
    hansonWrightQuadraticForm (B - C) z = hansonWrightQuadraticForm B z - hansonWrightQuadraticForm C z := by
  simp only [hansonWrightQuadraticForm, Matrix.sub_apply, sub_mul, Finset.sum_sub_distrib]

/-- Any two genuine original subgaussian coordinates have an
integrable actual product. Source: arXiv:2604.01978v1, §6.3,
the finite original quadratic-form expectation of `lem:Hanson_wright`. -/
theorem integrable_subgaussian_mul {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → ℝ) (c d : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (hY : HasSubgaussianMGF Y d P) :
    Integrable (fun ω => X ω * Y ω) P := by
  have hx : MemLp X 2 P := by simpa using hX.memLp 2
  have hy : MemLp Y 2 P := by simpa using hY.memLp 2
  exact hx.integrable_mul hy

/-- The actual original coordinate square has a finite expectation.
Source: arXiv:2604.01978v1, §6.3, the original second moments
in the diagonal Hanson–Wright centering. -/
theorem integrable_subgaussian_square {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X : Ω → ℝ) (c : ℝ≥0) (hX : HasSubgaussianMGF X c P) :
    Integrable (fun ω => (X ω) ^ 2) P := by
  simpa only [pow_two] using integrable_subgaussian_mul P X X c c hX hX

/-- Every original matrix-weighted coordinate product is integrable.
Source: arXiv:2604.01978v1, §6.3, commuting the genuine
quadratic-form expectation with its original finite coordinate sums. -/
theorem integrable_subgaussian_quadratic_coordinate {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (i j : Fin n) :
    Integrable (fun ω => B i j * Z ω i * Z ω j) P := by
  simpa only [mul_assoc] using
    (integrable_subgaussian_mul P (fun ω => Z ω i) (fun ω => Z ω j) c c (hZ i) (hZ j)).const_mul (B i j)

/-- The actual original quadratic form has a finite expectation.
Source: arXiv:2604.01978v1, §6.3, the genuine expectation
subtracted in `lem:Hanson_wright`. Independence is unnecessary for finiteness. -/
theorem integrable_hansonWrightQuadraticForm {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) : Integrable (fun ω => hansonWrightQuadraticForm B (Z ω)) P :=
  integrable_finsetSum _ (fun i _ => integrable_finsetSum _
    (fun j _ => integrable_subgaussian_quadratic_coordinate P Z c hZ B i j))

/-- Distinct independent original subgaussian coordinates have
actual product expectation zero. Source: arXiv:2604.01978v1,
§6.3, the original centering in `lem:Hanson_wright`. -/
theorem integral_subgaussian_coordinate_mul_of_ne {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (i j : Fin n) (hij : i ≠ j) :
    (∫ ω, Z ω i * Z ω j ∂P) = 0 := by
  let : IsProbabilityMeasure P := hindep.isProbabilityMeasure
  have h := (hindep.indepFun hij).integral_mul_eq_mul_integral
    (hZ i).aestronglyMeasurable (hZ j).aestronglyMeasurable
  simpa only [Pi.mul_apply, subgaussian_integral_eq_zero P _ c (hZ i),
    subgaussian_integral_eq_zero P _ c (hZ j), mul_zero] using h

/-- The original coordinate second moments give the actual diagonal
of the original product expectation. Source: arXiv:2604.01978v1,
§6.3, the original quadratic-form mean in `lem:Hanson_wright`. -/
theorem integral_subgaussian_coordinate_mul_eq_diagonal {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (i j : Fin n) :
    (∫ ω, Z ω i * Z ω j ∂P) = if i = j then ∫ ω, (Z ω i) ^ 2 ∂P else 0 := by
  by_cases hij : i = j
  · subst j
    simp only [ite_true, pow_two]
  · rw [ite_eq_right hij, integral_subgaussian_coordinate_mul_of_ne P Z c hindep hZ i j hij]

/-- The genuine original quadratic-form expectation is its original
diagonal weighted by the actual original second moments.
Source: arXiv:2604.01978v1, §6.3, the exact mean subtracted
in `lem:Hanson_wright`. No variance proxy replaces the actual moments. -/
theorem integral_hansonWrightQuadraticForm {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (B : Matrix (Fin n) (Fin n) ℝ) :
    (∫ ω, hansonWrightQuadraticForm B (Z ω) ∂P) =
      ∑ i : Fin n, B i i * ∫ ω, (Z ω i) ^ 2 ∂P := by
  have hint (i j : Fin n) : Integrable (fun ω => B i j * (Z ω i * Z ω j)) P :=
    (integrable_subgaussian_mul P (fun ω => Z ω i) (fun ω => Z ω j) c c (hZ i) (hZ j)).const_mul (B i j)
  simp only [hansonWrightQuadraticForm, mul_assoc]
  rw [integral_finsetSum _ (fun i hi => integrable_finsetSum _ (fun j hj => hint i j))]
  simp_rw [integral_finsetSum _ (fun j hj => hint _ j), integral_const_mul,
    integral_subgaussian_coordinate_mul_eq_diagonal P Z c hindep hZ]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

/-- The actual original centered quadratic form splits into the
original centered coordinate squares and its actual remaining matrix.
Source: arXiv:2604.01978v1, §6.3, the diagonal and remaining
terms in the Hanson–Wright exponential estimate. -/
theorem centered_hansonWrightQuadraticForm_split
    {Ω : Type*} [MeasurableSpace Ω] {n : ℕ} (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (ω : Ω) :
    hansonWrightQuadraticForm B (Z ω) - ∫ ω', hansonWrightQuadraticForm B (Z ω') ∂P =
      (∑ i : Fin n, B i i * ((Z ω i) ^ 2 - ∫ ω', (Z ω' i) ^ 2 ∂P)) +
        hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) (Z ω) := by
  rw [integral_hansonWrightQuadraticForm P Z c hindep hZ B]
  have h := hansonWrightQuadraticForm_sub B (Matrix.diagonal (fun i => B i i)) (Z ω)
  rw [hansonWrightQuadraticForm_diagonal] at h
  simp only [mul_sub, Finset.sum_sub_distrib]
  linarith

/-- Genuine independent standard coordinates jointly witness the
subgaussian and distinct-index inputs. The identity matrix gives a
genuine nonzero quadratic form taking the value two at the vector of ones. -/
example : iIndepFun (fun i : Fin 2 => fun ω : Fin 2 → ℝ => ω i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun ω : Fin 2 → ℝ => ω i) 1
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    (0 : Fin 2) ≠ 1 ∧ hansonWrightQuadraticForm (1 : Matrix (Fin 2) (Fin 2) ℝ)
      (WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) = 2 := by
  refine ⟨?_, hasSubgaussianMGF_standardProduct_coordinate, by decide, ?_⟩
  · simpa only [id_eq] using iIndepFun_pi (μ := fun _ : Fin 2 => gaussianReal 0 1)
      (X := fun _ : Fin 2 => id) (fun i => measurable_id.aemeasurable)
  · simp [hansonWrightQuadraticForm, Matrix.one_apply]

end Transformer.Homogenized
