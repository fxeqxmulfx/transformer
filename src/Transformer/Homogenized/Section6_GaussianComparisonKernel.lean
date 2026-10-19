module

public import Transformer.Homogenized.Section6_GaussianQuadraticMGF
public import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Basic

/-!
# The actual Gaussian comparison kernel

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The off-diagonal exponential estimate compares original independent
subgaussian coordinates with an auxiliary standard Gaussian. Its
linear exponential integral is computed exactly, giving the genuine
Gaussian representation of a squared Euclidean exponential. The
conditional original linear-form estimate retains the exact squared
norm of the transposed matrix image. All scalar integrals are finite;
the extended-real identities can therefore be used in Tonelli's theorem.
The zero vector and empty coordinate family are included in the exact
linear identities, without an inverse covariance assumption.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Every actual standard Gaussian linear exponential is integrable,
and its original MGF has its exact norm-square variance. Source:
arXiv:2604.01978v1, §6.3, the Gaussian-comparison linear identity
for `lem:Hanson_wright`. -/
theorem stdGaussian_linear_exp_integral {n : ℕ} (L : EucSpace n →L[ℝ] ℝ) (t : ℝ) :
    Integrable (fun g : EucSpace n => Real.exp (t * L g)) (stdGaussian (EucSpace n)) ∧
    (∫ g : EucSpace n, Real.exp (t * L g) ∂stdGaussian (EucSpace n)) =
      Real.exp (‖L‖ ^ 2 * t ^ 2 / 2) := by
  have hg : HasGaussianLaw (fun g : EucSpace n => L g) (stdGaussian (EucSpace n)) := by
    simpa only [Function.comp_def, id_eq] using
      (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace n))).map L
  have hl : HasLaw (fun g : EucSpace n => L g) (gaussianReal 0 (‖L‖ ^ 2).toNNReal)
      (stdGaussian (EucSpace n)) := by
    refine ⟨hg.aemeasurable, ?_⟩
    have h := hg.map_eq_gaussianReal
    rw [integral_strongDual_stdGaussian, variance_dual_stdGaussian] at h
    exact h
  refine ⟨hl.integrable_fun_comp (integrable_exp_mul_gaussianReal t), ?_⟩
  simpa only [mgf, zero_mul, zero_add, Real.coe_toNNReal _ (sq_nonneg _)] using mgf_gaussianReal hl t

/-- The original Gaussian inner-product exponential has its exact
squared-vector MGF, including genuine integrability. Source:
arXiv:2604.01978v1, §6.3, the comparison kernel for `lem:Hanson_wright`. -/
theorem stdGaussian_inner_exp_integral {n : ℕ} (v : EucSpace n) (t : ℝ) :
    Integrable (fun g : EucSpace n => Real.exp (t * inner ℝ g v)) (stdGaussian (EucSpace n)) ∧
    (∫ g : EucSpace n, Real.exp (t * inner ℝ g v) ∂stdGaussian (EucSpace n)) =
      Real.exp (‖v‖ ^ 2 * t ^ 2 / 2) := by
  have h := stdGaussian_linear_exp_integral (innerSL ℝ v) t
  have he : ∀ g : EucSpace n, inner ℝ v g = inner ℝ g v := fun g => real_inner_comm g v
  simpa only [innerSL_apply_apply, innerSL_apply_norm, he] using h

/-- The genuine Gaussian inner exponential also has its exact finite
extended-real integral. Source: arXiv:2604.01978v1, §6.3,
the Tonelli input to the comparison proof of `lem:Hanson_wright`. -/
theorem lintegral_stdGaussian_exp_inner {n : ℕ} (v : EucSpace n) (t : ℝ) :
    (∫⁻ g : EucSpace n, ENNReal.ofReal (Real.exp (t * inner ℝ g v)) ∂stdGaussian (EucSpace n)) =
      ENNReal.ofReal (Real.exp (‖v‖ ^ 2 * t ^ 2 / 2)) := by
  obtain ⟨hi, he⟩ := stdGaussian_inner_exp_integral v t
  rw [← ofReal_integral_eq_lintegral_ofReal hi (ae_of_all _ (fun g => Real.exp_nonneg _)), he]

/-- A positive square exponential is the actual Gaussian linear
exponential integral, with the exact square-root coefficient. Source:
arXiv:2604.01978v1, §6.3, the Gaussian comparison identity
used for `lem:Hanson_wright`. -/
theorem exp_norm_sq_eq_gaussian_integral {n : ℕ} (v : EucSpace n) (r : ℝ) (hr : 0 ≤ r) :
    Real.exp (r * ‖v‖ ^ 2) =
      ∫ g : EucSpace n, Real.exp (Real.sqrt (2 * r) * inner ℝ g v) ∂stdGaussian (EucSpace n) := by
  rw [(stdGaussian_inner_exp_integral v (Real.sqrt (2 * r))).2,
    Real.sq_sqrt (by positivity : 0 ≤ 2 * r)]
  congr 1
  ring

/-- The same genuine Gaussian representation holds in extended-real
integrals before any interchange of expectations. Source:
arXiv:2604.01978v1, §6.3, the Tonelli form of the Gaussian
comparison identity for `lem:Hanson_wright`. -/
theorem ofReal_exp_norm_sq_eq_gaussian_lintegral {n : ℕ}
    (v : EucSpace n) (r : ℝ) (hr : 0 ≤ r) :
    ENNReal.ofReal (Real.exp (r * ‖v‖ ^ 2)) =
      ∫⁻ g : EucSpace n, ENNReal.ofReal (Real.exp (Real.sqrt (2 * r) * inner ℝ g v))
        ∂stdGaussian (EucSpace n) := by
  rw [lintegral_stdGaussian_exp_inner, Real.sq_sqrt (by positivity : 0 ≤ 2 * r)]
  congr 2
  ring

/-- Pairing the actual matrix image with an original vector gives the
original transposed-image coefficient sum. Source: arXiv:2604.01978v1,
§6.3, the conditional linear form in `lem:Hanson_wright`. -/
theorem inner_matrix_eq_transpose_coordinate_sum {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (v z : EucSpace n) :
    inner ℝ v (Matrix.toEuclideanCLM (𝕜 := ℝ) B z) =
      ∑ j : Fin n, (Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose v) j * z j := by
  rw [Matrix.inner_toEuclideanCLM]
  simp only [dotProduct, Matrix.mulVec, Matrix.ofLp_toEuclideanCLM, Matrix.transpose_apply,
    Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- The original independent subgaussian vector gives the genuine
conditional bilinear exponential estimate with the exact transposed
image norm. Source: arXiv:2604.01978v1, §6.3, the Gaussian
comparison step in the proof of `lem:Hanson_wright`. -/
theorem subgaussian_matrix_bilinear_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (v : EucSpace n) (t : ℝ) :
    Integrable (fun ω => Real.exp (t * inner ℝ v (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)))) P ∧
    (∫ ω, Real.exp (t * inner ℝ v (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω))) ∂P) ≤
      Real.exp ((c : ℝ) * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose v‖ ^ 2 * t ^ 2 / 2) := by
  have hi := integrable_exp_subgaussian_linearForm P Z c hindep hZ
    (fun j => (Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose v) j) t
  have hb := subgaussian_linearForm_exp_integral_le P Z c hindep hZ
    (fun j => (Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose v) j) t
  simpa only [inner_matrix_eq_transpose_coordinate_sum, EuclideanSpace.real_norm_sq_eq] using And.intro hi hb

/-- The original conditional bilinear bound also controls its genuine
extended-real exponential integral before interchanging expectations.
Source: arXiv:2604.01978v1, §6.3, the Tonelli input to the
Gaussian comparison proof of `lem:Hanson_wright`. -/
theorem subgaussian_matrix_bilinear_lintegral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (v : EucSpace n) (t : ℝ) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (t * inner ℝ v (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)))) ∂P) ≤
      ENNReal.ofReal (Real.exp ((c : ℝ) * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose v‖ ^ 2 * t ^ 2 / 2)) := by
  obtain ⟨hi, hb⟩ := subgaussian_matrix_bilinear_exp_integral_le P Z c hindep hZ B v t
  have h := ENNReal.ofReal_le_ofReal hb
  rw [ofReal_integral_eq_lintegral_ofReal hi (ae_of_all _ (fun ω => Real.exp_nonneg _))] at h
  exact h

/-- Genuine independent standard coordinates and a positive scalar
coefficient jointly witness the original comparison hypotheses. -/
example : iIndepFun (fun i : Fin 2 => fun z : Fin 2 → ℝ => z i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun z : Fin 2 → ℝ => z i) 1
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧ (0 : ℝ) ≤ 1 / 8 := by
  refine ⟨?_, hasSubgaussianMGF_standardProduct_coordinate, by norm_num⟩
  simpa only [id_eq] using iIndepFun_pi (μ := fun _ : Fin 2 => gaussianReal 0 1)
    (X := fun _ => id) (fun i => measurable_id.aemeasurable)

end Transformer.Homogenized
