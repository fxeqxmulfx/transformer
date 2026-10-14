/-
# Actual subgaussian square-exponential comparison

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The original matrix-image square exponential is represented by a
genuine Gaussian linear exponential. Tonelli interchanges its
nonnegative expectations before integrability is known. The proved
original subgaussian linear bound and actual Gaussian quadratic bound
then establish finiteness, retaining the original matrix Frobenius
square and operator norm. Measurability of the original vector follows
from its original coordinate hypotheses; no joint-law shortcut is used.
The argument includes singular matrices and zero subgaussian proxies.
The real-valued estimate is derived after the extended-real bound
has proved finiteness of the original expectation.
-/

import Transformer.Homogenized.Section6_GaussianComparisonKernel
import Mathlib.MeasureTheory.Function.SpecialFunctions.Inner

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal ENNReal

namespace Transformer.Homogenized

/-- Actual coordinate AE measurability gives AE measurability of the
original finite Euclidean vector. Source: arXiv:2604.01978v1,
§6.3, the original random vector in `lem:Hanson_wright`. -/
theorem aemeasurable_eucSpace_of_coordinates {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n)
    (hZ : ∀ i : Fin n, AEMeasurable (fun ω => Z ω i) P) : AEMeasurable Z P := by
  have hm : AEMeasurable (fun ω => (Z ω).ofLp) P := aemeasurable_pi_iff.mpr hZ
  have h := (WithLp.measurable_toLp 2 (Fin n → ℝ)).comp_aemeasurable hm
  simpa only [Function.comp_def, WithLp.toLp_ofLp] using h

/-- The original matrix-image square exponential has a finite bound
derived by actual Gaussian comparison in extended-real integrals.
Source: arXiv:2604.01978v1, §6.3, the squared-linear-form
comparison step for `lem:Hanson_wright`. -/
theorem subgaussian_matrix_norm_sq_lintegral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) (hr : 0 ≤ r)
    (hsmall : r * (c : ℝ) * matOpNorm B ^ 2 ≤ 1 / 4) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (r * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)‖ ^ 2)) ∂P) ≤
      ENNReal.ofReal (Real.exp (8 * r * (c : ℝ) * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2)) := by
  let G := stdGaussian (EucSpace n)
  let T : EucSpace n →L[ℝ] EucSpace n := Matrix.toEuclideanCLM (𝕜 := ℝ) B
  let F : Ω → EucSpace n → ℝ≥0∞ := fun ω g =>
    ENNReal.ofReal (Real.exp (Real.sqrt (2 * r) * inner ℝ g (T (Z ω))))
  have hz : AEMeasurable Z P := aemeasurable_eucSpace_of_coordinates P Z (fun i => (hZ i).aemeasurable)
  have hfst : AEMeasurable (fun p : Ω × EucSpace n => Z p.1) (P.prod G) := by
    simpa only [Function.comp_def] using hz.comp_quasiMeasurePreserving
      (measurePreserving_fst (μ := P) (ν := G)).quasiMeasurePreserving
  have htz : AEMeasurable (fun p : Ω × EucSpace n => T (Z p.1)) (P.prod G) :=
    T.continuous.measurable.comp_aemeasurable hfst
  have hinner : AEMeasurable (fun p : Ω × EucSpace n => inner ℝ p.2 (T (Z p.1))) (P.prod G) :=
    measurable_snd.aemeasurable.inner htz
  have hm : AEMeasurable (Function.uncurry F) (P.prod G) :=
    ENNReal.measurable_ofReal.comp_aemeasurable (aemeasurable_const.fun_mul hinner).exp
  have he : (∫⁻ ω, ENNReal.ofReal (Real.exp (r * ‖T (Z ω)‖ ^ 2)) ∂P) =
      ∫⁻ g : EucSpace n, ∫⁻ ω, F ω g ∂P ∂G := by
    calc
      _ = ∫⁻ ω, ∫⁻ g : EucSpace n, F ω g ∂G ∂P := by
        apply lintegral_congr
        intro ω
        exact ofReal_exp_norm_sq_eq_gaussian_lintegral (T (Z ω)) r hr
      _ = _ := lintegral_lintegral_swap hm
  have hpoint : ∀ g : EucSpace n, (∫⁻ ω, F ω g ∂P) ≤
      ENNReal.ofReal (Real.exp (((c : ℝ) * r) * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose g‖ ^ 2)) := by
    intro g
    have h := subgaussian_matrix_bilinear_lintegral_le P Z c hindep hZ B g (Real.sqrt (2 * r))
    have hs : (c : ℝ) * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose g‖ ^ 2 * (Real.sqrt (2 * r)) ^ 2 / 2 =
        ((c : ℝ) * r) * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose g‖ ^ 2 := by
      rw [Real.sq_sqrt (by positivity : 0 ≤ 2 * r)]
      ring
    simpa only [F, T, hs] using h
  have hs : ((c : ℝ) * r) * matOpNorm B.transpose ^ 2 ≤ 1 / 4 := by
    simpa only [matOpNorm_transpose, mul_comm, mul_left_comm, mul_assoc] using hsmall
  obtain ⟨hi, hb⟩ := stdGaussian_matrix_norm_sq_exp B.transpose ((c : ℝ) * r) (by positivity) hs
  have hf : (∑ i : Fin n, ∑ j : Fin n, B.transpose i j ^ 2) = ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2 := by
    simp only [Matrix.transpose_apply]
    rw [Finset.sum_comm]
  have hb' : (∫ g : EucSpace n,
      Real.exp (((c : ℝ) * r) * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose g‖ ^ 2) ∂G) ≤
      Real.exp (8 * r * (c : ℝ) * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by
    have he' : 8 * ((c : ℝ) * r) * (∑ i : Fin n, ∑ j : Fin n, B.transpose i j ^ 2) =
        8 * r * (c : ℝ) * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2 := by rw [hf]; ring
    simpa only [he'] using hb
  rw [he]
  calc
    _ ≤ ∫⁻ g : EucSpace n, ENNReal.ofReal (Real.exp (((c : ℝ) * r) *
        ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose g‖ ^ 2)) ∂G := lintegral_mono hpoint
    _ = ENNReal.ofReal (∫ g : EucSpace n, Real.exp (((c : ℝ) * r) *
        ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B.transpose g‖ ^ 2) ∂G) :=
      (ofReal_integral_eq_lintegral_ofReal hi (ae_of_all _ (fun g => Real.exp_nonneg _))).symm
    _ ≤ _ := ENNReal.ofReal_le_ofReal hb'

/-- Actual Gaussian comparison proves integrability and the original
real-valued square-exponential bound without assuming that moment.
Source: arXiv:2604.01978v1, §6.3, the auxiliary subgaussian
quadratic step for `lem:Hanson_wright`. -/
theorem subgaussian_matrix_norm_sq_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) (hr : 0 ≤ r)
    (hsmall : r * (c : ℝ) * matOpNorm B ^ 2 ≤ 1 / 4) :
    Integrable (fun ω => Real.exp (r * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)‖ ^ 2)) P ∧
    (∫ ω, Real.exp (r * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)‖ ^ 2) ∂P) ≤
      Real.exp (8 * r * (c : ℝ) * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by
  have hb := subgaussian_matrix_norm_sq_lintegral_le P Z c hindep hZ B r hr hsmall
  have hz := aemeasurable_eucSpace_of_coordinates P Z (fun i => (hZ i).aemeasurable)
  have ht : AEMeasurable (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)) P :=
    (Matrix.toEuclideanCLM (𝕜 := ℝ) B).continuous.measurable.comp_aemeasurable hz
  have hm : AEMeasurable (fun ω => Real.exp (r * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)‖ ^ 2)) P := by
    have hh : AEMeasurable (fun ω => r * (‖Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)‖ *
        ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)‖)) P :=
      aemeasurable_const.fun_mul (ht.norm.fun_mul ht.norm)
    simpa only [pow_two] using hh.exp
  have hn : 0 ≤ᵐ[P] (fun ω => Real.exp (r * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)‖ ^ 2)) :=
    ae_of_all _ (fun ω => Real.exp_nonneg _)
  have hi : Integrable (fun ω => Real.exp (r * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B (Z ω)‖ ^ 2)) P :=
    ⟨hm.aestronglyMeasurable, (hasFiniteIntegral_iff_ofReal hn).mpr (hb.trans_lt ENNReal.ofReal_lt_top)⟩
  refine ⟨hi, ?_⟩
  have hh := ENNReal.toReal_mono ENNReal.ofReal_ne_top hb
  rw [← ofReal_integral_eq_lintegral_ofReal hi hn,
    ENNReal.toReal_ofReal (integral_nonneg (fun ω => Real.exp_nonneg _)),
    ENNReal.toReal_ofReal (Real.exp_nonneg _)] at hh
  exact hh

/-- Genuine standard coordinates, a positive coefficient and the
identity matrix jointly witness every original comparison hypothesis. -/
example : ∃ Z : (Fin 2 → ℝ) → EucSpace 2,
    iIndepFun (fun i : Fin 2 => fun z => Z z i) (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun z => Z z i) 1 (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    AEMeasurable Z (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧ (0 : ℝ) ≤ 1 / 8 ∧
    (1 / 8 : ℝ) * (1 : ℝ≥0) * matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) ^ 2 ≤ 1 / 4 := by
  have hn : matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) = 1 := by
    rw [← Matrix.diagonal_one, matOpNorm_diagonal]
    simp
  refine ⟨WithLp.toLp 2, ?_, hasSubgaussianMGF_standardProduct_coordinate,
    (WithLp.measurable_toLp 2 (Fin 2 → ℝ)).aemeasurable, inferInstance, by norm_num, ?_⟩
  · simpa only [WithLp.ofLp_toLp, id_eq] using iIndepFun_pi
      (μ := fun _ : Fin 2 => gaussianReal 0 1) (X := fun _ => id) (fun i => measurable_id.aemeasurable)
  · rw [hn]
    norm_num

end Transformer.Homogenized
