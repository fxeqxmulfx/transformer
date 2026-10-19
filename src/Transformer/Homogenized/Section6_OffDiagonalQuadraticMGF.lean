module

public import Transformer.Homogenized.Section6_RandomSplitDecoupling

/-!
# The original off-diagonal quadratic MGF

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The actual off-diagonal form is decoupled by the proved fair-split
averaging identity and Jensen inequality. Tonelli exchanges its
nonnegative expectations, and each original cross block has the proved
independent bilinear bound. This gives the universal quadratic MGF
estimate and genuine integrability on the original probability space.
The matrix norm condition uses the original matrix before removing
its diagonal. Both coefficient signs and singular matrices are covered.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal ENNReal

namespace Transformer.Homogenized

/-- The actual original quadratic polynomial is AE measurable whenever
its original vector is. Source: arXiv:2604.01978v1, §6.3,
the original random quadratic form in `lem:Hanson_wright`. -/
theorem aemeasurable_hansonWrightQuadraticForm {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (hZ : AEMeasurable Z P)
    (B : Matrix (Fin n) (Fin n) ℝ) : AEMeasurable (fun ω => hansonWrightQuadraticForm B (Z ω)) P := by
  simp_rw [hansonWrightQuadraticForm_eq_inner]
  exact hZ.inner ((Matrix.toEuclideanCLM (𝕜 := ℝ) B).continuous.measurable.comp_aemeasurable hZ)

/-- Jensen and Tonelli transfer genuine cross-block integral bounds
to the actual original off-diagonal exponential. Source:
arXiv:2604.01978v1, §6.3, the decoupling expectation exchange
for `lem:Hanson_wright`. The cross bounds are explicit inputs here. -/
theorem offDiagonal_lintegral_le_of_cross_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (hZ : AEMeasurable Z P)
    (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (K : ℝ≥0∞)
    (hcross : ∀ σ : Fin n → Bool, (∫⁻ ω, ENNReal.ofReal (Real.exp (4 * t * hansonWrightQuadraticForm
      (coordinateMask (randomCoordinateSubset σ) * B * coordinateMask (randomCoordinateSubset σ)ᶜ) (Z ω))) ∂P) ≤ K) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (t * hansonWrightQuadraticForm
      (B - Matrix.diagonal (fun i => B i i)) (Z ω))) ∂P) ≤ K := by
  let R := randomSplitLaw n
  let F : Ω → (Fin n → Bool) → ℝ≥0∞ := fun ω σ => ENNReal.ofReal (Real.exp (4 * t *
    hansonWrightQuadraticForm (coordinateMask (randomCoordinateSubset σ) * B *
      coordinateMask (randomCoordinateSubset σ)ᶜ) (Z ω)))
  have hfst : AEMeasurable (fun p : Ω × (Fin n → Bool) => Z p.1) (P.prod R) := by
    simpa only [Function.comp_def] using hZ.comp_quasiMeasurePreserving
      (measurePreserving_fst (μ := P) (ν := R)).quasiMeasurePreserving
  have hsnd : AEMeasurable (fun p : Ω × (Fin n → Bool) => p.2) (P.prod R) := measurable_snd.aemeasurable
  have hpair : AEMeasurable (fun p : Ω × (Fin n → Bool) => (p.2, Z p.1)) (P.prod R) := hsnd.prodMk hfst
  have hkernel : Measurable (fun p : (Fin n → Bool) × EucSpace n => ENNReal.ofReal (Real.exp (4 * t *
      hansonWrightQuadraticForm (coordinateMask (randomCoordinateSubset p.1) * B *
        coordinateMask (randomCoordinateSubset p.1)ᶜ) p.2))) := measurable_random_cross_exp B t
  have hfraw := hkernel.comp_aemeasurable hpair
  have hf : AEMeasurable (Function.uncurry F) (P.prod R) := by
    unfold Function.uncurry
    simpa only [Function.comp_def, F] using hfraw
  have hp : ∀ ω, ENNReal.ofReal (Real.exp (t * hansonWrightQuadraticForm
      (B - Matrix.diagonal (fun i => B i i)) (Z ω))) ≤ ∫⁻ σ, F ω σ ∂R :=
    fun ω => random_split_exp_lintegral_jensen B (Z ω) t
  calc
    _ ≤ ∫⁻ ω, ∫⁻ σ, F ω σ ∂R ∂P := lintegral_mono hp
    _ = ∫⁻ σ, ∫⁻ ω, F ω σ ∂P ∂R := lintegral_lintegral_swap hf
    _ ≤ ∫⁻ σ : Fin n → Bool, K ∂R := lintegral_mono hcross
    _ = _ := by simp only [lintegral_const, measure_univ, mul_one]

/-- The original off-diagonal exponential has its exact universal
quadratic bound in nonnegative integrals. Source: arXiv:2604.01978v1,
§6.3, the random-split off-diagonal estimate for `lem:Hanson_wright`. -/
theorem subgaussian_offDiagonal_lintegral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hsmall : 16 * t ^ 2 * (c : ℝ) ^ 2 * matOpNorm B ^ 2 ≤ 1 / 2) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (t * hansonWrightQuadraticForm
      (B - Matrix.diagonal (fun i => B i i)) (Z ω))) ∂P) ≤
      ENNReal.ofReal (Real.exp (64 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2)) := by
  apply offDiagonal_lintegral_le_of_cross_le P Z
    (aemeasurable_eucSpace_of_coordinates P Z (fun i => (hZ i).aemeasurable)) B t
  intro σ
  have hs : (4 * t) ^ 2 * (c : ℝ) ^ 2 * matOpNorm B ^ 2 ≤ 1 / 2 := by nlinarith [hsmall]
  have h := subgaussian_cross_mask_lintegral_le P Z c hindep hZ B (randomCoordinateSubset σ) (4 * t) hs
  have he : 4 * (4 * t) ^ 2 * (c : ℝ) ^ 2 * (∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) =
      64 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2 := by ring
  rw [he] at h
  exact h

/-- The actual off-diagonal exponential is genuinely integrable and
its original real expectation has the universal quadratic MGF bound.
Source: arXiv:2604.01978v1, §6.3, the off-diagonal contribution
to `lem:Hanson_wright`. -/
theorem subgaussian_offDiagonal_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hsmall : 16 * t ^ 2 * (c : ℝ) ^ 2 * matOpNorm B ^ 2 ≤ 1 / 2) :
    Integrable (fun ω => Real.exp (t * hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) (Z ω))) P ∧
    (∫ ω, Real.exp (t * hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) (Z ω)) ∂P) ≤
      Real.exp (64 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by
  have hb := subgaussian_offDiagonal_lintegral_le P Z c hindep hZ B t hsmall
  have hz := aemeasurable_eucSpace_of_coordinates P Z (fun i => (hZ i).aemeasurable)
  have hm : AEMeasurable (fun ω => Real.exp (t * hansonWrightQuadraticForm
      (B - Matrix.diagonal (fun i => B i i)) (Z ω))) P :=
    ((aemeasurable_hansonWrightQuadraticForm P Z hz _).const_mul t).exp
  have hn : 0 ≤ᵐ[P] (fun ω => Real.exp (t * hansonWrightQuadraticForm
      (B - Matrix.diagonal (fun i => B i i)) (Z ω))) := ae_of_all _ (fun ω => Real.exp_nonneg _)
  have hi : Integrable (fun ω => Real.exp (t * hansonWrightQuadraticForm
      (B - Matrix.diagonal (fun i => B i i)) (Z ω))) P :=
    ⟨hm.aestronglyMeasurable, (hasFiniteIntegral_iff_ofReal hn).mpr (hb.trans_lt ENNReal.ofReal_lt_top)⟩
  refine ⟨hi, ?_⟩
  have hh := ENNReal.toReal_mono ENNReal.ofReal_ne_top hb
  rw [← ofReal_integral_eq_lintegral_ofReal hi hn,
    ENNReal.toReal_ofReal (integral_nonneg (fun ω => Real.exp_nonneg _)), ENNReal.toReal_ofReal (Real.exp_nonneg _)] at hh
  exact hh

/-- The original off-diagonal quadratic form has actual expectation
zero. Source: arXiv:2604.01978v1, §6.3, the diagonal subtraction
in the proof of `lem:Hanson_wright`. -/
theorem integral_offDiagonal_quadratic_eq_zero {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) :
    (∫ ω, hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) (Z ω) ∂P) = 0 := by
  rw [integral_hansonWrightQuadraticForm P Z c hindep hZ]
  apply Finset.sum_eq_zero
  intro i _
  simp only [Matrix.sub_apply, Matrix.diagonal_apply_eq, sub_self, zero_mul]

/-- The actual off-diagonal estimate also holds on a fixed absolute
coefficient interval defined by the original operator norm. Source:
arXiv:2604.01978v1, §6.3, the coefficient interval used when
combining the two quadratic parts of `lem:Hanson_wright`. -/
theorem subgaussian_offDiagonal_exp_integral_of_abs_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hsmall : |t| * (c : ℝ) * matOpNorm B ≤ 1 / 8) :
    Integrable (fun ω => Real.exp (t * hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) (Z ω))) P ∧
    (∫ ω, Real.exp (t * hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) (Z ω)) ∂P) ≤
      Real.exp (64 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by
  have hn : 0 ≤ |t| * (c : ℝ) * matOpNorm B :=
    mul_nonneg (mul_nonneg (abs_nonneg t) c.coe_nonneg) (matOpNorm_nonneg B)
  have hs := (sq_le_sq₀ hn (by norm_num : (0 : ℝ) ≤ 1 / 8)).mpr hsmall
  simp only [mul_pow, sq_abs] at hs
  exact subgaussian_offDiagonal_exp_integral_le P Z c hindep hZ B t (by nlinarith [hs])

/-- Genuine independent standard coordinates, their original measurable
Euclidean vector and a positive coefficient jointly witness every
off-diagonal hypothesis. The dimension is nonempty and the matrix nonzero. -/
example : ∃ Z : (Fin 2 → ℝ) → EucSpace 2,
    iIndepFun (fun i : Fin 2 => fun z => Z z i) (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun z => Z z i) 1 (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    AEMeasurable Z (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    16 * (1 / 8 : ℝ) ^ 2 * ((1 : ℝ≥0) : ℝ) ^ 2 * matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) ^ 2 ≤ 1 / 2 ∧
    |(1 / 8 : ℝ)| * (1 : ℝ≥0) * matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) ≤ 1 / 8 := by
  refine ⟨WithLp.toLp 2, ?_, hasSubgaussianMGF_standardProduct_coordinate,
    (WithLp.measurable_toLp 2 (Fin 2 → ℝ)).aemeasurable, inferInstance, ?_⟩
  · simpa only [WithLp.ofLp_toLp, id_eq] using iIndepFun_pi
      (μ := fun _ : Fin 2 => gaussianReal 0 1) (X := fun _ => id) (fun i => measurable_id.aemeasurable)
  · have hn : matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) = 1 := by rw [← Matrix.diagonal_one, matOpNorm_diagonal]; simp
    rw [hn]
    norm_num

/-- A genuine probability space and finite cross bound witness the
general expectation-exchange assumptions for every split. -/
example : AEMeasurable (fun _ : Unit => (0 : EucSpace 2)) (Measure.dirac ()) ∧
    IsProbabilityMeasure (Measure.dirac ()) ∧
    (∀ σ : Fin 2 → Bool, (∫⁻ ω : Unit, ENNReal.ofReal (Real.exp (4 * (1 : ℝ) * hansonWrightQuadraticForm
      (coordinateMask (randomCoordinateSubset σ) * (1 : Matrix (Fin 2) (Fin 2) ℝ) *
        coordinateMask (randomCoordinateSubset σ)ᶜ) ((fun _ : Unit => (0 : EucSpace 2)) ω))) ∂Measure.dirac ()) ≤ 1) := by
  refine ⟨aemeasurable_const, inferInstance, ?_⟩
  intro σ
  simp [hansonWrightQuadraticForm]

end Transformer.Homogenized
