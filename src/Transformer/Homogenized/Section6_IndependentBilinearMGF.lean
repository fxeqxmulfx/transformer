module

public import Transformer.Homogenized.Section6_CoordinateBlocks

/-!
# Original independent bilinear-form exponential estimates

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
Independent original subgaussian vectors give a bilinear exponential
bound quadratic in its coefficient, with the exact original matrix
Frobenius square and Euclidean operator norm. Their genuine product
joint law justifies iterated nonnegative expectations. The inner
linear MGF and the outer square exponential have already been proved
from the original coordinate assumptions. Integrability is obtained
from the extended-real bound, including either coefficient sign.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal ENNReal

namespace Transformer.Homogenized

/-- The actual real Euclidean inner product is its original coordinate
linear form. Source: arXiv:2604.01978v1, §6.3, the conditional
linear form in `lem:Hanson_wright`. -/
theorem real_inner_eq_coordinate_sum {n : ℕ} (x v : EucSpace n) :
    inner ℝ x v = ∑ i : Fin n, v i * x i := by
  rw [EuclideanSpace.inner_eq_star_dotProduct]
  simp only [dotProduct, Pi.star_apply, star_trivial]

/-- The original inner-product exponential has its genuine subgaussian
extended-real integral bound. Source: arXiv:2604.01978v1, §6.3,
the conditional linear estimate for `lem:Hanson_wright`. -/
theorem subgaussian_inner_lintegral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => X ω i) P)
    (hX : ∀ i : Fin n, HasSubgaussianMGF (fun ω => X ω i) c P) (v : EucSpace n) (t : ℝ) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (t * inner ℝ (X ω) v)) ∂P) ≤
      ENNReal.ofReal (Real.exp ((c : ℝ) * ‖v‖ ^ 2 * t ^ 2 / 2)) := by
  have hi := integrable_exp_subgaussian_linearForm P X c hindep hX (fun i => v i) t
  have hb := ENNReal.ofReal_le_ofReal (subgaussian_linearForm_exp_integral_le P X c hindep hX (fun i => v i) t)
  rw [ofReal_integral_eq_lintegral_ofReal hi (ae_of_all _ (fun ω => Real.exp_nonneg _))] at hb
  simpa only [real_inner_eq_coordinate_sum, EuclideanSpace.real_norm_sq_eq] using hb

/-- Original independent subgaussian vectors have the exact matrix
bilinear exponential bound in nonnegative integrals. Source:
arXiv:2604.01978v1, §6.3, the fixed-split step for `lem:Hanson_wright`. -/
theorem independent_subgaussian_bilinear_lintegral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (X Y : Ω → EucSpace n) (c d : ℝ≥0)
    (hxy : IndepFun X Y P)
    (hxind : iIndepFun (fun i : Fin n => fun ω => X ω i) P)
    (hyind : iIndepFun (fun i : Fin n => fun ω => Y ω i) P)
    (hX : ∀ i : Fin n, HasSubgaussianMGF (fun ω => X ω i) c P)
    (hY : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Y ω i) d P)
    (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hsmall : t ^ 2 * (c : ℝ) * (d : ℝ) * matOpNorm B ^ 2 ≤ 1 / 2) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (t * inner ℝ (X ω) (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Y ω)))) ∂P) ≤
      ENNReal.ofReal (Real.exp (4 * t ^ 2 * (c : ℝ) * (d : ℝ) * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2)) := by
  let μ := P.map X
  let ν := P.map Y
  let T : EucSpace n →L[ℝ] EucSpace n := Matrix.toEuclideanCLM (𝕜 := ℝ) B
  let f : EucSpace n × EucSpace n → ℝ≥0∞ := fun p => ENNReal.ofReal (Real.exp (t * inner ℝ p.1 (T p.2)))
  have hx := aemeasurable_eucSpace_of_coordinates P X (fun i => (hX i).aemeasurable)
  have hy := aemeasurable_eucSpace_of_coordinates P Y (fun i => (hY i).aemeasurable)
  have : IsProbabilityMeasure μ := (Measure.isProbabilityMeasure_map_iff hx).mpr inferInstance
  have : IsProbabilityMeasure ν := (Measure.isProbabilityMeasure_map_iff hy).mpr inferInstance
  have hl : HasLaw (fun ω => (X ω, Y ω)) (μ.prod ν) P :=
    ⟨hx.prodMk hy, hxy.map_prod_eq_prod_map_map hx hy⟩
  have hm : Measurable f := by dsimp only [f]; fun_prop
  have hs : ((c : ℝ) * t ^ 2 / 2) * (d : ℝ) * matOpNorm B ^ 2 ≤ 1 / 4 := by nlinarith [hsmall]
  have houterb := subgaussian_matrix_norm_sq_lintegral_le P Y d hyind hY B
    ((c : ℝ) * t ^ 2 / 2) (by positivity) hs
  have hi : ∀ y : EucSpace n, (∫⁻ x : EucSpace n, f (x, y) ∂μ) ≤
      ENNReal.ofReal (Real.exp (((c : ℝ) * t ^ 2 / 2) * ‖T y‖ ^ 2)) := by
    intro y
    rw [lintegral_map' (by fun_prop) hx]
    have h := subgaussian_inner_lintegral_le P X c hxind hX (T y) t
    have he : (c : ℝ) * ‖T y‖ ^ 2 * t ^ 2 / 2 = ((c : ℝ) * t ^ 2 / 2) * ‖T y‖ ^ 2 := by ring
    simpa only [f, he] using h
  have hm' : AEMeasurable (Function.uncurry (fun y x : EucSpace n => f (x, y))) (ν.prod μ) := by
    apply Measurable.aemeasurable
    dsimp only [f]
    fun_prop
  calc
    _ = ∫⁻ p : EucSpace n × EucSpace n, f p ∂μ.prod ν := hl.lintegral_comp hm.aemeasurable
    _ = ∫⁻ y : EucSpace n, ∫⁻ x : EucSpace n, f (x, y) ∂μ ∂ν :=
      (lintegral_lintegral_swap hm').trans (lintegral_lintegral hm.aemeasurable) |>.symm
    _ ≤ ∫⁻ y : EucSpace n, ENNReal.ofReal (Real.exp (((c : ℝ) * t ^ 2 / 2) * ‖T y‖ ^ 2)) ∂ν :=
      lintegral_mono hi
    _ = ∫⁻ ω, ENNReal.ofReal (Real.exp (((c : ℝ) * t ^ 2 / 2) * ‖T (Y ω)‖ ^ 2)) ∂P :=
      lintegral_map' (by fun_prop) hy
    _ ≤ _ := by
      have he : 8 * ((c : ℝ) * t ^ 2 / 2) * (d : ℝ) * (∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) =
          4 * t ^ 2 * (c : ℝ) * (d : ℝ) * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2 := by ring
      simpa only [T, he] using houterb

/-- The same original bilinear exponential is genuinely integrable,
and its real expectation has the exact fixed-split bound. Source:
arXiv:2604.01978v1, §6.3, the off-diagonal MGF step for `lem:Hanson_wright`. -/
theorem independent_subgaussian_bilinear_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (X Y : Ω → EucSpace n) (c d : ℝ≥0)
    (hxy : IndepFun X Y P)
    (hxind : iIndepFun (fun i : Fin n => fun ω => X ω i) P)
    (hyind : iIndepFun (fun i : Fin n => fun ω => Y ω i) P)
    (hX : ∀ i : Fin n, HasSubgaussianMGF (fun ω => X ω i) c P)
    (hY : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Y ω i) d P)
    (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hsmall : t ^ 2 * (c : ℝ) * (d : ℝ) * matOpNorm B ^ 2 ≤ 1 / 2) :
    Integrable (fun ω => Real.exp (t * inner ℝ (X ω) (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Y ω)))) P ∧
    (∫ ω, Real.exp (t * inner ℝ (X ω) (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Y ω))) ∂P) ≤
      Real.exp (4 * t ^ 2 * (c : ℝ) * (d : ℝ) * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by
  have hb := independent_subgaussian_bilinear_lintegral_le P X Y c d hxy hxind hyind hX hY B t hsmall
  have hx := aemeasurable_eucSpace_of_coordinates P X (fun i => (hX i).aemeasurable)
  have hy := aemeasurable_eucSpace_of_coordinates P Y (fun i => (hY i).aemeasurable)
  have hm : AEMeasurable (fun ω => Real.exp (t * inner ℝ (X ω) (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Y ω)))) P :=
    (aemeasurable_const.fun_mul (hx.inner ((Matrix.toEuclideanCLM (𝕜 := ℝ) B).continuous.measurable.comp_aemeasurable hy))).exp
  have hn : 0 ≤ᵐ[P] (fun ω => Real.exp (t * inner ℝ (X ω) (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Y ω)))) :=
    ae_of_all _ (fun ω => Real.exp_nonneg _)
  have hi : Integrable (fun ω => Real.exp (t * inner ℝ (X ω) (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Y ω)))) P :=
    ⟨hm.aestronglyMeasurable, (hasFiniteIntegral_iff_ofReal hn).mpr (hb.trans_lt ENNReal.ofReal_lt_top)⟩
  refine ⟨hi, ?_⟩
  have hh := ENNReal.toReal_mono ENNReal.ofReal_ne_top hb
  rw [← ofReal_integral_eq_lintegral_ofReal hi hn,
    ENNReal.toReal_ofReal (integral_nonneg (fun ω => Real.exp_nonneg _)), ENNReal.toReal_ofReal (Real.exp_nonneg _)] at hh
  exact hh

/-- Genuine complementary Gaussian coordinate blocks witness the
independent-vector and coordinate hypotheses with a positive coefficient. -/
example : ∃ X Y : (Fin 2 → ℝ) → EucSpace 2,
    IndepFun X Y (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    iIndepFun (fun i : Fin 2 => fun ω => X ω i) (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    iIndepFun (fun i : Fin 2 => fun ω => Y ω i) (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun ω => X ω i) 1 (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun ω => Y ω i) 1 (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (1 / 8 : ℝ) ^ 2 * (1 : ℝ≥0) * (1 : ℝ≥0) * matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) ^ 2 ≤ 1 / 2 := by
  let P := Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)
  let Z : (Fin 2 → ℝ) → EucSpace 2 := WithLp.toLp 2
  let s : Finset (Fin 2) := {0}
  have hsg : ∀ i : Fin 2, HasSubgaussianMGF (fun ω => Z ω i) 1 P := hasSubgaussianMGF_standardProduct_coordinate
  have hind : iIndepFun (fun i : Fin 2 => fun ω => Z ω i) P := by
    simpa only [Z, WithLp.ofLp_toLp, id_eq] using iIndepFun_pi
      (μ := fun _ : Fin 2 => gaussianReal 0 1) (X := fun _ => id) (fun i => measurable_id.aemeasurable)
  refine ⟨fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) (Z ω),
    fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask sᶜ) (Z ω),
    indepFun_coordinateMasks_compl P Z hind (fun i => (hsg i).aemeasurable) s,
    iIndepFun_coordinateMask P Z hind s, iIndepFun_coordinateMask P Z hind sᶜ,
    hasSubgaussianMGF_coordinateMask P Z 1 hsg s, hasSubgaussianMGF_coordinateMask P Z 1 hsg sᶜ,
    inferInstance, ?_⟩
  have hn : matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) = 1 := by rw [← Matrix.diagonal_one, matOpNorm_diagonal]; simp
  rw [hn]
  norm_num

end Transformer.Homogenized
