module

public import Transformer.Homogenized.Section7_GaussianPolynomialStein
public import Transformer.Homogenized.Section7_GaussianVectorStein

/-!
# Polynomial bounds on the original Gaussian variables

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and the justification
after `eq:interpolation_interm`. The derivatives of the original bilinear
softmax test have polynomial growth. Gaussian moments make these genuine
derivatives integrable, including along affine coordinate slices.

The convenient growth bound here is `C (1 + ‖z‖)^p`. Composition with
an actual affine linear map preserves this bound with an explicit constant.
Multiplication by an original coordinate increases the degree by one.
Gaussian moments then give integrability on the original probability space.
The same moment bound applies to Gaussian pairs in their actual product norm.
The scalar Stein identity proved directly from the density is transferred
to the actual Gaussian random variable, retaining its original variance.
No integration-by-parts or moment identity is assumed in these transfers.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- An actual affine image has at most linear norm growth.
Source: arXiv:2604.01978v1, §7, the coordinate slices and linear
Gaussian representations in the proof of `lem:gaussian_interp`. -/
theorem one_add_norm_gaussian_affine_le {E F : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    (L : E →L[ℝ] F) (b : F) (z : E) :
    1 + ‖L z + b‖ ≤ (1 + ‖L‖ + ‖b‖) * (1 + ‖z‖) := by
  have h := (norm_add_le (L z) b).trans (add_le_add (L.le_opNorm z) le_rfl)
  nlinarith [norm_nonneg L, norm_nonneg b, norm_nonneg z,
    mul_nonneg (norm_nonneg b) (norm_nonneg z)]

/-- A polynomial bound on an actual test persists along each actual
affine Gaussian slice. Source: arXiv:2604.01978v1, §7,
polynomial domination after `eq:interpolation_interm`. -/
theorem norm_polynomial_gaussian_affine_le {E F : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    (L : E →L[ℝ] F) (b : F)
    (g : F → ℝ) (C : ℝ) (hC : 0 ≤ C) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p) (z : E) :
    ‖g (L z + b)‖ ≤ (C * (1 + ‖L‖ + ‖b‖) ^ p) * (1 + ‖z‖) ^ p := by
  calc
    _ ≤ C * (1 + ‖L z + b‖) ^ p := hbound _
    _ ≤ C * ((1 + ‖L‖ + ‖b‖) * (1 + ‖z‖)) ^ p :=
      mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (by positivity)
        (one_add_norm_gaussian_affine_le L b z) p) hC
    _ = _ := by rw [mul_pow]; ring

/-- Weighting the actual test by an actual coordinate raises its
polynomial degree by one. Source: arXiv:2604.01978v1, §7,
the coordinate factor in the original Gaussian integration by parts. -/
theorem norm_coordinate_mul_polynomial_le {m : ℕ}
    (g : EucSpace m → ℝ) (C : ℝ) (hC : 0 ≤ C) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p) (i : Fin m) (z : EucSpace m) :
    ‖z i * g z‖ ≤ C * (1 + ‖z‖) ^ (p + 1) := by
  rw [norm_mul]
  calc
    _ ≤ ‖z‖ * (C * (1 + ‖z‖) ^ p) :=
      mul_le_mul (PiLp.norm_apply_le z i) (hbound z) (norm_nonneg _) (norm_nonneg z)
    _ ≤ (1 + ‖z‖) * (C * (1 + ‖z‖) ^ p) :=
      mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    _ = _ := by rw [pow_succ]; ring

/-- The actual Gaussian vector has finite moments of the convenient
polynomial envelope. Source: arXiv:2604.01978v1, §7, the Gaussian
moment justification following `eq:interpolation_interm`. -/
theorem integrable_gaussian_polynomial_envelope {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [MeasurableSpace E] [BorelSpace E]
    [CompleteSpace E] [SecondCountableTopology E]
    (P : Measure Ω) (X : Ω → E) (hX : HasGaussianLaw X P) (p : ℕ) :
    Integrable (fun ω => (1 + ‖X ω‖) ^ p) P := by
  have : IsProbabilityMeasure P := hX.isProbabilityMeasure
  have hXmom : MemLp X (p : ENNReal) P := hX.memLp (by simp)
  have h : MemLp (fun ω => 1 + ‖X ω‖) (p : ENNReal) P :=
    (memLp_const (μ := P) (1 : ℝ)).add hXmom.norm
  apply h.integrable_norm_pow'.congr
  apply ae_of_all
  intro ω
  change ‖1 + ‖X ω‖‖ ^ p = (1 + ‖X ω‖) ^ p
  rw [Real.norm_of_nonneg (by positivity)]

/-- Any actual measurable quantity dominated by a polynomial in the
original Gaussian vector has a finite expectation. Source:
arXiv:2604.01978v1, §7, domination in `lem:gaussian_interp`.
This also applies to the original coordinate derivatives and products. -/
theorem integrable_of_gaussian_polynomial_bound {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [MeasurableSpace E] [BorelSpace E]
    [CompleteSpace E] [SecondCountableTopology E]
    (P : Measure Ω) (X : Ω → E) (hX : HasGaussianLaw X P)
    (f : Ω → ℝ) (hf : AEStronglyMeasurable f P) (C : ℝ) (p : ℕ)
    (hbound : ∀ᵐ ω ∂P, ‖f ω‖ ≤ C * (1 + ‖X ω‖) ^ p) : Integrable f P :=
  ((integrable_gaussian_polynomial_envelope P X hX p).const_mul C).mono' hf hbound

/-- Scalar Gaussian integration by parts holds for polynomially growing
tests on the actual original probability space. Source:
arXiv:2604.01978v1, §7, `lem:gaussian_interp`; the derivative is the
actual derivative of the original test, and both integrals use its actual law. -/
theorem gaussian_stein_random_variable_polynomial {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X : Ω → ℝ) (hX : HasGaussianLaw X P)
    (hcenter : ∫ ω, X ω ∂P = 0) (g g' : ℝ → ℝ)
    (hg : Continuous g) (hg' : Continuous g') (hderiv : ∀ x, HasDerivAt g (g' x) x)
    (C C' : ℝ) (p : ℕ) (hbound : ∀ x, ‖g x‖ ≤ C * (1 + |x|) ^ p)
    (hbound' : ∀ x, ‖g' x‖ ≤ C' * (1 + |x|) ^ p) :
    (∫ ω, X ω * g (X ω) ∂P) = variance X P * ∫ ω, g' (X ω) ∂P := by
  have hC : 0 ≤ C := (norm_nonneg (g 0)).trans (by simpa using hbound 0)
  have hC' : 0 ≤ C' := (norm_nonneg (g' 0)).trans (by simpa using hbound' 0)
  have hp (x : ℝ) : (1 + |x|) ^ p ≤ 2 ^ (p - 1) * (1 + |x| ^ p) := by
    simpa only [one_pow] using add_pow_le (by norm_num : (0 : ℝ) ≤ 1) (abs_nonneg x) p
  have hmap := hX.map_eq_gaussianReal
  rw [hcenter] at hmap
  calc
    _ = ∫ x, x * g x ∂P.map X :=
      (integral_map hX.aemeasurable (continuous_id.mul hg).aestronglyMeasurable).symm
    _ = (variance X P).toNNReal * ∫ x, g' x ∂P.map X := by
      rw [hmap]
      apply centered_gaussian_stein_polynomial _ g g' hg hg' hderiv
        (C * 2 ^ (p - 1)) (C' * 2 ^ (p - 1)) p
      · intro x
        exact (hbound x).trans (by simpa only [mul_assoc] using mul_le_mul_of_nonneg_left (hp x) hC)
      · intro x
        exact (hbound' x).trans (by simpa only [mul_assoc] using mul_le_mul_of_nonneg_left (hp x) hC')
    _ = _ := by
      rw [Real.coe_toNNReal (variance X P) (variance_nonneg X P),
        integral_map hX.aemeasurable hg'.aestronglyMeasurable]

/-- A genuine two-dimensional Gaussian and an unbounded coordinate
test witness the polynomial moment and affine growth hypotheses. -/
example : ∃ g : EucSpace 2 → ℝ, ContDiff ℝ 1 g ∧ (0 : ℝ) ≤ 1 ∧
    (∀ z, ‖g z‖ ≤ 1 * (1 + ‖z‖) ^ 1) ∧
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    AEStronglyMeasurable g (stdGaussian (EucSpace 2)) := by
  refine ⟨EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2), ContinuousLinearMap.contDiff _,
    zero_le_one, ?_, IsGaussian.hasGaussianLaw_id,
    (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)).continuous.aestronglyMeasurable⟩
  intro z
  change ‖z (0 : Fin 2)‖ ≤ 1 * (1 + ‖z‖) ^ 1
  simpa only [one_mul, pow_one] using (PiLp.norm_apply_le z (0 : Fin 2)).trans
    (le_add_of_nonneg_left zero_le_one)

/-- The nonconstant scalar test and its actual derivative satisfy the
original polynomial Stein hypotheses under a positive-variance law. -/
example : HasGaussianLaw (id : ℝ → ℝ) (gaussianReal 0 1) ∧
    (∫ x : ℝ, id x ∂gaussianReal 0 1) = 0 ∧
    Continuous (id : ℝ → ℝ) ∧ Continuous (fun _ : ℝ => (1 : ℝ)) ∧
    (∀ x : ℝ, HasDerivAt (id : ℝ → ℝ) 1 x) ∧
    (∀ x : ℝ, ‖id x‖ ≤ 1 * (1 + |x|) ^ 1) ∧
    (∀ x : ℝ, ‖(1 : ℝ)‖ ≤ 1 * (1 + |x|) ^ 1) := by
  exact ⟨IsGaussian.hasGaussianLaw_id, integral_id_gaussianReal, continuous_id, continuous_const,
    hasDerivAt_id, fun x => by simp [Real.norm_eq_abs], fun x => by simp [abs_nonneg x]⟩

end Transformer.Homogenized
