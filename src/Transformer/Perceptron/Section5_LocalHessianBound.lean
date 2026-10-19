module

public import Transformer.Perceptron.Section5_ExpandedHessian
public import Transformer.Perceptron.Section5_ZeroMeanDirections

/-!
# Bounding the Hessian along a localized direction with zero mean

ArXiv:2601.21366v2, §5.2, the final estimate in the proof of
`thm: circle.gelu`. A zero measure mean cancels any constant
added to the pair kernel. If the kernel oscillates by at most
`ε` on the support of the direction, its mixed integral is at
most `ε` times the squared `L²` norm, by probability Cauchy--Schwarz.

The same local upper bound for the first-variation curvature
then bounds the expanded Hessian by twice that amount. The
estimates below use explicit local bounds as inputs; the support
argument must produce them from continuity at an accumulation point.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- A constant added to the actual mixed pair integral cancels for zero
measure mean. Source: arXiv:2601.21366v2, §5.2, subtracting `K_β''(0)`
in the proof of `thm: circle.gelu`. No symmetry is needed for this step. -/
theorem mixed_integral_sub_const_of_mean_zero (μ : Perspective.ProbSphere 2)
    (K : SSphere 2 × SSphere 2 → ℝ) (hK : Continuous K)
    (s : SSphere 2 → ℝ) (hs : Continuous s)
    (hmean : (∫ x, s x ∂(μ : Measure (SSphere 2))) = 0) (c : ℝ) :
    (∫ p, K p * s p.1 * s p.2
      ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) =
      ∫ p, (K p - c) * s p.1 * s p.2
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)) := by
  have hI : Integrable (fun p => K p * s p.1 * s p.2)
      ((μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) :=
    ((hK.mul (hs.comp continuous_fst)).mul (hs.comp continuous_snd)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hP : Integrable (fun p : SSphere 2 × SSphere 2 => s p.1 * s p.2)
      ((μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) :=
    ((hs.comp continuous_fst).mul (hs.comp continuous_snd)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have heq : (fun p : SSphere 2 × SSphere 2 => (K p - c) * s p.1 * s p.2) =
      fun p => K p * s p.1 * s p.2 - c * (s p.1 * s p.2) := by
    funext p
    ring
  rw [heq, integral_sub hI (hP.const_mul c), integral_const_mul, integral_prod_mul,
    hmean, zero_mul, mul_zero, sub_zero]

/-- A small pair kernel on the direction's actual support gives a small
mixed integral. Source: arXiv:2601.21366v2, §5.2, the kernel-modulus and
Cauchy--Schwarz estimate in the proof of `thm: circle.gelu`. -/
theorem abs_local_mixed_integral_le (μ : Perspective.ProbSphere 2)
    (K : SSphere 2 × SSphere 2 → ℝ) (hK : Continuous K)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (U : Set (SSphere 2))
    (hsupp : Function.support s ⊆ U) (ε : ℝ) (hε : 0 ≤ ε)
    (hbound : ∀ x ∈ U, ∀ y ∈ U, |K (x, y)| ≤ ε) :
    |∫ p, K p * s p.1 * s p.2
      ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))| ≤
      ε * ∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2)) := by
  have hI : Integrable (fun p => K p * s p.1 * s p.2)
      ((μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) :=
    ((hK.mul (hs.comp continuous_fst)).mul (hs.comp continuous_snd)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hR : Integrable (fun p : SSphere 2 × SSphere 2 => ε * |s p.1| * |s p.2|)
      ((μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) :=
    (((hs.comp continuous_fst).abs.const_mul ε).mul
      (hs.comp continuous_snd).abs).integrable_of_hasCompactSupport
        (HasCompactSupport.of_compactSpace _)
  have hp (p : SSphere 2 × SSphere 2) :
      |K p * s p.1 * s p.2| ≤ ε * |s p.1| * |s p.2| := by
    by_cases hx : s p.1 = 0
    · simp only [hx, mul_zero, zero_mul, abs_zero, le_refl]
    by_cases hy : s p.2 = 0
    · simp only [hy, mul_zero, abs_zero, le_refl]
    have hxy : |K p| ≤ ε := hbound p.1 (hsupp (Function.mem_support.mpr hx))
      p.2 (hsupp (Function.mem_support.mpr hy))
    rw [abs_mul, abs_mul]
    exact mul_le_mul_of_nonneg_right
      (mul_le_mul_of_nonneg_right hxy (abs_nonneg _)) (abs_nonneg _)
  calc
    _ = ‖∫ p, K p * s p.1 * s p.2
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))‖ := (Real.norm_eq_abs _).symm
    _ ≤ ∫ p, ‖K p * s p.1 * s p.2‖
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)) := norm_integral_le_integral_norm _
    _ = ∫ p, |K p * s p.1 * s p.2|
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)) := by
      simp only [Real.norm_eq_abs]
    _ ≤ ∫ p, ε * |s p.1| * |s p.2|
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2)) := integral_mono hI.abs hR hp
    _ = ε * (∫ x, |s x| ∂(μ : Measure (SSphere 2))) ^ 2 := by
      simp_rw [mul_assoc]
      rw [integral_const_mul,
        integral_prod_mul (fun x : SSphere 2 => |s x|) (fun x : SSphere 2 => |s x|)]
      ring
    _ ≤ ε * ∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2)) :=
      mul_le_mul_of_nonneg_left (sq_integral_abs_le_integral_sq μ s hs) hε

/-- Small local first-variation curvature and kernel oscillation bound
the expanded Hessian by `2ε‖s‖²`. Source: arXiv:2601.21366v2, §5.2,
the final contradiction in the proof of `thm: circle.gelu`.
The curvature term is retained, so global analytic cancellation is
not assumed by this estimate. Its actual local bound must be supplied. -/
theorem local_expanded_hessian_le (μ : Perspective.ProbSphere 2)
    (A : SSphere 2 → ℝ) (hA : Continuous A)
    (K : SSphere 2 × SSphere 2 → ℝ) (hK : Continuous K)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (U : Set (SSphere 2))
    (hsupp : Function.support s ⊆ U)
    (hmean : (∫ x, s x ∂(μ : Measure (SSphere 2))) = 0)
    (c ε : ℝ) (hε : 0 ≤ ε) (hlocal : ∀ x ∈ U, A x ≤ ε)
    (hosc : ∀ x ∈ U, ∀ y ∈ U, |K (x, y) - c| ≤ ε) :
    (∫ x, A x * s x ^ 2 ∂(μ : Measure (SSphere 2))) -
      (∫ p, K p * s p.1 * s p.2
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) ≤
      2 * ε * ∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2)) := by
  have hIA : Integrable (fun x => A x * s x ^ 2) (μ : Measure (SSphere 2)) :=
    (hA.mul (hs.pow 2)).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hIsq : Integrable (fun x => s x ^ 2) (μ : Measure (SSphere 2)) :=
    (hs.pow 2).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hpoint (x : SSphere 2) : A x * s x ^ 2 ≤ ε * s x ^ 2 := by
    by_cases hx : s x = 0
    · simp only [hx, zero_pow (by norm_num : 2 ≠ 0), mul_zero, le_refl]
    exact mul_le_mul_of_nonneg_right (hlocal x (hsupp (Function.mem_support.mpr hx))) (sq_nonneg _)
  have hlocalI := integral_mono hIA (hIsq.const_mul ε) hpoint
  rw [integral_const_mul] at hlocalI
  have hmix := abs_local_mixed_integral_le μ (fun p => K p - c) (hK.sub continuous_const)
    s hs U hsupp ε hε hosc
  have hlow := (abs_le.mp hmix).1
  rw [← mixed_integral_sub_const_of_mean_zero μ K hK s hs hmean c] at hlow
  linarith

/-- All continuity, support, mean, nonnegative-bound, local-curvature and
oscillation inputs are jointly realized by zero coefficients and zero
speed on an actual Dirac probability, in the full nonempty circle. -/
example : Continuous (0 : SSphere 2 → ℝ) ∧
    Continuous (0 : SSphere 2 × SSphere 2 → ℝ) ∧
    Function.support (0 : SSphere 2 → ℝ) ⊆ (univ : Set (SSphere 2)) ∧
    (∫ x, (0 : SSphere 2 → ℝ) x
      ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2))) = 0 ∧
    (0 : ℝ) ≤ 1 ∧ (∀ x ∈ (univ : Set (SSphere 2)), (0 : SSphere 2 → ℝ) x ≤ 1) ∧
    (∀ x ∈ (univ : Set (SSphere 2)), ∀ y ∈ (univ : Set (SSphere 2)),
      |(0 : SSphere 2 × SSphere 2 → ℝ) (x, y) - 0| ≤ 1) := by
  refine ⟨continuous_const, continuous_const, subset_univ _, by simp,
    zero_le_one, fun _ _ => zero_le_one, fun _ _ _ _ => by norm_num⟩

/-- Zero mean is also realized with nonzero norm: the first coordinate
has mean zero and squared norm one on the actual equally weighted
antipodal pair. Thus the local estimates admit nonzero directions. -/
example :
    (∫ x, (x : EucSpace 2) 0
      ∂(twoAtomProb (circlePoint 0) (circlePoint π) : Measure (SSphere 2))) = 0 ∧
    (∫ x, ((x : EucSpace 2) 0) ^ 2
      ∂(twoAtomProb (circlePoint 0) (circlePoint π) : Measure (SSphere 2))) = 1 := by
  constructor <;> rw [integral_twoAtomProb] <;>
    norm_num only [smul_eq_mul, circlePoint_coord_zero, Real.cos_zero, Real.cos_pi]

end Transformer.Perceptron
