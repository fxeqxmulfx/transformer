module

public import Transformer.Homogenized.Section7_GaussianInterpolationContinuity
public import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# The actual Gaussian interpolation path, derivative and local bound

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The actual original path is the actual
square-root blend of its two original vectors. Its interior derivative
is the original vector divided by twice its square-root coefficient,
minus the other original vector divided by its corresponding coefficient.

On a fixed neighborhood of an original interior time, both square-root
denominators stay uniformly positive. The actual path derivative is
therefore bounded by an explicit fixed linear combination of the two
original vector norms. The original test derivative bound gives the
corresponding bound on its actual scalar path derivative.
Gaussian first moments will make that original bound integrable,
justifying differentiation of the original expectation.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Set Filter
open scoped Topology

namespace Transformer.Homogenized

/-- The actual original square-root blend at the original time.
Source: arXiv:2604.01978v1, §7, `eq:interpolation_gaussian`. -/
noncomputable def gaussianInterpolationPoint {m : ℕ} (t : ℝ) (x y : EucSpace m) : EucSpace m :=
  Real.sqrt t • x + Real.sqrt (1 - t) • y

/-- The actual interior derivative of the original square-root blend.
Source: arXiv:2604.01978v1, §7, the original path derivative
in the integration-by-parts proof of `lem:gaussian_interp`. -/
noncomputable def gaussianInterpolationVelocity {m : ℕ} (t : ℝ) (x y : EucSpace m) : EucSpace m :=
  (1 / (2 * Real.sqrt t)) • x - (1 / (2 * Real.sqrt (1 - t))) • y

/-- The actual original blend is continuous at every original time,
including both endpoints. Source: arXiv:2604.01978v1, §7,
the original square-root path in `eq:interpolation_gaussian`. -/
theorem continuous_gaussianInterpolationPoint {m : ℕ} (x y : EucSpace m) :
    Continuous (fun t : ℝ => gaussianInterpolationPoint t x y) := by
  unfold gaussianInterpolationPoint
  fun_prop

/-- The actual derivative candidate is the derivative of the actual
original path at every interior time. Source: arXiv:2604.01978v1,
§7, the original path in `lem:gaussian_interp`. -/
theorem hasDerivAt_gaussianInterpolationPoint {m : ℕ} (x y : EucSpace m)
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) :
    HasDerivAt (fun r => gaussianInterpolationPoint r x y) (gaussianInterpolationVelocity t x y) t := by
  have hx := (Real.hasDerivAt_sqrt (ne_of_gt ht.1)).smul_const x
  have hy := (((hasDerivAt_id t).const_sub 1).sqrt (by
    change 1 - t ≠ 0
    linarith [ht.2])).smul_const y
  have h := hx.add hy
  change HasDerivAt (fun r : ℝ => Real.sqrt r • x + Real.sqrt (1 - r) • y) _ t at h
  simpa only [gaussianInterpolationPoint, gaussianInterpolationVelocity, id_eq, Pi.add_apply,
    neg_div, neg_smul, sub_eq_add_neg] using h

/-- The actual original scalar path derivative is the original
Fréchet derivative evaluated on its actual original velocity.
Source: arXiv:2604.01978v1, §7, differentiation before the
Gaussian integration-by-parts step in `lem:gaussian_interp`. -/
theorem hasDerivAt_gaussianInterpolation_test {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (x y : EucSpace m) (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) :
    HasDerivAt (fun r => φ (gaussianInterpolationPoint r x y))
      (fderiv ℝ φ (gaussianInterpolationPoint t x y) (gaussianInterpolationVelocity t x y)) t :=
  ((hφ.differentiable (by norm_num)) _).hasFDerivAt.comp_hasDerivAt t
    (hasDerivAt_gaussianInterpolationPoint x y t ht)

/-- The actual velocity is bounded by the genuine original vector
norms with its actual scalar coefficients.
Source: arXiv:2604.01978v1, §7, the original first-moment
domination in the proof of `lem:gaussian_interp`. -/
theorem norm_gaussianInterpolationVelocity_le {m : ℕ} (t : ℝ) (x y : EucSpace m) :
    ‖gaussianInterpolationVelocity t x y‖ ≤
      (1 / (2 * Real.sqrt t)) * ‖x‖ + (1 / (2 * Real.sqrt (1 - t))) * ‖y‖ := by
  have hx : 0 ≤ 1 / (2 * Real.sqrt t) := by positivity
  have hy : 0 ≤ 1 / (2 * Real.sqrt (1 - t)) := by positivity
  simpa only [gaussianInterpolationVelocity, norm_smul, Real.norm_eq_abs, abs_of_nonneg hx,
    abs_of_nonneg hy] using norm_sub_le ((1 / (2 * Real.sqrt t)) • x)
      ((1 / (2 * Real.sqrt (1 - t))) • y)

/-- The actual square-root derivative coefficient decreases when its
positive original argument increases. Source: arXiv:2604.01978v1,
§7, the fixed local derivative bound in `lem:gaussian_interp`. -/
theorem gaussian_sqrt_velocity_coefficient_le (u v : ℝ) (hu : 0 < u) (huv : u ≤ v) :
    1 / (2 * Real.sqrt v) ≤ 1 / (2 * Real.sqrt u) := by
  apply one_div_le_one_div_of_le
  · exact mul_pos (by norm_num) (Real.sqrt_pos.mpr hu)
  · exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt huv) (by norm_num)

/-- The actual velocity has an explicit uniform local bound in the
original interior neighborhood. Source: arXiv:2604.01978v1, §7,
the domination needed to differentiate `lem:gaussian_interp`. -/
theorem norm_gaussianInterpolationVelocity_local_le {m : ℕ}
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) (r : ℝ) (hr : r ∈ Ioo (t / 2) ((t + 1) / 2))
    (x y : EucSpace m) : ‖gaussianInterpolationVelocity r x y‖ ≤
      (1 / (2 * Real.sqrt (t / 2))) * ‖x‖ +
        (1 / (2 * Real.sqrt ((1 - t) / 2))) * ‖y‖ := by
  apply (norm_gaussianInterpolationVelocity_le r x y).trans
  apply add_le_add
  · exact mul_le_mul_of_nonneg_right
      (gaussian_sqrt_velocity_coefficient_le (t / 2) r (by linarith [ht.1]) hr.1.le) (norm_nonneg x)
  · exact mul_le_mul_of_nonneg_right (gaussian_sqrt_velocity_coefficient_le ((1 - t) / 2) (1 - r)
      (by linarith [ht.2]) (by linarith [hr.2])) (norm_nonneg y)

/-- The actual original scalar path derivative is bounded by the
same explicit original first-moment dominator.
Source: arXiv:2604.01978v1, §7, differentiation under the
integral in the corrected bounded-test `lem:gaussian_interp`. -/
theorem norm_gaussianInterpolation_test_derivative_local_le {m : ℕ}
    (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) (r : ℝ) (hr : r ∈ Ioo (t / 2) ((t + 1) / 2))
    (x y : EucSpace m) :
    ‖fderiv ℝ φ (gaussianInterpolationPoint r x y) (gaussianInterpolationVelocity r x y)‖ ≤
      C * ((1 / (2 * Real.sqrt (t / 2))) * ‖x‖ +
        (1 / (2 * Real.sqrt ((1 - t) / 2))) * ‖y‖) := by
  calc
    _ ≤ ‖fderiv ℝ φ (gaussianInterpolationPoint r x y)‖ * ‖gaussianInterpolationVelocity r x y‖ :=
      (fderiv ℝ φ _).le_opNorm _
    _ ≤ C * ‖gaussianInterpolationVelocity r x y‖ := mul_le_mul_of_nonneg_right
      (norm_gaussian_test_fderiv_le φ C hbdd _) (norm_nonneg _)
    _ ≤ _ := mul_le_mul_of_nonneg_left
      (norm_gaussianInterpolationVelocity_local_le t ht r hr x y) (gaussian_test_bound_nonneg φ C hbdd)

/-- The actual original scalar path derivative is measurable on the
original probability space. Source: arXiv:2604.01978v1, §7,
the original derivative expectation in `lem:gaussian_interp`. -/
theorem aestronglyMeasurable_gaussianInterpolation_test_derivative
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω) (X Y : Ω → EucSpace m)
    (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (t : ℝ) :
    AEStronglyMeasurable (fun ω => fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
      (gaussianInterpolationVelocity t (X ω) (Y ω))) P := by
  have hp : Continuous (fun p : EucSpace m × EucSpace m => gaussianInterpolationPoint t p.1 p.2) := by
    unfold gaussianInterpolationPoint; fun_prop
  have hv : Continuous (fun p : EucSpace m × EucSpace m => gaussianInterpolationVelocity t p.1 p.2) := by
    unfold gaussianInterpolationVelocity; fun_prop
  have hD : Continuous (fderiv ℝ φ) := hφ.continuous_fderiv (by norm_num)
  have htest : Continuous (fun p : EucSpace m × EucSpace m =>
      fderiv ℝ φ (gaussianInterpolationPoint t p.1 p.2) (gaussianInterpolationVelocity t p.1 p.2)) :=
    (hD.comp hp).clm_apply hv
  have hpair : AEMeasurable (fun ω => (X ω, Y ω)) P := hX.prodMk hY
  have hm : AEStronglyMeasurable (fun p : EucSpace m × EucSpace m =>
      fderiv ℝ φ (gaussianInterpolationPoint t p.1 p.2) (gaussianInterpolationVelocity t p.1 p.2))
      (P.map (fun ω => (X ω, Y ω))) := htest.aestronglyMeasurable
  simpa only [Function.comp_def] using hm.comp_aemeasurable hpair

/-- The genuine nonconstant bounded `C²` sine and a genuine interior
time and local time jointly witness every bound and regularity premise. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ), ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧
    (1 / 2 : ℝ) ∈ Ioo ((1 / 2 : ℝ) / 2) (((1 / 2 : ℝ) + 1) / 2) ∧
    AEMeasurable (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    AEMeasurable (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧ fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  obtain ⟨φ, C, hφ, hbdd, hn⟩ := gaussian_interp_test_witness
  exact ⟨φ, C, hφ, hbdd, by norm_num, by norm_num, measurable_id.aemeasurable,
    measurable_const.aemeasurable, hn⟩

end Transformer.Homogenized
