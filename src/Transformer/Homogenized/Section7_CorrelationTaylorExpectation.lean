/-
# Integrating the actual correlation remainder

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, expectation of the
Taylor expansion of normalized query correlation. The pointwise
global cubic remainder proved previously is integrated here with
all required measurability and integrability conditions explicit.

For actual measurable query vectors `p,q`, the error between expected
normalized inner product and expected genuine quadratic polynomial
is at most forty times the actual third moment of the sum of their
three absolute deviations. The actual polynomial and actual error
are both proved integrable from that moment condition.

A scalar Hölder estimate then controls a third absolute moment by
the square root of the product of its second and fourth moments.
This provides the quantitative route to the source's remainder rate;
the Gaussian deviation moment estimates themselves must still be
established for the original product initialization. No covariance
identity or probabilistic remainder is included as an assumption.
-/

import Transformer.Homogenized.Section7_GlobalCorrelationTaylor
import Transformer.Homogenized.Section7_GaussianRowFourth
import Transformer.Homogenized.Basic

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The original normalization is bounded even at the origin.
Source: arXiv:2604.01978v1, §7.3, normalized queries in `lem:delta_method`. -/
theorem norm_normalizeLayer_le_one {d : ℕ} (p : EucSpace d) :
    ‖normalizeLayer p‖ ≤ 1 := by
  by_cases hp : p = 0
  · subst p; simp [normalizeLayer]
  · rw [norm_normalizeLayer hp]

/-- The actual normalized correlation is a measurable observable.
Source: arXiv:2604.01978v1, §7.3, the expectation in `lem:delta_method`. -/
theorem measurable_normalized_inner {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (p q : Ω → EucSpace d) (hp : Measurable p) (hq : Measurable q) :
    Measurable (fun ω => inner ℝ (normalizeLayer (p ω)) (normalizeLayer (q ω))) := by
  unfold normalizeLayer
  fun_prop

/-- Bounded actual normalized correlation is integrable under any
finite original law. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integrable_normalized_inner {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (P : Measure Ω) [IsFiniteMeasure P] (p q : Ω → EucSpace d)
    (hp : Measurable p) (hq : Measurable q) :
    Integrable (fun ω => inner ℝ (normalizeLayer (p ω)) (normalizeLayer (q ω))) P := by
  apply Integrable.of_bound (measurable_normalized_inner p q hp hq).aestronglyMeasurable 1
  filter_upwards with ω
  rw [Real.norm_eq_abs]
  exact (abs_real_inner_le_norm _ _).trans
    ((mul_le_mul (norm_normalizeLayer_le_one _) (norm_normalizeLayer_le_one _)
      (norm_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)).trans_eq (one_mul _))

/-- The actual error is integrable when its explicit cubic majorant
is integrable. Source: arXiv:2604.01978v1, §7.3,
the expectation of the Taylor remainder in `lem:delta_method`. -/
theorem integrable_normalized_inner_delta_error {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (P : Measure Ω) (r : ℝ) (hr : |r| ≤ 1) (p q : Ω → EucSpace d)
    (hp : Measurable p) (hq : Measurable q)
    (hthird : Integrable (fun ω =>
      (|inner ℝ (p ω) (q ω) - r| + |‖p ω‖ ^ 2 - 1| + |‖q ω‖ ^ 2 - 1|) ^ 3) P) :
    Integrable (fun ω => inner ℝ (normalizeLayer (p ω)) (normalizeLayer (q ω)) -
      deltaCorrelationTaylor r (inner ℝ (p ω) (q ω) - r) (‖p ω‖ ^ 2 - 1) (‖q ω‖ ^ 2 - 1)) P := by
  have hm : Measurable (fun ω => deltaCorrelationTaylor r (inner ℝ (p ω) (q ω) - r)
      (‖p ω‖ ^ 2 - 1) (‖q ω‖ ^ 2 - 1)) := by unfold deltaCorrelationTaylor; fun_prop
  apply (hthird.const_mul 40).mono'
    ((measurable_normalized_inner p q hp hq).sub hm).aestronglyMeasurable
  exact ae_of_all _ fun ω => by
    simpa only [Pi.sub_apply, Real.norm_eq_abs] using normalized_inner_delta_taylor_bound r hr (p ω) (q ω)

/-- The genuine quadratic Taylor polynomial is integrable under the
same actual cubic moment condition. Source: arXiv:2604.01978v1,
§7.3, the expected polynomial in `lem:delta_method`. -/
theorem integrable_normalized_inner_delta_polynomial {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (P : Measure Ω) [IsFiniteMeasure P] (r : ℝ) (hr : |r| ≤ 1) (p q : Ω → EucSpace d)
    (hp : Measurable p) (hq : Measurable q)
    (hthird : Integrable (fun ω =>
      (|inner ℝ (p ω) (q ω) - r| + |‖p ω‖ ^ 2 - 1| + |‖q ω‖ ^ 2 - 1|) ^ 3) P) :
    Integrable (fun ω => deltaCorrelationTaylor r (inner ℝ (p ω) (q ω) - r)
      (‖p ω‖ ^ 2 - 1) (‖q ω‖ ^ 2 - 1)) P := by
  have hi := (integrable_normalized_inner P p q hp hq).sub
    (integrable_normalized_inner_delta_error P r hr p q hp hq hthird)
  convert hi using 1
  funext ω
  simp only [Pi.sub_apply]
  ring

/-- **Expected correlation remainder with an actual moment bound.**
Every observable and the exact quadratic polynomial are integrated
under the same original measure. Source: arXiv:2604.01978v1,
§7.3, the expectation step of `lem:delta_method`. -/
theorem integral_normalized_inner_delta_taylor_error {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (P : Measure Ω) [IsFiniteMeasure P] (r : ℝ) (hr : |r| ≤ 1) (p q : Ω → EucSpace d)
    (hp : Measurable p) (hq : Measurable q)
    (hthird : Integrable (fun ω =>
      (|inner ℝ (p ω) (q ω) - r| + |‖p ω‖ ^ 2 - 1| + |‖q ω‖ ^ 2 - 1|) ^ 3) P) :
    |(∫ ω, inner ℝ (normalizeLayer (p ω)) (normalizeLayer (q ω)) ∂P) -
      ∫ ω, deltaCorrelationTaylor r (inner ℝ (p ω) (q ω) - r)
        (‖p ω‖ ^ 2 - 1) (‖q ω‖ ^ 2 - 1) ∂P| ≤
          40 * ∫ ω, (|inner ℝ (p ω) (q ω) - r| + |‖p ω‖ ^ 2 - 1| + |‖q ω‖ ^ 2 - 1|) ^ 3 ∂P := by
  have he := integrable_normalized_inner_delta_error P r hr p q hp hq hthird
  have hea : Integrable (fun ω => |inner ℝ (normalizeLayer (p ω)) (normalizeLayer (q ω)) -
      deltaCorrelationTaylor r (inner ℝ (p ω) (q ω) - r) (‖p ω‖ ^ 2 - 1) (‖q ω‖ ^ 2 - 1)|) P := by
    simpa only [Real.norm_eq_abs] using he.norm
  rw [← integral_sub (integrable_normalized_inner P p q hp hq)
    (integrable_normalized_inner_delta_polynomial P r hr p q hp hq hthird)]
  refine abs_integral_le_integral_abs.trans ?_
  rw [← integral_const_mul]
  exact integral_mono_ae hea (hthird.const_mul 40)
    (ae_of_all _ fun ω => normalized_inner_delta_taylor_bound r hr (p ω) (q ω))

/-- Actual second and fourth scalar moments control the third
absolute moment. Source: arXiv:2604.01978v1, §7.3,
the third-order remainder rate in `lem:delta_method`. -/
theorem third_abs_moment_le_sqrt_second_fourth {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X : Ω → ℝ) (hX : MemLp X 2 P)
    (hfourth : Integrable (fun ω => X ω ^ 4) P) :
    (∫ ω, |X ω| ^ 3 ∂P) ≤ Real.sqrt (∫ ω, X ω ^ 2 ∂P) * Real.sqrt (∫ ω, X ω ^ 4 ∂P) := by
  have hAbs : MemLp (fun ω => |X ω|) 2 P := by simpa only [Real.norm_eq_abs] using hX.norm
  have hSq : MemLp (fun ω => X ω ^ 2) 2 P := by
    apply (memLp_two_iff_integrable_sq (hX.aestronglyMeasurable.pow 2)).mpr
    simpa only [Pi.pow_apply, ← pow_mul] using hfourth
  have h := integral_mul_le_Lp_mul_Lq_of_nonneg (μ := P) Real.HolderConjugate.two_two
    (f := fun ω => |X ω|) (g := fun ω => X ω ^ 2)
    (ae_of_all _ fun ω => abs_nonneg _) (ae_of_all _ fun ω => sq_nonneg _)
    (by simpa using hAbs) (by simpa using hSq)
  simp only [Real.rpow_two, sq_abs, ← pow_mul, ← Real.sqrt_eq_rpow] at h
  calc
    _ = ∫ ω, |X ω| * X ω ^ 2 ∂P := by
      apply integral_congr_ae
      filter_upwards with ω
      rw [← sq_abs (X ω)]
      ring
    _ ≤ _ := h

/-- A nontrivial correlation and measurable nonzero vector queries
under an actual standard Gaussian law jointly satisfy the cubic
moment hypotheses. Its scalar identity has finite second and fourth
moments as required by the Hölder estimate. -/
example : |(1 / 2 : ℝ)| ≤ 1 ∧
    Measurable (fun _ : ℝ => EuclideanSpace.single (0 : Fin 2) (2 : ℝ)) ∧
    Measurable (fun _ : ℝ => EuclideanSpace.single (0 : Fin 2) (3 : ℝ)) ∧
    Integrable (fun _ : ℝ =>
      (|inner ℝ (EuclideanSpace.single (0 : Fin 2) (2 : ℝ)) (EuclideanSpace.single 0 (3 : ℝ)) - 1 / 2| +
        |‖EuclideanSpace.single (0 : Fin 2) (2 : ℝ)‖ ^ 2 - 1| +
          |‖EuclideanSpace.single (0 : Fin 2) (3 : ℝ)‖ ^ 2 - 1|) ^ 3)
      (gaussianReal 0 1) ∧ MemLp (id : ℝ → ℝ) 2 (gaussianReal 0 1) ∧
    Integrable (fun x : ℝ => x ^ 4) (gaussianReal 0 1) := by
  exact ⟨by norm_num, measurable_const, measurable_const, integrable_const _,
    memLp_id_gaussianReal 2, integrable_gaussianReal_pow 1 4⟩

end Transformer.Homogenized
