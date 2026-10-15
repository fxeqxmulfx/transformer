/-
# Controlling actual normalized correlation before composing factors

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, composition
of the two original Gaussian factors. The correlation after the
first factor is random. Its difference from the original input
correlation controls the change in the cubic correction of the
second factor.

Normalization moves a vector by at most its squared-norm deviation
from one. This gives a global bound on normalized correlation by
the sum of its original score and two squared-norm deviations.
The estimate also includes zero vectors and makes no assumption
that either random squared norm stays away from zero.

The genuine cubic bias `r³-r` is Lipschitz on the full correlation
interval, with explicit constant two. A probability-space Cauchy
estimate bounds each first absolute deviation by its second moment.
All these estimates concern the actual observables, rather than a
substitute covariance or an assumed Gaussian product matrix.
-/

import Transformer.Homogenized.Section7_CorrelationTaylorExpectation

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- Normalization moves the actual vector by at most its original
squared-norm deviation, including the origin.
Source: arXiv:2604.01978v1, §7.3, normalized queries
in `lem:delta_method`. -/
theorem delta_norm_normalization_deviation_le {d : ℕ} (p : EucSpace d) :
    ‖normalizeLayer p - p‖ ≤ |‖p‖ ^ 2 - 1| := by
  by_cases hp : p = 0
  · subst p
    simp [normalizeLayer]
  have hvec : ‖p‖ • normalizeLayer p = p := by
    rw [normalizeLayer, smul_smul, mul_inv_cancel₀ (norm_ne_zero_iff.mpr hp), one_smul]
  have hdiff : normalizeLayer p - p = (1 - ‖p‖) • normalizeLayer p := by
    rw [sub_smul, one_smul, hvec]
  rw [hdiff, norm_smul, Real.norm_eq_abs, norm_normalizeLayer hp, mul_one]
  rw [show ‖p‖ ^ 2 - 1 = (‖p‖ - 1) * (‖p‖ + 1) by ring,
    abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ ‖p‖ + 1), abs_sub_comm ‖p‖ 1]
  nlinarith [mul_nonneg (abs_nonneg (1 - ‖p‖)) (norm_nonneg p)]

/-- The difference between the actual normalized and raw query
scores is controlled by the original norm deviations.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the effect of normalizing each original query image. -/
theorem delta_normalized_inner_raw_error_le {d : ℕ} (p q : EucSpace d) :
    |inner ℝ (normalizeLayer p) (normalizeLayer q) - inner ℝ p q| ≤
      |‖p‖ ^ 2 - 1| + ‖p‖ * |‖q‖ ^ 2 - 1| := by
  calc
    _ = |inner ℝ (normalizeLayer p - p) (normalizeLayer q) +
        inner ℝ p (normalizeLayer q - q)| := by
      rw [inner_sub_left, inner_sub_right]
      congr 1
      ring
    _ ≤ |inner ℝ (normalizeLayer p - p) (normalizeLayer q)| +
        |inner ℝ p (normalizeLayer q - q)| := abs_add_le _ _
    _ ≤ ‖normalizeLayer p - p‖ * ‖normalizeLayer q‖ + ‖p‖ * ‖normalizeLayer q - q‖ :=
      add_le_add (abs_real_inner_le_norm _ _) (abs_real_inner_le_norm _ _)
    _ ≤ _ := add_le_add
      ((mul_le_mul (delta_norm_normalization_deviation_le p) (norm_normalizeLayer_le_one q)
        (norm_nonneg _) (abs_nonneg _)).trans_eq (mul_one _))
      (mul_le_mul_of_nonneg_left (delta_norm_normalization_deviation_le q) (norm_nonneg p))

/-- **Global linear control of the actual normalized correlation error.**
The original three deviations suffice, with no lower norm bound.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the random intermediate correlation when composing the original factors. -/
theorem delta_normalized_inner_deviation_le {d : ℕ} (r : ℝ) (hr : |r| ≤ 1) (p q : EucSpace d) :
    |inner ℝ (normalizeLayer p) (normalizeLayer q) - r| ≤
      2 * (|inner ℝ p q - r| + |‖p‖ ^ 2 - 1| + |‖q‖ ^ 2 - 1|) := by
  by_cases hp : ‖p‖ ≤ 2
  · have htri := abs_sub_le (inner ℝ (normalizeLayer p) (normalizeLayer q)) (inner ℝ p q) r
    have hraw := delta_normalized_inner_raw_error_le p q
    have hprod := mul_le_mul_of_nonneg_right hp (abs_nonneg (‖q‖ ^ 2 - 1))
    linarith [abs_nonneg (inner ℝ p q - r), abs_nonneg (‖p‖ ^ 2 - 1)]
  · have hp2 : 2 ≤ ‖p‖ := le_of_not_ge hp
    have hs : 4 ≤ ‖p‖ ^ 2 := by nlinarith [norm_nonneg p]
    have hc : |inner ℝ (normalizeLayer p) (normalizeLayer q)| ≤ 1 := by
      exact (abs_real_inner_le_norm _ _).trans
        ((mul_le_mul (norm_normalizeLayer_le_one p) (norm_normalizeLayer_le_one q)
          (norm_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)).trans_eq (one_mul _))
    have ht := abs_sub_le (inner ℝ (normalizeLayer p) (normalizeLayer q)) 0 r
    simp only [sub_zero, zero_sub, abs_neg] at ht
    linarith [le_abs_self (‖p‖ ^ 2 - 1), abs_nonneg (inner ℝ p q - r), abs_nonneg (‖q‖ ^ 2 - 1)]

/-- The genuine cubic correlation bias is uniformly Lipschitz on
the full interval of actual correlations. Source: arXiv:2604.01978v1,
§7.3, the correction term in `lem:delta_method`. -/
theorem delta_cubic_bias_lipschitz (s r : ℝ) (hs : |s| ≤ 1) (hr : |r| ≤ 1) :
    |(s ^ 3 - s) - (r ^ 3 - r)| ≤ 2 * |s - r| := by
  have hs2 : s ^ 2 ≤ 1 := by simpa only [sq_abs, one_pow] using pow_le_pow_left₀ (abs_nonneg s) hs 2
  have hr2 : r ^ 2 ≤ 1 := by simpa only [sq_abs, one_pow] using pow_le_pow_left₀ (abs_nonneg r) hr 2
  have hsr : |s * r| ≤ 1 := by
    rw [abs_mul]
    exact (mul_le_mul hs hr (abs_nonneg r) (by norm_num : (0 : ℝ) ≤ 1)).trans_eq (one_mul _)
  have hcoef : |s ^ 2 + s * r + r ^ 2 - 1| ≤ 2 := by
    apply abs_le.mpr
    constructor <;> nlinarith [(abs_le.mp hsr).1, (abs_le.mp hsr).2, sq_nonneg s, sq_nonneg r]
  rw [show (s ^ 3 - s) - (r ^ 3 - r) = (s - r) * (s ^ 2 + s * r + r ^ 2 - 1) by ring, abs_mul]
  exact (mul_le_mul_of_nonneg_left hcoef (abs_nonneg _)).trans_eq (mul_comm _ _)

/-- The actual expected cubic bias changes by at most twice the
first absolute correlation deviation. Every observable is integrable
under the original probability law, because the actual correlations
are bounded by one. Source: arXiv:2604.01978v1, §7.3,
`lem:delta_method`, composing the two single-factor correction terms. -/
theorem integral_delta_cubic_bias_deviation_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (R : Ω → ℝ) (hR : Measurable R)
    (hbound : ∀ ω, |R ω| ≤ 1) (r : ℝ) (hr : |r| ≤ 1) :
    |(∫ ω, (R ω ^ 3 - R ω) ∂P) - (r ^ 3 - r)| ≤ 2 * ∫ ω, |R ω - r| ∂P := by
  have idev : Integrable (fun ω => |R ω - r|) P := by
    apply Integrable.of_bound (by fun_prop) 2
    filter_upwards with ω
    rw [Real.norm_eq_abs, abs_abs]
    have h := abs_sub_le (R ω) 0 r
    simp only [sub_zero, zero_sub, abs_neg] at h
    linarith [hbound ω]
  have ierr : Integrable (fun ω => (R ω ^ 3 - R ω) - (r ^ 3 - r)) P := by
    apply (idev.const_mul 2).mono' (by fun_prop)
    exact ae_of_all _ fun ω => by
      rw [Real.norm_eq_abs]
      exact delta_cubic_bias_lipschitz (R ω) r (hbound ω) hr
  have ibias : Integrable (fun ω => R ω ^ 3 - R ω) P := by
    convert ierr.add (integrable_const (r ^ 3 - r)) using 1
    funext ω
    simp only [Pi.add_apply]
    ring
  have he : (∫ ω, (R ω ^ 3 - R ω) ∂P) - (r ^ 3 - r) =
      ∫ ω, ((R ω ^ 3 - R ω) - (r ^ 3 - r)) ∂P := by
    rw [integral_sub ibias (integrable_const _)]
    simp
  rw [he]
  refine abs_integral_le_integral_abs.trans ?_
  rw [← integral_const_mul]
  apply integral_mono (by simpa only [Real.norm_eq_abs] using ierr.norm) (idev.const_mul 2)
  exact fun ω => delta_cubic_bias_lipschitz (R ω) r (hbound ω) hr

/-- A genuine first absolute moment is controlled by the actual
second moment under any probability law.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the intermediate correlation estimate for composing factors. -/
theorem first_abs_moment_le_sqrt_second {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (hX : MemLp X 2 P) :
    (∫ ω, |X ω| ∂P) ≤ Real.sqrt (∫ ω, X ω ^ 2 ∂P) := by
  have hAbs : MemLp (fun ω => |X ω|) 2 P := by simpa only [Real.norm_eq_abs] using hX.norm
  have hc : MemLp (fun _ : Ω => (1 : ℝ)) 2 P := memLp_const 1
  have h := integral_mul_le_Lp_mul_Lq_of_nonneg (μ := P) Real.HolderConjugate.two_two
    (f := fun ω => |X ω|) (g := fun _ => (1 : ℝ))
    (ae_of_all _ fun ω => abs_nonneg _) (ae_of_all _ fun _ => by norm_num)
    (by simpa using hAbs) (by simpa using hc)
  simpa only [mul_one, Real.rpow_two, sq_abs, one_pow, integral_const, probReal_univ,
    one_smul, Real.one_rpow, ← Real.sqrt_eq_rpow, Real.sqrt_one] using h

/-- A genuine nonzero unit vector, two nontrivial input
correlations, a bounded measurable correlation and a Gaussian identity satisfy the
normalization, cubic-bias and probability-moment hypotheses. -/
example : |(1 / 2 : ℝ)| ≤ 1 ∧ |(-1 / 2 : ℝ)| ≤ 1 ∧
    (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) ≠ 0 ∧
    Measurable (fun _ : ℝ => (1 / 2 : ℝ)) ∧ (∀ _ : ℝ, |(1 / 2 : ℝ)| ≤ 1) ∧
    MemLp (id : ℝ → ℝ) 2 (gaussianReal 0 1) := by
  refine ⟨by norm_num, by norm_num, ?_, measurable_const, fun _ => by norm_num,
    memLp_id_gaussianReal 2⟩
  intro he
  have hn : ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 := by simp
  rw [he] at hn
  norm_num at hn

end Transformer.Homogenized
