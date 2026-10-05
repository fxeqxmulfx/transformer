import Transformer.GPTMini.Sparsemax.SharedRows

/-!
# Squared loss on a common affine output segment

Derived outer-loss facts for arXiv:1602.02068v2, §2.5. These are ordinary
squared output errors, not the supervised sparsemax routing loss of §3.
The actual shared Q/K path at `73f8a0b` is connected to these analytic
facts by `sharedRowReadouts_segment`. Convexity is used after proving
that the real operator has affine outputs under the explicit restrictions.

A better endpoint yields nearby strict descent, including when the
sparsemax operator is not differentiable across a support boundary.
Fitting a target gives an exact quadratic decay, and a negative one-sided
derivative along the actual path rules out a zero full derivative.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Ordinary squared output loss is convex on affine readouts.
Source: the derived outer-loss statement for §2.5 of arXiv:1602.02068v2;
this alone does not prove convexity in normalized Q/K parameters. -/
theorem squaredReadoutLoss_segment_le (target start stop : E) (t : ℝ)
    (ht : 0 ≤ t) (hu : t ≤ 1) :
    squaredReadoutLoss target ((1 - t) • start + t • stop) ≤
      (1 - t) * squaredReadoutLoss target start + t * squaredReadoutLoss target stop := by
  have hc : 0 ≤ 1 - t := by linarith
  have hd : (1 - t) • start + t • stop - target =
      (1 - t) • (start - target) + t • (stop - target) := by
    simp only [smul_sub, sub_smul, one_smul]
    abel
  have hn := norm_add_le ((1 - t) • (start - target)) (t • (stop - target))
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_nonneg hc, abs_of_nonneg ht] at hn
  have ha : 0 ≤ (1 - t) * ‖start - target‖ + t * ‖stop - target‖ :=
    add_nonneg (mul_nonneg hc (norm_nonneg _)) (mul_nonneg ht (norm_nonneg _))
  have hq : ‖(1 - t) • (start - target) + t • (stop - target)‖ ^ 2 ≤
      ((1 - t) * ‖start - target‖ + t * ‖stop - target‖) ^ 2 := by
    nlinarith [norm_nonneg ((1 - t) • (start - target) + t • (stop - target))]
  have hv := mul_nonneg (mul_nonneg ht hc) (sq_nonneg (‖start - target‖ - ‖stop - target‖))
  unfold squaredReadoutLoss
  rw [hd]
  nlinarith

/-- Distinct scalar endpoints inhabit both segment inequalities.
Source context: ordinary output interpolation in the derived §2.5 argument. -/
example : squaredReadoutLoss (0 : ℝ) ((1 - (1 / 2 : ℝ)) • (2 : ℝ) + (1 / 2 : ℝ) • 0) ≤
    (1 - (1 / 2 : ℝ)) * squaredReadoutLoss (0 : ℝ) 2 +
      (1 / 2 : ℝ) * squaredReadoutLoss (0 : ℝ) 0 :=
  squaredReadoutLoss_segment_le _ _ _ _ (by norm_num) (by norm_num)

/-- A segment ending at the ordinary target has exact quadratic error decay.
Source: the derived squared-loss consequence of §2.5; all real parameters
are allowed in this analytic identity, before restricting the actual path. -/
theorem squaredReadoutLoss_segment_target (target start : E) (t : ℝ) :
    squaredReadoutLoss target ((1 - t) • start + t • target) =
      (1 - t) ^ 2 * squaredReadoutLoss target start := by
  have hd : (1 - t) • start + t • target - target = (1 - t) • (start - target) := by
    simp only [smul_sub, sub_smul, one_smul]
    abel
  unfold squaredReadoutLoss
  rw [hd, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]

/-- Strictly smaller values at arbitrarily short positive segment times
exclude a local minimum without any differentiability assumption.
Source: the derived neighborhood argument for §2.5, used for the actual
shared matrix path at `73f8a0b`, including inactive threshold ties. -/
theorem not_isLocalMin_of_segment_decrease (loss : ℝ → ℝ)
    (hd : ∀ t, 0 < t → t ≤ 1 → loss t < loss 0) : ¬ IsLocalMin loss 0 := by
  intro hm
  obtain ⟨radius, hr, hn⟩ := Metric.eventually_nhds_iff.mp hm
  let t := min (radius / 2) (1 / 2)
  have ht : 0 < t := lt_min_iff.mpr ⟨by linarith, by norm_num⟩
  have hu : t ≤ 1 := le_trans (min_le_right _ _) (by norm_num)
  have hb : t < radius := lt_of_le_of_lt (min_le_left _ _) (by linarith)
  have hm' := hn (y := t) (by simpa [Real.dist_eq, abs_of_pos ht] using hb)
  exact (not_lt_of_ge hm') (hd t ht hu)

/-- An actual decreasing quadratic inhabits the neighborhood descent premise.
Source context: the fitted-output decay in the derived §2.5 proof. -/
example : ¬ IsLocalMin (fun t : ℝ => (1 - t) ^ 2) 0 := by
  apply not_isLocalMin_of_segment_decrease
  intro t ht hu
  nlinarith

/-- Every positive short time strictly improves a positive target-fit loss.
Source: the exact ordinary squared-error decay derived for §2.5. -/
theorem targetSegment_strict_decrease (initial t : ℝ) (hi : 0 < initial)
    (ht : 0 < t) (hu : t ≤ 1) : (1 - t) ^ 2 * initial < initial := by
  have hf : (1 - t) ^ 2 < 1 := by nlinarith
  have hd := mul_lt_mul_of_pos_right hf hi
  simpa only [one_mul] using hd

/-- Positive error and an interior time inhabit the strict quadratic decrease premises.
Source context: the fitted shared-matrix segment in the derived §2.5 transfer. -/
example : (1 - (1 / 2 : ℝ)) ^ 2 * (9 / 128 : ℝ) < 9 / 128 :=
  targetSegment_strict_decrease _ _ (by norm_num) (by norm_num) (by norm_num)

/-- An affine loss upper bound is strictly improving at positive time
whenever its endpoint loss is smaller.
Source: the derived common-path descent argument for §2.5. -/
theorem lossSegment_strict_decrease (start stop t : ℝ) (hb : stop < start) (ht : 0 < t) :
    (1 - t) * start + t * stop < start := by
  have hd := mul_pos ht (sub_pos.mpr hb)
  nlinarith

/-- Different endpoint losses and positive time inhabit the affine decrease premises.
Source context: the shared loss segment derived for §2.5. -/
example : (1 - (1 / 2 : ℝ)) * (9 / 128 : ℝ) + (1 / 2 : ℝ) * 0 < 9 / 128 :=
  lossSegment_strict_decrease _ _ _ (by norm_num) (by norm_num)

/-- The target-fit quadratic has derivative minus twice its initial loss.
Source: the derived path derivative for ordinary squared error in §2.5. -/
theorem targetSegment_hasDerivAt (initial : ℝ) :
    HasDerivAt (fun t : ℝ => (1 - t) ^ 2 * initial) (-2 * initial) 0 := by
  have hd := (((hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub
    (hasDerivAt_id (0 : ℝ))).pow 2).mul_const initial
  convert hd using 1 <;> norm_num [Pi.sub_apply, Pi.pow_apply]

/-- The analytic target segment reaches zero loss exactly at its final time.
Source: the fitted-output quadratic derived for §2.5. -/
theorem targetSegment_final_loss (initial : ℝ) : (1 - (1 : ℝ)) ^ 2 * initial = 0 := by
  ring

/-- A negative right derivative of an actual loss rules out a zero full derivative.
Source: the derived one-sided argument for §2.5, with no ambient sparsemax
differentiability hypothesis beyond the contradicted zero derivative. -/
theorem no_zero_derivative_of_segment (loss : ℝ → ℝ) (slope : ℝ)
    (hn : slope < 0) (hd : HasDerivWithinAt loss slope (Set.Icc 0 1) 0) :
    ¬ HasDerivAt loss 0 0 := by
  intro hz
  have hu : UniqueDiffWithinAt ℝ (Set.Icc (0 : ℝ) 1) 0 :=
    uniqueDiffOn_Icc_zero_one 0 ⟨by norm_num, by norm_num⟩
  have he := hd.derivWithin hu
  have he' := hz.hasDerivWithinAt.derivWithin hu
  rw [he'] at he
  linarith

/-- The concrete quadratic satisfies both the negative slope and derivative premises.
Source context: the target-fit path used in the derived §2.5 transfer. -/
example : ¬ HasDerivAt (fun t : ℝ => (1 - t) ^ 2) 0 0 := by
  apply no_zero_derivative_of_segment _ (-2) (by norm_num)
  have hd := (targetSegment_hasDerivAt 1).hasDerivWithinAt (s := Set.Icc 0 1)
  simpa only [mul_one] using hd

end Transformer.GPTMini.Sparsemax
