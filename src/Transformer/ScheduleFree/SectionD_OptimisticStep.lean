module

public import Transformer.ScheduleFree.SectionD_OptimisticBasic

/-!
# One-step optimistic mirror-descent inequalities

arXiv:2405.15682v4, Appendix D.1, proof of Lemma 1. Both projected
updates are used explicitly. Their negative squared distances cancel
the Young inequality, and the adaptive charge is paid by the increase
of the regularization scale.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix D.1, adaptive summation: the lagged quadratic charge is bounded
by twice the regularizer increment when the clipped square is at most `p²`
and at most `q²-p²`. Division at `p=0` is handled explicitly. -/
theorem clipped_charge {p q a : ℝ} (hp : 0 ≤ p) (hpq : p ≤ q)
    (hap : a ≤ p ^ 2) (haq : a ≤ q ^ 2 - p ^ 2) :
    a / (2 * p) ≤ 2 * (q - p) := by
  by_cases hz : p = 0
  · subst p
    simp only [mul_zero, div_zero, sub_zero]
    positivity
  · have hpos : 0 < p := lt_of_le_of_ne hp (Ne.symm hz)
    have hbound : a ≤ 4 * p * (q - p) := by
      by_cases hq : q ≤ 2 * p
      · have hprod := mul_nonneg (show 0 ≤ 2 * p - q by linarith)
          (sub_nonneg.mpr hpq)
        have hprod2 := mul_nonneg hp (sub_nonneg.mpr hpq)
        nlinarith
      · have hprod := mul_nonneg hp (show 0 ≤ q - 2 * p by linarith)
        nlinarith [sq_nonneg p]
    apply (div_le_iff₀ (by positivity : 0 < 2 * p)).mpr
    nlinarith

example : (0 : ℝ) / (2 * 0) ≤ 2 * (0 - 0) :=
  clipped_charge le_rfl le_rfl (by norm_num) (by norm_num)

/-- Appendix D.1: the exact scale after adding a clipped difference. -/
theorem optimisticStep_scale_sq (D : ℝ) (s : OptimisticState E) (g : E) (he : 0 ≤ s.energy) :
    optimisticScale (optimisticStep D s g) ^ 2 =
      max s.peak ‖g - s.previous‖ ^ 2 +
        (s.energy + ‖clipGradient s.peak s.previous g - s.previous‖ ^ 2) :=
  optimisticScale_sq _ (add_nonneg he (sq_nonneg _))

example : optimisticScale (optimisticStep 1 (⟨(0 : ℝ), 0, 0, 0, 0⟩ : OptimisticState ℝ) 0) ^ 2 =
    max 0 ‖(0 : ℝ) - 0‖ ^ 2 + (0 + ‖clipGradient 0 (0 : ℝ) 0 - 0‖ ^ 2) :=
  optimisticStep_scale_sq 1 _ 0 le_rfl

/-- Appendix D.1: scale monotonicity and both bounds needed for the clipped charge. -/
theorem optimisticStep_scale_properties (D : ℝ) (s : OptimisticState E) (g : E)
    (hm : 0 ≤ s.peak) (he : 0 ≤ s.energy) :
    optimisticScale s ≤ optimisticScale (optimisticStep D s g) ∧
      ‖clipGradient s.peak s.previous g - s.previous‖ ^ 2 ≤ optimisticScale s ^ 2 ∧
      ‖clipGradient s.peak s.previous g - s.previous‖ ^ 2 ≤
        optimisticScale (optimisticStep D s g) ^ 2 - optimisticScale s ^ 2 := by
  have hold := optimisticScale_sq s he
  have hnew := optimisticStep_scale_sq D s g he
  have hvariation : ‖clipGradient s.peak s.previous g - s.previous‖ ≤ s.peak := by
    rw [clipGradient_variation hm]
    exact min_le_left _ _
  have hsmall := (sq_le_sq₀ (norm_nonneg _) hm).mpr hvariation
  have hpeak := (sq_le_sq₀ hm (hm.trans (le_max_left _ _))).mpr
    (le_max_left s.peak ‖g - s.previous‖)
  have hincrement : ‖clipGradient s.peak s.previous g - s.previous‖ ^ 2 ≤
      optimisticScale (optimisticStep D s g) ^ 2 - optimisticScale s ^ 2 := by
    nlinarith only [hold, hnew, hpeak]
  refine ⟨?_, ?_, hincrement⟩
  · apply (sq_le_sq₀ (optimisticScale_nonneg _) (optimisticScale_nonneg _)).mp
    nlinarith only [hincrement, sq_nonneg ‖clipGradient s.peak s.previous g - s.previous‖]
  · nlinarith only [hold, hsmall, he]

example : optimisticScale (⟨(0 : ℝ), 0, 0, 0, 0⟩ : OptimisticState ℝ) ≤
    optimisticScale (optimisticStep 1 ⟨(0 : ℝ), 0, 0, 0, 0⟩ 0) :=
  (optimisticStep_scale_properties 1 _ 0 le_rfl le_rfl).1


/-- Appendix D.1: the scaled Young inequality that cancels the optimistic distance. -/
theorem optimistic_young {D q : ℝ} (hD : 0 < D) (hq : 0 < q) (a b : E) :
    inner ℝ a b ≤ D / (2 * q) * ‖a‖ ^ 2 + q / (2 * D) * ‖b‖ ^ 2 := by
  have hpos : 0 < 2 * D * q := by positivity
  apply (mul_le_mul_iff_of_pos_left hpos).mp
  have he : (2 * D * q) * (D / (2 * q) * ‖a‖ ^ 2 + q / (2 * D) * ‖b‖ ^ 2) =
      D ^ 2 * ‖a‖ ^ 2 + q ^ 2 * ‖b‖ ^ 2 := by
    field_simp
  rw [he]
  have hi := mul_le_mul_of_nonneg_left (real_inner_le_norm a b) hpos.le
  nlinarith [sq_nonneg (D * ‖a‖ - q * ‖b‖)]

example (a b : E) : inner ℝ a b ≤ (1 : ℝ) / (2 * 1) * ‖a‖ ^ 2 +
    (1 : ℝ) / (2 * 1) * ‖b‖ ^ 2 :=
  optimistic_young zero_lt_one zero_lt_one a b

/-- Appendix D.1: cancellation of the two projection inequalities at positive scales. -/
theorem optimistic_projection_step {D p q : ℝ} (hD : 0 < D) (hp : 0 < p) (hpq : p ≤ q)
    (y previous clipped u : E) (hu : ‖u‖ ≤ D) :
    let z := projectBall D (y - (D / p) • previous)
    let next := projectBall D (y - (D / q) • clipped)
    inner ℝ clipped (z - u) ≤
      q / (2 * D) * (‖y - u‖ ^ 2 - ‖next - u‖ ^ 2) +
        D / (2 * p) * ‖clipped - previous‖ ^ 2 := by
  let z := projectBall D (y - (D / p) • previous)
  let next := projectBall D (y - (D / q) • clipped)
  have hq : 0 < q := hp.trans_le hpq
  have ha := projected_three_point hD.le (D / q) y clipped u hu
  have hb := projected_three_point hD.le (D / p) y previous next
    (norm_projectBall hD.le _)
  change 2 * (D / q) * inner ℝ clipped (next - u) ≤
    ‖y - u‖ ^ 2 - ‖next - u‖ ^ 2 - ‖next - y‖ ^ 2 at ha
  change 2 * (D / p) * inner ℝ previous (z - next) ≤
    ‖y - next‖ ^ 2 - ‖z - next‖ ^ 2 - ‖z - y‖ ^ 2 at hb
  rw [norm_sub_rev y next] at hb
  have hA := mul_le_mul_of_nonneg_left ha (show 0 ≤ q / (2 * D) by positivity)
  have hB := mul_le_mul_of_nonneg_left hb (show 0 ≤ p / (2 * D) by positivity)
  have hcA : q / (2 * D) * (2 * (D / q) * inner ℝ clipped (next - u)) =
      inner ℝ clipped (next - u) := by field_simp
  have hcB : p / (2 * D) * (2 * (D / p) * inner ℝ previous (z - next)) =
      inner ℝ previous (z - next) := by field_simp
  rw [hcA] at hA
  rw [hcB] at hB
  have hyoung := optimistic_young hD hp (clipped - previous) (z - next)
  have horder : p / (2 * D) ≤ q / (2 * D) :=
    div_le_div_of_nonneg_right hpq (by positivity)
  have hdiscardA := mul_nonneg (sub_nonneg.mpr horder) (sq_nonneg ‖next - y‖)
  have hdiscardB := mul_nonneg (show 0 ≤ p / (2 * D) by positivity) (sq_nonneg ‖z - y‖)
  have hsplit : inner ℝ clipped (z - u) = inner ℝ clipped (next - u) +
      inner ℝ previous (z - next) + inner ℝ (clipped - previous) (z - next) := by
    simp only [inner_sub_left, inner_sub_right]
    ring
  change inner ℝ clipped (z - u) ≤ _
  rw [hsplit]
  nlinarith only [hA, hB, hyoung, hdiscardA, hdiscardB]

example (y previous clipped : E) :
    let z := projectBall 1 (y - ((1 : ℝ) / 1) • previous)
    let next := projectBall 1 (y - ((1 : ℝ) / 1) • clipped)
    inner ℝ clipped (z - 0) ≤
      (1 : ℝ) / (2 * 1) * (‖y - 0‖ ^ 2 - ‖next - 0‖ ^ 2) +
        (1 : ℝ) / (2 * 1) * ‖clipped - previous‖ ^ 2 :=
  optimistic_projection_step zero_lt_one zero_lt_one le_rfl y previous clipped 0 (by simp)


end Transformer.ScheduleFree
