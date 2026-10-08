import Transformer.Modes.Section3_WeightedPhase
import Transformer.Modes.Section3_MixedMoments
/-!
# Weighted Taylor polynomials for the fourth-moment comparison

The `s = 3` argument of arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
expands the actual characteristic function through degree three and
its derivatives through successively lower degrees. Polynomial weights
inside the phase integral contribute the missing moment orders.

The exact integrated polynomial below consists of literal weighted
moments, with the powers of `i` and factorials retained. The remainder
bound uses only the next weighted absolute moment; the unit phase
avoids any exponential-moment assumption in the Taylor estimate.

The quadratic weighted expansion supplies the first-derivative
remainder needed under finite fourth moments. The Gaussian examples
use actual coordinate weights and phases, with every polynomial moment
integrable, so all assumptions are simultaneously realized.
-/

open Real MeasureTheory Filter
open scoped ENNReal
namespace Transformer.Modes

/-- Integrating the weighted phase polynomial gives its actual weighted moments.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the characteristic Taylor
coefficients before normalized-sum powers and Gaussian damping. -/
theorem integral_real_weighted_phase_taylor
    (μ : Measure (ℝ × ℝ)) (w f : ℝ × ℝ → ℝ) (n : ℕ)
    (hpoly : ∀ k ∈ Finset.range (n + 1), Integrable (fun z => w z * f z ^ k) μ) :
    (∫ z, (w z : ℂ) * (∑ k ∈ Finset.range (n + 1),
      ((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ)) ∂μ) =
      ∑ k ∈ Finset.range (n + 1),
        ((∫ z, w z * f z ^ k ∂μ : ℝ) : ℂ) * Complex.I ^ k / (k.factorial : ℂ) := by
  have he (k : ℕ) : (fun z => (w z : ℂ) * (((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ))) =
      fun z => ((w z * f z ^ k : ℝ) : ℂ) * Complex.I ^ k / (k.factorial : ℂ) := by
    funext z
    push_cast
    rw [mul_pow]
    ring
  have hi (k : ℕ) (hk : k ∈ Finset.range (n + 1)) :
      Integrable (fun z => (w z : ℂ) * (((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ))) μ := by
    rw [he k]
    exact ((hpoly k hk).ofReal.mul_const _).div_const _
  simp_rw [Finset.mul_sum]
  rw [integral_finsetSum _ hi]
  apply Finset.sum_congr rfl
  intro k hk
  rw [he k, integral_div, integral_mul_const, integral_complex_ofReal]

example : ∀ k : ℕ, Integrable (fun z : ℝ × ℝ => z.1 * z.2 ^ k) stdGauss2 := by
  intro k
  simpa only [pow_one] using integrable_pow_mul_pow hasExpMomentsOn_stdGauss2 one_pos 1 k

/-- A weighted phase Taylor remainder needs only its next absolute moment.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4 `eq:br-9.10`.
The deliberately nonsharp denominator `n!` comes from the proved scalar bound. -/
theorem norm_integral_real_weighted_phase_sub_taylor
    (μ : Measure (ℝ × ℝ)) (w f : ℝ × ℝ → ℝ) (hf : Continuous f) (n : ℕ)
    (hpoly : ∀ k ∈ Finset.range (n + 1), Integrable (fun z => w z * f z ^ k) μ)
    (hmom : Integrable (fun z => |w z| * |f z| ^ (n + 1)) μ) :
    ‖(∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
      ∫ z, (w z : ℂ) * (∑ k ∈ Finset.range (n + 1),
        ((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ)) ∂μ‖ ≤
      (∫ z, |w z| * |f z| ^ (n + 1) ∂μ) / (n.factorial : ℝ) := by
  have hw : Integrable w μ := by
    simpa only [pow_zero, mul_one] using hpoly 0 (by simp)
  have hi (k : ℕ) (hk : k ∈ Finset.range (n + 1)) :
      Integrable (fun z => (w z : ℂ) * (((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ))) μ := by
    have he : (fun z => (w z : ℂ) * (((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ))) =
        fun z => ((w z * f z ^ k : ℝ) : ℂ) * Complex.I ^ k / (k.factorial : ℂ) := by
      funext z
      push_cast
      rw [mul_pow]
      ring
    rw [he]
    exact ((hpoly k hk).ofReal.mul_const _).div_const _
  have hP : Integrable (fun z => (w z : ℂ) * (∑ k ∈ Finset.range (n + 1),
      ((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ))) μ := by
    convert integrable_finsetSum (Finset.range (n + 1)) hi using 1
    funext z
    rw [Finset.mul_sum]
  have hE := integrable_real_weighted_phase μ w f hf hw
  rw [← integral_sub hE hP]
  have he : (fun z => (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) -
      (w z : ℂ) * (∑ k ∈ Finset.range (n + 1),
        ((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ))) =
      fun z => (w z : ℂ) * (Complex.exp ((f z : ℂ) * Complex.I) -
        ∑ k ∈ Finset.range (n + 1), ((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ)) := by
    funext z
    ring
  rw [he]
  apply (norm_integral_le_integral_norm _).trans
  calc
    _ ≤ ∫ z, (|w z| * |f z| ^ (n + 1)) / (n.factorial : ℝ) ∂μ := by
      apply integral_mono_of_nonneg (ae_of_all _ fun z => norm_nonneg _) (hmom.div_const _)
      exact ae_of_all _ fun z => by
        dsimp only
        rw [norm_mul, Complex.norm_real, Real.norm_eq_abs]
        calc
          _ ≤ |w z| * (|f z| ^ (n + 1) / (n.factorial : ℝ)) :=
            mul_le_mul_of_nonneg_left (norm_complexExp_imaginary_taylor (f z) n) (abs_nonneg _)
          _ = _ := by ring
    _ = _ := integral_div _ _

example : Continuous (fun z : ℝ × ℝ => z.2) ∧
    (∀ k : ℕ, Integrable (fun z : ℝ × ℝ => z.1 * z.2 ^ k) stdGauss2) ∧
    Integrable (fun z : ℝ × ℝ => |z.1| * |z.2| ^ 4) stdGauss2 := by
  refine ⟨continuous_snd, ?_, ?_⟩
  · intro k
    simpa only [pow_one] using integrable_pow_mul_pow hasExpMomentsOn_stdGauss2 one_pos 1 k
  · simpa only [abs_mul, abs_pow, pow_one] using
      (integrable_pow_mul_pow hasExpMomentsOn_stdGauss2 one_pos 1 4).abs

/-- The weighted quadratic polynomial has cubic phase error, bounded by the
next weighted absolute moment. Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
the unscaled first-derivative Taylor step for `s = 3`. -/
theorem norm_integral_real_weighted_phase_sub_quadratic
    (μ : Measure (ℝ × ℝ)) (w f : ℝ × ℝ → ℝ) (hf : Continuous f)
    (hw : Integrable w μ) (hwf : Integrable (fun z => w z * f z) μ)
    (hwf2 : Integrable (fun z => w z * f z ^ 2) μ)
    (hmom : Integrable (fun z => |w z| * |f z| ^ 3) μ) :
    ‖(∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
      (((∫ z, w z ∂μ : ℝ) : ℂ) + ((∫ z, w z * f z ∂μ : ℝ) : ℂ) * Complex.I -
        ((∫ z, w z * f z ^ 2 ∂μ : ℝ) : ℂ) / 2)‖ ≤
      (∫ z, |w z| * |f z| ^ 3 ∂μ) / 2 := by
  have hpoly (k : ℕ) (hk : k ∈ Finset.range (2 + 1)) :
      Integrable (fun z => w z * f z ^ k) μ := by
    have hk2 : k ≤ 2 := by have := Finset.mem_range.mp hk; omega
    interval_cases k
    · simpa only [pow_zero, mul_one] using hw
    · simpa only [pow_one] using hwf
    · exact hwf2
  have h := norm_integral_real_weighted_phase_sub_taylor μ w f hf 2 hpoly hmom
  rw [integral_real_weighted_phase_taylor μ w f 2 hpoly] at h
  norm_num [Finset.sum_range_succ, Complex.I_sq] at h
  convert h using 1
  congr 1
  ring


example : Continuous (fun z : ℝ × ℝ => z.2) ∧
    Integrable (fun z : ℝ × ℝ => z.1) stdGauss2 ∧
    Integrable (fun z : ℝ × ℝ => z.1 * z.2) stdGauss2 ∧
    Integrable (fun z : ℝ × ℝ => z.1 * z.2 ^ 2) stdGauss2 ∧
    Integrable (fun z : ℝ × ℝ => |z.1| * |z.2| ^ 3) stdGauss2 := by
  refine ⟨continuous_snd, ?_, ?_, ?_, ?_⟩
  · simpa only [pow_one, pow_zero, mul_one] using
      integrable_pow_mul_pow hasExpMomentsOn_stdGauss2 one_pos 1 0
  · simpa only [pow_one] using integrable_pow_mul_pow hasExpMomentsOn_stdGauss2 one_pos 1 1
  · simpa only [pow_one] using integrable_pow_mul_pow hasExpMomentsOn_stdGauss2 one_pos 1 2
  · simpa only [abs_mul, abs_pow, pow_one] using
      (integrable_pow_mul_pow hasExpMomentsOn_stdGauss2 one_pos 1 3).abs

end Transformer.Modes
