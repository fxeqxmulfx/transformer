module

public import Transformer.ScheduleFree.SectionD_Smooth
public import Mathlib.Analysis.Convex.Integral
public import Mathlib.Analysis.Convex.SpecificFunctions.Pow
public import Transformer.ScheduleFree.SectionE_Conversion

/-!
# The stochastic optimistic-regret estimate

arXiv:2405.15682v4, Appendix D, proof of Corollary 2. This module
proves the Jensen and variance steps, rather than assuming that a
pathwise regret estimate commutes with a square root and expectation.
-/

/-!
# Constants in the accelerated rate

arXiv:2405.15682v4, Appendix D, Corollary 2. The first population
gradient difference is retained explicitly; all later differences are
absorbed by the negative variation term. The final constant is 14.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory

noncomputable section

section

namespace Transformer.ScheduleFree

variable {E Ω : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- Appendix D: the smooth upper and convex lower models imply the usual
squared Lipschitz bound for the gradient. -/
theorem smooth_gradient_sq {F : E → ℝ} {grad : E → E} {L : ℝ} (hL : 0 < L)
    (hgrad : ∀ x, IsSubgradient F x (grad x)) (hsmooth : SmoothUpper F grad L)
    (a b : E) : ‖grad a - grad b‖ ^ 2 ≤ L ^ 2 * ‖a - b‖ ^ 2 := by
  have hc := (div_le_iff₀ (by positivity : 0 < 2 * L)).mp
    (smooth_bregman hL hgrad hsmooth a b)
  have hu := hsmooth b a
  have hb : bregman F grad a b ≤ L / 2 * ‖a - b‖ ^ 2 := by
    unfold bregman
    linarith
  have hm := mul_le_mul_of_nonneg_left hb (by positivity : 0 ≤ 2 * L)
  nlinarith

example (a b : ℝ) : ‖(0 : ℝ) - 0‖ ^ 2 ≤ 1 ^ 2 * ‖a - b‖ ^ 2 :=
  smooth_gradient_sq (F := fun _ : ℝ => 0) (grad := fun _ => 0) zero_lt_one
    (by intro x u; simp) (by intro x y; simp only [inner_zero_left, zero_add]; positivity) a b

/-- Jensen's square-root inequality with the required integrability explicit. -/
theorem sqrt_expectation (V : Ω → ℝ) (hV : ∀ ω, 0 ≤ V ω)
    (hi : Integrable V P) (hs : Integrable (fun ω => Real.sqrt (V ω)) P) :
    (∫ ω, Real.sqrt (V ω) ∂P) ≤ Real.sqrt (∫ ω, V ω ∂P) :=
  Real.strictConcaveOn_sqrt.concaveOn.le_map_integral Real.continuous_sqrt.continuousOn
    isClosed_Ici (Filter.Eventually.of_forall hV) hi hs

example : (∫ _ω : Ω, Real.sqrt (0 : ℝ) ∂P) ≤ Real.sqrt (∫ _ω : Ω, (0 : ℝ) ∂P) :=
  sqrt_expectation (fun _ => 0) (fun _ => le_rfl) (integrable_const 0) (by simp)

/-- Appendix D's optimistic regret and variance assumptions imply the expected
regret bound used in the accelerated rate. For the paper, `v_t` and `a_t`
are the observed and population squared gradient differences. -/
theorem optimistic_expectation {D : ℝ} (hD : 0 ≤ D) (r : Ω → ℝ)
    (v a : ℕ → Ω → ℝ) (σ : ℕ → ℝ) (T : ℕ)
    (hv : ∀ t ω, 0 ≤ v t ω)
    (hregret : ∀ ω, r ω ≤ D * Real.sqrt (∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * v t ω))
    (hvariance : ∀ t < T, (∫ ω, v t ω ∂P) ≤ (∫ ω, a t ω ∂P) + σ t ^ 2)
    (hr : Integrable r P) (hvi : ∀ t < T, Integrable (v t) P)
    (hsqrt : Integrable (fun ω => Real.sqrt (∑ t ∈ Finset.range T,
      ((t : ℝ) + 1) ^ 2 * v t ω)) P) :
    (∫ ω, r ω ∂P) ≤ D * Real.sqrt ((∑ t ∈ Finset.range T,
      ((t : ℝ) + 1) ^ 2 * ∫ ω, a t ω ∂P) +
        ∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * σ t ^ 2) := by
  have hi := integrable_finsetSum (Finset.range T)
    (fun t ht => (hvi t (Finset.mem_range.mp ht)).const_mul (((t : ℝ) + 1) ^ 2))
  have hj := sqrt_expectation _ (fun ω => Finset.sum_nonneg fun t _ =>
    mul_nonneg (sq_nonneg _) (hv t ω)) hi hsqrt
  have hb := integral_mono_ae hr (hsqrt.const_mul D) (Filter.Eventually.of_forall hregret)
  rw [integral_const_mul] at hb
  have he : (∫ ω, ∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * v t ω ∂P) ≤
      (∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * ∫ ω, a t ω ∂P) +
        ∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * σ t ^ 2 := by
    rw [integral_finsetSum (Finset.range T)
      (fun t ht => (hvi t (Finset.mem_range.mp ht)).const_mul (((t : ℝ) + 1) ^ 2)),
      ← Finset.sum_add_distrib]
    apply Finset.sum_le_sum
    intro t ht
    rw [integral_const_mul]
    nlinarith [mul_le_mul_of_nonneg_left (hvariance t (Finset.mem_range.mp ht))
      (sq_nonneg ((t : ℝ) + 1))]
  exact hb.trans (mul_le_mul_of_nonneg_left (hj.trans (Real.sqrt_le_sqrt he)) hD)

example (T : ℕ) : (∫ _ω : Ω, (0 : ℝ) ∂P) ≤ (0 : ℝ) *
    Real.sqrt ((∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * ∫ _ω : Ω, (0 : ℝ) ∂P) +
      ∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * (0 : ℝ) ^ 2) :=
  optimistic_expectation le_rfl (fun _ => 0) (fun _ _ => 0) (fun _ _ => 0) (fun _ => 0) T
    (by intro t ω; exact le_rfl) (by intro ω; simp) (by intro t ht; simp)
    (by simp) (by intro t ht; simp) (by simp)

end Transformer.ScheduleFree

end

section

namespace Transformer.ScheduleFree

/-- The `t(t-1)` variation weights absorb `t²`, except at the first step. -/
theorem variation_absorbs_energy {L : ℝ} (hL : 0 < L) (a : ℕ → ℝ)
    (ha : ∀ t, 0 ≤ a t) (T : ℕ) (hT : 0 < T) :
    (∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * a t) ≤
      24 * L * (∑ t ∈ Finset.range T,
        weightSum (fun k => (k : ℝ) + 1) t / (6 * L) * a t) + a 0 := by
  have hpoint (t : ℕ) : ((t : ℝ) + 1) ^ 2 * a t ≤
      24 * L * (weightSum (fun k => (k : ℝ) + 1) t / (6 * L) * a t) +
        if t = 0 then a 0 else 0 := by
    by_cases ht : t = 0
    · simp [ht, weightSum_zero]
    · have ht1 : (1 : ℝ) ≤ t := by exact_mod_cast (Nat.one_le_iff_ne_zero.mpr ht)
      rw [ite_eq_right ht, add_zero]
      have hc : 24 * L * (weightSum (fun k => (k : ℝ) + 1) t / (6 * L) * a t) =
          2 * (t : ℝ) * ((t : ℝ) + 1) * a t := by
        rw [weightSum_linear]
        field_simp
        ring
      rw [hc]
      nlinarith [mul_nonneg (show 0 ≤ (t : ℝ) ^ 2 - 1 by nlinarith) (ha t)]
  have h := Finset.sum_le_sum (s := Finset.range T) (fun t _ => hpoint t)
  simpa only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_ite_eq',
    Finset.mem_range, hT, ite_true] using h

example (T : ℕ) (hT : 0 < T) :
    (∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * 0) ≤
      24 * 1 * (∑ t ∈ Finset.range T,
        weightSum (fun k => (k : ℝ) + 1) t / (6 * 1) * 0) + 0 :=
  variation_absorbs_energy zero_lt_one (fun _ => 0) (fun _ => le_rfl) T hT

/-- Appendix D: completing the square gives the constant `7` before division
by the total weight `T(T+1)/2`. -/
theorem acceleration_algebra {L D A N A0 V R B : ℝ} (hL : 0 < L) (hD : 0 ≤ D)
    (hA : 0 ≤ A) (hN : 0 ≤ N) (hfirst : A0 ≤ L ^ 2 * D ^ 2)
    (hdebit : A ≤ 24 * L * V + A0) (hregret : R ≤ D * Real.sqrt (A + N))
    (herror : B ≤ R - V) : B ≤ 7 * D ^ 2 * L + D * Real.sqrt N := by
  have hsplit : Real.sqrt (A + N) ≤ Real.sqrt A + Real.sqrt N := by
    nlinarith [Real.sq_sqrt hA, Real.sq_sqrt hN, Real.sq_sqrt (add_nonneg hA hN),
      Real.sqrt_nonneg A, Real.sqrt_nonneg N, Real.sqrt_nonneg (A + N),
      mul_nonneg (Real.sqrt_nonneg A) (Real.sqrt_nonneg N)]
  have hr := hregret.trans (mul_le_mul_of_nonneg_left hsplit hD)
  have hpay : 24 * L * (D * Real.sqrt A) - A ≤ 144 * D ^ 2 * L ^ 2 := by
    nlinarith [sq_nonneg (Real.sqrt A - 12 * D * L), Real.sq_sqrt hA]
  have hm := mul_le_mul_of_nonneg_left (herror.trans (sub_le_sub_right hr V))
    (by positivity : 0 ≤ 24 * L)
  apply (mul_le_mul_iff_right₀ (by positivity : 0 < 24 * L)).mp
  nlinarith [mul_nonneg (sq_nonneg D) (sq_nonneg L)]

example : (0 : ℝ) ≤ 7 * 0 ^ 2 * 1 + 0 * Real.sqrt 0 :=
  acceleration_algebra (A0 := 0) (V := 0) (R := 0) zero_lt_one le_rfl le_rfl le_rfl (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

end Transformer.ScheduleFree

end
