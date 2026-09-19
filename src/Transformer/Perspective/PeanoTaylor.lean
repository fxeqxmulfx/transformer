/-
# Second-order expansions with a Peano remainder

The two analytic facts Appendix B's `e:Hessianincoord` runs on, both stated
for real functions of a real variable and both independent of the survey:

* `isLittleO_secondOrder` — Taylor–Young at order two.  A function whose
  derivative exists everywhere and is itself differentiable at `0` agrees with
  its second-order polynomial up to `o(t²)`.  Mathlib's Taylor theorems
  (`taylor_isLittleO`, `taylor_mean_remainder_lagrange`) all ask for `C^n`
  regularity on an interval, which is more than the hypothesis
  `Perspective.SecondDerivEBetaAt` provides.
* `isLittleO_inner` — a little-o times a big-O, paired by the inner product.

They are kept apart from the appendix because neither mentions it.
-/

import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.MeanValue

open scoped BigOperators
open Asymptotics Filter

namespace Transformer
namespace Perspective

/-- **Taylor–Young at order two.**  If `f` is differentiable everywhere with
derivative `f'`, and `f'` is differentiable at `0` with derivative `c`, then

  `f t = f 0 + f' 0 · t + (c/2) t² + o(t²)`.

The proof is the mean value inequality applied to the remainder on
`[-|t|, |t|]`: its derivative is `o(t)` there, so the remainder itself is
`|t| · o(t) = o(t²)`. -/
theorem isLittleO_secondOrder {f f' : ℝ → ℝ} {c : ℝ}
    (hf : ∀ t : ℝ, HasDerivAt f (f' t) t) (hc : HasDerivAt f' c 0) :
    (fun t : ℝ => f t - f 0 - f' 0 * t - c / 2 * t ^ 2) =o[nhds 0] fun t : ℝ => t ^ 2 := by
  set h : ℝ → ℝ := fun t => f t - f 0 - f' 0 * t - c / 2 * t ^ 2 with hh
  set h' : ℝ → ℝ := fun t => f' t - f' 0 - c * t with hh'
  have hderiv : ∀ t : ℝ, HasDerivAt h (h' t) t := by
    intro t
    have ha : HasDerivAt (fun s : ℝ => f s - f 0) (f' t) t := (hf t).sub_const (f 0)
    have hb : HasDerivAt (fun s : ℝ => f' 0 * s) (f' 0) t :=
      ((hasDerivAt_id' t).const_mul (f' 0)).congr_deriv (mul_one _)
    have h2 : HasDerivAt (fun s : ℝ => c / 2 * s ^ 2) (c * t) t :=
      ((hasDerivAt_pow 2 t).const_mul (c / 2)).congr_deriv (by push_cast; ring)
    exact (ha.sub hb).sub h2
  have hzero : h 0 = 0 := by simp [hh]
  have hlo : (fun t : ℝ => h' t) =o[nhds 0] fun t : ℝ => t := by
    have := hasDerivAt_iff_isLittleO.mp hc
    simpa [hh'] using this.congr' (by filter_upwards with t; ring) (by simp)
  rw [Asymptotics.isLittleO_iff]
  intro ε hε
  obtain ⟨δ, hδ, hball⟩ :=
    Metric.eventually_nhds_iff.mp (Asymptotics.isLittleO_iff.mp hlo hε)
  refine Metric.eventually_nhds_iff.mpr ⟨δ, hδ, fun t ht => ?_⟩
  have habs : |t| < δ := by simpa [Real.dist_eq] using ht
  have hbound : ∀ y ∈ Metric.closedBall (0 : ℝ) |t|, ‖h' y‖ ≤ ε * |t| := by
    intro y hy
    have hy' : |y| ≤ |t| := by simpa [Real.dist_eq] using hy
    have : ‖h' y‖ ≤ ε * ‖y‖ := hball (by simpa [Real.dist_eq] using lt_of_le_of_lt hy' habs)
    calc ‖h' y‖ ≤ ε * ‖y‖ := this
      _ ≤ ε * |t| := by
          have : ‖y‖ ≤ |t| := by simpa [Real.norm_eq_abs] using hy'
          exact mul_le_mul_of_nonneg_left this hε.le
  have hmvt : ‖h t - h 0‖ ≤ ε * |t| * ‖t - 0‖ :=
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (fun y _ => (hderiv y).hasDerivWithinAt) hbound (convex_closedBall 0 |t|)
      (Metric.mem_closedBall_self (abs_nonneg t))
      (by simp)
  rw [hzero, sub_zero] at hmvt
  calc ‖h t‖ ≤ ε * |t| * ‖t - 0‖ := hmvt
    _ = ε * ‖t ^ 2‖ := by
        rw [sub_zero, Real.norm_eq_abs, Real.norm_eq_abs, abs_pow]
        ring

/-- A monomial that is `o(t^k)` at `0` has zero coefficient: the coefficient
of the leading term of a Peano expansion is unique. -/
theorem eq_zero_of_isLittleO_pow {C : ℝ} {k : ℕ}
    (h : (fun t : ℝ => C * t ^ k) =o[nhds 0] fun t : ℝ => t ^ k) : C = 0 := by
  by_contra hC
  have hCpos : 0 < |C| := abs_pos.mpr hC
  obtain ⟨δ, hδ, hball⟩ :=
    Metric.eventually_nhds_iff.mp (Asymptotics.isLittleO_iff.mp h (half_pos hCpos))
  have hhalf : (0 : ℝ) < δ / 2 := by linarith
  have ht : dist (δ / 2) (0 : ℝ) < δ := by
    rw [Real.dist_eq, sub_zero, abs_of_pos hhalf]; linarith
  have hb := hball ht
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_mul, abs_pow] at hb
  have hpow : (0 : ℝ) < |δ / 2| ^ k := pow_pos (abs_pos.mpr (ne_of_gt hhalf)) k
  nlinarith [hb, hpow, hCpos]

end Perspective
end Transformer
