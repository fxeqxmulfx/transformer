import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

/-
# The number of modes of a Gaussian KDE — the Fejér kernel

The Fejér kernel `Λ_a(w) = ∫_0^a (1 - s/a) cos(s w) ds = (1 - cos(a w)) / (a w²)` is
nonnegative and at least `a/4` on `|w| ≤ 1/a`.  It is the test function that turns
Fourier decay into a small-ball estimate (`Section5_PtBddSmallBall.lean`).

Source: not a statement of arXiv:2412.09080v3; a tool for refuting its
`eq:uniform-decay` (§5.5) for `β > 2` (`Section5_PtBddDecayFalse.lean`).
-/

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Modes

/-- The Fejér kernel at scale `a`: `Λ_a(w) = ∫_0^a (1 - s/a) cos(s w) ds`. -/
noncomputable def fejer (a w : ℝ) : ℝ := ∫ s in (0 : ℝ)..a, (1 - s / a) * Real.cos (s * w)

/-- The closed form `Λ_a(w) = (1 - cos(a w)) / (a w²)` for `w ≠ 0`. -/
theorem fejer_eq {a w : ℝ} (ha : 0 < a) (hw : w ≠ 0) :
    fejer a w = (1 - Real.cos (a * w)) / (a * w ^ 2) := by
  have ha' : a ≠ 0 := ha.ne'
  have hderiv : ∀ s ∈ Set.uIcc 0 a, HasDerivAt
      (fun s => (1 - s / a) * Real.sin (s * w) / w - Real.cos (s * w) / (a * w ^ 2))
      ((1 - s / a) * Real.cos (s * w)) s := by
    intro s _
    have h1 : HasDerivAt (fun s : ℝ => 1 - s / a) (-(1 / a)) s := by
      simpa using ((hasDerivAt_id s).div_const a).const_sub 1
    have h2 : HasDerivAt (fun s : ℝ => Real.sin (s * w)) (Real.cos (s * w) * w) s := by
      simpa using ((hasDerivAt_id s).mul_const w).sin
    have h3 : HasDerivAt (fun s : ℝ => Real.cos (s * w)) (-Real.sin (s * w) * w) s := by
      simpa using ((hasDerivAt_id s).mul_const w).cos
    have h := ((h1.mul h2).div_const w).sub (h3.div_const (a * w ^ 2))
    convert h using 1
    field_simp
    ring
  rw [fejer, intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv
    ((by fun_prop : Continuous fun s : ℝ => (1 - s / a) * Real.cos (s * w)).intervalIntegrable _ _)]
  simp only [zero_mul, Real.sin_zero, mul_zero, zero_div, Real.cos_zero, div_self ha', sub_self,
    zero_sub]
  field_simp
  ring

/-- The Fejér kernel is nonnegative. -/
theorem fejer_nonneg {a : ℝ} (ha : 0 < a) (w : ℝ) : 0 ≤ fejer a w := by
  by_cases hw : w = 0
  · subst hw
    unfold fejer
    refine intervalIntegral.integral_nonneg ha.le fun s hs => ?_
    have : s / a ≤ 1 := (div_le_one ha).2 hs.2
    simp only [mul_zero, Real.cos_zero, mul_one]
    linarith
  · rw [fejer_eq ha hw]
    exact div_nonneg (by linarith [Real.cos_le_one (a * w)]) (by positivity)

/-- The Fejér kernel is at least `a/4` on `|w| ≤ 1/a`. -/
theorem fejer_ge {a w : ℝ} (ha : 0 < a) (hw : |w| ≤ 1 / a) : a / 4 ≤ fejer a w := by
  have ha' : a ≠ 0 := ha.ne'
  have hint : ∫ s in (0 : ℝ)..a, (1 - s / a) / 2 = a / 4 := by
    have hd : ∀ s ∈ Set.uIcc 0 a, HasDerivAt (fun s : ℝ => (s - s ^ 2 / (2 * a)) / 2)
        ((1 - s / a) / 2) s := by
      intro s _
      have h := ((hasDerivAt_id' s).sub ((hasDerivAt_pow 2 s).div_const (2 * a))).div_const 2
      refine h.congr_deriv ?_
      norm_num
      field_simp
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hd
      ((by fun_prop : Continuous fun s : ℝ => (1 - s / a) / 2).intervalIntegrable _ _)]
    field_simp
    ring
  rw [← hint, fejer]
  refine intervalIntegral.integral_mono_on ha.le
    ((by fun_prop : Continuous fun s : ℝ => (1 - s / a) / 2).intervalIntegrable _ _)
    ((by fun_prop : Continuous fun s : ℝ => (1 - s / a) * Real.cos (s * w)).intervalIntegrable _ _)
    fun s hs => ?_
  have h1 : 0 ≤ 1 - s / a := by linarith [(div_le_one ha).2 hs.2]
  have h2 : |s * w| ≤ 1 := by
    rw [abs_mul, abs_of_nonneg hs.1]
    calc s * |w| ≤ a * (1 / a) := mul_le_mul hs.2 hw (abs_nonneg w) ha.le
      _ = 1 := by field_simp
  have h3 : (1 : ℝ) / 2 ≤ Real.cos (s * w) := by
    have := Real.one_sub_sq_div_two_le_cos (x := s * w)
    have h4 : (s * w) ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one _).2 h2
    linarith
  calc (1 - s / a) / 2 = (1 - s / a) * (1 / 2) := by ring
    _ ≤ (1 - s / a) * Real.cos (s * w) := mul_le_mul_of_nonneg_left h3 h1

/-- The hypotheses of `fejer_nonneg` and `fejer_ge` are satisfiable: `a = 1`, `w = 0`. -/
example : 1 / 4 ≤ fejer 1 0 ∧ 0 ≤ fejer 1 0 :=
  ⟨fejer_ge one_pos (by simp), fejer_nonneg one_pos 0⟩

/-- The hypotheses of `fejer_eq` are satisfiable: `a = 1`, `w = 1`. -/
example : fejer 1 1 = (1 - Real.cos (1 * 1)) / (1 * 1 ^ 2) := fejer_eq one_pos one_ne_zero

end Modes
end Transformer
