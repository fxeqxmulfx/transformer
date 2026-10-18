/-
# The substitution of `lemma:beta1`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, Appendix A, proof of `lemma:beta1`.

On `[0, π]` the proof substitutes `z = β(1 - cos x) ∈ [0, 2β]`, so that
`f_β''(x) = √β g_β(z)` with

  `g_β(z) = √z √(2 - z/β) (2z - 3 + (3z - 1 - z²)/β) e^{-z}`.

With `t = 1/β` and `B = 2z - 3 + t(3z - 1 - z²)`,
`f_β''(x) = β sin x · B · e^{-z}` and `sin² x = tz(2 - tz)`
(`f_β''_substitution`).  So `f_β''(x)² = β z(2 - tz)B² e^{-2z}`.  For
`t ∈ [0, 1]`, `z ≥ 0` and `tz ≤ 2` this file bounds `z(2 - tz)B² e^{-2z}` by
`(2/5)²` where `B ≥ 0` (`beta1_upper_core`) and by `(8/5)²` where `B ≤ 0`
(`beta1_lower_core`).  `Perspective.SyncKernelBetaOne` turns this into
`-(8/5)√β ≤ f_β'' ≤ (2/5)√β` on `[0, π]`.  Numerically `sup g_β ≈ 0.375` and
`inf g_β ≈ -1.498` over `β ≥ 1`.

* `B ≥ 0` forces `z ≥ 1`.  For `z ≤ 11/5`, AM–GM `27XY² ≤ (X + 2Y)³` with
  `X = 2r(2 - tz)`, `Y = zB` and `r = 3z - 1 - z²` removes `t`, since
  `X + 2Y = 2(3z - 2)`.  For `z ≥ 11/5`, `(2 - tz)B² ≤ 2(2z - 3)²`.
* `B ≤ 0` forces `z ≤ 3/2`.  Then `|B| ≤ 3 - 2z` if `r ≥ 0`.  If `r < 0`,
  then `z ≤ 2/5` and `|B| ≤ 4 - 5z + z²`.

The bounds against `e^{2z}` are in `Perspective.SyncKernelExp`.

The source bounds `g_β` only at the sign changes `b < a` of `f_β'''`.  It
locates `0.18 < β(1 - cos b) < 1.4 < β(1 - cos a)` and uses `g¹ > -1.381`,
`g¹ < 0.375`, `g²(0.18) > -0.247` and `g²(1.4) < 0.512`.  Here the bounds hold
on all of `[0, π]`, and their difference is `2`.
-/

import Transformer.Perspective.SyncKernel
import Transformer.Perspective.SyncKernelExp

open Real

namespace Transformer
namespace Perspective

/-- **The upper case:** `z(2 - tz)B² ≤ (4/25) e^{2z}` where
`B = 2z - 3 + t(3z - 1 - z²) ≥ 0`, for `t ∈ [0, 1]`, `z ≥ 0`, `tz ≤ 2`. -/
theorem beta1_upper_core {t z : ℝ} (ht : 0 ≤ t) (ht1 : t ≤ 1) (hz : 0 ≤ z) (htz : t * z ≤ 2)
    (hB : 0 ≤ 2 * z - 3 + t * (3 * z - 1 - z ^ 2)) :
    z * (2 - t * z) * (2 * z - 3 + t * (3 * z - 1 - z ^ 2)) ^ 2 ≤ 4 / 25 * Real.exp (2 * z) := by
  set r := 3 * z - 1 - z ^ 2 with hr
  set B := 2 * z - 3 + t * r with hBdef
  have hz1 : 1 ≤ z := by
    by_contra h
    have h' : z < 1 := lt_of_not_ge h
    rcases le_or_gt r 0 with hr0 | hr0
    · nlinarith [mul_nonneg ht (neg_nonneg.2 hr0)]
    · nlinarith [mul_le_mul_of_nonneg_right ht1 hr0.le]
  rcases le_or_gt z (11 / 5) with hz2 | hz2
  · have hr0 : 0 < r := by nlinarith
    have hX : 0 ≤ 2 * r * (2 - t * z) := by nlinarith
    have hY : 0 ≤ z * B := mul_nonneg hz hB
    have hamgm : 27 * (2 * r * (2 - t * z)) * (z * B) ^ 2 ≤ 8 * (3 * z - 2) ^ 3 := by
      have hsum : 2 * r * (2 - t * z) + 2 * (z * B) = 2 * (3 * z - 2) := by
        simp only [hBdef, hr]; ring
      have key : (2 * r * (2 - t * z) + 2 * (z * B)) ^ 3 -
          27 * (2 * r * (2 - t * z)) * (z * B) ^ 2 =
          (2 * r * (2 - t * z) - z * B) ^ 2 * (2 * r * (2 - t * z) + 8 * (z * B)) := by ring
      have hnn : 0 ≤ (2 * r * (2 - t * z) - z * B) ^ 2 * (2 * r * (2 - t * z) + 8 * (z * B)) :=
        mul_nonneg (sq_nonneg _) (by linarith)
      rw [hsum] at key
      nlinarith
    have hE := beta1_upper_middle hz1 hz2
    have hrz : 0 < 54 * r * z := by positivity
    refine le_of_mul_le_mul_left ?_ hrz
    nlinarith
  · have hG : (2 - t * z) * B ^ 2 ≤ 2 * (2 * z - 3) ^ 2 := by
      rcases le_or_gt 0 r with hr0 | hr0
      · have ha : 0 ≤ 2 * z - 3 := by linarith
        have h1 : 0 ≤ z * (2 * z - 3) - 4 * r := by nlinarith
        have h2 : 0 ≤ (2 * z - 3) * z - r := by nlinarith
        have key : 2 * (2 * z - 3) ^ 2 - (2 - t * z) * B ^ 2 =
            t * ((2 * z - 3) * (z * (2 * z - 3) - 4 * r) +
              2 * t * r * ((2 * z - 3) * z - r) + t ^ 2 * r ^ 2 * z) := by
          simp only [hBdef]; ring
        have : 0 ≤ t * ((2 * z - 3) * (z * (2 * z - 3) - 4 * r) +
            2 * t * r * ((2 * z - 3) * z - r) + t ^ 2 * r ^ 2 * z) := by positivity
        linarith
      · have hBa : B ≤ 2 * z - 3 := by nlinarith [mul_nonneg ht (neg_nonneg.2 hr0.le)]
        have hB2 : B ^ 2 ≤ (2 * z - 3) ^ 2 := by nlinarith
        nlinarith [mul_nonneg ht hz, sq_nonneg B]
    have hE := beta1_upper_tail (by linarith : (3 : ℝ) / 2 ≤ z)
    nlinarith [mul_le_mul_of_nonneg_left hG hz]

/-- **The lower case:** `z(2 - tz)B² ≤ (64/25) e^{2z}` where
`B = 2z - 3 + t(3z - 1 - z²) ≤ 0`, for `t ∈ [0, 1]`, `z ≥ 0`, `tz ≤ 2`. -/
theorem beta1_lower_core {t z : ℝ} (ht : 0 ≤ t) (ht1 : t ≤ 1) (hz : 0 ≤ z) (htz : t * z ≤ 2)
    (hB : 2 * z - 3 + t * (3 * z - 1 - z ^ 2) ≤ 0) :
    z * (2 - t * z) * (2 * z - 3 + t * (3 * z - 1 - z ^ 2)) ^ 2 ≤
      64 / 25 * Real.exp (2 * z) := by
  set r := 3 * z - 1 - z ^ 2 with hr
  set B := 2 * z - 3 + t * r with hBdef
  have hz2 : z ≤ 3 / 2 := by
    by_contra h
    have h' : 3 / 2 < z := lt_of_not_ge h
    rcases le_or_gt 0 r with hr0 | hr0
    · nlinarith [mul_nonneg ht hr0]
    · nlinarith [mul_le_mul_of_nonpos_right htz hr0.le]
  have h2tz : 2 - t * z ≤ 2 := by nlinarith [mul_nonneg ht hz]
  rcases le_or_gt 0 r with hr0 | hr0
  · have hB2 : B ^ 2 ≤ (3 - 2 * z) ^ 2 := by nlinarith [mul_nonneg ht hr0]
    have hE := beta1_lower_middle hz hz2
    nlinarith [mul_le_mul_of_nonneg_left (mul_le_mul h2tz hB2 (sq_nonneg B) (by norm_num)) hz]
  · have hz25 : z ≤ 2 / 5 := by nlinarith
    have hB2 : B ^ 2 ≤ (4 - 5 * z + z ^ 2) ^ 2 := by
      nlinarith [mul_le_mul_of_nonpos_right ht1 hr0.le]
    have hE := beta1_lower_start hz hz25
    nlinarith [mul_le_mul_of_nonneg_left (mul_le_mul h2tz hB2 (sq_nonneg B) (by norm_num)) hz]

/-- The hypotheses of both cases are satisfiable: `B = 2` at `t = 1`, `z = 2`,
and `B = -76/25` at `t = 1`, `z = 1/5`. -/
example : (2 : ℝ) * (2 - 1 * 2) * (2 * 2 - 3 + 1 * (3 * 2 - 1 - 2 ^ 2)) ^ 2 ≤
      4 / 25 * Real.exp (2 * 2) ∧
    (1 / 5 : ℝ) * (2 - 1 * (1 / 5)) * (2 * (1 / 5) - 3 + 1 * (3 * (1 / 5) - 1 - (1 / 5) ^ 2)) ^ 2 ≤
      64 / 25 * Real.exp (2 * (1 / 5)) :=
  ⟨beta1_upper_core (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num),
    beta1_lower_core (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num)⟩

/-- `f_β''` in the variable `z = β(1 - cos x)` of `lemma:beta1`:
`f_β''(x) = sin x (β(2z - 3) + 3z - 1 - z²) e^{-z}`. -/
theorem f_β''_eq_substitution (β x : ℝ) :
    f_β'' β x = Real.sin x * (β * (2 * (β * (1 - Real.cos x)) - 3) +
      (3 * (β * (1 - Real.cos x)) - 1 - (β * (1 - Real.cos x)) ^ 2)) *
      Real.exp (-(β * (1 - Real.cos x))) := by
  rw [f_β'', show β * (Real.cos x - 1) = -(β * (1 - Real.cos x)) by ring]
  congr 1
  linear_combination β ^ 2 * Real.sin x * Real.sin_sq x

/-- **The substitution of `lemma:beta1`.**  For `β ≥ 1`, with `t = 1/β`,
`z = β(1 - cos x)` and `B = 2z - 3 + t(3z - 1 - z²)`:
`t ∈ [0, 1]`, `z ≥ 0`, `tz ≤ 2`, `f_β''(x) = β sin x · B · e^{-z}` and
`f_β''(x)² e^{2z} = β z(2 - tz)B²`. -/
theorem f_β''_substitution {β : ℝ} (x : ℝ) (hβ : 1 ≤ β) :
    ∃ t z B : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ z ∧ t * z ≤ 2 ∧
      B = 2 * z - 3 + t * (3 * z - 1 - z ^ 2) ∧
      f_β'' β x = β * Real.sin x * B * Real.exp (-z) ∧
      f_β'' β x ^ 2 * Real.exp (2 * z) = β * (z * (2 - t * z) * B ^ 2) := by
  have hβ0 : 0 < β := by linarith
  have hf := f_β''_eq_substitution β x
  have hsin := Real.sin_sq x
  have hc1 := Real.cos_le_one x
  have hc2 := Real.neg_one_le_cos x
  set c := Real.cos x with hc
  set z := β * (1 - c) with hz
  set t := β⁻¹ with ht
  have hβt : β * t = 1 := mul_inv_cancel₀ hβ0.ne'
  have ht0 : 0 ≤ t := inv_nonneg.2 hβ0.le
  have ht1 : t ≤ 1 := inv_le_one_of_one_le₀ hβ
  have hz0 : 0 ≤ z := mul_nonneg hβ0.le (by linarith)
  have hE : Real.exp (-z) ^ 2 * Real.exp (2 * z) = 1 := by
    rw [← Real.exp_nat_mul, ← Real.exp_add]; norm_num
  clear_value t z c
  have htz : t * z = 1 - c := by rw [hz]; linear_combination (1 - c) * hβt
  have hf' : f_β'' β x =
      β * Real.sin x * (2 * z - 3 + t * (3 * z - 1 - z ^ 2)) * Real.exp (-z) := by
    rw [hf]; linear_combination (-(Real.sin x * (3 * z - 1 - z ^ 2) * Real.exp (-z))) * hβt
  refine ⟨t, z, 2 * z - 3 + t * (3 * z - 1 - z ^ 2), ht0, ht1, hz0, by linarith, rfl, hf', ?_⟩
  have hs2 : Real.sin x ^ 2 = t * z * (2 - t * z) := by rw [hsin, htz]; ring
  rw [hf']
  linear_combination (β ^ 2 * (2 * z - 3 + t * (3 * z - 1 - z ^ 2)) ^ 2 *
    Real.exp (-z) ^ 2 * Real.exp (2 * z)) * hs2 +
    (β * z * (2 - t * z) * (2 * z - 3 + t * (3 * z - 1 - z ^ 2)) ^ 2 *
      Real.exp (-z) ^ 2 * Real.exp (2 * z)) * hβt +
    (β * (z * (2 - t * z) * (2 * z - 3 + t * (3 * z - 1 - z ^ 2)) ^ 2)) * hE

/-- The hypothesis `1 ≤ β` of `f_β''_substitution` holds at `β = 1`. -/
example : ∃ t z B : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ z ∧ t * z ≤ 2 ∧
    B = 2 * z - 3 + t * (3 * z - 1 - z ^ 2) ∧
    f_β'' 1 (π / 2) = 1 * Real.sin (π / 2) * B * Real.exp (-z) ∧
    f_β'' 1 (π / 2) ^ 2 * Real.exp (2 * z) = 1 * (z * (2 - t * z) * B ^ 2) :=
  f_β''_substitution _ le_rfl

end Perspective
end Transformer
