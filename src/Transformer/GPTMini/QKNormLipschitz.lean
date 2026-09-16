/-
# How much the QK-normalized score moves

`GPTMini.QKNorm` bounds the score; this module bounds its *variation*.  Two
ingredients, both elementary:

  - `normL2 eps` is Lipschitz with constant `2 / eps`.  As for `rmsNormEps`,
    the scaling factor `1 / (‖x‖ + eps)` moves with `x`, and the second term
    of `c(x)·x - c(y)·y = c(x)·(x - y) + (c(x) - c(y))·y` is controlled by
    `‖y‖ / (‖y‖ + eps) ≤ 1`.
  - the inner product of two vectors of norm at most one is 1-Lipschitz in
    each argument, so the score is `e^α`-Lipschitz in the normalized `q, k`.

Together with `RoPE.applyRope_dist` — the rotation is an isometry, so it drops
out — this is everything the head's Lipschitz estimate needs about the scores.
-/

import Transformer.GPTMini.QKNorm
import Transformer.GPTMini.RoPE

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable {d_head : ℕ}

/-- **`normL2 eps` is Lipschitz with constant `2 / eps`.**

Source: `reference/model.py` (`CausalMHA.forward`, `F.normalize(·, eps=1e-6)`).
As with `rmsNormEps_lipschitz`, small `eps` makes the normalization steep. -/
theorem normL2_lipschitz (eps : ℝ) (heps : 0 < eps) (x y : EucSpace d_head) :
    ‖normL2 eps x - normL2 eps y‖ ≤ 2 / eps * ‖x - y‖ := by
  have hx : (0 : ℝ) < ‖x‖ + eps := by linarith [norm_nonneg x]
  have hy : (0 : ℝ) < ‖y‖ + eps := by linarith [norm_nonneg y]
  have hcx : 1 / (‖x‖ + eps) ≤ 1 / eps :=
    one_div_le_one_div_of_le heps (by linarith [norm_nonneg x])
  have hnorm : |‖y‖ - ‖x‖| ≤ ‖x - y‖ := by
    rw [← norm_neg (x - y), neg_sub]
    exact abs_norm_sub_norm_le y x
  have hcoef : |1 / (‖x‖ + eps) - 1 / (‖y‖ + eps)| * ‖y‖ ≤ 1 / eps * ‖x - y‖ := by
    have hdiff : 1 / (‖x‖ + eps) - 1 / (‖y‖ + eps)
        = (‖y‖ - ‖x‖) / ((‖x‖ + eps) * (‖y‖ + eps)) := by
      field_simp
      ring
    have hsplit : |‖y‖ - ‖x‖| / ((‖x‖ + eps) * (‖y‖ + eps)) * ‖y‖
        = |‖y‖ - ‖x‖| * (1 / (‖x‖ + eps)) * (‖y‖ / (‖y‖ + eps)) := by
      field_simp
    have hlast : ‖y‖ / (‖y‖ + eps) ≤ 1 := by
      rw [div_le_one hy]; linarith
    rw [hdiff, abs_div, abs_of_pos (mul_pos hx hy), hsplit]
    calc |‖y‖ - ‖x‖| * (1 / (‖x‖ + eps)) * (‖y‖ / (‖y‖ + eps))
        ≤ ‖x - y‖ * (1 / eps) * 1 :=
          mul_le_mul
            (mul_le_mul hnorm hcx (one_div_nonneg.mpr hx.le) (norm_nonneg _))
            hlast (div_nonneg (norm_nonneg _) hy.le)
            (mul_nonneg (norm_nonneg _) (one_div_nonneg.mpr heps.le))
      _ = 1 / eps * ‖x - y‖ := by ring
  have hdecomp : normL2 eps x - normL2 eps y
      = (1 / (‖x‖ + eps)) • (x - y)
        + (1 / (‖x‖ + eps) - 1 / (‖y‖ + eps)) • y := by
    rw [normL2, normL2]; module
  calc ‖normL2 eps x - normL2 eps y‖
      ≤ ‖(1 / (‖x‖ + eps)) • (x - y)‖
        + ‖(1 / (‖x‖ + eps) - 1 / (‖y‖ + eps)) • y‖ := by
        rw [hdecomp]; exact norm_add_le _ _
    _ = 1 / (‖x‖ + eps) * ‖x - y‖
        + |1 / (‖x‖ + eps) - 1 / (‖y‖ + eps)| * ‖y‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
          abs_of_nonneg (one_div_nonneg.mpr hx.le)]
    _ ≤ 1 / eps * ‖x - y‖ + 1 / eps * ‖x - y‖ := by
        gcongr
    _ = 2 / eps * ‖x - y‖ := by ring

/-- The inner product of two vectors of norm at most one moves by at most the
sum of the displacements of its arguments. -/
theorem abs_inner_sub_inner_le {d : ℕ} (a b a' b' : EucSpace d)
    (hb : ‖b‖ ≤ 1) (ha' : ‖a'‖ ≤ 1) :
    |inner (𝕜 := ℝ) a b - inner (𝕜 := ℝ) a' b'| ≤ ‖a - a'‖ + ‖b - b'‖ := by
  have hsplit : inner (𝕜 := ℝ) a b - inner (𝕜 := ℝ) a' b'
      = inner (𝕜 := ℝ) (a - a') b + inner (𝕜 := ℝ) a' (b - b') := by
    rw [inner_sub_left, inner_sub_right]; ring
  have h1 : |inner (𝕜 := ℝ) (a - a') b| ≤ ‖a - a'‖ := by
    refine (abs_real_inner_le_norm _ _).trans ?_
    calc ‖a - a'‖ * ‖b‖ ≤ ‖a - a'‖ * 1 :=
          mul_le_mul_of_nonneg_left hb (norm_nonneg _)
      _ = ‖a - a'‖ := mul_one _
  have h2 : |inner (𝕜 := ℝ) a' (b - b')| ≤ ‖b - b'‖ := by
    refine (abs_real_inner_le_norm _ _).trans ?_
    calc ‖a'‖ * ‖b - b'‖ ≤ 1 * ‖b - b'‖ :=
          mul_le_mul_of_nonneg_right ha' (norm_nonneg _)
      _ = ‖b - b'‖ := one_mul _
  rw [hsplit]
  exact (abs_add_le _ _).trans (by linarith)

/-- **The score is `e^α`-Lipschitz in the normalized queries and keys.**

Both arguments have norm at most one after `normL2`, so the bilinear form is
1-Lipschitz in each; the temperature multiplies it.  Source:
`reference/model.py` (`CausalMHA.forward`, `scores = (q @ k^T) * alpha`). -/
theorem score_lipschitz (alpha eps : ℝ) (heps : 0 ≤ eps) (q k q' k' : EucSpace d_head) :
    |score alpha eps q k - score alpha eps q' k'|
      ≤ Real.exp alpha
          * (‖normL2 eps q - normL2 eps q'‖ + ‖normL2 eps k - normL2 eps k'‖) := by
  rw [score, score, ← mul_sub, abs_mul, abs_of_pos (Real.exp_pos alpha)]
  exact mul_le_mul_of_nonneg_left
    (abs_inner_sub_inner_le _ _ _ _ (normL2_norm_le eps heps k) (normL2_norm_le eps heps q'))
    (Real.exp_pos alpha).le

/-- The hypotheses are satisfiable at the values the implementation uses:
`eps = 1e-6` in `CausalMHA.forward`. -/
example (x y : EucSpace 64) :
    ‖normL2 1e-6 x - normL2 1e-6 y‖ ≤ 2 / 1e-6 * ‖x - y‖ :=
  normL2_lipschitz 1e-6 (by norm_num) x y

end GPTMini
end Transformer
