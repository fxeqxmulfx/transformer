/-
# Metastability — bounds on the collapse time `T₁`

The collapse time `T₁ = t_a + t_b` of `Metastability.CollapseTime` satisfies, for `r = 8ε`
and `δ = e^{-λβ} < r` (arXiv:2410.06833v1, §2, `eq: an.ineq` and the bound on `T₁` in
`thm: metastability`):

* `collapseTime_le` — `T₁ ≤ 2 n e^{βr} + e n λ β²/(β-1)`: the linear phase lasts at most
  `n r/F_β(r) = n e^{βr}/(1 - r) ≤ 2 n e^{βr}`, the exponential phase at most
  `log(σ₀/δ)/c₂ ≤ λβ · β e n/(β-1)`;
* `collapseTime_pos` — `T₁ > 0`.
-/

import Transformer.Metastability.CollapseTime

open Real Set

namespace Transformer
namespace Metastability

/-- The linear phase lasts at most `2 n e^{βr}`: `t_a ≤ n r / F_β(r) = n e^{βr}/(1 - r)`. -/
theorem collapseTimeLinear_le {n : ℕ} {β r δ : ℝ} (hn : 1 ≤ n) (hδ : 0 < δ) (hδr : δ < r)
    (hr : r ≤ 1 / 2) (hFr : 0 < collapseRate β r) :
    collapseTimeLinear n β r δ ≤ 2 * n * Real.exp (β * r) := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hr0 : 0 < r := hδ.trans hδr
  have hσ0 : 0 ≤ collapseSwitch β r δ := hδ.le.trans (le_collapseSwitch β r δ)
  have hc : 0 < collapseRate β r / n := div_pos hFr hn'
  have h1r : 1 - r ≠ 0 := (by linarith : (0 : ℝ) < 1 - r).ne'
  calc collapseTimeLinear n β r δ = (r - collapseSwitch β r δ) / (collapseRate β r / n) := rfl
    _ ≤ r / (collapseRate β r / n) := div_le_div_of_nonneg_right (by linarith) hc.le
    _ = n * Real.exp (β * r) / (1 - r) := by
        unfold collapseRate
        rw [Real.exp_neg]
        field_simp
    _ ≤ 2 * n * Real.exp (β * r) := by
        rw [div_le_iff₀ (by linarith)]
        nlinarith [mul_pos hn' (Real.exp_pos (β * r))]

/-- The exponential phase lasts at most `e n λ β²/(β - 1)`: `log(σ₀/δ) ≤ log(1/δ) = λβ` and
`c₂ = (β-1)/(β e n)`. -/
theorem collapseTimeExp_le {n : ℕ} {β r lam : ℝ} (hβ : 1 < β) (hn : 1 ≤ n)
    (hδr : Real.exp (-(lam * β)) < r) (hr : r ≤ 1 / 2) :
    collapseTimeExp n β r (Real.exp (-(lam * β))) ≤ Real.exp 1 * n * lam * β ^ 2 / (β - 1) := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hδpos : 0 < Real.exp (-(lam * β)) := Real.exp_pos _
  have hσpos : 0 < collapseSwitch β r (Real.exp (-(lam * β))) :=
    hδpos.trans_le (le_collapseSwitch β r _)
  have hσ1 : collapseSwitch β r (Real.exp (-(lam * β))) ≤ 1 :=
    (collapseSwitch_le hδr.le).trans (by linarith)
  have hlog : Real.log (collapseSwitch β r (Real.exp (-(lam * β))) / Real.exp (-(lam * β)))
      ≤ lam * β := by
    rw [Real.log_div hσpos.ne' hδpos.ne', Real.log_exp]
    have := Real.log_nonpos hσpos.le hσ1
    linarith
  have hc₂ : 0 < (β - 1) / (β * Real.exp 1 * n) := div_pos (by linarith) (by positivity)
  have hβ0 : 0 < β := by linarith
  have hβ1 : β - 1 ≠ 0 := (sub_pos.2 hβ).ne'
  calc collapseTimeExp n β r (Real.exp (-(lam * β)))
      = Real.log (collapseSwitch β r (Real.exp (-(lam * β))) / Real.exp (-(lam * β)))
          / ((β - 1) / (β * Real.exp 1 * n)) := rfl
    _ ≤ lam * β / ((β - 1) / (β * Real.exp 1 * n)) := div_le_div_of_nonneg_right hlog hc₂.le
    _ = Real.exp 1 * n * lam * β ^ 2 / (β - 1) := by field_simp

/-- **`T₁ ≤ 2 n e^{βr} + e n λ β²/(β-1)`**, the upper bound on the collapse time in
`thm: metastability` (`r = 8ε`). -/
theorem collapseTime_le {n : ℕ} {β r lam : ℝ} (hβ : 1 < β) (hn : 1 ≤ n)
    (hδr : Real.exp (-(lam * β)) < r) (hr : r ≤ 1 / 2) (hFr : 0 < collapseRate β r) :
    collapseTime n β r (Real.exp (-(lam * β)))
      ≤ 2 * n * Real.exp (β * r) + Real.exp 1 * n * lam * β ^ 2 / (β - 1) :=
  add_le_add (collapseTimeLinear_le hn (Real.exp_pos _) hδr hr hFr)
    (collapseTimeExp_le hβ hn hδr hr)

/-- **`T₁ > 0`** when `δ < r`: either the linear phase or the exponential one has positive
length. -/
theorem collapseTime_pos {n : ℕ} {β r δ : ℝ} (hβ : 1 < β) (hn : 1 ≤ n) (hδ : 0 < δ)
    (hδr : δ < r) (hFr : 0 < collapseRate β r) : 0 < collapseTime n β r δ := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hc₁ : 0 < collapseRate β r / n := div_pos hFr hn'
  have hc₂ : 0 < (β - 1) / (β * Real.exp 1 * n) := div_pos (by linarith) (by positivity)
  have hσ : collapseSwitch β r δ ≤ r := collapseSwitch_le hδr.le
  have hδσ := le_collapseSwitch β r δ
  have hb : 0 ≤ collapseTimeExp n β r δ :=
    div_nonneg (Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)) hc₂.le
  have ha : 0 ≤ collapseTimeLinear n β r δ := div_nonneg (by linarith) hc₁.le
  unfold collapseTime
  rcases hσ.lt_or_eq with h | h
  · have : 0 < collapseTimeLinear n β r δ := div_pos (by linarith) hc₁
    linarith
  · have : 0 < collapseTimeExp n β r δ := by
      refine div_pos (Real.log_pos ?_) hc₂
      rw [h, one_lt_div hδ]
      exact hδr
    linarith

end Metastability
end Transformer
