/-
# Metastability — the parameter conditions of `thm: metastability`, made explicit

The three phases of `Metastability.CollapseTime` need `F_β(s) = s (1 - s) e^{-βs}` above the
leakage level `2 n² e^{-(1-α)β}` at the two ends `s = 8ε` and `s = δ = e^{-λβ}` of the collapse,
and the resulting time `T₁` has to be below `T₂ = (ε/n) e^{(1-α)β}`.  This file derives all of
it from the hypotheses of the paper (`eq: gamma`, `eq: lambda.3`):

* `collapseRate_gt_of_gamma` — `γ(β) > 0` gives `F_β(8ε) > 2 n² e^{-(1-α)β}`;
* `collapseRate_gt_of_lam` — the second bound of `eq: lambda.3` gives `F_β(δ) > 2 n² e^{-(1-α)β}`;
* `collapse_bound_lt` — the first bound of `eq: lambda.3` gives
  `2 n e^{8εβ} + e n λ β²/(β-1) < T₂`.

The bounds on the collapse time itself (`T₁ ≤ 2 n e^{8εβ} + e n λ β²/(β-1)`, `T₁ > 0`) are in
`Metastability.CollapseTimeBounds`.
-/

import Transformer.Metastability.Basic
import Transformer.Metastability.CollapseScalar

open Real Set

namespace Transformer
namespace Metastability

/-- **`γ(β) > 0` puts `F_β(8ε)` above the leakage level.**  `γ > 0` says
`2n²/ε < e^{β(1-α-8ε)}`, i.e. `2 n² e^{-(1-α)β} < ε e^{-8εβ}`, and `ε ≤ 8ε (1 - 8ε)`. -/
theorem collapseRate_gt_of_gamma {n : ℕ} {β ε α : ℝ} (hβ : 0 < β) (hε : 0 < ε)
    (hε' : ε < 1 / 16) (hγ : 0 < γβ n β α ε) (hn : 1 ≤ n) :
    2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β (8 * ε) := by
  have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hpos : 0 < 2 * (n : ℝ) ^ 2 / ε := by positivity
  have h1 : Real.log (2 * (n : ℝ) ^ 2 / ε) < β * (1 - α - 8 * ε) := by
    unfold γβ at hγ
    have : β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / ε) < 1 - α - 8 * ε := by linarith
    rwa [inv_mul_lt_iff₀ hβ] at this
  have h2 : 2 * (n : ℝ) ^ 2 / ε < Real.exp (β * (1 - α - 8 * ε)) :=
    (Real.log_lt_iff_lt_exp hpos).1 h1
  have h3 : 2 * (n : ℝ) ^ 2 < ε * Real.exp (β * (1 - α - 8 * ε)) := by
    rw [div_lt_iff₀ hε] at h2
    linarith
  have h4 : 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < ε * Real.exp (-(β * (8 * ε))) := by
    have h := mul_lt_mul_of_pos_right h3 (Real.exp_pos (-((1 - α) * β)))
    rw [mul_assoc ε, ← Real.exp_add] at h
    convert h using 3
    ring
  have h5 : ε * Real.exp (-(β * (8 * ε))) ≤ 8 * ε * (1 - 8 * ε) * Real.exp (-(β * (8 * ε))) := by
    refine mul_le_mul_of_nonneg_right ?_ (Real.exp_pos _).le
    nlinarith
  unfold collapseRate
  linarith

/-- **The second bound of `eq: lambda.3` puts `F_β(δ)` above the leakage level.**  For
`δ = e^{-λβ} < 8ε` and `λ < 1 - α - β⁻¹ log(2n²/(1 - 8ε)) - 8ε` (`e^{-λ_*β} = 8ε`):
`F_β(δ) ≥ (1 - 8ε) e^{-8εβ} δ > 2 n² e^{-(1-α)β}`. -/
theorem collapseRate_gt_of_lam {n : ℕ} {β ε α lam : ℝ} (hβ : 0 < β) (hε : 0 < ε)
    (hε' : ε < 1 / 16) (hn : 1 ≤ n) (hδ : Real.exp (-(lam * β)) < 8 * ε)
    (hlam : lam < 1 - α - Real.log (2 * (n : ℝ) ^ 2 / (1 - 8 * ε)) / β - 8 * ε) :
    2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β (Real.exp (-(lam * β))) := by
  have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have h8 : 0 < 1 - 8 * ε := by linarith
  have hpos : 0 < 2 * (n : ℝ) ^ 2 / (1 - 8 * ε) := by positivity
  have h1 : Real.log (2 * (n : ℝ) ^ 2 / (1 - 8 * ε)) < (1 - α) * β - 8 * ε * β - lam * β := by
    have : Real.log (2 * (n : ℝ) ^ 2 / (1 - 8 * ε)) / β < 1 - α - 8 * ε - lam := by linarith
    rw [div_lt_iff₀ hβ] at this
    linarith
  have h2 := (Real.log_lt_iff_lt_exp hpos).1 h1
  have h3 : 2 * (n : ℝ) ^ 2 < (1 - 8 * ε) * Real.exp ((1 - α) * β - 8 * ε * β - lam * β) := by
    rw [div_lt_iff₀ h8] at h2
    linarith
  have h4 : 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β))
      < (1 - 8 * ε) * (Real.exp (-(β * (8 * ε))) * Real.exp (-(lam * β))) := by
    have h := mul_lt_mul_of_pos_right h3 (Real.exp_pos (-((1 - α) * β)))
    rw [mul_assoc (1 - 8 * ε), ← Real.exp_add] at h
    rw [← Real.exp_add]
    convert h using 3
    ring
  have hδpos : 0 < Real.exp (-(lam * β)) := Real.exp_pos _
  have hexp : Real.exp (-(β * (8 * ε))) ≤ Real.exp (-(β * Real.exp (-(lam * β)))) :=
    Real.exp_le_exp.2 (by nlinarith)
  have h5 : (1 - 8 * ε) * (Real.exp (-(β * (8 * ε))) * Real.exp (-(lam * β)))
      ≤ Real.exp (-(lam * β)) * (1 - Real.exp (-(lam * β)))
          * Real.exp (-(β * Real.exp (-(lam * β)))) := by
    calc (1 - 8 * ε) * (Real.exp (-(β * (8 * ε))) * Real.exp (-(lam * β)))
        = Real.exp (-(lam * β)) * (1 - 8 * ε) * Real.exp (-(β * (8 * ε))) := by ring
      _ ≤ Real.exp (-(lam * β)) * (1 - Real.exp (-(lam * β)))
          * Real.exp (-(β * Real.exp (-(lam * β)))) :=
        mul_le_mul (mul_le_mul_of_nonneg_left (by linarith) hδpos.le) hexp
          (Real.exp_pos _).le (mul_nonneg hδpos.le (by linarith))
  unfold collapseRate
  linarith

/-- **The first bound of `eq: lambda.3` puts `T₁` below `T₂`.**  Since
`(ε/n) e^{(1-α)β} e^{-γβ} = 2 n e^{8εβ}` by the definition of `γ`, the first bound

  `λ < e^{(1 - α + β⁻¹ log((β-1)ε/(β²n²e)))β} (1 - e^{-γβ})`

(the sign of the logarithm is corrected, see `metastability`) is equivalent to
`e n λ β²/(β-1) < (ε/n) e^{(1-α)β} (1 - e^{-γβ})`, hence to
`2 n e^{8εβ} + e n λ β²/(β-1) < (ε/n) e^{(1-α)β}`. -/
theorem collapse_bound_lt {n : ℕ} {β ε α lam : ℝ} (hβ : 1 < β) (hε : 0 < ε) (hn : 1 ≤ n)
    (hlam : lam < Real.exp ((1 - α + β⁻¹ * Real.log ((β - 1) * ε /
        (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1))) * β) * (1 - Real.exp (-(γβ n β α ε * β)))) :
    2 * n * Real.exp (8 * ε * β) + Real.exp 1 * n * lam * β ^ 2 / (β - 1)
      < (ε / n) * Real.exp ((1 - α) * β) := by
  have hβ0 : 0 < β := by linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hβ1 : 0 < β - 1 := by linarith
  have hXpos : 0 < Real.exp ((1 - α) * β) := Real.exp_pos _
  have hcpos : 0 < (β - 1) * ε / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1) :=
    div_pos (mul_pos hβ1 hε) (by positivity)
  have hpos : 0 < 2 * (n : ℝ) ^ 2 / ε := by positivity
  have hexp1 : Real.exp ((1 - α + β⁻¹ * Real.log ((β - 1) * ε /
      (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1))) * β)
      = Real.exp ((1 - α) * β) * ((β - 1) * ε / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1)) := by
    have : (1 - α + β⁻¹ * Real.log ((β - 1) * ε / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1))) * β
        = (1 - α) * β + Real.log ((β - 1) * ε / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1)) := by
      field_simp
    rw [this, Real.exp_add, Real.exp_log hcpos]
  have hγ : γβ n β α ε * β
      = (1 - α) * β - 8 * ε * β - Real.log (2 * (n : ℝ) ^ 2 / ε) := by
    unfold γβ
    field_simp
  have hEγ : Real.exp (-(γβ n β α ε * β))
      = Real.exp (8 * ε * β) * (2 * (n : ℝ) ^ 2 / ε) / Real.exp ((1 - α) * β) := by
    rw [hγ, show -((1 - α) * β - 8 * ε * β - Real.log (2 * (n : ℝ) ^ 2 / ε))
        = (8 * ε * β + Real.log (2 * (n : ℝ) ^ 2 / ε)) - (1 - α) * β by ring,
      Real.exp_sub, Real.exp_add, Real.exp_log hpos]
  rw [hexp1] at hlam
  have hKpos : 0 < Real.exp 1 * n * β ^ 2 / (β - 1) := div_pos (by positivity) hβ1
  have hKc : Real.exp 1 * n * β ^ 2 / (β - 1)
      * ((β - 1) * ε / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1)) = ε / n := by
    field_simp
  have h1 := mul_lt_mul_of_pos_left hlam hKpos
  have h2 : Real.exp 1 * n * β ^ 2 / (β - 1) * (Real.exp ((1 - α) * β) *
      ((β - 1) * ε / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1))
        * (1 - Real.exp (-(γβ n β α ε * β))))
      = ε / n * Real.exp ((1 - α) * β) * (1 - Real.exp (-(γβ n β α ε * β))) := by
    rw [← hKc]; ring
  have h3 : 2 * (n : ℝ) * Real.exp (8 * ε * β)
      = ε / n * Real.exp ((1 - α) * β) * Real.exp (-(γβ n β α ε * β)) := by
    rw [hEγ]
    field_simp
  have h4 : Real.exp 1 * n * lam * β ^ 2 / (β - 1)
      = Real.exp 1 * n * β ^ 2 / (β - 1) * lam := by ring
  rw [h2] at h1
  rw [h4]
  linarith

end Metastability
end Transformer
