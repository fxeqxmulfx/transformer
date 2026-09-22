/-
# The number of modes of a Gaussian KDE — `lem:phi-t`, `δ_t` and the inner integral

§2.2 of arXiv:2412.09080v3, `lem:phi-t`: the check on `δ_t` in its proof, and
the second display, the size of the inner Kac-Rice integral over the Gaussian
proxy, uniformly over `t ∈ T` (read as in `Section2_PhiTAsymp.lean`).

**What the source says and what is carried here.**

* **The second display needs `n ≲ β^{5/2}`.**  The source checks
  "`α_t^{-1}δ_t² ≪ 1`", which is dimensionally inconsistent: `α_t` scales as
  `y⁻²`, and the quantity that makes `∫_0^∞ y e^{-α(y-δ)²/2} dy ≍ α⁻¹` is
  `α_t δ_t² = O(1)`.  That is `n β^{-5/2} e^{-t²/2} = O(1)`, i.e. `n ≲ β^{5/2}`
  at `t = 0`, which the regime `n^c ≲ β` does not give for `c < 2/5`.  Without
  it the display fails at `t = 0`: numerically, for `β = 100`, the ratio of the
  integral to `α_t^{-1}e^{-A_t/2}` is `0.26, 1.7, 17, 171, 1712` for
  `n = 10⁴, 10⁶, 10⁸, 10¹⁰, 10¹²`, growing as `√n β^{-5/4}`.  Both
  `phiAlpha_mul_phiDelta_sq_le` and `integral_krPhi_isTheta` carry `n ≲ β^{5/2}`
  as a hypothesis.

* The exponent is `e^{-A_t/2}`, with the exact `A_t`: `φ` carries `e^{-‖z‖²/2}`.
  The source writes `e^{-A_t}` with an `A_t` defined only up to constants.

Source: arXiv:2412.09080v3, `lem:phi-t`, `eq:phi-t`.
-/

import Transformer.Modes.Section2_PhiTAsymp

open Real Filter Asymptotics MeasureTheory
open scoped Topology

namespace Transformer
namespace Modes

/-- **Lemma (lem:phi-t), the check on `δ_t`, corrected.**  If moreover
`n ≲ β^{5/2}`, then `α_t δ_t² = O(1)` uniformly on `T`.  The source checks
`α_t^{-1}δ_t² ≪ 1` instead; see the module docstring.

Source: arXiv:2412.09080v3, proof of `lem:phi-t`. -/
theorem phiAlpha_mul_phiDelta_sq_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω)
    (hN : (fun k => (N k : ℝ)) =O[atTop] fun k => B k ^ ((5 : ℝ) / 2)) :
    ∃ C : ℝ, ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2 ≤ C := by
  obtain ⟨C₁, C₂, -, hC₂, hα⟩ := phiAlpha_isTheta hreg hω
  obtain ⟨CN, hCN⟩ := hN.bound
  refine ⟨C₂ * CN * 400, ?_⟩
  filter_upwards [hα, hCN, eventually_quot_mem_Ioo hreg hω,
    hreg.tendsto_B.eventually_ge_atTop 1] with k hαk hNk hk hB t ht
  have hβ := hreg.B_pos k
  obtain ⟨⟨a1, b1⟩, ⟨a2, b2⟩, ⟨a3, b3⟩, ⟨a4, b4⟩, -⟩ := hk t ht
  have hε0 : 0 ≤ (B k)⁻¹ := inv_nonneg.2 hβ.le
  have hε1 : (B k)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hB
  -- `n β^{-5/2} ≤ C_N`
  have h52 : 0 < B k ^ (-(5 : ℝ) / 2) := by positivity
  have hnm : (N k : ℝ) * B k ^ (-(5 : ℝ) / 2) ≤ CN := by
    rw [Real.norm_of_nonneg (Nat.cast_nonneg _), Real.norm_of_nonneg (by positivity)] at hNk
    calc (N k : ℝ) * B k ^ (-(5 : ℝ) / 2)
        ≤ CN * B k ^ ((5 : ℝ) / 2) * B k ^ (-(5 : ℝ) / 2) := by gcongr
      _ = CN := by rw [mul_assoc, ← Real.rpow_add hβ]; norm_num
  have hCN0 : 0 ≤ CN := le_trans (by positivity) hnm
  -- `e^{-t²/2} E² ≤ 400`
  set E := gMean' ((B k)⁻¹, t ^ 2 / B k) * (1 - t ^ 2 + (B k)⁻¹)
      + t ^ 2 * (qCov (B k) t * gMean ((B k)⁻¹, t ^ 2 / B k) / (2 * qVar (B k) t)) with hE
  have hR0 : 0 < qCov (B k) t * gMean ((B k)⁻¹, t ^ 2 / B k) / (2 * qVar (B k) t) := by
    apply div_pos <;> nlinarith
  have hR4 : qCov (B k) t * gMean ((B k)⁻¹, t ^ 2 / B k) / (2 * qVar (B k) t) < 4 := by
    rw [div_lt_iff₀ (by linarith)]; nlinarith [mul_lt_mul b4 b1.le (by linarith) (by norm_num)]
  have ht2 : 0 ≤ t ^ 2 := sq_nonneg t
  have hEu : E ≤ 4 + 6 * t ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hR4.le ht2,
      mul_nonneg (by linarith : (0 : ℝ) ≤ gMean' ((B k)⁻¹, t ^ 2 / B k)) ht2,
      mul_le_mul_of_nonneg_right b2.le (by linarith : (0 : ℝ) ≤ 1 + (B k)⁻¹)]
  have hEl : -(4 + 6 * t ^ 2) ≤ E := by
    nlinarith [mul_nonneg ht2 hR0.le, mul_le_mul_of_nonneg_right b2.le ht2,
      mul_nonneg (by linarith : (0 : ℝ) ≤ gMean' ((B k)⁻¹, t ^ 2 / B k)) hε0]
  have hg := Real.exp_pos (-(t ^ 2) / 2)
  have hg1 : Real.exp (-(t ^ 2) / 2) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
  have hgE : Real.exp (-(t ^ 2) / 2) * E ^ 2 ≤ 400 := by
    have hsq : E ^ 2 ≤ (4 + 6 * t ^ 2) ^ 2 := sq_le_sq' hEl hEu
    nlinarith [mul_le_mul_of_nonneg_left hsq hg.le, sq_mul_exp_neg_half_sq_le t,
      pow_four_mul_exp_neg_half_sq_le t]
  -- assemble
  have h1 : B k ^ ((1 : ℝ) / 2) * (B k ^ (-(3 : ℝ) / 2)) ^ 2 = B k ^ (-(5 : ℝ) / 2) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hβ.le, ← Real.rpow_add hβ]; norm_num
  have h2 : Real.exp (t ^ 2 / 2) * Real.exp (-(t ^ 2) / 2) ^ 2 = Real.exp (-(t ^ 2) / 2) := by
    rw [show Real.exp (-(t ^ 2) / 2) ^ 2 = Real.exp (-(t ^ 2) / 2) * Real.exp (-(t ^ 2) / 2)
      from sq _, ← mul_assoc, ← Real.exp_add, ← Real.exp_add]
    congr 1; ring
  rw [phiDelta_eq hβ t (by linarith), ← hE]
  calc phiAlpha (B k) t * (√(N k) * momentScale (B k) t * E) ^ 2
      ≤ C₂ * (B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2))
          * (√(N k) * momentScale (B k) t * E) ^ 2 :=
        mul_le_mul_of_nonneg_right (hαk t ht).2 (sq_nonneg _)
    _ = C₂ * ((N k : ℝ) * B k ^ (-(5 : ℝ) / 2)) * (Real.exp (-(t ^ 2) / 2) * E ^ 2) := by
        rw [mul_pow, mul_pow, Real.sq_sqrt (Nat.cast_nonneg _), momentScale]
        linear_combination (C₂ * N k * E ^ 2 * (Real.exp (t ^ 2 / 2)
          * Real.exp (-(t ^ 2) / 2) ^ 2)) * h1 + (C₂ * N k * E ^ 2 * B k ^ (-(5 : ℝ) / 2)) * h2
    _ ≤ C₂ * CN * 400 := by gcongr

/-- **Lemma (lem:phi-t), second display, corrected.**  If moreover
`n ≲ β^{5/2}`, then `∫_0^∞ y φ(Σ_t^{-1/2}[(0,y) - μ_t]) dy ≍ α_t^{-1} e^{-A_t/2}`
uniformly on `T`.  Without `n ≲ β^{5/2}` it fails at `t = 0`; see the module
docstring.

Not proved here.

Source: arXiv:2412.09080v3, `lem:phi-t`, `eq:phi-t`. -/
theorem integral_krPhi_isTheta {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω)
    (hN : (fun k => (N k : ℝ)) =O[atTop] fun k => B k ^ ((5 : ℝ) / 2)) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      C₁ * ((phiAlpha (B k) t)⁻¹ * Real.exp (-phiA (N k) (B k) t / 2))
          ≤ ∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y ∧
        ∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y
          ≤ C₂ * ((phiAlpha (B k) t)⁻¹ * Real.exp (-phiA (N k) (B k) t / 2)) := by
  sorry

/-- The hypotheses above are satisfiable: `β = n`, `ω(β) = √(log log β)`, and
`n ≤ n^{5/2}`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) ∧
    (fun k => ((k + 1 : ℕ) : ℝ)) =O[atTop] fun k => ((k + 1 : ℕ) : ℝ) ^ ((5 : ℝ) / 2) := by
  refine ⟨isRegime_succ, isSlowGrowth_sqrt_log_log, ?_⟩
  · refine IsBigO.of_bound 1 ?_
    filter_upwards with k
    have h1 : (1 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) := by push_cast; linarith [k.cast_nonneg (α := ℝ)]
    rw [one_mul, Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg (by positivity)]
    calc ((k + 1 : ℕ) : ℝ) = ((k + 1 : ℕ) : ℝ) ^ (1 : ℝ) := (Real.rpow_one _).symm
      _ ≤ _ := Real.rpow_le_rpow_of_exponent_le h1 (by norm_num)

end Modes
end Transformer
