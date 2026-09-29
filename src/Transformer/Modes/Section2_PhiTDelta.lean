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

* What holds with no bound on `n` is the same bound with the rate kept,
  `α_t δ_t² ≲ n β^{-5/2} e^{-t²/2} (4 + 6t²)²`
  (`phiAlpha_mul_phiDelta_sq_le_rate`, `Section2_PhiTDeltaRate.lean`); it is what
  `main_int_phi_T'` uses on `T'`, where the shift is paid for by `e^{-A_t/2}`.

* The exponent is `e^{-A_t/2}`, with the exact `A_t`: `φ` carries `e^{-‖z‖²/2}`.
  The source writes `e^{-A_t}` with an `A_t` defined only up to constants.

Source: arXiv:2412.09080v3, `lem:phi-t`, `eq:phi-t`.
-/

import Transformer.Modes.Section2_PhiTDeltaRate
import Transformer.Modes.Section2_GaussianShiftBounds

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
  obtain ⟨C, hC, hCk⟩ := phiAlpha_mul_phiDelta_sq_le_rate hreg hω
  obtain ⟨CN, hCN⟩ := hN.bound
  refine ⟨C * CN * 400, ?_⟩
  filter_upwards [hCk, hCN] with k hk hNk t ht
  have hβ := hreg.B_pos k
  -- `n β^{-5/2} ≤ C_N`
  have h52 : 0 < B k ^ (-(5 : ℝ) / 2) := by positivity
  have hnm : (N k : ℝ) * B k ^ (-(5 : ℝ) / 2) ≤ CN := by
    rw [Real.norm_of_nonneg (Nat.cast_nonneg _), Real.norm_of_nonneg (by positivity)] at hNk
    calc (N k : ℝ) * B k ^ (-(5 : ℝ) / 2)
        ≤ CN * B k ^ ((5 : ℝ) / 2) * B k ^ (-(5 : ℝ) / 2) := by gcongr
      _ = CN := by rw [mul_assoc, ← Real.rpow_add hβ]; norm_num
  have hCN0 : 0 ≤ CN := le_trans (by positivity) hnm
  -- `e^{-t²/2} (4 + 6t²)² ≤ 400`
  have hg1 : Real.exp (-(t ^ 2) / 2) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
  have hgE : Real.exp (-(t ^ 2) / 2) * (4 + 6 * t ^ 2) ^ 2 ≤ 400 := by
    nlinarith [sq_mul_exp_neg_half_sq_le t, pow_four_mul_exp_neg_half_sq_le t]
  calc phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2
      ≤ C * ((N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)) * (4 + 6 * t ^ 2) ^ 2 :=
        hk t ht
    _ = C * ((N k : ℝ) * B k ^ (-(5 : ℝ) / 2)) *
          (Real.exp (-(t ^ 2) / 2) * (4 + 6 * t ^ 2) ^ 2) := by ring
    _ ≤ C * CN * 400 := by gcongr

/-- **Lemma (lem:phi-t), second display, corrected.**  If moreover
`n ≲ β^{5/2}`, then `∫_0^∞ y φ(Σ_t^{-1/2}[(0,y) - μ_t]) dy ≍ α_t^{-1} e^{-A_t/2}`
uniformly on `T`.  Without `n ≲ β^{5/2}` it fails at `t = 0`; see the module
docstring.

Source: arXiv:2412.09080v3, `lem:phi-t`, `eq:phi-t`. -/
theorem integral_krPhi_isTheta {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω)
    (hN : (fun k => (N k : ℝ)) =O[atTop] fun k => B k ^ ((5 : ℝ) / 2)) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      C₁ * ((phiAlpha (B k) t)⁻¹ * Real.exp (-phiA (N k) (B k) t / 2))
          ≤ ∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y ∧
        ∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y
          ≤ C₂ * ((phiAlpha (B k) t)⁻¹ * Real.exp (-phiA (N k) (B k) t / 2)) := by
  obtain ⟨C, hC⟩ := phiAlpha_mul_phiDelta_sq_le hreg hω hN
  obtain ⟨L, U, hL, -, hα⟩ := phiAlpha_isTheta hreg hω
  let D₁ : ℝ := (2 * Real.pi)⁻¹ * Real.exp (-C) / 2
  let D₂ : ℝ := (2 * Real.pi)⁻¹ * Real.exp (C / 2) * 2
  refine ⟨D₁, D₂, by dsimp [D₁]; positivity, by dsimp [D₂]; positivity, ?_⟩
  filter_upwards [hC, hα] with k hCk hαk t ht
  let a := phiAlpha (B k) t
  let d := phiDelta (N k) (B k) t
  let A := phiA (N k) (B k) t
  have ha : 0 < a := by
    have hβ := hreg.B_pos k
    have hp : 0 < B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by positivity
    exact lt_of_lt_of_le (mul_pos hL hp) (hαk t ht).1
  have hF : sigmaFst (B k) t ≠ 0 := by
    intro h
    simp [a, phiAlpha, h] at ha
  have hD : sigmaDet (B k) t ≠ 0 := by
    intro h
    simp [a, phiAlpha, h] at ha
  have hquad (y : ℝ) : krQuad (N k) (B k) t 0 y = A + a * (y - d) ^ 2 :=
    krQuad_zero_eq hF hD y
  have hI : (∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y) =
      ((2 * Real.pi)⁻¹ * Real.exp (-A / 2)) *
        (∫ y in Set.Ioi (0 : ℝ), y * Real.exp (-(a / 2) * (y - d) ^ 2)) := by
    calc
      _ = ∫ y in Set.Ioi (0 : ℝ),
          ((2 * Real.pi)⁻¹ * Real.exp (-A / 2)) *
            (y * Real.exp (-(a / 2) * (y - d) ^ 2)) := by
          apply setIntegral_congr_fun measurableSet_Ioi
          intro y hy
          change y * ((2 * Real.pi)⁻¹ * Real.exp (-krQuad (N k) (B k) t 0 y / 2)) = _
          rw [hquad y]
          rw [show -(A + a * (y - d) ^ 2) / 2 = -A / 2 + -(a / 2) * (y - d) ^ 2 by ring,
            Real.exp_add]
          ring
      _ = _ := integral_const_mul _ _
  have hb := integral_mul_exp_neg_shift_sq_bounds ha (hCk t ht)
  have hcoeff : 0 ≤ (2 * Real.pi)⁻¹ * Real.exp (-A / 2) := by positivity
  constructor
  · change D₁ * (a⁻¹ * Real.exp (-A / 2)) ≤ _
    have h := mul_le_mul_of_nonneg_left hb.1 hcoeff
    rw [hI]
    calc
      D₁ * (a⁻¹ * Real.exp (-A / 2))
          = ((2 * Real.pi)⁻¹ * Real.exp (-A / 2)) * (Real.exp (-C) * (2 * a)⁻¹) := by
              dsimp [D₁]
              field_simp [ha.ne']
      _ ≤ _ := h
  · change _ ≤ D₂ * (a⁻¹ * Real.exp (-A / 2))
    have h := mul_le_mul_of_nonneg_left hb.2 hcoeff
    rw [hI]
    calc
      _ ≤ ((2 * Real.pi)⁻¹ * Real.exp (-A / 2)) * (Real.exp (C / 2) * (a / 2)⁻¹) := h
      _ = D₂ * (a⁻¹ * Real.exp (-A / 2)) := by
          dsimp [D₂]
          field_simp [ha.ne']

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
