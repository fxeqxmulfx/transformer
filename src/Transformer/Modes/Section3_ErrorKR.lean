import Transformer.Modes.Section3_ErrorHigher

/-
# The number of modes of a Gaussian KDE — higher-order errors in the Kac–Rice integral

§3 and §3.2 of arXiv:2412.09080v3: `cor:error-higher`, the bounds
`lem:error-higher` gives on the Kac–Rice integrals of `g₂ = q_t - φ` and
`g₃ = q_t - φ - n^{-1/2}ψ`, and `eq:error-goal`, the goal of §3.

**What the source says and what is carried here.**

* The integrals are `ℝ≥0∞`-valued and every bound is on the integral itself,
  so an infinite integral cannot satisfy them.

* The source's integral bounds use "the density" `q_t`, together with the
  continuity asserted in `lem: pt.bdd`. The continuous-family versions on
  `T`, expressed by `IsDensityFamily`, are false in the allowed regime
  `β = n`: no such family exists. The refutations below concern that
  assertion, and do not refute integral estimates for other density versions.
  The two claims on `T'` remain open; this regime has empty `T'` eventually.

* "The second equation holds upon replacing `y ≥ 0` by `y ≥ Δ_t`, by
  non-negativity of the integrand" is proved, `gThreeKR_Ioi_le`.  The integrand
  is clipped at `0`, so for `Δ_t < 0` the extra `y ≤ 0` contribute nothing.

* `eq:error-goal`'s `≪` is `o`.

Source: arXiv:2412.09080v3, `eq:error-goal`, `cor:error-higher` and the remark
after its proof.
-/

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

/-- `Σ_t^{-1/2}[(0, y) - μ_t]`, the point the Kac–Rice integrand evaluates a
density of `n^{-1/2} Σ Yᵢ(t)` at.  arXiv:2412.09080v3, `eq:error-goal`. -/
noncomputable def krPoint (n : ℕ) (β t y : ℝ) : ℝ × ℝ :=
  whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t) (-muFst n β t, y - muSnd n β t)

/-- `q t` is, for every `t ∈ S`, a continuous density of `n^{-1/2} Σ Yᵢ(t)`:
the `q_t` of arXiv:2412.09080v3, `eq:qt`. -/
structure IsDensityFamily (n : ℕ) (β : ℝ) (S : Set ℝ) (q : ℝ → ℝ × ℝ → ℝ) : Prop where
  /-- Each `q t` is continuous. -/
  continuous : ∀ t ∈ S, Continuous (q t)
  /-- Each `q t` is a density of `n^{-1/2} Σ Yᵢ(t)`. -/
  isDensityOf : ∀ t ∈ S,
    IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) (q t)

/-- Continuous density families on `T` are eventually impossible in `β = n`.
Source: arXiv:2412.09080v3, §3.2, `cor:error-higher`; §4.1, `lem: pt.bdd`. -/
theorem not_eventually_densityFamily_T_succ (ω : ℝ → ℝ) :
    ¬ ∀ᶠ k : ℕ in atTop, ∃ q,
      IsDensityFamily (k + 1) (k + 1) (intervalT (k + 1) (k + 1) (ω (k + 1))) q := by
  intro h
  apply not_eventually_continuous_density_on_T_succ ω
  filter_upwards [h] with k hk
  obtain ⟨q, hq⟩ := hk
  intro t ht
  exact ⟨q t, hq.continuous t ht, hq.isDensityOf t ht⟩

/-- `∫_S ∫_{R_t} (det Σ_t)^{-1/2} y |q_t - φ|(Σ_t^{-1/2}[(0,y) - μ_t]) dy dt`.
arXiv:2412.09080v3, `eq:error-goal`, `cor:error-higher`. -/
noncomputable def gTwoKR (n : ℕ) (β : ℝ) (S : Set ℝ) (q : ℝ → ℝ × ℝ → ℝ)
    (R : ℝ → Set ℝ) : ℝ≥0∞ :=
  ∫⁻ t in S, ∫⁻ y in R t, ENNReal.ofReal (y * sigmaDet β t ^ (-(1 : ℝ) / 2) *
    |q t (krPoint n β t y) - phi2 (krPoint n β t y)|)

/-- `∫_S ∫_{R_t} (det Σ_t)^{-1/2} y |q_t - φ - n^{-1/2}ψ|(Σ_t^{-1/2}[(0,y) - μ_t]) dy dt`.
arXiv:2412.09080v3, `cor:error-higher`. -/
noncomputable def gThreeKR (n : ℕ) (β : ℝ) (S : Set ℝ) (q : ℝ → ℝ × ℝ → ℝ)
    (R : ℝ → Set ℝ) : ℝ≥0∞ :=
  ∫⁻ t in S, ∫⁻ y in R t, ENNReal.ofReal (y * sigmaDet β t ^ (-(1 : ℝ) / 2) *
    |q t (krPoint n β t y) - phi2 (krPoint n β t y)
      - (Real.sqrt n)⁻¹ * psiOf (lawY β t) (krPoint n β t y)|)

/-! ### `cor:error-higher` -/

/-- **Counterexample to the continuous-family version of the first display
of `cor:error-higher` on `T`.** The allowed regime `β = n` has no such family,
independently of the proposed integral bound.
Source: arXiv:2412.09080v3, §3.2, `cor:error-higher`; §4.1, `lem: pt.bdd`. -/
theorem not_error_higher_two_KR_T (ω : ℝ → ℝ) :
    ¬ ∃ C : ℝ, ∀ᶠ k : ℕ in atTop, ∃ q,
      IsDensityFamily (k + 1) (k + 1) (intervalT (k + 1) (k + 1) (ω (k + 1))) q ∧
      gTwoKR (k + 1) (k + 1) (intervalT (k + 1) (k + 1) (ω (k + 1))) q
          (fun t => Set.Ioc 0 (deltaCut (k + 1) (k + 1) t))
        ≤ ENNReal.ofReal
          (C * (Real.exp (-(ω (k + 1)) / 4) * Real.sqrt ((k + 1) * Real.log (k + 1)))) := by
  rintro ⟨C, hC⟩
  apply not_eventually_densityFamily_T_succ ω
  exact hC.mono fun k ⟨q, hq, _⟩ => ⟨q, hq⟩

/-- **The continuous-family version of the second display is also false.**
Its rate cannot restore the density family missing in the allowed `β = n`.
Source: arXiv:2412.09080v3, §3.2, `cor:error-higher`; §4.1, `lem: pt.bdd`. -/
theorem not_error_higher_three_KR_T (ω : ℝ → ℝ) :
    ¬ ∃ C : ℝ, ∀ᶠ k : ℕ in atTop, ∃ q,
      IsDensityFamily (k + 1) (k + 1) (intervalT (k + 1) (k + 1) (ω (k + 1))) q ∧
      gThreeKR (k + 1) (k + 1) (intervalT (k + 1) (k + 1) (ω (k + 1))) q (fun _ => Set.Ioi 0)
        ≤ ENNReal.ofReal
          (C * (Real.exp (-(ω (k + 1)) / 2) * Real.sqrt ((k + 1) * Real.log (k + 1)))) := by
  rintro ⟨C, hC⟩
  apply not_eventually_densityFamily_T_succ ω
  exact hC.mono fun k ⟨q, hq, _⟩ => ⟨q, hq⟩

/-- **Corollary (cor:error-higher), first display, on `T'`:** `O(√β)`.

Not proved here.

Source: arXiv:2412.09080v3, `cor:error-higher`, "Moreover". -/
theorem error_higher_two_KR_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B) :
    ∃ C : ℝ, ∀ᶠ k in atTop, ∃ q, IsDensityFamily (N k) (B k) (intervalT' (N k) (B k)) q ∧
      gTwoKR (N k) (B k) (intervalT' (N k) (B k)) q
          (fun t => Set.Ioc 0 (deltaCut (N k) (B k) t))
        ≤ ENNReal.ofReal (C * Real.sqrt (B k)) := by
  sorry

/-- **Corollary (cor:error-higher), second display, on `T'`:** `O(√β)`.

Not proved here.

Source: arXiv:2412.09080v3, `cor:error-higher`, "Moreover". -/
theorem error_higher_three_KR_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B) :
    ∃ C : ℝ, ∀ᶠ k in atTop, ∃ q, IsDensityFamily (N k) (B k) (intervalT' (N k) (B k)) q ∧
      gThreeKR (N k) (B k) (intervalT' (N k) (B k)) q (fun _ => Set.Ioi 0)
        ≤ ENNReal.ofReal (C * Real.sqrt (B k)) := by
  sorry

/-- The hypotheses of `cor:error-higher` are satisfiable. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

/-! ### The form used later: `y ≥ Δ_t` -/

/-- `x^{-1/2} ≥ 0` for every real `x`: for `x < 0`, Mathlib's `rpow` is
`e^{-log|x|/2} cos(-π/2) = 0`. -/
theorem rpow_neg_half_nonneg (x : ℝ) : 0 ≤ x ^ (-(1 : ℝ) / 2) := by
  rcases le_or_gt 0 x with hx | hx
  · exact Real.rpow_nonneg hx _
  · rw [Real.rpow_def_of_neg hx, show -(1 : ℝ) / 2 * π = -(π / 2) by ring, Real.cos_neg,
      Real.cos_pi_div_two, mul_zero]

/-- **The remark after `cor:error-higher`:** the second display holds with
`y ≥ Δ_t` in place of `y ≥ 0`, by non-negativity of the integrand.

Source: arXiv:2412.09080v3, the remark after the proof of `cor:error-higher`. -/
theorem gThreeKR_Ioi_le (n : ℕ) (β : ℝ) (S : Set ℝ) (q : ℝ → ℝ × ℝ → ℝ) :
    gThreeKR n β S q (fun t => Set.Ioi (deltaCut n β t))
      ≤ gThreeKR n β S q (fun _ => Set.Ioi 0) := by
  refine lintegral_mono fun t => ?_
  rw [← lintegral_indicator measurableSet_Ioi, ← lintegral_indicator measurableSet_Ioi]
  refine lintegral_mono fun y => ?_
  simp only [Set.indicator_apply, Set.mem_Ioi]
  split_ifs with h1 h2 h2
  · exact le_rfl
  · rw [ENNReal.ofReal_eq_zero.2]
    exact mul_nonpos_of_nonpos_of_nonneg
      (mul_nonpos_of_nonpos_of_nonneg (not_lt.1 h2) (rpow_neg_half_nonneg _)) (abs_nonneg _)
  · exact zero_le
  · exact le_rfl

/-! ### `eq:error-goal` -/

/-- **The continuous-family version of `eq:error-goal` is false.** The
allowed regime `β = n` lacks the asserted family, already when `ε = 1`.
Source: arXiv:2412.09080v3, §3, `eq:error-goal`; §4.1, `lem: pt.bdd`. -/
theorem not_error_goal (ω : ℝ → ℝ) :
    ¬ ∀ ε : ℝ, 0 < ε → ∀ᶠ k : ℕ in atTop, ∃ q,
      IsDensityFamily (k + 1) (k + 1) (intervalT (k + 1) (k + 1) (ω (k + 1))) q ∧
      gTwoKR (k + 1) (k + 1) (intervalT (k + 1) (k + 1) (ω (k + 1))) q (fun _ => Set.Ioi 0)
        ≤ ENNReal.ofReal (ε * Real.sqrt ((k + 1) * Real.log (k + 1))) := by
  intro h
  apply not_eventually_densityFamily_T_succ ω
  exact (h 1 one_pos).mono fun k ⟨q, hq, _⟩ => ⟨q, hq⟩

/-- The `β = n` counterexample satisfies the original growth assumptions
and the positive test tolerance used in the refutation. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) ∧ (0 : ℝ) < 1 :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log, one_pos⟩

end Modes
end Transformer
