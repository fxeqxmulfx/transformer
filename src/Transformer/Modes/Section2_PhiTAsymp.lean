/-
# The number of modes of a Gaussian KDE — `lem:phi-t`, the asymptotics

§2.2 of arXiv:2412.09080v3, `lem:phi-t`: the sizes of `A_t`, `α_t`, `δ_t` and
of the inner Kac-Rice integral over the Gaussian proxy, uniformly over `t ∈ T`.

**What the source says and what is carried here.**

* `≍` over `t ∈ T` is read uniformly: two constants `0 < C₁ ≤ C₂` and, for all
  large `k`, the two-sided bound at every `t ∈ T_k`, along a regime sequence
  `(n_k, β_k)`.  `A_t, α_t, δ_t` are the exact coefficients of
  `krQuad_zero_eq`, so the source's "there exist" is discharged by them.

* `A_t ≍ β^{-3/2} n t² e^{-t²/2}` and `α_t ≍ β^{1/2} e^{t²/2}`: as in the
  source.

* **`δ_t` is not carried as the source writes it.**  The source claims
  `δ_t ≍ n^{1/2}β^{-3/2}e^{-t²/2}(1 - t²/2)`.  The right side vanishes at
  `t = √2 ∈ T`; from the closed forms of §5.2,
  `δ_t = n^{1/2}β^{-3/2}e^{-t²/2}(1 - t²/2 + (-3/2 + 3t²/2 - t⁴/4)/β + O(β⁻²))`,
  which is `≈ n^{1/2}β^{-5/2}e^{-1}/2 > 0` there (checked numerically to seven
  digits), so the claim is false at `t = √2`.  The refutation needs the closed
  forms, which are unproved, and is not carried yet.

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

import Transformer.Modes.Section2_PhiT

open Real Filter Asymptotics MeasureTheory
open scoped Topology

namespace Transformer
namespace Modes

/-- **Lemma (lem:phi-t), `A_t ≍ β^{-3/2} n t² e^{-t²/2}`, uniformly on `T`.**

Not proved here.

Source: arXiv:2412.09080v3, `lem:phi-t`. -/
theorem phiA_isTheta {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      C₁ * (B k ^ (-(3 : ℝ) / 2) * N k * t ^ 2 * Real.exp (-(t ^ 2) / 2))
          ≤ phiA (N k) (B k) t ∧
        phiA (N k) (B k) t
          ≤ C₂ * (B k ^ (-(3 : ℝ) / 2) * N k * t ^ 2 * Real.exp (-(t ^ 2) / 2)) := by
  sorry

/-- **Lemma (lem:phi-t), `α_t ≍ β^{1/2} e^{t²/2}`, uniformly on `T`.**

Not proved here.

Source: arXiv:2412.09080v3, `lem:phi-t`. -/
theorem phiAlpha_isTheta {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      C₁ * (B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2)) ≤ phiAlpha (B k) t ∧
        phiAlpha (B k) t ≤ C₂ * (B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2)) := by
  sorry

/-- **Lemma (lem:phi-t), the check on `δ_t`, corrected.**  If moreover
`n ≲ β^{5/2}`, then `α_t δ_t² = O(1)` uniformly on `T`.  The source checks
`α_t^{-1}δ_t² ≪ 1` instead; see the module docstring.

Not proved here.

Source: arXiv:2412.09080v3, proof of `lem:phi-t`. -/
theorem phiAlpha_mul_phiDelta_sq_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω)
    (hN : (fun k => (N k : ℝ)) =O[atTop] fun k => B k ^ ((5 : ℝ) / 2)) :
    ∃ C : ℝ, ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2 ≤ C := by
  sorry

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
