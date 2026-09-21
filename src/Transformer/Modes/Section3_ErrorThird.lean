import Transformer.Modes.Section3_Edgeworth

/-
# The number of modes of a Gaussian KDE — the third-order error for large `y`

§3 and §3.1 of arXiv:2412.09080v3: why `thm:br` with `s = 2` does not suffice
(`eq:tilde-y-fail`), the bound `eq:H3-bound` on the Hermite polynomials along
the Kac–Rice line `x = 0`, and `lem:error-3`, the contribution of the Edgeworth
term `n^{-1/2}ψ` for `y ≥ Δ_t`.

**What the source says and what is carried here.**

* `eq:tilde-y-fail`: `∫_0^∞ ỹ/(1 + ỹ²) dỹ` diverges.  Proved,
  `not_integrableOn_tildeY`.

* **`eq:H3-bound`, "`|H^{(k,3-k)}(Σ_t^{-1/2}[(0,y) - μ_t])| ≲ ỹ³` for `y ≥ Δ_t`",
  fails for small `ỹ`**, as the cubic bound it rests on does
  (`not_exists_hermite_le_cube`).  At `t = 0` one has `𝔼G(0) = 0` (the
  integrand is odd), so `A_0 = 0`, `Δ_0 = δ_0`, the whitened point is
  `(0, ỹ)`, and `H^{(2,1)}(0, ỹ) = He₂(0) He₁(ỹ) = -ỹ`, which no `C ỹ³` bounds
  as `ỹ → 0⁺`.  What holds, for every positive definite `Σ` and every centre,
  is `ỹ ≤ ‖Σ^{-1/2}[(0,y) - μ]‖ ≤ 2ỹ` and `|H^{(k,3-k)}| ≤ 24(ỹ + ỹ³)` for
  `y ≥ Δ`: `eucl_whiten_bounds`, `abs_hermite_whiten_le`, both proved.  The
  extra `ỹ` changes the proof of `lem:error-3` only by the harmless
  `∫ ỹ² e^{-ỹ²/2}`.

* `lem:error-3` is stated with `|ψ|` in the integrand: the proof bounds
  `|ψ|`, and `eq:error-goal` needs the absolute value.  Both bounds are on the
  `ℝ≥0∞`-valued integral itself, so an infinite integral cannot satisfy them.

Source: arXiv:2412.09080v3, §3, `eq:tilde-y-fail`, `eq:H3-bound`, `lem:error-3`.
-/

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

/-- **Equation (eq:tilde-y-fail).**  `∫_0^∞ ỹ/(1 + ỹ²) dỹ` diverges: this is why
`thm:br` with `s = 2` cannot control the inner integral of `eq:error-goal`.

Source: arXiv:2412.09080v3, §3, `eq:tilde-y-fail`. -/
theorem not_integrableOn_tildeY :
    ¬ IntegrableOn (fun y : ℝ => y / (1 + y ^ 2)) (Set.Ioi 0) := by
  intro h
  have h1 : IntegrableOn (fun y : ℝ => y / (1 + y ^ 2)) (Set.Ioi 1) :=
    h.mono_set (Set.Ioi_subset_Ioi zero_le_one)
  refine not_integrableOn_Ioi_inv (a := 1) ((h1.const_mul 2).mono' ?_ ?_)
  · exact measurable_inv.aestronglyMeasurable
  · refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun y hy => ?_)
    have hy : (1 : ℝ) < y := hy
    have hy0 : 0 < y := by linarith
    rw [Real.norm_of_nonneg (inv_nonneg.2 hy0.le), inv_eq_one_div, div_le_iff₀ hy0]
    rw [mul_div_assoc', div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
    nlinarith

/-- **`ỹ ≍ ‖Σ^{-1/2}[(0,y) - μ]‖` for `y ≥ Δ`, exactly.**  For `Σ = [[a, b], [b, d]]`
positive definite, `μ = (m₁, m₂)`, `A = m₁²/a`, `α = a/(ad - b²)`,
`δ = m₂ - b m₁/a`, `Δ = δ + √(A/α)` and `ỹ = √α (y - δ)`: if `y ≥ Δ`, then
`ỹ ≤ ‖Σ^{-1/2}[(0,y) - μ]‖ ≤ 2ỹ`.

Source: arXiv:2412.09080v3, §3.1, before `eq:H3-bound`. -/
theorem eucl_whiten_bounds {a b d : ℝ} (ha : 0 < a) (hD : 0 < a * d - b ^ 2) (m₁ m₂ y : ℝ)
    (hy : m₂ - b * m₁ / a + Real.sqrt (m₁ ^ 2 / a / (a / (a * d - b ^ 2))) ≤ y) :
    Real.sqrt (a / (a * d - b ^ 2)) * (y - (m₂ - b * m₁ / a))
        ≤ eucl (whiten a b d (-m₁, y - m₂)) ∧
      eucl (whiten a b d (-m₁, y - m₂))
        ≤ 2 * (Real.sqrt (a / (a * d - b ^ 2)) * (y - (m₂ - b * m₁ / a))) := by
  set α := a / (a * d - b ^ 2) with hα
  set A := m₁ ^ 2 / a with hA
  set δ := m₂ - b * m₁ / a with hδ
  set u := Real.sqrt α * (y - δ) with hu
  have hα0 : 0 < α := div_pos ha hD
  have hA0 : 0 ≤ A := div_nonneg (sq_nonneg _) ha.le
  have he : eucl (whiten a b d (-m₁, y - m₂)) ^ 2 = A + u ^ 2 := by
    rw [eucl_whiten_sq ha hD, quadForm_complete_square ha.ne' hD.ne', hu, mul_pow,
      Real.sq_sqrt hα0.le]
  have hyδ : Real.sqrt (A / α) ≤ y - δ := by linarith
  have hu0 : Real.sqrt A ≤ u := by
    calc Real.sqrt A = Real.sqrt α * Real.sqrt (A / α) := by
          rw [← Real.sqrt_mul hα0.le, mul_div_cancel₀ _ hα0.ne']
      _ ≤ u := mul_le_mul_of_nonneg_left hyδ (Real.sqrt_nonneg _)
  have hu0' : 0 ≤ u := (Real.sqrt_nonneg _).trans hu0
  have hAu : A ≤ u ^ 2 := by
    calc A = Real.sqrt A ^ 2 := (Real.sq_sqrt hA0).symm
      _ ≤ u ^ 2 := pow_le_pow_left₀ (Real.sqrt_nonneg _) hu0 2
  have hE : 0 ≤ eucl (whiten a b d (-m₁, y - m₂)) := Real.sqrt_nonneg _
  constructor
  · nlinarith
  · nlinarith

/-- **Equation (eq:H3-bound), corrected.**  In the notation of
`eucl_whiten_bounds`, `|H^{(k,3-k)}(Σ^{-1/2}[(0,y) - μ])| ≤ 24(ỹ + ỹ³)` for
`y ≥ Δ`.  The source claims `≲ ỹ³`, which fails as `ỹ → 0⁺`; see the module
docstring.

Source: arXiv:2412.09080v3, §3.1, `eq:H3-bound`. -/
theorem abs_hermite_whiten_le {a b d : ℝ} (ha : 0 < a) (hD : 0 < a * d - b ^ 2) (m₁ m₂ y : ℝ)
    (hy : m₂ - b * m₁ / a + Real.sqrt (m₁ ^ 2 / a / (a / (a * d - b ^ 2))) ≤ y)
    (k : ℕ) (hk : k ≤ 3) :
    |hermite3 k (whiten a b d (-m₁, y - m₂))|
      ≤ 24 * (Real.sqrt (a / (a * d - b ^ 2)) * (y - (m₂ - b * m₁ / a))
        + (Real.sqrt (a / (a * d - b ^ 2)) * (y - (m₂ - b * m₁ / a))) ^ 3) := by
  obtain ⟨h1, h2⟩ := eucl_whiten_bounds ha hD m₁ m₂ y hy
  set e := eucl (whiten a b d (-m₁, y - m₂))
  set u := Real.sqrt (a / (a * d - b ^ 2)) * (y - (m₂ - b * m₁ / a))
  have he : 0 ≤ e := Real.sqrt_nonneg _
  have h3 : e ^ 3 ≤ (2 * u) ^ 3 := pow_le_pow_left₀ he h2 3
  have hu : 0 ≤ u := by linarith
  calc _ ≤ 3 * (e + e ^ 3) := abs_hermite_le k hk _
    _ ≤ 24 * (u + u ^ 3) := by nlinarith [pow_nonneg hu 3]

/-- The hypotheses of `eucl_whiten_bounds` and `abs_hermite_whiten_le` are
satisfiable: `Σ = I₂`, `μ = 0`, `y = 0`, `k = 0`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 * 1 - 0 ^ 2 ∧
    (0 : ℝ) - 0 * 0 / 1 + Real.sqrt (0 ^ 2 / 1 / (1 / (1 * 1 - 0 ^ 2))) ≤ 0 ∧ 0 ≤ 3 := by
  norm_num

/-! ### `lem:error-3` -/

/-- `Δ_t = δ_t + √(A_t/α_t)`, the cut in `y` of `lem:error-3`.
arXiv:2412.09080v3, `lem:error-3`. -/
noncomputable def deltaCut (n : ℕ) (β t : ℝ) : ℝ :=
  phiDelta n β t + Real.sqrt (phiA n β t / phiAlpha β t)

/-- `∫_S ∫_{Δ_t}^∞ y (n det Σ_t)^{-1/2} |ψ|(Σ_t^{-1/2}[(0,y) - μ_t]) dy dt`, the
contribution of the Edgeworth term `n^{-1/2}ψ` to the Kac–Rice integral for
large `y`.  arXiv:2412.09080v3, `lem:error-3`. -/
noncomputable def psiKR (n : ℕ) (β : ℝ) (S : Set ℝ) : ℝ≥0∞ :=
  ∫⁻ t in S, ∫⁻ y in Set.Ioi (deltaCut n β t),
    ENNReal.ofReal (y * ((n : ℝ) * sigmaDet β t) ^ (-(1 : ℝ) / 2) *
      |psiOf (lawY β t)
        (whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t) (-muFst n β t, y - muSnd n β t))|)

/-- **Lemma (lem:error-3), on `T`.**  In the regime `n^c ≲ β ≲ n^{2-c}`,
`∫_T ∫_{Δ_t}^∞ y (n det Σ_t)^{-1/2} |ψ|(…) dy dt ≲ e^{-ω(β)/4} √(β log β)`.

Not proved here.

Source: arXiv:2412.09080v3, `lem:error-3`. -/
theorem error_three_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, ∀ᶠ k in atTop,
      psiKR (N k) (B k) (intervalT (N k) (B k) (ω (B k)))
        ≤ ENNReal.ofReal (C * (Real.exp (-(ω (B k)) / 4) * Real.sqrt (B k * Real.log (B k)))) := by
  sorry

/-- **Lemma (lem:error-3), on `T'`.**  In the regime `n^c ≲ β ≲ n^{2-c}`,
`∫_{T'} ∫_{Δ_t}^∞ y (n det Σ_t)^{-1/2} |ψ|(…) dy dt ≲ √β`.

Not proved here.

Source: arXiv:2412.09080v3, `lem:error-3`. -/
theorem error_three_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B) :
    ∃ C : ℝ, ∀ᶠ k in atTop,
      psiKR (N k) (B k) (intervalT' (N k) (B k)) ≤ ENNReal.ofReal (C * Real.sqrt (B k)) := by
  sorry

/-- The hypotheses of `error_three_T` and `error_three_T'` are satisfiable. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

end Modes
end Transformer
