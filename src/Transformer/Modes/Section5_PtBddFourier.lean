import Transformer.Modes.Section5_PtBdd
import Transformer.Modes.Section3_Cumulants
import Mathlib.Analysis.SpecialFunctions.JapaneseBracket
import Mathlib.Analysis.SpecialFunctions.Gaussian.PoissonSummation

/-
# The number of modes of a Gaussian KDE — the Fourier transform of `ν_t`

§5.5 of arXiv:2412.09080v3, `sec: proof.pt.bdd`: the decay of `𝓕ν_t`, where
`ν_t` is the law of `(G(t), G'(t))`, the integrability of `|𝓕ν_t|ⁿ` it gives
for `n > 4`.

**What the source says and what is carried here.**

* `eq:uniform-decay`, `|𝓕ν_t(ξ)| ≲ (1 + ‖ξ‖)^{-1/2}` "where the implicit
  constant depends only on `β`", is stated for each fixed `t`, with a constant
  depending on `β` and `t`: `uniform_decay`.  The source reduces to `t = 0`
  "by translation invariance of the standard Gaussian", but translating
  `Z ~ N(t, 1)` moves the amplitude `e^{-x²/2}`, not the phase `φ_θ`, so the
  reduction does not hold; and no constant is uniform in `t`, since
  `ν_t → δ₀` as `t → ∞` and `𝓕δ₀ ≡ 1`.  That is proved, `not_uniform_decay`.
  Only a fixed `t` is used afterwards.

* "`∫|𝓕ν_t|ⁿ ≲ ∫_{‖ξ‖≤1} 1 + ∫_{‖ξ‖>1} |ξ|^{-n/2}`, which is finite as long
  as `n > 4`" is proved, for any function with the decay of
  `eq:uniform-decay`: `lintegral_pow_lt_top_of_decay`.

Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology

namespace Transformer
namespace Modes

/-- `𝓕ν_t(ξ) = 𝔼 e^{-i(ξ₁ G(t) + ξ₂ G'(t))}`, the Fourier transform of the law
`ν_t` of `(G(t), G'(t))`.  arXiv:2412.09080v3, §5.5. -/
noncomputable def fourierNu (β t : ℝ) (ξ : ℝ × ℝ) : ℂ :=
  ∫ x, Complex.exp (-(Complex.I * ((ξ.1 * bigG β t x + ξ.2 * bigG' β t x : ℝ) : ℂ)))
    ∂gaussianReal 0 1

/-- **Equation (eq:uniform-decay)**, for a fixed `t`:
`|𝓕ν_t(ξ)| ≲ (1 + ‖ξ‖)^{-1/2}` for all `ξ ∈ ℝ²`.

Not proved here.  The constant depends on `t`; see the module docstring and
`not_uniform_decay`.

Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`. -/
theorem uniform_decay {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    ∃ C : ℝ, ∀ ξ : ℝ × ℝ, ‖fourierNu β t ξ‖ ≤ C / Real.sqrt (1 + ‖ξ‖) := by
  sorry

/-! ### The constant of `eq:uniform-decay` cannot be uniform in `t` -/

/-- `|u|ᵏ e^{-βu²/2} → 0` as `u → ∞`. -/
theorem tendsto_pow_mul_gaussFactor {β : ℝ} (hβ : 0 < β) (k : ℕ) :
    Tendsto (fun u : ℝ => u ^ k * Real.exp (-(β / 2) * u ^ 2)) atTop (𝓝 0) := by
  have h := (tendsto_rpow_abs_mul_exp_neg_mul_sq_cocompact (half_pos hβ) k).mono_left
    atTop_le_cocompact
  refine tendsto_zero_iff_norm_tendsto_zero.2 (h.congr fun u => ?_)
  rw [Real.rpow_natCast, norm_mul, norm_pow, Real.norm_eq_abs,
    Real.norm_of_nonneg (Real.exp_pos _).le]

/-- `(G(t), G'(t)) → 0` as `t → ∞`, for every sample point. -/
theorem tendsto_bigG {β : ℝ} (hβ : 0 < β) (x : ℝ) :
    Tendsto (fun t => bigG β t x) atTop (𝓝 0) ∧
      Tendsto (fun t => bigG' β t x) atTop (𝓝 0) := by
  have hu : Tendsto (fun t : ℝ => t - x) atTop atTop :=
    tendsto_atTop_add_const_right _ _ tendsto_id
  have h0 := (tendsto_pow_mul_gaussFactor hβ 0).comp hu
  have h1 := (tendsto_pow_mul_gaussFactor hβ 1).comp hu
  have h2 := (tendsto_pow_mul_gaussFactor hβ 2).comp hu
  refine ⟨h1.congr fun t => ?_, ?_⟩
  · simp only [Function.comp, bigG]; ring
  · have h := h0.sub (h2.const_mul β)
    rw [mul_zero, sub_zero] at h
    refine h.congr fun t => ?_
    simp only [Function.comp, bigG']; ring

/-- `𝓕ν_t(ξ) → 1` as `t → ∞`, by dominated convergence: `ν_t → δ₀`. -/
theorem tendsto_fourierNu {β : ℝ} (hβ : 0 < β) (ξ : ℝ × ℝ) :
    Tendsto (fun t => fourierNu β t ξ) atTop (𝓝 1) := by
  have hc : Continuous fun r : ℝ => Complex.exp (-(Complex.I * (r : ℂ))) := by fun_prop
  have h := tendsto_integral_filter_of_dominated_convergence (μ := gaussianReal 0 1)
    (l := atTop) (f := fun _ => (1 : ℂ)) (fun _ => (1 : ℝ))
    (F := fun t x => Complex.exp (-(Complex.I * ((ξ.1 * bigG β t x + ξ.2 * bigG' β t x : ℝ) : ℂ))))
    (Eventually.of_forall fun t => (hc.comp (by unfold bigG bigG'; fun_prop)).aestronglyMeasurable)
    (Eventually.of_forall fun t => ae_of_all _ fun x => by simp [Complex.norm_exp])
    (integrable_const _)
    (ae_of_all _ fun x => by
      have hx := ((tendsto_bigG hβ x).1.const_mul ξ.1).add ((tendsto_bigG hβ x).2.const_mul ξ.2)
      rw [mul_zero, mul_zero, add_zero] at hx
      have h0 := (hc.tendsto 0).comp hx
      simp only [Complex.ofReal_zero, mul_zero, neg_zero, Complex.exp_zero] at h0
      exact h0)
  simpa [fourierNu] using h

/-- **`eq:uniform-decay` is false with a constant depending only on `β`.**
No `C` bounds `|𝓕ν_t(ξ)| ≤ C (1 + ‖ξ‖)^{-1/2}` for all `t` and `ξ`: at
`ξ = (C², 0)` the bound is below `1`, while `𝓕ν_t(ξ) → 1` as `t → ∞`.

Source: arXiv:2412.09080v3, §5.5, the sentence after `eq:uniform-decay`. -/
theorem not_uniform_decay {β : ℝ} (hβ : 0 < β) :
    ¬ ∃ C : ℝ, ∀ t : ℝ, ∀ ξ : ℝ × ℝ, ‖fourierNu β t ξ‖ ≤ C / Real.sqrt (1 + ‖ξ‖) := by
  rintro ⟨C, hC⟩
  set ξ : ℝ × ℝ := (C ^ 2, 0)
  have hξ : C ^ 2 ≤ ‖ξ‖ := by
    have := norm_fst_le ξ
    rwa [Real.norm_of_nonneg (sq_nonneg C)] at this
  have hlt : C / Real.sqrt (1 + ‖ξ‖) < 1 := by
    have hs : 0 < Real.sqrt (1 + ‖ξ‖) := Real.sqrt_pos.2 (by positivity)
    rcases le_or_gt C 0 with hC0 | hC0
    · exact (div_nonpos_of_nonpos_of_nonneg hC0 hs.le).trans_lt one_pos
    · rw [div_lt_one hs, Real.lt_sqrt hC0.le]
      linarith
  have h := (tendsto_fourierNu hβ ξ).norm
  rw [norm_one] at h
  exact hlt.not_ge (le_of_tendsto' h fun t => hC t ξ)

/-- The hypothesis of `uniform_decay`, `tendsto_fourierNu` and
`not_uniform_decay` is satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

/-! ### Integrability of `|𝓕ν_t|ⁿ` -/

/-- **`|𝓕ν_t|ⁿ ∈ L¹(ℝ²)` for `n > 4`**, from the decay `eq:uniform-decay`
alone: `∫(1 + ‖ξ‖)^{-n/2} dξ < ∞` on `ℝ²` once `n/2 > 2`.

Source: arXiv:2412.09080v3, §5.5, the display after `eq:uniform-decay`. -/
theorem lintegral_pow_lt_top_of_decay {F : ℝ × ℝ → ℂ} {C : ℝ}
    (hF : ∀ ξ, ‖F ξ‖ ≤ C / Real.sqrt (1 + ‖ξ‖)) {n : ℕ} (hn : 5 ≤ n) :
    ∫⁻ ξ, ‖F ξ‖ₑ ^ n < ∞ := by
  have hint : Integrable (fun ξ : ℝ × ℝ => C ^ n * (1 + ‖ξ‖) ^ (-((n : ℝ) / 2))) volume := by
    refine (integrable_one_add_norm ?_).const_mul _
    have : (5 : ℝ) ≤ n := by exact_mod_cast hn
    simp only [Module.finrank_prod, Module.finrank_self]
    push_cast
    linarith
  refine lt_of_le_of_lt (lintegral_mono fun ξ => ?_) hint.lintegral_lt_top
  have ha : 0 < 1 + ‖ξ‖ := by positivity
  have hpow : ‖F ξ‖ ^ n ≤ C ^ n * (1 + ‖ξ‖) ^ (-((n : ℝ) / 2)) := by
    refine (pow_le_pow_left₀ (norm_nonneg _) (hF ξ) n).trans_eq ?_
    rw [div_pow, Real.sqrt_eq_rpow, ← Real.rpow_mul_natCast ha.le,
      Real.rpow_neg ha.le, div_eq_mul_inv]
    ring_nf
  rw [← ofReal_norm, ← ENNReal.ofReal_pow (norm_nonneg _)]
  exact ENNReal.ofReal_le_ofReal hpow

/-- The hypothesis of `lintegral_pow_lt_top_of_decay` is satisfiable. -/
example : ∀ ξ : ℝ × ℝ, ‖(0 : ℂ)‖ ≤ 0 / Real.sqrt (1 + ‖ξ‖) := fun _ => by simp

end Modes
end Transformer
