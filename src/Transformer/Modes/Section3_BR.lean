import Transformer.Modes.Section3_Cumulants

/-
# The number of modes of a Gaussian KDE — the cited Edgeworth theorem

§3 of arXiv:2412.09080v3 (`sec:error`) cites Bhattacharya–Rao, *Normal
Approximation and Asymptotic Expansions*, Theorems 19.2–19.3, as `thm:br`: for
i.i.d. `X₁, X₂, …` in `ℝ^k` with mean `0`, covariance `I` and
`𝔼‖X₁‖^{s+1} < ∞`, "under suitable conditions", the density `q_n` of
`n^{-1/2}(X₁ + ⋯ + Xₙ)` satisfies
`sup_x (1 + ‖x‖^s) |q_n(x) - Σ_{j=0}^{s-2} n^{-j/2} Q_j(x)| ≲ n^{-(s-1)/2}`.

**What the source says and what is carried here.**

* Only the two instances the paper uses are stated: `k = 2` and `s = 2, 3`
  (`edgeworth_two`, `edgeworth_three`), with `Q₀ = φ` and `Q₁ = ψ` of `eq:psi`.

* The "suitable conditions" are the ones of Bhattacharya–Rao, Theorem 19.2:
  `|𝔼 e^{i⟨ξ, X₁⟩}|^ν` is integrable for some `ν ≥ 1` (`HasIntegrableCharFun`).
  Under it `q_n` exists and is continuous for `n ≥ ν`; the statement is made
  for every continuous density, which then coincides with that one.

* `𝔼‖X₁‖^{s+1} < ∞` is `MemLp id (s+1) μ`; all norms of `ℝ²` being
  equivalent, the sup norm of `ℝ × ℝ` gives the same condition.

* For `s = 3` the cumulants in `ψ` are the derivatives of `log 𝔼 e^{⟨u, X⟩}`
  (`cumulantOf`), as the paper defines them, so exponential moments are
  assumed (`HasExpMoments`).  The law the paper applies it to, `Y(t)` of
  `eq:Yi`, is bounded.

Source: arXiv:2412.09080v3, §3, `thm:br`; Bhattacharya–Rao, Theorems 19.2–19.3.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

/-- Some power `|𝔼 e^{i⟨ξ, X⟩}|^ν`, `ν ≥ 1`, of the characteristic function of
`μ` is integrable: the condition under which Bhattacharya–Rao, Theorem 19.2,
cited as `thm:br` in arXiv:2412.09080v3, §3, gives a density expansion. -/
def HasIntegrableCharFun (μ : Measure (ℝ × ℝ)) : Prop :=
  ∃ ν : ℕ, 1 ≤ ν ∧ Integrable (fun ξ : ℝ × ℝ =>
    ‖∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂μ‖ ^ ν)

/-- `thm:br` for `s = 2`: `sup_x (1 + ‖x‖²) |q_n(x) - φ(x)| ≲ n^{-1/2}`.
arXiv:2412.09080v3, §3, `thm:br` (Bhattacharya–Rao, Theorems 19.2–19.3). -/
theorem edgeworth_two (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C, ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ, Continuous q →
      IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, (1 + eucl x ^ 2) * |q x - phi2 x| ≤ C * (n : ℝ) ^ (-(1 : ℝ) / 2) := by
  sorry

/-- `thm:br` for `s = 3`:
`sup_x (1 + ‖x‖³) |q_n(x) - φ(x) - n^{-1/2} ψ(x)| ≲ n^{-1}`.
arXiv:2412.09080v3, §3, `thm:br` (Bhattacharya–Rao, Theorems 19.2–19.3), with
`ψ` of `eq:psi`; exponential moments are assumed so that the cumulants in `ψ`
are those the paper defines (see the module docstring). -/
theorem edgeworth_three (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ)
    (hcf : HasIntegrableCharFun μ) :
    ∃ C, ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ, Continuous q →
      IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, (1 + eucl x ^ 3) * |q x - phi2 x - (Real.sqrt n)⁻¹ * psiOf μ x|
        ≤ C * (n : ℝ)⁻¹ := by
  sorry

/-- The characteristic function of `N(0, I₂)` has modulus
`e^{-ξ₁²/2} e^{-ξ₂²/2}`.  arXiv:2412.09080v3, §3.1. -/
theorem norm_charFun_stdGauss2 (ξ : ℝ × ℝ) :
    ‖∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2‖
      = Real.exp (-(1 / 2) * ξ.1 ^ 2) * Real.exp (-(1 / 2) * ξ.2 ^ 2) := by
  have h := integral_prod_mul (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)
    (fun a => Complex.exp (ξ.1 * a * Complex.I)) (fun a => Complex.exp (ξ.2 * a * Complex.I))
  have e : (fun x : ℝ × ℝ => Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I))
      = fun x => Complex.exp (ξ.1 * x.1 * Complex.I) * Complex.exp (ξ.2 * x.2 * Complex.I) := by
    funext x; rw [← Complex.exp_add]; push_cast; ring_nf
  rw [e, h, ← charFun_apply_real, ← charFun_apply_real, charFun_gaussianReal,
    charFun_gaussianReal, norm_mul, Complex.norm_exp, Complex.norm_exp]
  simp [← Complex.ofReal_pow]
  ring_nf

/-- `N(0, I₂)` satisfies the condition of `thm:br` with `ν = 1`.
arXiv:2412.09080v3, §3, `thm:br`. -/
theorem hasIntegrableCharFun_stdGauss2 : HasIntegrableCharFun stdGauss2 := by
  refine ⟨1, le_rfl, ?_⟩
  simp only [pow_one, norm_charFun_stdGauss2]
  rw [Measure.volume_eq_prod]
  exact (integrable_exp_neg_mul_sq (by norm_num : (0 : ℝ) < 1 / 2)).mul_prod
    (integrable_exp_neg_mul_sq (by norm_num : (0 : ℝ) < 1 / 2))

/-- The hypotheses of `edgeworth_two` are satisfiable. -/
example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasIntegrableCharFun_stdGauss2⟩

/-- The hypotheses of `edgeworth_three` are satisfiable. -/
example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

end Modes
end Transformer
