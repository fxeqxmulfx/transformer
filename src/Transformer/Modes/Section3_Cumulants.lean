import Transformer.Modes.Section3_Hermite
import Mathlib.Probability.Distributions.Gaussian.Fernique

/-
# The number of modes of a Gaussian KDE — cumulants and the Edgeworth term

§3.1 of arXiv:2412.09080v3 (`sec:error-3`): the cumulants `κ^α` of a random
vector of `ℝ²`, the third Edgeworth term `ψ` (`eq:psi`), and the moment
identity for `|α| = 3`.  Everything here is stated for an arbitrary law `μ` on
`ℝ²`; §3 applies it to `μ = Law(Y(t))` of `eq:Yi`, which is bounded (`G` and
`G'` are), so every hypothesis on exponential moments below holds there.

**What the source says and what is carried here.**

* A density is a continuous function: "`q` is the density of `S`" is
  `IsDensityOf`, nonnegativity together with `Law(S) = q · Lebesgue`.

* `κ^α` is "the `α`-th mixed derivative at `0` of `log 𝔼 e^{⟨u, Y⟩}`":
  `cumulantOf`, with the partial derivatives taken one variable at a time.

* **The moment identity is off by `√n`.**  The source writes
  `κ_t^α = 𝔼[H^α(Y(t))] = 𝔼_{Z ~ N(0,I₂)}[(q_t/φ)(Z) H^α(Z)]` with `q_t` the
  density of `n^{-1/2} Σ Yᵢ(t)`.  The last expectation is `𝔼[H^α(S_n)]`, the
  third cumulant of the normalized sum, which is `n^{-1/2} κ_t^α`, not `κ_t^α`
  (cumulants are additive over independent summands and homogeneous of
  degree three).  `cumulant_three_eq_integral_hermite` carries the first
  equality as written; `integral_density_mul_hermite` carries the second with
  the factor `n^{-1/2}` restored.

Source: arXiv:2412.09080v3, §3.1, `eq:psi` and the display after it.
-/

open Real MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace Transformer
namespace Modes

/-- `q` is a density of the law of `S` under `P`: it is nonnegative and
`Law(S) = q · Lebesgue`.  arXiv:2412.09080v3, §3 ("the density `q_t` of
`n^{-1/2} Σ Yᵢ(t)`"). -/
structure IsDensityOf {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) (S : Ω → ℝ × ℝ)
    (q : ℝ × ℝ → ℝ) : Prop where
  nonneg : ∀ x, 0 ≤ q x
  map_eq : P.map S = volume.withDensity (fun x => ENNReal.ofReal (q x))

/-- `μ` is standardized: square integrable, mean `0`, covariance `I₂`.
arXiv:2412.09080v3, §3 ("by construction `q_t` has mean `0` and covariance
`I₂`"). -/
structure IsStandardized (μ : Measure (ℝ × ℝ)) : Prop where
  memLp : MemLp id 2 μ
  mean_fst : ∫ x, x.1 ∂μ = 0
  mean_snd : ∫ x, x.2 ∂μ = 0
  var_fst : ∫ x, x.1 ^ 2 ∂μ = 1
  var_snd : ∫ x, x.2 ^ 2 ∂μ = 1
  cov : ∫ x, x.1 * x.2 ∂μ = 0

/-- The moment generating function of `μ` is finite near `0`, so that the
cumulant generating function `log 𝔼 e^{⟨u, Y⟩}` of arXiv:2412.09080v3, §3.1, is
defined and smooth there. -/
def HasExpMoments (μ : Measure (ℝ × ℝ)) : Prop :=
  ∃ ε > 0, ∀ u v : ℝ, |u| < ε → |v| < ε →
    Integrable (fun x : ℝ × ℝ => Real.exp (u * x.1 + v * x.2)) μ

/-- The cumulant `κ^{(a,b)}` of `μ`: the mixed derivative `∂_u^a ∂_v^b` at `0`
of `log ∫ e^{u x₁ + v x₂} dμ`.  arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
noncomputable def cumulantOf (μ : Measure (ℝ × ℝ)) (a b : ℕ) : ℝ :=
  iteratedDeriv a (fun u => iteratedDeriv b
    (fun v => Real.log (∫ x, Real.exp (u * x.1 + v * x.2) ∂μ)) 0) 0

/-- The third Edgeworth term
`ψ(x) = φ(x) Σ_{k=0}^{3} κ^{(k,3-k)} / (k!(3-k)!) · H^{(k,3-k)}(x)`.
arXiv:2412.09080v3, `eq:psi`. -/
noncomputable def psiOf (μ : Measure (ℝ × ℝ)) (x : ℝ × ℝ) : ℝ :=
  phi2 x * ∑ k ∈ Finset.range 4,
    cumulantOf μ k (3 - k) / ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * hermite3 k x

/-- The normalized sum `n^{-1/2} (X₁ + ⋯ + Xₙ)`.  arXiv:2412.09080v3, §3. -/
noncomputable def scaledSum (n : ℕ) (X : Fin n → ℝ × ℝ) : ℝ × ℝ :=
  (Real.sqrt n)⁻¹ • ∑ i, X i

/-- The second equality of the moment identity, corrected: if `q` is the
density of `n^{-1/2} Σ Xᵢ` for `X₁, …, Xₙ` i.i.d. of law `μ`, then
`𝔼_{Z ~ N(0,I₂)}[(q/φ)(Z) H^α(Z)] = ∫ q H^α = n^{-1/2} κ^α` for `|α| = 3`.
arXiv:2412.09080v3, §3.1, the display after `eq:psi`, which states it without
the factor `n^{-1/2}` (see the module docstring). -/
theorem integral_density_mul_hermite (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hexp : HasExpMoments μ) (n : ℕ) (hn : 1 ≤ n)
    (q : ℝ × ℝ → ℝ) (hq : IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q)
    (k : ℕ) (hk : k ≤ 3) :
    ∫ x, q x * hermite3 k x = (Real.sqrt n)⁻¹ * cumulantOf μ k (3 - k) := by
  sorry

/-- The standard normal law `N(0, I₂)` on `ℝ²`, whose density is `φ`.
arXiv:2412.09080v3, §3.1. -/
noncomputable abbrev stdGauss2 : Measure (ℝ × ℝ) := (gaussianReal 0 1).prod (gaussianReal 0 1)

/-- `𝔼 Z² = 1` for `Z ~ N(0,1)`; the variance of the standard normal in
arXiv:2412.09080v3, §3.1. -/
theorem integral_sq_stdGaussian : ∫ x, x ^ 2 ∂gaussianReal 0 1 = 1 := by
  have h := variance_id_gaussianReal (μ := 0) (v := 1)
  rw [variance_of_integral_eq_zero (memLp_id_gaussianReal 2).aemeasurable (by simp)] at h
  simpa using h

/-- `N(0, I₂)` is standardized.  arXiv:2412.09080v3, §3.1. -/
theorem isStandardized_stdGauss2 : IsStandardized stdGauss2 where
  memLp := IsGaussian.memLp_id _ _ (by simp)
  mean_fst := by
    simpa using integral_prod_mul (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)
      (fun a => a) (fun _ => (1 : ℝ))
  mean_snd := by
    simpa using integral_prod_mul (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)
      (fun _ => (1 : ℝ)) (fun a => a)
  var_fst := by
    have := integral_prod_mul (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)
      (fun a => a ^ 2) (fun _ => (1 : ℝ))
    simp only [mul_one, integral_const, smul_eq_mul, probReal_univ] at this
    rw [this, integral_sq_stdGaussian]
  var_snd := by
    have := integral_prod_mul (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)
      (fun _ => (1 : ℝ)) (fun a => a ^ 2)
    simp only [one_mul, integral_const, smul_eq_mul, probReal_univ] at this
    rw [this, integral_sq_stdGaussian]
  cov := by
    simpa using integral_prod_mul (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)
      (fun a => a) (fun a => a)

/-- `N(0, I₂)` has every exponential moment.  arXiv:2412.09080v3, §3.1. -/
theorem hasExpMoments_stdGauss2 : HasExpMoments stdGauss2 := by
  refine ⟨1, one_pos, fun u v _ _ => ?_⟩
  simpa [Real.exp_add] using
    (integrable_exp_mul_gaussianReal (μ := 0) (v := 1) u).mul_prod
      (integrable_exp_mul_gaussianReal (μ := 0) (v := 1) v)

/-- For a single summand the normalized sum of `N(0, I₂)` has density `φ`.
arXiv:2412.09080v3, §3.1. -/
theorem isDensityOf_stdGauss2 :
    IsDensityOf (Measure.pi fun _ : Fin 1 => stdGauss2) (scaledSum 1) phi2 := by
  refine ⟨fun x => by unfold phi2; positivity, ?_⟩
  have hS : scaledSum 1 = Function.eval 0 := by
    funext X; simp [scaledSum]
  rw [hS, (measurePreserving_eval (μ := fun _ : Fin 1 => stdGauss2) 0).map_eq, stdGauss2,
    gaussianReal_of_var_ne_zero _ one_ne_zero,
    prod_withDensity (measurable_gaussianPDF _ _) (measurable_gaussianPDF _ _),
    ← Measure.volume_eq_prod]
  congr 1
  funext x
  rw [gaussianPDF, gaussianPDF, ← ENNReal.ofReal_mul (gaussianPDFReal_nonneg _ _ _)]
  congr 1
  simp only [gaussianPDFReal, phi2, NNReal.coe_one, mul_one, sub_zero]
  rw [mul_mul_mul_comm, ← Real.exp_add, ← mul_inv, ← Real.sqrt_mul (by positivity),
    Real.sqrt_mul_self (by positivity)]
  ring_nf

/-- The hypotheses of `integral_density_mul_hermite` are satisfiable. -/
example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ 1 ≤ 1 ∧
    IsDensityOf (Measure.pi fun _ : Fin 1 => stdGauss2) (scaledSum 1) phi2 ∧ (0 : ℕ) ≤ 3 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2, le_rfl,
    isDensityOf_stdGauss2, by norm_num⟩

end Modes
end Transformer
