import Transformer.Modes.Section3_Cumulants
import Mathlib.MeasureTheory.Measure.Haar.NormedSpace

/-
# The cited Edgeworth condition and the Gaussian reference law

§3 of arXiv:2412.09080v3 (`sec:error`) cites Bhattacharya–Rao, *Normal
Approximation and Asymptotic Expansions*, Theorems 19.2–19.3, as `thm:br`: for
i.i.d. `X₁, X₂, …` in `ℝ^k` with mean `0`, covariance `I` and
`𝔼‖X₁‖^{s+1} < ∞`, "under suitable conditions", the density `q_n` of
`n^{-1/2}(X₁ + ⋯ + Xₙ)` satisfies
`sup_x (1 + ‖x‖^s) |q_n(x) - Σ_{j=0}^{s-2} n^{-j/2} Q_j(x)| ≲ n^{-(s-1)/2}`.

**What the source says and what is carried here.**

* The two instances `k = 2`, `s = 2, 3` (`edgeworth_two`,
  `edgeworth_three`) are in `Section3_BREstimates`, after the Fourier
  comparison modules. This foundation defines their characteristic-power
  condition and verifies the Gaussian reference law.

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


/-- The actual Gaussian exponential expectation equals `exp(-|ξ|²/2)`.
Source: arXiv:2412.09080v3, §3 `thm:br`, the reference term `Q₀ = φ`. -/
theorem charFun_stdGauss2_eq_exp (ξ : ℝ × ℝ) :
    (∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2) =
      ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ) := by
  have h := integral_prod_mul (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)
    (fun a => Complex.exp (ξ.1 * a * Complex.I)) (fun a => Complex.exp (ξ.2 * a * Complex.I))
  have e : (fun x : ℝ × ℝ => Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I)) =
      fun x => Complex.exp (ξ.1 * x.1 * Complex.I) * Complex.exp (ξ.2 * x.2 * Complex.I) := by
    funext x
    rw [← Complex.exp_add]
    push_cast
    ring_nf
  rw [e, h, ← charFun_apply_real, ← charFun_apply_real, charFun_gaussianReal,
    charFun_gaussianReal, ← Complex.exp_add, Complex.ofReal_exp]
  congr 1
  push_cast
  simp only [mul_zero, one_mul, zero_mul, zero_sub]
  ring

/-- Every Gaussian characteristic norm power has its exact separable exponent.
Source: arXiv:2412.09080v3, §3 `thm:br`, the integrable-power condition. -/
theorem norm_charFun_stdGauss2_pow (ξ : ℝ × ℝ) (ν : ℕ) :
    ‖∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2‖ ^ ν =
      Real.exp (-((ν : ℝ) / 2) * ξ.1 ^ 2) * Real.exp (-((ν : ℝ) / 2) * ξ.2 ^ 2) := by
  rw [norm_charFun_stdGauss2, mul_pow, ← Real.exp_nat_mul, ← Real.exp_nat_mul]
  congr 1 <;> congr 1 <;> ring

/-- The actual Gaussian characteristic norm power is integrable for every
positive integer power, not just the witness `ν = 1`. Source:
arXiv:2412.09080v3, §3 `thm:br`, its suitable characteristic-power condition. -/
theorem integrable_charFun_stdGauss2_pow {ν : ℕ} (hν : 1 ≤ ν) :
    Integrable (fun ξ : ℝ × ℝ =>
      ‖∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2‖ ^ ν) := by
  have hν0 : 0 < (ν : ℝ) := by exact_mod_cast (show 0 < ν by omega)
  have he : (fun ξ : ℝ × ℝ =>
      ‖∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2‖ ^ ν) =
      fun ξ => Real.exp (-((ν : ℝ) / 2) * ξ.1 ^ 2) * Real.exp (-((ν : ℝ) / 2) * ξ.2 ^ 2) :=
    funext fun ξ => norm_charFun_stdGauss2_pow ξ ν
  rw [he, Measure.volume_eq_prod]
  exact (integrable_exp_neg_mul_sq (by positivity : (0 : ℝ) < (ν : ℝ) / 2)).mul_prod
    (integrable_exp_neg_mul_sq (by positivity : (0 : ℝ) < (ν : ℝ) / 2))

/-- The complex Gaussian exponential expectation itself is integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly`, the Gaussian Fourier term. -/
theorem integrable_charFun_stdGauss2 : Integrable (fun ξ : ℝ × ℝ =>
    ∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2) := by
  have hC : Continuous (fun ξ : ℝ × ℝ =>
      ∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2) := by
    have he : (fun ξ : ℝ × ℝ =>
        ∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2) =
        fun ξ => ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ) := funext charFun_stdGauss2_eq_exp
    rw [he]
    fun_prop
  apply (integrable_norm_iff hC.aestronglyMeasurable).mp
  simpa only [pow_one] using integrable_charFun_stdGauss2_pow (ν := 1) le_rfl

example : 1 ≤ (2 : ℕ) := by norm_num

/-- Nonzero frequency scaling preserves integrability of every positive
Gaussian characteristic norm power. Source: arXiv:2412.09080v3, §5.4,
the frequency scaling in `eq:big-phi-poly`. -/
theorem integrable_charFun_stdGauss2_pow_scaled {ν : ℕ} (hν : 1 ≤ ν) {c : ℝ} (hc : c ≠ 0) :
    Integrable (fun ξ : ℝ × ℝ =>
      ‖∫ x, Complex.exp (((((c • ξ).1) * x.1 + ((c • ξ).2) * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2‖ ^ ν) := by
  exact (integrable_charFun_stdGauss2_pow hν).comp_smul hc

example : 1 ≤ (2 : ℕ) ∧ (2 * Real.pi : ℝ) ≠ 0 := ⟨by norm_num, by positivity⟩

/-- Positive Gaussian frequency scaling has the exact two-dimensional
Jacobian `c⁻²`. Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`.
The identity uses the Bochner convention for all powers; density inversion
uses the positive, proved integrable powers above. -/
theorem integral_charFun_stdGauss2_pow_scaled (ν : ℕ) (c : ℝ) (hc : 0 < c) :
    (∫ ξ : ℝ × ℝ,
      ‖∫ x, Complex.exp (((((c • ξ).1) * x.1 + ((c • ξ).2) * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2‖ ^ ν) =
      (c ^ 2)⁻¹ * ∫ ξ : ℝ × ℝ,
        ‖∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2‖ ^ ν := by
  have h := Measure.integral_comp_smul_of_nonneg volume
    (fun ξ : ℝ × ℝ =>
      ‖∫ x, Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I) ∂stdGauss2‖ ^ ν)
    c (hR := hc.le)
  rw [h]
  simp [Module.finrank_prod, smul_eq_mul]

example : (0 : ℝ) < 2 * Real.pi := by positivity

end Modes
end Transformer
