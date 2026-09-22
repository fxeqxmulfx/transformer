import Transformer.Modes.Section3_MixedMoments
import Transformer.Modes.Section3_LogDeriv

/-
# The number of modes of a Gaussian KDE — third cumulants are third moments

The first equality of the moment identity of arXiv:2412.09080v3, §3.1:
`κ^α = 𝔼[H^α(Y)]` for `|α| = 3` and a standardized law.

Both sides reduce to the same plain moment `∫ x₁^k x₂^{3-k} dμ`.

* On the right, the lower-order part of `H^{(k,3-k)}` is a first moment, which
  vanishes because the law is centred (`integral_hermite3_eq`).

* On the left, the cumulant is the mixed derivative at `0` of
  `log ∫ e^{u x₁ + v x₂} dμ`.  Each derivative of the logarithm produces a
  quotient whose correction terms carry a factor of the moment generating
  function's first derivatives, and those vanish at the origin of a centred
  law, while the function itself is `1` there.  What is left is the mixed
  moment, one differentiation under the integral sign per index.

The proof therefore runs over the four indices separately: `k = 0` and `k = 3`
are a single-variable third derivative (`iteratedDeriv_three_log`), `k = 2` a
quotient differentiated twice (`iteratedDeriv_two_div`) and `k = 1` a second
derivative of a logarithm differentiated once more
(`deriv_mul_sub_sq_div_sq`).

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`.
-/

open Real MeasureTheory Filter
open scoped Topology

namespace Transformer
namespace Modes

variable {μ : Measure (ℝ × ℝ)} {ε : ℝ}

/-- **The third Hermite integrals of a standardized law are its third
moments.**  `H^{(k,3-k)}` is `x₁^k x₂^{3-k}` plus a multiple of a single
coordinate, and a standardized law has mean `0`.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`. -/
theorem integral_hermite3_eq (hE : HasExpMomentsOn μ ε) (hε : 0 < ε) (hμ : IsStandardized μ)
    (k : ℕ) (hk : k ≤ 3) :
    ∫ x, hermite3 k x ∂μ = ∫ x, x.1 ^ k * x.2 ^ (3 - k) ∂μ := by
  have hint : ∀ i j : ℕ, Integrable (fun x : ℝ × ℝ => x.1 ^ i * x.2 ^ j) μ :=
    fun i j => integrable_pow_mul_pow hE hε i j
  have hfst : ∫ x : ℝ × ℝ, x.1 ^ 1 * x.2 ^ 0 ∂μ = 0 := by simpa using hμ.mean_fst
  have hsnd : ∫ x : ℝ × ℝ, x.1 ^ 0 * x.2 ^ 1 ∂μ = 0 := by simpa using hμ.mean_snd
  interval_cases k
  · have h : (fun x : ℝ × ℝ => hermite3 0 x)
        = fun x : ℝ × ℝ => x.1 ^ 0 * x.2 ^ 3 - 3 * (x.1 ^ 0 * x.2 ^ 1) := by
      funext x
      simp only [hermite3, hermite1]
      ring
    rw [h, integral_sub (hint 0 3) ((hint 0 1).const_mul 3), integral_const_mul, hsnd]
    ring
  · have h : (fun x : ℝ × ℝ => hermite3 1 x)
        = fun x : ℝ × ℝ => x.1 ^ 1 * x.2 ^ 2 - 1 * (x.1 ^ 1 * x.2 ^ 0) := by
      funext x
      simp only [hermite3, hermite1]
      ring
    rw [h, integral_sub (hint 1 2) ((hint 1 0).const_mul 1), integral_const_mul, hfst]
    ring
  · have h : (fun x : ℝ × ℝ => hermite3 2 x)
        = fun x : ℝ × ℝ => x.1 ^ 2 * x.2 ^ 1 - 1 * (x.1 ^ 0 * x.2 ^ 1) := by
      funext x
      simp only [hermite3, hermite1]
      ring
    rw [h, integral_sub (hint 2 1) ((hint 0 1).const_mul 1), integral_const_mul, hsnd]
    ring
  · have h : (fun x : ℝ × ℝ => hermite3 3 x)
        = fun x : ℝ × ℝ => x.1 ^ 3 * x.2 ^ 0 - 3 * (x.1 ^ 1 * x.2 ^ 0) := by
      funext x
      simp only [hermite3, hermite1]
      ring
    rw [h, integral_sub (hint 3 0) ((hint 1 0).const_mul 3), integral_const_mul, hfst]
    ring

/-- The hypotheses of `integral_hermite3_eq` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ (0 : ℝ) < 1 ∧ IsStandardized stdGauss2 ∧ (0 : ℕ) ≤ 3 :=
  ⟨hasExpMomentsOn_stdGauss2, one_pos, isStandardized_stdGauss2, by norm_num⟩

/-- **The first equality of the moment identity**: for `|α| = 3` and a
standardized law, `κ^α = 𝔼[H^α(Y)]`.

The source states it for `α` ranging over the multi-indices of order three,
which are the `(k, 3-k)` of `eq:psi`, and for "our normalization", which is
`IsStandardized`.  Two differences from the source, both recorded rather than
silent:

* `hexp` is an added hypothesis.  The source calls `κ^α` "the `α`-th mixed
  derivative at `0` of the cumulant generating function", which presupposes
  that the generating function is finite near `0`; in Lean `cumulantOf` is
  total, so without exponential moments the statement is false as written.

* Of the normalization only the centring is used.  Unit variances and zero
  covariance play no part at order three, so the theorem holds under the
  weaker hypothesis of a centred law; it is stated with the source's
  `IsStandardized` all the same.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`. -/
theorem cumulant_three_eq_integral_hermite (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hexp : HasExpMoments μ) (k : ℕ) (hk : k ≤ 3) :
    cumulantOf μ k (3 - k) = ∫ x, hermite3 k x ∂μ := by
  obtain ⟨ε, hε, hE⟩ := hexp
  have hε0 : |(0 : ℝ)| < ε := by simpa using hε
  have hball : Metric.ball (0 : ℝ) ε ∈ 𝓝 (0 : ℝ) := Metric.ball_mem_nhds 0 hε
  have habs : ∀ t ∈ Metric.ball (0 : ℝ) ε, |t| < ε := by
    intro t ht
    simpa [Real.dist_eq] using ht
  rw [integral_hermite3_eq hE hε hμ k hk, ← expMoment_origin]
  interval_cases k
  · -- `κ^{(0,3)}`: a third derivative in the second variable alone
    show cumulantOf μ 0 3 = expMoment μ 0 3 0 0
    have hc : cumulantOf μ 0 3 = iteratedDeriv 3 (fun v => Real.log (expMoment μ 0 0 0 v)) 0 := by
      simp only [cumulantOf, iteratedDeriv_zero, expMoment_zero_zero]
    rw [hc]
    exact iteratedDeriv_three_log (a₃ := fun v => expMoment μ 0 3 0 v)
      (eventually_of_mem hball fun t ht => hasDerivAt_expMoment_snd hE hε0 (habs t ht) 0 0)
      (eventually_of_mem hball fun t ht => hasDerivAt_expMoment_snd hE hε0 (habs t ht) 0 1)
      (hasDerivAt_expMoment_snd hE hε0 hε0 0 2)
      (eventually_of_mem hball fun t ht => expMoment_zero_zero_pos hE hε0 (habs t ht))
      (expMoment_origin_zero_zero μ) (expMoment_origin_zero_one hμ)
  · -- `κ^{(1,2)}`: a second derivative in `v`, then one in `u`
    show cumulantOf μ 1 2 = expMoment μ 1 2 0 0
    have hc : cumulantOf μ 1 2
        = deriv (fun u => iteratedDeriv 2 (fun v => Real.log (expMoment μ 0 0 u v)) 0) 0 := by
      simp only [cumulantOf, iteratedDeriv_one, expMoment_zero_zero]
    have heq : (fun u => iteratedDeriv 2 (fun v => Real.log (expMoment μ 0 0 u v)) 0)
        =ᶠ[𝓝 (0 : ℝ)] fun u => (expMoment μ 0 2 u 0 * expMoment μ 0 0 u 0
          - expMoment μ 0 1 u 0 * expMoment μ 0 1 u 0)
          / (expMoment μ 0 0 u 0 * expMoment μ 0 0 u 0) := by
      refine eventually_of_mem hball fun u hu => ?_
      show iteratedDeriv 2 (fun v => Real.log (expMoment μ 0 0 u v)) 0
          = (expMoment μ 0 2 u 0 * expMoment μ 0 0 u 0
            - expMoment μ 0 1 u 0 * expMoment μ 0 1 u 0)
            / (expMoment μ 0 0 u 0 * expMoment μ 0 0 u 0)
      exact iteratedDeriv_two_log
        (eventually_of_mem hball fun t ht =>
          hasDerivAt_expMoment_snd hE (habs u hu) (habs t ht) 0 0)
        (hasDerivAt_expMoment_snd hE (habs u hu) hε0 0 1)
        (eventually_of_mem hball fun t ht => expMoment_zero_zero_pos hE (habs u hu) (habs t ht))
    rw [hc, heq.deriv_eq]
    exact deriv_mul_sub_sq_div_sq (P₁ := fun u => expMoment μ 1 2 u 0)
      (hasDerivAt_expMoment_fst hE hε0 hε0 0 2) (hasDerivAt_expMoment_fst hE hε0 hε0 0 0)
      (hasDerivAt_expMoment_fst hE hε0 hε0 0 1) (expMoment_origin_zero_zero μ)
      (expMoment_origin_one_zero hμ) (expMoment_origin_zero_one hμ)
  · -- `κ^{(2,1)}`: one derivative in `v`, then two in `u`
    show cumulantOf μ 2 1 = expMoment μ 2 1 0 0
    have hc : cumulantOf μ 2 1
        = iteratedDeriv 2 (fun u => deriv (fun v => Real.log (expMoment μ 0 0 u v)) 0) 0 := by
      simp only [cumulantOf, iteratedDeriv_one, expMoment_zero_zero]
    have heq : (fun u => deriv (fun v => Real.log (expMoment μ 0 0 u v)) 0)
        =ᶠ[𝓝 (0 : ℝ)] fun u => expMoment μ 0 1 u 0 / expMoment μ 0 0 u 0 := by
      refine eventually_of_mem hball fun u hu => ?_
      show deriv (fun v => Real.log (expMoment μ 0 0 u v)) 0
          = expMoment μ 0 1 u 0 / expMoment μ 0 0 u 0
      exact ((hasDerivAt_expMoment_snd hE (habs u hu) hε0 0 0).log
        (expMoment_zero_zero_pos hE (habs u hu) hε0).ne').deriv
    rw [hc, heq.iteratedDeriv_eq 2]
    exact iteratedDeriv_two_div (b₂ := fun u => expMoment μ 2 1 u 0)
      (eventually_of_mem hball fun t ht => hasDerivAt_expMoment_fst hE (habs t ht) hε0 0 1)
      (hasDerivAt_expMoment_fst hE hε0 hε0 1 1)
      (eventually_of_mem hball fun t ht => hasDerivAt_expMoment_fst hE (habs t ht) hε0 0 0)
      (hasDerivAt_expMoment_fst hE hε0 hε0 1 0)
      (eventually_of_mem hball fun t ht => (expMoment_zero_zero_pos hE (habs t ht) hε0).ne')
      (expMoment_origin_zero_zero μ) (expMoment_origin_one_zero hμ)
      (expMoment_origin_zero_one hμ)
  · -- `κ^{(3,0)}`: a third derivative in the first variable alone
    show cumulantOf μ 3 0 = expMoment μ 3 0 0 0
    have hc : cumulantOf μ 3 0 = iteratedDeriv 3 (fun u => Real.log (expMoment μ 0 0 u 0)) 0 := by
      simp only [cumulantOf, iteratedDeriv_zero, expMoment_zero_zero]
    rw [hc]
    exact iteratedDeriv_three_log (a₃ := fun u => expMoment μ 3 0 u 0)
      (eventually_of_mem hball fun t ht => hasDerivAt_expMoment_fst hE (habs t ht) hε0 0 0)
      (eventually_of_mem hball fun t ht => hasDerivAt_expMoment_fst hE (habs t ht) hε0 1 0)
      (hasDerivAt_expMoment_fst hE hε0 hε0 2 0)
      (eventually_of_mem hball fun t ht => expMoment_zero_zero_pos hE (habs t ht) hε0)
      (expMoment_origin_zero_zero μ) (expMoment_origin_one_zero hμ)

/-- The hypotheses of `cumulant_three_eq_integral_hermite` are satisfiable. -/
example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ (0 : ℕ) ≤ 3 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2, by norm_num⟩

end Modes
end Transformer
