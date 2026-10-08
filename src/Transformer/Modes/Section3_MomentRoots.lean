import Transformer.Modes.Section3_MixedMoments
import Mathlib.Analysis.Convex.Integral
import Mathlib.Analysis.Convex.SpecificFunctions.Pow

/-!
# Comparing every mixed moment with one Euclidean moment

The proof of `lem:eta` in arXiv:2412.09080v3, §5.3, uses that a cumulant of
order `s` is bounded by the `s`th Euclidean moment. This module supplies the
moment comparison needed for each factor in its moment expansion.

Exponential moments imply integrability of every Euclidean power. For a
probability law, Jensen's inequality for the concave power `k/s` then gives
`E ‖X‖^k ≤ (E ‖X‖^s)^(k/s)`. A mixed monomial of degree `k` is bounded by
`‖X‖^k`. Using the common root `R = (E ‖X‖^s)^(1/s)` makes products of such
bounds multiply to `R^s`, exactly the moment in the source.

This includes vanishing moments: nonnegative fractional powers and the
positive total degree make the same identities valid at zero. The moment
root is only a scalar used in the bound; the original moment integral is
kept in the final cumulant inequality.

No centring or covariance hypothesis is needed for these comparisons.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`, and its proof in §5.3.
-/

open Real MeasureTheory Filter

namespace Transformer.Modes

/-- A mixed monomial is bounded by the Euclidean power of the same order.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`. -/
theorem abs_mixed_monomial_le (z : ℝ × ℝ) (i j : ℕ) :
    |z.1 ^ i * z.2 ^ j| ≤ eucl z ^ (i + j) := by
  have h1 : |z.1| ≤ eucl z := by
    rw [eucl, ← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_le_sqrt (by nlinarith)
  have h2 : |z.2| ≤ eucl z := by
    rw [eucl, ← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_le_sqrt (by nlinarith)
  have hr : 0 ≤ eucl z := Real.sqrt_nonneg _
  rw [abs_mul, abs_pow, abs_pow, pow_add]
  gcongr

/-- Exponential moments imply finite Euclidean moments of every order.
Source: arXiv:2412.09080v3, §3.1, definition of the cumulants. -/
theorem integrable_eucl_pow {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    (hexp : HasExpMoments μ) (s : ℕ) : Integrable (fun z => eucl z ^ s) μ := by
  obtain ⟨ε, hε, hE⟩ := hexp
  have h1 : Integrable (fun z : ℝ × ℝ => |z.1| ^ s) μ := by
    simpa only [pow_zero, mul_one, abs_pow] using (integrable_pow_mul_pow hE hε s 0).abs
  have h2 : Integrable (fun z : ℝ × ℝ => |z.2| ^ s) μ := by
    simpa only [pow_zero, one_mul, abs_pow] using (integrable_pow_mul_pow hE hε 0 s).abs
  have hp := (h1.add h2).const_mul ((2 : ℝ) ^ (s - 1))
  refine hp.mono' (by unfold eucl; fun_prop) (ae_of_all _ fun z => ?_)
  have hr : 0 ≤ eucl z := Real.sqrt_nonneg _
  rw [Real.norm_eq_abs, abs_of_nonneg (pow_nonneg hr s)]
  have he : eucl z ≤ |z.1| + |z.2| := by
    rw [eucl, ← Real.sqrt_sq (add_nonneg (abs_nonneg z.1) (abs_nonneg z.2))]
    exact Real.sqrt_le_sqrt (by
      nlinarith [sq_abs z.1, sq_abs z.2, mul_nonneg (abs_nonneg z.1) (abs_nonneg z.2)])
  exact (pow_le_pow_left₀ (Real.sqrt_nonneg _) he s).trans
    (add_pow_le (abs_nonneg z.1) (abs_nonneg z.2) s)

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨inferInstance, hasExpMoments_stdGauss2⟩

/-- The power appearing in Jensen's inequality returns the lower moment.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem pow_rpow_ratio (r : ℝ) (hr : 0 ≤ r) (n k : ℕ) (hn : 0 < n) :
    (r ^ n) ^ ((k : ℝ) / n) = r ^ k := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul hr]
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [mul_div_cancel₀ _ hn', Real.rpow_natCast]

example : (0 : ℝ) ≤ 2 ∧ (0 : ℕ) < 3 := ⟨by norm_num, by norm_num⟩

/-- An integer power of the common moment root is the fractional moment
bound. Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem moment_root_pow (M : ℝ) (hM : 0 ≤ M) (n k : ℕ) :
    (M ^ ((n : ℝ)⁻¹)) ^ k = M ^ ((k : ℝ) / n) := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul hM]
  congr 1
  ring

example : (0 : ℝ) ≤ 8 := by norm_num

/-- The common moment root is nonnegative, also when the moment is zero.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem moment_root_nonneg (M : ℝ) (hM : 0 ≤ M) (n : ℕ) :
    0 ≤ M ^ ((n : ℝ)⁻¹) := Real.rpow_nonneg hM _

example : (0 : ℝ) ≤ 0 := le_rfl

/-- Multiplying factors of total positive degree recovers the original
moment exactly. Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem moment_root_pow_self (M : ℝ) (hM : 0 ≤ M) (n : ℕ) (hn : 0 < n) :
    (M ^ ((n : ℝ)⁻¹)) ^ n = M := by
  rw [moment_root_pow M hM n n]
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [div_self hn', Real.rpow_one]

example : (0 : ℝ) ≤ 8 ∧ (0 : ℕ) < 3 := ⟨by norm_num, by norm_num⟩

/-- Lyapunov's moment inequality from Jensen's concave power inequality.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem integral_eucl_pow_le_root {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    (hexp : HasExpMoments μ) (n k : ℕ) (hn : 0 < n) (hk : k ≤ n) :
    (∫ z, eucl z ^ k ∂μ) ≤ (∫ z, eucl z ^ n ∂μ) ^ ((k : ℝ) / n) := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hk0 : (0 : ℝ) ≤ (k : ℝ) / n := by positivity
  have hk1 : (k : ℝ) / n ≤ 1 := (div_le_one hn').2 (by exact_mod_cast hk)
  have hg := Real.concaveOn_rpow hk0 hk1
  have heq : (fun z => (eucl z ^ n) ^ ((k : ℝ) / n)) = fun z => eucl z ^ k := by
    funext z
    exact pow_rpow_ratio _ (Real.sqrt_nonneg _) n k hn
  have h := hg.le_map_integral (Real.continuous_rpow_const hk0).continuousOn
    isClosed_Ici (ae_of_all _ fun z => pow_nonneg (Real.sqrt_nonneg _) n)
    (integrable_eucl_pow hexp n) (by
      change Integrable (fun z => (eucl z ^ n) ^ ((k : ℝ) / n)) μ
      rw [heq]
      exact integrable_eucl_pow hexp k)
  change (∫ z, (eucl z ^ n) ^ ((k : ℝ) / n) ∂μ) ≤
    (∫ z, eucl z ^ n ∂μ) ^ ((k : ℝ) / n) at h
  rw [heq] at h
  exact h

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMoments stdGauss2 ∧
    (0 : ℕ) < 3 ∧ (2 : ℕ) ≤ 3 :=
  ⟨inferInstance, hasExpMoments_stdGauss2, by norm_num, by norm_num⟩

/-- Every lower mixed moment is controlled by a power of the common root.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem expMoment_origin_le_root {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    (hexp : HasExpMoments μ) (n i j : ℕ) (hn : 0 < n) (hij : i + j ≤ n) :
    |expMoment μ i j 0 0| ≤ ((∫ z, eucl z ^ n ∂μ) ^ ((n : ℝ)⁻¹)) ^ (i + j) := by
  have hM : 0 ≤ ∫ z, eucl z ^ n ∂μ := integral_nonneg fun z => by unfold eucl; positivity
  rw [expMoment_origin, moment_root_pow _ hM]
  obtain ⟨ε, hε, hE⟩ := hexp
  calc |∫ z, z.1 ^ i * z.2 ^ j ∂μ| ≤ ∫ z, |z.1 ^ i * z.2 ^ j| ∂μ :=
      abs_integral_le_integral_abs
    _ ≤ ∫ z, eucl z ^ (i + j) ∂μ := integral_mono
      (integrable_pow_mul_pow hE hε i j).abs (integrable_eucl_pow ⟨ε, hε, hE⟩ (i + j))
      (fun z => abs_mixed_monomial_le z i j)
    _ ≤ _ := integral_eucl_pow_le_root ⟨ε, hε, hE⟩ n (i + j) hn hij

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMoments stdGauss2 ∧
    (0 : ℕ) < 3 ∧ (1 : ℕ) + 2 ≤ 3 :=
  ⟨inferInstance, hasExpMoments_stdGauss2, by norm_num, by norm_num⟩

end Transformer.Modes
