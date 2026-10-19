module

public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
public import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

/-!
# Weighted interval partitions for the exponential martingale argument

ArXiv:2604.01978v1, §7, the maximal inequality after
eq:Gronwall_E_selfcontained, and def:ito.formula.
To compensate the exponential observable, its random integrating
factor must be multiplied through the actual martingale problem.
This module proves the finite-partition algebra and integration
bounds used to pass that argument to continuous time.

The error is controlled by the actual oscillation of the continuous
weight on each interval. The integration-by-parts identity is an
exact telescoping identity for the original sampled paths.
No stochastic integral, probabilistic estimate or convergence
assumption is introduced into a definition.
-/

@[expose] public section

open scoped BigOperators
open Set MeasureTheory

namespace Transformer.Homogenized

/-- The actual weighted interval integral has error at most oscillation times the bound for its integrand and interval length.
Source: arXiv:2604.01978v1, §7, the exponential maximal-inequality argument. -/
theorem abs_weighted_integral_error_le (f g : ℝ → ℝ) (a b c ε A : ℝ) (hab : a ≤ b) (hA : 0 ≤ A)
    (hf : ContinuousOn f (Icc a b)) (hg : ContinuousOn g (Icc a b))
    (hclose : ∀ x ∈ Icc a b, |c - f x| ≤ ε) (hbound : ∀ x ∈ Icc a b, |g x| ≤ A) :
    |c * (∫ x in a..b, g x) - ∫ x in a..b, f x * g x| ≤ ε * A * (b - a) := by
  have hgInt : IntervalIntegrable g volume a b := by
    apply ContinuousOn.intervalIntegrable
    simpa only [uIcc_of_le hab] using hg
  have hfgInt : IntervalIntegrable (fun x => f x * g x) volume a b := by
    apply ContinuousOn.intervalIntegrable
    rw [uIcc_of_le hab]
    exact hf.mul hg
  have heq : c * (∫ x in a..b, g x) - ∫ x in a..b, f x * g x =
      ∫ x in a..b, (c - f x) * g x := by
    simp_rw [sub_mul]
    rw [intervalIntegral.integral_sub (hgInt.const_mul c) hfgInt, intervalIntegral.integral_const_mul]
  rw [heq]
  have hnorm := intervalIntegral.norm_integral_le_of_norm_le_const (a := a) (b := b)
    (C := ε * A) (f := fun x => (c - f x) * g x) (fun x hx => ?_)
  · simpa only [Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr hab)] using hnorm
  · rw [uIoc_of_le hab] at hx
    have hxi : x ∈ Icc a b := ⟨hx.1.le, hx.2⟩
    rw [Real.norm_eq_abs, abs_mul]
    calc
      _ ≤ |c - f x| * A := mul_le_mul_of_nonneg_left (hbound x hxi) (abs_nonneg _)
      _ ≤ ε * A := mul_le_mul_of_nonneg_right (hclose x hxi) hA

/-- Adding the actual interval errors gives the same bound on an arbitrary finite monotone partition.
Source: arXiv:2604.01978v1, §7, the exponential maximal-inequality argument. -/
theorem abs_weighted_partition_error_le (f g : ℝ → ℝ) (θ : ℕ → ℝ) (hθ : Monotone θ) (N : ℕ)
    (c : ℕ → ℝ) (ε A : ℝ) (hA : 0 ≤ A)
    (hf : ContinuousOn f (Icc (θ 0) (θ N))) (hg : ContinuousOn g (Icc (θ 0) (θ N)))
    (hclose : ∀ k < N, ∀ x ∈ Icc (θ k) (θ (k + 1)), |c k - f x| ≤ ε)
    (hbound : ∀ x ∈ Icc (θ 0) (θ N), |g x| ≤ A) :
    |(∑ k ∈ Finset.range N, c k * ∫ x in θ k..θ (k + 1), g x) -
      ∫ x in θ 0..θ N, f x * g x| ≤ ε * A * (θ N - θ 0) := by
  have hsub (k : ℕ) (hk : k < N) : Icc (θ k) (θ (k + 1)) ⊆ Icc (θ 0) (θ N) :=
    fun x hx => ⟨(hθ (Nat.zero_le k)).trans hx.1, hx.2.trans (hθ (Nat.succ_le_of_lt hk))⟩
  have hInt (k : ℕ) (hk : k < N) : IntervalIntegrable (fun x => f x * g x) volume (θ k) (θ (k + 1)) := by
    apply ContinuousOn.intervalIntegrable
    rw [uIcc_of_le (hθ (Nat.le_succ k))]
    exact (hf.mul hg).mono (hsub k hk)
  rw [← intervalIntegral.sum_integral_adjacent_intervals hInt, ← Finset.sum_sub_distrib]
  calc
    _ ≤ ∑ k ∈ Finset.range N, |c k * (∫ x in θ k..θ (k + 1), g x) -
        ∫ x in θ k..θ (k + 1), f x * g x| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k ∈ Finset.range N, ε * A * (θ (k + 1) - θ k) := by
      apply Finset.sum_le_sum
      intro k hk
      have hkN := Finset.mem_range.mp hk
      exact abs_weighted_integral_error_le f g _ _ _ ε A (hθ (Nat.le_succ k)) hA
        (hf.mono (hsub k hkN)) (hg.mono (hsub k hkN)) (hclose k hkN) (fun x hx => hbound x (hsub k hkN hx))
    _ = _ := by rw [← Finset.mul_sum, Finset.sum_range_sub]

/-- Bounded actual weights give a uniform bound for the weighted partition sum.
Source: arXiv:2604.01978v1, §7, the exponential maximal-inequality argument. -/
theorem abs_weighted_partition_sum_le (g : ℝ → ℝ) (θ : ℕ → ℝ) (hθ : Monotone θ) (N : ℕ)
    (c : ℕ → ℝ) (B A : ℝ) (hA : 0 ≤ A) (hg : ContinuousOn g (Icc (θ 0) (θ N)))
    (hc : ∀ k < N, |c k| ≤ B) (hbound : ∀ x ∈ Icc (θ 0) (θ N), |g x| ≤ A) :
    |∑ k ∈ Finset.range N, c k * ∫ x in θ k..θ (k + 1), g x| ≤ B * A * (θ N - θ 0) := by
  have h := abs_weighted_partition_error_le (fun _ => 0) g θ hθ N c B A hA continuousOn_const hg
    (fun k hk x hx => by simpa only [sub_zero] using hc k hk) hbound
  simpa only [zero_mul, intervalIntegral.integral_zero, sub_zero] using h


/-- Even the sum of the absolute weighted interval contributions is
uniformly bounded. Source: arXiv:2604.01978v1, §7,
the bounded partition approximations in the exponential martingale argument. -/
theorem sum_abs_weighted_partition_le (g : ℝ → ℝ) (θ : ℕ → ℝ) (hθ : Monotone θ) (N : ℕ)
    (c : ℕ → ℝ) (B A : ℝ) (hA : 0 ≤ A) (hg : ContinuousOn g (Icc (θ 0) (θ N)))
    (hc : ∀ k < N, |c k| ≤ B) (hbound : ∀ x ∈ Icc (θ 0) (θ N), |g x| ≤ A) :
    (∑ k ∈ Finset.range N, |c k * ∫ x in θ k..θ (k + 1), g x|) ≤ B * A * (θ N - θ 0) := by
  have hsub (k : ℕ) (hk : k < N) : Icc (θ k) (θ (k + 1)) ⊆ Icc (θ 0) (θ N) :=
    fun x hx => ⟨(hθ (Nat.zero_le k)).trans hx.1, hx.2.trans (hθ (Nat.succ_le_of_lt hk))⟩
  calc
    _ ≤ ∑ k ∈ Finset.range N, B * A * (θ (k + 1) - θ k) := by
      apply Finset.sum_le_sum
      intro k hk
      have hkN := Finset.mem_range.mp hk
      have h := abs_weighted_integral_error_le (fun _ => 0) g (θ k) (θ (k + 1)) (c k) B A
        (hθ (Nat.le_succ k)) hA continuousOn_const (hg.mono (hsub k hkN))
        (fun x hx => by simpa only [sub_zero] using hc k hkN)
        (fun x hx => hbound x (hsub k hkN hx))
      simpa only [zero_mul, intervalIntegral.integral_zero, sub_zero] using h
    _ = _ := by rw [← Finset.mul_sum, Finset.sum_range_sub]

/-- Exact integration by parts for the original sampled scalar paths. This identity precedes any probabilistic or limiting argument.
Source: arXiv:2604.01978v1, §7, the exponential maximal-inequality argument. -/
theorem weighted_partition_integration_by_parts (C Y : ℝ → ℝ) (θ : ℕ → ℝ) (N : ℕ) :
    (∑ k ∈ Finset.range N, C (θ k) * (Y (θ (k + 1)) - Y (θ k))) +
      (∑ k ∈ Finset.range N, (C (θ (k + 1)) - C (θ k)) * Y (θ (k + 1))) =
        C (θ N) * Y (θ N) - C (θ 0) * Y (θ 0) := by
  rw [← Finset.sum_add_distrib]
  calc
    _ = ∑ k ∈ Finset.range N,
        (C (θ (k + 1)) * Y (θ (k + 1)) - C (θ k) * Y (θ k)) := by
      apply Finset.sum_congr rfl
      intro k hk
      ring
    _ = _ := Finset.sum_range_sub (fun k => C (θ k) * Y (θ k)) N

/-- A nonconstant weight, a nonzero integrand and an actual monotone
partition satisfy every premise and have nonzero error 1/2. -/
example : Monotone (fun k : ℕ => (k : ℝ)) ∧ (0 : ℝ) ≤ 1 ∧
    ContinuousOn (fun x : ℝ => x) (Icc (0 : ℝ) 1) ∧
    ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Icc (0 : ℝ) 1) ∧
    (∀ k : ℕ, k < 1 → ∀ x ∈ Icc (k : ℝ) ((k + 1 : ℕ) : ℝ), |1 - x| ≤ (1 : ℝ)) ∧
    (∀ k : ℕ, k < 1 → |(1 : ℝ)| ≤ 1) ∧
    (∀ x ∈ Icc (0 : ℝ) 1, |(1 : ℝ)| ≤ 1) ∧
    |(1 : ℝ) * (∫ _ in (0 : ℝ)..1, (1 : ℝ)) - ∫ x in (0 : ℝ)..1, x * 1| = 1 / 2 := by
  refine ⟨fun i j hij => by change (i : ℝ) ≤ (j : ℝ); exact_mod_cast hij, by norm_num, continuous_id.continuousOn,
    continuousOn_const, ?_, fun k hk => by norm_num, fun x hx => by norm_num, ?_⟩
  · intro k hk x hx
    have hk0 : k = 0 := by omega
    subst k
    norm_num at hx
    exact abs_le.mpr ⟨by linarith [hx.2], by linarith [hx.1]⟩
  · norm_num [integral_id]

/-- Two nonconstant sampled paths give nonzero exact integration by parts.
Source: arXiv:2604.01978v1, §7, the integrating-factor argument. -/
example : (∑ k ∈ Finset.range 1, (k : ℝ) * (((k + 1 : ℕ) : ℝ) ^ 2 - (k : ℝ) ^ 2)) +
    (∑ k ∈ Finset.range 1, (((k + 1 : ℕ) : ℝ) - (k : ℝ)) * ((k + 1 : ℕ) : ℝ) ^ 2) = 1 := by
  norm_num

end Transformer.Homogenized
