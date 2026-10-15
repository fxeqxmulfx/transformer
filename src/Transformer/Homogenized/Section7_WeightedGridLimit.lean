/-
# Actual weighted partition sums converge to the time integral

ArXiv:2604.01978v1, §7, the exponential maximal inequality
following eq:Gronwall_E_selfcontained and def:ito.formula.
The finite-partition estimates now give convergence for the actual
interval grids. Each weight is evaluated at an actual tag inside
its interval, including either endpoint.

The proof derives the oscillation bound from continuity on the
original compact time interval and the actual mesh T/(n+1).
The weighted integrals and their limit remain the original ones;
no convergence assumption is placed in a definition.
The endpoint versions will be used on the two sides of the
finite integration-by-parts identity for the random factor.
-/

import Transformer.Homogenized.Section7_WeightedPartition
import Transformer.Homogenized.Section7_IntervalGrid
import Mathlib.Topology.UniformSpace.HeineCantor

open scoped BigOperators
open Set Filter MeasureTheory Topology

namespace Transformer.Homogenized

/-- The actual step length is exactly T/(n+1) before the clock is clipped.
Source: arXiv:2604.01978v1, §7, the exponential martingale argument. -/
theorem intervalGridTime_succ_sub (T : ℝ) (n k : ℕ) (hk : k < n + 1) :
    intervalGridTime T n (k + 1) - intervalGridTime T n k = T / ((n : ℝ) + 1) := by
  rw [intervalGridTime, intervalGridTime, min_eq_left (Nat.succ_le_of_lt hk), min_eq_left hk.le]
  simp only [Nat.cast_succ]
  ring

/-- Every interval between consecutive clock times lies in [0,T].
Source: arXiv:2604.01978v1, §7, the exponential martingale argument. -/
theorem intervalGridTime_subinterval_subset (T : ℝ) (hT : 0 ≤ T) (n k : ℕ) :
    Icc (intervalGridTime T n k) (intervalGridTime T n (k + 1)) ⊆ Icc (0 : ℝ) T :=
  fun _ hx => ⟨(intervalGridTime_mem T hT n k).1.trans hx.1,
    hx.2.trans (intervalGridTime_mem T hT n (k + 1)).2⟩

/-- Any tag and integration point in one actual interval are at distance at most its mesh.
Source: arXiv:2604.01978v1, §7, the exponential martingale argument. -/
theorem dist_intervalGrid_tag_le (T : ℝ) (n k : ℕ) (hk : k < n + 1) (t x : ℝ)
    (ht : t ∈ Icc (intervalGridTime T n k) (intervalGridTime T n (k + 1)))
    (hx : x ∈ Icc (intervalGridTime T n k) (intervalGridTime T n (k + 1))) :
    dist t x ≤ T / ((n : ℝ) + 1) := by
  have hs := intervalGridTime_succ_sub T n k hk
  rw [Real.dist_eq]
  exact abs_le.mpr ⟨by linarith [ht.1, hx.2], by linarith [ht.2, hx.1]⟩

/-- The actual tagged weighted partition sums converge to the original product integral. Continuity gives the oscillation estimate used in the proof.
Source: arXiv:2604.01978v1, §7, the exponential martingale argument. -/
theorem weighted_intervalGrid_integrals_tendsto (T : ℝ) (hT : 0 < T) (f g : ℝ → ℝ)
    (hf : ContinuousOn f (Icc (0 : ℝ) T)) (hg : ContinuousOn g (Icc (0 : ℝ) T))
    (tag : ℕ → ℕ → ℝ)
    (htag : ∀ n k, k < n + 1 → tag n k ∈ Icc (intervalGridTime T n k) (intervalGridTime T n (k + 1)))
    (A : ℝ) (hA : 0 ≤ A) (hbound : ∀ x ∈ Icc (0 : ℝ) T, |g x| ≤ A) :
    Tendsto (fun n => ∑ k ∈ Finset.range (n + 1),
      f (tag n k) * ∫ x in intervalGridTime T n k..intervalGridTime T n (k + 1), g x)
      atTop (𝓝 (∫ x in (0 : ℝ)..T, f x * g x)) := by
  apply Metric.tendsto_atTop.mpr
  intro ε hε
  let η := ε / (A * T + 1)
  have hden : 0 < A * T + 1 := by positivity
  have hη : 0 < η := div_pos hε hden
  have hηmul : η * A * T < ε := by
    have hcancel : η * (A * T + 1) = ε := div_mul_cancel₀ ε hden.ne'
    nlinarith
  obtain ⟨δ, hδ, hmod⟩ := Metric.uniformContinuousOn_iff.mp
    (isCompact_Icc.uniformContinuousOn_of_continuous hf) η hη
  have he : Tendsto (fun n : ℕ => T / ((n : ℝ) + 1)) atTop (𝓝 0) := by
    simpa only [mul_one_div, mul_zero] using
      (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).const_mul T
  obtain ⟨N, hN⟩ := eventually_atTop.mp (he.eventually (eventually_lt_nhds hδ))
  refine ⟨N, fun n hn => ?_⟩
  have hclose (k : ℕ) (hk : k < n + 1) (x : ℝ)
      (hx : x ∈ Icc (intervalGridTime T n k) (intervalGridTime T n (k + 1))) :
      |f (tag n k) - f x| ≤ η := by
    have hd := (dist_intervalGrid_tag_le T n k hk (tag n k) x (htag n k hk) hx).trans_lt (hN n hn)
    have hfclose := hmod (tag n k) (intervalGridTime_subinterval_subset T hT.le n k (htag n k hk))
      x (intervalGridTime_subinterval_subset T hT.le n k hx) hd
    simpa only [Real.dist_eq] using hfclose.le
  have h := abs_weighted_partition_error_le f g (intervalGridTime T n)
    (intervalGridTime_monotone T hT.le n) (n + 1) (fun k => f (tag n k)) η A hA
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hf)
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hg) hclose
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hbound)
  simp only [intervalGridTime_zero, intervalGridTime_last, sub_zero] at h
  rw [Real.dist_eq]
  exact h.trans_lt hηmul

/-- Actual left endpoint weights give the original time integral.
Source: arXiv:2604.01978v1, §7, the adapted martingale partition sums. -/
theorem left_weighted_intervalGrid_integrals_tendsto (T : ℝ) (hT : 0 < T) (f g : ℝ → ℝ)
    (hf : ContinuousOn f (Icc (0 : ℝ) T)) (hg : ContinuousOn g (Icc (0 : ℝ) T))
    (A : ℝ) (hA : 0 ≤ A) (hbound : ∀ x ∈ Icc (0 : ℝ) T, |g x| ≤ A) :
    Tendsto (fun n => ∑ k ∈ Finset.range (n + 1), f (intervalGridTime T n k) *
      ∫ x in intervalGridTime T n k..intervalGridTime T n (k + 1), g x)
      atTop (𝓝 (∫ x in (0 : ℝ)..T, f x * g x)) := by
  exact weighted_intervalGrid_integrals_tendsto T hT f g hf hg
    (fun n k => intervalGridTime T n k)
    (fun n k hk => ⟨le_rfl, intervalGridTime_monotone T hT.le n (Nat.le_succ k)⟩) A hA hbound

/-- Actual right endpoint weights give the same original time integral.
Source: arXiv:2604.01978v1, §7, the integrating-factor partition sums. -/
theorem right_weighted_intervalGrid_integrals_tendsto (T : ℝ) (hT : 0 < T) (f g : ℝ → ℝ)
    (hf : ContinuousOn f (Icc (0 : ℝ) T)) (hg : ContinuousOn g (Icc (0 : ℝ) T))
    (A : ℝ) (hA : 0 ≤ A) (hbound : ∀ x ∈ Icc (0 : ℝ) T, |g x| ≤ A) :
    Tendsto (fun n => ∑ k ∈ Finset.range (n + 1), f (intervalGridTime T n (k + 1)) *
      ∫ x in intervalGridTime T n k..intervalGridTime T n (k + 1), g x)
      atTop (𝓝 (∫ x in (0 : ℝ)..T, f x * g x)) := by
  exact weighted_intervalGrid_integrals_tendsto T hT f g hf hg
    (fun n k => intervalGridTime T n (k + 1))
    (fun n k hk => ⟨intervalGridTime_monotone T hT.le n (Nat.le_succ k), le_rfl⟩) A hA hbound

/-- The actual weighted sums have a bound independent of the grid size.
Source: arXiv:2604.01978v1, §7, the dominated limit used in the
exponential martingale argument. The tags are actual times in [0,T]. -/
theorem abs_weighted_intervalGrid_sum_le (T : ℝ) (hT : 0 ≤ T) (f g : ℝ → ℝ)
    (n : ℕ) (tag : ℕ → ℝ) (htag : ∀ k < n + 1, tag k ∈ Icc (0 : ℝ) T)
    (B A : ℝ) (hA : 0 ≤ A) (hg : ContinuousOn g (Icc (0 : ℝ) T))
    (hfBound : ∀ x ∈ Icc (0 : ℝ) T, |f x| ≤ B) (hgBound : ∀ x ∈ Icc (0 : ℝ) T, |g x| ≤ A) :
    |∑ k ∈ Finset.range (n + 1), f (tag k) *
      ∫ x in intervalGridTime T n k..intervalGridTime T n (k + 1), g x| ≤ B * A * T := by
  have h := abs_weighted_partition_sum_le g (intervalGridTime T n)
    (intervalGridTime_monotone T hT n) (n + 1) (fun k => f (tag k)) B A hA
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hg)
    (fun k hk => hfBound _ (htag k hk))
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hgBound)
  simpa only [intervalGridTime_zero, intervalGridTime_last, sub_zero] using h

/-- A nonconstant weight and nonzero integrand satisfy every convergence
premise. The actual left endpoint sums converge to the nonzero integral 1/2. -/
example : ∃ (f g : ℝ → ℝ) (tag : ℕ → ℕ → ℝ),
    (0 : ℝ) < 1 ∧ ContinuousOn f (Icc (0 : ℝ) 1) ∧ ContinuousOn g (Icc (0 : ℝ) 1) ∧
    (∀ n k, k < n + 1 → tag n k ∈ Icc (intervalGridTime 1 n k) (intervalGridTime 1 n (k + 1))) ∧
    (0 : ℝ) ≤ 1 ∧ (∀ x ∈ Icc (0 : ℝ) 1, |g x| ≤ 1) ∧
    (∀ x ∈ Icc (0 : ℝ) 1, |f x| ≤ 1) ∧
    Tendsto (fun n => ∑ k ∈ Finset.range (n + 1), f (intervalGridTime 1 n k) *
      ∫ x in intervalGridTime 1 n k..intervalGridTime 1 n (k + 1), g x) atTop (𝓝 (1 / 2)) := by
  refine ⟨fun x => x, fun _ => 1, fun n k => intervalGridTime 1 n k,
    one_pos, continuous_id.continuousOn, continuousOn_const, ?_, zero_le_one,
    fun x hx => by norm_num, fun x hx => by simpa only [abs_of_nonneg hx.1] using hx.2, ?_⟩
  · intro n k hk
    exact ⟨le_rfl, intervalGridTime_monotone 1 zero_le_one n (Nat.le_succ k)⟩
  · simpa using left_weighted_intervalGrid_integrals_tendsto 1 one_pos (fun x => x) (fun _ => 1)
      continuous_id.continuousOn continuousOn_const 1 zero_le_one (fun x hx => by norm_num)

end Transformer.Homogenized
