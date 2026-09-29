/-
# Kinetic theory for Transformers — right-endpoint Riemann sums

`N⁻¹ Σ_{j=1}^N g(j/N) → ∫_0^1 g` for continuous `g`: the empirical measures of the cursor
`σ_j = j/N` converge to the Lebesgue measure on `(0,1]`, which is what makes the configurations
of tokens that never move witnesses of the initial convergence hypothesis of `mean_field_limit`.
-/

import Transformer.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Topology.UniformSpace.HeineCantor

open scoped BigOperators
open Real MeasureTheory Filter Set

namespace Transformer
namespace Kinetic

/-- **Right-endpoint Riemann sums.**  For a continuous `g`,

  `N⁻¹ Σ_{j=1}^N g(j/N) → ∫_0^1 g`.

The positions `j = 1,…,N` of the source are the elements of `Idx N = Fin N` shifted by one, as in
`EmpiricalTendsto`. -/
theorem tendsto_rightRiemann {g : ℝ → ℝ} (hg : Continuous g) :
    Tendsto (fun N : ℕ => (N : ℝ)⁻¹ * ∑ j : Idx N, g (((j : ℝ) + 1) / N)) atTop
      (nhds (∫ σ in (0 : ℝ)..1, g σ)) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  obtain ⟨δ, hδ, hδg⟩ := Metric.uniformContinuousOn_iff.1
    (isCompact_Icc.uniformContinuousOn_of_continuous hg.continuousOn) (ε / 2) (half_pos hε)
  obtain ⟨N₀, hN₀⟩ := exists_nat_gt (1 / δ)
  refine ⟨N₀ + 1, fun N hN => ?_⟩
  have hN' : (N₀ : ℝ) + 1 ≤ N := by exact_mod_cast hN
  have hNpos : (0 : ℝ) < N := by
    have h0 : (0 : ℝ) ≤ N₀ := Nat.cast_nonneg _
    linarith
  have hNδ : 1 / (N : ℝ) < δ := by
    rw [div_lt_iff₀ hδ] at hN₀
    rw [div_lt_iff₀ hNpos]
    nlinarith
  have hsum : ∫ σ in (0 : ℝ)..1, g σ
      = ∑ k ∈ Finset.range N, ∫ σ in ((k : ℝ) / N)..(((k : ℝ) + 1) / N), g σ := by
    have h := intervalIntegral.sum_integral_adjacent_intervals (f := g) (μ := volume)
      (a := fun k : ℕ => (k : ℝ) / N) (n := N) (fun k _ => hg.intervalIntegrable _ _)
    simp only [Nat.cast_zero, zero_div, Nat.cast_add, Nat.cast_one] at h
    rw [div_self hNpos.ne'] at h
    exact h.symm
  have hterm : ∀ k ∈ Finset.range N,
      ‖(g (((k : ℝ) + 1) / N)) / N - ∫ σ in ((k : ℝ) / N)..(((k : ℝ) + 1) / N), g σ‖
        ≤ ε / 2 / N := by
    intro k hk
    have hkN : (k : ℝ) + 1 ≤ N := by exact_mod_cast Finset.mem_range.1 hk
    have hab : (k : ℝ) / N ≤ ((k : ℝ) + 1) / N := by gcongr; linarith
    have hlen : ((k : ℝ) + 1) / N - (k : ℝ) / N = 1 / N := by ring
    have hc : ∫ σ in ((k : ℝ) / N)..(((k : ℝ) + 1) / N), g (((k : ℝ) + 1) / N)
        = g (((k : ℝ) + 1) / N) / N := by
      rw [intervalIntegral.integral_const, smul_eq_mul, hlen]
      ring
    have hdiff : g (((k : ℝ) + 1) / N) / N - ∫ σ in ((k : ℝ) / N)..(((k : ℝ) + 1) / N), g σ
        = -∫ σ in ((k : ℝ) / N)..(((k : ℝ) + 1) / N), (g σ - g (((k : ℝ) + 1) / N)) := by
      rw [intervalIntegral.integral_sub (hg.intervalIntegrable _ _) intervalIntegrable_const, hc]
      ring
    rw [hdiff, norm_neg]
    have hle := intervalIntegral.norm_integral_le_of_norm_le_const
      (a := (k : ℝ) / N) (b := ((k : ℝ) + 1) / N) (C := ε / 2)
      (f := fun σ => g σ - g (((k : ℝ) + 1) / N)) (fun σ hσ => by
        rw [Set.uIoc_of_le hab] at hσ
        have hk0 : (0 : ℝ) ≤ (k : ℝ) / N := by positivity
        have hk1 : ((k : ℝ) + 1) / N ≤ 1 := by rw [div_le_one hNpos]; exact hkN
        have hσ0 : σ ∈ Icc (0 : ℝ) 1 := ⟨hk0.trans hσ.1.le, hσ.2.trans hk1⟩
        have hσ1 : ((k : ℝ) + 1) / N ∈ Icc (0 : ℝ) 1 := ⟨hk0.trans hab, hk1⟩
        have hd : dist σ (((k : ℝ) + 1) / N) < δ := by
          rw [Real.dist_eq, abs_of_nonpos (by linarith [hσ.2])]
          linarith [hσ.1, hlen, hNδ]
        have := hδg σ hσ0 _ hσ1 hd
        rw [Real.dist_eq] at this
        exact this.le)
    calc _ ≤ ε / 2 * |((k : ℝ) + 1) / N - (k : ℝ) / N| := hle
      _ = ε / 2 / N := by rw [hlen, abs_of_pos (by positivity)]; ring
  rw [hsum, Real.dist_eq, Fin.sum_univ_eq_sum_range (fun k : ℕ => g (((k : ℝ) + 1) / N)) N,
    Finset.mul_sum, ← Finset.sum_sub_distrib]
  have hcard : ∑ k ∈ Finset.range N, ε / 2 / (N : ℝ) = ε / 2 := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    field_simp
  calc |∑ k ∈ Finset.range N, ((N : ℝ)⁻¹ * g (((k : ℝ) + 1) / N)
        - ∫ σ in ((k : ℝ) / N)..(((k : ℝ) + 1) / N), g σ)|
      ≤ ∑ k ∈ Finset.range N, ε / 2 / (N : ℝ) := by
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k hk => ?_)
        have := hterm k hk
        rw [Real.norm_eq_abs] at this
        rwa [inv_mul_eq_div]
    _ = ε / 2 := hcard
    _ < ε := half_lt_self hε

end Kinetic
end Transformer
