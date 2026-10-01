/-
# Relative asymptotics for the correlation formula

arXiv:2402.19449v2, Proposition 2, equation (2), Appendix H.
The ratio limit proved for actual datasets gives the source's asymptotic
equivalence when its coefficient is nonzero. The cπ_k domination condition
also gives exactly the relative-error criterion used in that proof.
-/

import Transformer.Imbalance.Section3_Correlation
import Mathlib.Analysis.Asymptotics.AsymptoticEquivalent

open Filter Asymptotics
open scoped Topology

namespace Transformer.Imbalance

/-- Appendix H's frequency condition converts q=C/c into q/π_k -> 0.
The functions are class count and class frequency, and C is the fixed
off-class probability bound. -/
theorem frequency_domination_criterion (c : ℕ → ℕ) (f : ℕ → ℝ) (C : ℝ)
    (h : Tendsto (fun N => 1 / ((c N : ℝ) * f N)) atTop (𝓝 0)) :
    Tendsto (fun N => (C / (c N : ℝ)) / f N) atTop (𝓝 0) := by
  have hh := tendsto_const_nhds (x := C) |>.mul h
  convert hh using 1
  · funext N
    rw [div_div]
    ring
  · simp

/-- Nonvacuity of the relative-frequency criterion; Appendix H. -/
example : Tendsto (fun N : ℕ => 1 / (((N + 1 : ℕ) : ℝ) * 1)) atTop (𝓝 0) := by
  have h : Tendsto (fun N : ℕ => (N : ℝ) + 1) atTop atTop :=
    tendsto_natCast_atTop_atTop.atTop_add tendsto_const_nhds
  convert h.inv_tendsto_atTop using 1
  funext N
  simp

/-- A nonzero limiting gradient/trace ratio a yields G ~ a*T, the
asymptotic-equivalence form used in Proposition 2, equation (2).
For the dataset theorem, a=norm(μ)/(pH); its nonzero requirement is the
missing nondegenerate-mean condition in the source's literal ~ assertion. -/
theorem correlation_equivalence (G T : ℕ → ℝ) (a : ℝ) (ha : a ≠ 0)
    (h : Tendsto (fun N => G N / T N) atTop (𝓝 a)) :
    IsEquivalent atTop G (fun N => a * T N) := by
  apply isEquivalent_of_tendsto_one
  have hh := h.div_const a
  rw [div_self ha] at hh
  convert hh using 1
  funext N
  simp only [Pi.div_apply, div_div]
  rw [mul_comm]

/-- Nonvacuity of the nonzero correlation coefficient and ratio limit;
Proposition 2's analytic conversion. -/
example : (2 : ℝ) ≠ 0 ∧ Tendsto (fun _ : ℕ => (2 : ℝ) / 1) atTop (𝓝 2) := by
  refine ⟨by norm_num, ?_⟩
  simp only [div_one]
  exact tendsto_const_nhds

end Transformer.Imbalance
