/-
# The Zipf-law asymptotic consequences

arXiv:2402.19449v2, Appendix H.
-/

import Transformer.Imbalance.Section3_ZipfLimits

open Filter Asymptotics
open scoped Topology

noncomputable section

namespace Transformer.Imbalance

/-- The first ceil(c/log(c)^2) classes account for mass tending to one;
Appendix H. This uses the numerator's cutoff, correcting the source's
displayed sum, which instead has ceil(c/log(c)). -/
theorem zipf_partial_mass_tendsto :
    Tendsto (fun c => zipfPartialMass c (zipfCutoff c)) atTop (𝓝 1) := by
  have hne : ∀ᶠ c : ℕ in atTop, harmonicMass c ≠ 0 := by
    filter_upwards [eventually_gt_atTop 0] with c hc
    exact (harmonicMass_pos c hc).ne'
  have hratio := (isEquivalent_iff_tendsto_one hne).mp zipf_harmonic_cutoff_equiv
  apply hratio.congr'
  filter_upwards with c
  simp only [Pi.div_apply, zipfPartialMass]
  rw [Nat.min_eq_left (show zipfCutoff c ≤ c from Nat.min_le_left _ _)]

/-- Even the least frequent rank in the rounded Zipf prefix has c*pi_k
diverging; Appendix H. The conclusion uses the exact harmonic/ceiling
bound, correcting the source's literal inequality c*pi_k >= log(c). -/
theorem zipf_prefix_min_frequency_tendsto :
    Tendsto (fun c : ℕ => (c : ℝ) / ((zipfCutoff c : ℝ) * harmonicMass c))
      atTop atTop :=
  zipf_prefix_min_frequency_equiv.symm.tendsto_atTop zipf_log_tendsto

/-- Appendix H's Zipf consequence, with the cutoff typo and literal
frequency lower bound corrected: the first c/log(c)^2 classes account for
mass tending to one, and even the least frequent class in this prefix
has c*pi_k tending to infinity. Both limits retain the harmonic
normalization and integer rounding rather than discarding their errors. -/
theorem zipf_cutoff_asymptotics :
    Tendsto (fun c => zipfPartialMass c (zipfCutoff c)) atTop (𝓝 1) ∧
    Tendsto (fun c : ℕ => (c : ℝ) / ((zipfCutoff c : ℝ) * harmonicMass c)) atTop atTop := by
  exact ⟨zipf_partial_mass_tendsto, zipf_prefix_min_frequency_tendsto⟩

end Transformer.Imbalance
