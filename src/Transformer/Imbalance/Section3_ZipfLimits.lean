/-
# Harmonic, logarithmic, and rounding limits for the Zipf argument

arXiv:2402.19449v2, Appendix H's opening paragraph.
-/

import Transformer.Imbalance.Section3_ZipfBasic
import Mathlib.NumberTheory.Harmonic.EulerMascheroni
import Mathlib.Analysis.Asymptotics.SpecificAsymptotics

open Filter Asymptotics
open scoped Topology

noncomputable section

namespace Transformer.Imbalance

/-- The logarithm of the vocabulary size diverges; Appendix H. -/
theorem zipf_log_tendsto :
    Tendsto (fun c : ℕ => Real.log c) atTop atTop :=
  Real.tendsto_log_atTop.comp tendsto_natCast_atTop_atTop

/-- The harmonic normalizer is asymptotic to log(c), strengthening the
source's Theta(log(c)) estimate enough to justify its limit; Appendix H. -/
theorem harmonicMass_equiv_log :
    (fun c : ℕ => harmonicMass c) ~[atTop] (fun c : ℕ => Real.log c) := by
  apply isEquivalent_of_tendsto_one
  have h0 := Real.tendsto_harmonic_sub_log.div_atTop zipf_log_tendsto
  have h1 : Tendsto (fun c : ℕ =>
      ((harmonic c : ℝ) - Real.log c) / Real.log c + 1) atTop (𝓝 1) := by
    simpa using h0.add_const 1
  apply h1.congr'
  filter_upwards [zipf_log_tendsto.eventually_ne_atTop 0] with c hc
  simp only [Pi.div_apply, harmonicMass, sub_div, div_self hc]
  ring

/-- The unrounded cutoff c/log(c)^2 diverges; Appendix H. -/
theorem zipf_real_cutoff_tendsto :
    Tendsto (fun c : ℕ => (c : ℝ) / (Real.log c) ^ 2) atTop atTop := by
  have h := (Real.tendsto_exp_div_pow_atTop 2).comp zipf_log_tendsto
  apply h.congr'
  filter_upwards [(tendsto_natCast_atTop_atTop :
    Tendsto (fun c : ℕ => (c : ℝ)) atTop atTop).eventually_gt_atTop 0] with c hc
  simp only [Function.comp_apply, Real.exp_log hc]

/-- The iterated logarithm is negligible compared with log(c);
Appendix H's prefix-mass computation. -/
theorem zipf_log_log_div_log_tendsto :
    Tendsto (fun c : ℕ => Real.log (Real.log c) / Real.log c) atTop (𝓝 0) := by
  exact Real.isLittleO_log_id_atTop.tendsto_div_nhds_zero.comp zipf_log_tendsto

/-- Taking the logarithm of c/log(c)^2 preserves the leading log(c)
term; Appendix H's prefix-mass computation. -/
theorem zipf_log_real_cutoff_equiv :
    (fun c : ℕ => Real.log ((c : ℝ) / (Real.log c) ^ 2)) ~[atTop]
      (fun c : ℕ => Real.log c) := by
  apply isEquivalent_of_tendsto_one
  have h : Tendsto (fun c : ℕ => 1 - 2 * (Real.log (Real.log c) / Real.log c))
      atTop (𝓝 1) := by
    simpa using tendsto_const_nhds.sub (zipf_log_log_div_log_tendsto.const_mul 2)
  apply h.congr'
  filter_upwards [(tendsto_natCast_atTop_atTop :
    Tendsto (fun c : ℕ => (c : ℝ)) atTop atTop).eventually_gt_atTop 0,
    zipf_log_tendsto.eventually_gt_atTop 0] with c hc hl
  simp only [Pi.div_apply, Real.log_div hc.ne' (pow_ne_zero 2 hl.ne'), Real.log_pow]
  field_simp
  ring

/-- The ceiling of c/log(c)^2 eventually lies inside the vocabulary,
so the finite-size truncation no longer changes it; Appendix H. -/
theorem zipfCutoff_eventually_eq :
    (fun c => zipfCutoff c) =ᶠ[atTop]
      (fun c : ℕ => ⌈(c : ℝ) / (Real.log c) ^ 2⌉₊) := by
  filter_upwards [zipf_log_tendsto.eventually_ge_atTop 1] with c hc
  apply Nat.min_eq_right
  apply Nat.ceil_le.mpr
  have hpow : 1 ≤ (Real.log c) ^ 2 := by nlinarith
  apply (div_le_iff₀ (by linarith : 0 < (Real.log c) ^ 2)).mpr
  simpa using mul_le_mul_of_nonneg_left hpow (Nat.cast_nonneg (α := ℝ) c)

/-- Integer rounding does not change the leading cutoff c/log(c)^2;
Appendix H's prefix-mass argument. -/
theorem zipfCutoff_equiv :
    (fun c => (zipfCutoff c : ℝ)) ~[atTop]
      (fun c : ℕ => (c : ℝ) / (Real.log c) ^ 2) := by
  have h : (fun c : ℕ => (⌈(c : ℝ) / (Real.log c) ^ 2⌉₊ : ℝ)) ~[atTop]
      (fun c : ℕ => (c : ℝ) / (Real.log c) ^ 2) :=
    isEquivalent_nat_ceil.comp_tendsto zipf_real_cutoff_tendsto
  apply h.congr_left
  exact zipfCutoff_eventually_eq.symm.mono (fun c hc => congrArg Nat.cast hc)

/-- The number of classes in the retained prefix diverges; Appendix H. -/
theorem zipfCutoff_tendsto : Tendsto zipfCutoff atTop atTop := by
  apply (tendsto_natCast_atTop_iff (R := ℝ)).mp
  exact zipfCutoff_equiv.symm.tendsto_atTop zipf_real_cutoff_tendsto

/-- The logarithm of the rounded cutoff is still asymptotic to log(c);
Appendix H's prefix-mass computation. -/
theorem zipf_log_cutoff_equiv :
    (fun c => Real.log (zipfCutoff c)) ~[atTop]
      (fun c : ℕ => Real.log c) :=
  (zipfCutoff_equiv.log zipf_real_cutoff_tendsto).trans zipf_log_real_cutoff_equiv

/-- The prefix's harmonic normalizer is asymptotic to the full
vocabulary's normalizer; Appendix H's prefix-mass computation. -/
theorem zipf_harmonic_cutoff_equiv :
    (fun c => harmonicMass (zipfCutoff c)) ~[atTop] (fun c => harmonicMass c) :=
  ((harmonicMass_equiv_log.comp_tendsto zipfCutoff_tendsto).trans
    zipf_log_cutoff_equiv).trans harmonicMass_equiv_log.symm

/-- The product of the rounded rank cutoff and the harmonic normalizer
is asymptotic to c/log(c); Appendix H's frequency argument. -/
theorem zipf_cutoff_normalizer_equiv :
    (fun c => (zipfCutoff c : ℝ) * harmonicMass c) ~[atTop]
      (fun c : ℕ => (c : ℝ) / Real.log c) := by
  apply (zipfCutoff_equiv.mul harmonicMass_equiv_log).congr_right
  filter_upwards [zipf_log_tendsto.eventually_ne_atTop 0] with c hc
  simp only [Pi.mul_apply]
  field_simp

/-- The exact least-frequency bound grows as log(c), rather than
satisfying the manuscript's literal finite inequality c*pi_k >= log(c);
Appendix H's frequency argument. -/
theorem zipf_prefix_min_frequency_equiv :
    (fun c : ℕ => (c : ℝ) / ((zipfCutoff c : ℝ) * harmonicMass c)) ~[atTop]
      (fun c : ℕ => Real.log c) := by
  have hid : (fun c : ℕ => (c : ℝ)) ~[atTop] (fun c : ℕ => (c : ℝ)) :=
    IsEquivalent.refl
  apply (hid.div zipf_cutoff_normalizer_equiv).congr_right
  filter_upwards [(tendsto_natCast_atTop_atTop :
    Tendsto (fun c : ℕ => (c : ℝ)) atTop atTop).eventually_ne_atTop 0] with c hc
  simp only [Pi.div_apply]
  exact div_div_cancel₀ hc

end Transformer.Imbalance
