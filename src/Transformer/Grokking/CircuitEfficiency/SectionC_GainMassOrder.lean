import Transformer.Grokking.CircuitEfficiency.SectionC_GainUnequalEscape

/-!
# Ordered balanced mass proxies and a positive unequal-pair error tolerance

Sources: Varma et al., arXiv:2309.02390v1, section 3's efficiency
competition and appendix C's product inputs; original current-CE
native mass laws and initialized estimates at lab commit 68d90fa.

When Gen mass is already at least Mem mass, the Gen balanced input
at that larger mass is no smaller than its input at Mem mass. At
the common smaller mass, the physical efficiency gap supplies a
strict positive relative-rate advantage. Hence the Gen-minus-Mem
balanced-proxy update is bounded below by remaining decay times
their mass difference plus rate times the gap times Mem mass.

Unequal Gen factors subtract their exact balancing correction. The
positive tolerance here is at most half the gap and, after multiplying
by rate, at most half the remaining decay. It can be generated along
an initialized actual path by the already-proved relative-error limit.
This prepares a strict order-preserving actual unequal update without
assuming convergence of individual parameters or positive mass limits.

Mass order does not assert factor-by-factor order. The comparison
uses only pair sums and then corrects the true unequal inputs.
Mem's nonnegative balancing error helps the Gen-minus-Mem bound.
Absolute masses can tend to zero; the tolerance concerns correction
divided by the current positive Gen mass.

All proxies use the original shared current CE multiplier. They do
not define another feedback trajectory or assume a successful learned
reference. Exact reals, zero betas, fixed gained tables and uniform
native decay differ from appendix C's coupled assigned norm/GD and
the preserved nonzero-beta learned GPTMini configuration.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Balanced pair normalized input grows with nonnegative current
mass at fixed original scale/gain. Sources: appendix C partner
partials and native epsilon normalization at 68d90fa; no future
path property is built into this scalar monotonicity statement. -/
theorem gain_balanced_mass_input_monotone (gain scale eps u v : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (hu : 0 ≤ u) (horder : u ≤ v) :
    u * gainBalancedRelativeRate gain scale eps (u / 2) ≤
      v * gainBalancedRelativeRate gain scale eps (v / 2) := by
  have hv : 0 ≤ v := le_trans hu horder
  have hdu : 0 < scale * gain * (u / 2) + eps := by positivity
  have hdv : 0 < scale * gain * (v / 2) + eps := by positivity
  have hdiff : 0 ≤ v - u := by linarith only [horder]
  have hweighted := mul_nonneg (mul_nonneg (mul_nonneg hs hg) (le_of_lt he)) hdiff
  have huGroup : u * gainBalancedRelativeRate gain scale eps (u / 2) =
      (scale * gain * u) / (scale * gain * (u / 2) + eps) := by
    unfold gainBalancedRelativeRate
    ring
  have hvGroup : v * gainBalancedRelativeRate gain scale eps (v / 2) =
      (scale * gain * v) / (scale * gain * (v / 2) + eps) := by
    unfold gainBalancedRelativeRate
    ring
  rw [huGroup, hvGroup]
  apply (div_le_div_iff₀ hdu hdv).mpr
  nlinarith only [hweighted]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 := by norm_num

/-- Ordered pair masses give this quantitative balanced-proxy
advantage at the original common scale. Sources: section 3 efficiency
and native product normalization at 68d90fa; remaining decay need
not be positive for this exact lower difference estimate. -/
theorem gain_balanced_mass_proxy_order_margin (lower scale eps genGain memGain ceiling decay rate u v : ℝ)
    (hlower : 0 < lower) (hscale : lower ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (heta : 0 ≤ rate)
    (hv : 0 < v) (horder : v ≤ u) (hceiling : v ≤ 2 * ceiling) :
    0 < gainBalancedRelativeGap lower eps genGain memGain ceiling ∧
      (1 - rate * decay) * (u - v) + rate * gainBalancedRelativeGap lower eps genGain memGain ceiling * v ≤
        2 * gainBalancedFactorStep genGain scale eps decay rate (u / 2) -
          2 * gainBalancedFactorStep memGain scale eps decay rate (v / 2) := by
  have hs := lt_of_lt_of_le hlower hscale
  have hg := lt_trans hmem hgain
  have hmean : 0 ≤ v / 2 := by positivity
  have hmeanBox : v / 2 ≤ ceiling := by linarith only [hceiling]
  have hgap := gain_balanced_relative_gap_floor lower scale eps genGain memGain ceiling (v / 2) (v / 2)
    hlower hscale hunit he hmem hgain hmean (le_refl _) hmeanBox
  have hinput := gain_balanced_mass_input_monotone genGain scale eps v u
    (le_of_lt hg) (le_of_lt hs) he (le_of_lt hv) horder
  have hi := mul_le_mul_of_nonneg_left hinput heta
  have hd := mul_le_mul_of_nonneg_left hgap.2 (mul_nonneg heta (le_of_lt hv))
  refine ⟨hgap.1, ?_⟩
  rw [gain_balanced_factor_step_multiplier, gain_balanced_factor_step_multiplier]
  nlinarith only [hi, hd]

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 2 ∧
    (2 : ℝ) ≤ 3 ∧ (2 : ℝ) ≤ 2 * 2 := by norm_num

/-- A positive original-scale efficiency gap and nonnegative
remaining decay make ordered balanced mass proxies strictly ordered.
Sources: appendix C product inputs and the native margin estimate
at 68d90fa; this current conclusion is not a future convergence claim. -/
theorem gain_balanced_mass_proxy_order_strict (lower scale eps genGain memGain ceiling decay rate u v : ℝ)
    (hlower : 0 < lower) (hscale : lower ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hv : 0 < v) (horder : v ≤ u) (hceiling : v ≤ 2 * ceiling) :
    2 * gainBalancedFactorStep memGain scale eps decay rate (v / 2) <
      2 * gainBalancedFactorStep genGain scale eps decay rate (u / 2) := by
  have hm := gain_balanced_mass_proxy_order_margin lower scale eps genGain memGain ceiling decay rate u v
    hlower hscale hunit he hmem hgain (le_of_lt heta) hv horder hceiling
  have hdiff : 0 ≤ u - v := by linarith only [horder]
  have hbase := mul_nonneg hd hdiff
  have hboost := mul_pos (mul_pos heta hm.1) hv
  linarith only [hm.2, hbase, hboost]

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (2 : ℝ) ≤ 2 * 2 := by norm_num

/-- Numerical tolerance for preserving unequal current mass order.
Sources: appendix C efficiency and native balancing error at 68d90fa;
the expression depends only on fixed floor/box/native constants. -/
noncomputable def gainUnequalOrderTolerance (lower eps genGain memGain ceiling decay rate : ℝ) : ℝ :=
  min (gainBalancedRelativeGap lower eps genGain memGain ceiling / 2) ((1 - rate * decay) / (2 * rate))

/-- The order tolerance is positive, at most half the efficiency gap,
and after multiplying by rate at most half the remaining decay.
Sources: actual native mass estimates at 68d90fa; positivity is a
static nonempty condition, without an assumed successful path. -/
theorem gain_unequal_order_tolerance_bounds (lower eps genGain memGain ceiling decay rate : ℝ)
    (hlower : 0 < lower) (he : 0 < eps) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hc : 0 ≤ ceiling) (heta : 0 < rate) (hd : 0 < 1 - rate * decay) :
    0 < gainUnequalOrderTolerance lower eps genGain memGain ceiling decay rate ∧
      gainUnequalOrderTolerance lower eps genGain memGain ceiling decay rate ≤
        gainBalancedRelativeGap lower eps genGain memGain ceiling / 2 ∧
      rate * gainUnequalOrderTolerance lower eps genGain memGain ceiling decay rate ≤ (1 - rate * decay) / 2 := by
  have hg := lt_trans hmem hgain
  have hgap : 0 < gainBalancedRelativeGap lower eps genGain memGain ceiling := by
    unfold gainBalancedRelativeGap
    positivity
  have hright := min_le_right (gainBalancedRelativeGap lower eps genGain memGain ceiling / 2)
    ((1 - rate * decay) / (2 * rate))
  have heq : rate * ((1 - rate * decay) / (2 * rate)) = (1 - rate * decay) / 2 := by
    field_simp [ne_of_gt heta]
  refine ⟨lt_min (div_pos hgap (by norm_num)) (div_pos hd (by positivity)), min_le_left _ _, ?_⟩
  calc
    _ ≤ rate * ((1 - rate * decay) / (2 * rate)) := mul_le_mul_of_nonneg_left hright (le_of_lt heta)
    _ = _ := heq

example : (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) := by norm_num

end Transformer.Grokking.CircuitEfficiency
