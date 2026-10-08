import Transformer.Grokking.CircuitEfficiency.SectionC_GainProxyLimits
import Transformer.Grokking.CircuitEfficiency.SectionC_GainRatioGrowth

/-!
# Unequal native pair mass rates at the actual current CE scale

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
compositional factors and appendix C's product partials; retained
native zero-beta normalization and balancing error at commit d655756.

The exact unequal-pair law is a current mass times the balanced
relative multiplier, minus rate times the nonnegative balancing
error. Divide by the current positive mass to expose the error
per unit mass. This quantity already tends to zero on initialized
actual native paths; no limiting individual weights are needed.

The sufficient small-rate condition gives strictly positive remaining
decay, rather than merely nonnegative remaining decay. It follows
that positive current pair mass stays positive after an actual-form
step, even if one current factor is zero. The instantaneous mass
multiplier is bounded above by the fixed native epsilon ceiling.
No retained momentum is replaced or reset in the native specialization.

These scalar statements retain the present CE multiplier as an
argument. Their subsequent actual-path use must derive that multiplier
and its bounds from the original closed feedback. They concern exact
reals, zero betas and fixed tables; appendix C used coupled-cost GD,
and preserved learned GPTMini uses nonzero betas. A current mass rate
is neither a positive mass limit nor a held-out success certificate.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- The explicit sufficient native small rate leaves a strictly
positive pure-decay multiplier. Sources: actual zero-beta estimates
at d655756 and appendix C's positive gained partner inputs; no sign
or convergence of a future trajectory is assumed. -/
theorem gain_small_rate_decay_remaining_positive (gain eps decay rate : ℝ)
    (hg : 0 < gain) (he : 0 < eps) (heta : 0 < rate)
    (hsmall : rate * (decay + gain / eps) ≤ 1) :
    0 < 1 - rate * decay := by
  have hpart := mul_pos heta (div_pos hg he)
  nlinarith only [hsmall, hpart]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- Unequal current mass obeys the same-scale balanced relative
multiplier with its exact additive correction. Sources: appendix C
partner partials and the native proxy identity at d655756; the
original present CE scale is kept rather than recomputed. -/
theorem gain_pair_step_mass_multiplier (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2 =
      (a + b) * (1 - rate * decay + rate * gainBalancedRelativeRate gain scale eps ((a + b) / 2)) -
        rate * gainPairBalancingError gain scale eps a b := by
  rw [gain_pair_step_balanced_proxy gain scale eps decay rate a b hg hs he ha hb,
    gain_balanced_factor_step_multiplier]
  ring

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- The actual-form relative pair mass multiplier subtracts precisely
rate times the balancing error per unit mass. Sources: appendix C's
product dynamics and native proxy law at d655756; current mass is
positive, without a uniform future lower bound or parameter limit. -/
theorem gain_pair_step_relative_mass_multiplier (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hmass : 0 < a + b) :
    ((gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2) / (a + b) =
      1 - rate * decay + rate * gainBalancedRelativeRate gain scale eps ((a + b) / 2) -
        rate * (gainPairBalancingError gain scale eps a b / (a + b)) := by
  rw [gain_pair_step_mass_multiplier gain scale eps decay rate a b hg hs he ha hb]
  field_simp [ne_of_gt hmass]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 := by norm_num

/-- A current positive pair mass remains positive at the next
actual-form native step, including a single zero factor. Sources:
section 3's zero-first-factor seeds and the native update at d655756;
both normalized partner increments are derived nonnegative. -/
theorem gain_pair_step_mass_positive (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (heta : 0 ≤ rate)
    (hd : 0 < 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hmass : 0 < a + b) :
    0 < (gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2 := by
  have hda : 0 < scale * gain * a + eps := by positivity
  have hdb : 0 < scale * gain * b + eps := by positivity
  have hia : 0 ≤ rate * (scale * gain * a) / (scale * gain * a + eps) := by positivity
  have hib : 0 ≤ rate * (scale * gain * b) / (scale * gain * b + eps) := by positivity
  have hbase := mul_pos hd hmass
  unfold gainPairStep
  dsimp only
  nlinarith only [hbase, hia, hib]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 := by norm_num

/-- Actual-form current mass growth has the fixed native epsilon
ceiling, despite unequal factors. Sources: appendix C product inputs
and the normalized native law at d655756; use the nonnegative
correction and the current balanced relative-rate ceiling. -/
theorem gain_pair_step_relative_mass_ceiling (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hu : scale ≤ 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hmass : 0 < a + b) :
    ((gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2) / (a + b) ≤
      1 - rate * decay + rate * gain / eps := by
  have hf : 0 ≤ (a + b) / 2 := by positivity
  have hrate := gain_balanced_relative_rate_ceiling gain scale eps ((a + b) / 2) hg hs hu he hf
  have hweighted := mul_le_mul_of_nonneg_left hrate heta
  have herror := div_nonneg (gain_pair_balancing_error_nonneg gain scale eps a b hg hs he ha hb) (le_of_lt hmass)
  have hsub := mul_nonneg heta herror
  rw [gain_pair_step_relative_mass_multiplier gain scale eps decay rate a b hg hs he ha hb hmass]
  have hgroup : rate * (gain / eps) = rate * gain / eps := by ring
  rw [hgroup] at hweighted
  linarith only [hweighted, hsub]

example : (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 := by norm_num

/-- An explicit current error budget yields a lower next mass
multiplier. Sources: appendix C's product partials and the exact
native proxy at d655756; this is a pointwise estimate. Its future
use must generate the error budget from actual initialized balance,
rather than supply it as a successful path premise. No positive
limiting mass is assumed; the bound compares relative multipliers
even when the absolute present mass is small or decreasing. -/
theorem gain_pair_step_mass_floor_from_relative_error (gain scale eps decay rate a b allowance : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (heta : 0 ≤ rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hmass : 0 < a + b)
    (herror : gainPairBalancingError gain scale eps a b / (a + b) ≤ allowance) :
    (a + b) * (1 - rate * decay + rate * gainBalancedRelativeRate gain scale eps ((a + b) / 2) -
      rate * allowance) ≤
        (gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2 := by
  have heq := gain_pair_step_relative_mass_multiplier gain scale eps decay rate a b hg hs he ha hb hmass
  have hw := mul_le_mul_of_nonneg_left herror heta
  have hbound : 1 - rate * decay + rate * gainBalancedRelativeRate gain scale eps ((a + b) / 2) -
      rate * allowance ≤
        ((gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2) / (a + b) := by
    linarith only [heq, hw]
  simpa only [mul_comm] using (le_div_iff₀ hmass).mp hbound

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 / 2 + 1 / 2 ∧
    gainPairBalancingError 3 (1 / 2) 1 (1 / 2) (1 / 2) / (1 / 2 + 1 / 2) ≤ 1 / 10 := by
  norm_num [gainPairBalancingError]

end Transformer.Grokking.CircuitEfficiency
