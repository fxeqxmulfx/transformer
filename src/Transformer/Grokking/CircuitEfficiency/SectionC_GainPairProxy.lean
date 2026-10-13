import Transformer.Grokking.CircuitEfficiency.SectionC_GainRelativePath

/-!
# Exact unequal-pair error relative to a balanced native proxy

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
factors and appendix C's product partials; true zero-beta native
partner update and relative-balance estimates at lab commit 1b6785b.

Keep the same current CE scale when evaluating the balanced proxy at
the pair's mean factor. The actual-form pair sum equals this proxy
sum minus an explicit nonnegative correction. This is not a new
native trajectory: re-evaluating CE/clipping on a balanced state
would generally change the shared scale and is not done here.

The correction is at most gain^2*difference^2/(2*epsilon^2).
On a finite physical box, its ratio to positive current pair mass
is at most gain^2*ceiling/epsilon^2 times squared relative asymmetry.
The initialized actual relative-balance result can therefore control
this error without individual parameter limits or a future reference.

These current scalar identities/bounds do not yet prove efficient
Gen/Mem crossing or a successful unequal-seed held-out tail. They
use exact reals and the actual-form zero-beta normalization, with
uniform native decay differing from coupled-cost GD and GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Exact normalized-input correction from unequal factors. Sources:
appendix C product partials and native update at 1b6785b; this number
depends only on the present scale/factors, not a future path property. -/
noncomputable def gainPairBalancingError (gain scale eps a b : ℝ) : ℝ :=
  eps * (scale * gain) ^ 2 * (a - b) ^ 2 /
    (2 * (scale * gain * a + eps) * (scale * gain * b + eps) *
      (scale * gain * ((a + b) / 2) + eps))

/-- Actual-form pair mass is the same-scale balanced proxy minus
its exact unequal-factor error. Sources: appendix C partner partials
and native normalization at 1b6785b; the CE scale is not recomputed. -/
theorem gain_pair_step_balanced_proxy (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2 =
      2 * gainBalancedFactorStep gain scale eps decay rate ((a + b) / 2) -
        rate * gainPairBalancingError gain scale eps a b := by
  have hda : 0 < scale * gain * a + eps := by positivity
  have hdb : 0 < scale * gain * b + eps := by positivity
  have hdm : 0 < scale * gain * ((a + b) / 2) + eps := by positivity
  unfold gainPairStep gainBalancedFactorStep gainPairBalancingError
  dsimp only
  generalize hdae : scale * gain * a + eps = da at hda ⊢
  generalize hdbe : scale * gain * b + eps = db at hdb ⊢
  generalize hdme : scale * gain * ((a + b) / 2) + eps = dm at hdm ⊢
  field_simp [ne_of_gt hda, ne_of_gt hdb, ne_of_gt hdm]
  rw [← hdae, ← hdbe, ← hdme]
  ring

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Nonnegative current factors make the exact balancing correction
nonnegative. Source: appendix C partner normalization at 1b6785b;
the difference is squared, including either factor ordering. -/
theorem gain_pair_balancing_error_nonneg (gain scale eps a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    0 ≤ gainPairBalancingError gain scale eps a b := by
  unfold gainPairBalancingError
  positivity

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- The same-scale balanced proxy bounds actual-form pair mass above.
Source: the native correction identity at 1b6785b; this Jensen-type
comparison keeps the original current CE rather than changing forward. -/
theorem gain_pair_step_balanced_proxy_ceiling (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (heta : 0 ≤ rate) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2 ≤
      2 * gainBalancedFactorStep gain scale eps decay rate ((a + b) / 2) := by
  rw [gain_pair_step_balanced_proxy gain scale eps decay rate a b hg hs he ha hb]
  have hnn := mul_nonneg heta (gain_pair_balancing_error_nonneg gain scale eps a b hg hs he ha hb)
  linarith only [hnn]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Actual-form unit CE scale bounds the balancing correction by a
quadratic difference. Source: positive-epsilon native partner inputs
at 1b6785b; no mass limit or supplied small asymmetry is assumed. -/
theorem gain_pair_balancing_error_ceiling (gain scale eps a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hu : scale ≤ 1) (he : 0 < eps) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    gainPairBalancingError gain scale eps a b ≤ gain ^ 2 * (a - b) ^ 2 / (2 * eps ^ 2) := by
  have hda : 0 < scale * gain * a + eps := by positivity
  have hdb : 0 < scale * gain * b + eps := by positivity
  have hdm : 0 < scale * gain * ((a + b) / 2) + eps := by positivity
  have hsa : 0 ≤ scale * gain * a := by positivity
  have hsb : 0 ≤ scale * gain * b := by positivity
  have hsm : 0 ≤ scale * gain * ((a + b) / 2) := by positivity
  have hfa : eps ≤ scale * gain * a + eps := by linarith only [hsa]
  have hfb : eps ≤ scale * gain * b + eps := by linarith only [hsb]
  have hfm : eps ≤ scale * gain * ((a + b) / 2) + eps := by
    linarith only [hsm]
  have hab := mul_le_mul hfa hfb (le_of_lt he) (le_of_lt hda)
  have habm := mul_le_mul hab hfm (le_of_lt he) (mul_nonneg (le_of_lt hda) (le_of_lt hdb))
  have hfloor : 2 * eps ^ 3 ≤ 2 * (scale * gain * a + eps) * (scale * gain * b + eps) *
      (scale * gain * ((a + b) / 2) + eps) := by
    calc
      _ = 2 * ((eps * eps) * eps) := by ring
      _ ≤ 2 * (((scale * gain * a + eps) * (scale * gain * b + eps)) *
          (scale * gain * ((a + b) / 2) + eps)) := mul_le_mul_of_nonneg_left habm (by norm_num)
      _ = _ := by ring
  have hscaleSq : scale ^ 2 ≤ 1 := by nlinarith only [hs, hu]
  have hscaled := mul_le_mul_of_nonneg_right hscaleSq (mul_nonneg (sq_nonneg gain) (sq_nonneg (a - b)))
  have hnum : eps * (scale * gain) ^ 2 * (a - b) ^ 2 ≤ eps * gain ^ 2 * (a - b) ^ 2 := by
    nlinarith only [mul_le_mul_of_nonneg_left hscaled (le_of_lt he)]
  have hden : 0 < 2 * (scale * gain * a + eps) * (scale * gain * b + eps) *
      (scale * gain * ((a + b) / 2) + eps) := by positivity
  unfold gainPairBalancingError
  apply (div_le_div_iff₀ hden (by positivity)).mpr
  have hleft := mul_le_mul_of_nonneg_right hnum (show 0 ≤ 2 * eps ^ 2 by positivity)
  have hright := mul_le_mul_of_nonneg_left hfloor (mul_nonneg (sq_nonneg gain) (sq_nonneg (a - b)))
  nlinarith only [hleft, hright]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Error per positive current mass is controlled by squared relative
asymmetry and the finite box. Source: the native quadratic correction
at 1b6785b; total mass may itself approach zero and is not bounded below. -/
theorem gain_pair_balancing_error_relative_ceiling (gain scale eps ceiling a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hu : scale ≤ 1) (he : 0 < eps)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hac : a ≤ ceiling) (hbc : b ≤ ceiling) (hmass : 0 < a + b) :
    gainPairBalancingError gain scale eps a b / (a + b) ≤
      (gain ^ 2 * ceiling / eps ^ 2) * gainPairRelativeDifference a b ^ 2 := by
  have herror := gain_pair_balancing_error_ceiling gain scale eps a b hg hs hu he ha hb
  have hsum : a + b ≤ 2 * ceiling := by linarith only [hac, hbc]
  have hfactor : 0 ≤ gain ^ 2 / (2 * eps ^ 2) * gainPairRelativeDifference a b ^ 2 := by positivity
  have hweighted := mul_le_mul_of_nonneg_left hsum hfactor
  calc
    _ ≤ (gain ^ 2 * (a - b) ^ 2 / (2 * eps ^ 2)) / (a + b) :=
      div_le_div_of_nonneg_right herror (le_of_lt hmass)
    _ = gain ^ 2 / (2 * eps ^ 2) * gainPairRelativeDifference a b ^ 2 * (a + b) := by
      unfold gainPairRelativeDifference
      rw [div_pow, sq_abs]
      field_simp [ne_of_gt hmass, ne_of_gt he]
    _ ≤ gain ^ 2 / (2 * eps ^ 2) * gainPairRelativeDifference a b ^ 2 * (2 * ceiling) := hweighted
    _ = _ := by ring

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 2 ∧ (1 : ℝ) ≤ 2 ∧ (0 : ℝ) < 0 + 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
