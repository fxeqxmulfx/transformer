import Transformer.Grokking.AdamW.ScalarRecurrence

/-!
# Noncollapse under a retained native gradient-sign history

Source: PyTorch 2.14.1 AdamW at lab commit 88aa892, parameter decay
and retained exponential buffers. The predicates below describe
numerical parameter/moment signs, not convergence or task success.
Show that an actual nonpositive gradient history preserves them,
and that a negative current gradient activates a zero parameter.

The next fixed-circuit module derives these signs from actual CE and
closes the history through its current factors. No supplied arbitrary
history is identified with GPTMini. Positive remaining decay factor is
explicit; neither unrestricted prescribed-rate descent nor convergence
to the coupled-penalty optimum is asserted.
-/

namespace Transformer.Grokking.AdamW

/-- Numerical nonnegative-parameter, nonpositive-moment and physical-
variance region. Source: native AdamW signs at 88aa892; this is a
predicate of a state, not an assumed theorem about training. -/
def NonnegativeScalarState (state : ScalarState) : Prop :=
  0 ≤ state.parameter ∧ state.moment ≤ 0 ∧ 0 ≤ state.variance

/-- Strictly positive parameter within the same moment-sign region.
Source: native AdamW at 88aa892, for seeded-factor noncollapse. -/
def PositiveScalarState (state : ScalarState) : Prop :=
  0 < state.parameter ∧ state.moment ≤ 0 ∧ 0 ≤ state.variance

/-- Nonnegative initialization satisfies the actual numeric sign
conditions. Source: native AdamW at 88aa892, zero initial buffers. -/
theorem seeded_scalar_nonnegative (parameter : ℝ) (hp : 0 ≤ parameter) :
    NonnegativeScalarState (seededScalarState parameter) := by
  exact ⟨hp, le_rfl, le_rfl⟩

example : (0 : ℝ) ≤ 1 / 200 := by norm_num

/-- A positive seed satisfies the stronger numeric region without
requiring a preexisting gradient. Source: native AdamW at 88aa892. -/
theorem seeded_scalar_positive (parameter : ℝ) (hp : 0 < parameter) :
    PositiveScalarState (seededScalarState parameter) := by
  exact ⟨hp, le_rfl, le_rfl⟩

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- Strict positivity implies nonnegativity of the same retained
state. Source: native sign-region predicates at 88aa892. -/
theorem positive_scalar_nonnegative (state : ScalarState) (hp : PositiveScalarState state) :
    NonnegativeScalarState state := by
  exact ⟨le_of_lt hp.1, hp.2⟩

example : PositiveScalarState (seededScalarState (1 / 200)) := by
  exact seeded_scalar_positive _ (by norm_num)

/-- Retained adaptation cannot reduce the parameter below the pure
decay contribution under a nonpositive numerator history. Source:
native AdamW at 88aa892; this bound does not say the parameter is
monotone when decay is positive. -/
theorem scalar_parameter_decay_floor (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hm : state.moment ≤ 0) (hg : gradient ≤ 0) :
    (1 - rate * decay) * state.parameter ≤
      (scalarNativeStep b1 b2 eps decay rate state gradient).parameter := by
  have hd := retained_direction_nonpos b1 b2 eps gradient state hb h1 he hm hg
  change (1 - rate * decay) * state.parameter ≤ (1 - rate * decay) * state.parameter - rate * _
  nlinarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ (seededScalarState 1).moment ≤ 0 ∧ (-1 : ℝ) ≤ 0 := by
  norm_num [seededScalarState]

/-- A negative current gradient contributes strictly more than pure
decay at a positive rate. Source: native AdamW at 88aa892; old
parameter sign and monotonicity of the loss are not assumed. -/
theorem scalar_strict_parameter_decay_floor (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hm : state.moment ≤ 0) (hg : gradient < 0) :
    (1 - rate * decay) * state.parameter <
      (scalarNativeStep b1 b2 eps decay rate state gradient).parameter := by
  have hd := retained_direction_negative b1 b2 eps gradient state hb h1 he hm hg
  change (1 - rate * decay) * state.parameter < (1 - rate * decay) * state.parameter - rate * _
  nlinarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 < (1 / 1000 : ℝ) ∧ (seededScalarState 1).moment ≤ 0 ∧ (-1 : ℝ) < 0 := by
  norm_num [seededScalarState]

/-- All numeric sign conditions persist under a nonpositive current
gradient and nonnegative remaining decay factor. Source: native AdamW
at 88aa892; validity of variance and first moment is proved separately
from parameter sign, rather than placed in the update definition. -/
theorem scalar_nonnegative_step (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 ≤ 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeScalarState state) (hg : gradient ≤ 0) :
    NonnegativeScalarState (scalarNativeStep b1 b2 eps decay rate state gradient) := by
  have hf := scalar_parameter_decay_floor b1 b2 eps decay rate gradient state hb1 h1 he heta hs.2.1 hg
  refine ⟨le_trans (mul_nonneg hd hs.1) hf, ?_, ?_⟩
  · exact scalar_native_moment_nonpos b1 b2 eps decay rate gradient state hb1 (le_of_lt h1) hs.2.1 hg
  · exact scalar_native_variance_nonneg b1 b2 eps decay rate gradient state hb2 h2 hs.2.2

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ 0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeScalarState (seededScalarState 1) ∧ (-1 : ℝ) ≤ 0 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, seeded_scalar_nonnegative _ (by norm_num), by norm_num⟩

/-- Positive seeded parameters cannot collapse while the remaining
decay factor is positive and the native gradient-sign region persists.
Source: native AdamW at 88aa892; no lower bound independent of time
or convergence to a particular circuit allocation is claimed. -/
theorem scalar_positive_step (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 ≤ 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 < 1 - rate * decay)
    (hs : PositiveScalarState state) (hg : gradient ≤ 0) :
    PositiveScalarState (scalarNativeStep b1 b2 eps decay rate state gradient) := by
  have hn := scalar_nonnegative_step b1 b2 eps decay rate gradient state hb1 h1 hb2 h2 he heta
    (le_of_lt hd) (positive_scalar_nonnegative state hs) hg
  have hf := scalar_parameter_decay_floor b1 b2 eps decay rate gradient state hb1 h1 he heta hs.2.1 hg
  exact ⟨lt_of_lt_of_le (mul_pos hd hs.1) hf, hn.2⟩

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ 0 ≤ (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    PositiveScalarState (seededScalarState (1 / 200)) ∧ (-1 : ℝ) ≤ 0 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, seeded_scalar_positive _ (by norm_num), by norm_num⟩

/-- A strictly negative current gradient activates a zero parameter
within the native sign region, despite any retained nonpositive first
moment. Source: native AdamW at 88aa892; subsequent CE signs will be
derived from the currently trained factor partner. -/
theorem scalar_negative_gradient_activates (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 ≤ 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeScalarState state) (hg : gradient < 0) :
    PositiveScalarState (scalarNativeStep b1 b2 eps decay rate state gradient) := by
  have hn := scalar_nonnegative_step b1 b2 eps decay rate gradient state hb1 h1 hb2 h2 he
    (le_of_lt heta) hd hs (le_of_lt hg)
  have hdir := retained_direction_negative b1 b2 eps gradient state hb1 h1 he hs.2.1 hg
  have hp := mul_nonneg hd hs.1
  refine ⟨?_, hn.2⟩
  change 0 < (1 - rate * decay) * state.parameter - rate * _
  nlinarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ 0 < (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeScalarState (seededScalarState 0) ∧ (-1 : ℝ) < 0 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, seeded_scalar_nonnegative _ (by norm_num), by norm_num⟩

end Transformer.Grokking.AdamW
