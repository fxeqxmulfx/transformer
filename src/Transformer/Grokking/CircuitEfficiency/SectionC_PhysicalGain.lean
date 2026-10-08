import Transformer.Grokking.CircuitEfficiency.SectionC_NativeSigns

/-!
# Fixed forward gains and actual two-factor parameter efficiency

Source: Varma et al., arXiv:2309.02390v1, section 3, Efficiency,
and appendix C, sim-overall-logits. Implement efficiency in the
actual fixed-table forward: a circuit produces gain * first * second
from two trainable coordinates with squared norm first^2 + second^2.
Gain is a fixed readout coefficient, not a trainable coordinate or
an unequal optimizer-decay group. All four trainable parameters enter
the Gen/Mem score, and unit gains recover the earlier source forward.

Prove the minimum actual two-coordinate squared norm needed for a
nonnegative logit weight: 2 * weight / gain, attained by equal factors.
Greater positive gain therefore produces the same positive logit with
strictly smaller trainable norm. No norm cost is inserted into CE.

Deviation: this is a degree-two homogeneous physical factor model,
corresponding to the source's scaling exponent two, rather than its
simulation exponent 1.2 and assumed circuit-norm penalty. The tables
and gains are fixed; learned GPTMini feature discovery, native path
convergence and numerical-kernel bridges are not supplied here.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Actual logit weight produced by two trainable factors and a fixed
readout. Source: section 3, degree-two specialization of Efficiency,
and appendix C product logits; gain is counted only in the forward. -/
def physicalCircuitScore (gain first second : ℝ) : ℝ := gain * (first * second)

/-- Squared Euclidean norm of these two trainable coordinates.
Source: section 3's actual parameter norm, specialized to two factors;
the fixed readout coefficient is not a trainable parameter. -/
def physicalPairSquaredNorm (first second : ℝ) : ℝ := first ^ 2 + second ^ 2

/-- Both physically weighted circuit contributions to the correct
training logit. Source: appendix C, sim-overall-logits with explicitly
different fixed readout efficiencies in place of an assumed norm cost. -/
def gainNativeTotalScore (genGain memGain : ℝ) (state : NativeSubweightState) : ℝ :=
  physicalCircuitScore genGain (state 0).parameter (state 1).parameter +
    physicalCircuitScore memGain (state 2).parameter (state 3).parameter

/-- Degree-two output homogeneity follows from the actual trainable
forward. Source: section 3, Efficiency paragraph, exponent two;
gain is fixed while both trainable factors are scaled. -/
theorem physical_circuit_score_scaling (gain scale first second : ℝ) :
    physicalCircuitScore gain (scale * first) (scale * second) =
      scale ^ 2 * physicalCircuitScore gain first second := by
  unfold physicalCircuitScore
  ring

/-- Scaling both actual parameters scales their squared norm by
the square of the same factor. Source: section 3, parameter norm;
this is not a manually assigned circuit-efficiency penalty. -/
theorem physical_pair_norm_scaling (scale first second : ℝ) :
    physicalPairSquaredNorm (scale * first) (scale * second) =
      scale ^ 2 * physicalPairSquaredNorm first second := by
  unfold physicalPairSquaredNorm
  ring

/-- Every pair producing a physical logit obeys the actual norm
budget. Source: section 3, Efficiency, corrected to the explicit
degree-two physical model; absolute value permits signed factors. -/
theorem physical_gain_norm_lower_bound (gain first second : ℝ) (hg : 0 < gain) :
    2 * |physicalCircuitScore gain first second| / gain ≤ physicalPairSquaredNorm first second := by
  have hb : |first * second| ≤ (first ^ 2 + second ^ 2) / 2 := by
    apply abs_le.mpr
    constructor
    · nlinarith [sq_nonneg (first + second)]
    · nlinarith [sq_nonneg (first - second)]
  have hc : 2 * |physicalCircuitScore gain first second| / gain = 2 * |first * second| := by
    unfold physicalCircuitScore
    rw [abs_mul, abs_of_pos hg]
    field_simp
  rw [hc]
  unfold physicalPairSquaredNorm
  linarith

example : (0 : ℝ) < 3 := by norm_num

/-- Equal actual factors attain this budget at every nonnegative
target logit. Source: section 3, degree-two Efficiency model; the
readout, actual output and both coordinate norms are checked. -/
theorem physical_gain_budget_attained (gain weight : ℝ) (hg : 0 < gain) (hw : 0 ≤ weight) :
    physicalCircuitScore gain (Real.sqrt (weight / gain)) (Real.sqrt (weight / gain)) = weight ∧
      physicalPairSquaredNorm (Real.sqrt (weight / gain)) (Real.sqrt (weight / gain)) = 2 * weight / gain := by
  have hd : 0 ≤ weight / gain := by positivity
  constructor
  · unfold physicalCircuitScore
    rw [Real.mul_self_sqrt hd]
    field_simp
  · unfold physicalPairSquaredNorm
    rw [Real.sq_sqrt hd]
    ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Equality in the positive-logit norm budget forces balanced actual
factors. Source: section 3, degree-two Efficiency specialization;
this characterizes the physical minimizer, rather than postulating
that an optimizer keeps the two trainable coordinates balanced. -/
theorem physical_gain_budget_equality (gain first second : ℝ) (hg : 0 < gain) :
    physicalPairSquaredNorm first second =
      2 * physicalCircuitScore gain first second / gain ↔ first = second := by
  have hc : 2 * physicalCircuitScore gain first second / gain = 2 * (first * second) := by
    unfold physicalCircuitScore
    field_simp
  rw [hc]
  unfold physicalPairSquaredNorm
  constructor
  · intro he
    have hz : (first - second) ^ 2 = 0 := by nlinarith [he]
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp hz)
  · intro he
    rw [he]
    ring

example : (0 : ℝ) < 3 := by norm_num

/-- The explicit minimum squared parameter norm is attained, rather
than declared as an effective cost. Source: section 3, Efficiency
specialized to the actual degree-two forward; every feasible pair is
bounded below and a concrete balanced pair achieves the lower bound. -/
theorem physical_gain_budget_minimum (gain weight : ℝ) (hg : 0 < gain) (hw : 0 ≤ weight) :
    (∀ first second, physicalCircuitScore gain first second = weight →
      2 * weight / gain ≤ physicalPairSquaredNorm first second) ∧
    (∃ first second, physicalCircuitScore gain first second = weight ∧
      physicalPairSquaredNorm first second = 2 * weight / gain) := by
  constructor
  · intro first second hs
    have hb := physical_gain_norm_lower_bound gain first second hg
    rw [hs, abs_of_nonneg hw] at hb
    exact hb
  · exact ⟨Real.sqrt (weight / gain), Real.sqrt (weight / gain),
      physical_gain_budget_attained gain weight hg hw⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Greater physical gain is strictly more efficient at a common
positive target logit. Source: section 3's lower-norm criterion;
the compared costs are attained actual trainable squared norms. -/
theorem physical_gain_efficiency_order (genGain memGain weight : ℝ)
    (hm : 0 < memGain) (hgain : memGain < genGain) (hw : 0 < weight) :
    2 * weight / genGain < 2 * weight / memGain := by
  exact div_lt_div_of_pos_left (by positivity) hm hgain

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 := by norm_num

/-- Unit readouts recover the earlier equal-coefficient actual
training score. Source: appendix C, sim-overall-logits; this records
the precise forward specialization before changing CE derivatives. -/
theorem gain_native_unit_score (state : NativeSubweightState) :
    gainNativeTotalScore 1 1 state = nativeTotalScore state := by
  unfold gainNativeTotalScore physicalCircuitScore nativeTotalScore
  ring

end Transformer.Grokking.CircuitEfficiency
