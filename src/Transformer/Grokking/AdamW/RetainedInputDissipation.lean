import Transformer.Grokking.AdamW.ScalarInputDissipation

/-!
# Retained-input dissipation with a nonzero native first beta

Source: PyTorch 2.14.1 AdamW, adam.py lines 445--475 moment
insertions and lines 529--547 completed-clock corrections, ported at
79f4fb0. Extend the inserted-input denominator bounds at c555a63
without replacing the actual retained first numerator.

The corrected denominator has a current squared-input floor times
its first-clock correction. For a fixed input floor, every strict
subfloor eventually holds uniformly over nonnegative retained variance
and all inputs meeting that floor. The late-clock bound uses no input
or variance convergence. Legal first beta need not be zero.

A sufficiently large actual denominator gives a positive adaptive
gap relative to the epsilon ceiling. A negative retained first moment
cannot cancel the negative new input; its new magnitude is at least
(1-beta1) times the supplied input floor. Both numerator history and
the complete denominator stay in the true native update.

These are scalar current estimates and a conditional uniform clock
bound. A CE application must generate large inputs on its own path.
The generic estimates do not assert parameter convergence at critical
decay, nor do they make all coordinate denominators exceed epsilon:
a coordinate without a nonzero input can retain a smaller correction.
A separate weighted mass argument must absorb those clock errors.

Exact reals and non-AMSGrad native AdamW are explicit. No buffer,
optimizer or checkpoint is changed, and no learned GPTMini,
floating-point or thermodynamic system-size claim follows.
-/

namespace Transformer.Grokking.AdamW

open Filter

/-- The actual retained denominator covers its inserted-input floor
times the true first-clock correction. Source: native AdamW at
79f4fb0 and c555a63; neither beta nor the retained variance is reset. -/
theorem next_buffer_retained_denominator_input_floor
    (b1 b2 eps variance gradient lower : ℝ) (clock : ℕ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hv : 0 ≤ variance) (hl : 0 ≤ lower) (hg : lower ≤ |gradient|) :
    (1 - b1 ^ (clock + 1)) * (eps + Real.sqrt ((1 - b2) * lower ^ 2)) ≤
      nextBufferDenominator b1 b2 eps variance gradient clock := by
  have hfloor := next_buffer_zero_first_denominator_input_floor b2 eps variance gradient lower clock hb2 h2 hv hl hg
  have hc := le_of_lt (bias_correction_positive b1 (clock + 1) hb1 h1 (by omega))
  have hscaled := mul_le_mul_of_nonneg_left hfloor hc
  simpa only [nextBufferDenominator, zero_pow (by omega : clock + 1 ≠ 0),
    sub_zero, one_mul] using hscaled

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ |(-1 : ℝ)| := by norm_num

/-- A fixed inserted-input floor supplies a uniform late-clock
denominator bound below its limiting square-root floor. Source:
native corrections at 79f4fb0; current inputs and retained variances
are universally quantified, not required to converge in the future. -/
theorem next_buffer_retained_input_floor_tail
    (b1 b2 eps lower floor : ℝ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hl : 0 ≤ lower) (hd : floor < eps + Real.sqrt ((1 - b2) * lower ^ 2)) :
    ∃ start : ℕ, ∀ clock, start ≤ clock → ∀ variance gradient : ℝ,
      0 ≤ variance → lower ≤ |gradient| →
      floor ≤ nextBufferDenominator b1 b2 eps variance gradient clock := by
  have hp := (tendsto_add_atTop_iff_nat 1).mpr (tendsto_pow_atTop_nhds_zero_of_lt_one hb1 h1)
  have hc : Tendsto (fun clock : ℕ => 1 - b1 ^ (clock + 1)) atTop (nhds 1) := by
    simpa only [sub_zero] using tendsto_const_nhds.sub hp
  have ht : Tendsto (fun clock : ℕ =>
      (1 - b1 ^ (clock + 1)) * (eps + Real.sqrt ((1 - b2) * lower ^ 2))) atTop
      (nhds (eps + Real.sqrt ((1 - b2) * lower ^ 2))) := by
    simpa only [one_mul] using hc.mul_const (eps + Real.sqrt ((1 - b2) * lower ^ 2))
  obtain ⟨start, htail⟩ := eventually_atTop.mp (ht.eventually_const_lt hd)
  refine ⟨start, ?_⟩
  intro clock hclock variance gradient hv hg
  exact le_trans (le_of_lt (htail clock hclock))
    (next_buffer_retained_denominator_input_floor b1 b2 eps variance gradient lower clock hb1 h1 hb2 h2 hv hl hg)

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) < 1 + Real.sqrt ((1 - 49 / 50) * (1 : ℝ) ^ 2) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  have hroot := Real.sqrt_pos.mpr (show (0 : ℝ) < (1 - 49 / 50) * (1 : ℝ) ^ 2 by norm_num)
  linarith only [hroot]

/-- A true denominator above epsilon and a nonzero new input give
an explicit positive adaptive gap with the first moment retained.
Source: native AdamW at 79f4fb0; late-clock applications can derive
the current denominator premise from the preceding uniform bound. -/
theorem scalar_retained_input_adaptive_gap
    (b1 b2 eps decay rate gradient lower floor : ℝ) (state : ScalarState)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hm : state.moment ≤ 0)
    (he : 0 < eps) (heta : 0 ≤ rate) (hl : 0 < lower) (hg : gradient ≤ -lower)
    (hfloor : eps < floor)
    (hd : floor ≤ nextBufferDenominator b1 b2 eps state.variance gradient state.clock) :
    let gap := ((1 - b1) * lower) / eps - ((1 - b1) * lower) / floor
    0 < gap ∧ (scalarNativeStep b1 b2 eps decay rate state gradient).parameter ≤
      (1 - rate * decay) * state.parameter + rate *
        ((-(scalarNativeStep b1 b2 eps decay rate state gradient).moment) / eps - gap) := by
  dsimp only
  let magnitude := -(scalarNativeStep b1 b2 eps decay rate state gradient).moment
  let input := (1 - b1) * lower
  have hinput : 0 < input := mul_pos (by linarith only [h1]) hl
  have hold := mul_nonpos_of_nonneg_of_nonpos hb1 hm
  have hnew := mul_le_mul_of_nonneg_left hg (show 0 ≤ 1 - b1 by linarith only [h1])
  have hmag : input ≤ magnitude := by
    dsimp only [input, magnitude, scalarNativeStep]
    nlinarith only [hold, hnew]
  have hnonnegative : 0 ≤ magnitude := le_of_lt (lt_of_lt_of_le hinput hmag)
  have hpositive : 0 < floor := lt_trans he hfloor
  have hgap : 0 < input / eps - input / floor := by
    have h := div_lt_div_of_pos_left hinput he hfloor
    linarith only [h]
  have hreciprocal : (1 : ℝ) / floor ≤ 1 / eps :=
    div_le_div_of_nonneg_left (by norm_num) he (le_of_lt hfloor)
  have hproduct := mul_nonneg (sub_nonneg.mpr hmag) (sub_nonneg.mpr hreciprocal)
  have hidentity : magnitude / eps - (input / eps - input / floor) - magnitude / floor =
      (magnitude - input) * (1 / eps - 1 / floor) := by ring
  have hlinear : magnitude / floor ≤ magnitude / eps - (input / eps - input / floor) := by
    linarith only [hproduct, hidentity]
  have hactual := div_le_div_of_nonneg_left hnonnegative hpositive hd
  refine ⟨hgap, ?_⟩
  rw [scalar_parameter_denominator]
  change (1 - rate * decay) * state.parameter + rate * magnitude /
      nextBufferDenominator b1 b2 eps state.variance gradient state.clock ≤ _
  have hscaled := mul_le_mul_of_nonneg_left (le_trans hactual hlinear) heta
  have hgroup : rate * magnitude / nextBufferDenominator b1 b2 eps state.variance gradient state.clock =
      rate * (magnitude / nextBufferDenominator b1 b2 eps state.variance gradient state.clock) := by ring
  rw [hgroup]
  exact add_le_add le_rfl hscaled

/- The joint current-state fixture retains nonzero first and second
moments at a positive completed clock. The scalar theorem requires
these numerical data; the CE application separately derives its own
initialized state history and current denominator bound. -/
example :
    let state : ScalarState := ⟨1, -10, 200, 1⟩
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ state.moment ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 1 / 2 ∧ (-1 : ℝ) ≤ -(1 / 2 : ℝ) ∧
      (1 : ℝ) < 2 ∧ 2 ≤ nextBufferDenominator (9 / 10) (49 / 50) 1 state.variance (-1) state.clock := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  have hroot := Real.sqrt_le_sqrt (show (361 : ℝ) ≤ 490050 / 99 by norm_num)
  norm_num at hroot
  norm_num [nextBufferDenominator]
  nlinarith only [hroot]

end Transformer.Grokking.AdamW
