import Transformer.Grokking.CircuitEfficiency.SectionC_GainMassDecay

/-!
# Initialized actual strong-decay paths collapse in absolute mass

Sources: Varma et al., arXiv:2309.02390v1, section 3's CE/decay
competition and appendix C's product partials; exact clipped native
feedback and mass multiplier estimates at lab commit 4e9829f.

The actual current shared CE/clipping coefficient remains endogenous.
Induction gives a power ceiling on each physical pair sum from any
nonnegative retained initial state, including zero pair sums. The
initial moments, variances and completed clocks need not be reset.

If decay is greater than the larger physical gain divided by epsilon,
and remaining decay is nonnegative, this ceiling tends to zero.
Actual numerical signs give both pair-mass limits and all four physical
parameter limits by squeezing. No parameter convergence, future box,
limiting reference, positive mass floor or successful decision is assumed.

For q = 1 - rate * decay + rate * genGain / epsilon,
each sum at clock n is at most q^n times its initial sum. The proof
uses the same common q for Gen and Mem; lesser Mem gain can only
lower its own instantaneous ceiling. For gain 3, epsilon 1, decay 10
and rate 1/1000, the common q is 993/1000. All clocks remain actual
completed native updates, rather than a rescaled limiting flow.

This is absolute collapse, not failure of relative efficient-circuit
selection. A later theorem must combine it with actual task decisions
before drawing an implication about accuracy versus confidence/loss.

Fixed physical tables, exact reals, zero betas and uniform decoupled
native decay differ from appendix C's coupled norm/GD and preserved
learned nonzero-beta GPTMini. No floating-point transfer is claimed.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter

/-- Every initialized actual native pair mass has the explicit power
ceiling, even if the initial mass is zero. Sources: appendix C partner
inputs and native mass/sign laws at 4e9829f; initial retained signs
generate all path signs, without future bounds or buffer resets. -/
theorem gain_native_mass_power_ceiling (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 ≤ rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial) :
    let q := gainPairMassCeiling genGain eps decay rate
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    ∀ n, (path n 0).parameter + (path n 1).parameter ≤
        q ^ n * ((initial 0).parameter + (initial 1).parameter) ∧
      (path n 2).parameter + (path n 3).parameter ≤
        q ^ n * ((initial 2).parameter + (initial 3).parameter) := by
  let q := gainPairMassCeiling genGain eps decay rate
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
  have hq : 0 ≤ q := add_nonneg hd (div_nonneg (mul_nonneg heta (le_of_lt hgen)) (le_of_lt he))
  have hn : ∀ n, NonnegativeNativeState (path n) := by
    intro n
    exact gain_native_nonnegative_path remaining genGain memGain bound 0 0 eps decay rate initial n
      hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he heta hd hs
  dsimp only
  intro n
  induction n with
  | zero =>
    change (initial 0).parameter + (initial 1).parameter ≤ q ^ 0 * ((initial 0).parameter + (initial 1).parameter) ∧
      (initial 2).parameter + (initial 3).parameter ≤ q ^ 0 * ((initial 2).parameter + (initial 3).parameter)
    simp only [pow_zero, one_mul, le_refl, and_self]
  | succ n ih =>
    have hstep := gain_native_pair_mass_ceiling remaining genGain memGain bound eps decay rate (path n)
      hgen hmem hgain hclip he heta (fun i => (hn n i).1)
    have hg := le_trans hstep.1 (mul_le_mul_of_nonneg_left ih.1 hq)
    have hm := le_trans hstep.2 (mul_le_mul_of_nonneg_left ih.2 hq)
    constructor
    · simpa only [path, gainNativePath, q, pow_succ, mul_assoc, mul_comm, mul_left_comm] using hg
    · simpa only [path, gainNativePath, q, pow_succ, mul_assoc, mul_comm, mul_left_comm] using hm

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Strong native decay forces both actual initialized pair masses
to zero. Sources: section 3 CE/decay competition and native estimates
at 4e9829f; static gain/epsilon/decay conditions and initial retained
sign data derive convergence, including zero initial masses. -/
theorem gain_native_strong_decay_mass_tendsto_zero (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : genGain / eps < decay) (hs : NonnegativeNativeState initial) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    Tendsto (fun n => (path n 0).parameter + (path n 1).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => (path n 2).parameter + (path n 3).parameter) atTop (nhds 0) := by
  have hq := gain_pair_mass_ceiling_strong_decay genGain eps decay rate hgen he heta hd hstrong
  have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one (le_of_lt hq.1) hq.2
  have hpowers := gain_native_mass_power_ceiling remaining genGain memGain bound eps decay rate initial
    hgen hmem hgain hclip he (le_of_lt heta) hd hs
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
  have hn : ∀ n, NonnegativeNativeState (path n) := by
    intro n
    exact gain_native_nonnegative_path remaining genGain memGain bound 0 0 eps decay rate initial n
      hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he (le_of_lt heta) hd hs
  constructor
  · apply squeeze_zero (fun n => add_nonneg (hn n 0).1 (hn n 1).1) (fun n => (hpowers n).1)
    simpa using hpow.mul_const ((initial 0).parameter + (initial 1).parameter)
  · apply squeeze_zero (fun n => add_nonneg (hn n 2).1 (hn n 3).1) (fun n => (hpowers n).2)
    simpa using hpow.mul_const ((initial 2).parameter + (initial 3).parameter)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧ (3 : ℝ) / 1 < 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Every physical parameter of the original initialized strong-decay
path tends to zero. Sources: appendix C physical factors and native
mass/sign iteration at 4e9829f; this conclusion is derived from static
data, not supplied as a limiting reference or attraction hypothesis. -/
theorem gain_native_strong_decay_parameters_tendsto_zero (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : genGain / eps < decay) (hs : NonnegativeNativeState initial) :
    ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial n i).parameter)
      atTop (nhds 0) := by
  have hmass := gain_native_strong_decay_mass_tendsto_zero remaining genGain memGain bound eps decay rate initial
    hgen hmem hgain hclip he heta hd hstrong hs
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
  have hn : ∀ n, NonnegativeNativeState (path n) := by
    intro n
    exact gain_native_nonnegative_path remaining genGain memGain bound 0 0 eps decay rate initial n
      hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he (le_of_lt heta) hd hs
  intro i
  fin_cases i
  · apply squeeze_zero (fun n => (hn n 0).1) (fun n => ?_) hmass.1
    linarith only [(hn n 1).1]
  · apply squeeze_zero (fun n => (hn n 1).1) (fun n => ?_) hmass.1
    linarith only [(hn n 0).1]
  · apply squeeze_zero (fun n => (hn n 2).1) (fun n => ?_) hmass.2
    linarith only [(hn n 3).1]
  · apply squeeze_zero (fun n => (hn n 3).1) (fun n => ?_) hmass.2
    linarith only [(hn n 2).1]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧ (3 : ℝ) / 1 < 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.CircuitEfficiency
