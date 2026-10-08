import Transformer.Grokking.CircuitEfficiency.SectionC_GainWarmupTail
import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdLimits
import Transformer.Grokking.CircuitEfficiency.SectionC_GainWeakBalanceBoundary

/-!
# Conditional original-warmup Gen selection from numerical initialization

Sources: Varma et al., arXiv:2309.02390v1, section 3 Efficiency and
appendix C product CE; actual retained warmup at d54cd6d, cold
boundary exclusion at 92002ec and original no-coexistence at 8bc3453.
Keep gains 3/2, cap one, epsilon 1e-8, decay 0.1, betas 0.9/0.98
and the actual ten-completed-update warmup with base rate 0.001.

Starting in the nonnegative physical/sign region with one positive
Gen partner, any finite actual physical parameter limit has strictly
positive Gen mass. The full then-current warmup state enters the
cold-boundary exclusion; both native moments and growing bias clocks
are retained. No positive reference factor or live Mem limit enters
as a premise. The source-style double-zero first factors are allowed.

For at most 113 classes, the limiting Gen factors must be strictly
positive and equal, and both limiting Mem factors must be zero.
This is selection conditional on actual finite parameter convergence,
not a proof of convergence, attraction, transition delay or a basin.
Actual positive pure-Gen original-warmup paths witness the full joint
hypotheses at the fixed numerical recipe, without fitting epsilon.
The source uses coupled norm-cost GD; this exact-real fixed-table
plain-CE/native-decay model does not identify its circuits with
learned GPTMini heads. Numerical, stochastic and learned composition
bridges remain open, and all frozen experiments are unchanged.

The pure Mem paths with wrong held-out answers start with both
Gen factors zero. They therefore do not meet the positive-partner
hypothesis of this selection law. Positive Gen examples ensure that
its convergence premises are jointly satisfiable, but do not prove
those premises for the small source-style competing initialization.
No first successful clock, uniform time bound or monotone allocation
is obtained by classifying a finite limit.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Positive actual original-warmup Gen paths jointly witness the
numerical sign, positive initial partner and finite-limit hypotheses.
Sources: appendix C products and pure-root histories at 792e3bc;
the optimizer constants and epsilon are the fixed original ones. -/
theorem original_gain_warmup_positive_limit_witness (remaining : ℕ) :
    ∃ point : NativeSubweightState, NonnegativeNativeState point ∧ 0 < (point 1).parameter ∧
      ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) point n i).parameter)
        atTop (nhds (point i).parameter) := by
  obtain ⟨amplitude, ha, _, hp⟩ := original_pure_circuit_warmup_parameter_limits_exist remaining 0
  refine ⟨originalPureCircuitPoint 0 amplitude, ?_, ?_, hp⟩
  · change NonnegativeNativeState (seededNativeSubweights ((amplitude, amplitude), (0, 0)))
    exact native_seeded_nonnegative _ _ _ _ (le_of_lt ha) (le_of_lt ha) le_rfl le_rfl
  · change (0 : ℝ) < amplitude
    exact ha

/-- Actual finite original-warmup parameter limits inherit physical
nonnegativity from initialization. Sources: appendix C partners and
native scheduled signs at 8b41020; no reference signs are assumed. -/
theorem original_gain_warmup_limit_nonnegative (remaining : ℕ) (initial reference : NativeSubweightState)
    (hs : NonnegativeNativeState initial)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    ∀ i, 0 ≤ (reference i).parameter := by
  apply gain_native_nonnegative_parameter_limit _ reference _ hp
  intro n
  exact gain_native_scheduled_nonnegative_path remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000)
    (1 / 10) grokkingWarmupRate initial n (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun k => ⟨(grokking_warmup_rate_legal k).1, (grokking_warmup_rate_legal k).2.2⟩) hs

example : ∃ initial reference : NativeSubweightState, NonnegativeNativeState initial ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 95 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
  obtain ⟨point, hs, _, hp⟩ := original_gain_warmup_positive_limit_witness 95
  exact ⟨point, point, hs, hp⟩

/-- A positive initial Gen partner excludes every zero-Gen finite
original-warmup limit. Sources: section 3 formation/efficiency and
actual cold-regime retained noncollapse at 92002ec; finite physical
convergence stays explicit, while live Mem is not a premise. -/
theorem original_gain_warmup_limit_gen_mass (remaining : ℕ) (initial reference : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    0 < gainGenParameterMass reference := by
  let start := gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial 10
  have hstart : NonnegativeNativeState start :=
    gain_native_scheduled_nonnegative_path remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) grokkingWarmupRate initial 10 (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (fun k => ⟨(grokking_warmup_rate_legal k).1, (grokking_warmup_rate_legal k).2.2⟩) hs
  have hpositive : 0 < (start 1).parameter :=
    (gain_native_warmup_positive_coordinate remaining 3 2 1 (1 / 100000000) initial 10 1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs hi).1
  have htail := gain_native_warmup_tail_parameter_limits remaining 3 2 1 (1 / 100000000) initial reference hp
  exact gain_native_source_cold_regime_gen_mass remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000)
    (1 / 10) (1 / 1000) start reference (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) hstart hpositive htail (cold_gain_small_decay_epsilon_regime remaining)

example : ∃ initial reference : NativeSubweightState,
    NonnegativeNativeState initial ∧ 0 < (initial 1).parameter ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 111 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
  obtain ⟨point, hs, hi, hp⟩ := original_gain_warmup_positive_limit_witness 111
  exact ⟨point, point, hs, hi, hp⟩

/-- At most 113 classes force every finite original-warmup limit
from positive-Gen-partner data to select the pure positive Gen pair.
Sources: section 3 efficiency, appendix C complete CE and necessary
original balance at 8bc3453; neither successful limits nor absent Mem
are hypotheses, and global convergence is not asserted. -/
theorem original_gain_warmup_limit_gen_selected (remaining : ℕ) (initial reference : NativeSubweightState)
    (hq : remaining ≤ 111) (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    0 < (reference 0).parameter ∧ (reference 1).parameter = (reference 0).parameter ∧
      (reference 2).parameter = 0 ∧ (reference 3).parameter = 0 := by
  have hn := original_gain_warmup_limit_nonnegative remaining initial reference hs hp
  have hb := gain_native_warmup_limit_balance remaining 3 2 1 (1 / 100000000) initial reference (by norm_num) hp
  have hg := original_gain_warmup_limit_gen_mass remaining initial reference hs hi hp
  have hpartner := gain_native_nonnegative_partner_balance remaining 3 2 1 (1 / 100000000) (1 / 10)
    reference (by norm_num) (by norm_num) (by norm_num) hn hb
  have hscale := gain_ce_gradient_scale_pos remaining 3 2 1 reference (by norm_num)
  have hGenPair := native_nonnegative_pair_balance (gainCEGradientScale remaining 3 2 1 reference * 3)
    (1 / 100000000) (1 / 10) (reference 0).parameter (reference 1).parameter
    (by positivity) (by norm_num) (hn 0) (hn 1)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hpartner 0)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hpartner 1)
  rcases hGenPair with hz | ⟨_, hp0, _, hequal, _⟩
  · exfalso
    unfold gainGenParameterMass at hg
    linarith only [hg, hz.1, hz.2]
  · have habsent := original_gain_balanced_pair_absent remaining 1 reference hq (by norm_num) hn hb
    have hMem : (reference 2).parameter = 0 ∧ (reference 3).parameter = 0 := by
      rcases habsent with hz | hm
      · exfalso
        linarith only [hp0, hz.1]
      · exact hm
    exact ⟨hp0, hequal.symm, hMem⟩

example : (95 : ℕ) ≤ 111 ∧ ∃ initial reference : NativeSubweightState,
    NonnegativeNativeState initial ∧ 0 < (initial 1).parameter ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 95 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
  obtain ⟨point, hs, hi, hp⟩ := original_gain_warmup_positive_limit_witness 95
  exact ⟨by norm_num, point, point, hs, hi, hp⟩

end Transformer.Grokking.CircuitEfficiency
