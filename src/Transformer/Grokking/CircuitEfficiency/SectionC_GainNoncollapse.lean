import Transformer.Grokking.CircuitEfficiency.SectionC_GainEnvelope
import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedPath

/-!
# Positive retained Gen mass excludes zero limits above the decay threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3's circuit
formation and appendix C's actual two-factor CE; retained native
AdamW/clipping at lab commit 0ecfae4. Analyze a closed gained-CE
path whose physical parameters have finite limits. Positive Gen
mass is required at finite clocks, not at the limiting point.

If its actual limiting CE coefficient exceeds decay * epsilon,
Gen's parameter mass has a positive limit. Under a hypothetical
zero mass, the actual partner inputs and both retained first
moments tend to zero, and the corrected denominators tend to
epsilon. An eventual coefficient/ceiling pair then satisfies the
retained weighted-mass bound, excluding collapse to zero.

Every input, buffer and denominator tail comes from actual CE
feedback. Finite parameter convergence and the coefficient
threshold remain explicit; no attraction or delayed crossing is
inferred. Uniform native decay/plain CE differ from the source's
GD/coupled norm cost. Tables/readout gains remain fixed, and
learned stochastic/numerical GPTMini transfer is still unproved.

The reference contributes physical parameter data to the actual
callback. Its buffers and clock are not declared limiting optimizer
fields: completed clocks diverge, while causal buffer convergence
is derived separately. Finite-clock positive mass allows a source
factor to start at zero with a positive partner; it does not require
a future positive lower bound, successful margin or train/test label
correspondence to be inserted into the optimizer recurrence.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Positive finite-clock Gen mass has positive limiting mass
when the actual limiting feedback dominates decay at epsilon.
Sources: section 3 formation and appendix C product CE, with native
retained moments at 0ecfae4; successful future Gen mass is a conclusion. -/
theorem gain_gen_mass_positive_limit (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (state n))
    (hgen : 0 < genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 ≤ rate)
    (hs : ∀ n, NonnegativeNativeState (state n)) (hu : ∀ n, 0 < gainGenParameterMass (state n))
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter))
    (hc : decay * eps < gainCEGradientScale remaining genGain memGain bound reference * genGain) :
    0 < gainGenParameterMass reference := by
  by_contra hnot
  have href := gain_native_nonnegative_parameter_limit state reference hs hp
  have hz : (reference 0).parameter = 0 ∧ (reference 1).parameter = 0 := by
    have hmass := le_of_not_gt hnot
    unfold gainGenParameterMass at hmass
    constructor <;> linarith [href 0, href 1]
  have hgz0 : appliedGainNativeGradient remaining genGain memGain bound reference 0 = 0 := by
    rw [gain_native_applied_gradient_scale]
    change -(gainCEGradientScale remaining genGain memGain bound reference * genGain * (reference 1).parameter) = 0
    rw [hz.2, mul_zero, neg_zero]
  have hgz1 : appliedGainNativeGradient remaining genGain memGain bound reference 1 = 0 := by
    rw [gain_native_applied_gradient_scale]
    change -(gainCEGradientScale remaining genGain memGain bound reference * genGain * (reference 0).parameter) = 0
    rw [hz.1, mul_zero, neg_zero]
  have hg0 := gain_native_applied_gradient_tendsto remaining genGain memGain bound state reference 0 hp
  have hg1 := gain_native_applied_gradient_tendsto remaining genGain memGain bound state reference 1 hp
  rw [hgz0] at hg0
  rw [hgz1] at hg1
  let coefficient := gainCEGradientScale remaining genGain memGain bound reference * genGain
  let gap := coefficient - decay * eps
  let ceiling := eps + gap / (2 * decay)
  have hgap : 0 < gap := by dsimp [gap, coefficient]; linarith only [hc]
  have hd : eps < ceiling := by
    have ht := div_pos hgap (show 0 < 2 * decay by positivity)
    dsimp only [ceiling]
    linarith only [ht]
  have hcoefficient : decay * ceiling < coefficient := by
    have hcancel := div_mul_cancel₀ gap (show 2 * decay ≠ 0 by positivity)
    dsimp only [ceiling, gap] at *
    nlinarith only [hcancel, hgap]
  have hi0 : ∀ n, state (n + 1) 0 = scalarNativeStep b1 b2 eps decay rate (state n 0)
      (appliedGainNativeGradient remaining genGain memGain bound (state n) 0) := fun n => congrFun (hstep n) 0
  have hi1 : ∀ n, state (n + 1) 1 = scalarNativeStep b1 b2 eps decay rate (state n 1)
      (appliedGainNativeGradient remaining genGain memGain bound (state n) 1) := fun n => congrFun (hstep n) 1
  have hd0 := scalar_zero_next_denominator_tail b1 b2 eps decay rate ceiling (fun n => state n 0)
    (fun n => appliedGainNativeGradient remaining genGain memGain bound (state n) 0) hi0 hb1 h1 hb2 h2 hg0 hd
  have hd1 := scalar_zero_next_denominator_tail b1 b2 eps decay rate ceiling (fun n => state n 1)
    (fun n => appliedGainNativeGradient remaining genGain memGain bound (state n) 1) hi1 hb1 h1 hb2 h2 hg1 hd
  have hscale := (gain_ce_gradient_scale_tendsto remaining genGain memGain bound state reference hp).mul_const genGain
  obtain ⟨start, htail⟩ := eventually_atTop.mp
    ((hscale.eventually_const_lt hcoefficient).and (hd0.and hd1))
  have hm0 := (gain_native_closed_buffer_limits remaining genGain memGain bound b1 b2 eps decay rate
    state reference 0 hstep hb1 h1 hb2 h2 hp).1
  have hm1 := (gain_native_closed_buffer_limits remaining genGain memGain bound b1 b2 eps decay rate
    state reference 1 hstep hb1 h1 hb2 h2 hp).1
  rw [hgz0] at hm0
  rw [hgz1] at hm1
  have hwl : Tendsto (fun n => gainGenNegativeMomentMass (state n)) atTop (nhds 0) := by
    simpa only [gainGenNegativeMomentMass, zero_add, neg_zero] using (hm0.add hm1).neg
  have hul : Tendsto (fun n => gainGenParameterMass (state n)) atTop (nhds 0) := by
    simpa only [gainGenParameterMass, hz.1, hz.2, zero_add] using (hp 0).add (hp 1)
  have hpositive := retained_pair_positive_limit b1 ceiling rate decay (decay * ceiling) 0
    (fun n => gainGenParameterMass (state n)) (fun n => gainGenNegativeMomentMass (state n)) start
    hb1 h1 (lt_trans he hd) heta (by simp only [sub_self, le_refl]) hu
    (fun n => (gain_gen_masses_nonnegative (state n) (hs n)).2) ?_ ?_ hul hwl
  · exact lt_irrefl 0 hpositive
  · intro n hn
    rw [hstep n]
    exact gain_gen_moment_mass_floor remaining genGain memGain bound b1 b2 eps decay rate (decay * ceiling)
      (state n) (le_of_lt h1) (hs n) (le_of_lt (htail n hn).1)
  · intro n hn
    rw [hstep n]
    exact gain_gen_parameter_mass_floor remaining genGain memGain bound b1 b2 eps decay rate ceiling
      (state n) hgen hclip hb1 h1 he heta (lt_trans he hd) (hs n)
      (le_of_lt (htail n hn).2.1) (le_of_lt (htail n hn).2.2)

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    let state := gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point
    (∀ n, state (n + 1) = gainNativeStep 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) (state n)) ∧
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (∀ n, NonnegativeNativeState (state n)) ∧ (∀ n, 0 < gainGenParameterMass (state n)) ∧
    (∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (point i).parameter)) ∧
    (1 / 2) * eps < gainCEGradientScale 111 3 2 1 point * 3 := by
  dsimp only
  let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
  let eps := 3 * gainCEGradientScale 111 3 2 1 point
  have hb := gain_native_efficient_point_balance 111 1 (by norm_num)
  have hs0 : NonnegativeNativeState point := native_seeded_nonnegative _ _ _ _
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  refine ⟨fun _ => rfl, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, hb.1, by norm_num, by norm_num, ?_, ?_, ?_, ?_⟩
  · intro n
    exact gain_native_nonnegative_path 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point n
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      hb.1 (by norm_num) (by norm_num) hs0
  · intro n
    rw [gain_native_balanced_path_history 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000)
      ((1, 1), (1 / 2, 1 / 2)) n (by norm_num) (by norm_num) (by norm_num) (by norm_num) hb.2]
    norm_num [gainGenParameterMass, gainBalancedHistory, seededNativeSubweights, seededScalarState]
  · intro i
    exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2)
      (1 / 1000) ((1, 1), (1 / 2, 1 / 2)) i (by norm_num) (by norm_num) (by norm_num) (by norm_num) hb.2
  · have hs := gain_ce_gradient_scale_pos 111 3 2 1 point (by norm_num)
    linarith

end Transformer.Grokking.CircuitEfficiency
