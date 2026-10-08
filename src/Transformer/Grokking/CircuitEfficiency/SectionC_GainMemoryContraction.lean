import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryMass

/-!
# A static contraction ceiling retaining native first-moment memory

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C partner inputs; native mass and corrected
clock/epsilon floor at lab commit ada36f3.

Use total parameter mass plus rate/((1-beta)^2*epsilon) times total
negative retained first-moment mass. The two current mass inequalities
give its common ceiling factor: the maximum of remaining decay plus
rate*gain*(2-beta)/((1-beta)*epsilon), and beta*(2-beta).
All actual CE/clipping feedback, second moments and clocks stay intact.

If native decay exceeds gain*(2-beta)/((1-beta)*epsilon), with legal
first beta, positive rate/epsilon/gain and nonnegative remaining decay,
the derived factor lies strictly between zero and one. The condition
is sufficient and deliberately uses the uniform first-clock epsilon
floor; it is not asserted to be the sharp attraction threshold.

Every quantity is numerical current state data. No future parameter
limit, prescribed input, supplied reference or task-success premise is
assumed. These current estimates prepare an initialized contraction;
path convergence and decisions require subsequent proof. Fixed physical
tables and uniform decoupled AdamW differ from appendix C coupled-cost
GD and changing learned GPTMini. No floating-point transfer is claimed.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Positive retained-memory weight derived from the native epsilon
floor. Source: native denominator at ada36f3; all arguments occur. -/
noncomputable def gainNativeMemoryWeight (b1 eps rate : ℝ) : ℝ :=
  rate / ((1 - b1) ^ 2 * eps)

/-- Common numerical ceiling for the two weighted mass contributions.
Sources: appendix C partner gain and native memory at ada36f3. -/
noncomputable def gainNativeMemoryRatio (b1 eps decay rate genGain : ℝ) : ℝ :=
  max (1 - rate * decay + rate * genGain * (2 - b1) / ((1 - b1) * eps)) (b1 * (2 - b1))

/-- Current physical mass plus weighted negative retained memory.
Sources: native mass laws at ada36f3; no limiting property is encoded. -/
noncomputable def gainNativeMemoryMass (b1 eps rate : ℝ) (state : NativeSubweightState) : ℝ :=
  gainNativeParameterMass state + gainNativeMemoryWeight b1 eps rate * gainNativeNegativeMomentMass state

/-- The current nonnegative memory mass covers every physical
coordinate. Sources: appendix C physical factors and retained memory
at ada36f3; no decay or future-state condition is required. -/
theorem gain_native_memory_mass_covers (b1 eps rate : ℝ) (state : NativeSubweightState)
    (he : 0 < eps) (heta : 0 ≤ rate) (hs : NonnegativeNativeState state) :
    0 ≤ gainNativeMemoryMass b1 eps rate state ∧
      gainNativeParameterMass state ≤ gainNativeMemoryMass b1 eps rate state ∧
      ∀ i, (state i).parameter ≤ gainNativeMemoryMass b1 eps rate state := by
  have hm := gain_native_masses_nonnegative state hs
  have hc : 0 ≤ gainNativeMemoryWeight b1 eps rate := div_nonneg heta (mul_nonneg (sq_nonneg _) (le_of_lt he))
  have hweighted := mul_nonneg hc hm.2.1
  have hmass : gainNativeParameterMass state ≤ gainNativeMemoryMass b1 eps rate state := by
    unfold gainNativeMemoryMass
    linarith only [hweighted]
  exact ⟨le_trans hm.1 hmass, hmass, fun i => le_trans (hm.2.2 i) hmass⟩

example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- The explicit memory weight yields both exact coefficient identities.
Source: native first-clock denominator at ada36f3; the weight is computed
from static constants rather than defined as a contraction witness. -/
theorem gain_native_memory_coefficients (b1 eps rate genGain : ℝ) (h1 : b1 < 1) (he : 0 < eps) :
    (rate / ((1 - b1) * eps) + gainNativeMemoryWeight b1 eps rate) * (1 - b1) * genGain =
        rate * genGain * (2 - b1) / ((1 - b1) * eps) ∧
      b1 * (rate / ((1 - b1) * eps) + gainNativeMemoryWeight b1 eps rate) =
        (b1 * (2 - b1)) * gainNativeMemoryWeight b1 eps rate := by
  have hb : 1 - b1 ≠ 0 := by linarith only [h1]
  unfold gainNativeMemoryWeight
  constructor <;> field_simp [hb, ne_of_gt he] <;> ring

example : (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 := by norm_num

/-- Static strong decay makes the generated native memory factor a
strict contraction. Sources: section 3 decay competition and native
mass/epsilon laws at ada36f3; the sufficient threshold is explicit,
nonempty, and no future trajectory data are supplied. -/
theorem gain_native_memory_ratio_strong_decay (b1 eps decay rate genGain : ℝ)
    (h1 : b1 < 1) (he : 0 < eps) (heta : 0 < rate) (hg : 0 < genGain)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    0 < gainNativeMemoryRatio b1 eps decay rate genGain ∧ gainNativeMemoryRatio b1 eps decay rate genGain < 1 := by
  have hb : 0 < 1 - b1 := by linarith only [h1]
  have ht : 0 < 2 - b1 := by linarith only [h1]
  have hinput := mul_pos heta (div_pos (mul_pos hg ht) (mul_pos hb he))
  have hgroup : rate * (genGain * (2 - b1) / ((1 - b1) * eps)) =
      rate * genGain * (2 - b1) / ((1 - b1) * eps) := by ring
  rw [hgroup] at hinput
  have hgap : 0 < decay - genGain * (2 - b1) / ((1 - b1) * eps) := by linarith only [hstrong]
  have hscaled := mul_pos heta hgap
  have hsplit : rate * (decay - genGain * (2 - b1) / ((1 - b1) * eps)) =
      rate * decay - rate * genGain * (2 - b1) / ((1 - b1) * eps) := by ring
  rw [hsplit] at hscaled
  have hp : 0 < 1 - rate * decay + rate * genGain * (2 - b1) / ((1 - b1) * eps) := by
    linarith only [hd, hinput]
  have hp1 : 1 - rate * decay + rate * genGain * (2 - b1) / ((1 - b1) * eps) < 1 := by
    linarith only [hscaled]
  have hm1 : b1 * (2 - b1) < 1 := by nlinarith only [pow_pos hb 2]
  unfold gainNativeMemoryRatio
  exact ⟨lt_of_lt_of_le hp (le_max_left _ _), max_lt hp1 hm1⟩

example : (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 3 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by norm_num

/-- The actual native weighted parameter/moment mass has the derived
current common ceiling, allowing legal nonzero beta. Sources: appendix C
true partner feedback and native mass laws at ada36f3; no convergence,
variance matching, moment reset or successful margin is assumed. -/
theorem gain_native_memory_mass_step_ceiling (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hs : NonnegativeNativeState state) :
    gainNativeMemoryMass b1 eps rate (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      gainNativeMemoryRatio b1 eps decay rate genGain * gainNativeMemoryMass b1 eps rate state := by
  let t := rate / ((1 - b1) * eps)
  let c := gainNativeMemoryWeight b1 eps rate
  let a := 1 - rate * decay + rate * genGain * (2 - b1) / ((1 - b1) * eps)
  let b := b1 * (2 - b1)
  let next := gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state
  have ht : 0 ≤ t := div_nonneg heta (by positivity)
  have hc : 0 ≤ c := div_nonneg heta (mul_nonneg (sq_nonneg _) (le_of_lt he))
  have hm := gain_native_masses_nonnegative state hs
  have hp := gain_native_parameter_mass_ceiling remaining genGain memGain bound b1 b2 eps decay rate state
    hgen hmem hclip hb1 h1 he heta hs
  have hmoment := gain_native_moment_mass_ceiling remaining genGain memGain bound b1 b2 eps decay rate state
    hgen hmem hgain hclip (le_of_lt h1) hs
  have hu := mul_le_mul_of_nonneg_left hmoment (add_nonneg ht hc)
  have hcoeff := gain_native_memory_coefficients b1 eps rate genGain h1 he
  have hparamMax := mul_le_mul_of_nonneg_right (le_max_left a b) hm.1
  have hmomentMax := mul_le_mul_of_nonneg_right (le_max_right a b) (mul_nonneg hc hm.2.1)
  change gainNativeParameterMass next + c * gainNativeNegativeMomentMass next ≤
    max a b * (gainNativeParameterMass state + c * gainNativeNegativeMomentMass state)
  calc
    _ ≤ (1 - rate * decay) * gainNativeParameterMass state + (t + c) * gainNativeNegativeMomentMass next := by
      nlinarith only [hp]
    _ ≤ (1 - rate * decay) * gainNativeParameterMass state +
        (t + c) * (b1 * gainNativeNegativeMomentMass state + (1 - b1) * genGain * gainNativeParameterMass state) := by
      linarith only [hu]
    _ = a * gainNativeParameterMass state + b * (c * gainNativeNegativeMomentMass state) := by
      calc
        _ = (1 - rate * decay + (t + c) * (1 - b1) * genGain) * gainNativeParameterMass state +
            (b1 * (t + c)) * gainNativeNegativeMomentMass state := by ring
        _ = _ := by rw [hcoeff.1, hcoeff.2]; ring
    _ ≤ max a b * gainNativeParameterMass state + max a b * (c * gainNativeNegativeMomentMass state) :=
      add_le_add hparamMax hmomentMax
    _ = _ := by ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

end Transformer.Grokking.CircuitEfficiency
