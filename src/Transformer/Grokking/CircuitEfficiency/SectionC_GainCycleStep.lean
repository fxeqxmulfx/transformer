import Transformer.Grokking.CircuitEfficiency.SectionC_GainCycleFeedback
import Transformer.Grokking.AdamW.ScalarDenominator

/-!
# Legal native constants and actual zero-beta amplitude updates

Sources: Varma et al., arXiv:2309.02390v1, appendix C's actual
two-factor binary CE, and PyTorch AdamW at lab commit dc4009c.
Choose epsilon, uniform decay and rate from the verified current
CE inputs at amplitudes one/two. Prove the constants are positive,
the remaining decay factor is positive, and the cold Gen threshold
holds. The subsequent path uses legal beta1=beta2=0 throughout.

Derive an actual step from arbitrary retained moments, variances and
clocks at the stated present parameter point. Both native insertions
and completed-clock corrections remain in the optimizer definition.
Zero betas make the new direction instantaneous; they are a fixed
legal configuration rather than an artificial moment-reset intervention.
The two normalized amplitude equations are then verified. Infinite
physical-parameter periodicity and nonconvergence still require
an actual path proof; completed optimizer clocks keep increasing.
This chosen large-rate binary-table specialization is not a claim
about the preserved nonzero-beta GPTMini runs or coupled-cost GD.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Positive numerical epsilon from the actual low-amplitude input.
Source: true binary CE at dc4009c; no path property is encoded. -/
noncomputable def gainCycleEpsilon : ℝ := gainCycleInputMagnitude 1

/-- Actual high-amplitude normalized input at the chosen epsilon.
Source: native zero-beta normalization at dc4009c. -/
noncomputable def gainCycleHighDirection : ℝ :=
  gainCycleInputMagnitude 2 / (gainCycleInputMagnitude 2 + gainCycleEpsilon)

/-- Numerical uniform decay chosen from the two current directions.
Source: native parameter-decay formula at dc4009c. -/
noncomputable def gainCycleDecay : ℝ := (1 / 2 + gainCycleHighDirection) / 3

/-- Numerical rate solving the two amplitude equations. Source:
native AdamW at dc4009c; positivity and the actual equations are proved. -/
noncomputable def gainCycleRate : ℝ :=
  2 / (gainCycleDecay + 1 / 2 - gainCycleHighDirection)

/-- Actual high-amplitude normalization is strictly between zero
and one quarter. Sources: appendix C binary CE and the native
epsilon choice at dc4009c; use the proved actual input separation. -/
theorem gain_cycle_high_direction_interval :
    0 < gainCycleHighDirection ∧ gainCycleHighDirection < 1 / 4 := by
  have hs := gain_cycle_input_separation
  have hd : 0 < gainCycleInputMagnitude 2 + gainCycleInputMagnitude 1 := by linarith only [hs.1, hs.2.1]
  constructor
  · exact div_pos hs.2.1 hd
  · change gainCycleInputMagnitude 2 / (gainCycleInputMagnitude 2 + gainCycleInputMagnitude 1) < 1 / 4
    apply (div_lt_iff₀ hd).mpr
    linarith only [hs.2.2.1]

/-- This actual configuration meets positive native constants,
positive remaining decay and the static cold Gen regime. Sources:
appendix C's current CE and native AdamW at dc4009c; no finite
parameter convergence or future positive limiting Gen is supplied. -/
theorem gain_cycle_configuration_regime :
    0 < gainCycleEpsilon ∧ 0 < gainCycleDecay ∧ 0 < gainCycleRate ∧
      0 < 1 - gainCycleRate * gainCycleDecay ∧
      gainCycleDecay * gainCycleEpsilon < coldGainCEGradientScale 0 10 * 3 := by
  have hb := gain_cycle_high_direction_interval
  have hs := gain_cycle_input_separation
  have he : 0 < gainCycleEpsilon := hs.1
  have hd : 0 < gainCycleDecay := by unfold gainCycleDecay; linarith only [hb.1]
  have hden : 0 < gainCycleDecay + 1 / 2 - gainCycleHighDirection := by linarith only [hd, hb.2]
  have hr : 0 < gainCycleRate := div_pos (by norm_num) hden
  have hbudget : 2 * gainCycleDecay / (gainCycleDecay + 1 / 2 - gainCycleHighDirection) < 1 := by
    apply (div_lt_iff₀ hden).mpr
    unfold gainCycleDecay
    linarith only [hb.2]
  have heq : gainCycleRate * gainCycleDecay =
      2 * gainCycleDecay / (gainCycleDecay + 1 / 2 - gainCycleHighDirection) := by
    unfold gainCycleRate
    ring
  have hremain : 0 < 1 - gainCycleRate * gainCycleDecay := by rw [heq]; linarith only [hbudget]
  have hdecay : gainCycleDecay < 1 / 4 := by unfold gainCycleDecay; linarith only [hb.2]
  have hscale : gainCycleEpsilon < 3 := hs.2.2.2
  have hfirst := mul_lt_mul_of_pos_right hdecay he
  have hsecond := mul_lt_mul_of_pos_left hscale (by norm_num : (0 : ℝ) < 1 / 4)
  have hcold : coldGainCEGradientScale 0 10 = 1 / 2 := by norm_num [coldGainCEGradientScale]
  rw [hcold]
  exact ⟨he, hd, hr, hremain, by linarith only [hfirst, hsecond]⟩

/-- At any retained clock, the true zero-beta step at an equal-Gen
physical point uses its actual present CE magnitude. Sources:
appendix C product CE and native AdamW at dc4009c; supplied parameter
equalities determine the current callback, not a future input stream. -/
theorem gain_cycle_zero_beta_amplitude_step (amplitude eps decay rate : ℝ) (state : NativeSubweightState)
    (h0 : 0 ≤ amplitude) (h2 : amplitude ≤ 2)
    (hp : ∀ i, (state i).parameter = (seededNativeSubweights ((amplitude, amplitude), (0, 0)) i).parameter) :
    ∀ i, (gainNativeStep 0 3 2 10 0 0 eps decay rate state i).parameter =
      ![(1 - rate * decay) * amplitude + rate * gainCycleInputMagnitude amplitude /
          (gainCycleInputMagnitude amplitude + eps),
        (1 - rate * decay) * amplitude + rate * gainCycleInputMagnitude amplitude /
          (gainCycleInputMagnitude amplitude + eps), 0, 0] i := by
  have hg := gain_native_equal_parameters_applied_gradient 0 3 2 10 state
    (seededNativeSubweights ((amplitude, amplitude), (0, 0))) hp
  rw [(gain_cycle_unclipped_callback amplitude h0 h2).2] at hg
  have hv : 0 ≤ gainCycleInputMagnitude amplitude := by unfold gainCycleInputMagnitude; positivity
  have hn : -gainCycleInputMagnitude amplitude ≤ 0 := by linarith only [hv]
  intro i
  change (scalarNativeStep 0 0 eps decay rate (state i)
    (appliedGainNativeGradient 0 3 2 10 state i)).parameter = _
  rw [scalar_zero_betas_parameter, hg, hp i]
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState, abs_of_nonpos hn] <;> ring

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 ∧ ∀ i : Fin 4,
    (⟨(seededNativeSubweights ((1, 1), (0, 0)) i).parameter,
      -1, 1, 37⟩ : ScalarState).parameter =
        (seededNativeSubweights ((1, 1), (0, 0)) i).parameter := by
  exact ⟨by norm_num, by norm_num, fun i => rfl⟩

/-- The actual normalized inputs and chosen native constants solve
both numerical swap equations. Sources: appendix C's binary CE
and native zero-beta parameter update at dc4009c; this derives
one/two amplitude movement, not an assumed infinite periodic path. -/
theorem gain_cycle_swapping_equations :
    (1 - gainCycleRate * gainCycleDecay) * 1 + gainCycleRate * gainCycleInputMagnitude 1 /
      (gainCycleInputMagnitude 1 + gainCycleEpsilon) = 2 ∧
    (1 - gainCycleRate * gainCycleDecay) * 2 + gainCycleRate * gainCycleHighDirection = 1 := by
  have hb := gain_cycle_high_direction_interval
  have hne : 1 - gainCycleHighDirection ≠ 0 := by linarith only [hb.2]
  have hr : gainCycleRate = 3 / (1 - gainCycleHighDirection) := by
    have hden : (1 / 2 + gainCycleHighDirection) / 3 +
        1 / 2 - gainCycleHighDirection = (2 / 3) *
          (1 - gainCycleHighDirection) := by ring
    unfold gainCycleRate gainCycleDecay
    rw [hden]
    field_simp [hne]
  have hs := gain_cycle_input_separation
  have heps : gainCycleInputMagnitude 1 / (gainCycleInputMagnitude 1 + gainCycleEpsilon) = 1 / 2 := by
    unfold gainCycleEpsilon
    field_simp [ne_of_gt hs.1]
    ring
  rw [hr]
  have hgroup : 3 / (1 - gainCycleHighDirection) * gainCycleInputMagnitude 1 /
      (gainCycleInputMagnitude 1 + gainCycleEpsilon) =
      3 / (1 - gainCycleHighDirection) * (gainCycleInputMagnitude 1 /
        (gainCycleInputMagnitude 1 + gainCycleEpsilon)) := by ring
  rw [hgroup, heps]
  constructor <;> unfold gainCycleDecay <;> field_simp [hne] <;> ring

end Transformer.Grokking.CircuitEfficiency
