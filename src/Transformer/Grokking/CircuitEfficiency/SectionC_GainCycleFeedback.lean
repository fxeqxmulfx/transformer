import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdRegime

/-!
# Actual binary gained-CE feedback at two proposed cycle amplitudes

Sources: Varma et al., arXiv:2309.02390v1, appendix C's physical
product train CE, and native clipping at lab commit 94c077f.
Use two classes, gains three/two and clip cap ten. Gen's factors
are equal; both Mem factors and initial buffers are zero. Derive
the complete actual callback and prove the wrapper is inactive
on every Gen amplitude in [0, 2]. No future input is prescribed.

The exact input magnitudes at amplitudes one and two are positive,
but the second is less than one third of the first. This is current
CE feedback at explicit parameter points, not a supplied update,
periodic-path predicate or convergence claim. A subsequent native
zero-beta construction must derive the proposed period-two path.
That legal beta specialization and a large chosen rate do not model
the preserved beta1=0.9/beta2=0.98 GPTMini run. Plain CE, physical
gains, real arithmetic and uniform decoupled native decay remain
explicit deviations from appendix C's coupled-cost GD.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW
open scoped BigOperators

/-- Numerical partner-input magnitude of the equal-Gen binary table.
Source: appendix C true product CE at 94c077f; its actual callback
identity and the clip's inactivity are proved below. -/
noncomputable def gainCycleInputMagnitude (amplitude : ℝ) : ℝ :=
  3 * amplitude / (Real.exp (3 * amplitude ^ 2) + 1)

/-- Evaluate the entire true raw coordinate derivative vector on an
equal-Gen/zero-Mem physical point. Source: appendix C product CE,
with the explicit binary/gain specialization at 94c077f. -/
theorem gain_cycle_raw_callback (amplitude : ℝ) :
    rawGainNativeGradient 0 3 2 (seededNativeSubweights ((amplitude, amplitude), (0, 0))) =
      ![-gainCycleInputMagnitude amplitude, -gainCycleInputMagnitude amplitude, 0, 0] := by
  have hscore : gainNativeTotalScore 3 2 (seededNativeSubweights ((amplitude, amplitude), (0, 0))) =
      3 * amplitude ^ 2 := by
    norm_num [gainNativeTotalScore, physicalCircuitScore,
      seededNativeSubweights, seededScalarState]
    ring
  funext i
  simp only [rawGainNativeGradient, hscore]
  fin_cases i <;> norm_num [nativeFactorGain, nativeFactorPartner, seededNativeSubweights,
    seededScalarState, gainCycleInputMagnitude] <;> ring

/-- The actual raw norm is at most nine throughout this amplitude
interval. Sources: appendix C binary slope and native norm-two clip
at 94c077f; bound the true full callback rather than supplying its norm. -/
theorem gain_cycle_raw_norm_ceiling (amplitude : ℝ)
    (h0 : 0 ≤ amplitude) (h2 : amplitude ≤ 2) :
    coordinateGradientNorm (rawGainNativeGradient 0 3 2
      (seededNativeSubweights ((amplitude, amplitude), (0, 0)))) ≤ 9 := by
  have hz0 : 0 ≤ gainCycleInputMagnitude amplitude := by
    unfold gainCycleInputMagnitude
    exact div_nonneg (by positivity) (by positivity)
  have hz6 : gainCycleInputMagnitude amplitude ≤ 6 := by
    unfold gainCycleInputMagnitude
    have hd : 0 < Real.exp (3 * amplitude ^ 2) + 1 := by positivity
    apply (div_le_iff₀ hd).mpr
    nlinarith only [h2, Real.exp_pos (3 * amplitude ^ 2)]
  rw [gain_cycle_raw_callback]
  norm_num [coordinateGradientNorm, Fin.sum_univ_succ]
  have hsq : gainCycleInputMagnitude amplitude ^ 2 +
      gainCycleInputMagnitude amplitude ^ 2 ≤ 81 := by
    nlinarith only [hz0, hz6]
  exact le_trans (Real.sqrt_le_sqrt hsq) (by norm_num)

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 := by norm_num

/-- Native clipping at cap ten leaves both actual inputs unchanged
on the verified amplitude interval and gives the exact current CE
scale. Sources: appendix C binary CE and the actual norm+1e-6
wrapper at 94c077f; the first-moment state is not reset by this identity. -/
theorem gain_cycle_unclipped_callback (amplitude : ℝ)
    (h0 : 0 ≤ amplitude) (h2 : amplitude ≤ 2) :
    gainCEGradientScale 0 3 2 10 (seededNativeSubweights ((amplitude, amplitude), (0, 0))) =
      1 / (Real.exp (3 * amplitude ^ 2) + 1) ∧
    appliedGainNativeGradient 0 3 2 10 (seededNativeSubweights ((amplitude, amplitude), (0, 0))) =
      ![-gainCycleInputMagnitude amplitude, -gainCycleInputMagnitude amplitude, 0, 0] := by
  have hn := gain_cycle_raw_norm_ceiling amplitude h0 h2
  have hclip : coordinateClipFactor 10
      (rawGainNativeGradient 0 3 2
      (seededNativeSubweights ((amplitude, amplitude), (0, 0)))) = 1 := by
    apply (coordinate_clip_exact_threshold _ _).mpr
    linarith only [hn]
  have hscore : gainNativeTotalScore 3 2 (seededNativeSubweights ((amplitude, amplitude), (0, 0))) =
      3 * amplitude ^ 2 := by
    norm_num [gainNativeTotalScore, physicalCircuitScore,
      seededNativeSubweights, seededScalarState]
    ring
  constructor
  · simp only [gainCEGradientScale, hclip, hscore]
    norm_num
  · change (fun i => coordinateClipFactor 10
      (rawGainNativeGradient 0 3 2 (seededNativeSubweights ((amplitude, amplitude), (0, 0)))) *
      rawGainNativeGradient 0 3 2 (seededNativeSubweights ((amplitude, amplitude), (0, 0))) i) = _
    rw [hclip]
    simpa only [one_mul] using gain_cycle_raw_callback amplitude

example : (0 : ℝ) ≤ 2 ∧ (2 : ℝ) ≤ 2 := by norm_num

/-- The true two amplitudes have sharply separated positive CE
inputs. Sources: appendix C's exponential train denominator and
the actual binary gained callback at 94c077f; the inequalities are
derived from exp(3) >= 4, not from chosen future optimizer inputs. -/
theorem gain_cycle_input_separation :
    0 < gainCycleInputMagnitude 1 ∧ 0 < gainCycleInputMagnitude 2 ∧
      3 * gainCycleInputMagnitude 2 < gainCycleInputMagnitude 1 ∧
        gainCycleInputMagnitude 1 < 3 := by
  have hlow : 0 < gainCycleInputMagnitude 1 := by
    unfold gainCycleInputMagnitude
    positivity
  have hhigh : 0 < gainCycleInputMagnitude 2 := by
    unfold gainCycleInputMagnitude
    positivity
  have he : 4 ≤ Real.exp 3 := by
    linarith only [Real.add_one_le_exp 3]
  have hp : (64 : ℝ) ≤ Real.exp 3 ^ 3 := by
    have ht := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 4) he 3
    norm_num at ht
    exact ht
  have hquartic :=
    mul_le_mul_of_nonneg_right hp (le_of_lt (Real.exp_pos 3))
  have h12 : Real.exp 12 = Real.exp 3 ^ 4 := by
    have ht := Real.exp_nat_mul (3 : ℝ) 4
    norm_num at ht
    exact ht
  have hgap : 3 * gainCycleInputMagnitude 2 < gainCycleInputMagnitude 1 := by
    norm_num [gainCycleInputMagnitude]
    rw [h12]
    field_simp
    nlinarith only [he, hquartic]
  have hc :=
    (gain_cycle_unclipped_callback 1 (by norm_num) (by norm_num)).1
  have hs := (gain_ce_gradient_scale_unit_interval 0 3 2 10
    (seededNativeSubweights ((1, 1), (0, 0))) (by norm_num)).2
  rw [hc] at hs
  have hbound : gainCycleInputMagnitude 1 < 3 := by
    norm_num [gainCycleInputMagnitude] at hs ⊢
    rw [div_eq_mul_inv]
    nlinarith only [hs]
  exact ⟨hlow, hhigh,
    hgap, hbound⟩

end Transformer.Grokking.CircuitEfficiency
