import Transformer.Grokking.CircuitEfficiency.SectionC_GainGradientLimits

/-!
# Necessary limits of actual uniformly decayed physical-gain feedback

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C;
retained native AdamW/clipping at lab commit c5c2105. Apply the scalar
finite-limit law to the actual gained-forward CE trajectory. Derive
every input limit from current parameters; native moment limits and
the growing clock then follow from the retained scalar recurrence.

Any finite parameter limit at a positive constant rate must balance
its uniform decay against the actual normalized clipped derivative.
For positive physical gains and positive cap, the only finite
no-decay limit is the fully zero parameter point. No independent
input/moment convergence or instantaneous buffer matching enters.
Fully cold actual traces witness every joint convergence hypothesis.

These are necessary conditional laws, not attraction, global
convergence or rule selection. Physical readout gains and plain CE
remain distinct from the source's asymmetric coupled norm cost/GD.
The source tables remain fixed, and stochastic learned GPTMini,
finite-precision forward/autograd and kernel bridges remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Actual finite parameter limits satisfy native coordinate
balance with their actual gained CE input. Sources: appendix C
product CE and native AdamW at c5c2105; input and moment limits are
derived and the reference clock is not declared a finite clock limit. -/
theorem gain_native_closed_limit_balance (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState) (i : Fin 4)
    (hstep : ∀ n, state (n + 1) = gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (state n))
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    decay * (reference i).parameter + appliedGainNativeGradient remaining genGain memGain bound reference i /
      (|appliedGainNativeGradient remaining genGain memGain bound reference i| + eps) = 0 := by
  apply scalar_limit_balance b1 b2 eps decay rate (reference i).parameter
    (appliedGainNativeGradient remaining genGain memGain bound reference i) (fun n => state n i)
    (fun n => appliedGainNativeGradient remaining genGain memGain bound (state n) i)
    _ hb1 h1 hb2 h2 he heta (hp i)
    (gain_native_applied_gradient_tendsto remaining genGain memGain bound state reference i hp)
  intro n
  exact congrFun (hstep n) i

example : (∀ n : ℕ, (fun _ : Fin 4 => zeroScalarStateAt (n + 1)) =
      gainNativeStep 111 3 2 1 (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (fun _ => zeroScalarStateAt n)) ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 1000 ∧
    (∀ i : Fin 4, Tendsto (fun n : ℕ => (zeroScalarStateAt n).parameter)
      atTop (nhds ((fun _ : Fin 4 => zeroScalarStateAt 0) i).parameter)) := by
  refine ⟨fun n => (gain_native_cold_step 111 3 2 1 _ _ _ _ _ n).symm,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  exact tendsto_const_nhds

/-- Both retained coordinate buffers have limits forced by the
actual gained-CE feedback. Sources: appendix C true CE and native
AdamW at c5c2105; all earlier inputs and finite initial buffers remain
in the causal recurrences, and no limiting finite clock is assumed. -/
theorem gain_native_closed_buffer_limits (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState) (i : Fin 4)
    (hstep : ∀ n, state (n + 1) = gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (state n))
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    Tendsto (fun n => (state n i).moment) atTop
      (nhds (appliedGainNativeGradient remaining genGain memGain bound reference i)) ∧
    Tendsto (fun n => (state n i).variance) atTop
      (nhds (appliedGainNativeGradient remaining genGain memGain bound reference i ^ 2)) := by
  let input := fun n => appliedGainNativeGradient remaining genGain memGain bound (state n) i
  let value := appliedGainNativeGradient remaining genGain memGain bound reference i
  have hi : ∀ n, state (n + 1) i = scalarNativeStep b1 b2 eps decay rate (state n i) (input n) := by
    intro n
    exact congrFun (hstep n) i
  have hg : Tendsto input atTop (nhds value) := gain_native_applied_gradient_tendsto remaining genGain memGain bound state reference i hp
  constructor
  · exact Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate (fun n => state n i) input n hi).1.symm)
      (moment_tendsto_of_input_tendsto b1 (state 0 i).moment value input hb1 h1 hg)
  · exact Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate (fun n => state n i) input n hi).2.1.symm)
      (moment_tendsto_of_input_tendsto b2 (state 0 i).variance (value ^ 2) (fun n => input n ^ 2) hb2 h2 (hg.pow 2))

example : (∀ n : ℕ, (fun _ : Fin 4 => zeroScalarStateAt (n + 1)) =
      gainNativeStep 111 3 2 1 (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (fun _ => zeroScalarStateAt n)) ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (∀ i : Fin 4, Tendsto (fun n : ℕ => (zeroScalarStateAt n).parameter)
      atTop (nhds ((fun _ : Fin 4 => zeroScalarStateAt 0) i).parameter)) := by
  refine ⟨fun n => (gain_native_cold_step 111 3 2 1 _ _ _ _ _ n).symm,
    by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  exact tendsto_const_nhds

/-- With positive physical readouts, zero actual applied derivatives
force every physical coordinate to vanish. Sources: appendix C's
partner chain rule and native clipping at c5c2105; the shared true CE
scale is strictly positive at every finite parameter reference. -/
theorem gain_native_zero_gradient_parameters (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hg : ∀ i, appliedGainNativeGradient remaining genGain memGain bound state i = 0) :
    ∀ i, (state i).parameter = 0 := by
  have hs := gain_ce_gradient_scale_pos remaining genGain memGain bound state hclip
  have hgain : ∀ i, 0 < nativeFactorGain genGain memGain i := by
    intro i
    fin_cases i
    · exact hgen
    · exact hgen
    · exact hmem
    · exact hmem
  intro i
  have hi : nativeFactorPartner (nativeFactorPartner i) = i := by fin_cases i <;> rfl
  have hz := hg (nativeFactorPartner i)
  rw [gain_native_applied_gradient_scale, hi] at hz
  have hp : (gainCEGradientScale remaining genGain memGain bound state *
      nativeFactorGain genGain memGain (nativeFactorPartner i)) * (state i).parameter = 0 := by linarith only [hz]
  exact (mul_eq_zero.mp hp).resolve_left (ne_of_gt (mul_pos hs (hgain _)))

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧
    (∀ i, appliedGainNativeGradient 111 3 2 1 (fun _ => zeroScalarStateAt 0) i = 0) := by
  exact ⟨by norm_num, by norm_num, by norm_num, gain_native_cold_applied_gradient 111 3 2 1 0⟩

/-- No-decay actual gained native feedback has only the fully zero
possible finite parameter limit. Sources: appendix C true product CE
and native AdamW at c5c2105; gradient, moment and parameter-sign
convergence assumptions are not independently supplied. Both positive
readouts, positive cap, epsilon and constant rate remain explicit. -/
theorem gain_native_no_decay_finite_limit_zero (remaining : ℕ) (genGain memGain bound b1 b2 eps rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = gainNativeStep remaining genGain memGain bound b1 b2 eps 0 rate (state n))
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    ∀ i, (reference i).parameter = 0 := by
  apply gain_native_zero_gradient_parameters remaining genGain memGain bound reference hgen hmem hclip
  intro i
  have hb := gain_native_closed_limit_balance remaining genGain memGain bound b1 b2 eps 0 rate state reference i
    hstep hb1 h1 hb2 h2 he heta hp
  have hz : appliedGainNativeGradient remaining genGain memGain bound reference i /
      (|appliedGainNativeGradient remaining genGain memGain bound reference i| + eps) = 0 := by
    simpa only [zero_mul, zero_add] using hb
  have hd : |appliedGainNativeGradient remaining genGain memGain bound reference i| + eps ≠ 0 := by positivity
  exact (div_eq_zero_iff.mp hz).resolve_right hd

example : (∀ n : ℕ, (fun _ : Fin 4 => zeroScalarStateAt (n + 1)) =
      gainNativeStep 111 3 2 1 (9 / 10) (49 / 50) (1 / 100000000) 0 (1 / 1000)
        (fun _ => zeroScalarStateAt n)) ∧
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 1000 ∧
    (∀ i : Fin 4, Tendsto (fun n : ℕ => (zeroScalarStateAt n).parameter)
      atTop (nhds ((fun _ : Fin 4 => zeroScalarStateAt 0) i).parameter)) := by
  refine ⟨fun n => (gain_native_cold_step 111 3 2 1 _ _ _ _ _ n).symm,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  exact tendsto_const_nhds

end Transformer.Grokking.CircuitEfficiency
