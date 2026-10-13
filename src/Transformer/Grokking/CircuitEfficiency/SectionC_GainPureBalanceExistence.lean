import Transformer.Grokking.CircuitEfficiency.SectionC_GainContinuity
import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdRegime
import Mathlib.Topology.Order.IntermediateValue

/-!
# Actual positive pure-circuit balance at the original numerical settings

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C
product CE; native necessary balance at c5c2105 and actual gained
callback at c268c1f. Keep gains 3/2, cap one, decay 0.1 and epsilon
1e-8. Either circuit admits a positive equal-factor pure balance.

The numerical residual evaluates the actual complete clipped CE at
its physical point. Continuity, all cold competitors and opposite
endpoint signs give a root strictly between amplitudes zero and ten.
Its four true applied CE partials then satisfy original native balance.
No feedback value, epsilon or decay is chosen to fit a supplied point.

These are genuine balanced points with native zero-buffer seeds.
Retained trajectories and their selection from other seeds need
separate proofs. In particular, coexistence exclusion does not choose
Gen over Mem: both pure alternatives have positive actual points.
The source's coupled circuit-norm GD, learned transformer features,
floating-point kernels and stochastic transfer are not asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Actual equal-factor Gen-only (side zero) or Mem-only (side one)
physical point with native zero buffers. Source: appendix C products;
its amplitude and pair index both enter the four numerical coordinates. -/
def originalPureCircuitPoint (side : Fin 2) (amplitude : ℝ) : NativeSubweightState :=
  if side = 0 then seededNativeSubweights ((amplitude, amplitude), (0, 0))
  else seededNativeSubweights ((0, 0), (amplitude, amplitude))

/-- Actual normalized native balance residual before multiplying by
the positive amplitude. Sources: appendix C true CE and native
normalization at c5c2105, with original numerical epsilon/decay. -/
noncomputable def originalPureCircuitResidual (remaining : ℕ) (side : Fin 2) (amplitude : ℝ) : ℝ :=
  gainCEGradientScale remaining 3 2 1 (originalPureCircuitPoint side amplitude) *
    (if side = 0 then 3 else 2) * (1 - (1 / 10) * amplitude) - 1 / 1000000000

/-- The true pure-circuit residual is continuous, including the cold
point and clipping boundary. Sources: appendix C full CE and native
clip strip at 9970d92; no assigned scale family is supplied. -/
theorem original_pure_circuit_residual_continuous (remaining : ℕ) (side : Fin 2) :
    Continuous (originalPureCircuitResidual remaining side) := by
  change Continuous (fun amplitude => originalPureCircuitResidual remaining side amplitude)
  have hp : ∀ i, Continuous (fun x => (originalPureCircuitPoint side x i).parameter) := by
    intro i
    fin_cases side <;> fin_cases i <;> norm_num [originalPureCircuitPoint, seededNativeSubweights, seededScalarState]
    all_goals first | exact continuous_id | exact continuous_const
  have hc := gain_ce_gradient_scale_continuous remaining 3 2 1 (originalPureCircuitPoint side) hp
  have hunit : Continuous (fun _ : ℝ => (1 : ℝ)) := continuous_const
  have htail := hunit.sub (continuous_id.const_mul (1 / 10 : ℝ))
  have hcost : Continuous (fun _ : ℝ => (1 / 1000000000 : ℝ)) := continuous_const
  simpa only [originalPureCircuitResidual, Pi.mul_def, Pi.sub_def, id_eq] using
    ((hc.mul_const (if side = 0 then 3 else 2)).mul htail).sub hcost

/-- Actual cold CE and complete native clipping give a positive
residual at zero and negative residual at ten for either circuit.
Sources: appendix C all competitors, cold feedback at ca253e4 and
original native epsilon/decay at 0033b1b; no class bound is required. -/
theorem original_pure_circuit_residual_endpoint_signs (remaining : ℕ) (side : Fin 2) :
    0 < originalPureCircuitResidual remaining side 0 ∧ originalPureCircuitResidual remaining side 10 < 0 := by
  have hz : ∀ i, (originalPureCircuitPoint side 0 i).parameter = 0 := by
    intro i
    fin_cases side <;> fin_cases i <;> norm_num [originalPureCircuitPoint, seededNativeSubweights, seededScalarState]
  have hc := gain_ce_gradient_scale_zero_parameters remaining 3 2 1 (originalPureCircuitPoint side 0) hz
  have hf : (1 / 2 : ℝ) ≤ ((remaining : ℝ) + 1) / ((remaining : ℝ) + 2) := by
    apply (le_div_iff₀ (by positivity)).mpr
    have hn : (0 : ℝ) ≤ remaining := by positivity
    linarith only [hn]
  have hg : (2 : ℝ) ≤ (if side = 0 then 3 else 2) := by split_ifs <;> norm_num
  have hprod := mul_le_mul hf hg (by norm_num) (le_trans (by norm_num) hf)
  constructor
  · unfold originalPureCircuitResidual
    rw [hc, cold_gain_ce_gradient_scale_unit_cap]
    nlinarith only [hprod]
  · norm_num [originalPureCircuitResidual]

/-- A strictly positive pure-circuit amplitude solves the actual
original-settings balance residual. Sources: appendix C true CE,
continuity and native normalized balance at c5c2105; the root lies
inside a fixed finite interval, with no chosen-feedback premise. -/
theorem original_pure_circuit_residual_root (remaining : ℕ) (side : Fin 2) :
    ∃ amplitude : ℝ, 0 < amplitude ∧ amplitude < 10 ∧ originalPureCircuitResidual remaining side amplitude = 0 := by
  obtain ⟨hzero, hten⟩ := original_pure_circuit_residual_endpoint_signs remaining side
  have hc := original_pure_circuit_residual_continuous remaining side
  have himage : (0 : ℝ) ∈ originalPureCircuitResidual remaining side '' Set.Icc 0 10 :=
    intermediate_value_Icc' (by norm_num) hc.continuousOn ⟨le_of_lt hten, le_of_lt hzero⟩
  obtain ⟨amplitude, ha, hr⟩ := himage
  have hpos : 0 < amplitude := by
    by_contra hn
    have heq : amplitude = 0 := by linarith only [ha.1, le_of_not_gt hn]
    rw [heq] at hr
    linarith only [hzero, hr]
  have hlt : amplitude < 10 := by
    by_contra hn
    have heq : amplitude = 10 := by linarith only [ha.2, le_of_not_gt hn]
    rw [heq] at hr
    linarith only [hten, hr]
  exact ⟨amplitude, hpos, hlt, hr⟩

/-- An actual residual root satisfies all four original native
coordinate balance equations with its own complete clipped CE.
Sources: appendix C partners and normalized AdamW at c5c2105;
the inactive pair has true zero inputs, not dropped equations. -/
theorem original_pure_circuit_root_balance (remaining : ℕ) (side : Fin 2) (amplitude : ℝ)
    (ha : 0 < amplitude) (hr : originalPureCircuitResidual remaining side amplitude = 0) :
    ∀ i, (1 / 10 : ℝ) * (originalPureCircuitPoint side amplitude i).parameter +
      appliedGainNativeGradient remaining 3 2 1 (originalPureCircuitPoint side amplitude) i /
        (|appliedGainNativeGradient remaining 3 2 1 (originalPureCircuitPoint side amplitude) i| + 1 / 100000000) = 0 := by
  let point := originalPureCircuitPoint side amplitude
  let scale := gainCEGradientScale remaining 3 2 1 point
  let gain : ℝ := if side = 0 then 3 else 2
  have hs : 0 < scale := gain_ce_gradient_scale_pos remaining 3 2 1 point (by norm_num)
  have hg : 0 < gain := by dsimp [gain]; split_ifs <;> norm_num
  have hd : scale * gain * amplitude + 1 / 100000000 ≠ 0 := by positivity
  have hres : scale * gain * (1 - (1 / 10) * amplitude) - 1 / 1000000000 = 0 := hr
  have hsteady : (1 / 10 : ℝ) * amplitude + -(scale * gain * amplitude) /
      (scale * gain * amplitude + 1 / 100000000) = 0 := by
    have hscaled := congrArg (fun value : ℝ => value * amplitude) hres
    field_simp [hd]
    nlinarith only [hscaled]
  have habs : |-(scale * gain * amplitude)| = scale * gain * amplitude := by
    rw [abs_of_neg (by have hp := mul_pos (mul_pos hs hg) ha; linarith only [hp])]
    ring
  intro i
  rw [gain_native_applied_gradient_scale]
  fin_cases side <;> fin_cases i
  all_goals first
    | change (1 / 10 : ℝ) * amplitude + -(scale * gain * amplitude) /
        (|-(scale * gain * amplitude)| + 1 / 100000000) = 0
      rw [habs]
      exact hsteady
    | norm_num [point, originalPureCircuitPoint, seededNativeSubweights, seededScalarState, nativeFactorGain, nativeFactorPartner]

example : ∃ amplitude : ℝ, 0 < amplitude ∧ originalPureCircuitResidual 95 0 amplitude = 0 := by
  obtain ⟨amplitude, ha, _, hr⟩ := original_pure_circuit_residual_root 95 0
  exact ⟨amplitude, ha, hr⟩

/-- Either pure circuit has a positive actual original-settings
balanced point, with both buffers initialized normally. Sources:
appendix C products and native normalized balance at c5c2105;
existence is proved at fixed epsilon, decay and full CE feedback. -/
theorem original_pure_circuit_balanced_point_exists (remaining : ℕ) (side : Fin 2) :
    ∃ amplitude : ℝ, 0 < amplitude ∧ amplitude < 10 ∧
      ∀ i, (1 / 10 : ℝ) * (originalPureCircuitPoint side amplitude i).parameter +
        appliedGainNativeGradient remaining 3 2 1 (originalPureCircuitPoint side amplitude) i /
          (|appliedGainNativeGradient remaining 3 2 1 (originalPureCircuitPoint side amplitude) i| + 1 / 100000000) = 0 := by
  obtain ⟨amplitude, ha, hlt, hr⟩ := original_pure_circuit_residual_root remaining side
  exact ⟨amplitude, ha, hlt, original_pure_circuit_root_balance remaining side amplitude ha hr⟩

end Transformer.Grokking.CircuitEfficiency
