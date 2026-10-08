import Transformer.Grokking.CircuitEfficiency.SectionC_GainPureBalanceExistence
import Transformer.Grokking.CircuitEfficiency.SectionC_GainWarmupBounds
import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedPath

/-!
# Actual original-warmup pure-circuit retained trajectories

Sources: Varma et al., arXiv:2309.02390v1, appendix C product CE;
actual native balanced histories at 68402e2 and the completed-update
warmup at 0033b1b. Generalize the initialized actual balanced-path
history to arbitrary supplied rates, including the first zero rate.
The entire corrected numerical history stays the native one.

At original gains 3/2, cap one, betas 0.9/0.98, decay 0.1 and epsilon
1e-8, both pure-circuit roots give actual original-warmup trajectories.
Their physical parameters stay constant; both moment histories are
computed from their actual constant CE inputs, and clocks grow.
The first zero-rate step still inserts those inputs and advances.

No independently assigned gradient or future buffer constraint is
introduced. These paths begin at their proved balanced amplitudes;
they do not show attraction from the competing source-style seeds.
Both buffers start at zero and the optimizer configuration is fixed.
Legal native constants alone therefore do not select a physical circuit.
The source's coupled norm-cost GD, learned GPTMini mechanisms,
numerical and stochastic implementation bridges remain separate tasks.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Supplied current rates preserve actual retained balanced history.
Sources: appendix C full CE and native path at 68402e2; induction is
ported from gain_native_balanced_path_history, replacing its fixed
rate by the current rate without skipping zero-rate insertions. -/
theorem gain_native_scheduled_balanced_path_history (remaining : ℕ) (genGain memGain bound b1 b2 eps decay : ℝ)
    (rate : ℕ → ℝ) (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (clock : ℕ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hb : ∀ i, decay * (seededNativeSubweights parameters i).parameter +
      appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) i /
        (|appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) i| + eps) = 0) :
    gainNativeScheduledPath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights parameters) clock =
      gainBalancedHistory remaining genGain memGain bound b1 b2 parameters clock := by
  induction clock with
  | zero =>
    funext i
    fin_cases i <;> rfl
  | succ clock ih =>
    change gainNativeStep remaining genGain memGain bound b1 b2 eps decay (rate clock) _ = _
    rw [ih]
    exact gain_native_balanced_history_step remaining genGain memGain bound b1 b2 eps decay (rate clock)
      parameters clock hb1 h1 hb2 h2 hb

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (∀ i, (1 / 2 : ℝ) * (point i).parameter + appliedGainNativeGradient 111 3 2 1 point i /
      (|appliedGainNativeGradient 111 3 2 1 point i| + eps) = 0) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    (gain_native_efficient_point_balance 111 1 (by norm_num)).2⟩

/-- Candidate numerical native history of either proved pure point.
Sources: appendix C pair order and actual retained histories at
68402e2; every argument enters the true point and full native state. -/
noncomputable def originalPureCircuitHistory (remaining : ℕ) (side : Fin 2) (amplitude : ℝ) (clock : ℕ) : NativeSubweightState :=
  gainBalancedHistory remaining 3 2 1 (9 / 10) (49 / 50)
    (if side = 0 then ((amplitude, amplitude), (0, 0)) else ((0, 0), (amplitude, amplitude))) clock

/-- A true pure residual root generates its candidate as the entire
actual original-warmup path, including both retained buffers/clocks.
Sources: appendix C CE and native balanced history at 68402e2;
original root balance supplies all four numerical equations. -/
theorem original_pure_circuit_warmup_root_history (remaining : ℕ) (side : Fin 2) (amplitude : ℝ)
    (ha : 0 < amplitude) (hr : originalPureCircuitResidual remaining side amplitude = 0) :
    ∀ n, gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) (originalPureCircuitPoint side amplitude) n =
      originalPureCircuitHistory remaining side amplitude n := by
  let parameters := if side = 0 then ((amplitude, amplitude), (0, 0)) else ((0, 0), (amplitude, amplitude))
  have hp : seededNativeSubweights parameters = originalPureCircuitPoint side amplitude := by
    fin_cases side <;> rfl
  have hb : ∀ i, (1 / 10 : ℝ) * (seededNativeSubweights parameters i).parameter +
      appliedGainNativeGradient remaining 3 2 1 (seededNativeSubweights parameters) i /
        (|appliedGainNativeGradient remaining 3 2 1 (seededNativeSubweights parameters) i| + 1 / 100000000) = 0 := by
    rw [hp]
    exact original_pure_circuit_root_balance remaining side amplitude ha hr
  intro n
  unfold gainNativeWarmupPath originalPureCircuitHistory
  rw [← hp]
  exact gain_native_scheduled_balanced_path_history remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000)
    (1 / 10) grokkingWarmupRate parameters n (by norm_num) (by norm_num) (by norm_num) (by norm_num) hb

example : ∃ amplitude : ℝ, 0 < amplitude ∧ originalPureCircuitResidual 95 1 amplitude = 0 := by
  obtain ⟨amplitude, ha, _, hr⟩ := original_pure_circuit_residual_root 95 1
  exact ⟨amplitude, ha, hr⟩

/-- All four numerical fields follow from the actual pure-root
warmup trajectory. Sources: appendix C true CE and retained native
history at 68402e2; only physical parameters are constant, while
moments and completed clocks keep their actual initialized evolution. -/
theorem original_pure_circuit_warmup_root_fields (remaining : ℕ) (side : Fin 2) (amplitude : ℝ)
    (ha : 0 < amplitude) (hr : originalPureCircuitResidual remaining side amplitude = 0) :
    let point := originalPureCircuitPoint side amplitude
    let path := gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) point
    ∀ n i, (path n i).parameter = (point i).parameter ∧
      (path n i).moment = firstMomentAt (9 / 10) (fun _ => appliedGainNativeGradient remaining 3 2 1 point i) n ∧
      (path n i).variance = secondMomentAt (49 / 50) (fun _ => appliedGainNativeGradient remaining 3 2 1 point i) n ∧
      (path n i).clock = n := by
  dsimp only
  intro n i
  rw [original_pure_circuit_warmup_root_history remaining side amplitude ha hr n]
  fin_cases side <;> exact ⟨rfl, rfl, rfl, rfl⟩

example : ∃ amplitude : ℝ, 0 < amplitude ∧ originalPureCircuitResidual 111 0 amplitude = 0 := by
  obtain ⟨amplitude, ha, _, hr⟩ := original_pure_circuit_residual_root 111 0
  exact ⟨amplitude, ha, hr⟩

/-- Either pure circuit has an actual original-warmup trajectory
at positive amplitude with the full retained numerical history.
Sources: appendix C feedback and original native recipe at 0033b1b;
existence is derived at fixed numerical epsilon, betas and decay. -/
theorem original_pure_circuit_warmup_fields_exist (remaining : ℕ) (side : Fin 2) :
    ∃ amplitude : ℝ, 0 < amplitude ∧ amplitude < 10 ∧
      let point := originalPureCircuitPoint side amplitude
      let path := gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) point
      ∀ n i, (path n i).parameter = (point i).parameter ∧
        (path n i).moment = firstMomentAt (9 / 10) (fun _ => appliedGainNativeGradient remaining 3 2 1 point i) n ∧
        (path n i).variance = secondMomentAt (49 / 50) (fun _ => appliedGainNativeGradient remaining 3 2 1 point i) n ∧
        (path n i).clock = n := by
  obtain ⟨amplitude, ha, hlt, hr⟩ := original_pure_circuit_residual_root remaining side
  exact ⟨amplitude, ha, hlt, original_pure_circuit_warmup_root_fields remaining side amplitude ha hr⟩

/-- Both pure original-warmup trajectories have actual finite
physical parameter limits. Sources: appendix C CE and retained
histories at 68402e2; limits are proved for the constructed paths,
without requiring finite limiting clocks or restarted moments. -/
theorem original_pure_circuit_warmup_parameter_limits_exist (remaining : ℕ) (side : Fin 2) :
    ∃ amplitude : ℝ, 0 < amplitude ∧ amplitude < 10 ∧
      ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000)
        (originalPureCircuitPoint side amplitude) n i).parameter) atTop
          (nhds (originalPureCircuitPoint side amplitude i).parameter) := by
  obtain ⟨amplitude, ha, hlt, hfields⟩ := original_pure_circuit_warmup_fields_exist remaining side
  refine ⟨amplitude, ha, hlt, ?_⟩
  intro i
  apply Tendsto.congr (f₁ := fun _ : ℕ => (originalPureCircuitPoint side amplitude i).parameter)
    _ tendsto_const_nhds
  intro n
  exact ((hfields n i).1).symm

end Transformer.Grokking.CircuitEfficiency
