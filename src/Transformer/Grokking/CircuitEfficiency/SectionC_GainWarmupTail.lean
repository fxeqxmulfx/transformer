import Transformer.Grokking.CircuitEfficiency.SectionC_GainPureWarmupPaths
import Transformer.Grokking.CircuitEfficiency.SectionC_GainLimitBalance

/-!
# Full retained warmup tails and actual finite-limit balance

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C,
eq. sim-overall-logits and complete product CE; native AdamW at
88aa892 and completed-update warmup at 0033b1b. The source uses
coupled circuit-norm GD; here fixed physical gains, plain CE,
uniform native decoupled decay and retained buffers remain explicit.

After ten actual warmup steps, the entire native trajectory is
exactly the constant-rate trajectory initialized with its actual
then-current state. Neither moment nor variance nor clock is reset.
The initial zero rate still inserts the true CE input into both
buffers and advances the clock. The tail recomputes every later
input from its current physical parameters and shared clipping.

A positive initial coordinate stays positive through all finite
warmup clocks. Finite physical parameter convergence transfers to
the full retained constant-rate tail. Its necessary numerical
balance is therefore the original decay against the actual complete
clipped CE derivative normalized by its magnitude plus epsilon.
Input and moment convergence follow from the existing closed law;
no independent future input, buffer matching or finite clock limit
is assumed. Positive pure-circuit paths witness the convergence
hypotheses at original epsilon, rather than an epsilon fitted to a
proposed point. Attraction from competing seeds, existence of their
limits and learned stochastic/numerical GPTMini remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- A positive initial physical coordinate persists at every actual
warmup clock. Sources: section 3 formation, appendix C partners and
native sign induction at 65ce284; only initial numerical signs enter. -/
theorem gain_native_warmup_positive_coordinate (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (i : Fin 4)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (he : 0 < eps)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial i).parameter) :
    PositiveScalarState (gainNativeWarmupPath remaining genGain memGain bound eps initial n i) := by
  induction n with
  | zero => exact ⟨hi, (hs i).2⟩
  | succ n ih =>
    have hn := gain_native_scheduled_nonnegative_path remaining genGain memGain bound (9 / 10) (49 / 50)
      eps (1 / 10) grokkingWarmupRate initial n hgen hmem hclip (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) he
      (fun k => ⟨(grokking_warmup_rate_legal k).1, (grokking_warmup_rate_legal k).2.2⟩) hs
    have hd : (0 : ℝ) < 1 - grokkingWarmupRate n * (1 / 10) := by
      nlinarith only [(grokking_warmup_rate_legal n).2.1]
    change PositiveScalarState (scalarNativeStep (9 / 10) (49 / 50) eps (1 / 10) (grokkingWarmupRate n) _ _)
    exact scalar_positive_step (9 / 10) (49 / 50) eps (1 / 10) (grokkingWarmupRate n) _ _
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) he
      (grokking_warmup_rate_legal n).1 hd ih
      (gain_native_applied_gradient_nonpos remaining genGain memGain bound _ i hclip
        (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hn _).1)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 1).parameter := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState]⟩

/-- The complete original-warmup tail equals the actual fixed-rate
path from its true tenth state. Sources: native AdamW at 88aa892 and
completed-update schedule at 0033b1b; all retained fields survive. -/
theorem gain_native_warmup_tail_history (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (initial : NativeSubweightState) (n : ℕ) :
    gainNativeWarmupPath remaining genGain memGain bound eps initial (n + 10) =
      gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
        (gainNativeWarmupPath remaining genGain memGain bound eps initial 10) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have ht : n + 1 + 10 = (n + 10) + 1 := by omega
    rw [ht]
    change gainNativeStep remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10)
        (grokkingWarmupRate (n + 10)) (gainNativeWarmupPath remaining genGain memGain bound eps initial (n + 10)) =
      gainNativeStep remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000) _
    rw [grokking_warmup_rate_profile.2.2 (n + 10) (by omega), ih]

/-- Actual finite warmup parameter limits are also finite limits of
the retained constant-rate tail. Sources: appendix C feedback and
native schedule at 0033b1b; this does not assume moment limits. -/
theorem gain_native_warmup_tail_parameter_limits (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (initial reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining genGain memGain bound eps initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps
      (1 / 10) (1 / 1000) (gainNativeWarmupPath remaining genGain memGain bound eps initial 10) n i).parameter)
        atTop (nhds (reference i).parameter) := by
  intro i
  have hshift := (tendsto_add_atTop_iff_nat
    (f := fun n => (gainNativeWarmupPath remaining genGain memGain bound eps initial n i).parameter) 10).2 (hp i)
  exact Tendsto.congr (fun n => congrArg (fun state : NativeSubweightState => (state i).parameter)
    (gain_native_warmup_tail_history remaining genGain memGain bound eps initial n)) hshift

example : ∃ amplitude : ℝ, 0 < amplitude ∧ amplitude < 10 ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 95 3 2 1 (1 / 100000000)
      (originalPureCircuitPoint 0 amplitude) n i).parameter) atTop
        (nhds (originalPureCircuitPoint 0 amplitude i).parameter) := by
  exact original_pure_circuit_warmup_parameter_limits_exist 95 0

/-- Every actual finite warmup parameter limit satisfies the original
native normalized balance. Sources: appendix C complete CE and native
necessary balance at c5c2105; the warmup tail retains its full history. -/
theorem gain_native_warmup_limit_balance (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (initial reference : NativeSubweightState) (he : 0 < eps)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining genGain memGain bound eps initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    ∀ i, (1 / 10 : ℝ) * (reference i).parameter + appliedGainNativeGradient remaining genGain memGain bound reference i /
      (|appliedGainNativeGradient remaining genGain memGain bound reference i| + eps) = 0 := by
  let state := gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
    (gainNativeWarmupPath remaining genGain memGain bound eps initial 10)
  have htail := gain_native_warmup_tail_parameter_limits remaining genGain memGain bound eps initial reference hp
  intro i
  exact gain_native_closed_limit_balance remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10)
    (1 / 1000) state reference i (fun _ => rfl) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) he (by norm_num) htail

example : (0 : ℝ) < 1 / 100000000 ∧ ∃ amplitude : ℝ, 0 < amplitude ∧ amplitude < 10 ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 111 3 2 1 (1 / 100000000)
      (originalPureCircuitPoint 0 amplitude) n i).parameter) atTop
        (nhds (originalPureCircuitPoint 0 amplitude i).parameter) := by
  exact ⟨by norm_num, original_pure_circuit_warmup_parameter_limits_exist 111 0⟩

/-- Both actual retained warmup buffers have the limits forced by
its finite physical parameter limits. Sources: appendix C feedback
and native history at c5c2105; no independent buffer convergence,
zero restart or finite limiting bias clock is supplied. -/
theorem gain_native_warmup_limit_buffers (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (initial reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining genGain memGain bound eps initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining genGain memGain bound eps initial n i).moment)
        atTop (nhds (appliedGainNativeGradient remaining genGain memGain bound reference i)) ∧
      Tendsto (fun n => (gainNativeWarmupPath remaining genGain memGain bound eps initial n i).variance)
        atTop (nhds (appliedGainNativeGradient remaining genGain memGain bound reference i ^ 2)) := by
  let state := gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
    (gainNativeWarmupPath remaining genGain memGain bound eps initial 10)
  have htail := gain_native_warmup_tail_parameter_limits remaining genGain memGain bound eps initial reference hp
  intro i
  have hb := gain_native_closed_buffer_limits remaining genGain memGain bound (9 / 10) (49 / 50) eps
    (1 / 10) (1 / 1000) state reference i (fun _ => rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) htail
  constructor
  · apply (tendsto_add_atTop_iff_nat 10).1
    exact Tendsto.congr (fun n => congrArg (fun current : NativeSubweightState => (current i).moment)
      (gain_native_warmup_tail_history remaining genGain memGain bound eps initial n).symm) hb.1
  · apply (tendsto_add_atTop_iff_nat 10).1
    exact Tendsto.congr (fun n => congrArg (fun current : NativeSubweightState => (current i).variance)
      (gain_native_warmup_tail_history remaining genGain memGain bound eps initial n).symm) hb.2

example : ∃ amplitude : ℝ, 0 < amplitude ∧ amplitude < 10 ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 95 3 2 1 (1 / 100000000)
      (originalPureCircuitPoint 1 amplitude) n i).parameter) atTop
        (nhds (originalPureCircuitPoint 1 amplitude i).parameter) := by
  exact original_pure_circuit_warmup_parameter_limits_exist 95 1

end Transformer.Grokking.CircuitEfficiency
