import Transformer.Grokking.CircuitEfficiency.SectionC_GainAllocation
import Transformer.Grokking.CircuitEfficiency.SectionC_NativeHistory

/-!
# Native balanced points are actual retained zero-buffer CE trajectories

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C;
native AdamW/clipping at lab commit 68402e2. Starting at any physical
parameter point satisfying the actual normalized native balance,
construct its entire zero-initialized retained history and identify
it with the actual closed gained-CE path. Constant inputs are derived
from unchanged actual parameters, not stipulated as a future stream.

Both buffers evolve from zero, valid bias clocks grow, and every
corrected direction is derived from those full histories. All gains,
parameters, native betas and the shared clip bound enter the candidate.
One uniform decay/epsilon is used. The positive unequal-gain point
supplies joint witnesses at ordinary beta1=0.9/beta2=0.98, with its
explicitly chosen positive epsilon and uniform decay one half.

Generic identities use the exact-real division convention and do
not need a positive epsilon as an unused hypothesis; the concrete
witness has positive epsilon. This proves actual nonzero paths and
finite positive parameter limits, not their attraction from other
seeds, a delayed transition, learned feature discovery, stochastic
GPTMini or a floating-point implementation bridge.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Candidate full numerical history at a physical point, later
identified with actual CE feedback. Source: native retained histories
at 68402e2; each coordinate uses its actual gained-point CE partial. -/
noncomputable def gainBalancedHistory (remaining : ℕ) (genGain memGain bound b1 b2 : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (clock : ℕ) : NativeSubweightState :=
  let point := seededNativeSubweights parameters
  fun i =>
    { parameter := (point i).parameter
      moment := firstMomentAt b1 (fun _ => appliedGainNativeGradient remaining genGain memGain bound point i) clock
      variance := secondMomentAt b2 (fun _ => appliedGainNativeGradient remaining genGain memGain bound point i) clock
      clock := clock }

/-- Actual gained CE inputs along the candidate are the point inputs.
Sources: appendix C product forward and native clipping at 68402e2;
the numerical buffers/clocks are not assumed equal to point data. -/
theorem gain_native_balanced_history_gradient (remaining : ℕ) (genGain memGain bound b1 b2 : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (clock : ℕ) :
    appliedGainNativeGradient remaining genGain memGain bound
      (gainBalancedHistory remaining genGain memGain bound b1 b2 parameters clock) =
      appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) := by
  apply gain_native_equal_parameters_applied_gradient
  intro i
  rfl

/-- The actual native update advances both retained histories and
keeps the balanced physical parameters. Sources: true appendix C CE
and native AdamW at 68402e2; bias correction follows the growing clock. -/
theorem gain_native_balanced_history_step (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (clock : ℕ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hb : ∀ i, decay * (seededNativeSubweights parameters i).parameter +
      appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) i /
        (|appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) i| + eps) = 0) :
    gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate
      (gainBalancedHistory remaining genGain memGain bound b1 b2 parameters clock) =
      gainBalancedHistory remaining genGain memGain bound b1 b2 parameters (clock + 1) := by
  let point := seededNativeSubweights parameters
  let input := appliedGainNativeGradient remaining genGain memGain bound point
  have hd : ∀ i, nextBufferDirection b1 b2 eps (firstMomentAt b1 (fun _ => input i) clock)
      (secondMomentAt b2 (fun _ => input i) clock) (input i) clock = input i / (|input i| + eps) := by
    intro i
    rw [← history_direction_succ b1 b2 eps (fun _ => input i),
      constant_gradient_direction b1 b2 eps (input i) (clock + 1) hb1 h1 hb2 h2 (by omega)]
  funext i
  change scalarNativeStep b1 b2 eps decay rate _
    (appliedGainNativeGradient remaining genGain memGain bound
      (gainBalancedHistory remaining genGain memGain bound b1 b2 parameters clock) i) = _
  rw [gain_native_balanced_history_gradient]
  unfold scalarNativeStep gainBalancedHistory
  dsimp only
  rw [hd]
  congr 1
  change (1 - rate * decay) * (point i).parameter - rate * (input i / (|input i| + eps)) = (point i).parameter
  calc
    _ = (point i).parameter - rate * (decay * (point i).parameter + input i / (|input i| + eps)) := by ring
    _ = (point i).parameter := by rw [hb i]; ring

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (∀ i, (1 / 2 : ℝ) * (point i).parameter + appliedGainNativeGradient 111 3 2 1 point i /
      (|appliedGainNativeGradient 111 3 2 1 point i| + eps) = 0) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    (gain_native_efficient_point_balance 111 1 (by norm_num)).2⟩

/-- The entire candidate history equals the actual closed path from
native zero-buffer seeds. Sources: appendix C CE and native AdamW at
68402e2; induction verifies actual current-state feedback at each step. -/
theorem gain_native_balanced_path_history (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (clock : ℕ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hb : ∀ i, decay * (seededNativeSubweights parameters i).parameter +
      appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) i /
        (|appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) i| + eps) = 0) :
    gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights parameters) clock =
      gainBalancedHistory remaining genGain memGain bound b1 b2 parameters clock := by
  induction clock with
  | zero =>
    funext i
    fin_cases i <;> rfl
  | succ clock ih =>
    change gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate _ = _
    rw [ih]
    exact gain_native_balanced_history_step remaining genGain memGain bound b1 b2 eps decay rate parameters clock hb1 h1 hb2 h2 hb

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (∀ i, (1 / 2 : ℝ) * (point i).parameter + appliedGainNativeGradient 111 3 2 1 point i /
      (|appliedGainNativeGradient 111 3 2 1 point i| + eps) = 0) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    (gain_native_efficient_point_balance 111 1 (by norm_num)).2⟩

/-- Actual retained zero-buffer balanced paths have their finite
physical parameter limits. Sources: appendix C CE and native AdamW
at 68402e2; convergence is derived for these paths, not supplied. -/
theorem gain_native_balanced_path_parameter_tendsto (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (i : Fin 4)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hb : ∀ j, decay * (seededNativeSubweights parameters j).parameter +
      appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) j /
        (|appliedGainNativeGradient remaining genGain memGain bound (seededNativeSubweights parameters) j| + eps) = 0) :
    Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights parameters) n i).parameter) atTop
      (nhds (seededNativeSubweights parameters i).parameter) := by
  apply Tendsto.congr (f₁ := fun _ : ℕ => (seededNativeSubweights parameters i).parameter) _ tendsto_const_nhds
  intro n
  rw [gain_native_balanced_path_history remaining genGain memGain bound b1 b2 eps decay rate parameters n hb1 h1 hb2 h2 hb]
  rfl

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (∀ i, (1 / 2 : ℝ) * (point i).parameter + appliedGainNativeGradient 111 3 2 1 point i /
      (|appliedGainNativeGradient 111 3 2 1 point i| + eps) = 0) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    (gain_native_efficient_point_balance 111 1 (by norm_num)).2⟩

end Transformer.Grokking.CircuitEfficiency
