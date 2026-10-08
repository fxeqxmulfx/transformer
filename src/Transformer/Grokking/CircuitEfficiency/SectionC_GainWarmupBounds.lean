import Transformer.Grokking.CircuitEfficiency.SectionC_GainSquareBounds
import Transformer.Grokking.AdamW.ScheduledParameterBox

/-!
# Closed gained-CE feedback with the original native warmup

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C,
eq. sim-overall-logits and product train CE; native PyTorch AdamW
and domain.training.rate at 88aa892/0033b1b. Advance the existing
actual gained CE callback with its shared clipping and native
retained scalar update, using the current scheduled rate.

The numerical path computes every new input from its current
physical parameters. Its initial moments and clock are retained,
and every coordinate advances even at rate zero. Numerical sign
preservation extends to any legal nonnegative changing rate.

For original betas, decay and completed-update warmup, zero-buffer
physical seeds within absolute size 17 stay within that same box.
Positive ordered gains and nonnegative seeds give the explicit
positive complete CE and partner-input floors at every actual clock.
No independent future gradient, state bound, input limit, buffer
inequality, parameter convergence or successful decision is assumed.

This is the exact-real fixed-table two-circuit model, with constant
physical gains and uniform native decoupled decay. It differs from
the source's coupled circuit-norm GD and from learned GPTMini Q/K,
attention and FFN mechanisms. The floor can remain very small and
does not predict a transition time. Floating-point schedule/kernel,
stochastic transfer and weak-decay attraction remain separate tasks.
Frozen training, checkpoints and measurements are unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Actual complete gained-CE feedback with a supplied current rate.
Sources: appendix C CE and native update/schedule at c268c1f/0033b1b;
all arguments enter the retained state recurrence and its true input. -/
noncomputable def gainNativeScheduledPath (remaining : ℕ) (genGain memGain bound b1 b2 eps decay : ℝ)
    (rate : ℕ → ℝ) (initial : NativeSubweightState) : ℕ → NativeSubweightState
  | 0 => initial
  | n + 1 => gainNativeStep remaining genGain memGain bound b1 b2 eps decay (rate n)
      (gainNativeScheduledPath remaining genGain memGain bound b1 b2 eps decay rate initial n)

/-- Numerical instantiation of the original retained optimizer and
ten-completed-update warmup. Sources: original lab at 0033b1b;
the path still generates true current full-CE clipped inputs. -/
noncomputable def gainNativeWarmupPath (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (initial : NativeSubweightState) : ℕ → NativeSubweightState :=
  gainNativeScheduledPath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) grokkingWarmupRate initial

/-- Actual scheduled full-CE feedback preserves all numerical signs.
Sources: appendix C partner gradients and native updates at 65ce284;
only rate legality and initial signs, not future CE signs, are inputs. -/
theorem gain_native_scheduled_nonnegative_path (remaining : ℕ) (genGain memGain bound b1 b2 eps decay : ℝ)
    (rate : ℕ → ℝ) (initial : NativeSubweightState) (n : ℕ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps)
    (hlegal : ∀ k, 0 ≤ rate k ∧ 0 ≤ 1 - rate k * decay) (hs : NonnegativeNativeState initial) :
    NonnegativeNativeState (gainNativeScheduledPath remaining genGain memGain bound b1 b2 eps decay rate initial n) := by
  induction n with
  | zero => exact hs
  | succ n ih =>
    exact gain_native_nonnegative_step remaining genGain memGain bound b1 b2 eps decay (rate n) _
      hgen hmem hclip hb1 h1 hb2 h2 he (hlegal n).1 (hlegal n).2 ih

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (∀ n : ℕ, 0 ≤ grokkingWarmupRate n ∧ 0 ≤ 1 - grokkingWarmupRate n * (1 / 10)) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    fun n => ⟨(grokking_warmup_rate_legal n).1, (grokking_warmup_rate_legal n).2.2⟩,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Actual closed original-warmup CE paths retain physical box 17.
Sources: appendix C feedback and native schedule at 0033b1b;
no independent input stream, clipping bound or future state guard
is supplied to the zero-buffer initialized parameter estimate. -/
theorem gain_native_seeded_warmup_parameter_box (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (he : 0 < eps)
    (hi : ∀ i, |(seededNativeSubweights parameters i).parameter| ≤ 17) :
    ∀ n i, |(gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights parameters) n i).parameter| ≤ 17 := by
  let path := gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights parameters)
  have hz : ∀ i, (path 0 i).moment = 0 ∧ (path 0 i).variance = 0 ∧ (path 0 i).clock = 0 := by
    intro i
    fin_cases i <;> simp [path, gainNativeWarmupPath, gainNativeScheduledPath, seededNativeSubweights, seededScalarState]
  exact native_original_warmup_coordinate_box (Fin 4) eps path
    (fun n => appliedGainNativeGradient remaining genGain memGain bound (path n))
    (fun n i => rfl) (fun i => (hz i).1) (fun i => (hz i).2.1) (fun i => (hz i).2.2) he hi

example : (0 : ℝ) < 1 / 100000000 ∧
    ∀ i, |(seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter| ≤ 17 := by
  refine ⟨by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Original-warmup full CE has the generated positive ceiling-17
feedback floor and unit scale ceiling at every retained clock.
Sources: appendix C multiclass CE and native clipping at ee02740;
physical bounds and signs follow from initialization and actual feedback. -/
theorem gain_native_seeded_warmup_feedback_floor (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ))
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (horder : memGain ≤ genGain)
    (he : 0 < eps) (hs : NonnegativeNativeState (seededNativeSubweights parameters))
    (hi : ∀ i, |(seededNativeSubweights parameters i).parameter| ≤ 17) :
    0 < gainCEFeedbackFloor remaining genGain memGain bound 17 ∧
      let path := gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights parameters)
      ∀ n, (∀ i, 0 ≤ (path n i).parameter ∧ (path n i).parameter ≤ 17) ∧
        gainCEFeedbackFloor remaining genGain memGain bound 17 ≤ gainCEGradientScale remaining genGain memGain bound (path n) ∧
          gainCEGradientScale remaining genGain memGain bound (path n) < 1 := by
  have hbox := gain_native_seeded_warmup_parameter_box remaining genGain memGain bound eps parameters he hi
  refine ⟨gain_ce_feedback_floor_pos remaining genGain memGain bound 17 (le_of_lt hgen) (by norm_num) hclip, ?_⟩
  dsimp only
  intro n
  let state := gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights parameters) n
  have hnonneg := gain_native_scheduled_nonnegative_path remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10)
    grokkingWarmupRate (seededNativeSubweights parameters) n hgen hmem hclip (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) he
    (fun k => ⟨(grokking_warmup_rate_legal k).1, (grokking_warmup_rate_legal k).2.2⟩) hs
  have hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ 17 :=
    fun i => ⟨(hnonneg i).1, le_of_abs_le (hbox n i)⟩
  exact ⟨hp, gain_ce_gradient_scale_box_floor remaining genGain memGain bound 17 state hgen hmem horder
    (by norm_num) hclip hp, (gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip).2⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (2 : ℝ) ≤ 3 ∧
    (0 : ℝ) < 1 / 100000000 ∧ NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    (∀ i, |(seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter| ≤ 17) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Every generated original-warmup coordinate input has its
ceiling-17 partner floor at all clocks. Sources: appendix C chain
rule and native shared clipping at ee02740/0033b1b; no future
partner sign, gradient stream or physical box is postulated. -/
theorem gain_native_seeded_warmup_partner_floor (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ))
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (horder : memGain ≤ genGain)
    (he : 0 < eps) (hs : NonnegativeNativeState (seededNativeSubweights parameters))
    (hi : ∀ i, |(seededNativeSubweights parameters i).parameter| ≤ 17) :
    let path := gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights parameters)
    ∀ n i, gainCEFeedbackFloor remaining genGain memGain bound 17 * nativeFactorGain genGain memGain i *
      (path n (nativeFactorPartner i)).parameter ≤ -appliedGainNativeGradient remaining genGain memGain bound (path n) i := by
  have ht := gain_native_seeded_warmup_feedback_floor remaining genGain memGain bound eps parameters
    hgen hmem hclip horder he hs hi
  dsimp only
  intro n i
  exact gain_native_applied_gradient_box_floor remaining genGain memGain bound 17 _ i
    hgen hmem horder (by norm_num) hclip (ht.2 n).1

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (2 : ℝ) ≤ 3 ∧
    (0 : ℝ) < 1 / 100000000 ∧ NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    (∀ i, |(seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter| ≤ 17) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
