import Transformer.Grokking.CircuitEfficiency.SectionC_GainFeedbackFloor
import Transformer.Grokking.AdamW.SquareParameterBox

/-!
# Sharper physical boxes for actual retained gained-CE feedback

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C,
eq. sim-overall-logits and product train CE; native PyTorch AdamW
at 88aa892/0033b1b. Actual full-CE partials and shared norm clipping
generate all four coordinate inputs on the closed native path.

Transfer the initialized same-input moment-square estimate to these
generated inputs. Positive decay produces a numerical physical box
independent of epsilon magnitude and clipping magnitude, with no
future gradient, parameter, moment, denominator or convergence
premise. Original retained betas/rate/decay keep absolute initial
coordinates at most 17 inside that same box.

For nonnegative seeds and positive ordered gains, derive the explicit
complete clipped CE floor at ceiling 17, and its actual partner-input
floor at every clock. These are still qualitative exact-real bounds:
the exponential score ceiling can make the feedback floor very small.
Boundedness and a positive CE coefficient do not establish Gen/Mem
attraction, a transition time or generalization by learned mechanisms.
Fixed gained tables, constant rate and uniform native decoupled decay
differ from the source's coupled circuit-norm GD and real GPTMini
learning. No floating-point, warmup or stochastic transfer is supplied.

The score bound retains both gained product logits and all
remaining+2 finite classes. The clipping floor keeps the actual
added 1e-6 norm denominator; neither contribution is discarded.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Actual closed seeded full-CE feedback inherits the generated
moment-square physical box. Sources: appendix C product CE and
native AdamW at 88aa892; no independent gradient stream, clipping
bound or future state inequality enters this initialized result. -/
theorem gain_native_seeded_square_parameter_box (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (parameters : (ℝ × ℝ) × (ℝ × ℝ))
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2) (he : 0 < eps)
    (hdecay : 0 < decay) (hrate : 0 ≤ rate) (hkeep : 0 ≤ 1 - rate * decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights parameters)
    ∀ n i, |(path n i).parameter| ≤
      max |(seededNativeSubweights parameters i).parameter| (Real.sqrt (nativeMomentSquareScale b1 b2) / decay) := by
  dsimp only
  intro n i
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights parameters)
  have hz : (path 0 i).moment = 0 ∧ (path 0 i).variance = 0 ∧ (path 0 i).clock = 0 := by
    fin_cases i <;> simp [path, gainNativePath, seededNativeSubweights, seededScalarState]
  exact scalar_initialized_parameter_box b1 b2 eps decay rate (fun k => path k i)
    (fun k => appliedGainNativeGradient remaining genGain memGain bound (path k) i)
    (fun k => rfl) hz.1 hz.2.1 hz.2.2 hb1 h1 h2 hgap he hdecay hrate hkeep n

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧
    (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧ (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (1 / 10) := by norm_num

/-- Original weak-decay full-CE paths preserve the numerical box
17 from bounded physical initialization. Sources: appendix C
two-factor feedback and native original constants at 0033b1b;
epsilon stays positive and no future input/state bounds are assumed. -/
theorem gain_native_seeded_original_weak_box (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (he : 0 < eps)
    (hi : ∀ i, |(seededNativeSubweights parameters i).parameter| ≤ 17) :
    ∀ n i, |(gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
      (seededNativeSubweights parameters) n i).parameter| ≤ 17 := by
  let path := gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
    (seededNativeSubweights parameters)
  have hz : ∀ i, (path 0 i).moment = 0 ∧ (path 0 i).variance = 0 ∧ (path 0 i).clock = 0 := by
    intro i
    fin_cases i <;> simp [path, gainNativePath, seededNativeSubweights, seededScalarState]
  exact native_original_weak_decay_coordinate_box (Fin 4) eps path
    (fun n => appliedGainNativeGradient remaining genGain memGain bound (path n))
    (fun n i => rfl) (fun i => (hz i).1) (fun i => (hz i).2.1) (fun i => (hz i).2.2) he hi

example : (0 : ℝ) < 1 / 100000000 ∧
    ∀ i, |(seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter| ≤ 17 := by
  refine ⟨by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Actual original weak-decay feedback has the explicit positive
ceiling-17 CE floor at every retained clock. Sources: appendix C
multiclass CE and native clipping at ee02740/0033b1b; the physical
box and sign data are generated from initial data, not future bounds. -/
theorem gain_native_seeded_original_weak_feedback_floor (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ))
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (horder : memGain ≤ genGain)
    (he : 0 < eps) (hs : NonnegativeNativeState (seededNativeSubweights parameters))
    (hi : ∀ i, |(seededNativeSubweights parameters i).parameter| ≤ 17) :
    0 < gainCEFeedbackFloor remaining genGain memGain bound 17 ∧
      let path := gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
        (seededNativeSubweights parameters)
      ∀ n, (∀ i, 0 ≤ (path n i).parameter ∧ (path n i).parameter ≤ 17) ∧
        gainCEFeedbackFloor remaining genGain memGain bound 17 ≤ gainCEGradientScale remaining genGain memGain bound (path n) ∧
          gainCEGradientScale remaining genGain memGain bound (path n) < 1 := by
  have hbox := gain_native_seeded_original_weak_box remaining genGain memGain bound eps parameters he hi
  refine ⟨gain_ce_feedback_floor_pos remaining genGain memGain bound 17 (le_of_lt hgen) (by norm_num) hclip, ?_⟩
  dsimp only
  intro n
  let state := gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
    (seededNativeSubweights parameters) n
  have hnonneg := gain_native_nonnegative_path remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10)
    (1 / 1000) (seededNativeSubweights parameters) n hgen hmem hclip (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) he (by norm_num) (by norm_num) hs
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

/-- Every actual original weak-decay coordinate input has its
generated ceiling-17 partner floor at all clocks. Sources: appendix C
chain rule and native full-vector clipping at ee02740/0033b1b;
no independent input stream, future sign or box is prescribed. -/
theorem gain_native_seeded_original_weak_partner_floor (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ))
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (horder : memGain ≤ genGain)
    (he : 0 < eps) (hs : NonnegativeNativeState (seededNativeSubweights parameters))
    (hi : ∀ i, |(seededNativeSubweights parameters i).parameter| ≤ 17) :
    let path := gainNativePath remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
      (seededNativeSubweights parameters)
    ∀ n i, gainCEFeedbackFloor remaining genGain memGain bound 17 * nativeFactorGain genGain memGain i *
      (path n (nativeFactorPartner i)).parameter ≤ -appliedGainNativeGradient remaining genGain memGain bound (path n) i := by
  have ht := gain_native_seeded_original_weak_feedback_floor remaining genGain memGain bound eps parameters
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
