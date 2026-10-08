import Transformer.Grokking.CircuitEfficiency.SectionC_NativePositiveBalance
import Transformer.Grokking.CircuitEfficiency.SectionC_NativeHistory

/-!
# A nonzero balanced point is an actual retained native CE trajectory

Sources: Varma et al., arXiv:2309.02390v1, appendix C's four product
factors and actual CE; native AdamW/clipping at lab commit 9c2c42b.
Start all parameters at one with the actual zero native buffers. Choose
epsilon from this point's actual clipped CE scale, and uniform decay
one half. Prove the full closed path has unit parameters and its actual
constant-CE-gradient moment histories, at every completed clock.

This works at every valid native beta1/beta2, including 0.9/0.98; both
buffers evolve from zero and the clock grows, rather than being reset
or identified with the current gradient. The constant gradients are
derived from the current parameters and true CE callback, not supplied
as a successful future stream. The chosen epsilon is a mathematical
witness, not the preserved GPTMini experiment's hyperparameter.

This nonzero path establishes joint satisfiability of positive finite
parameter convergence and the actual retained update. It does not
establish attraction from the paper's one-zero-factor seeds, global
convergence, a grokking delay, physical circuit efficiency or any
stochastic/numerical-kernel bridge.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Candidate unit-parameter numerical history, to be identified with
the actual closed native path below. Source: native zero-initialized
moment recurrences at 9c2c42b; all beta, scale and clock arguments enter. -/
noncomputable def nativeUnitHistoryState (b1 b2 scale : ℝ) (clock : ℕ) : ScalarState :=
  { parameter := 1
    moment := firstMomentAt b1 (fun _ => -scale) clock
    variance := secondMomentAt b2 (fun _ => -scale) clock
    clock := clock }

/-- Actual CE callbacks agree when current parameters agree, even
when retained buffers and clocks differ. Sources: appendix C product
derivatives and native clipping at 9c2c42b; all four parameters enter. -/
theorem native_equal_parameters_applied_gradient (remaining : ℕ) (bound : ℝ)
    (state reference : NativeSubweightState)
    (hp : ∀ i, (state i).parameter = (reference i).parameter) :
    appliedNativeSubweightGradient remaining bound state = appliedNativeSubweightGradient remaining bound reference := by
  have hs : nativeTotalScore state = nativeTotalScore reference := by
    unfold nativeTotalScore
    rw [hp 0, hp 1, hp 2, hp 3]
  have hr : rawNativeSubweightGradient remaining state = rawNativeSubweightGradient remaining reference := by
    funext i
    rw [native_raw_gradient_partner, native_raw_gradient_partner, hp, hs]
  unfold appliedNativeSubweightGradient
  rw [hr]

example : ∀ i : Fin 4,
    ({ parameter := 1, moment := -1, variance := 1, clock := 37 } : ScalarState).parameter =
      (seededNativeSubweights ((1, 1), (1, 1)) i).parameter := by
  intro i
  fin_cases i <;> rfl

/-- The unit-history state's actual CE gradient is exactly the
unit-point gradient at every clock. Sources: appendix C's product
CE and shared native clipping at 9c2c42b; buffers do not enter CE. -/
theorem native_unit_history_gradient (remaining : ℕ) (bound b1 b2 : ℝ) (clock : ℕ) (i : Fin 4) :
    let scale := nativeCEGradientScale remaining bound (seededNativeSubweights ((1, 1), (1, 1)))
    appliedNativeSubweightGradient remaining bound (fun _ => nativeUnitHistoryState b1 b2 scale clock) i = -scale := by
  dsimp only
  rw [native_equal_parameters_applied_gradient remaining bound _ (seededNativeSubweights ((1, 1), (1, 1)))
    (by intro j; fin_cases j <;> rfl), native_applied_gradient_scale]
  have hp : ∀ j : Fin 4, (seededNativeSubweights ((1, 1), (1, 1)) j).parameter = 1 := by
    intro j
    fin_cases j <;> rfl
  rw [hp, mul_one]

/-- Actual native insertion advances the candidate full history and
keeps its unit parameters. Sources: appendix C actual CE and native
AdamW at 9c2c42b; derive the bias-corrected direction from both retained
moment histories, including the next completed clock and shared clip. -/
theorem native_unit_history_step (remaining : ℕ) (bound b1 b2 rate : ℝ) (clock : ℕ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) :
    let scale := nativeCEGradientScale remaining bound (seededNativeSubweights ((1, 1), (1, 1)))
    nativeSubweightStep remaining bound b1 b2 scale (1 / 2) rate
      (fun _ => nativeUnitHistoryState b1 b2 scale clock) =
      (fun _ => nativeUnitHistoryState b1 b2 scale (clock + 1)) := by
  dsimp only
  let scale := nativeCEGradientScale remaining bound (seededNativeSubweights ((1, 1), (1, 1)))
  have he : 0 < scale := native_ce_gradient_scale_pos remaining bound _ hclip
  have hd : nextBufferDirection b1 b2 scale (firstMomentAt b1 (fun _ => -scale) clock)
      (secondMomentAt b2 (fun _ => -scale) clock) (-scale) clock = -(1 / 2 : ℝ) := by
    rw [← history_direction_succ b1 b2 scale (fun _ => -scale),
      constant_gradient_direction b1 b2 scale (-scale) (clock + 1) hb1 h1 hb2 h2 (by omega),
      abs_of_neg (by linarith : -scale < 0), neg_neg]
    field_simp
    ring
  funext i
  change scalarNativeStep b1 b2 scale (1 / 2) rate (nativeUnitHistoryState b1 b2 scale clock)
    (appliedNativeSubweightGradient remaining bound (fun _ => nativeUnitHistoryState b1 b2 scale clock) i) = _
  rw [native_unit_history_gradient]
  unfold scalarNativeStep nativeUnitHistoryState
  dsimp only
  rw [hd]
  congr 1
  ring

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 := by norm_num

/-- The candidate history is exactly the actual closed CE path from
unit zero-buffer seeds. Sources: appendix C product CE and native
AdamW at 9c2c42b; induction checks the actual callback at every step,
without separately stipulating constant inputs or buffer matching. -/
theorem native_unit_path_history (remaining : ℕ) (bound b1 b2 rate : ℝ) (clock : ℕ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) :
    let scale := nativeCEGradientScale remaining bound (seededNativeSubweights ((1, 1), (1, 1)))
    nativeSubweightPath remaining bound b1 b2 scale (1 / 2) rate
      (seededNativeSubweights ((1, 1), (1, 1))) clock =
      (fun _ => nativeUnitHistoryState b1 b2 scale clock) := by
  dsimp only
  induction clock with
  | zero =>
    funext i
    fin_cases i <;> rfl
  | succ clock ih =>
    change nativeSubweightStep remaining bound b1 b2 _ (1 / 2) rate _ = _
    rw [ih]
    exact native_unit_history_step remaining bound b1 b2 rate clock hclip hb1 h1 hb2 h2

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 := by norm_num

/-- The actual nonzero retained path has a positive finite parameter
limit in each coordinate, with an unbounded clock and evolving buffers.
Sources: appendix C CE and native AdamW at 9c2c42b; no convergence
premise is supplied, and this witness uses its explicitly chosen epsilon. -/
theorem native_unit_path_parameter_tendsto (remaining : ℕ) (bound b1 b2 rate : ℝ) (i : Fin 4)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) :
    let scale := nativeCEGradientScale remaining bound (seededNativeSubweights ((1, 1), (1, 1)))
    Tendsto (fun n => (nativeSubweightPath remaining bound b1 b2 scale (1 / 2) rate
      (seededNativeSubweights ((1, 1), (1, 1))) n i).parameter) atTop (nhds 1) := by
  dsimp only
  apply Tendsto.congr (f₁ := fun _ : ℕ => (1 : ℝ)) _ tendsto_const_nhds
  intro n
  rw [native_unit_path_history remaining bound b1 b2 rate n hclip hb1 h1 hb2 h2]
  rfl

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
