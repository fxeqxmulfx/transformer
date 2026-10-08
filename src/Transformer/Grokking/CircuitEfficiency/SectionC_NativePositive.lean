import Transformer.Grokking.CircuitEfficiency.SectionC_NativeSigns

/-!
# Positive circuit formation on the closed native CE trajectory

Sources: Varma et al., arXiv:2309.02390v1, appendix C, one-zero-factor
seeds and product logits; native AdamW/clipping at lab commit 793b191.
Positive second-factor seeds activate both initially zero first factors
through their actual negative CE derivatives. All four parameters then
remain strictly positive at every finite subsequent iteration, with
retained moments, shared clipping and positive remaining decay factor.

Deviation: this is plain fixed-table CE and parameter-wise native decay,
not the paper's GD/coupled circuit-norm penalty. Positivity gives strict
training correctness but does not select Gen over Mem, cross a held-out
margin, identify learned GPTMini features or certify floating-point code.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Strict positivity of every actual factor, with numerical retained
moment and variance signs. Source: native AdamW at 793b191. -/
def PositiveNativeState (state : NativeSubweightState) : Prop :=
  ∀ i, PositiveScalarState (state i)

/-- Once formed, positive factors persist under their actual CE
feedback. Sources: appendix C product logits and native AdamW at
793b191. No positive lower bound independent of time is claimed. -/
theorem native_positive_step (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 < 1 - rate * decay)
    (hs : PositiveNativeState state) :
    PositiveNativeState (nativeSubweightStep remaining bound b1 b2 eps decay rate state) := by
  intro i
  exact scalar_positive_step b1 b2 eps decay rate _ (state i) hb1 h1 hb2 (le_of_lt h2) he heta hd (hs i)
    (native_applied_gradient_nonpos remaining bound state i hclip (le_of_lt (hs _).1))

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    PositiveNativeState (seededNativeSubweights ((1, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> exact seeded_scalar_positive _ (by norm_num)

/-- A positive factor survives; a zero factor with a positive partner
is activated by its actual current derivative. Sources: appendix C
product CE and native AdamW at 793b191. Retained old first moments
may already be nonzero; no reset is used in either case. -/
theorem native_step_positive_coordinate (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState state)
    (hi : 0 < (state i).parameter ∨ 0 < (state (nativeFactorPartner i)).parameter) :
    PositiveScalarState (nativeSubweightStep remaining bound b1 b2 eps decay rate state i) := by
  rcases hi with hp | hp
  · exact scalar_positive_step b1 b2 eps decay rate _ (state i) hb1 h1 hb2 (le_of_lt h2) he (le_of_lt heta)
      hd ⟨hp, (hs i).2⟩ (native_applied_gradient_nonpos remaining bound state i hclip (hs _).1)
  · exact scalar_negative_gradient_activates b1 b2 eps decay rate _ (state i) hb1 h1 hb2 (le_of_lt h2)
      he heta (le_of_lt hd) (hs i) (native_applied_gradient_negative remaining bound state i hclip hp)

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 < (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    (0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 0).parameter ∨
      0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num)
      (by norm_num) (by norm_num), ?_⟩
  exact Or.inr (by norm_num [seededNativeSubweights, nativeFactorPartner, seededScalarState])

/-- Both initially absent first factors actually emerge after one
step from positive second seeds. Sources: appendix C simulation
initialization and native AdamW at 793b191. All four buffers remain
valid; the shared gradient may be clipped and moments are retained. -/
theorem native_one_zero_seed_first_positive (remaining : ℕ)
    (bound b1 b2 eps decay rate genSeed memSeed : ℝ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hg : 0 < genSeed) (hm : 0 < memSeed) :
    PositiveNativeState (nativeSubweightStep remaining bound b1 b2 eps decay rate
      (seededNativeSubweights ((0, genSeed), (0, memSeed)))) := by
  have hs := native_seeded_nonnegative 0 genSeed 0 memSeed le_rfl (le_of_lt hg) le_rfl (le_of_lt hm)
  intro i
  apply native_step_positive_coordinate remaining bound b1 b2 eps decay rate _ i hclip
    hb1 h1 hb2 h2 he heta hd hs
  fin_cases i
  · exact Or.inr hg
  · exact Or.inl hg
  · exact Or.inr hm
  · exact Or.inl hm

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 < (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

/-- Every finite step after the first has both positive products on
the actual closed CE trajectory. Sources: appendix C's positive
second seeds and native AdamW at 793b191. This proves formation and
noncollapse, independently of whether a generalizing margin grows. -/
theorem native_one_zero_seed_path_positive (remaining : ℕ)
    (bound b1 b2 eps decay rate genSeed memSeed : ℝ) (n : ℕ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hg : 0 < genSeed) (hm : 0 < memSeed) :
    PositiveNativeState (nativeSubweightPath remaining bound b1 b2 eps decay rate
      (seededNativeSubweights ((0, genSeed), (0, memSeed))) (n + 1)) := by
  induction n with
  | zero =>
    exact native_one_zero_seed_first_positive remaining bound b1 b2 eps decay rate
      genSeed memSeed hclip hb1 h1 hb2 h2 he heta hd hg hm
  | succ n ih =>
    exact native_positive_step remaining bound b1 b2 eps decay rate _ hclip hb1 h1 hb2 h2
      he (le_of_lt heta) hd ih

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 < (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

/-- Formed positive products give strict correctness against all
training classes. Source: appendix C's actual multiclass training
logits. Held-out success requires a comparison of the two products;
this consequence deliberately makes no such conclusion. -/
theorem native_positive_train_correct (remaining : ℕ) (state : NativeSubweightState)
    (hs : PositiveNativeState state) :
    Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining
      ((state 0).parameter * (state 1).parameter)
      ((state 2).parameter * (state 3).parameter)) 0 := by
  apply train_table_strict_correct_iff _ _ _ |>.mpr
  have hg := mul_pos (hs 0).1 (hs 1).1
  have hm := mul_pos (hs 2).1 (hs 3).1
  linarith

example : PositiveNativeState (seededNativeSubweights ((1, 1 / 200), (1, 1))) := by
  intro i
  fin_cases i <;> exact seeded_scalar_positive _ (by norm_num)

/-- Actual training correctness holds at every positive update clock
from positive source seeds. Sources: appendix C's multiclass tables
and the retained native CE iteration at 793b191. This is a decision
statement; arbitrarily low training CE and held-out success are separate. -/
theorem native_one_zero_seed_path_train_correct (remaining : ℕ)
    (bound b1 b2 eps decay rate genSeed memSeed : ℝ) (n : ℕ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hg : 0 < genSeed) (hm : 0 < memSeed) :
    let state := nativeSubweightPath remaining bound b1 b2 eps decay rate
      (seededNativeSubweights ((0, genSeed), (0, memSeed))) (n + 1)
    Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining
      ((state 0).parameter * (state 1).parameter)
      ((state 2).parameter * (state 3).parameter)) 0 := by
  exact native_positive_train_correct remaining _
    (native_one_zero_seed_path_positive remaining bound b1 b2 eps decay rate genSeed memSeed n
      hclip hb1 h1 hb2 h2 he heta hd hg hm)

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 < (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
