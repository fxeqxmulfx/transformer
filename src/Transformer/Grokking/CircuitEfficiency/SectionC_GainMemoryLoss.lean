import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryCollapse
import Transformer.Grokking.CircuitEfficiency.SectionC_GainGradientLimits

/-!
# Actual loss and confidence limits with retained nonzero-beta memory

Sources: Varma et al., arXiv:2309.02390v1, section 3 decisions versus
confidence and appendix C true product CE; initialized native retained
contraction at lab commit 1880c50, continuity at 9970d92 and zero-beta
loss derivation at a47cc78, extended here to legal retained betas.

Static strong decay generates zero physical parameter limits on the
original native path, with both buffers and full correction clocks.
Derive actual gained score/margin limits, full train and held-out CE
limits log(remaining+2), and every actual held-out softmax class
probability limit 1/(remaining+2). All finite classes enter the sum.
No future convergence, score, confidence or successful reference is
supplied. The zero reference is a numerical point for continuous CE,
not a finite optimizer-state limit: actual completed clocks diverge.

These absolute limits can be combined with native permanent decision
selection and arbitrary delay on the same configuration. They do not
assert positive limiting confidence or stable numerical correctness.
The sufficient strong-decay bound may be conservative. Fixed gained
tables, exact reals and uniform decoupled decay differ from appendix C
coupled-cost GD and from learned stochastic/floating-point GPTMini.
The actual trained transformer transfer remains open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss
open scoped BigOperators

/-- Both original physical scores and their actual held-out Gen/Mem
margin vanish under static strong decay. Sources: appendix C product
logits and initialized native limits at 1880c50; convergence is derived
from initial retained signs rather than a future reference premise. -/
theorem gain_native_memory_scores_tendsto_zero (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter -
        physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) := by
  have hp := gain_native_memory_parameters_tendsto_zero remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  have hg : Tendsto (fun n => physicalCircuitScore genGain
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 0).parameter
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 1).parameter) atTop (nhds 0) := by
    simpa only [physicalCircuitScore, mul_zero] using ((hp 0).mul (hp 1)).const_mul genGain
  have hm : Tendsto (fun n => physicalCircuitScore memGain
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 2).parameter
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 3).parameter) atTop (nhds 0) := by
    simpa only [physicalCircuitScore, mul_zero] using ((hp 2).mul (hp 3)).const_mul memGain
  exact ⟨hg, hm, by simpa only [sub_zero] using hg.sub hm⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- Original train and held-out CE both approach uniform-class loss
under initialized strong native decay. Sources: appendix C true CE
formulas and native convergence at 1880c50; a finite parameter limit
is derived, and neither loss is replaced by a correctness indicator. -/
theorem gain_native_memory_ce_tendsto_uniform (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    Tendsto (fun n => tableTrainCE remaining (gainNativeTotalScore genGain memGain (path n)))
      atTop (nhds (Real.log ((remaining : ℝ) + 2))) ∧
      Tendsto (fun n => crossEntropy (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0)
        atTop (nhds (Real.log ((remaining : ℝ) + 2))) := by
  have hp := gain_native_memory_parameters_tendsto_zero remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  let reference := seededNativeSubweights ((0, 0), (0, 0))
  have hr : ∀ i, (reference i).parameter = 0 := by
    intro i
    fin_cases i <;> rfl
  have hpr : ∀ i, Tendsto (fun n =>
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
    intro i
    rw [hr i]
    exact hp i
  have hg : physicalCircuitScore genGain (reference 0).parameter (reference 1).parameter = 0 := by
    rw [hr 0, hr 1, physicalCircuitScore, mul_zero, mul_zero]
  have hm : physicalCircuitScore memGain (reference 2).parameter (reference 3).parameter = 0 := by
    rw [hr 2, hr 3, physicalCircuitScore, mul_zero, mul_zero]
  have ht := gain_native_train_ce_tendsto remaining genGain memGain _ reference hpr
  have hh := gain_native_heldout_ce_tendsto remaining genGain memGain _ reference hpr
  have htwo : (1 : ℝ) + 1 = 2 := by norm_num
  constructor
  · simpa only [gainNativeTotalScore, hg, hm, zero_add, table_train_ce_initial] using ht
  · simpa only [hg, hm, table_heldout_ce_formula, Real.exp_zero, sub_zero,
      add_comm, add_left_comm, add_assoc, htwo] using hh

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- Every actual held-out class softmax probability tends to its uniform
value under static strong native decay. Sources: appendix C full
exponential denominator and initialized collapse at 1880c50; the sum
ranges over every class, including the wrong Mem and remaining classes. -/
theorem gain_native_memory_probabilities_tendsto_uniform (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    let logits := fun n => heldoutTableLogits remaining
      (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
      (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)
    ∀ k, Tendsto (fun n => Real.exp (logits n k) / ∑ j, Real.exp (logits n j))
      atTop (nhds (1 / ((remaining : ℝ) + 2))) := by
  have hmargins := gain_native_memory_scores_tendsto_zero remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  let logits := fun n => heldoutTableLogits remaining
    (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
    (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)
  have hden : Tendsto (fun n => ∑ j, Real.exp (logits n j)) atTop (nhds ((remaining : ℝ) + 2)) := by
    convert (hmargins.1.rexp.add hmargins.2.1.rexp).add_const (remaining : ℝ) using 1
    · funext n
      exact heldout_table_exp_sum remaining _ _
    · simp only [Real.exp_zero]
      congr 1
      ring
  dsimp only
  intro k
  have hlogit : Tendsto (fun n => logits n k) atTop (nhds 0) := by
    by_cases hk : k = 0
    · simpa only [logits, heldoutTableLogits, hk, ite_true] using hmargins.1
    · by_cases hm : k = (0 : Fin (remaining + 1)).succ
      · simpa only [logits, heldoutTableLogits, ite_eq_right hk, ite_eq_left hm] using hmargins.2.1
      · simp only [logits, heldoutTableLogits, ite_eq_right hk, ite_eq_right hm]
        exact tendsto_const_nhds
  simpa only [Pi.div_def, Real.exp_zero] using hlogit.rexp.div hden (by positivity)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

end Transformer.Grokking.CircuitEfficiency
