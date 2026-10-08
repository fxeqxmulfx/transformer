import Transformer.Grokking.CircuitEfficiency.SectionC_GainStrongDecayPath
import Transformer.Grokking.CircuitEfficiency.SectionC_GainGradientLimits

/-!
# Actual strong-decay logit, loss and softmax confidence limits

Sources: Varma et al., arXiv:2309.02390v1, section 3's distinction
between decisions and confidence, appendix C's product train/test CE;
initialized native collapse at lab commit a47cc78.

Strong static decay generates zero limits of all actual physical
parameters. Derive both circuit score limits, the Gen-minus-Mem margin
limit, and the original train and held-out CE limits log(remaining+2).
Every actual held-out softmax class probability approaches 1/(remaining+2),
with its denominator the full finite-class exponential sum.

No parameter limit, successful reference, future score, confidence,
clipping stream or loss limit is supplied as an independent premise.
The all-zero reference used in the existing continuity theorem is only
a numerical point; its buffers and clock are not path-state limits.
The class count and every competing class remain in the actual CE.

These absolute limits do not yet assert task success. Strong-decay
accuracy selection must be proved on the same initialized path before
using them as a counterexample to accuracy implying confidence or
vanishing loss. Exact-real decisions can persist under vanishing margins.

Zero betas, fixed physical tables and uniform decoupled native decay
differ from appendix C's coupled norm/GD and the preserved learned
nonzero-beta GPTMini. No numerical-kernel transfer is claimed.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss
open scoped BigOperators

/-- Both original physical scores and their actual held-out Gen/Mem
margin vanish under static strong decay. Sources: appendix C product
logits and initialized native limits at a47cc78; convergence is derived
from initial retained signs rather than a future reference premise. -/
theorem gain_native_strong_decay_scores_tendsto_zero (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : genGain / eps < decay) (hs : NonnegativeNativeState initial) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter -
        physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) := by
  have hp := gain_native_strong_decay_parameters_tendsto_zero remaining genGain memGain bound eps decay rate initial
    hgen hmem hgain hclip he heta hd hstrong hs
  have hg : Tendsto (fun n => physicalCircuitScore genGain
      (gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial n 0).parameter
      (gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial n 1).parameter) atTop (nhds 0) := by
    simpa only [physicalCircuitScore, mul_zero] using ((hp 0).mul (hp 1)).const_mul genGain
  have hm : Tendsto (fun n => physicalCircuitScore memGain
      (gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial n 2).parameter
      (gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial n 3).parameter) atTop (nhds 0) := by
    simpa only [physicalCircuitScore, mul_zero] using ((hp 2).mul (hp 3)).const_mul memGain
  exact ⟨hg, hm, by simpa only [sub_zero] using hg.sub hm⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧ (3 : ℝ) / 1 < 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Original train and held-out CE both approach uniform-class loss
under initialized strong native decay. Sources: appendix C true CE
formulas and native convergence at a47cc78; a finite parameter limit
is derived, and neither loss is replaced by a correctness indicator. -/
theorem gain_native_strong_decay_ce_tendsto_uniform (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : genGain / eps < decay) (hs : NonnegativeNativeState initial) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    Tendsto (fun n => tableTrainCE remaining (gainNativeTotalScore genGain memGain (path n)))
      atTop (nhds (Real.log ((remaining : ℝ) + 2))) ∧
      Tendsto (fun n => crossEntropy (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0)
        atTop (nhds (Real.log ((remaining : ℝ) + 2))) := by
  have hp := gain_native_strong_decay_parameters_tendsto_zero remaining genGain memGain bound eps decay rate initial
    hgen hmem hgain hclip he heta hd hstrong hs
  let reference := seededNativeSubweights ((0, 0), (0, 0))
  have hr : ∀ i, (reference i).parameter = 0 := by
    intro i
    fin_cases i <;> rfl
  have hpr : ∀ i, Tendsto (fun n =>
      (gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial n i).parameter)
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

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧ (3 : ℝ) / 1 < 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Every actual held-out class softmax probability tends to its uniform
value under static strong native decay. Sources: appendix C full
exponential denominator and initialized collapse at a47cc78; the sum
ranges over every class, including the wrong Mem and remaining classes. -/
theorem gain_native_strong_decay_probabilities_tendsto_uniform (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : genGain / eps < decay) (hs : NonnegativeNativeState initial) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    let logits := fun n => heldoutTableLogits remaining
      (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
      (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)
    ∀ k, Tendsto (fun n => Real.exp (logits n k) / ∑ j, Real.exp (logits n j))
      atTop (nhds (1 / ((remaining : ℝ) + 2))) := by
  have hmargins := gain_native_strong_decay_scores_tendsto_zero remaining genGain memGain bound eps decay rate initial
    hgen hmem hgain hclip he heta hd hstrong hs
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
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

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧ (3 : ℝ) / 1 < 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.CircuitEfficiency
