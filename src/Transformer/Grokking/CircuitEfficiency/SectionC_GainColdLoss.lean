import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdCollapse
import Transformer.Grokking.CircuitEfficiency.SectionC_GainGradientLimits

/-!
# Actual output, loss and confidence limits at the strict cold threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3 decisions versus
confidence and appendix C true train/test CE; full retained output port
at 2c2a85b and sharp initialized parameter collapse at 1692df8.

Replace the earlier conservative first-clock condition with cold
coefficient times greater physical gain < decay times epsilon.
Actual physical parameter limits give zero Gen and Mem scores and
a vanishing held-out margin. Original train/held-out CE both approach
log(class count), and every full held-out softmax probability tends
to the uniform value. Every finite competing class remains in the sum.

No future parameter, score, CE, confidence or successful reference is
an independent hypothesis. Both buffers and actual completed clocks
are unchanged. The zero point only evaluates the numerical continuous
readout; it is not an assumed finite whole-state optimizer attractor.

These absolute limits are not finite-clock accuracy results. An exact
strict decision can be correct with a margin approaching zero, and its
selected class is not certified here for every strict-threshold path.
Critical equality and stable weak-decay selection remain separate.
Exact-real fixed gained tables/plain CE/uniform native decay differ
from appendix C's assigned coupled norm-cost GD. Learned stochastic/
numerical GPTMini and thermodynamic size transfer remain unproved;
no optimizer, experiment or checkpoint is modified.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss
open scoped BigOperators

/-- Both actual physical circuit scores and their held-out margin
vanish above the sharp strict cold threshold. Sources: appendix C
product logits and generated native parameter limits at 1692df8;
no future score or Gen dominance is supplied as a premise. -/
theorem gain_native_strict_cold_scores_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter -
        physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) := by
  have hp := gain_native_strict_cold_parameters_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap
  have hg : Tendsto (fun n => physicalCircuitScore genGain
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 0).parameter
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 1).parameter) atTop (nhds 0) := by
    simpa only [physicalCircuitScore, mul_zero] using ((hp 0).mul (hp 1)).const_mul genGain
  have hm : Tendsto (fun n => physicalCircuitScore memGain
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 2).parameter
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 3).parameter) atTop (nhds 0) := by
    simpa only [physicalCircuitScore, mul_zero] using ((hp 2).mul (hp 3)).const_mul memGain
  exact ⟨hg, hm, by simpa only [sub_zero] using hg.sub hm⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Original train and held-out CE approach uniform-class loss above
the sharp strict cold threshold. Sources: appendix C actual CE and
initialized native parameter limits at 1692df8; neither loss nor a
successful decision is inserted in the recurrence or its hypotheses. -/
theorem gain_native_strict_cold_ce_tendsto_uniform (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    Tendsto (fun n => tableTrainCE remaining (gainNativeTotalScore genGain memGain (path n)))
      atTop (nhds (Real.log ((remaining : ℝ) + 2))) ∧
      Tendsto (fun n => crossEntropy (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0)
        atTop (nhds (Real.log ((remaining : ℝ) + 2))) := by
  have hp := gain_native_strict_cold_parameters_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap
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

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Every full held-out softmax class probability approaches uniform
confidence above the sharp strict cold threshold. Sources: appendix C
complete exponential sum and native parameter limits at 1692df8;
the true class, wrong Mem class and all remaining classes are retained. -/
theorem gain_native_strict_cold_probabilities_tendsto_uniform (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    let logits := fun n => heldoutTableLogits remaining
      (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
      (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)
    ∀ k, Tendsto (fun n => Real.exp (logits n k) / ∑ j, Real.exp (logits n j))
      atTop (nhds (1 / ((remaining : ℝ) + 2))) := by
  have hmargins := gain_native_strict_cold_scores_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap
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

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

end Transformer.Grokking.CircuitEfficiency

