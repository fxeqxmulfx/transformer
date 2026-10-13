import Transformer.Grokking.CircuitEfficiency.SectionC_GainStrongDecayLoss
import Transformer.Grokking.CircuitEfficiency.SectionC_GainUnequalDelay

/-!
# Arbitrarily delayed accuracy success with vanishing native confidence

Sources: Varma et al., arXiv:2309.02390v1, section 3's grokking
ingredients and confidence/decay competition, appendix C's product
train/test tables; actual initialized native selection at 4e9829f
and absolute strong-decay collapse at a47cc78.

One fixed task and native configuration admit every finite wrong-test
prefix with perfect training decisions. A positive first-zero Gen
partner seed varies with the budget; already train-fitted Mem starts
with equal positive factors. The same bounded initialized retained
path permanently becomes strictly correct on held-out classes.

At the same time, all four parameters and the held-out Gen/Mem margin
tend to zero, actual train and held-out CE tend to log(remaining+2),
and every held-out softmax class probability tends to 1/(remaining+2).
No limiting weights, future success, margin or feedback stream are
supplied. Static strong decay and the sufficient small rate are compatible.

The pinned gains are 3 and 2, clipping cap and epsilon are 1,
decay is 10 and rate is 1/1000. Remaining decay is 99/100;
the larger pair-mass ceiling is 993/1000. The sufficient small-rate
quantity is 13/1000, so the actual relative selection theorem applies.
All these numerical constants, the task and Mem initialization stay
fixed as the prescribed delay budget grows.

After the generated success start, each finite decision has a strictly
positive margin. Its zero limit allows no uniform positive tail margin.
The loss limit is positive for every admissible finite class count.

The pinned configuration provides a formal counterexample to delayed
permanent strict accuracy implying a zero held-out CE limit. This
separates exact decision selection from confidence improvement; it
is not a counterexample to the source's coupled-cost GD dynamics.

Zero betas, exact reals, fixed tables and uniform native decoupled
decay differ from learned nonzero-beta GPTMini and floating-point
kernels. Seed/margin scales can leave machine range. The theorem
makes no claim of a measured grokking event on preserved GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss
open scoped BigOperators

/-- The same actual initialized native path has any prescribed
train-correct/wrong-test prefix, permanent later success, zero physical
limits and uniform confidence/loss limits. Sources: section 3 and
appendix C tables, native selection at 4e9829f and collapse at a47cc78;
only initial partner data vary, and every future property is derived. -/
theorem gain_native_delayed_accuracy_with_collapse (remaining : ℕ)
    (genGain memGain bound eps decay rate amplitude : ℝ)
    (hmem : 0 < memGain) (hbetter : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) (ha : 0 < amplitude)
    (hstrong : genGain / eps < decay) :
    ∀ budget : ℕ, ∃ seed ceiling : ℝ, 0 < seed ∧ 0 < ceiling ∧
      let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
        (seededNativeSubweights ((0, seed), (amplitude, amplitude)))
      (∀ n, NonnegativeNativeState (path n) ∧ ∀ i,
        (path n i).parameter ≤ ceiling ∧ -(path n i).moment ≤ bound ∧ (path n i).variance ≤ bound ^ 2) ∧
      (∀ n, StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ n, 0 < physicalCircuitScore genGain (path (n + 1) 0).parameter (path (n + 1) 1).parameter) ∧
      (∀ n, n ≤ budget → StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter))
          (0 : Fin (remaining + 1)).succ) ∧
      (∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n → StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ i, Tendsto (fun n => (path n i).parameter) atTop (nhds 0)) ∧
      Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter -
        physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => tableTrainCE remaining (gainNativeTotalScore genGain memGain (path n)))
        atTop (nhds (Real.log ((remaining : ℝ) + 2))) ∧
      Tendsto (fun n => crossEntropy (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0)
        atTop (nhds (Real.log ((remaining : ℝ) + 2))) ∧
      ∀ k, Tendsto (fun n => Real.exp (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) k) /
        ∑ j, Real.exp (heldoutTableLogits remaining
          (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
          (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) j))
        atTop (nhds (1 / ((remaining : ℝ) + 2))) := by
  intro budget
  have hgen := lt_trans hmem hbetter
  have hd := le_of_lt (gain_small_rate_decay_remaining_positive genGain eps decay rate hgen he heta hsmall)
  obtain ⟨seed, ceiling, hseed, hceiling, hbox, htrain, hpositive, hprefix, hsuccess⟩ :=
    gain_native_unequal_arbitrary_delayed_generalization remaining genGain memGain bound eps decay rate amplitude
      hmem hbetter hclip he hdecay heta hsmall ha budget
  have hs := native_seeded_nonnegative 0 seed amplitude amplitude
    le_rfl (le_of_lt hseed) (le_of_lt ha) (le_of_lt ha)
  have hp := gain_native_strong_decay_parameters_tendsto_zero remaining genGain memGain bound eps decay rate _
    hgen hmem (le_of_lt hbetter) hclip he heta hd hstrong hs
  have hm := gain_native_strong_decay_scores_tendsto_zero remaining genGain memGain bound eps decay rate _
    hgen hmem (le_of_lt hbetter) hclip he heta hd hstrong hs
  have hce := gain_native_strong_decay_ce_tendsto_uniform remaining genGain memGain bound eps decay rate _
    hgen hmem (le_of_lt hbetter) hclip he heta hd hstrong hs
  have hprob := gain_native_strong_decay_probabilities_tendsto_uniform remaining genGain memGain bound eps decay rate _
    hgen hmem (le_of_lt hbetter) hclip he heta hd hstrong hs
  exact ⟨seed, ceiling, hseed, hceiling, hbox, htrain, hpositive, hprefix, hsuccess, hp, hm.2.2, hce.1, hce.2, hprob⟩

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 10 ∧ (0 : ℝ) < 1 / 1000 ∧
    (1 / 1000 : ℝ) * (10 + 3 / 1) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (3 : ℝ) / 1 < 10 := by norm_num

/-- A pinned actual native configuration disproves zero held-out CE
convergence from arbitrarily delayed permanent strict accuracy. Sources:
section 3 confidence competition and appendix C CE, extended to native
strong decay at a47cc78; all four physical weights still tend to zero. -/
theorem fixed_native_delayed_accuracy_without_zero_ce (remaining : ℕ) :
    ∀ budget : ℕ, ∃ seed : ℝ, 0 < seed ∧
      let path := gainNativePath remaining 3 2 1 0 0 1 10 (1 / 1000)
        (seededNativeSubweights ((0, seed), (1, 1)))
      (∀ n, StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter))
          (0 : Fin (remaining + 1)).succ) ∧
      (∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n → StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ i, Tendsto (fun n => (path n i).parameter) atTop (nhds 0)) ∧
      ¬Tendsto (fun n => crossEntropy (heldoutTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0) atTop (nhds 0) := by
  intro budget
  obtain ⟨seed, _, hseed, _, _, htrain, _, hprefix, hsuccess, hp, _, _, hce, _⟩ :=
    gain_native_delayed_accuracy_with_collapse remaining 3 2 1 1 10 (1 / 1000) 1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) budget
  refine ⟨seed, hseed, htrain, hprefix, hsuccess, hp, ?_⟩
  intro hzero
  have hr : 0 ≤ (remaining : ℝ) := by positivity
  have hclasses : 1 < (remaining : ℝ) + 2 := by linarith only [hr]
  have hloss : 0 < Real.log ((remaining : ℝ) + 2) := Real.log_pos hclasses
  have heq := tendsto_nhds_unique hce hzero
  linarith only [hloss, heq]

end Transformer.Grokking.CircuitEfficiency
