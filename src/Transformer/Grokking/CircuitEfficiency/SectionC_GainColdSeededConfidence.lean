import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdSeededReadouts

/-!
# Full initialized confidence and CE limits at critical equality

Sources: Varma et al., arXiv:2309.02390v1, section 3 decisions
versus confidence and appendix C true finite-class CE; actual
critical collapse at cb3a0d6 and physical readouts at 63f857e.

At or above the cold threshold, original training and held-out CE
approach log(class count). Every full held-out softmax probability
approaches the uniform value, and the true-versus-Mem probability
gap tends to zero. The true class, wrong Mem class and every other
finite class remain in the original exponential denominator.

All limits include exact critical equality and legal retained betas
with standard nonnegative zero-buffer seeds. Positive rate/epsilon,
positive gains/cap and nonnegative remaining decay are explicit.
No future parameter, score, CE, probability or successful-reference
limit is an independent hypothesis. The actual CE callback generates
the path; the zero point only evaluates the continuous readout.

These are absolute confidence limits, not finite-clock accuracy
statements. Strict exact decisions can persist while the limiting
probabilities tie. Neither zero confidence gap nor uniform limiting
CE determines which class dominates at every finite clock. This
distinction is necessary for comparing norms and confidence with
observed held-out decision changes during delayed generalization.

Exact-real fixed gained tables/plain CE/uniform decoupled AdamW
differ from appendix C's assigned coupled norm-cost GD. Stable
weak-decay rule selection, learned stochastic/numerical GPTMini and
thermodynamic system-size transfer remain separate. Frozen model,
optimizer, measurements, runs and checkpoints are unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss
open scoped BigOperators

/-- Actual train and held-out CE approach uniform-class loss,
including critical equality. Sources: appendix C complete losses,
initialized collapse at cb3a0d6 and physical readouts at 63f857e;
neither loss convergence nor a successful decision is a premise. -/
theorem gain_native_seeded_cold_ce_tendsto_uniform (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
    Tendsto (fun n => tableTrainCE remaining (gainNativeTotalScore genGain memGain (path n)))
      atTop (nhds (Real.log ((remaining : ℝ) + 2))) ∧
      Tendsto (fun n => crossEntropy (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0)
        atTop (nhds (Real.log ((remaining : ℝ) + 2))) := by
  have hp := gain_native_seeded_cold_parameters_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  let reference := seededNativeSubweights ((0, 0), (0, 0))
  have hr : ∀ i, (reference i).parameter = 0 := by
    intro i
    fin_cases i <;> rfl
  have hpr : ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n i).parameter) atTop (nhds (reference i).parameter) := by
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
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- Every full held-out softmax probability approaches uniform
confidence, including critical equality. Sources: appendix C's true
exponential sum and generated readouts at 63f857e; all finite classes
remain in the denominator, without an assumed probability limit. -/
theorem gain_native_seeded_cold_probabilities_tendsto_uniform (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
    let logits := fun n => heldoutTableLogits remaining
      (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
      (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)
    ∀ k, Tendsto (fun n => Real.exp (logits n k) / ∑ j, Real.exp (logits n j))
      atTop (nhds (1 / ((remaining : ℝ) + 2))) := by
  have hscores := gain_native_seeded_cold_scores_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
  let logits := fun n => heldoutTableLogits remaining
    (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
    (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)
  have hden : Tendsto (fun n => ∑ j, Real.exp (logits n j)) atTop (nhds ((remaining : ℝ) + 2)) := by
    convert (hscores.1.rexp.add hscores.2.1.rexp).add_const (remaining : ℝ) using 1
    · funext n
      exact heldout_table_exp_sum remaining _ _
    · simp only [Real.exp_zero]
      congr 1
      ring
  dsimp only
  intro k
  have hlogit : Tendsto (fun n => logits n k) atTop (nhds 0) := by
    by_cases hk : k = 0
    · simpa only [logits, heldoutTableLogits, hk, ite_true] using hscores.1
    · by_cases hm : k = (0 : Fin (remaining + 1)).succ
      · simpa only [logits, heldoutTableLogits, ite_eq_right hk, ite_eq_left hm] using hscores.2.1
      · simp only [logits, heldoutTableLogits, ite_eq_right hk, ite_eq_right hm]
        exact tendsto_const_nhds
  simpa only [Pi.div_def, Real.exp_zero] using hlogit.rexp.div hden (by positivity)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- The true-versus-Mem softmax probability gap vanishes, including
critical equality. Sources: appendix C's distinct test classes and
the complete generated uniform limit above; this does not declare
which exact finite-clock decision wins when the limit ties. -/
theorem gain_native_seeded_cold_probability_gap_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
    let logits := fun n => heldoutTableLogits remaining
      (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
      (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)
    Tendsto (fun n => Real.exp (logits n 0) / (∑ j, Real.exp (logits n j)) -
      Real.exp (logits n (0 : Fin (remaining + 1)).succ) / (∑ j, Real.exp (logits n j))) atTop (nhds 0) := by
  have hp := gain_native_seeded_cold_probabilities_tendsto_uniform remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  simpa only [sub_self] using (hp 0).sub (hp (0 : Fin (remaining + 1)).succ)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

end Transformer.Grokking.CircuitEfficiency
