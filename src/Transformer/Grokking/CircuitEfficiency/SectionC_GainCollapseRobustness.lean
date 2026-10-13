import Transformer.Grokking.CircuitEfficiency.SectionC_GainAccuracyCollapse

/-!
# Delayed native accuracy without a positive perturbation radius

Sources: Varma et al., arXiv:2309.02390v1, section 3 confidence versus
decisions and appendix C's actual held-out table; derived native
accuracy/collapse paths at lab commit cb1e286.

First construct a bounded perturbation of the actual multiclass logits:
subtract one positive radius from the target coordinate only. If the
Gen-minus-Mem margin is less than that radius, the wrong Mem coordinate
beats the perturbed target. Every coordinate error is bounded by the
same radius, with no replacement of the decision by a binary surrogate.

Initialized strong decay makes the actual margin tend to zero, so every
fixed positive radius eventually admits such a perturbation at every
later clock. Combine this with arbitrary delayed permanent accuracy on
the original retained path. The same fixed configuration can be perfectly
correct on every sufficiently late exact-real test decision and lack
any fixed positive tail robustness radius.

The perturbation is a mathematical control, not an intervention on
training and not a claim that a floating-point kernel generates it.
This does not prove a numerical failure or kernel-error magnitude.
Zero betas, fixed physical tables and native decoupled decay differ
from appendix C's coupled-cost GD and learned nonzero-beta GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.NaiveLoss

/-- A target-only logit perturbation bounded by the positive radius
breaks strict target correctness whenever the actual Gen/Mem gap is
smaller. Sources: appendix C multiclass table and the confidence/decision
separation at cb1e286; the competing Mem class is an actual class. -/
theorem heldout_table_target_perturbation (remaining : ℕ) (x y radius : ℝ)
    (hr : 0 < radius) (hgap : x - y < radius) :
    ∃ perturbed : Fin (remaining + 2) → ℝ,
      (∀ k, |perturbed k - heldoutTableLogits remaining x y k| ≤ radius) ∧ ¬StrictCorrect perturbed 0 := by
  let perturbed := fun k => heldoutTableLogits remaining x y k - if k = 0 then radius else 0
  refine ⟨perturbed, ?_, ?_⟩
  · intro k
    have heq : perturbed k - heldoutTableLogits remaining x y k = -(if k = 0 then radius else 0) := by
      dsimp only [perturbed]
      ring
    rw [heq]
    by_cases hk : k = 0
    · simp only [ite_eq_left hk, abs_neg, abs_of_pos hr, le_refl]
    · simp only [ite_eq_right hk, neg_zero, abs_zero]
      exact le_of_lt hr
  · intro hcorrect
    have hn := Fin.succ_ne_zero (0 : Fin (remaining + 1))
    have hm : perturbed (0 : Fin (remaining + 1)).succ = y := by
      simp only [perturbed, heldoutTableLogits, ite_eq_right hn, ite_true, sub_zero]
    have ht : perturbed 0 = x - radius := by
      simp only [perturbed, heldoutTableLogits, ite_true]
    have hwrong := hcorrect (0 : Fin (remaining + 1)).succ hn
    rw [hm, ht] at hwrong
    linarith only [hgap, hwrong]

example : (0 : ℝ) < 1 ∧ (1 / 10 : ℝ) - 0 < 1 := by norm_num

/-- Each fixed positive perturbation radius eventually defeats actual
held-out strict correctness under initialized strong native decay.
Sources: appendix C table and margin collapse at cb1e286; every path
limit and perturbation is generated from static data, without a future
successful-margin, numerical-error or limiting-reference premise. -/
theorem gain_native_strong_decay_eventually_perturbable (remaining : ℕ)
    (genGain memGain bound eps decay rate radius : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : genGain / eps < decay)
    (hs : NonnegativeNativeState initial) (hr : 0 < radius) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    ∃ start : ℕ, ∀ n, start ≤ n → ∃ perturbed : Fin (remaining + 2) → ℝ,
      (∀ k, |perturbed k - heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) k| ≤ radius) ∧
      ¬StrictCorrect perturbed 0 := by
  have hm := gain_native_strong_decay_scores_tendsto_zero remaining genGain memGain bound eps decay rate initial
    hgen hmem hgain hclip he heta hd hstrong hs
  obtain ⟨start, htail⟩ := eventually_atTop.mp (hm.2.2.eventually_lt_const hr)
  refine ⟨start, ?_⟩
  intro n hn
  exact heldout_table_target_perturbation remaining _ _ radius hr (htail n hn)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧ (3 : ℝ) / 1 < 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧ (0 : ℝ) < 1 / 100 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_, by norm_num⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The actual held-out Gen/Mem margin has no fixed positive lower
bound on any tail under initialized strong native decay. Sources:
appendix C table and collapse at cb1e286; this is a quantitative
obstruction distinct from the finite-clock strict decision property. -/
theorem gain_native_strong_decay_no_positive_margin_tail (remaining : ℕ)
    (genGain memGain bound eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : genGain / eps < decay)
    (hs : NonnegativeNativeState initial) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    ¬∃ radius : ℝ, 0 < radius ∧ ∃ start : ℕ, ∀ n, start ≤ n →
      radius ≤ physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter -
        physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter := by
  dsimp only
  intro hex
  obtain ⟨radius, hr, start, htail⟩ := hex
  have hm := gain_native_strong_decay_scores_tendsto_zero remaining genGain memGain bound eps decay rate initial
    hgen hmem hgain hclip he heta hd hstrong hs
  obtain ⟨after, hlt⟩ := eventually_atTop.mp (hm.2.2.eventually_lt_const hr)
  have hbelow := hlt (start + after) (by omega)
  have habove := htail (start + after) (by omega)
  linarith only [hbelow, habove]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧ (3 : ℝ) / 1 < 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The same pinned original path can have any finite wrong-test
prefix, permanent later strict accuracy, and an eventual bounded
counterperturbation for every positive radius. Sources: section 3
accuracy and appendix C true logits, extended to native collapse at
cb1e286; perturbations certify lack of a uniform positive radius. -/
theorem fixed_native_delayed_accuracy_without_tail_robustness (remaining : ℕ) :
    ∀ budget : ℕ, ∃ seed : ℝ, 0 < seed ∧
      let path := gainNativePath remaining 3 2 1 0 0 1 10 (1 / 1000)
        (seededNativeSubweights ((0, seed), (1, 1)))
      let logits := fun n => heldoutTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)
      (∀ n, StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (logits n) (0 : Fin (remaining + 1)).succ) ∧
      ∃ start : ℕ, budget < start ∧ (∀ n, start ≤ n → StrictCorrect (logits n) 0) ∧
        ∀ radius : ℝ, 0 < radius → ∃ after : ℕ, start ≤ after ∧
          ∀ n, after ≤ n → ∃ perturbed : Fin (remaining + 2) → ℝ,
            (∀ k, |perturbed k - logits n k| ≤ radius) ∧ ¬StrictCorrect perturbed 0 := by
  intro budget
  obtain ⟨seed, _, hseed, _, _, htrain, _, hprefix, ⟨start, hlate, hcorrect⟩, _, _, _, _, _⟩ :=
    gain_native_delayed_accuracy_with_collapse remaining 3 2 1 1 10 (1 / 1000) 1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) budget
  refine ⟨seed, hseed, htrain, hprefix, start, hlate, hcorrect, ?_⟩
  intro radius hr
  have hs := native_seeded_nonnegative 0 seed 1 1
    le_rfl (le_of_lt hseed) (by norm_num) (by norm_num)
  obtain ⟨after, htail⟩ := gain_native_strong_decay_eventually_perturbable remaining 3 2 1 1 10 (1 / 1000) radius _
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) hs hr
  exact ⟨after + start, by omega, fun n hn => htail n (by omega)⟩

end Transformer.Grokking.CircuitEfficiency
