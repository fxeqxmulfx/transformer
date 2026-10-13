import Transformer.Grokking.CircuitEfficiency.SectionC_GainPositiveLimits

/-!
# An efficient absent circuit cannot beat a native Mem-only boundary path

Sources: Varma et al., arXiv:2309.02390v1, section 3, the three
ingredients and two-factor formation, and appendix C product/test
tables; native AdamW/clipping at lab commit 4965889. Set both Gen
factors to zero and both Mem factors to a positive amplitude. Keep
Gen's larger physical gain 3 versus Mem's 2, one uniform positive
decay and epsilon, valid ordinary betas and actual zero buffers.

Choose epsilon and decay from this point's actual clipped CE scale.
Prove its normalized balance, full actual retained history and all
physical parameter coordinates. At every clock the train class is
uniquely correct, but the actual wrong Mem test class is uniquely
correct according to the model. Gen remains absent even though its
physical norm efficiency is higher. This supplies a genuine nonzero
boundary obstruction to dropping the positive-interior premise.

Amplitude, epsilon and decay are explicitly chosen mathematical
witness settings, not the frozen experiment. This is not attraction
from positive Gen source seeds, arbitrary-time zero train CE, a
universal grokking impossibility or learned stochastic GPTMini.
Both retained buffers and growing clocks remain in the actual path;
no stale nonzero Gen numerator is assumed to vanish or reset.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- A nonzero Mem-only point satisfies actual uniform native balance
despite Gen's larger gain. Sources: appendix C product CE and native
normalization at 4965889; positive settings are derived from actual CE. -/
theorem gain_native_mem_only_point_balance (remaining : ℕ) (bound amplitude : ℝ)
    (hclip : 0 < bound) (ha : 0 < amplitude) :
    let point := seededNativeSubweights ((0, 0), (amplitude, amplitude))
    let eps := gainCEGradientScale remaining 3 2 bound point * 2 * amplitude
    0 < eps ∧ 0 < 1 / (2 * amplitude) ∧ ∀ i,
      (1 / (2 * amplitude)) * (point i).parameter + appliedGainNativeGradient remaining 3 2 bound point i /
        (|appliedGainNativeGradient remaining 3 2 bound point i| + eps) = 0 := by
  dsimp only
  let point := seededNativeSubweights ((0, 0), (amplitude, amplitude))
  let scale := gainCEGradientScale remaining 3 2 bound point
  have hs : 0 < scale := gain_ce_gradient_scale_pos remaining 3 2 bound point hclip
  have hinput : 0 < scale * 2 * amplitude := by positivity
  have hp : ∀ i : Fin 4, (point i).parameter = ![0, 0, amplitude, amplitude] i := by
    intro i
    fin_cases i <;> rfl
  have hg : ∀ i, appliedGainNativeGradient remaining 3 2 bound point i =
      ![0, 0, -(scale * 2 * amplitude), -(scale * 2 * amplitude)] i := by
    intro i
    rw [gain_native_applied_gradient_scale, hp]
    change -(scale * nativeFactorGain 3 2 i * ![0, 0, amplitude, amplitude] (nativeFactorPartner i)) = _
    fin_cases i <;> simp [nativeFactorGain, nativeFactorPartner]
  refine ⟨hinput, by positivity, ?_⟩
  intro i
  change (1 / (2 * amplitude)) * (point i).parameter + appliedGainNativeGradient remaining 3 2 bound point i /
    (|appliedGainNativeGradient remaining 3 2 bound point i| + scale * 2 * amplitude) = 0
  rw [hg, hp]
  fin_cases i
  · change 1 / (2 * amplitude) * 0 + 0 / (|0| + scale * 2 * amplitude) = 0
    norm_num
  · change 1 / (2 * amplitude) * 0 + 0 / (|0| + scale * 2 * amplitude) = 0
    norm_num
  · change 1 / (2 * amplitude) * amplitude + -(scale * 2 * amplitude) /
      (|-(scale * 2 * amplitude)| + scale * 2 * amplitude) = 0
    rw [abs_of_neg (neg_neg_of_pos hinput)]
    field_simp
    ring
  · change 1 / (2 * amplitude) * amplitude + -(scale * 2 * amplitude) /
      (|-(scale * 2 * amplitude)| + scale * 2 * amplitude) = 0
    rw [abs_of_neg (neg_neg_of_pos hinput)]
    field_simp
    ring

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 10 := by norm_num

/-- The Mem-only boundary point is an actual zero-buffer retained
CE path at every valid native beta. Sources: appendix C products and
native AdamW at 4965889; its point equations are realized by the full
generated histories, not promoted directly to an attracting state. -/
theorem gain_native_mem_only_path_history (remaining : ℕ) (bound amplitude b1 b2 rate : ℝ) (clock : ℕ)
    (hclip : 0 < bound) (ha : 0 < amplitude) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) :
    let point := seededNativeSubweights ((0, 0), (amplitude, amplitude))
    let eps := gainCEGradientScale remaining 3 2 bound point * 2 * amplitude
    gainNativePath remaining 3 2 bound b1 b2 eps (1 / (2 * amplitude)) rate point clock =
      gainBalancedHistory remaining 3 2 bound b1 b2 ((0, 0), (amplitude, amplitude)) clock := by
  dsimp only
  exact gain_native_balanced_path_history remaining 3 2 bound b1 b2 _ (1 / (2 * amplitude)) rate
    ((0, 0), (amplitude, amplitude)) clock hb1 h1 hb2 h2
    (gain_native_mem_only_point_balance remaining bound amplitude hclip ha).2.2

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 10 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 := by norm_num

/-- Every physical coordinate on the actual boundary path retains
its seeded value, including the absent efficient Gen pair. Sources:
appendix C products and native AdamW at 4965889; this follows from
the complete retained history identity, not an invariant predicate. -/
theorem gain_native_mem_only_path_parameters (remaining : ℕ) (bound amplitude b1 b2 rate : ℝ) (clock : ℕ) (i : Fin 4)
    (hclip : 0 < bound) (ha : 0 < amplitude) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) :
    let point := seededNativeSubweights ((0, 0), (amplitude, amplitude))
    let eps := gainCEGradientScale remaining 3 2 bound point * 2 * amplitude
    (gainNativePath remaining 3 2 bound b1 b2 eps (1 / (2 * amplitude)) rate point clock i).parameter =
      (point i).parameter := by
  dsimp only
  rw [gain_native_mem_only_path_history remaining bound amplitude b1 b2 rate clock hclip ha hb1 h1 hb2 h2]
  rfl

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 10 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 := by norm_num

/-- Every actual boundary clock fits the train class uniquely and
selects the wrong Mem held-out class uniquely. Sources: appendix C
test tables and native AdamW at 4965889; Gen's greater efficiency
does not recreate an absent cold pair. There is no argmax tie here. -/
theorem gain_native_mem_only_path_decisions (remaining : ℕ) (bound amplitude b1 b2 rate : ℝ) (clock : ℕ)
    (hclip : 0 < bound) (ha : 0 < amplitude) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) :
    let point := seededNativeSubweights ((0, 0), (amplitude, amplitude))
    let eps := gainCEGradientScale remaining 3 2 bound point * 2 * amplitude
    let state := gainNativePath remaining 3 2 bound b1 b2 eps (1 / (2 * amplitude)) rate point clock
    Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining
      (physicalCircuitScore 3 (state 0).parameter (state 1).parameter)
      (physicalCircuitScore 2 (state 2).parameter (state 3).parameter)) 0 ∧
    Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining
      (physicalCircuitScore 3 (state 0).parameter (state 1).parameter)
      (physicalCircuitScore 2 (state 2).parameter (state 3).parameter)) (0 : Fin (remaining + 1)).succ := by
  dsimp only
  rw [gain_native_mem_only_path_parameters remaining bound amplitude b1 b2 rate clock 0 hclip ha hb1 h1 hb2 h2,
    gain_native_mem_only_path_parameters remaining bound amplitude b1 b2 rate clock 1 hclip ha hb1 h1 hb2 h2,
    gain_native_mem_only_path_parameters remaining bound amplitude b1 b2 rate clock 2 hclip ha hb1 h1 hb2 h2,
    gain_native_mem_only_path_parameters remaining bound amplitude b1 b2 rate clock 3 hclip ha hb1 h1 hb2 h2]
  have hgen : physicalCircuitScore 3 (seededNativeSubweights ((0, 0), (amplitude, amplitude)) 0).parameter
      (seededNativeSubweights ((0, 0), (amplitude, amplitude)) 1).parameter = 0 := by
    norm_num [physicalCircuitScore, seededNativeSubweights, seededScalarState]
  have hmem : physicalCircuitScore 2 (seededNativeSubweights ((0, 0), (amplitude, amplitude)) 2).parameter
      (seededNativeSubweights ((0, 0), (amplitude, amplitude)) 3).parameter = 2 * (amplitude * amplitude) := rfl
  rw [hgen, hmem]
  have hp : 0 < 2 * (amplitude * amplitude) := by positivity
  constructor
  · exact (train_table_strict_correct_iff remaining 0 _).mpr (by linarith)
  · intro k hk
    have hw : heldoutTableLogits remaining 0 (2 * (amplitude * amplitude)) (0 : Fin (remaining + 1)).succ =
        2 * (amplitude * amplitude) := by simp [heldoutTableLogits]
    rw [hw]
    by_cases hz : k = 0
    · rw [hz]
      simpa only [heldoutTableLogits, ite_true] using hp
    · simpa only [heldoutTableLogits, hz, hk, ite_false] using hp

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 10 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
