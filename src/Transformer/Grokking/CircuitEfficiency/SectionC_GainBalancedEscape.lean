import Transformer.Grokking.CircuitEfficiency.SectionC_GainRatioGrowth
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Finite Gen/Mem factor crossing on the actual initialized native path

Sources: Varma et al., arXiv:2309.02390v1, section 3's competing
circuits and appendix C's product CE; native AdamW at lab commit
b7c4293. Derive finite crossing from positive balanced initialization,
greater Gen physical gain, positive decay/epsilon/rate and positive
remaining decay. The path is the actual shared clipped CE recurrence.

Its physical box, positive finite-clock coordinates, shared scale
floor and full pair-state symmetry are derived from initialization.
If crossing never happens, the actual Gen/Mem ratio grows at least
as q^n for one fixed q>1, yet is bounded by one. Unbounded powers give
a contradiction. No finite parameter limit, successful reference,
prescribed gradient stream or future attraction premise is supplied.

This is an exact-real balanced zero-beta specialization with fixed
gained tables and uniform decoupled AdamW, differing from appendix C's
coupled assigned norm/GD and unbalanced first-zero-factor seeds, and
from learned nonzero-beta GPTMini. The bound is qualitative, not a
measured transition time. Persistent held-out ordering and a combined
arbitrarily long wrong-prefix/later-success statement follow separately.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- All factors on initialized balanced native paths stay positive
at every finite clock. Sources: appendix C's paired physical factors
and native AdamW at b7c4293; no positive limiting margin is assumed. -/
theorem gain_native_seeded_balanced_positive (remaining : ℕ) (genGain memGain bound eps decay rate a b : ℝ)
    (n : ℕ) (i : Fin 4) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 < 1 - rate * decay) (ha : 0 < a) (hb : 0 < b) :
    PositiveScalarState (gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, a), (b, b))) n i) := by
  have hs := native_seeded_nonnegative a a b b (le_of_lt ha) (le_of_lt ha) (le_of_lt hb) (le_of_lt hb)
  have hi : 0 < (seededNativeSubweights ((a, a), (b, b)) i).parameter := by
    fin_cases i
    · exact ha
    · exact ha
    · exact hb
    · exact hb
  exact gain_native_positive_coordinate_path remaining genGain memGain bound 0 0 eps decay rate
    _ n i hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he heta hd hs hi

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by
  norm_num

/-- Actual native Gen factors catch Mem at a finite update without
a convergence premise. Sources: appendix C competing gained products
and native AdamW at b7c4293; all future bounds used in the argument
are proved consequences of the stated numerical initialization. -/
theorem gain_native_balanced_finite_crossing (remaining : ℕ) (genGain memGain bound eps decay rate a b : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (ha : 0 < a) (hb : 0 < b) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, a), (b, b)))
    ∃ n, (path n 2).parameter < (path n 0).parameter := by
  dsimp only
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, a), (b, b)))
  change ∃ n, (path n 2).parameter < (path n 0).parameter
  by_contra hnever
  push Not at hnever
  have hg : 0 < genGain := lt_trans hmem hgain
  obtain ⟨ceiling, hceiling, hbox⟩ := gain_native_seeded_bounded remaining genGain memGain bound 0 0 eps decay rate
    a a b b hg hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he hdecay
    (le_of_lt heta) (le_of_lt hd) (le_of_lt ha) (le_of_lt ha) (le_of_lt hb) (le_of_lt hb)
  let lower := gainCEFeedbackFloor remaining genGain memGain bound ceiling
  let growth := gainBalancedRatioGrowth lower eps genGain memGain ceiling decay rate
  have hlower : 0 < lower := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling
    (le_of_lt hg) (le_of_lt hceiling) hclip
  have hgrowth : 1 < growth := gain_balanced_ratio_growth_one_lt lower eps genGain memGain ceiling decay rate
    hlower he hmem hgain (le_of_lt hceiling) heta hd
  have hpos : ∀ n i, 0 < (path n i).parameter := by
    intro n i
    exact (gain_native_seeded_balanced_positive remaining genGain memGain bound eps decay rate a b n i
      hg hmem hclip he (le_of_lt heta) hd ha hb).1
  have hstep : ∀ n, growth * ((path n 0).parameter / (path n 2).parameter) ≤
      (path (n + 1) 0).parameter / (path (n + 1) 2).parameter := by
    intro n
    have hparam : ∀ i, 0 ≤ (path n i).parameter ∧ (path n i).parameter ≤ ceiling := by
      intro i
      exact ⟨((hbox n).1 i).1, ((hbox n).2 i).1⟩
    exact gain_native_balanced_ratio_growth remaining genGain memGain bound ceiling eps decay rate (path n)
      hmem hgain hclip he heta hd hparam
      (gain_native_equal_pair_path remaining genGain memGain bound 0 0 eps decay rate a b n) (hpos n 2) (hnever n)
  have hpower : ∀ n, growth ^ n * (a / b) ≤ (path n 0).parameter / (path n 2).parameter := by
    intro n
    induction n with
    | zero => simp [path, gainNativePath, seededNativeSubweights, seededScalarState]
    | succ n ih =>
      calc
        growth ^ (n + 1) * (a / b) = growth * (growth ^ n * (a / b)) := by rw [pow_succ]; ring
        _ ≤ growth * ((path n 0).parameter / (path n 2).parameter) :=
          mul_le_mul_of_nonneg_left ih (by linarith only [hgrowth])
        _ ≤ (path (n + 1) 0).parameter / (path (n + 1) 2).parameter := hstep n
  obtain ⟨n, hn⟩ := ((tendsto_pow_atTop_atTop_of_one_lt hgrowth).eventually_gt_atTop (b / a)).exists
  have hlarge : b < growth ^ n * a := (div_lt_iff₀ ha).mp hn
  have hratio : 1 < growth ^ n * (a / b) := by
    have heq : growth ^ n * (a / b) = (growth ^ n * a) / b := by ring
    rw [heq]
    apply (lt_div_iff₀ hb).mpr
    simpa only [one_mul] using hlarge
  have hupper : (path n 0).parameter / (path n 2).parameter ≤ 1 := by
    apply (div_le_iff₀ (hpos n 2)).mpr
    simpa only [one_mul] using hnever n
  linarith only [hratio, hpower n, hupper]

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 10 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by
  norm_num

/-- One fixed legal binary native configuration has finite factor
crossing for every positive balanced Gen seed against unit Mem.
Sources: appendix C binary CE and the native port at b7c4293;
this fixes actual hyperparameters rather than supplying a later limit. -/
theorem fixed_native_balanced_factor_crossing (seed : ℝ) (hseed : 0 < seed) :
    ∃ n, (gainNativePath 0 3 2 1 0 0 1 (1 / 10) (1 / 1000)
      (seededNativeSubweights ((seed, seed), (1, 1))) n 2).parameter <
      (gainNativePath 0 3 2 1 0 0 1 (1 / 10) (1 / 1000)
        (seededNativeSubweights ((seed, seed), (1, 1))) n 0).parameter := by
  exact gain_native_balanced_finite_crossing 0 3 2 1 1 (1 / 10) (1 / 1000) seed 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hseed (by norm_num)

example : (0 : ℝ) < 1 / 200 := by
  norm_num

/-- Actual balanced native feedback reaches strictly correct train
and held-out decisions at a finite update. Sources: appendix C's
true Gen/Mem logits and native AdamW at b7c4293; derive the successful
physical scores from the generated crossing, not a target premise. -/
theorem gain_native_balanced_finite_correct (remaining : ℕ) (genGain memGain bound eps decay rate a b : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (ha : 0 < a) (hb : 0 < b) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, a), (b, b)))
    ∃ n, StrictCorrect (trainTableLogits remaining
      (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
      (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 ∧
      StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 := by
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, a), (b, b)))
  obtain ⟨n, hcross⟩ := gain_native_balanced_finite_crossing remaining genGain memGain bound eps decay rate a b
    hmem hgain hclip he hdecay heta hd ha hb
  have hg : 0 < genGain := lt_trans hmem hgain
  have hgp := (gain_native_seeded_balanced_positive remaining genGain memGain bound eps decay rate a b n 0
    hg hmem hclip he (le_of_lt heta) hd ha hb).1
  have hmp := (gain_native_seeded_balanced_positive remaining genGain memGain bound eps decay rate a b n 2
    hg hmem hclip he (le_of_lt heta) hd ha hb).1
  have hs := gain_native_equal_pair_path remaining genGain memGain bound 0 0 eps decay rate a b n
  change (path n 2).parameter < (path n 0).parameter at hcross
  change 0 < (path n 0).parameter at hgp
  change 0 < (path n 2).parameter at hmp
  change NativePairSymmetry (path n) at hs
  have hx : 0 < physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter := by
    rw [← hs.1]
    exact mul_pos hg (mul_pos hgp hgp)
  have hy : 0 < physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter := by
    rw [← hs.2]
    exact mul_pos hmem (mul_pos hmp hmp)
  have hsq : (path n 2).parameter * (path n 2).parameter < (path n 0).parameter * (path n 0).parameter := by
    nlinarith only [hcross, hgp, hmp]
  have hfirst := mul_lt_mul_of_pos_right hgain (mul_pos hmp hmp)
  have hsecond := mul_lt_mul_of_pos_left hsq hg
  have hscore : physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter <
      physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter := by
    rw [← hs.1, ← hs.2]
    exact lt_trans hfirst hsecond
  refine ⟨n, ?_, heldout_table_strict_correct remaining _ _ hx hscore⟩
  apply (train_table_strict_correct_iff remaining _ _).mpr
  linarith only [hx, hy]

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 10 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency
