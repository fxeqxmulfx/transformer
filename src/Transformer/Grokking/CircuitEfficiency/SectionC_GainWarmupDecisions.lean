import Transformer.Grokking.CircuitEfficiency.SectionC_GainWarmupSelection

/-!
# Actual original-warmup output limits, decisions and logit robustness

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C
train/test logits; full retained original-warmup selection at fe23124.
Keep the original numerical optimizer, gains 3/2 and cap one. Every
observed score is computed from its actual current physical factors.

Under actual finite parameter convergence, nonnegative initialization
and one positive Gen partner, at most 113 classes force positive
limiting Gen score and zero Mem score. The actual held-out margin
therefore has a positive limit. A generated half-limit tail guarantees
strict train/held-out decisions against every actual competing class.
It also yields a fixed positive radius for coordinatewise output
errors on the same tail, rather than only finite-clock robustness.

The finite convergence premise is still explicit and jointly witnessed
by positive original pure-Gen paths. No attracting parameter basin,
transition delay, floating-point parameter/kernel error bound or
learned GPTMini head decomposition is proved. A bound on perturbed
logits concerns decisions only; it is not dynamical stability of the
weights. Plain CE/uniform native decay differs from the source's
coupled circuit-norm GD. Frozen training and measurements are unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- Actual gained Gen/Mem scores of the original retained warmup.
Sources: appendix C physical products and native warmup at d54cd6d;
class count, initial state and completed clock all enter the path. -/
noncomputable def originalGainWarmupScores (remaining : ℕ) (initial : NativeSubweightState) (n : ℕ) : ℝ × ℝ :=
  let state := gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial n
  (physicalCircuitScore 3 (state 0).parameter (state 1).parameter,
    physicalCircuitScore 2 (state 2).parameter (state 3).parameter)

/-- Actual full held-out logits from the current original-warmup
scores. Source: appendix C target, Mem label and all other labels. -/
noncomputable def originalGainWarmupHeldoutLogits (remaining : ℕ) (initial : NativeSubweightState) (n : ℕ) :
    Fin (remaining + 2) → ℝ :=
  heldoutTableLogits remaining (originalGainWarmupScores remaining initial n).1 (originalGainWarmupScores remaining initial n).2

/-- Actual finite warmup limits imply positive Gen score, zero Mem
score and a positive full margin limit. Sources: appendix C products
and original finite-limit selection at fe23124; no successful output
or positive limiting factor is independently assumed. -/
theorem original_gain_warmup_score_limits (remaining : ℕ) (initial reference : NativeSubweightState)
    (hq : remaining ≤ 111) (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    0 < 3 * (reference 0).parameter ^ 2 ∧
      Tendsto (fun n => (originalGainWarmupScores remaining initial n).1) atTop (nhds (3 * (reference 0).parameter ^ 2)) ∧
      Tendsto (fun n => (originalGainWarmupScores remaining initial n).2) atTop (nhds 0) ∧
      Tendsto (fun n => (originalGainWarmupScores remaining initial n).1 - (originalGainWarmupScores remaining initial n).2)
        atTop (nhds (3 * (reference 0).parameter ^ 2)) := by
  obtain ⟨hg, heq, hm0, hm1⟩ := original_gain_warmup_limit_gen_selected remaining initial reference hq hs hi hp
  have hGen : Tendsto (fun n => (originalGainWarmupScores remaining initial n).1) atTop
      (nhds (3 * (reference 0).parameter ^ 2)) := by
    simpa only [originalGainWarmupScores, physicalCircuitScore, heq, pow_two] using ((hp 0).mul (hp 1)).const_mul 3
  have hMem : Tendsto (fun n => (originalGainWarmupScores remaining initial n).2) atTop (nhds 0) := by
    simpa only [originalGainWarmupScores, physicalCircuitScore, hm0, hm1, mul_zero] using ((hp 2).mul (hp 3)).const_mul 2
  exact ⟨by positivity, hGen, hMem, by simpa only [sub_zero] using hGen.sub hMem⟩

example : (95 : ℕ) ≤ 111 ∧ ∃ initial reference : NativeSubweightState,
    NonnegativeNativeState initial ∧ 0 < (initial 1).parameter ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 95 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
  obtain ⟨point, hs, hi, hp⟩ := original_gain_warmup_positive_limit_witness 95
  exact ⟨by norm_num, point, point, hs, hi, hp⟩

/-- The actual warmup margin eventually exceeds half its derived
positive limit. Sources: appendix C held-out products and finite
Gen selection at fe23124; no positive future margin is a premise. -/
theorem original_gain_warmup_positive_margin_tail (remaining : ℕ) (initial reference : NativeSubweightState)
    (hq : remaining ≤ 111) (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    0 < 3 * (reference 0).parameter ^ 2 ∧ ∀ᶠ n in atTop,
      (3 * (reference 0).parameter ^ 2) / 2 <
        (originalGainWarmupScores remaining initial n).1 - (originalGainWarmupScores remaining initial n).2 := by
  have hlimits := original_gain_warmup_score_limits remaining initial reference hq hs hi hp
  exact ⟨hlimits.1, hlimits.2.2.2.eventually_const_lt (by linarith only [hlimits.1])⟩

example : (111 : ℕ) ≤ 111 ∧ ∃ initial reference : NativeSubweightState,
    NonnegativeNativeState initial ∧ 0 < (initial 1).parameter ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 111 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
  obtain ⟨point, hs, hi, hp⟩ := original_gain_warmup_positive_limit_witness 111
  exact ⟨by norm_num, point, point, hs, hi, hp⟩

/-- The actual original-warmup path eventually answers every train
and held-out class strictly correctly. Sources: appendix C full
logits and the derived positive score/margin limits; this concerns
the actual path and does not postulate a successful future start. -/
theorem original_gain_warmup_eventually_correct (remaining : ℕ) (initial reference : NativeSubweightState)
    (hq : remaining ≤ 111) (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    ∀ᶠ n in atTop, StrictCorrect (trainTableLogits remaining
      (originalGainWarmupScores remaining initial n).1 (originalGainWarmupScores remaining initial n).2) 0 ∧
      StrictCorrect (originalGainWarmupHeldoutLogits remaining initial n) 0 := by
  have hl := original_gain_warmup_score_limits remaining initial reference hq hs hi hp
  have hz : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0) := tendsto_const_nhds
  have ht : Tendsto (fun n => (originalGainWarmupScores remaining initial n).1 +
      (originalGainWarmupScores remaining initial n).2) atTop (nhds (3 * (reference 0).parameter ^ 2)) := by
    simpa only [add_zero] using hl.2.1.add hl.2.2.1
  have htrain := hz.eventually_lt ht hl.1
  have hGen := hz.eventually_lt hl.2.1 hl.1
  have hMem := hl.2.2.1.eventually_lt hl.2.1 hl.1
  exact (htrain.and (hGen.and hMem)).mono fun n hh =>
    ⟨(train_table_strict_correct_iff remaining _ _).2 hh.1,
      heldout_table_strict_correct remaining _ _ hh.2.1 hh.2.2⟩

example : (95 : ℕ) ≤ 111 ∧ ∃ initial reference : NativeSubweightState,
    NonnegativeNativeState initial ∧ 0 < (initial 1).parameter ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 95 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
  obtain ⟨point, hs, hi, hp⟩ := original_gain_warmup_positive_limit_witness 95
  exact ⟨by norm_num, point, point, hs, hi, hp⟩

/-- A fixed positive coordinatewise logit-error radius preserves
all-class held-out correctness on an actual original-warmup tail.
Sources: appendix C full competitors and derived half-margin tail;
this is output robustness, not an attracting parameter neighborhood. -/
theorem original_gain_warmup_robust_radius_tail (remaining : ℕ) (initial reference : NativeSubweightState)
    (hq : remaining ≤ 111) (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    ∃ radius : ℝ, 0 < radius ∧ ∀ᶠ n in atTop, ∀ perturbed : Fin (remaining + 2) → ℝ,
      (∀ k, |perturbed k - originalGainWarmupHeldoutLogits remaining initial n k| ≤ radius) → StrictCorrect perturbed 0 := by
  have hmargin := original_gain_warmup_positive_margin_tail remaining initial reference hq hs hi hp
  let radius := (3 * (reference 0).parameter ^ 2) / 8
  have hr : 0 < radius := by dsimp [radius]; linarith only [hmargin.1]
  refine ⟨radius, hr, hmargin.2.mono ?_⟩
  intro n hgap perturbed hbound k hk
  have hsign := gain_native_scheduled_nonnegative_path remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000)
    (1 / 10) grokkingWarmupRate initial n (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun j => ⟨(grokking_warmup_rate_legal j).1, (grokking_warmup_rate_legal j).2.2⟩) hs
  have hm : 0 ≤ (originalGainWarmupScores remaining initial n).2 :=
    mul_nonneg (by norm_num) (mul_nonneg (hsign 2).1 (hsign 3).1)
  have hc : originalGainWarmupHeldoutLogits remaining initial n k ≤ (originalGainWarmupScores remaining initial n).2 := by
    change (if k = 0 then _ else if k = (0 : Fin (remaining + 1)).succ then _ else 0) ≤ _
    rw [ite_eq_right hk]
    split_ifs
    · exact le_rfl
    · exact hm
  have h0 := (abs_le.mp (hbound 0)).1
  have hother := (abs_le.mp (hbound k)).2
  change -radius ≤ perturbed 0 - (originalGainWarmupScores remaining initial n).1 at h0
  dsimp only [radius] at h0 hother
  linarith only [hmargin.1, hgap, hc, h0, hother]

example : (111 : ℕ) ≤ 111 ∧ ∃ initial reference : NativeSubweightState,
    NonnegativeNativeState initial ∧ 0 < (initial 1).parameter ∧
    ∀ i, Tendsto (fun n => (gainNativeWarmupPath 111 3 2 1 (1 / 100000000) initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
  obtain ⟨point, hs, hi, hp⟩ := original_gain_warmup_positive_limit_witness 111
  exact ⟨by norm_num, point, point, hs, hi, hp⟩

end Transformer.Grokking.CircuitEfficiency
