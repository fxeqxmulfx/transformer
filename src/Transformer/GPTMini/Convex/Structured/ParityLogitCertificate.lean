import Transformer.GPTMini.Convex.Structured.TensorParityCertificate

/-!
# Rationally checkable actual learned parity logit gaps

Source: plan.md §4 and the actual finite-row tensor parity certificate at
9b86323. The conditions below compare raw unrestricted learned logits,
not numerical exp/softmax estimates or assumed correct output logits.
They are finite differences of the actual initial, nine needed transition,
fifteen needed value and learned branch rows.

A common gap eleven gives a conservative sufficient budget, because
exp(11) >= 2^11 = 2048 and 142/2048 < 1/11. Thus finite exact rational
comparisons on dyadic learned weights can certify all raw valid parity
prefixes for the real tensor model. Checking those actual checkpoint
coordinates and IEEE preservation are separate tasks. No property of
AdamW convergence is assumed; every other parameter remains unrestricted.

The source data labels select which coordinates to check only. They
remain absent from the learned forward call. Actual normalized confidence,
whole-vocabulary margin and full raw integer transfer are derived through
the previously proved true likelihood/inference coupling.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- A predicate on actual raw logits and their finite target/rival differences.
Source: Concentration.stateRow_gap's local learned-energy hypothesis, before normalization. -/
def RowLogitGap {S : Type*} (score : S → ℝ) (selected : S) (gap : ℝ) : Prop :=
  ∀ rival, rival ≠ selected → score rival ≤ score selected - gap

/-- Only the nine transition rows required by the independent parity grammar need raw logit gaps.
Source: ParityTransitionRows's true BOS/bit/SEP/answer cases, now with exact unnormalized coordinate comparisons. -/
def ParityTransitionLogitGaps (transition : Fin 68 → Fin 6 → Fin 6 → ℝ) (gap : ℝ) : Prop :=
  RowLogitGap (transition 1 0) 0 gap ∧
  (∀ bit : Bool, ∀ phase : Fin 6, phase = 0 ∨ phase = 1 →
    RowLogitGap (transition (if bit then 22 else 21) phase) (paritySharedRule (if bit then 22 else 21) phase) gap) ∧
  (∀ phase : Fin 6, phase = 0 ∨ phase = 1 → RowLogitGap (transition 18 phase) (paritySharedRule 18 phase) gap) ∧
  RowLogitGap (transition 24 2) 4 gap ∧ RowLogitGap (transition 25 3) 4 gap

/-- Needed finite raw coordinate comparisons on arbitrary actual full learned parameters.
Source: independently required parity path/value coordinates and actual learned initial/branch reads; all other weights stay free. -/
def ParityModelLogitGaps (θ : BindingParameters 68 19) (gap : ℝ) : Prop :=
  RowLogitGap (fun s => sharedInitialRead s θ.1) 0 gap ∧
  ParityTransitionLogitGaps (fun token state next => sharedTransitionRead token state next θ.1) gap ∧
  (∀ state : Fin 6, state = 2 ∨ state = 3 ∨ state = 4 → ∀ h,
    RowLogitGap (fun d => sharedEmissionRead state h d θ.1)
      (outputDigit (vocabularyCode (by decide) (paritySharedLabel state)) h) gap) ∧
  RowLogitGap (fun head => mixedHeadRead head θ) 0 gap

/-- Real finite full shared parameters meet every needed raw coordinate gap exactly.
Source: genuine referenceShared/mixedReference coordinate identities and sharpRowLogits_gap, without a hard inference replacement. -/
theorem parityReference_logit_gaps (gap : ℝ) : ParityModelLogitGaps
    (mixedReferenceParameters (C := 19) paritySharedRule
      (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 gap) gap := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro rival hr
    unfold mixedReferenceParameters
    simp_rw [referenceShared_initial]
    exact sharpRowLogits_gap (0 : Fin 6) rival gap hr
  · have ht : ∀ token state, RowLogitGap (fun next => sharedTransitionRead token state next
        (mixedReferenceParameters (C := 19) paritySharedRule
          (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 gap).1)
        (paritySharedRule token state) gap := by
      intro token state rival hr
      unfold mixedReferenceParameters
      simp_rw [referenceShared_transition]
      exact sharpRowLogits_gap (paritySharedRule token state) rival gap hr
    refine ⟨ht 1 0, ?_, ?_, ht 24 2, ht 25 3⟩
    · rintro bit phase (rfl | rfl)
      · exact ht (if bit then 22 else 21) 0
      · exact ht (if bit then 22 else 21) 1
    · rintro phase (rfl | rfl)
      · exact ht 18 0
      · exact ht 18 1
  · rintro state (rfl | rfl | rfl) h rival hr <;>
      unfold mixedReferenceParameters <;> simp_rw [referenceShared_emission] <;>
      exact sharpRowLogits_gap _ rival gap hr
  · intro rival hr
    dsimp only
    rw [mixedReference_head, mixedReference_head]
    exact sharpRowLogits_gap (0 : Fin 2) rival gap hr

/-- Needed raw transition differences derive every corresponding genuine learned normalized row confidence.
Source: actual six-state softmax gap bound, applied only to the nine independently required transitions. -/
theorem parityTransitionLogitGaps_rows (transition : Fin 68 → Fin 6 → Fin 6 → ℝ) (gap : ℝ)
    (hgaps : ParityTransitionLogitGaps transition gap) :
    ParityTransitionRows transition (6 * Real.exp (-gap)) := by
  obtain ⟨hb, ht, hs, he, ho⟩ := hgaps
  refine ⟨stateRow_gap _ 0 gap hb, ?_, ?_, stateRow_gap _ 4 gap he, stateRow_gap _ 4 gap ho⟩
  · intro bit phase hphase
    exact stateRow_gap _ _ gap (ht bit phase hphase)
  · intro phase hphase
    exact stateRow_gap _ _ gap (hs phase hphase)

example : ParityTransitionLogitGaps (fun token state next => sharedTransitionRead token state next
    (mixedReferenceParameters (C := 19) paritySharedRule
      (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 11).1) 11 :=
  (parityReference_logit_gaps 11).2.1

/-- Exact finite raw logit conditions imply all actual needed initial/path/value/branch confidence conditions.
Source: true learned row softmax and row-size tails six/four/two, with arbitrary remaining full parameters. -/
theorem parityModelLogitGaps_rows (θ : BindingParameters 68 19) (gap : ℝ)
    (hgaps : ParityModelLogitGaps θ gap) :
    ParityModelRows θ (6 * Real.exp (-gap)) (6 * Real.exp (-gap))
      (4 * Real.exp (-gap)) (2 * Real.exp (-gap)) := by
  obtain ⟨hi, ht, hv, hh⟩ := hgaps
  refine ⟨stateRow_gap _ 0 gap hi, parityTransitionLogitGaps_rows _ gap ht, ?_, ?_⟩
  · intro state hstate h
    exact stateRow_gap _ _ gap (hv state hstate h)
  · exact stateRow_gap (fun head => mixedHeadRead head θ) 0 gap hh

example : ParityModelLogitGaps (mixedReferenceParameters (C := 19) paritySharedRule
    (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 11) 11 :=
  parityReference_logit_gaps 11

/-- Gap eleven is sufficient for the entire true nineteen-position mixed-model decoder budget.
Source: actual 142 row-size deficit coefficient and the elementary real bound exp(11) >= 2^11; no numerical transcendental evaluation. -/
theorem parityLogitGap_eleven_budget : (0 : ℝ) ≤ 6 * Real.exp (-11) ∧
    2 * Real.exp (-11) + 6 * Real.exp (-11) + 19 * (6 * Real.exp (-11)) +
      5 * (4 * Real.exp (-11)) < (1 / 11 : ℝ) := by
  have h1 : (2 : ℝ) ≤ Real.exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
  have hpow := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 2) h1 11
  have he : (2048 : ℝ) ≤ Real.exp 11 := by
    rw [← Real.exp_nat_mul] at hpow
    norm_num only [Nat.cast_ofNat, mul_one, pow_succ, pow_zero] at hpow
    exact hpow
  have hb := div_le_div_of_nonneg_left (by norm_num : (0 : ℝ) ≤ 142)
    (by norm_num : (0 : ℝ) < 2048) he
  rw [div_eq_mul_inv, ← Real.exp_neg] at hb
  have hp := Real.exp_pos (-11)
  constructor <;> norm_num at hb ⊢ <;> nlinarith

/-- Exact finite actual learned logit differences certify every valid parity input of the genuine full tensor integer function.
Source: derived normalized row bounds, the rational sufficient whole-model budget and actual all-prefix tensor semantics. -/
theorem tensorParity_solves_of_logit_gaps (mode : Mode) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters 68 19) (hgaps : ParityModelLogitGaps θ 11) :
    SolvesTask (tensorFunction (basisTensorConfig mode .parity) (basisTensorConfig_fits mode .parity).1
      (basisTensorConfig_fits mode .parity).2 eps θ) mode .parity := by
  exact tensorParity_solves_of_rows mode eps heps θ _ _ _ _
    parityLogitGap_eleven_budget.1 parityLogitGap_eleven_budget.2 (parityModelLogitGaps_rows θ 11 hgaps)

example : (0 : ℝ) < 1 / 100000 ∧ ParityModelLogitGaps (mixedReferenceParameters (C := 19) paritySharedRule
    (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 11) 11 :=
  ⟨by norm_num, parityReference_logit_gaps 11⟩

end
end Transformer.GPTMini.Convex.Structured
