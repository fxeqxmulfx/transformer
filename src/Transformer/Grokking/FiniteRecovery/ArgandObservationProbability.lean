module

public import Transformer.Grokking.FiniteRecovery.ArgandEnergyRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandMomentumLimits
public import Transformer.Grokking.FiniteRecovery.Observations
public import Mathlib.FieldTheory.Finiteness
public import Mathlib.MeasureTheory.Measure.Real

/-! Finite observation counting for the actual randomized training run.
For a fixed mask the total success probability across distinct targets
is at most the number of possible observed label vectors. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Finset MeasureTheory Set Transformer.Grokking.FlatSpectrum.TablePart
open scoped Topology BigOperators

variable {A Ω I O : Type*} {T H d layers : ℕ}

private theorem exists_prob_le_labels_div_targets [Fintype I] [Nonempty I] [Fintype O]
    [MeasurableSpace Ω] (mu : Measure Ω) [IsProbabilityMeasure mu]
    (labels : I → O) (success : I → Set Ω) (hm : ∀ i, MeasurableSet (success i))
    (hd : ∀ i j, i ≠ j → labels i = labels j → Disjoint (success i) (success j)) :
    ∃ i, mu.real (success i) ≤ (Fintype.card O : ℝ) / Fintype.card I := by
  classical
  have hf (o : O) : ∑ i ∈ Finset.univ.filter (fun i => labels i = o), mu.real (success i) ≤ 1 := by
    rw [← measureReal_biUnion_finset (f := success) (fun i hi j hj hij => hd i j hij
      ((Finset.mem_filter.mp hi).2.trans (Finset.mem_filter.mp hj).2.symm)) (fun i _ => hm i)]
    exact measureReal_le_one
  have ht := Finset.sum_le_sum (s := (Finset.univ : Finset O)) fun o _ => hf o
  rw [Finset.sum_fiberwise_of_maps_to (fun i _ => Finset.mem_univ (labels i))] at ht
  have ht' : ∑ i, mu.real (success i) ≤ (Fintype.card O : ℝ) := by simpa using ht
  by_contra! h
  have hs := Finset.sum_lt_sum_of_nonempty (Finset.univ_nonempty (α := I)) (fun i _ => h i)
  have hn : (Fintype.card I : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_pos.ne'
  simpa [nsmul_eq_mul, mul_div_cancel₀ _ hn] using hs.trans_le ht'

variable [Fintype A]

/-- Eventual strict correctness also contains every draw with a strict
correct weight limit, by continuity of the logits on the finite domain. -/
def eventualRecoveryEvent (realScoreInit : Bool) (repeats : ℕ) (lam : ℝ) (rate : ℕ → ℝ)
    (inputs : Finset (Word A (T + 1))) (target : Word A (T + 1) → A) :
    Set (ComplexParameter A H d layers → ℂ) :=
  {draw | ∀ᶠ t in atTop, (Model.fromWeights ((fullGradientMomentumRun realScoreInit repeats
    (observedLastPositions target inputs) lam rate draw t).weights)).realizes repeats target}

theorem mem_eventualRecoveryEvent_of_limit (realScoreInit : Bool) (repeats : ℕ)
    (lam : ℝ) (rate : ℕ → ℝ) (inputs : Finset (Word A (T + 1))) (target : Word A (T + 1) → A)
    (draw : ComplexParameter A H d layers → ℂ) (star : Weights A H d layers)
    (h : Tendsto (fun t => (fullGradientMomentumRun realScoreInit repeats
      (observedLastPositions target inputs) lam rate draw t).weights) atTop (𝓝 star))
    (hs : (Model.fromWeights star).realizes repeats target) :
    draw ∈ eventualRecoveryEvent realScoreInit repeats lam rate inputs target :=
  realizes_eventually_of_weight_limit h repeats target hs

private theorem measurable_eventualRecoveryEvent (realScoreInit : Bool) (repeats : ℕ)
    (lam : ℝ) (rate : ℕ → ℝ) (inputs : Finset (Word A (T + 1))) (target : Word A (T + 1) → A) :
    MeasurableSet (eventualRecoveryEvent (H := H) (d := d) (layers := layers)
      realScoreInit repeats lam rate inputs target) := by
  let batch := observedLastPositions target inputs
  let fit := fullGradientMomentumRun (H := H) (d := d) (layers := layers) realScoreInit repeats batch lam rate
  have ht (t : ℕ) : MeasurableSet {draw | (Model.fromWeights (fit draw t).weights).realizes repeats target} := by
    simp only [Model.realizes, ofPred_forall]
    refine MeasurableSet.iInter fun word => MeasurableSet.iInter fun a => ?_
    by_cases ha : a = target word
    · simp [ha]
    · simp only [ha, ne_eq, not_false_eq_true, iInter_true]
      have hx := (continuous_prefix_weights_velocity (H := H) (d := d) (layers := layers)
        realScoreInit repeats batch lam rate t).1
      exact (isOpen_lt ((continuous_logits_fromWeights repeats word _ a).comp hx)
        ((continuous_logits_fromWeights repeats word _ (target word)).comp hx)).measurableSet
  unfold eventualRecoveryEvent
  simp only [eventually_atTop, ofPred_exists, ofPred_forall]
  refine MeasurableSet.iUnion fun n => MeasurableSet.iInter fun t => ?_
  by_cases h : n ≤ t
  · simpa only [h, iInter_true] using ht t
  · simp [h]

/-- At least one target in any injected finite family has success
probability at most the number of label vectors divided by the family size,
for every architecture, learning rate and initialization distribution. -/
theorem exists_argand_recovery_prob_le [Fintype I] [Nonempty I]
    (mu : Measure (ComplexParameter A H d layers → ℂ)) [IsProbabilityMeasure mu]
    (realScoreInit : Bool) (repeats : ℕ) (lam : ℝ) (rate : ℕ → ℝ)
    (inputs : Finset (Word A (T + 1))) (target : I → Word A (T + 1) → A)
    (hinj : Function.Injective target) :
    ∃ i, mu.real (eventualRecoveryEvent realScoreInit repeats lam rate inputs (target i)) ≤
      (Fintype.card A : ℝ) ^ inputs.card / Fintype.card I := by
  classical
  let labels (i : I) : inputs → A := fun w => target i w
  let E i := eventualRecoveryEvent (H := H) (d := d) (layers := layers)
    realScoreInit repeats lam rate inputs (target i)
  have hd (i j : I) (hne : i ≠ j) (heq : labels i = labels j) : Disjoint (E i) (E j) := by
    have hb : observedLastPositions (target i) inputs = observedLastPositions (target j) inputs := by
      apply List.map_congr_left
      intro w hw
      have h := congrFun heq ⟨w, by simpa using hw⟩
      simp only [labels] at h
      rw [h]
    apply Set.disjoint_left.mpr
    intro draw hi hj
    change ∀ᶠ t in atTop, (Model.fromWeights ((fullGradientMomentumRun realScoreInit repeats
      (observedLastPositions (target i) inputs) lam rate draw t).weights)).realizes repeats (target i) at hi
    rw [hb] at hi
    obtain ⟨t, h1, h2⟩ := (hi.and hj).exists
    apply hne (hinj (funext fun w => ?_))
    by_contra h
    exact lt_asymm (h1 w _ (Ne.symm h)) (h2 w _ h)
  simpa [E, Fintype.card_fun] using exists_prob_le_labels_div_targets mu labels E
    (fun i => measurable_eventualRecoveryEvent _ _ _ _ _ _) hd

/-- For PT4's unknown coupling the exponent is the exact dimension of
the Flat Spectrum target class. This bound applies to full Argand even
when model capacity or the optimizer imposes further restrictions. -/
theorem unknownCoupling_prob_le {p left right : ℕ} [Fact p.Prime]
    (mu : Measure (ComplexParameter (ZMod p) H d layers → ℂ)) [IsProbabilityMeasure mu]
    (realScoreInit : Bool) (repeats : ℕ) (lam : ℝ) (rate : ℕ → ℝ)
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) (T + 1))
    (inputs : Finset (Word (ZMod p) (T + 1))) :
    ∃ f ∈ unknownCouplingFamily cut,
      mu.real (eventualRecoveryEvent realScoreInit repeats lam rate inputs f) ≤
        (p : ℝ) ^ inputs.card / (p : ℝ) ^ (p ^ left + p ^ right + left * right - 1) := by
  classical
  let F := flatClass (ZMod p) (Fin left) (Fin right)
  let target (i : F) := (i : _ → ZMod p) ∘ cut.symm
  have hi : Function.Injective target := by
    intro i j h
    exact Subtype.ext (funext fun c => by simpa [target] using congrFun h (cut c))
  have hc : Fintype.card F = p ^ (p ^ left + p ^ right + left * right - 1) := by
    rw [Module.card_eq_pow_finrank (K := ZMod p), ZMod.card]
    have h := finrank_flatClass (F := ZMod p) (ι := Fin left) (κ := Fin right)
    simp only [ZMod.card, Fintype.card_fin] at h
    exact congrArg (p ^ ·) (show Module.finrank (ZMod p) F = _ from
      Nat.eq_sub_of_add_eq h)
  obtain ⟨i, h⟩ := exists_argand_recovery_prob_le mu realScoreInit repeats lam rate inputs target hi
  refine ⟨target i, ?_, ?_⟩
  · simp [unknownCouplingFamily, target, Function.comp_assoc]
  · simpa [hc, ZMod.card] using h

/-- Each missing independent binary label halves the worst-case bound. -/
theorem unknownCoupling_prob_le_of_deficit {left right : ℕ}
    (mu : Measure (ComplexParameter (ZMod 2) H d layers → ℂ)) [IsProbabilityMeasure mu]
    (realScoreInit : Bool) (repeats : ℕ) (lam : ℝ) (rate : ℕ → ℝ)
    (cut : Word (ZMod 2) left × Word (ZMod 2) right ≃ Word (ZMod 2) (T + 1))
    (inputs : Finset (Word (ZMod 2) (T + 1))) {k : ℕ}
    (hk : 2 ^ left + 2 ^ right + left * right - 1 = inputs.card + k) :
    ∃ f ∈ unknownCouplingFamily cut,
      mu.real (eventualRecoveryEvent realScoreInit repeats lam rate inputs f) ≤ (1 / 2 : ℝ) ^ k := by
  obtain ⟨f, hf, h⟩ := unknownCoupling_prob_le mu realScoreInit repeats lam rate cut inputs
  refine ⟨f, hf, h.trans_eq ?_⟩
  rw [hk, pow_add, one_div_pow]
  field_simp
  norm_num

example (mu : Measure (ComplexParameter (ZMod 2) 4 16 1 → ℂ)) [IsProbabilityMeasure mu]
    (inputs : Finset (Word (ZMod 2) 8)) (hc : inputs.card = 31) :=
  unknownCoupling_prob_le_of_deficit mu true 1 (1 / 10) (wsdRate 20000)
    (appendCut (ZMod 2) 4 4) inputs (k := 16) (by rw [hc]; decide)

example (mu : Measure (ComplexParameter (ZMod 2) 4 16 1 → ℂ)) [IsProbabilityMeasure mu]
    (inputs : Finset (Word (ZMod 2) 16)) (hc : inputs.card = 511) :=
  unknownCoupling_prob_le_of_deficit mu true 1 (1 / 10) (wsdRate 20000)
    (appendCut (ZMod 2) 8 8) inputs (k := 64) (by rw [hc]; decide)

example (mu : Measure (ComplexParameter (ZMod 2) 4 16 1 → ℂ)) [IsProbabilityMeasure mu]
    (inputs : Finset (Word (ZMod 2) 32)) (hc : inputs.card = 131071) :=
  unknownCoupling_prob_le_of_deficit mu true 1 (1 / 10) (wsdRate 20000)
    (appendCut (ZMod 2) 16 16) inputs (k := 256) (by rw [hc]; decide)

end Transformer.Grokking.FiniteRecovery.Argand
