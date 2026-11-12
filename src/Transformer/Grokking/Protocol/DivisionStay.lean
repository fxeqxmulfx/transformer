module

public import Transformer.Grokking.Protocol.DivisionMeasurable
public import Transformer.Grokking.Protocol.EventualStay
public import Transformer.Grokking.Protocol.EventualSuccess

/-!
# Division: eventual correctness and the complete states from which a run stays right

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 (left) and the appendix *Additional
experimental details*: "AdamW optimizer with learning rate `10⁻³`, weight decay `1`,
`β₁ = 0.9`, `β₂ = 0.98`, … minibatch size 512"; the paper states no invariant set and no
probability. openai/grok at 3d64b1d8, `grok/training.py` and `grok/data.py`;
`experiments/grok_division/PROTOCOL.md`; `review/grok_division_convergence.md`, section 5.

The question: can an all-correct region of the complete state of the optimizer (parameters,
both moments, the count of updates) be kept under the minibatch sampler and be reached with
positive probability? An epoch of the protocol has `10` updates. At the end of epoch `e` the
split and the complete state (`divisionState`) fix the law of what follows; `divisionStay` is
the probability, over the later shuffles, that every held-out answer is right at every later
update, and a state is robust (`DivisionRobust`) if every sequence of shuffles keeps them
right. For Adam and for AdamW at every weight decay, and every law of the initial parameters:

* the probability of staying right is `1` exactly at the robust states
  (`divisionStay_eq_one_iff`), and reaching a robust state gives eventual correctness
  (`law_divisionRobust_le`);
* for almost every run that is eventually right, the probability of staying right from its
  state at the end of epoch `e` tends to `1` (`division_ae_tendsto_stay`);
* a run is eventually right with positive probability exactly if with positive probability it
  reaches a state from which it stays right with positive probability, and exactly if for
  every `δ < 1` it reaches with positive probability a state from which it stays right with
  probability above `δ` (`division_law_eventuallyRight_ne_zero_iff`,
  `division_law_eventuallyRight_ne_zero_iff_forall`).

So the question is the question whether the event of the target has positive probability, put
in terms of the state: a region kept with probability above `δ` and reached with positive
probability exists for every `δ < 1` if the event has positive probability, and for no `δ` if
it has none. Whether it has is not known here: the tasks of the review stay open. What the
measured runs do is in `experiments/grok_division/README.md`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- The complete state of a division run after `t` updates: parameters and both moments. -/
noncomputable abbrev divisionState (decay : ℝ) (ω : Sample (97 * 96) 4656 (Param paperSizes))
    (t : ℕ) : AdamState (Param paperSizes) :=
  stateAt (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 ω t

/-- **The probability of staying right for division**, from the complete state `s` at the end
of epoch `e` under the split `π`: over fresh shuffles of the later epochs, every held-out
answer is right at every later update. -/
noncomputable abbrev divisionStay (decay : ℝ) (π : Equiv.Perm (Fin (97 * 96))) (e : ℕ)
    (s : AdamState (Param paperSizes)) : ENNReal :=
  stayRight answerLogits (fun x => residueToken x.1) (Hyper.source decay) divisionGrad
    equations (by norm_num : 4656 ≤ 97 * 96) 512 π e s

/-- **A robust state for division**: every sequence of later shuffles keeps every held-out
answer right at every update. -/
abbrev DivisionRobust (decay : ℝ) (π : Equiv.Perm (Fin (97 * 96))) (e : ℕ)
    (s : AdamState (Param paperSizes)) : Prop :=
  Robust answerLogits (fun x => residueToken x.1) (Hyper.source decay) divisionGrad equations
    (by norm_num : 4656 ≤ 97 * 96) 512 π e s

/-- An epoch of the division protocol has ten updates. -/
lemma batchesPerEpoch_division : batchesPerEpoch 4656 512 = 10 := by decide

/-- The event of the target, in terms of the held-out answers. -/
lemma mem_eventuallyRight_iff (decay : ℝ) (ω : Sample (97 * 96) 4656 (Param paperSizes)) :
    ω ∈ eventuallyRight decay ↔ ∀ᶠ t in atTop, AnswersHeldout answerLogits
      (fun x => residueToken x.1) equations 4656 ω.1 (divisionRun decay ω t) := by
  simp only [eventuallyRight, wrongAt, AnswersHeldout, Set.mem_ofPred_eq, not_exists, not_and,
    not_not]

lemma eventuallyRight_eq (decay : ℝ) : eventuallyRight decay = {ω | ∀ᶠ t in atTop,
    AnswersHeldout answerLogits (fun x => residueToken x.1) equations 4656 ω.1
      (divisionRun decay ω t)} :=
  Set.ext (mem_eventuallyRight_iff decay)

variable (decay : ℝ)

/-- **Division: the probability of staying right is `1` exactly at the robust states.** -/
theorem divisionStay_eq_one_iff {π : Equiv.Perm (Fin (97 * 96))} {e : ℕ}
    {s : AdamState (Param paperSizes)} :
    divisionStay decay π e s = 1 ↔ DivisionRobust decay π e s :=
  stayRight_eq_one_iff (by decide)

/-- **Division: the probability that a run is eventually right is at least the probability
that its complete state at the end of some epoch is robust.** -/
theorem law_divisionRobust_le (ν : Measure (Param paperSizes → ℝ)) :
    law (97 * 96) 4656 ν {ω | ∃ e, DivisionRobust decay ω.1 e (divisionState decay ω (10 * e))} ≤
      law (97 * 96) 4656 ν (eventuallyRight decay) := by
  rw [eventuallyRight_eq]
  exact law_robust_le (b := 512) (by decide) ν

variable (ν : Measure (Param paperSizes → ℝ)) [IsProbabilityMeasure ν]

/-- **Division: for almost every run that is eventually right, the probability of staying
right from its complete state at the end of epoch `e` tends to `1`**, for Adam and for AdamW
at every weight decay and every law of the initial parameters. -/
theorem division_ae_tendsto_stay :
    ∀ᵐ ω ∂law (97 * 96) 4656 ν, ω ∈ eventuallyRight decay →
      Tendsto (fun e => divisionStay decay ω.1 e (divisionState decay ω (10 * e))) atTop
        (𝓝 1) := by
  filter_upwards [ae_tendsto_stayRight (logits := answerLogits)
    (label := fun x => residueToken x.1) (h := Hyper.source decay) (eqs := equations)
    (hk := (by norm_num : 4656 ≤ 97 * 96)) (b := 512) measurable_answerLogits
    measurable_divisionGrad (by decide) ν] with ω hω hev
  exact hω ((mem_eventuallyRight_iff decay ω).1 hev)

/-- **Division: a run is eventually right with positive probability exactly if with positive
probability it reaches a complete state from which it stays right with positive
probability.** -/
theorem division_law_eventuallyRight_ne_zero_iff :
    law (97 * 96) 4656 ν (eventuallyRight decay) ≠ 0 ↔ ∃ e, law (97 * 96) 4656 ν
      {ω | divisionStay decay ω.1 e (divisionState decay ω (10 * e)) ≠ 0} ≠ 0 := by
  rw [eventuallyRight_eq]
  refine ⟨fun hpos => ?_, fun ⟨e, he⟩ => law_eventually_ne_zero_of_stayRight (b := 512)
    measurable_answerLogits measurable_divisionGrad (by decide) ν he⟩
  obtain ⟨e, he⟩ := exists_lt_stayRight (b := 512) measurable_answerLogits
    measurable_divisionGrad (by decide) ν hpos (δ := 0) zero_lt_one
  refine ⟨e, fun hzero => he (measure_mono_null (fun ω hω => ?_) hzero)⟩
  exact (show (0 : ENNReal) < _ from hω).ne'

/-- **Division: a run is eventually right with positive probability exactly if for every
`δ < 1` it reaches with positive probability a complete state from which it stays right with
probability above `δ`.** -/
theorem division_law_eventuallyRight_ne_zero_iff_forall :
    law (97 * 96) 4656 ν (eventuallyRight decay) ≠ 0 ↔ ∀ δ : ENNReal, δ < 1 → ∃ e,
      law (97 * 96) 4656 ν
        {ω | δ < divisionStay decay ω.1 e (divisionState decay ω (10 * e))} ≠ 0 := by
  refine ⟨fun hpos δ hδ => ?_, fun hall => ?_⟩
  · rw [eventuallyRight_eq] at hpos
    exact exists_lt_stayRight (b := 512) measurable_answerLogits measurable_divisionGrad
      (by decide) ν hpos hδ
  · obtain ⟨e, he⟩ := hall 0 zero_lt_one
    refine (division_law_eventuallyRight_ne_zero_iff decay ν).2
      ⟨e, fun hzero => he (measure_mono_null (fun ω hω => ?_) hzero)⟩
    exact (show (0 : ENNReal) < _ from hω).ne'

/-! The witnesses: the law of the source's initialization, with the weight decay `1`, and the
law whose initial parameters are zero. -/

example : divisionState 1 zeroSample 0 = ⟨0, 0, 0⟩ := rfl

example (π : Equiv.Perm (Fin (97 * 96))) (s : AdamState (Param paperSizes)) :=
  divisionStay_eq_one_iff 1 (π := π) (e := 3) (s := s)

example := law_divisionRobust_le 1 (initMeasure paperSizes)

example := division_ae_tendsto_stay 1 (initMeasure paperSizes)

example := division_law_eventuallyRight_ne_zero_iff 1 (initMeasure paperSizes)

example := division_law_eventuallyRight_ne_zero_iff_forall 1
  (Measure.dirac (0 : Param paperSizes → ℝ))

end Transformer.Grokking.Protocol
