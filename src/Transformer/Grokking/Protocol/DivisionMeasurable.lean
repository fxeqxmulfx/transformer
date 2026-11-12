module

public import Transformer.Grokking.Protocol.AccuracyMeasurable
public import Transformer.Grokking.Protocol.GradientMeasurable
public import Transformer.Grokking.Protocol.ReadoutStationary
public import Transformer.Grokking.Protocol.Refutation

/-!
# For division the held-out accuracy and the wrong answers are measurable

Source: arXiv:2201.02177v1 (Power et al.), Figure 1; openai/grok at 3d64b1d8;
`experiments/grok_division/PROTOCOL.md`; `review/grok_division_convergence.md`, bridge 2.

The logits of the decoder are continuous in the parameters (`Continuity`), so those of the
answer and of the closing token are measurable in them (`measurable_answerLogits`,
`measurable_closingLogits`); the gradient of the backward pass is measurable in them
(`measurable_divisionGrad`, from `GradientMeasurable`). So for the division protocol, under
Adam and under AdamW at every weight decay, after every update: every parameter of the run is
a measurable function of the sample (`measurable_divisionRun`); so is the held-out accuracy,
of the answer and of the closing token (`measurable_divisionAccuracy`,
`measurable_closingAccuracy`); and the samples with a wrong held-out answer form an event
(`measurableSet_wrongAt`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- The logits of the answer are measurable in the parameters. -/
theorem measurable_answerLogits (x : Equation) (c : Fin 239) :
    Measurable fun θ : Param paperSizes → ℝ => answerLogits θ x c :=
  (continuous_logits_relu (S := paperSizes) (read x) 4 c).measurable

/-- The logits of the closing token are measurable in the parameters. -/
theorem measurable_closingLogits (x : Equation) (c : Fin 239) :
    Measurable fun θ : Param paperSizes → ℝ => closingLogits θ x c :=
  (continuous_logits_relu (S := paperSizes) (read x) 5 c).measurable

/-- The gradient of an update is measurable in the parameters. -/
theorem measurable_divisionGrad (B : List Equation) (p : Param paperSizes) :
    Measurable fun θ : Param paperSizes → ℝ => divisionGrad B θ p :=
  measurable_sourceGrad (B.map supervised) p

/-- **Every parameter of a division run is a measurable function of the sample**, after every
update, for Adam and for AdamW at every weight decay. -/
theorem measurable_divisionRun (decay : ℝ) (t : ℕ) (p : Param paperSizes) :
    Measurable fun ω : Sample (97 * 96) 4656 (Param paperSizes) => divisionRun decay ω t p :=
  measurable_trajectory (Hyper.source decay) measurable_divisionGrad equations _ 512 t p

/-- The held-out accuracy of the answer after `t` updates of the division protocol. -/
noncomputable abbrev divisionAccuracy (decay : ℝ) (t : ℕ)
    (ω : Sample (97 * 96) 4656 (Param paperSizes)) : ℝ :=
  heldoutAccuracy answerLogits (fun x => residueToken x.1) (Hyper.source decay) divisionGrad
    equations (by norm_num : 4656 ≤ 97 * 96) 512 t ω

/-- **The held-out accuracy of the answer is a measurable function of the sample**, after
every update. -/
theorem measurable_divisionAccuracy (decay : ℝ) (t : ℕ) :
    Measurable (divisionAccuracy decay t) :=
  measurable_heldoutAccuracy measurable_answerLogits (fun x => residueToken x.1)
    (Hyper.source decay) measurable_divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 t

/-- **The held-out accuracy of the closing token is a measurable function of the sample**,
after every update. -/
theorem measurable_closingAccuracy (decay : ℝ) (t : ℕ) :
    Measurable (heldoutAccuracy closingLogits (fun _ => eosToken) (Hyper.source decay)
      divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 t) :=
  measurable_heldoutAccuracy measurable_closingLogits (fun _ => eosToken)
    (Hyper.source decay) measurable_divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 t

/-- **The samples with a wrong held-out answer after `t` updates form an event.** -/
theorem measurableSet_wrongAt (decay : ℝ) (t : ℕ) : MeasurableSet (wrongAt decay t) := by
  have hset : wrongAt decay t = ⋃ x : Equation,
      (Prod.fst ⁻¹' {π : Equiv.Perm (Fin (97 * 96)) | x ∈ heldout equations 4656 π}) ∩
        {ω : Sample (97 * 96) 4656 (Param paperSizes) |
          predict (answerLogits (divisionRun decay ω t) x) = residueToken x.1}ᶜ := by
    ext ω
    simp [wrongAt]
  rw [hset]
  have hrun : Measurable fun ω : Sample (97 * 96) 4656 (Param paperSizes) =>
      divisionRun decay ω t := measurable_pi_iff.2 fun p => measurable_divisionRun decay t p
  exact MeasurableSet.iUnion fun x => (measurable_fst MeasurableSpace.measurableSet_top).inter
    (measurableSet_predict_eq (fun c => (measurable_answerLogits x c).comp hrun) _).compl

/-- **The mean held-out accuracy is the integral of a bounded measurable function**: it is
integrable for the law of a run. -/
theorem integrable_divisionAccuracy (decay : ℝ) (t : ℕ)
    (μ : Measure (Sample (97 * 96) 4656 (Param paperSizes))) [IsFiniteMeasure μ] :
    Integrable (divisionAccuracy decay t) μ :=
  Integrable.of_bound (measurable_divisionAccuracy decay t).aestronglyMeasurable 1
    (ae_of_all _ fun ω => by
      have h := heldoutAccuracy_mem answerLogits (fun x => residueToken x.1) (Hyper.source decay)
        divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 t ω
      rw [Real.norm_eq_abs, abs_of_nonneg h.1]
      exact h.2)

/-! The witnesses: the law of a run of the protocol, for which the accuracy is integrable; and
the probability that a held-out answer is wrong after `t` updates and the probability that
none is, which add up to `1` because the first is that of an event. -/

example (decay : ℝ) (t : ℕ) :
    Integrable (divisionAccuracy decay t) (law (97 * 96) 4656 (initMeasure paperSizes)) :=
  integrable_divisionAccuracy decay t _

example (decay : ℝ) (t : ℕ) :
    law (97 * 96) 4656 (initMeasure paperSizes) (wrongAt decay t) +
      law (97 * 96) 4656 (initMeasure paperSizes) (wrongAt decay t)ᶜ = 1 := by
  rw [measure_add_measure_compl (measurableSet_wrongAt decay t), measure_univ]

end Transformer.Grokking.Protocol
