module

public import Transformer.Grokking.Protocol.Target

/-!
# The convergence target is satisfiable and falsifiable

Source: `experiments/grok_division/PROTOCOL.md`; the target is `Generalizes` of
`Transformer.Grokking.Protocol.Target`, for arXiv:2201.02177v1 (Power et al.), Figure 1.

The target is a predicate of a model, its labels and a gradient oracle. Two models over two
equations, of which one is trained on, show that it is neither always true nor always false:

* a model whose logits mark the label of its input, whatever its parameters, generalizes
  (`generalizes_indicator`);
* a model with equal logits, whose prediction is the first class, does not generalize when
  the label is the second class (`not_generalizes_const`).

Both models ignore their parameters, so neither says anything about training. The modes of
convergence are witnessed by an accuracy that is `0` before the update `3` and `1` from it on.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- **A model that marks the label of its input generalizes**, whatever its training. -/
theorem generalizes_indicator (label : Fin 2 → Bool) :
    Generalizes (N := 2) (k := 1) (Measure.dirac (0 : Unit → ℝ))
      (fun _ x c => if c = label x then (1 : ℝ) else 0) label (Hyper.source 0) (fun _ _ _ => 0) id
      (by norm_num) 1 :=
  (generalizes_iff (by norm_num : 1 < 2)).2 <| Eventually.of_forall fun _ =>
    Eventually.of_forall fun _ x _ => predict_indicator_bool (label x)

/-- **A model with equal logits does not generalize** when the label is the second class: its
prediction is the first class at every update of every sample. -/
theorem not_generalizes_const :
    ¬ Generalizes (N := 2) (k := 1) (Measure.dirac (0 : Unit → ℝ))
      (fun _ _ (_ : Bool) => (0 : ℝ)) (fun _ => true) (Hyper.source 0) (fun _ _ _ => 0) id
      (by norm_num) 1 := fun hgen => by
  obtain ⟨ω, hω⟩ := ((generalizes_iff (by norm_num : 1 < 2)).1 hgen).exists
  obtain ⟨t, ht⟩ := hω.exists
  obtain ⟨x, hx⟩ := heldout_nonempty (id : Fin 2 → Fin 2) (by norm_num : 1 < 2) ω.1
  have h := ht x hx
  rw [predict_const_bool] at h
  exact Bool.false_ne_true h

/-- The accuracy that is `0` before the update `3` and `1` from it on, on the one-point space,
tends to `1` almost surely. -/
lemma tendsToOneAlmostSurely_step :
    TendsToOneAlmostSurely (Measure.dirac ())
      fun (t : ℕ) (_ : Unit) => if 3 ≤ t then (1 : ℝ) else 0 :=
  Eventually.of_forall fun _ => tendsto_const_nhds.congr' <| by
    filter_upwards [eventually_ge_atTop 3] with t ht
    rw [ite_eq_left ht]

example : TendsToOneInProbability (Measure.dirac ())
    fun (t : ℕ) (_ : Unit) => if 3 ≤ t then (1 : ℝ) else 0 :=
  tendsToOneAlmostSurely_step.inProbability fun _ => aestronglyMeasurable_const

example : TendsToOneInMean (Measure.dirac ())
    fun (t : ℕ) (_ : Unit) => if 3 ≤ t then (1 : ℝ) else 0 :=
  tendsToOneAlmostSurely_step.inMean (fun _ => aestronglyMeasurable_const) fun t _ => by
    split_ifs <;> norm_num

/-- The held-out accuracy of the model with equal logits is `0` at every update, in `[0, 1]`. -/
example (t : ℕ) (ω : Sample 2 1 Unit) :=
  heldoutAccuracy_mem (fun _ _ (_ : Bool) => (0 : ℝ)) (fun _ => true) (Hyper.source 0)
    (fun _ _ _ => 0) (id : Fin 2 → Fin 2) (by norm_num : 1 ≤ 2) 1 t ω

end Transformer.Grokking.Protocol
