module

public import Transformer.Grokking.Protocol.RunMeasurable
public import Transformer.Grokking.Protocol.Witness

/-!
# The held-out accuracy is a measurable function of the sample

Source: arXiv:2201.02177v1 (Power et al.), Figure 1; `experiments/grok_division/PROTOCOL.md`,
*Randomness*; `review/grok_division_convergence.md`, bridge 2. The target and its modes are
those of `Transformer.Grokking.Protocol.Target`.

A class is predicted exactly when its logit is no smaller than any other and larger than that
of every earlier class, so the samples at which a model with measurable logits predicts a
class form an event (`measurableSet_predict_eq`), and its accuracy on a finite set is a
measurable function (`measurable_accuracy`). The held-out set is a function of the first
shuffle, which takes finitely many values. So for a model whose logits and whose gradient
oracle are measurable in the parameters, the held-out accuracy after every update is a
measurable function of the sample (`measurable_heldoutAccuracy`), and the almost sure target
gives the two weaker modes (`Generalizes.inProbability`, `Generalizes.inMean`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

variable {Ω ι X C : Type*} [MeasurableSpace Ω] [Fintype C] [LinearOrder C] [Nonempty C]
  {N k : ℕ}

/-- **The samples at which a class is predicted form an event**, for measurable logits. -/
theorem measurableSet_predict_eq {z : Ω → C → ℝ} (hz : ∀ c, Measurable fun ω => z ω c) (y : C) :
    MeasurableSet {ω | predict (z ω) = y} := by
  have hset : {ω | predict (z ω) = y} =
      (⋂ c, {ω | z ω c ≤ z ω y}) ∩ ⋂ c, ⋂ (_ : c < y), {ω | z ω c < z ω y} := by
    ext ω
    simp [predict_eq_iff]
  rw [hset]
  exact (MeasurableSet.iInter fun c => measurableSet_le (hz c) (hz y)).inter
    (MeasurableSet.iInter fun c => MeasurableSet.iInter fun _ => measurableSet_lt (hz c) (hz y))

/-- **The accuracy on a finite set is measurable**, for measurable logits. -/
theorem measurable_accuracy {f : Ω → X → C → ℝ} (hf : ∀ x c, Measurable fun ω => f ω x c)
    (y : X → C) (H : Finset X) : Measurable fun ω => accuracy (f ω) y H := by
  have hsum : ∀ ω, accuracy (f ω) y H =
      (∑ x ∈ H, if predict (f ω x) = y x then (1 : ℝ) else 0) / H.card := fun ω => by
    rw [accuracy, Finset.card_filter]
    push_cast
    rfl
  simp only [hsum]
  exact (Finset.measurable_sum H fun x _ => Measurable.ite
    (measurableSet_predict_eq (hf x) (y x)) measurable_const measurable_const).div_const _

variable [DecidableEq X]

/-- **The held-out accuracy after every update is a measurable function of the sample**, for a
model whose logits and whose gradient oracle are measurable in the parameters. -/
theorem measurable_heldoutAccuracy {logits : (ι → ℝ) → X → C → ℝ}
    (hlogits : ∀ x c, Measurable fun θ => logits θ x c) (label : X → C) (h : Hyper)
    {lossGrad : List X → (ι → ℝ) → ι → ℝ} (hL : ∀ B i, Measurable fun θ => lossGrad B θ i)
    (eqs : Fin N → X) (hk : k ≤ N) (b t : ℕ) :
    Measurable (heldoutAccuracy logits label h lossGrad eqs hk b t) := by
  have htraj : Measurable fun ω : Sample N k ι => trajectory h lossGrad eqs hk b ω t :=
    measurable_pi_iff.2 fun i => measurable_trajectory h hL eqs hk b t i
  have hF : Measurable fun x : Equiv.Perm (Fin N) × Sample N k ι =>
      accuracy (logits (trajectory h lossGrad eqs hk b x.2 t)) label (heldout eqs k x.1) :=
    measurable_from_prod_countable_right fun π =>
      measurable_accuracy (fun x c => (hlogits x c).comp htraj) label (heldout eqs k π)
  have h2 := hF.comp (measurable_fst.prodMk measurable_id)
  exact h2

variable {ν : Measure (ι → ℝ)} [IsProbabilityMeasure ν] {logits : (ι → ℝ) → X → C → ℝ}
  {label : X → C} {h : Hyper} {lossGrad : List X → (ι → ℝ) → ι → ℝ} {eqs : Fin N → X}
  {hk : k ≤ N} {b : ℕ}

/-- **The target gives convergence in probability**, for a model whose logits and whose
gradient oracle are measurable in the parameters. -/
theorem Generalizes.inProbability (hlogits : ∀ x c, Measurable fun θ => logits θ x c)
    (hL : ∀ B i, Measurable fun θ => lossGrad B θ i)
    (hgen : Generalizes ν logits label h lossGrad eqs hk b) :
    TendsToOneInProbability (law N k ν) (heldoutAccuracy logits label h lossGrad eqs hk b) :=
  TendsToOneAlmostSurely.inProbability (fun t =>
    (measurable_heldoutAccuracy hlogits label h hL eqs hk b t).aestronglyMeasurable) hgen

/-- **The target gives convergence of the mean accuracy**, for a model whose logits and whose
gradient oracle are measurable in the parameters. -/
theorem Generalizes.inMean (hlogits : ∀ x c, Measurable fun θ => logits θ x c)
    (hL : ∀ B i, Measurable fun θ => lossGrad B θ i)
    (hgen : Generalizes ν logits label h lossGrad eqs hk b) :
    TendsToOneInMean (law N k ν) (heldoutAccuracy logits label h lossGrad eqs hk b) :=
  TendsToOneAlmostSurely.inMean (fun t =>
    (measurable_heldoutAccuracy hlogits label h hL eqs hk b t).aestronglyMeasurable)
    (fun t ω => heldoutAccuracy_mem logits label h lossGrad eqs hk b t ω) hgen

/-! The witnesses: the event that the second of two real logits is predicted; the accuracy of
that model on two inputs; and the model of `Witness` that marks the label of its input, whose
held-out accuracy is measurable and tends to `1` in probability and in mean. -/

example : MeasurableSet {a : ℝ | predict (fun c : Bool => if c then a else 0) = true} :=
  measurableSet_predict_eq (fun c => by cases c <;> fun_prop) true

example : Measurable fun a : ℝ =>
    accuracy (fun (x : Fin 2) (c : Bool) => if c then a + x else 0) (fun _ => true) {0, 1} :=
  measurable_accuracy (fun x c => by cases c <;> fun_prop) _ _

example (label : Fin 2 → Bool) (t : ℕ) :
    Measurable (heldoutAccuracy (N := 2) (k := 1) (ι := Unit)
      (fun _ x c => if c = label x then (1 : ℝ) else 0) label (Hyper.source 0) (fun _ _ _ => 0) id
      (by norm_num) 1 t) :=
  measurable_heldoutAccuracy (ι := Unit)
    (logits := fun _ x c => if c = label x then (1 : ℝ) else 0) (fun _ _ => measurable_const) label
    (Hyper.source 0) (lossGrad := fun _ _ _ => 0) (fun _ _ => measurable_const) id
    (by norm_num : 1 ≤ 2) 1 t

example (label : Fin 2 → Bool) :
    TendsToOneInProbability (law 2 1 (Measure.dirac (0 : Unit → ℝ)))
      (heldoutAccuracy (fun _ x c => if c = label x then (1 : ℝ) else 0) label (Hyper.source 0)
        (fun _ _ _ => 0) id (by norm_num) 1) :=
  (generalizes_indicator label).inProbability (fun _ _ => measurable_const)
    fun _ _ => measurable_const

example (label : Fin 2 → Bool) :
    TendsToOneInMean (law 2 1 (Measure.dirac (0 : Unit → ℝ)))
      (heldoutAccuracy (fun _ x c => if c = label x then (1 : ℝ) else 0) label (Hyper.source 0)
        (fun _ _ _ => 0) id (by norm_num) 1) :=
  (generalizes_indicator label).inMean (fun _ _ => measurable_const)
    fun _ _ => measurable_const

end Transformer.Grokking.Protocol
