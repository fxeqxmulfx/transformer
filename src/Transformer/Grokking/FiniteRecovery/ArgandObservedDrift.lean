module

public import Transformer.Grokking.FiniteRecovery.ObservedDrift
public import Transformer.Grokking.FiniteRecovery.ArgandEqualMoment
public import Transformer.Grokking.FiniteRecovery.TwoHiddenUncorrected

/-! Ordinary corrected and uncorrected AdamW with their actual clipped
observed minibatches. Both updates can be tested against the same
observable risk on one uninterrupted trajectory. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.ObservedDrift

open MeasureTheory Set Filter ObservedConstraints
open scoped Topology NNReal

variable {X A J : Type*} [Fintype A] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)
local notation "W" => Weights A H d L

/-- The selected optimizer retains both native moment buffers. -/
noncomputable def state (correct : Bool) (r : ℕ) (batch : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (z : Draw) (t : ℕ) : AdamW.State (Parameter A H d L) :=
  if correct then adamWRun r batch rate z t else
    uncorrectedAdamRun r (9/10) (19/20) (1/100000000) (1/10) batch rate z t

/-- Actual signed parameter displacement, including decoupled weight decay.
The moments in the direction already include the clipped observed gradient
of the current minibatch. -/
noncomputable def displacement (correct : Bool) (η : ℝ) (t : ℕ)
    (now next : AdamW.State (Parameter A H d L)) : W := fun i =>
  -η*(now.weights i/10 + if correct then AdamW.direction t (next.first i) (next.second i)
    else EqualMoment.direction (1/100000000) (next.first i) (next.second i))

theorem weights_step (correct : Bool) (r : ℕ) (batch : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (z : Draw) (t : ℕ) :
    let s := state correct r batch rate z
    (s (t+1)).weights = (s t).weights + displacement correct (rate t) t (s t) (s (t+1)) := by
  intro s
  funext i
  cases correct with
  | false =>
    let run := uncorrectedAdamRun r (9/10) (19/20) (1/100000000) (1/10) batch rate z
    let dir := EqualMoment.direction (1/100000000) ((run (t+1)).first i) ((run (t+1)).second i)
    change (1-rate t*(1/10))*(run t).weights i - rate t*dir =
      (run t).weights i + -rate t*((run t).weights i/10+dir)
    ring
  | true =>
    let run := adamWRun r batch rate z
    let dir := AdamW.direction t ((run (t+1)).first i) ((run (t+1)).second i)
    change (1-rate t/10)*(run t).weights i - rate t*dir =
      (run t).weights i + -rate t*((run t).weights i/10+dir)
    ring

def labeledBatch (S : Finset X) (y : S → A) (prompt : X → Word A (T+1))
    (pos : X → Fin (T+1)) (xs : List S) : List (LabeledPosition A) :=
  xs.map fun (x : S) => ⟨T,prompt x.val,pos x.val,y x⟩

/-- Strict recovery with probability above three quarters on one ordinary
trajectory. The dynamic hypothesis is a bound on signed risk derivatives
over actual updates. Only observed labels enter the minibatches and risk.
The shuffle may depend on the same draw as the initialized weights. -/
theorem native_three_quarters [Fintype X] [Fintype J] [Nonempty A] [MeasurableSpace A]
    [MeasurableSingletonClass A] (μ : Measure Draw) [IsProbabilityMeasure μ]
    (correct : Bool) (r : ℕ) (S : Finset X) (rule : J → Set (X → A))
    (f : X → A) (hf : ∀ j, f ∈ rule j)
    (id : InjOn (fun g : X → A => fun x : S => g x) (⋂ j, rule j))
    (prompt : X → Word A (T+1)) (pos : X → Fin (T+1)) (batches : Draw → ℕ → List S)
    (rate : ℕ → ℝ) (B : ℕ) {ρ floor : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : floor < 1/2)
    (excess : Draw → ℝ) (hm : Measurable excess) (hi : Integrable excess μ) (hp : ∀ᵐ z ∂μ, 0 ≤ excess z)
    (mean : ∫ z, excess z ∂μ < (1-ρ)*(1/2-floor)/4) :
    let s (z : Draw) := state correct r (fun t => labeledBatch S (fun x => f x) prompt pos (batches z t)) rate z
    let score (w : W) x := (Model.fromWeights w).logits r (prompt x) (pos x)
    let V (w : W) := risk (tableLaw (score w)) S (fun x => f x) rule
    (∀ z t, B ≤ t →
      let start := (s z t).weights
      let Δ := displacement correct (rate t) t (s z t) (s z (t+1))
      let line := segmentRisk V start Δ
      ContinuousOn line (Icc 0 1) ∧ DifferentiableOn ℝ line (Ioo 0 1) ∧
      ∀ u ∈ Ioo 0 1, deriv line u ≤ -(1-ρ)*(V start-floor)+excess z) →
    ∃ G : Set Draw, MeasurableSet G ∧ 3/4 < μ.real G ∧
      ∀ z ∈ G, ∀ᶠ t in atTop, StrictOn (score (s z t).weights) f := by
  intro s score V slope
  exact permanent_of_observed_drift μ score S rule f hf id V (fun _ => le_rfl) (fun z t => (s z t).weights)
    (fun z t => displacement correct (rate t) t (s z t) (s z (t+1)))
    (fun z t => weights_step correct r (fun j => labeledBatch S (fun x => f x) prompt pos (batches z j)) rate z t)
    B hρ hlt hr excess hm hi hp (by linarith only [mean]) slope

/-- The five-observation example excludes a high-probability permanent
low-risk regime for every collection of identifiable rules containing
its target, including any presentation of its phase-table class. -/
theorem counterexample_risk_ceiling [Fintype J] (stream : ℕ → List (Fin 5))
    (S : Finset TwoHiddenAnswers.Cell) (rule : J → Set (TwoHiddenAnswers.Cell → TwoHiddenAnswers.Token))
    (hf : ∀ j, TwoHiddenAnswers.label ∈ rule j)
    (id : InjOn (fun g : TwoHiddenAnswers.Cell → TwoHiddenAnswers.Token => fun x : S => g x) (⋂ j, rule j)) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter TwoHiddenAnswers.Token 4 16 6 => (1/50 : ℝ≥0)^2)
    μ.real {z | ∀ᶠ t in atTop, risk (tableLaw (TwoHiddenAnswers.score stream z t)) S
      (fun x => TwoHiddenAnswers.label x) rule < 1/2} ≤ 1/2756 := by
  intro μ
  apply (measureReal_mono (μ := μ) ?_ (measure_ne_top μ _)).trans (TwoHiddenAnswers.hidden_eventually_correct stream)
  intro z hz
  filter_upwards [hz] with t ht
  exact fun x _ => strict_of_risk _ S rule id _ hf ht x

/-- Full native weights satisfying the certificate exist on the very table
where the ordinary dynamics has the 1/2756 permanent-success ceiling. -/
example [Fintype J] (S : Finset TwoHiddenAnswers.Cell) (rule : J → Set (TwoHiddenAnswers.Cell → TwoHiddenAnswers.Token))
    (hf : ∀ j, TwoHiddenAnswers.label ∈ rule j)
    (id : InjOn (fun g : TwoHiddenAnswers.Cell → TwoHiddenAnswers.Token => fun x : S => g x) (⋂ j, rule j)) :
    ∃ (w : Weights TwoHiddenAnswers.Token 4 16 6) (β : ℝ), 0 < β ∧
      risk (tableLaw (fun x a => β*(Model.fromWeights w).logits 1 (TwoHiddenAnswers.prompt x) 0 a))
        S (fun x => TwoHiddenAnswers.label x) rule < 1/2 := by
  obtain ⟨w,win⟩ := TwoHiddenAnswers.representable
  exact ⟨w,(strict_iff_scaled_risk _ S rule id _ hf).mp win⟩

end Transformer.Grokking.FiniteRecovery.Argand.ObservedDrift
