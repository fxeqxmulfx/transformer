module

public import Transformer.Grokking.FiniteRecovery.ArgandObservedDrift
public import Transformer.Grokking.FiniteRecovery.ArgandBatchDifferentiability

/-! A differentiable upper bound on structural risk. Evaluating it costs
one softmax row per input. Its reference table is a feasible completion
verified against the observations. The minibatch loss stays unchanged. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.ObservedDrift

open MeasureTheory Set Filter ObservedConstraints
open scoped Topology

variable {X A J : Type*} [Fintype X] [Fintype A] [Nonempty A] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)
local notation "W" => Weights A H d L

/-- The first sum uses observed labels. Each structural rule contributes
at most the second sum, evaluated against any feasible completion g. -/
noncomputable def marginalBound (r count : ℕ) (S : Finset X) (y : S → A) (g : X → A)
    (prompt : X → Word A (T+1)) (pos : X → Fin (T+1)) (w : W) : ℝ :=
  let miss x a := 1 - (Model.fromWeights w).probability r (prompt x) (pos x) a
  (∑ x : S, miss x.val (y x)) + count * ∑ x, miss x (g x)

theorem risk_le_marginalBound [MeasurableSpace A] [MeasurableSingletonClass A] [Fintype J]
    (r : ℕ) (S : Finset X) (y : S → A) (g : X → A) (rule : J → Set (X → A))
    (feasible : ∀ j, g ∈ rule j) (fit : ∀ x : S, g x = y x)
    (prompt : X → Word A (T+1)) (pos : X → Fin (T+1)) (w : W) :
    risk (tableLaw (fun x => (Model.fromWeights w).logits r (prompt x) (pos x))) S y rule ≤
      marginalBound r (Fintype.card J) S y g prompt pos w := by
  let score x := (Model.fromWeights w).logits r (prompt x) (pos x)
  have miss (x : X) : (tableLaw score).real {k | k x ≠ g x} =
      1-(Model.fromWeights w).probability r (prompt x) (pos x) (g x) := by
    let E := {k : X → A | k x = g x}
    change (tableLaw score).real Eᶜ = _
    rw [measureReal_compl (measurableSet_eq_fun (measurable_pi_apply x) measurable_const),probReal_univ]
    exact congrArg (fun v : ℝ => 1-v) (table_marginal score x (g x))
  rw [← funext fit]
  simpa only [marginalBound,miss] using risk_le_of_feasible (tableLaw score) S rule g feasible

/-- A feasible completion identified by the observed labels gives a strict
checkpoint certificate from a single softmax evaluation at each input. -/
theorem strict_of_marginalBound [MeasurableSpace A] [MeasurableSingletonClass A] [Fintype J]
    (r : ℕ) (S : Finset X) (g : X → A) (rule : J → Set (X → A)) (feasible : ∀ j, g ∈ rule j)
    (id : InjOn (fun v : X → A => fun x : S => v x) (⋂ j, rule j))
    (prompt : X → Word A (T+1)) (pos : X → Fin (T+1)) (w : W)
    (low : marginalBound r (Fintype.card J) S (fun x => g x) g prompt pos w < 1/2) :
    StrictOn (fun x => (Model.fromWeights w).logits r (prompt x) (pos x)) g := by
  have bound := risk_le_marginalBound r S (fun x => g x) g rule feasible (fun _ => rfl) prompt pos w
  exact strict_of_risk _ S rule id g feasible (bound.trans_lt low)

omit [Nonempty A] in
theorem differentiable_marginalBound (r count : ℕ) (S : Finset X) (y : S → A) (g : X → A)
    (prompt : X → Word A (T+1)) (pos : X → Fin (T+1)) :
    Differentiable ℝ (marginalBound (H := H) (d := d) (L := L) r count S y g prompt pos) := by
  have miss (x : X) (a : A) : Differentiable ℝ (fun w : W =>
      1-(Model.fromWeights w).probability r (prompt x) (pos x) a) := by
    exact (differentiable_const (1 : ℝ)).sub
      ((c1Lipschitz_probability_fromWeights r (prompt x) (pos x) a).contDiff.differentiable one_ne_zero)
  exact (Differentiable.fun_sum fun (x : S) _ => miss x.val (y x)).add
    ((Differentiable.fun_sum fun (x : X) _ => miss x (g x)).const_mul _)

/-- A sufficient condition involving only the Jacobian of the computable
upper bound against the actual parameter step. Observations and a verified
completion suffice for every target in the class compatible with y. -/
theorem three_quarters_of_marginal_jacobian [MeasurableSpace A] [MeasurableSingletonClass A] [Fintype J]
    (μ : Measure Draw) [IsProbabilityMeasure μ] (correct : Bool) (r : ℕ) (S : Finset X)
    (y : S → A) (rule : J → Set (X → A)) (g : X → A) (feasible : ∀ j, g ∈ rule j) (fit : ∀ x : S, g x = y x)
    (id : InjOn (fun v : X → A => fun x : S => v x) (⋂ j, rule j))
    (prompt : X → Word A (T+1)) (pos : X → Fin (T+1)) (batches : Draw → ℕ → List S)
    (rate : ℕ → ℝ) (B : ℕ) {ρ floor : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : floor < 1/2)
    (e : Draw → ℝ) (hm : Measurable e) (hi : Integrable e μ) (hp : ∀ᵐ z ∂μ, 0 ≤ e z)
    (mean : ∫ z, e z ∂μ < (1-ρ)*(1/2-floor)/4) :
    let s (z : Draw) := state correct r (fun t => labeledBatch S y prompt pos (batches z t)) rate z
    let U := marginalBound (H := H) (d := d) (L := L) r (Fintype.card J) S y g prompt pos
    (∀ z t, B ≤ t → ∀ u ∈ Ioo (0:ℝ) 1,
      let w := (s z t).weights
      let Δ := displacement correct (rate t) t (s z t) (s z (t+1))
      fderiv ℝ U (w + u • Δ) Δ ≤ -(1-ρ)*(U w-floor)+e z) →
    ∃ G : Set Draw, MeasurableSet G ∧ 3/4 < μ.real G ∧ ∀ z ∈ G, ∀ᶠ t in atTop,
      ∀ f : X → A, (∀ j, f ∈ rule j) → (∀ x : S, f x = y x) →
        StrictOn (fun x => (Model.fromWeights (s z t).weights).logits r (prompt x) (pos x)) f := by
  intro s U jac
  let score (w : W) x := (Model.fromWeights w).logits r (prompt x) (pos x)
  let Δ z t := displacement correct (rate t) t (s z t) (s z (t+1))
  have hU : Differentiable ℝ U := differentiable_marginalBound r (Fintype.card J) S y g prompt pos
  have major (w : W) : risk (tableLaw (score w)) S (fun x => g x) rule ≤ U w := by
    rw [funext fit]
    exact risk_le_marginalBound r S y g rule feasible fit prompt pos w
  obtain ⟨G,mG,pG,ok⟩ := permanent_of_observed_drift μ score S rule g feasible id U major
    (fun z t => (s z t).weights) Δ (fun z t => weights_step correct r _ rate z t)
    B hρ hlt hr e hm hi hp (q := 3/4) (by linarith only [mean]) (by
      intro z t ht
      let w := (s z t).weights
      have along (u : ℝ) : HasDerivAt (fun v : ℝ => w+v • Δ z t) (Δ z t) u := by
        simpa using ((hasDerivAt_id u).smul_const (Δ z t)).const_add w
      have slope (u : ℝ) : HasDerivAt (segmentRisk U w (Δ z t))
          (fderiv ℝ U (w+u • Δ z t) (Δ z t)) u := (hU _).hasFDerivAt.comp_hasDerivAt u (along u)
      have hd : Differentiable ℝ (segmentRisk U w (Δ z t)) := fun u => (slope u).differentiableAt
      exact ⟨hd.continuous.continuousOn,hd.differentiableOn,fun u hu => (slope u).deriv ▸ jac z t ht u hu⟩)
  refine ⟨G,mG,pG,fun z hz => ?_⟩
  filter_upwards [ok z hz] with t win
  intro f hf hy
  have he : f = g := id (mem_iInter.mpr hf) (mem_iInter.mpr feasible) ((funext hy).trans (funext fit).symm)
  exact he ▸ win

end Transformer.Grokking.FiniteRecovery.Argand.ObservedDrift
