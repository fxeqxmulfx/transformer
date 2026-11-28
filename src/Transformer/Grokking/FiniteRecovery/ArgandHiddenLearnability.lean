module

public import Transformer.Grokking.FiniteRecovery.HiddenLearnability
public import Transformer.Grokking.FiniteRecovery.ArgandObservedJacobian

/-! # Learning hidden answers with ordinary AdamW on the Argand model

The necessary conditions of `HiddenLearnability` for ordinary
bias-corrected AdamW in real arithmetic: label-only dependence, and the
seen-answer condition that the 3×3 phase table fails and modular division
satisfies. The sufficient side is `division_three_quarters`, the conditional
theorem `three_quarters_of_marginal_jacobian` on a/b mod 97 for every member
of the class; its Jacobian bound and mean excess remain hypotheses. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set MeasureTheory Filter
open scoped NNReal

variable {X A : Type*} [Fintype A] {H d L T : ℕ}

/-- **Identifiability for AdamW.** Ordinary AdamW trains on minibatches of
observed inputs labeled by `y`, in any fixed order. Two targets with the
labels `y` that differ have hidden success probabilities summing to at most
one under every law of the draw. -/
theorem adamW_shared_labels [Finite X] (μ : Measure (ComplexParameter A H d L → ℂ)) [IsProbabilityMeasure μ]
    (r : ℕ) (S : Finset X) (prompt : X → Word A (T+1)) (rate : ℕ → ℝ) (order : ℕ → List S)
    (y : S → A) {f g : X → A} (hf : ∀ x : S, f x = y x) (hg : ∀ x : S, g x = y x) (hne : f ≠ g) :
    let s (y : S → A) (z : ComplexParameter A H d L → ℂ) t x := (Model.fromWeights (adamWRun r
      (fun t => ObservedDrift.labeledBatch S y prompt (fun _ => Fin.last T) (order t)) rate z t).weights).logits
        r (prompt x) (Fin.last T)
    μ.real {z | HiddenSuccess (S : Set X) (s y z) f} + μ.real {z | HiddenSuccess (S : Set X) (s y z) g} ≤ 1 := by
  intro s
  refine hidden_success_add_le_one μ (S : Set X) s (fun y t k => ?_) y hf hg hne
  have ho := adamW_strict_open (H := H) (d := d) (L := L) r t rate
    (fun t => ObservedDrift.labeledBatch S y prompt (fun _ => Fin.last T) (order t))
    (fun x : {x // x ∉ (S : Set X)} => prompt x.val) (k ∘ Subtype.val)
  convert ho.measurableSet using 1
  ext z
  simp only [mem_ofPred_eq, StrictOn, Subtype.forall, Function.comp_apply, s]

/-- Observing only the input `false` cannot separate the identity from the
constant `false`. -/
example := adamW_shared_labels (H := 1) (d := 1) (L := 1) (T := 0)
  (independentGaussianDraws fun _ => (1 : ℝ≥0)) 1 ({false} : Finset Bool) (fun _ _ => false)
  (fun _ => 0) (fun _ => []) (fun _ => false) (f := id) (g := fun _ => false)
  (fun x => Finset.mem_singleton.mp x.2) (fun _ => rfl) (fun h => absurd (congrFun h true) (by decide))

/-- **Unseen answers cap success at one half.** `X` indexes the hidden
inputs. Ordinary AdamW starts from i.i.d. centered Gaussian native
coordinates and trains on a fixed minibatch sequence. Every prompt and
label token of the minibatches and every hidden prompt token lies in
`seen`. If a hidden answer and another token lie outside it, their swap is
a symmetry of the run, and eventual strict success has probability at most
one half. -/
theorem adamW_unseen_answer_le_half [Finite X] (seen : Set A) (r : ℕ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (f : X → A)
    (trained : ∀ t, ∀ p ∈ batch t, (∀ i, p.2.1 i ∈ seen) ∧ p.2.2.2 ∈ seen)
    (shown : ∀ x i, prompt x i ∈ seen) {x : X} {b : A} (hx : f x ∉ seen) (hb : b ∉ seen)
    (hne : b ≠ f x) (variance : ℝ≥0) :
    (independentGaussianDraws (fun _ : ComplexParameter A H d L => variance)).real {z | ∀ᶠ t in atTop,
      StrictOn (fun x => (Model.fromWeights (adamWRun r batch rate z t).weights).logits r
        (prompt x) (Fin.last T)) f} ≤ 1/2 := by
  let G := fixingSubgroup (Equiv.Perm A) seen
  have fix (e : G) {a : A} (ha : a ∈ seen) : e.val a = a :=
    (mem_fixingSubgroup_iff (Equiv.Perm A)).mp e.property a ha
  have batches (e : G) (t : ℕ) : (batch t).map (relabelPosition e.val) = batch t := by
    conv_rhs => rw [← List.map_id (batch t)]
    refine List.map_congr_left fun p hp => ?_
    obtain ⟨n, w, i, a⟩ := p
    obtain ⟨hw, ha⟩ : (∀ j, w j ∈ seen) ∧ a ∈ seen := trained t _ hp
    change (⟨n, e.val ∘ w, i, e.val a⟩ : LabeledPosition A) = ⟨n, w, i, a⟩
    rw [fix e ha, show e.val ∘ w = w from funext fun j => fix e (hw j)]
  obtain ⟨-, bound⟩ := normalizedAdam_hidden_orbit (H := H) (d := d) (L := L) G r rate batch prompt f
    batches (fun e x => funext fun i => fix e (shown x i)) variance
  simp only [normalizedAdam_weights] at bound
  exact bound.trans (one_div_le_one_div_of_le (by norm_num)
    (by exact_mod_cast orbit_two_le_of_unseen seen f hx hb hne))

open TwoHiddenAnswers in
/-- The 3×3 phase table fails the seen-answer condition. Its minibatches
and prompts use only the eleven tokens of `used`, while the first hidden
probe has answer 2, and token 4 is unused as well. -/
example (stream : ℕ → List (Fin 5)) := adamW_unseen_answer_le_half (H := 4) (d := 16) (L := 6)
    (used : Set Token) 1 (fun k => 3*wsdRate 20000 k) (batch stream) (prompt ∘ probe) (label ∘ probe)
    (by
      intro t p hp
      obtain ⟨j, -, rfl⟩ := List.mem_map.mp hp
      simp only [Finset.mem_coe]
      fin_cases j <;> decide)
    (by simp only [Finset.mem_coe]; decide) (x := 0) (b := 4)
    (by simp only [Finset.mem_coe]; decide) (by simp only [Finset.mem_coe]; decide) (by decide) ((1/50)^2)

open TwoHiddenAnswers in
example : 2 ≤ (hiddenLabelOrbit (fixingSubgroup (Equiv.Perm Token) (used : Set Token)) (label ∘ probe)).card :=
  orbit_two_le_of_unseen _ _ (x := 0) (b := 4) (by simp only [Finset.mem_coe]; decide)
    (by simp only [Finset.mem_coe]; decide) (by decide)

/-- Tokens the observed equations of a division table show: their answers
and every token of their prompts. -/
def divisionSeen (S : Set DivisionInput) (f : DivisionInput → Fin 239) : Set (Fin 239) :=
  {a | ∃ y ∈ S, f y = a ∨ ∃ i, divisionPrompt y i = a}

/-- **Division sees every answer.** If the unit pattern graph of the mask is
connected and one zero-row equation is observed, each answer of each member
of the class is an observed denominator token or the observed zero-row answer. -/
theorem division_answers_seen (S : Set DivisionInput)
    (connected : (FlatSpectrum.patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none, j) ∈ S)
    {f : DivisionInput → Fin 239} (hf : f ∈ factorClass unitToken) (x : DivisionInput) :
    f x ∈ divisionSeen S f := by
  obtain ⟨j, hj⟩ := hz
  obtain ⟨r, c, z, rfl⟩ := hf
  obtain ⟨i, b⟩ := x
  cases i with
  | none => exact ⟨(none, j), hj, Or.inl rfl⟩
  | some i =>
    obtain ⟨v, adj⟩ := connected.exists_adj_of_nontrivial (Sum.inr (r i + c b).toMul)
    cases v with
    | inl k =>
      have observed : (k, (r i + c b).toMul) ∈ unitMask S := FlatSpectrum.patternGraph_adj_inr_inl.mp adj
      exact ⟨(some k, (r i + c b).toMul), observed, Or.inr ⟨3, rfl⟩⟩
    | inr _ => exact False.elim (FlatSpectrum.not_patternGraph_adj_inr_inr adj)

/-- The seen-answer condition holds for modular division: the stabilizer of
the observed tokens fixes the hidden assignment of every member of the class. -/
theorem division_orbit_singleton [Finite X] (S : Set DivisionInput)
    (connected : (FlatSpectrum.patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none, j) ∈ S)
    {f : DivisionInput → Fin 239} (hf : f ∈ factorClass unitToken) (hidden : X → DivisionInput) :
    (hiddenLabelOrbit (fixingSubgroup (Equiv.Perm (Fin 239)) (divisionSeen S f)) (f ∘ hidden)).card = 1 :=
  orbit_singleton_of_seen _ _ fun x => division_answers_seen S connected hz hf (hidden x)

/-- A sparse mask: the anchor row, one cell of every other row and one
zero-row equation. -/
private def sparseRows (i : Fin 97) : ℕ := if i = 1 then 2^96 - 1 else 1

private theorem sparse_graph : (FlatSpectrum.patternGraph (unitMask (bitDivisionMask sparseRows))).Preconnected ∧
    ∃ j, (none, j) ∈ bitDivisionMask sparseRows :=
  bitMask_identifying_graph sparseRows (fun _ => 0) (fun _ => 0) 0 (by decide) (by decide) (by decide)

example := division_orbit_singleton _ sparse_graph.1 sparse_graph.2 divisionAnswer_mem
  (fun x : ((bitDivisionMask sparseRows)ᶜ : Set DivisionInput) => x.val)

example := division_answers_seen _ sparse_graph.1 sparse_graph.2 divisionAnswer_mem

namespace ObservedDrift

/-- **Three quarters for modular division, conditionally.** The full Argand
model (4 heads, width 64, 6 layers, all 239 output tokens) trains in real
arithmetic with ordinary bias-corrected AdamW: clipped gradients of
minibatches of observed equations, both moment buffers, decoupled weight
decay 1/10, and WSD with peak 0.003 over 20,000 updates and the floor
3·10⁻⁸ after them. The draw has any law, on which the minibatch order may
depend. On a mask with a connected unit graph and one zero-row equation,
each member `F` of the class is learned on the hidden inputs with
probability above three quarters if the returned implication's Jacobian
bound on `U` holds along the actual updates from step `B` on. That bound
and `mean` are hypotheses; neither is proved for this model. -/
theorem division_three_quarters (μ : Measure (ComplexParameter (Fin 239) 4 16 6 → ℂ))
    [IsProbabilityMeasure μ] (S : Finset DivisionInput)
    (connected : (FlatSpectrum.patternGraph (unitMask (S : Set DivisionInput))).Preconnected)
    (hz : ∃ j, (none, j) ∈ S) {F : DivisionInput → Fin 239} (hF : F ∈ factorClass unitToken)
    (batches : (ComplexParameter (Fin 239) 4 16 6 → ℂ) → ℕ → List S) (B : ℕ) {ρ floor : ℝ}
    (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : floor < 1/2) (e : (ComplexParameter (Fin 239) 4 16 6 → ℂ) → ℝ)
    (hm : Measurable e) (hi : Integrable e μ) (hp : ∀ᵐ z ∂μ, 0 ≤ e z)
    (mean : ∫ z, e z ∂μ < (1-ρ)*(1/2-floor)/4) :
    let s z := state true 1 (fun t => labeledBatch S (fun x => F x) divisionPrompt (fun _ => Fin.last 4)
      (batches z t)) (fun t => 3*wsdRate 20000 t) z
    let U := marginalBound (H := 4) (d := 16) (L := 6) 1 1 S (fun x => F x) F divisionPrompt (fun _ => Fin.last 4)
    (∀ z t, B ≤ t → ∀ u ∈ Ioo (0:ℝ) 1,
      let w := (s z t).weights
      let Δ := displacement true (3*wsdRate 20000 t) t (s z t) (s z (t+1))
      fderiv ℝ U (w + u • Δ) Δ ≤ -(1-ρ)*(U w-floor)+e z) →
    3/4 < μ.real {z | HiddenSuccess (S : Set DivisionInput) (fun t x =>
      (Model.fromWeights (s z t).weights).logits 1 (divisionPrompt x) (Fin.last 4)) F} := by
  intro s U jac
  have id : InjOn (fun v : DivisionInput → Fin 239 => fun x : S => v x) (⋂ _ : Unit, factorClass unitToken) := by
    rw [iInter_const]
    intro f hf g hg eq
    exact factorClass_identified unitToken _ connected hz hf hg (funext fun x => congrFun eq ⟨x.val, x.property⟩)
  obtain ⟨G, -, pG, ok⟩ := three_quarters_of_marginal_jacobian μ true 1 S (fun x => F x)
    (fun _ : Unit => factorClass unitToken) F (fun _ => hF) (fun _ => rfl) id divisionPrompt
    (fun _ => Fin.last 4) batches (fun t => 3*wsdRate 20000 t) B hρ hlt hr e hm hi hp mean jac
  refine pG.trans_le (measureReal_mono fun z hz => ?_)
  obtain ⟨N, hN⟩ := eventually_atTop.mp (ok z hz)
  exact ⟨N, fun t ht x _ a ha => hN t ht F (fun _ => hF) (fun _ => rfl) x a ha⟩

private noncomputable def sparseSites : Finset DivisionInput := (bitDivisionMask sparseRows).toFinite.toFinset

example (batches : (ComplexParameter (Fin 239) 4 16 6 → ℂ) → ℕ → List sparseSites) :=
  division_three_quarters (independentGaussianDraws fun _ => (1/50 : ℝ≥0)^2) sparseSites
    (by rw [sparseSites, Set.Finite.coe_toFinset]; exact sparse_graph.1)
    (sparse_graph.2.imp fun _ h => (Set.Finite.mem_toFinset _).mpr h) divisionAnswer_mem batches 20000
    (ρ := 0) (floor := 0) le_rfl one_pos (by norm_num) (fun _ => 0) measurable_const (integrable_const 0)
    (ae_of_all _ fun _ => le_rfl) (by norm_num)

end ObservedDrift

end Transformer.Grokking.FiniteRecovery.Argand
