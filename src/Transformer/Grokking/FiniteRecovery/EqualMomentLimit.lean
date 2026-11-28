module

public import Transformer.Grokking.FiniteRecovery.ArgandEqualMoment

/-! An explicit full-model limit for ordinary equal-decay Adam on a positive
WSD floor. Identifiability transfers a structurally valid limit prediction
and its observed labels to all hidden inputs; no perturbed copies are used. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.EqualMoment

open Filter
open Transformer.AMSGradW.Kernite (clip)
open scoped Topology

variable {ι : Type*} [Fintype ι]

/-- On a positive constant tail, a common limiting clipped gradient G gives
the exact equilibrium in every coordinate. The start and minibatch sequence
are arbitrary. Changing epsilon or decay changes the equilibrium. -/
theorem uncorrected_limit_of_common_gradient {β γ ε decay η : ℝ} (hβ : 0 ≤ β) (hβ' : β < 1)
    (hγ : 0 ≤ γ) (hγ' : γ < 1) (hε : 0 < ε)
    (hd : 0 < decay) (hη : 0 < η) (hc : η*decay ≤ 1)
    (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial G : ι → ℝ)
    (hr : ∀ᶠ t in atTop, rate t = η)
    (hg : Tendsto (fun t => clip (grad t (uncorrectedRun β γ ε decay rate grad initial t).weights)) atTop (𝓝 G)) :
    Tendsto (fun t => (uncorrectedRun β γ ε decay rate grad initial t).weights) atTop
      (𝓝 (fun i => -(G i / (|G i| + ε)) / decay)) := by
  let s := uncorrectedRun β γ ε decay rate grad initial
  let g t := clip (grad t (s t).weights)
  apply DecoupledDecay.tendsto_of_direction_limit (fun t => (s t).weights)
    (fun t i => direction ε ((s (t+1)).first i) ((s (t+1)).second i)) _ hd hη hc rate hr (fun _ _ => rfl)
  apply tendsto_pi_nhds.mpr
  intro i
  have hgi : Tendsto (fun t => g t i) atTop (𝓝 (G i)) := tendsto_pi_nhds.mp hg i
  have hm : Tendsto (fun t => (s t).first i) atTop (𝓝 (G i)) :=
    DecoupledDecay.moving_average_limit hβ hβ' hgi (fun _ => rfl)
  have hv : Tendsto (fun t => (s t).second i) atTop (𝓝 ((G i)^2)) :=
    DecoupledDecay.moving_average_limit hγ hγ' (hgi.pow 2) (fun _ => rfl)
  have hden := (Real.continuous_sqrt.continuousAt.tendsto.comp ((tendsto_add_atTop_iff_nat 1).2 hv)).add_const ε
  rw [Real.sqrt_sq_eq_abs] at hden
  exact ((tendsto_add_atTop_iff_nat 1).2 hm).div hden (add_pos_of_nonneg_of_pos (abs_nonneg _) hε).ne'

theorem limit_of_common_gradient {β ε decay η : ℝ} (hβ : 0 ≤ β) (hβ' : β < 1) (hε : 0 < ε)
    (hd : 0 < decay) (hη : 0 < η) (hc : η*decay ≤ 1) (rate : ℕ → ℝ)
    (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial G : ι → ℝ) (hr : ∀ᶠ t in atTop, rate t = η)
    (hg : Tendsto (fun t => clip (grad t (run β ε decay rate grad initial t).weights)) atTop (𝓝 G)) :
    Tendsto (fun t => (run β ε decay rate grad initial t).weights) atTop
      (𝓝 (fun i => -(G i / (|G i| + ε)) / decay)) :=
  uncorrected_limit_of_common_gradient hβ hβ' hβ hβ' hε hd hη hc rate grad initial G hr hg

example (w c : ι → ℝ) := limit_of_common_gradient (β := 9/10) (ε := 1/100000000)
  (decay := 1/10) (η := 1/50000000) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (fun _ => 1/50000000) (fun _ _ => c)
  w (clip c) (Eventually.of_forall fun _ => rfl) tendsto_const_nhds

end Transformer.Grokking.FiniteRecovery.EqualMoment

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Set FlatSpectrum
open Transformer.AMSGradW.Kernite (clip)
open scoped Topology

variable {A X : Type*} [Fintype A] [Finite X] {H d L T : ℕ}

/-- Any convergent weight path inherits the correct strict predictions from
a limit whose winner table is structurally admissible and fits the observed
labels. The optimizer and the hidden labels do not enter the certificate. -/
theorem eventually_strict_of_identified_limit (r : ℕ) (prompt : X → Word A (T+1)) (S : Set X)
    (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (path : ℕ → Weights A H d L) (w : Weights A H d L)
    (hw : Tendsto path atTop (𝓝 w)) :
    let score := fun (u : Weights A H d L) x => (Model.fromWeights u).logits r (prompt x) (Fin.last T)
    (∃ g ∈ family, (∀ x : S, g x = f x) ∧ StrictOn (score w) g) →
      ∀ᶠ t in atTop, StrictOn (score (path t)) f := by
  intro score cert
  have strict := (classCertificate_iff S family hid f hf (score w)).1 cert
  exact hw.eventually ((strict_prompt_weights_open r prompt f).mem_nhds strict)

/-- For an identifiable class, the limiting winner on each unlabeled input
need only be strict and the table of winners must belong to the prior class
and agree on the observations. Combined with the common-gradient condition,
this gives eventual correctness of every ordinary iterate on the whole task. -/
theorem equalMoment_wsd_generalizes (r : ℕ) (scale : ℝ) (hs : 0 < scale) (hcap : scale ≤ 10000)
    (batch : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d L → ℂ)
    (prompt : X → Word A (T+1)) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (G : Weights A H d L) :
    let rate t := scale*wsdRate 20000 t
    let fit := equalMomentRun r (1/10) batch rate draw
    let w : Weights A H d L := fun i => -(G i / (|G i|+1/100000000))/(1/10)
    let score := fun (u : Weights A H d L) x => (Model.fromWeights u).logits r (prompt x) (Fin.last T)
    Tendsto (fun t => clip (batchGradient r (batch t) (fit t).weights)) atTop (𝓝 G) →
    (∃ g ∈ family, (∀ x : S, g x = f x) ∧ StrictOn (score w) g) →
    Tendsto (fun t => (fit t).weights) atTop (𝓝 w) ∧
      ∀ᶠ t in atTop, StrictOn (score (fit t).weights) f := by
  intro rate fit w score hg cert
  have hw : Tendsto (fun t => (fit t).weights) atTop (𝓝 w) := by
    apply EqualMoment.limit_of_common_gradient (η := scale*(1/100000000)) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by positivity) (by linarith) rate _ _ G _ hg
    filter_upwards [eventually_ge_atTop (20000-1 : ℕ)] with t ht
    rw [show rate t = scale*wsdRate 20000 t from rfl,wsdRate_eq_floor (by decide) (by decide) ht]
  exact ⟨hw,eventually_strict_of_identified_limit r prompt S family hid f hf _ w hw cert⟩

open Division.Factors

example (s : ℝ) (hs : 0 < s) (hc : s ≤ 10000) (batch : ℕ → List (LabeledPosition (Fin 239)))
    (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  equalMoment_wsd_generalizes 1 s hs hc batch draw divisionPrompt (factorMask (1 : DivisionUnit))
    (factorClass unitToken) (factorClass_identified _ _ (by
      rw [unitMask_factorMask]
      exact (diagRow_connected _).preconnected) ⟨1,mem_insert _ _⟩) divisionAnswer divisionAnswer_mem

end Transformer.Grokking.FiniteRecovery.Argand
