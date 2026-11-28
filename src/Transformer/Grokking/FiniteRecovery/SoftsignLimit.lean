module

public import Transformer.Grokking.FiniteRecovery.EqualMomentLimit
public import Transformer.Grokking.FiniteRecovery.ArgandSoftsign

/-! The ordinary one-moment trajectory has an explicit equilibrium if its
observed clipped gradients converge on the positive WSD floor. The equilibrium
direction has the same functional form as equal-moment Adam, with Lion's
own epsilon. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.SoftsignMomentum

open Filter
open Transformer.AMSGradW.Kernite (clip)
open scoped Topology

variable {ι : Type*} [Fintype ι]

/-- Both the stored average and the direction blend approach the limiting
clipped gradient. No bound on the early rates or on the initial point is used. -/
theorem limit_of_common_gradient {decay η : ℝ} (hd : 0 < decay) (hη : 0 < η) (hc : η*decay ≤ 1)
    (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial G : ι → ℝ)
    (hr : ∀ᶠ t in atTop, rate t = η)
    (hg : Tendsto (fun t => clip (grad t (run decay rate grad initial t).weights)) atTop (𝓝 G)) :
    Tendsto (fun t => (run decay rate grad initial t).weights) atTop (𝓝 fun i => -direction (G i)/decay) := by
  let s := run decay rate grad initial
  let g t := clip (grad t (s t).weights)
  apply DecoupledDecay.tendsto_of_direction_limit (fun t => (s t).weights)
    (fun t i => direction (9/10*(s t).moment i + 1/10*g t i)) _ hd hη hc rate hr (fun _ _ => rfl)
  apply tendsto_pi_nhds.mpr
  intro i
  have hgi : Tendsto (fun t => g t i) atTop (𝓝 (G i)) := tendsto_pi_nhds.mp hg i
  have hm : Tendsto (fun t => (s t).moment i) atTop (𝓝 (G i)) := by
    apply DecoupledDecay.moving_average_limit (q := 99/100) (by norm_num) (by norm_num) hgi
    intro t
    change 99/100*(s t).moment i + 1/100*g t i = _
    ring
  have hc' := (hm.const_mul (9/10 : ℝ)).add (hgi.const_mul (1/10 : ℝ))
  rw [show 9/10*G i + 1/10*G i = G i by ring] at hc'
  exact continuous_direction.continuousAt.tendsto.comp hc'

example (w c : ι → ℝ) := limit_of_common_gradient (decay := 2/3) (η := 3/1000000000)
  (by norm_num) (by norm_num) (by norm_num) (fun _ => 3/1000000000) (fun _ _ => c) w
  (clip c) (Eventually.of_forall fun _ => rfl) tendsto_const_nhds

end Transformer.Grokking.FiniteRecovery.SoftsignMomentum

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Set FlatSpectrum
open Transformer.AMSGradW.Kernite (clip)
open scoped Topology

variable {A X : Type*} [Fintype A] [Finite X] {H d L T : ℕ}

/-- An ordinary positive-floor Lion run eventually generalizes if its
observed gradients have a common limit and the resulting strict winner table
belongs to the identifiable class and agrees on observed labels. -/
theorem softsign_wsd_generalizes (r : ℕ) (scale decay : ℝ) (hs : 0 < scale) (hd : 0 < decay)
    (hcap : scale*decay ≤ 1000) (batch : ℕ → List (LabeledPosition A))
    (draw : ComplexParameter A H d L → ℂ) (prompt : X → Word A (T+1)) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family) (G : Weights A H d L) :
    let rate t := scale*wsdRate 20000 t
    let fit := softsignRun r decay batch rate draw
    let w : Weights A H d L := fun i => -SoftsignMomentum.direction (G i)/decay
    let score := fun (u : Weights A H d L) x => (Model.fromWeights u).logits r (prompt x) (Fin.last T)
    Tendsto (fun t => clip (batchGradient r (batch t) (fit t).weights)) atTop (𝓝 G) →
    (∃ g ∈ family, (∀ x : S, g x = f x) ∧ StrictOn (score w) g) →
    Tendsto (fun t => (fit t).weights) atTop (𝓝 w) ∧ ∀ᶠ t in atTop, StrictOn (score (fit t).weights) f := by
  intro rate fit w score hg cert
  have hw : Tendsto (fun t => (fit t).weights) atTop (𝓝 w) := by
    apply SoftsignMomentum.limit_of_common_gradient (η := scale*(1/100000000)) hd (by positivity)
      (by linarith) rate _ _ G _ hg
    filter_upwards [eventually_ge_atTop (20000-1 : ℕ)] with t ht
    rw [show rate t = scale*wsdRate 20000 t from rfl,wsdRate_eq_floor (by decide) (by decide) ht]
  exact ⟨hw,eventually_strict_of_identified_limit r prompt S family hid f hf _ w hw cert⟩

open Division.Factors

example (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  softsign_wsd_generalizes 1 (3/10) (2/3) (by norm_num) (by norm_num) (by norm_num) batch draw divisionPrompt
    (factorMask (1 : DivisionUnit)) (factorClass unitToken) (factorClass_identified _ _ (by
      rw [unitMask_factorMask]
      exact (diagRow_connected _).preconnected) ⟨1,mem_insert _ _⟩) divisionAnswer divisionAnswer_mem

end Transformer.Grokking.FiniteRecovery.Argand
