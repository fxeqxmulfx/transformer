module

public import Transformer.Grokking.FiniteRecovery.ArgandPAGEGaussian
public import Transformer.Grokking.FiniteRecovery.StaircaseDecoder
public import Transformer.Optimization.PAGE.SummableSteps

/-! An infinite, uninterrupted PAGE run on the observed binary cross-entropy.
With a sufficiently small total learning rate there are native Gaussian
starts converging to strictly correct limits, and others converging to
strictly reversed limits, at the same schedule and for every sampler. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.Optimization.PAGE Filter MeasureTheory Set Metric
open scoped Topology NNReal

variable {I : Type*} [Fintype I] {H d L T : ℕ}

noncomputable def observedScheduledPAGE (u : I → Word (ZMod 2) (T+1))
    (f : Word (ZMod 2) (T+1) → ZMod 2) (r : ℕ) (rate : ℕ → ℝ)
    (draw : ComplexParameter (ZMod 2) H d L → ℂ) (path : ℕ → Option I) (t : ℕ) :
    Weights (ZMod 2) H d L := WithLp.ofLp (scheduledRun (observedComponent r u f) rate
      (WithLp.toLp 2 (sgdInitialWeights true draw)) path t).weights

/-- Gaussian starts in a neighborhood of a native strict classifier keep
its decisions even at the infinite weight limit for all small-total schedules. -/
theorem gaussian_page_summable (variance : ComplexParameter (ZMod 2) H d L → ℝ≥0)
    (hv : ∀ i, 0 < variance i) (r : ℕ) (u : I → Word (ZMod 2) (T+1))
    (f g : Word (ZMod 2) (T+1) → ZMod 2) (draw : ComplexParameter (ZMod 2) H d L → ℂ)
    (hw : (Model.fromWeights (sgdInitialWeights true draw)).realizes r g) :
    ∃ δ > 0, ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ z ∈ U, ∀ rate : ℕ → ℝ, Summable (fun t => |rate t|) → (∑' t, |rate t|) ≤ δ →
        ∀ path : ℕ → Option I, let w := observedScheduledPAGE u f r rate z path
          (∀ t, (Model.fromWeights (w t)).realizes r g) ∧
            ∃ star, Tendsto w atTop (𝓝 star) ∧ (Model.fromWeights star).realizes r g := by
  let C := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter (ZMod 2) H d L => ℝ)
  let init := fun z => C.symm (sgdInitialWeights true z)
  have hi : Continuous init := C.symm.continuous.comp (continuous_sgdInitialWeights true)
  let G := {w : WeightSpace (ZMod 2) H d L | (Model.fromWeights (C w)).realizes r g}
  have hG : IsOpen G := (isOpen_realizingWeights r g).preimage C.continuous
  obtain ⟨ρ,hρ,δ,hδ,h⟩ := stays_open_summable (observedComponent (H := H) (d := d) (layers := L) r u f)
    (fun _ => locallyLipschitz_fullGradient _ _ _) hG (c := init draw) hw
  let U := init ⁻¹' ball (init draw) ρ
  have hU : IsOpen U := isOpen_ball.preimage hi
  let := independentGaussianDraws_openPos variance hv
  refine ⟨δ,hδ,U,hU,hU.measure_pos _ ⟨draw, mem_ball_self hρ⟩,?_⟩
  intro z hz rate hs hb path
  obtain ⟨hall,star,hstar,ht⟩ := h (init z) hz rate hs hb path
  exact ⟨hall,C star,(C.continuous.tendsto star).comp ht,hstar⟩

/-- One nonzero rate budget works for both opposite limits. Training reads
exactly the staircase, even on runs whose limiting classifier reverses parity. -/
theorem modularSum_page_summable {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (hk : (k+1)+(k+1) ≤ 32) (variance : ComplexParameter (ZMod 2) 4 16 6 → ℝ≥0)
    (hv : ∀ i, 0 < variance i) :
    ∃ δ > 0, ∀ b : ZMod 2, ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ z ∈ U, ∀ rate : ℕ → ℝ, Summable (fun t => |rate t|) → (∑' t, |rate t|) ≤ δ →
        ∀ path : ℕ → Option (Fin (s+1) ⊕ Fin s),
          let w := observedScheduledPAGE (T := (k+1)+k) (staircaseWord e) (modularSum e) 1 rate z path
          let g := fun x => modularSum e x + b
          (∀ t, (Model.fromWeights (w t)).realizes (T := (k+1)+k) 1 g) ∧
            ∃ star, Tendsto w atTop (𝓝 star) ∧ (Model.fromWeights star).realizes (T := (k+1)+k) 1 g := by
  choose d hd using modularSum_initial_or_complement e hk
  choose δ hδ U hU hp hrun using fun b => gaussian_page_summable (T := (k+1)+k) variance hv 1
    (staircaseWord e) (modularSum e) (fun x => modularSum e x+b) (d b) (hd b)
  obtain ⟨a,ha,hle⟩ := exists_pos_le_all δ hδ
  exact ⟨a,ha,fun b => ⟨U b,hU b,hp b,fun z hz rate hs hb =>
    hrun b z hz rate hs (hb.trans (hle b))⟩⟩

/-- A strictly positive schedule with any prescribed positive total rate.
Together with the preceding theorem this gives a failure event over an
infinite run on which an update is taken at every time. -/
theorem geometric_rate_budget {a : ℝ} (ha : 0 < a) :
    let rate := fun t : ℕ => (a/2) * (1/2 : ℝ)^t
    (∀ t, 0 < rate t) ∧ HasSum (fun t => |rate t|) a := by
  have hp (t : ℕ) : 0 < a/2 * (1/2 : ℝ)^t := by positivity
  refine ⟨hp,?_⟩
  simp only [abs_of_pos (hp _)]
  simpa using hasSum_geometric_two.mul_left (a/2)

example := modularSum_page_summable (s := 15) (wordEnumeration 2 4) (by decide)
  (fun _ => (1/50 : ℝ≥0)^2) (fun _ => by norm_num)
example := modularSum_page_summable (s := 255) (wordEnumeration 2 8) (by decide)
  (fun _ => (1/50 : ℝ≥0)^2) (fun _ => by norm_num)
example := modularSum_page_summable (s := 65535) (wordEnumeration 2 16) (by decide)
  (fun _ => (1/50 : ℝ≥0)^2) (fun _ => by norm_num)

end Transformer.Grokking.FiniteRecovery.Argand
