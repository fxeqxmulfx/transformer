module

public import Transformer.Grokking.FiniteRecovery.ArgandFirstNoise

/-! Sampling the native initialization again on each trial. The native
draw may have any probability law. A discrete seed can additionally drive
the order of every minibatch. Only the first-update Gaussian perturbation
is drawn independently of that seed. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory ProbabilityTheory Set
open scoped NNReal

private theorem positive_product_of_sections {D E : Type*} [MeasurableSpace D] [MeasurableSpace E]
    (π : Measure D) [IsProbabilityMeasure π] (μ : Measure E) [IsProbabilityMeasure μ]
    (U : Set (D × E)) (hu : MeasurableSet U) (hp : ∀ d, 0 < μ (Prod.mk d ⁻¹' U)) :
    0 < π.prod μ U := by
  rw [Measure.prod_apply hu,lintegral_pos_iff_support (measurable_measure_prodMk_left hu)]
  have h : Function.support (fun d => μ (Prod.mk d ⁻¹' U)) = univ := eq_univ_of_forall (fun d => (hp d).ne')
  simp [h]

variable {A X D : Type*} [Fintype A] [Finite X] {H d L T : ℕ}

/-- Any law of the native raw tensors, including the native Gaussian law,
can be independently redrawn on each restart. This keeps the exact
single-run mass and the geometric finite-restart miss probability. -/
theorem firstNoise_randomStart_restarts (π : Measure (ComplexParameter A H d L → ℂ)) [IsProbabilityMeasure π]
    (r n : ℕ) (rate : ℕ → ℝ) (hr : ∀ t < n+1, 0 ≤ rate t ∧ rate t < 10)
    (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (hrep : ∃ w : Weights A H d L, StrictOn (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) f)
    (v : ℝ≥0) (hv : 0 < v) :
    let P := π.prod (Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v))
    let scores (z : (ComplexParameter A H d L → ℂ) × Weights A H d L) :=
      firstNoiseScores r (n+1) rate batch prompt z.1 z.2
    let ok z := classCertificate S family (fun x => f x) (scores z)
    let U := {z | StrictOn (scores z) f}
    0 < P U ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => P) {z | retained ok z m = none} = (1-P U)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => P), ∃ j, StrictOn (scores (z j)) f ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  intro P scores ok U
  have ho : IsOpen U := (strict_prompt_weights_open r prompt f).preimage
    ((AdamW.continuous_firstNoise_joint rate (fun t => batchGradient r (batch t))
      (fun t => continuous_batchGradient r (batch t)) n).comp
      (((continuous_sgdInitialWeights true).comp continuous_fst).prodMk continuous_snd))
  have hp : 0 < P U := positive_product_of_sections π _ U ho.measurableSet (fun d =>
    (firstNoise_strict_mass r n rate hr batch prompt f d hrep v hv).2)
  exact ⟨hp,certified_restarts P S family hid f hf scores ho.measurableSet hp⟩

/-- A seed may determine both the initial tensors and a different minibatch
stream for each trial. No support assumption is imposed on the seed law. -/
theorem firstNoise_seeded_restarts [MeasurableSpace D] [MeasurableSingletonClass D] [Countable D]
    (π : Measure D) [IsProbabilityMeasure π] (r n : ℕ) (rate : ℕ → ℝ)
    (hr : ∀ t < n+1, 0 ≤ rate t ∧ rate t < 10) (batch : D → ℕ → List (LabeledPosition A))
    (draw : D → ComplexParameter A H d L → ℂ) (prompt : X → Word A (T+1)) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (hrep : ∃ w : Weights A H d L, StrictOn (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) f)
    (v : ℝ≥0) (hv : 0 < v) :
    let P := π.prod (Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v))
    let scores (z : D × Weights A H d L) := firstNoiseScores r (n+1) rate (batch z.1) prompt (draw z.1) z.2
    let ok z := classCertificate S family (fun x => f x) (scores z)
    let U := {z | StrictOn (scores z) f}
    0 < P U ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => P) {z | retained ok z m = none} = (1-P U)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => P), ∃ j, StrictOn (scores (z j)) f ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  intro P scores ok U
  have section_ok d := firstNoise_strict_mass r n rate hr (batch d) prompt f (draw d) hrep v hv
  let V d := {z | StrictOn (firstNoiseScores r (n+1) rate (batch d) prompt (draw d) z) f}
  have he : U = ⋃ d, {d} ×ˢ V d := by
    ext ⟨d,z⟩
    simp [U,scores,V]
  have hu : MeasurableSet U := by
    rw [he]
    exact MeasurableSet.iUnion (fun d => (measurableSet_singleton d).prod (section_ok d).1.measurableSet)
  have hp : 0 < P U := positive_product_of_sections π _ U hu (fun d => (section_ok d).2)
  exact ⟨hp,certified_restarts P S family hid f hf scores hu hp⟩

end Transformer.Grokking.FiniteRecovery.Argand
