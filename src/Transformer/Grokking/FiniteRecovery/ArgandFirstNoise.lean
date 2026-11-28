module

public import Transformer.Grokking.FiniteRecovery.AdamWFirstNoise

/-! Certified generalization at the original AdamW rates with a single
independent weight perturbation. The first gradient uses the ordinary
Argand initialization. A strictly representable target in any identifiable
class has positive single-run mass. Independent restarts eventually keep
a strictly correct model using only the observed answers to certify it. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal Topology

variable {A X : Type*} [Fintype A] {H d L T : ℕ}

noncomputable def firstNoiseScores (r K : ℕ) (rate : ℕ → ℝ) (batch : ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (draw : ComplexParameter A H d L → ℂ) (noise : Weights A H d L) :
    X → A → ℝ :=
  let w := (AdamW.firstNoiseRun rate (fun t => batchGradient r (batch t))
    (sgdInitialWeights true draw) noise K).weights
  fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)

variable [Finite X]

/-- A single-trial guarantee for every fixed standard initialization.
Representability is required only in the full weight space. The proof
does not require a stock training trajectory that already generalizes. -/
theorem firstNoise_strict_mass (r n : ℕ) (rate : ℕ → ℝ)
    (hr : ∀ t < n+1, 0 ≤ rate t ∧ rate t < 10) (batch : ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) (draw : ComplexParameter A H d L → ℂ)
    (hrep : ∃ w : Weights A H d L, StrictOn (fun x => (Model.fromWeights w).logits r
      (prompt x) (Fin.last T)) f) (v : ℝ≥0) (hv : 0 < v) :
    let U := {z | StrictOn (firstNoiseScores r (n+1) rate batch prompt draw z) f}
    IsOpen U ∧ 0 < Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v) U := by
  let g (t : ℕ) := batchGradient (H := H) (d := d) (layers := L) r (batch t)
  have hg t : Continuous (g t) := continuous_batchGradient r (batch t)
  have ho := strict_prompt_weights_open (H := H) (d := d) (L := L) r prompt f
  exact ⟨ho.preimage (AdamW.continuous_firstNoise_weights rate g hg _ n),
    AdamW.firstNoise_open_probability rate g hg _ n hr v hv _ ho hrep⟩

/-- Exact finite-restart miss probability and almost-sure stabilization
for arbitrary native draws, step counts, architectures and identifiable
finite tasks. The observation mask is the verifier's only source of labels. -/
theorem firstNoise_certified_restarts (r n : ℕ) (rate : ℕ → ℝ)
    (hr : ∀ t < n+1, 0 ≤ rate t ∧ rate t < 10) (batch : ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (draw : ComplexParameter A H d L → ℂ) (hrep : ∃ w : Weights A H d L, StrictOn
      (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) f) (v : ℝ≥0) (hv : 0 < v) :
    let μ := Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v)
    let scores := firstNoiseScores r (n+1) rate batch prompt draw
    let ok z := classCertificate S family (fun x => f x) (scores z)
    let p := μ {z | StrictOn (scores z) f}
    0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {z | retained ok z m = none} = (1-p)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) f ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  obtain ⟨ho,hp⟩ := firstNoise_strict_mass r n rate hr batch prompt f draw hrep v hv
  exact ⟨hp,certified_restarts _ S family hid f hf _ ho.measurableSet hp⟩

/-- At each strictly successful stock endpoint, all sufficiently small
first-update perturbations preserve every strict answer at the same step. -/
theorem firstNoise_preserves_strict_near_zero (r n : ℕ) (rate : ℕ → ℝ) (batch : ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) (draw : ComplexParameter A H d L → ℂ)
    (hstock : StrictOn (fun x => (Model.fromWeights (adamWRun r batch rate draw (n+1)).weights).logits r
      (prompt x) (Fin.last T)) f) :
    ∀ᶠ z in 𝓝 (0 : Weights A H d L), StrictOn (firstNoiseScores r (n+1) rate batch prompt draw z) f := by
  have ho := (strict_prompt_weights_open (H := H) (d := d) (L := L) r prompt f).preimage
    (AdamW.continuous_firstNoise_weights rate (fun t => batchGradient r (batch t))
      (fun t => continuous_batchGradient r (batch t)) (sgdInitialWeights true draw) n)
  apply ho.mem_nhds
  simpa only [Set.mem_preimage,Set.mem_ofPred_eq,AdamW.firstNoiseRun_zero_noise,adamWRun] using hstock

end Transformer.Grokking.FiniteRecovery.Argand
