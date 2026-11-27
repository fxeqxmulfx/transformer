module

public import Transformer.Grokking.FiniteRecovery.FactorLayer
public import Transformer.Grokking.FiniteRecovery.DivisionNative
public import Transformer.Grokking.FiniteRecovery.FactorObservations

/-! Every member of the unknown-factor class is strictly represented in
the same native architecture, including arbitrary constant-row outputs.
The general AdamW theorem gives a uniform confidence threshold, for
fixed training horizons and sufficiently small total learning rate. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division.Factors

open Transformer.PhasorV2 FlatSpectrum MeasureTheory Set
open scoped NNReal ENNReal

noncomputable def model (χ : Character) (r c : UnitFactor) (z : Fin 239) : Model (Fin 239) 4 16 6 :=
  sixLayers (vocab χ r c z) layer

theorem model_strict (χ : Character) (hinj : Function.Injective χ) (r c : UnitFactor) (z : Fin 239) :
    StrictOn (fun x => (model χ r c z).logits 1 (divisionPrompt x) (Fin.last 4)) (factorTable unitToken r c z) := by
  intro x t ht
  apply readout χ hinj r c z x _ _ t ht
  rw [model,sixLayers_hidden]
  exact scaled_state (layer_state χ r c z x layerScale_pos) (Parity.gain_pos _)

/-- The *entire* class, with arbitrary unit factors and arbitrary zero-row
token, fits the standard initialization support at H=4, d=16, depth=6,
FFN width 256 and one pass. The construction varies only the embedding. -/
theorem class_native_realization (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken) :
    ∃ w : ComplexParameter (Fin 239) 4 16 6 → ℂ, StrictOn (fun x =>
      (Model.fromWeights (sgdInitialWeights true w)).logits 1 (divisionPrompt x) (Fin.last 4)) f := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  obtain ⟨r,c,z,rfl⟩ := hf
  obtain ⟨χ,hinj⟩ := exists_injective_mulChar (p := 97)
  refine ⟨(model χ r c z).tensorEntry,?_⟩
  rw [show sgdInitialWeights true (model χ r c z).tensorEntry = (model χ r c z).toWeights from
    sixLayers_initial _ _ _,Model.fromWeights_toWeights _ rfl]
  exact model_strict χ hinj r c z

/-- A single bound on the total rate and a single restart count work for
every table in the class. The verification uses labels only on the
identifying mask. Each trial may use an adaptive minibatch stream. -/
theorem class_adamW_confidence (K : ℕ) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (variance : ComplexParameter (Fin 239) 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i)
    {failure : ℝ≥0∞} (hf : 0 < failure) :
    let μ := independentGaussianDraws variance
    ∃ δ > (0:ℝ), ∃ N, ∀ m ≥ N, ∀ (f : DivisionInput → Fin 239), f ∈ factorClass unitToken → ∀ rate : ℕ → ℝ,
      (∀ t, 0 ≤ rate t ∧ rate t ≤ 10) → (∑ t ∈ Finset.range K, rate t) ≤ δ →
      ∀ batch : (ℕ → ComplexParameter (Fin 239) 4 16 6 → ℂ) → ℕ → ℕ → List (LabeledPosition (Fin 239)),
        Measure.infinitePi (fun _ : ℕ => μ) {w | ∀ j ∈ Finset.range m,
          ¬classCertificate S (factorClass unitToken) (fun x => f x) (fun x =>
            (Model.fromWeights (adamWRun 1 (batch w j) rate (w j) K).weights).logits 1
              (divisionPrompt x) (Fin.last 4))} < failure :=
  adamW_uniform_class_confidence 1 K divisionPrompt S (factorClass unitToken)
    (factorClass_identified _ _ hc hz) variance hv class_native_realization hf

/-- On the exact 192-cell mask the confidence after sufficiently many
independent 100,000-step trials exceeds three quarters for every target
in the class. -/
example := class_adamW_confidence 100000 (factorMask (1 : DivisionUnit))
  (by rw [unitMask_factorMask]; exact (diagRow_connected _).preconnected)
  ⟨1,mem_insert _ _⟩ (fun _ => 1) (by intro; norm_num) (failure := 1/4) (by norm_num)

end Transformer.Grokking.FiniteRecovery.Argand.Division.Factors
