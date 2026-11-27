module

public import Transformer.Grokking.FiniteRecovery.NativeSixLayers
public import Transformer.Grokking.FiniteRecovery.AdamWNativeBasin

/-! Strict realization of modular division in the complete architecture
used by the experiment in experiments/argand, including the zero numerator.
The native draw sets every score, gate and frequency to its usual initial
value. The capacity witness is used only to prove a positive-measure event
under independent Gaussian draws and sufficiently small total learning rate. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2 FlatSpectrum MeasureTheory Set
open scoped ENNReal NNReal

/-- One layer routes and multiplies the characters. The five additional
layers embed this construction into the tested architecture. -/
noncomputable def model (χ : Character) : Model (Fin 239) 4 16 6 :=
  sixLayers (embedding χ) routingLayer

theorem model_strict (χ : Character) (hinj : Function.Injective χ) :
    StrictOn (fun x => (model χ).logits 1 (divisionPrompt x) (Fin.last 4)) divisionAnswer := by
  intro x t ht
  rw [answer_token] at ht ⊢
  apply strict_readout χ hinj (quotient x) _ _ t ht
  rw [model,sixLayers_hidden]
  exact positive_scale (routingLayer_answer χ x layerScale_pos) (Parity.gain_pos _)

/-- From the actual raw initialization support, a tied width-64 Argand
(four heads of width 16, FFN 256, six layers, one pass) strictly predicts
all 9,312 quotients among all 239 tokens. -/
theorem division_native_realization : ∃ z : ComplexParameter (Fin 239) 4 16 6 → ℂ,
    StrictOn (fun x => (Model.fromWeights (sgdInitialWeights true z)).logits 1
      (divisionPrompt x) (Fin.last 4)) divisionAnswer := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  obtain ⟨χ,hinj⟩ := exists_injective_mulChar (p := 97)
  refine ⟨(model χ).tensorEntry,?_⟩
  rw [show sgdInitialWeights true (model χ).tensorEntry = (model χ).toWeights from
    sixLayers_initial _ _ _,Model.fromWeights_toWeights _ rfl]
  exact model_strict χ hinj

/-- For any finite training horizon, a positive-measure set of native
draws succeeds at every iterate and for every minibatch stream when the
sum of learning rates is sufficiently small. Every parameter is trained. -/
theorem division_adamW_native_basin (K : ℕ)
    (variance : ComplexParameter (Fin 239) 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    ∃ δ > (0:ℝ), ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ z ∈ U, ∀ rate : ℕ → ℝ, (∀ t, 0 ≤ rate t ∧ rate t ≤ 10) →
        (∑ t ∈ Finset.range K, rate t) ≤ δ → ∀ batch : ℕ → List (LabeledPosition (Fin 239)), ∀ n ≤ K,
          StrictOn (fun x => (Model.fromWeights (adamWRun 1 batch rate z n).weights).logits 1
            (divisionPrompt x) (Fin.last 4)) divisionAnswer := by
  obtain ⟨z,hz⟩ := division_native_realization
  exact adamW_native_basin 1 K divisionPrompt divisionAnswer variance hv z hz

/-- A connected unit mask and one zero-row observation suffice to verify
the trained quotient. For a small enough rate sum, the probability of
missing all certified endpoints can be made less than any positive
threshold by independent native starts, even with adaptive batches. -/
theorem division_adamW_verified_confidence (K : ℕ) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (variance : ComplexParameter (Fin 239) 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i)
    {failure : ℝ≥0∞} (hf : 0 < failure) :
    let μ := independentGaussianDraws variance
    ∃ δ > (0:ℝ), ∃ N, ∀ m ≥ N, ∀ rate : ℕ → ℝ, (∀ t, 0 ≤ rate t ∧ rate t ≤ 10) →
      (∑ t ∈ Finset.range K, rate t) ≤ δ →
      ∀ batch : (ℕ → ComplexParameter (Fin 239) 4 16 6 → ℂ) → ℕ → ℕ → List (LabeledPosition (Fin 239)),
        Measure.infinitePi (fun _ : ℕ => μ) {z | ∀ j ∈ Finset.range m,
          ¬classCertificate S (factorClass unitToken) (fun x => divisionAnswer x) (fun x =>
            (Model.fromWeights (adamWRun 1 (batch z j) rate (z j) K).weights).logits 1
              (divisionPrompt x) (Fin.last 4))} < failure := by
  let μ := independentGaussianDraws variance
  obtain ⟨δ,hδ,U,hU,hp,basin⟩ := division_adamW_native_basin K variance hv
  obtain ⟨N,hN⟩ := uniform_restart_confidence (fun _ : Unit => μ U) (fun _ => hp) hf
  refine ⟨δ,hδ,N,fun m hm rate hr hs batch => ?_⟩
  have he := iid_miss_window μ hU.measurableSet 0 m
  simp only [zero_add,Nat.Ico_zero_eq_range] at he
  apply lt_of_le_of_lt _ (hN m hm ())
  rw [← he]
  apply measure_mono
  intro z h j hj hu
  exact h j hj ((division_certificate_iff S hc hz _).2 (basin _ hu rate hr hs (batch z j) K le_rfl))

example := division_adamW_native_basin 100000 (fun _ => 1) (by intro; norm_num)
example := division_adamW_native_basin 200000 (fun _ => 1) (by intro; norm_num)

end Transformer.Grokking.FiniteRecovery.Argand.Division
