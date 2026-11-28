module

public import Transformer.Grokking.FiniteRecovery.FirstNoiseRandomStart
public import Transformer.Grokking.FiniteRecovery.FactorNative

/-! Ordinary WSD with a single Gaussian perturbation after update one,
for modular division and all other unknown group-factor tables. Full
native Argand capacity and identification on the observed mask discharge
the hypotheses of the general certified-restart theorem. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division.Factors

open MeasureTheory ProbabilityTheory Set FlatSpectrum
open scoped NNReal

/-- At every positive stock WSD horizon, every unknown-factor table has
positive strict endpoint mass conditional on any native draw and fixed
minibatch order. The first endpoint passing the observation-only check
is almost surely eventually retained. The numeric kernel uses 10⁻⁶ as
the noise standard deviation. -/
theorem class_firstNoise_wsd_restarts (n : ℕ) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    let μ := Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 => gaussianReal 0 ((1/1000000 : ℝ≥0)^2))
    let scores := firstNoiseScores 1 (n+1) (wsdRate (n+1)) batch divisionPrompt draw
    let ok z := classCertificate S (factorClass unitToken) (fun x => f x) (scores z)
    let p := μ {z | StrictOn (scores z) f}
    0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {z | retained ok z m = none} = (1-p)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) f ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  obtain ⟨z,hrep⟩ := class_native_realization f hf
  exact firstNoise_certified_restarts 1 n (wsdRate (n+1)) (fun t _ =>
    ⟨(wsdRate_bounds _ t).1,lt_of_le_of_lt (wsdRate_bounds _ t).2 (by norm_num)⟩) batch divisionPrompt S
    (factorClass unitToken) (factorClass_identified _ _ hc hz) f hf draw ⟨sgdInitialWeights true z,hrep⟩
    ((1/1000000 : ℝ≥0)^2) (by norm_num)

/-- The optimal 192-observation mask and the full experiment architecture.
All 9,312 answers are checked against all 239 competing tokens. -/
example (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  class_firstNoise_wsd_restarts 99999 (factorMask (1 : DivisionUnit))
    (by rw [unitMask_factorMask]; exact (diagRow_connected _).preconnected) ⟨1,mem_insert _ _⟩
    divisionAnswer divisionAnswer_mem batch draw

example (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  class_firstNoise_wsd_restarts 199999 (factorMask (1 : DivisionUnit))
    (by rw [unitMask_factorMask]; exact (diagRow_connected _).preconnected) ⟨1,mem_insert _ _⟩
    divisionAnswer divisionAnswer_mem batch draw

/-- On each trial a native seed determines both initialization and batch
order. The ideal independent Gaussian perturbation has the standard
deviation of the numerical implementation. Any distribution on the
countable native seed space gives positive single-trial success mass. -/
theorem class_firstNoise_seeded_wsd {D : Type*} [MeasurableSpace D] [MeasurableSingletonClass D]
    [Countable D] (π : Measure D) [IsProbabilityMeasure π] (n : ℕ) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : D → ℕ → List (LabeledPosition (Fin 239))) (draw : D → ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    let P := π.prod (Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 => gaussianReal 0 ((1/1000000 : ℝ≥0)^2)))
    let scores (z : D × Weights (Fin 239) 4 16 6) :=
      firstNoiseScores 1 (n+1) (wsdRate (n+1)) (batch z.1) divisionPrompt (draw z.1) z.2
    let ok z := classCertificate S (factorClass unitToken) (fun x => f x) (scores z)
    let p := P {z | StrictOn (scores z) f}
    0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => P) {z | retained ok z m = none} = (1-p)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => P), ∃ j, StrictOn (scores (z j)) f ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  exact firstNoise_seeded_restarts π 1 n _ (fun t _ => ⟨(wsdRate_bounds _ t).1,
    lt_of_le_of_lt (wsdRate_bounds _ t).2 (by norm_num)⟩) batch draw divisionPrompt S (factorClass unitToken)
    (factorClass_identified _ _ hc hz) f hf ⟨sgdInitialWeights true w,hw⟩ _ (by norm_num)

end Transformer.Grokking.FiniteRecovery.Argand.Division.Factors
