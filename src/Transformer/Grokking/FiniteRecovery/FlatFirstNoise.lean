module

public import Transformer.Grokking.FiniteRecovery.ArgandFirstNoise
public import Transformer.Grokking.FiniteRecovery.FlatAnchoredRecovery

/-! The Flat Spectrum observation criterion gives an observation-only
verifier for the perturbed AdamW endpoint. In particular the binary
modular sums use their exact staircase minima without changing the
stock WSD rates, width, depth or number of passes. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory ProbabilityTheory Set FlatSpectrum Matrix
open scoped NNReal

variable {A I J : Type*} [Fintype A] [Nonempty A] {H d L T : ℕ}

/-- Any cut and injective phase code with unique minimum nuclear-norm
completion supply the identification condition. Static representability
of this target suffices at every positive WSD horizon. -/
theorem firstNoise_of_flat_spectrum [Fintype I] [Fintype J] [DecidableEq J]
    (r n : ℕ) (cut : I × J ≃ Word A (T+1)) (code : A → ℂ) (hc : Function.Injective code)
    (family : Set (Word A (T+1) → A)) (S : Finset (Word A (T+1)))
    (hn : ∀ f ∈ family, NuclearRecovers (observedCells cut S) (targetKernel cut code f))
    (f : Word A (T+1) → A) (hf : f ∈ family) (batch : ℕ → List (LabeledPosition A))
    (draw : ComplexParameter A H d L → ℂ)
    (hrep : ∃ w : Weights A H d L, (Model.fromWeights w).realizes r f)
    (v : ℝ≥0) (hv : 0 < v) :
    let μ := Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v)
    let scores := firstNoiseScores r (n+1) (wsdRate (n+1)) batch id draw
    let ok z := classCertificate (S : Set _) family (fun x => f x) (scores z)
    let p := μ {z | StrictOn (scores z) f}
    0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {z | retained ok z m = none} = (1-p)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) f ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  have hi := (exists_learner_iff_injOn S family).mp (exists_learner_of_nuclear_recovers cut code hc S family hn)
  exact firstNoise_certified_restarts r n _ (fun t _ => ⟨(wsdRate_bounds _ t).1,
    lt_of_le_of_lt (wsdRate_bounds _ t).2 (by norm_num)⟩) batch id S family
    (fun a ha b hb he => hi ha hb (observe_eq_iff_eqOn.mpr fun x hx => congrFun he ⟨x,hx⟩))
    f hf draw hrep v hv

/-- The learner receives the unit-modulus rank-one class and its observed
labels. For 8, 16 and 32 bits the same H=4, d=16, six layers and one pass
give almost-sure strict recovery by certified independent restarts. -/
theorem modularSum_firstNoise {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (hk : (k+1)+(k+1) ≤ 32) (n : ℕ) (batch : ℕ → List (LabeledPosition (ZMod 2)))
    (draw : ComplexParameter (ZMod 2) 4 16 6 → ℂ) :
    let cut := enumeratedCut e
    let S := inputsAtCells cut (staircase s)
    let family := unitRankOneFamily cut ZMod.stdAddChar
    let μ := Measure.pi (fun _ : Parameter (ZMod 2) 4 16 6 => gaussianReal 0 ((1/1000000 : ℝ≥0)^2))
    let scores := firstNoiseScores (T := (k+1)+k) 1 (n+1) (wsdRate (n+1)) batch id draw
    let ok z := classCertificate (S : Set _) family (fun x => modularSum e x) (scores z)
    (observe S (modularSum e)).card = 2*s+1 ∧ 0 < μ {z | StrictOn (scores z) (modularSum e)} ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) (modularSum e) ∧
        ∀ l, retained ok z (j+1+l) = some (z j) := by
  let cut := enumeratedCut e
  let S := inputsAtCells cut (staircase s)
  let family := unitRankOneFamily cut ZMod.stdAddChar
  have hf : modularSum e ∈ family :=
    ⟨_,_,targetKernel_modularSum e,fun _ _ => by simp [vecMulVec_apply]⟩
  have hn : ∀ f ∈ family, NuclearRecovers (observedCells cut S) (targetKernel cut ZMod.stdAddChar f) := by
    rintro f ⟨u,v,hu,hv⟩
    rw [show S = inputsAtCells cut _ from rfl,observedCells_inputsAtCells,hu]
    exact forall_isMinOn_vecMulVec_staircase hv
  obtain ⟨hp,_,hs⟩ := firstNoise_of_flat_spectrum (T := (k+1)+k) 1 n cut ZMod.stdAddChar
    ZMod.injective_stdAddChar family S hn _ hf batch draw ⟨_,modularSum_native_realizes e hk⟩
    ((1/1000000 : ℝ≥0)^2) (by norm_num)
  exact ⟨by rw [card_observe,card_inputsAtCells,ncard_staircase],hp,hs⟩

example := modularSum_firstNoise (wordEnumeration 2 4) (by decide) 20479
example := modularSum_firstNoise (wordEnumeration 2 8) (by decide) 20479
example := modularSum_firstNoise (wordEnumeration 2 16) (by decide) 20479

end Transformer.Grokking.FiniteRecovery.Argand
