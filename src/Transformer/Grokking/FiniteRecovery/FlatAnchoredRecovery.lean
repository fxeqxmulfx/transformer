module

public import Transformer.Grokking.FiniteRecovery.ArgandHiddenConfidence
public import Transformer.Grokking.FiniteRecovery.ArgandModularSum
public import Transformer.Grokking.FiniteRecovery.ArgandGaussianRecovery

/-! Flat Spectrum supplies identifiability for anchored PAGE. On each binary
staircase the learner is told the class of unit-modulus rank-one phase tables.
The class does not specify modular addition or its representing weights.
Theorem E in review/flat_spectrum.md supplies the mask and its size. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open FlatSpectrum Matrix Filter Set MeasureTheory
open scoped ENNReal NNReal

variable {A I J : Type*} [Fintype A] [Nonempty A] {H d L T s : ℕ}

/-- Unique minimum nuclear completion supplies the identifiability hypothesis
of the one-run confidence theorem, for any phase code, cut and target class. -/
theorem anchoredPAGE_of_nuclear_recovery [Fintype I] [Fintype J] [DecidableEq J]
    (mu : Measure (ComplexParameter A H d L → ℂ)) [IsProbabilityMeasure mu]
    {q : ℝ≥0∞} (hq : q < 1) (r : ℕ) (cut : I × J ≃ Word A (T+1))
    (code : A → ℂ) (hc : Function.Injective code) (family : Set (Word A (T+1) → A))
    (S : Finset (Word A (T+1))) (hS : S.Nonempty)
    (hn : ∀ f ∈ family, NuclearRecovers (observedCells cut S) (targetKernel cut code f)) :
    AnchoredRecoveryAt mu q r family S :=
  anchoredPAGE_hidden_confidence mu hq r family S hS ((exists_learner_iff_injOn S family).mp
    (exists_learner_of_nuclear_recovers cut code hc S family hn))

/-- The prior tests only the rank-one phase factorization of a candidate's
predictions. Its row and column factors are unknown to the learner. -/
def unitRankOneFamily (cut : Fin (s+1) × Fin (s+1) ≃ Word A T) (code : A → ℂ) : Set (Word A T → A) :=
  {f | ∃ u v, targetKernel cut code f = vecMulVec u v ∧ ∀ i j, ‖vecMulVec u v i j‖ = 1}

theorem rankOne_anchoredPAGE_staircase (mu : Measure (ComplexParameter A H d L → ℂ))
    [IsProbabilityMeasure mu] {q : ℝ≥0∞} (hq : q < 1) (r : ℕ)
    (cut : Fin (s+1) × Fin (s+1) ≃ Word A (T+1)) (code : A → ℂ) (hc : Function.Injective code) :
    AnchoredRecoveryAt mu q r (unitRankOneFamily cut code) (inputsAtCells cut (staircase s)) := by
  apply anchoredPAGE_of_nuclear_recovery mu hq r cut code hc
  · apply Finset.card_pos.mp
    rw [card_inputsAtCells, ncard_staircase]
    omega
  · rintro f ⟨u,v,he,hu⟩
    rw [observedCells_inputsAtCells,he]
    exact forall_isMinOn_vecMulVec_staircase hu

/-- At the standard independent tensor variance and one pass of the 64-wide,
six-layer Argand, a single anchored PAGE run recovers modular addition above
confidence 3/4. Only the rank-one prior and the staircase labels are supplied. -/
theorem modularSum_anchoredPAGE {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (hk : (k+1)+(k+1) ≤ 32) :
    let cut := enumeratedCut e
    let S := inputsAtCells cut (staircase s)
    let family := unitRankOneFamily cut ZMod.stdAddChar
    let part := observe S (modularSum e)
    let mu := independentGaussianDraws (ι := ComplexParameter (ZMod 2) 4 16 6) (fun _ => (1/50 : ℝ≥0)^2)
    part.card = 2*s+1 ∧ ∃ U, MeasurableSet U ∧ (3/4 : ℝ≥0∞) < mu U ∧
      ∃ a > 0, ∀ η, 0 < η → η ≤ a → ∀ draw ∈ U, ∀ path,
        StrictRecoveryAfterUpdate (T := (k+1)+k) 1 (modularSum e)
          (observedAnchoredPAGE 1 family part η (sgdInitialWeights true draw) path) := by
  dsimp only
  refine ⟨by rw [card_observe,card_inputsAtCells,ncard_staircase],?_⟩
  let mu := independentGaussianDraws (ι := ComplexParameter (ZMod 2) 4 16 6) (fun _ => (1/50 : ℝ≥0)^2)
  have hq : (3/4 : ℝ≥0∞) < 1 := by
    rw [ENNReal.div_lt_iff (by norm_num) (by simp)]
    norm_num
  obtain ⟨U,hU,hp,a,ha,fit⟩ := rankOne_anchoredPAGE_staircase (T := (k+1)+k) mu hq 1
    (enumeratedCut e) ZMod.stdAddChar ZMod.injective_stdAddChar
  have hf : modularSum e ∈ unitRankOneFamily (enumeratedCut e) ZMod.stdAddChar :=
    ⟨_,_,targetKernel_modularSum e,fun _ _ => by simp [vecMulVec_apply]⟩
  exact ⟨U,hU,hp,a,ha,fun η hη hb => fit η hη hb _ hf ⟨_,modularSum_native_realizes e hk⟩⟩

example := modularSum_anchoredPAGE (s := 15) (wordEnumeration 2 4) (by decide)
example := modularSum_anchoredPAGE (s := 255) (wordEnumeration 2 8) (by decide)
example := modularSum_anchoredPAGE (s := 65535) (wordEnumeration 2 16) (by decide)

/-- Two observations suffice for the unknown Boolean constant or copy task
at any positive input length, with strict predictions at the limit of PAGE. -/
example (T layers r : ℕ) (mu : Measure (ComplexParameter Bool 1 1 layers → ℂ))
    [IsProbabilityMeasure mu] {q : ℝ≥0∞} (hq : q < 1) :
    ∃ S : Finset (Word Bool (T+1)), S.card = 2 ∧
      ∃ U, MeasurableSet U ∧ q < mu U ∧ ∃ a > 0, ∀ η, 0 < η → η ≤ a →
        ∀ f ∈ lastTokenFamily T, ∀ draw ∈ U, ∀ path, StrictRecoveryAfterUpdate r f
          (observedAnchoredPAGE r (lastTokenFamily T) (observe S f) η (sgdInitialWeights true draw) path) := by
  obtain ⟨S,hs,learn,hlearn⟩ := (isLeast_lastToken_observations T).1
  have hS : S.Nonempty := Finset.card_pos.mp (by omega)
  obtain ⟨U,hU,hp,a,ha,fit⟩ := anchoredPAGE_hidden_confidence mu hq r (lastTokenFamily T) S hS
    ((exists_learner_iff_injOn _ _).mp ⟨learn,hlearn⟩)
  exact ⟨S,hs,U,hU,hp,a,ha,fun η hη hb f hf => fit η hη hb f hf (lastTokenFamily_native_realizes _ _ _ _ hf)⟩

end Transformer.Grokking.FiniteRecovery.Argand
