module

public import Transformer.Grokking.FiniteRecovery.ArgandFirstNoise
public import Transformer.Grokking.FiniteRecovery.NativeSixLayers
public import Transformer.Grokking.FiniteRecovery.TransitionForecast

/-! Forecasting any deterministic transition on 64 states with the native
six-layer Argand and first-update Gaussian AdamW. A common transition may
be observed at sites from several series with overlapping contexts. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transition MeasureTheory ProbabilityTheory Set
open scoped InnerProductSpace NNReal

/-- One product layer executes an arbitrary transition. The remaining
five layers preserve the residual. Both RMS normalizations use the
native offset, and the tied embedding compares all 64 possible answers. -/
theorem sixLayer_transition_realization {T : ℕ} (next : Fin 64 → Fin 64) :
    ∃ w : Weights (Fin 64) 4 16 6, (Model.fromWeights w).realizes 1 (transitionTask (T := T) next) := by
  let eps := Parity.offset
  have he : 0 < eps := by norm_num [eps,Parity.offset]
  have hc : 0 < 4 * 16 := by decide
  let a := rmsAmplitude 64 1 eps
  have ha : 0 < a := rmsAmplitude_pos_at he hc one_pos
  let layer := transitionLayer (H := 4) (d := 16) next (rmsAmplitude 64 a eps)
  let model := Division.sixLayers basis layer
  let out := rmsAmplitude 64 (Division.layerScale * a) eps
  have hb : 0 < Division.layerScale * a := mul_pos Division.layerScale_pos ha
  have ho : 0 < out := rmsAmplitude_pos_at he hc hb
  refine ⟨model.toWeights,?_⟩
  rw [Model.fromWeights_toWeights _ rfl]
  intro word token hne
  let i := Fin.last T
  have hi : (fun j => argandRMS eps (basis (word j))) = fun j => (a : ℂ) • basis (word j) := by
    funext j
    simpa only [Complex.ofReal_one,one_smul] using argandRMS_basis_at eps (word j) (a := 1) zero_le_one
  have hh : model.hidden 1 word i = (out : ℂ) • basis (next (word i)) := by
    rw [Division.sixLayers_hidden,hi,transitionLayer_forward_at eps he hc next _ _ ha]
    exact argandRMS_basis_at eps _ hb.le
  have logits (b : Fin 64) : model.logits 1 word i b = if next (word i) = b then out/2 else 0 := by
    rw [Model.logits,hh]
    change (⟪basis b, (out : ℂ) • basis (next (word i))⟫_ℂ).re / 2 = _
    rw [inner_smul_right]
    by_cases h : next (word i) = b <;>
      simp [basis,EuclideanSpace.inner_single_left,h]
  change token ≠ next (word i) at hne
  change model.logits 1 word i token < model.logits 1 word i (next (word i))
  rw [logits,logits,ite_eq_right hne.symm,ite_eq_left rfl]
  positivity

/-- The number of required sites depends on the number of different final
states, even when many series contribute histories. Once the mask covers
them, the ordinary WSD rates and independent first-update perturbations
almost surely retain a strictly correct forecast on every site. -/
theorem series_transition_firstNoise {I : Type*} [Fintype I] {T : ℕ}
    (prompt : I → Word (Fin 64) (T+1)) (S : Set I)
    (hcover : ∀ x, ∃ y : S, prompt y.val (Fin.last T) = prompt x (Fin.last T))
    (next : Fin 64 → Fin 64) (n : ℕ) (batch : ℕ → List (LabeledPosition (Fin 64)))
    (draw : ComplexParameter (Fin 64) 4 16 6 → ℂ) :
    let last x := prompt x (Fin.last T)
    let family := range fun g : Fin 64 → Fin 64 => g ∘ last
    let μ := Measure.pi (fun _ : Parameter (Fin 64) 4 16 6 => gaussianReal 0 ((1/1000000 : ℝ≥0)^2))
    let scores := firstNoiseScores 1 (n+1) (wsdRate (n+1)) batch prompt draw
    let ok z := classCertificate S family (fun x => next (last x)) (scores z)
    let p := μ {z | StrictOn (scores z) (next ∘ last)}
    IsLeast {k : ℕ | ∃ U : Finset I, U.card = k ∧ DeterminesOnContexts last (univ : Set (Fin 64 → Fin 64)) U}
      (Finset.univ.image last).card ∧
    0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {z | retained ok z m = none} = (1-p)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) (next ∘ last) ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  intro last family μ scores ok p
  have hid : InjOn (fun g : I → Fin 64 => fun x : S => g x) family := by
    rintro _ ⟨g,rfl⟩ _ ⟨f,rfl⟩ hagree
    funext x
    obtain ⟨y,hy⟩ := hcover x
    change last y.val = last x at hy
    change g (last x) = f (last x)
    rw [← hy]
    exact congrFun hagree y
  obtain ⟨w,hw⟩ := sixLayer_transition_realization (T := T) next
  refine ⟨isLeast_unrestricted_series_observations (A := Fin 64) last,?_⟩
  exact firstNoise_certified_restarts 1 n _ (fun t _ => ⟨(wsdRate_bounds _ t).1,
    lt_of_le_of_lt (wsdRate_bounds _ t).2 (by norm_num)⟩) batch prompt S family hid
    (next ∘ last) (mem_range_self next) draw ⟨w,fun x => hw (prompt x)⟩ _ (by norm_num)

/-- All length-32 histories are identified by the 64 constant histories. -/
example (next : Fin 64 → Fin 64) :=
  series_transition_firstNoise (T := 31) id (transitionInputs (Fin 64) 31 : Set _)
    (fun x => ⟨⟨fun _ => x (Fin.last 31),Finset.mem_map.mpr ⟨_,Finset.mem_univ _,rfl⟩⟩,rfl⟩)
    next 99999

end Transformer.Grokking.FiniteRecovery.Argand
