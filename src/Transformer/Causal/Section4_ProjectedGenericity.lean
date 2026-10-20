module

public import Transformer.Perspective.UniformHemisphere
public import Transformer.Perspective.UniformShell

/-!
# Generic positive leading covectors up to the leading dimension

ArXiv:2411.04990v2, §4, the initialization condition for a restricted
case of `thm2`. If the number of original tokens is at most the leading
dimension, their actual leading projections are independent almost
surely and admit a common positive leading covector.

The Fubini proof adapts `Perspective.ae_linearIndependent_pi` from
`Perspective/UniformHemisphere.lean`: a projected dependent last token
lies in the inverse image of the preceding projected span. This is
a proper ambient subspace and the original invariant marginal does
not charge it. Source of that auxiliary argument: arXiv:2312.10794v5,
§6.1, the footnote after `thm: d.infty`. No distribution of normalized
projected tokens or independence assumption is supplied as an input.
-/

@[expose] public section

open MeasureTheory

namespace Transformer.Causal

variable {d n : ℕ}

/-- Independence of the actual initial leading projections is Borel.
Source: arXiv:2411.04990v2, §4, the explicit generic initialization
used for the count-at-most-leading-dimension restriction of `thm2`. -/
theorem measurableSet_projectedLinearIndependent (L : Submodule ℝ (EucSpace d)) (k : ℕ) :
    MeasurableSet {X : SphereTuple d k |
      LinearIndependent ℝ (fun i => L.starProjection (X i : EucSpace d))} :=
  (isOpen_setOfPred_linearIndependent.preimage (continuous_pi fun i =>
    L.starProjection.continuous.comp
      (continuous_subtype_val.comp (continuous_apply i)))).measurableSet

/-- Up to the actual leading dimension, the original i.i.d. invariant
initialization has independent leading projections almost surely.
Source: arXiv:2411.04990v2, §4, this restricted initialization case of
`thm2`; proof adapted from the cited `Perspective.ae_linearIndependent_pi`. -/
theorem ae_projectedLinearIndependent_pi (L : Submodule ℝ (EucSpace d))
    (σ : Measure (SSphere d)) [IsFiniteMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (Perspective.sphereMap d U) = σ) :
    ∀ k, k ≤ Module.finrank ℝ L → ∀ᵐ X ∂(Measure.pi fun _ : Idx k => σ),
      LinearIndependent ℝ (fun i => L.starProjection (X i : EucSpace d)) := by
  intro k
  induction k with
  | zero => exact fun _ => Filter.Eventually.of_forall fun _ => linearIndependent_empty_type
  | succ k ih =>
    intro hk
    set e := MeasurableEquiv.piFinSuccAbove (fun _ : Idx (k + 1) => SSphere d) (Fin.last k)
    have hmp :=
      (measurePreserving_piFinSuccAbove (fun _ : Idx (k + 1) => σ) (Fin.last k)).symm e
    have hS : MeasurableSet {X : SphereTuple d (k + 1) | ¬LinearIndependent ℝ
        (fun i => L.starProjection (X i : EucSpace d))} :=
      (measurableSet_projectedLinearIndependent L (k + 1)).compl
    rw [ae_iff, ← hmp.measure_preimage hS.nullMeasurableSet,
      Measure.prod_apply_symm (e.symm.measurable hS)]
    refine (lintegral_congr_ae ?_).trans (lintegral_zero (μ := Measure.pi fun _ : Idx k => σ))
    filter_upwards [ih (by omega)] with Y hY
    let S := Submodule.span ℝ (Set.range fun i => L.starProjection (Y i : EucSpace d))
    let W := S.comap L.starProjection.toLinearMap
    have hW : W ≠ ⊤ := by
      intro hW
      have hLS : L ≤ S := by
        intro z hz
        have hzW : z ∈ W := by rw [hW]; exact Submodule.mem_top
        change L.starProjection z ∈ S at hzW
        rwa [L.starProjection_eq_self_iff.mpr hz] at hzW
      have hdim := Submodule.finrank_mono hLS
      have hdimS : Module.finrank ℝ S = k := by
        simpa using finrank_span_eq_card hY
      rw [hdimS] at hdim
      omega
    refine measure_mono_null (t := {x : SSphere d | (x : EucSpace d) ∈ W}) (fun x hx => ?_)
      (Perspective.measure_mem_submodule_eq_zero d σ hσ hW)
    simp only [Set.mem_preimage, Set.mem_ofPred_eq] at hx
    by_contra hxW
    apply hx
    have hsnoc : e.symm (x, Y) = Fin.snoc Y x := by
      funext j
      rw [MeasurableEquiv.piFinSuccAbove_symm_apply, Fin.insertNthEquiv_apply,
        Fin.insertNth_last']
    have he : (fun i => L.starProjection (e.symm (x, Y) i : EucSpace d)) =
        Fin.snoc (fun i => L.starProjection (Y i : EucSpace d))
          (L.starProjection (x : EucSpace d)) := by
      rw [hsnoc]
      exact Fin.comp_snoc
        (fun y : SSphere d => L.starProjection (y : EucSpace d)) Y x
    rw [he, linearIndependent_finSnoc]
    exact ⟨hY, hxW⟩

/-- The same projected independence statement holds under the original
uniform tuple law, with its actual leading dimension as the count bound.
Source: arXiv:2411.04990v2, §4, this restricted initialization case of `thm2`. -/
theorem ae_projectedLinearIndependent_of_uniformTuple (L : Submodule ℝ (EucSpace d))
    (hcount : n ≤ Module.finrank ℝ L)
    (P : Measure (SphereTuple d n)) (hP : Perspective.UniformTuple d n P) :
    ∀ᵐ X ∂P, LinearIndependent ℝ (fun i => L.starProjection (X i : EucSpace d)) := by
  obtain ⟨σ, hprob, hinv, rfl⟩ := hP
  have := hprob
  exact ae_projectedLinearIndependent_pi L σ hinv n hcount

/-- Independent actual leading projections admit a common positive
covector in the actual leading space. Source: arXiv:2411.04990v2, §4,
the explicit leading-hemisphere restriction of `thm2`; the hemisphere
construction is the cited `Perspective` linear-independence argument. -/
theorem exists_positive_leading_covector_of_independent (L : Submodule ℝ (EucSpace d))
    (X : SphereTuple d n) (hn : 1 ≤ n)
    (hX : LinearIndependent ℝ (fun i => L.starProjection (X i : EucSpace d))) :
    ∃ w ∈ L, ∀ i, 0 < inner ℝ w (X i : EucSpace d) := by
  obtain ⟨u, hu⟩ :=
    Perspective.exists_common_hemisphere_of_linearIndependent d _ hX hn
  refine ⟨L.starProjection (u : EucSpace d),
    L.starProjection_apply_mem _, fun i => ?_⟩
  rw [L.inner_starProjection_left_eq_right,
    real_inner_comm (L.starProjection (X i : EucSpace d)) (u : EucSpace d)]
  exact hu i

/-- The original uniform initialization admits a common positive
leading covector almost surely when its token count does not exceed
the leading dimension. This count restriction is explicit and is
not asserted for longer tuples. Source: arXiv:2411.04990v2, §4,
this restricted case of `thm2`. -/
theorem ae_exists_positive_leading_covector (L : Submodule ℝ (EucSpace d))
    (hn : 1 ≤ n) (hcount : n ≤ Module.finrank ℝ L)
    (P : Measure (SphereTuple d n)) (hP : Perspective.UniformTuple d n P) :
    ∀ᵐ X ∂P, ∃ w ∈ L, ∀ i, 0 < inner ℝ w (X i : EucSpace d) := by
  filter_upwards [ae_projectedLinearIndependent_of_uniformTuple L hcount P hP]
    with X hX
  exact exists_positive_leading_covector_of_independent L X hn hX

/-- The original shell law, full leading plane and a nonempty tuple
realize the measure and dimension premises. The actual unit coordinate
vector also realizes the projected independence premise.
Source: arXiv:2411.04990v2, §4, the restricted initialization of `thm2`. -/
example : ∃ P : Measure (SphereTuple 2 1), Perspective.UniformTuple 2 1 P ∧
    (1 : ℕ) ≤ 1 ∧ 1 ≤ Module.finrank ℝ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    LinearIndependent ℝ (fun _ : Idx 1 => (⊤ : Submodule ℝ (EucSpace 2)).starProjection
      (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))) := by
  refine ⟨Measure.pi (fun _ : Idx 1 => Perspective.shellLaw (d := 2) (by norm_num)),
    Perspective.uniformTuple_shellLaw (by norm_num), le_rfl, by simp, ?_⟩
  rw [linearIndependent_unique_iff]
  simp [Submodule.starProjection_top]

end Transformer.Causal
