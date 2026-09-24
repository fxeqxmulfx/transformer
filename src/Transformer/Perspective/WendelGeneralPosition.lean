/-
# General position of a uniform spherical sample

For Wendel's formula, every subfamily of at most `d` sampled points must be
linearly independent almost surely.  This extends the initial-segment result
of `UniformHemisphere` to arbitrary finite sets of sample indices.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`.
-/

import Transformer.Perspective.UniformHemisphere
import Mathlib.Probability.ProductMeasure

open MeasureTheory

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-- Almost-sure linear independence for a product sample indexed by an
arbitrary finite type, provided its cardinality does not exceed the ambient
dimension. Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem ae_linearIndependent_pi_fintype (σ : Measure (SSphere d))
    [IsProbabilityMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ)
    (ι : Type*) [Fintype ι] (hcard : Fintype.card ι ≤ d) :
    ∀ᵐ X ∂(Measure.pi fun _ : ι => σ),
      LinearIndependent ℝ fun i : ι => (X i : EucSpace d) := by
  classical
  let e : ι ≃ Fin (Fintype.card ι) := Fintype.equivFin ι
  let F : (ι → SSphere d) → Fin (Fintype.card ι) → SSphere d :=
    fun X j => X (e.symm j)
  have hmap : Measure.map F (Measure.pi fun _ : ι => σ) =
      Measure.pi fun _ : Fin (Fintype.card ι) => σ := by
    let E := MeasurableEquiv.piCongrLeft (fun _ : Fin (Fintype.card ι) => SSphere d) e
    have hp : MeasurePreserving E (Measure.pi fun _ : ι => σ)
        (Measure.pi fun _ : Fin (Fintype.card ι) => σ) :=
      measurePreserving_piCongrLeft (fun _ : Fin (Fintype.card ι) => σ) e
    have hFE : F = E := by
      funext X j
      simpa [F, E, MeasurableEquiv.piCongrLeft] using
        (Equiv.piCongrLeft_apply_apply
          (fun _ : Fin (Fintype.card ι) => SSphere d) e X (e.symm j)).symm
    rw [hFE]
    exact hp.map_eq
  have hfin := ae_linearIndependent_pi d σ hσ (Fintype.card ι) hcard
  rw [← hmap] at hfin
  have hF : AEMeasurable F (Measure.pi fun _ : ι => σ) := by
    have hm : Measurable F := by
      fun_prop
    exact hm.aemeasurable
  have hmeas := measurableSet_linearIndependent d (Fintype.card ι)
  have hpull := (ae_map_iff hF hmeas).mp hfin
  filter_upwards [hpull] with X hX
  have heq : (fun i : ι => (F X (e i) : EucSpace d)) =
      fun i : ι => (X i : EucSpace d) := by
    funext i
    simp [F]
  rw [← heq]
  exact (linearIndependent_equiv e).2 hX

/-- Every fixed subfamily of at most `d` points in an i.i.d. uniform
spherical sample is linearly independent almost surely.  The product-law
marginal on the chosen indices is again a product law.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem ae_linearIndependent_on_finset (σ : Measure (SSphere d))
    [IsProbabilityMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ)
    {n : ℕ} (I : Finset (Idx n)) (hI : I.card ≤ d) :
    ∀ᵐ X ∂(Measure.pi fun _ : Idx n => σ),
      LinearIndependent ℝ fun i : I => (X i : EucSpace d) := by
  classical
  have hmap : Measure.map I.restrict (Measure.pi fun _ : Idx n => σ) =
      Measure.pi fun _ : I => σ := by
    simpa only [Measure.infinitePi_eq_pi] using
      (Measure.infinitePi_map_restrict (fun _ : Idx n => σ) (I := I))
  have hres : Measurable (I.restrict : (Idx n → SSphere d) → I → SSphere d) := by
    fun_prop
  have hmeas : MeasurableSet
      {Y : I → SSphere d | LinearIndependent ℝ fun i : I => (Y i : EucSpace d)} :=
    (isOpen_setOfPred_linearIndependent.preimage
      (continuous_pi fun i => continuous_subtype_val.comp (continuous_apply i))).measurableSet
  have hae := ae_linearIndependent_pi_fintype d σ hσ I (by simpa using hI)
  rw [← hmap] at hae
  exact (ae_map_iff hres.aemeasurable hmeas).mp hae

/-- A finite uniform spherical sample is almost surely in linear general
position: every subfamily of at most `d` points is independent.  This is the
nondegeneracy condition used in the counting proof of Wendel's formula.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem ae_linearGeneralPosition (n : ℕ)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P) :
    ∀ᵐ X ∂P, ∀ I : Finset (Idx n), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (X i : EucSpace d) := by
  classical
  obtain ⟨σ, hσprob, hσinv, rfl⟩ := hP
  let _ : IsProbabilityMeasure σ := hσprob
  apply Filter.eventually_all.mpr
  intro I
  by_cases hI : I.card ≤ d
  · filter_upwards [ae_linearIndependent_on_finset d σ hσinv I hI] with X hX
    exact fun _ => hX
  · exact Filter.Eventually.of_forall fun _ h => (hI h).elim

/-- The cardinality condition for general position is satisfiable at
`d = n = 1`, with the unique sample index selected. -/
example : (Finset.univ : Finset (Idx 1)).card ≤ 1 := by decide

end Perspective
end Transformer
