/-
# Homogenized Transformers — the overlap observable and its derivatives

The calculus behind `lem:Ito_formula` of arXiv:2604.01978v1, *Homogenized
Transformers*, §5: the two derivatives of the observable

  `φ(X) = ⟨x_i, x_j⟩`

on `(𝕊^{d-1})^n`, its Riemannian Hessian, and the generator of an Itô SDE
applied to it.  `φ` is a quadratic form, so `Dφ` is linear and `D²φ(X) = Dφ`;
`overlapForm` is `Dφ` bundled as a continuous linear map of `X`, which is what
makes that reading available.

The statements of `lem:Ito_formula` itself are in
`Transformer.Homogenized.OverlapDrift`.
-/

import Transformer.Homogenized.Generator
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.ContDiff.FTaylorSeries

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The first derivative -/

/-- The derivative of `X ↦ ⟨x_i, x_j⟩`, as the continuous linear map it is:

  `Dφ(X)[U] = ⟨u_i, x_j⟩ + ⟨x_i, u_j⟩`.

It is bundled — and bundled *linearly in `X`* — because the second derivative
of `φ` is read off from it. -/
noncomputable def overlapForm {d n : ℕ} (i j : Idx n) :
    (Idx n → EucSpace d) →L[ℝ] (Idx n → EucSpace d) →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := fun y =>
        (innerSL ℝ (y j)).comp (ContinuousLinearMap.proj (R := ℝ) i) +
          (innerSL ℝ (y i)).comp (ContinuousLinearMap.proj (R := ℝ) j)
      map_add' := by
        intro y z
        ext u
        simp
        ring
      map_smul' := by
        intro c y
        ext u
        simp }

theorem overlapForm_apply {d n : ℕ} (i j : Idx n) (y u : Idx n → EucSpace d) :
    overlapForm i j y u = inner (𝕜 := ℝ) (u i) (y j) + inner (𝕜 := ℝ) (y i) (u j) := by
  simp [overlapForm, real_inner_comm (y j) (u i)]

theorem hasFDerivAt_overlap {d n : ℕ} (i j : Idx n) (X : Idx n → EucSpace d) :
    HasFDerivAt (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j))
      (overlapForm i j X) X := by
  have hi := (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Idx n => EucSpace d)
    i).hasFDerivAt (x := X)
  have hj := (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Idx n => EucSpace d)
    j).hasFDerivAt (x := X)
  refine (hi.inner ℝ hj).congr_fderiv ?_
  ext u
  simp [overlapForm_apply, real_inner_comm]
  ring

/-- **The integrand of the martingale part `eq:martingale_Mij_clean`.**  Along
a tangent direction `U`,

  `Dφ(X)[U] = ⟨u_i, x_j⟩ + ⟨x_i, u_j⟩`,

so at `U = G(θ,X)` this is the source's `⟨G_i(t,θ), x_j(t)⟩ + ⟨x_i(t), G_j(t,θ)⟩`,
the function integrated against `W(dθ,dt)`.

Source: arXiv:2604.01978v1, `eq:martingale_Mij_clean`. -/
theorem fderiv_overlap {d n : ℕ} (i j : Idx n) (X U : Idx n → EucSpace d) :
    fderiv ℝ (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j)) X U
      = inner (𝕜 := ℝ) (U i) (X j) + inner (𝕜 := ℝ) (X i) (U j) := by
  rw [(hasFDerivAt_overlap i j X).fderiv, overlapForm_apply]

/-! ### The second derivative and the Hessian -/

theorem fderiv_fderiv_overlap {d n : ℕ} (i j : Idx n) (X : Idx n → EucSpace d) :
    fderiv ℝ (fderiv ℝ (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j))) X
      = overlapForm i j := by
  have h : fderiv ℝ (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j))
      = ⇑(overlapForm (d := d) i j) :=
    funext fun Y => (hasFDerivAt_overlap i j Y).fderiv
  rw [h, (overlapForm (d := d) i j).fderiv]

theorem iteratedFDeriv_two_overlap {d n : ℕ} (i j : Idx n) (X V : Idx n → EucSpace d) :
    iteratedFDeriv ℝ 2 (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j)) X ![V, V]
      = 2 * inner (𝕜 := ℝ) (V i) (V j) := by
  rw [iteratedFDeriv_two_apply]
  simp [fderiv_fderiv_overlap, overlapForm_apply]
  ring

/-- The Riemannian Hessian of the overlap observable:

  `Hess φ(X)[V,V] = 2⟨v_i, v_j⟩ - ⟨x_i, x_j⟩(‖v_i‖² + ‖v_j‖²)`.

The second term is the sphere correction of `lem:toolkit_geo_riem`; it is what
turns the flat identity `d⟨x_i,x_j⟩ = ⟨dx_i,x_j⟩ + ⟨x_i,dx_j⟩ + d⟨x_i,x_j⟩_t`
into `eq:drift_Dij_clean`.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem sphHess_overlap {d n : ℕ} (i j : Idx n) (X V : Idx n → EucSpace d) :
    sphHess (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j)) X V
      = 2 * inner (𝕜 := ℝ) (V i) (V j)
        - inner (𝕜 := ℝ) (X i) (X j) * (‖V i‖ ^ 2 + ‖V j‖ ^ 2) := by
  simp only [sphHess, iteratedFDeriv_two_overlap]
  congr 1
  have hterm : ∀ k : Idx n,
      fderiv ℝ (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j)) X
          (Pi.single k (X k))
        = (if i = k then inner (𝕜 := ℝ) (X i) (X j) else 0)
          + (if j = k then inner (𝕜 := ℝ) (X i) (X j) else 0) := by
    intro k
    rw [fderiv_overlap, Pi.single_apply, Pi.single_apply]
    by_cases hik : i = k <;> by_cases hjk : j = k <;> simp [hik, hjk]
  simp only [hterm, add_mul, Finset.sum_add_distrib, ite_mul, zero_mul,
    Finset.sum_ite_eq, Finset.mem_univ, ite_true]
  ring

/-- The generator of any Itô SDE on `(𝕊^{d-1})^n`, applied to the overlap
observable: the flat identity `d⟨x_i,x_j⟩ = ⟨dx_i,x_j⟩ + ⟨x_i,dx_j⟩ +
d⟨x_i,x_j⟩_t` together with the sphere correction.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem sphGenerator_overlap {d n : ℕ} (ρ : Measure (HeadParam d))
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    (X : Idx n → EucSpace d) (i j : Idx n) :
    sphGenerator ρ B G (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j)) X
      = inner (𝕜 := ℝ) (B X i) (X j) + inner (𝕜 := ℝ) (X i) (B X j)
        + (1 / 2 : ℝ) * ∫ θ, (2 * inner (𝕜 := ℝ) (G θ X i) (G θ X j)
            - inner (𝕜 := ℝ) (X i) (X j) * (‖G θ X i‖ ^ 2 + ‖G θ X j‖ ^ 2)) ∂ρ := by
  simp only [sphGenerator, fderiv_overlap, sphHess_overlap]

end Homogenized
end Transformer
