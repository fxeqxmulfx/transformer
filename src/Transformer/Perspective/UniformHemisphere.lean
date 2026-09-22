/-
# `n ≤ d` uniform points lie in an open hemisphere almost surely

The survey proves `thm: d.infty` by cone collapse, and justifies its
hypothesis in a footnote: when `d ≥ n`, `n` uniform points lie in an open
hemisphere with probability one, a "weak version of Wendel's theorem" that
"is easy to see directly".  Directly: they are almost surely linearly
independent (`ae_linearIndependent_of_uniformTuple`, by Fubini from
`measure_mem_submodule_eq_zero`), and independent points lie in a common open
hemisphere (`exists_common_hemisphere_of_linearIndependent`); together, this
is `ae_exists_openHemisphere`.

Source: arXiv:2312.10794v5, §6.1, the paragraph after `thm: d.infty` and its
footnote; arXiv:2512.01868v4, §4, `cor: d-ge-n`.
-/

import Transformer.Perspective.SphereHyperplane
import Transformer.Perspective.Section3_SmallBeta
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.LinearAlgebra.LinearIndependent.Lemmas
import Mathlib.MeasureTheory.Constructions.Pi

open MeasureTheory

namespace Transformer
namespace Perspective

variable (d : ℕ) {n : ℕ}

/-- Linear independence of `k` points of the sphere is an open, hence
measurable, condition. -/
theorem measurableSet_linearIndependent (k : ℕ) :
    MeasurableSet {X : Fin k → SSphere d | LinearIndependent ℝ fun i => (X i : EucSpace d)} :=
  (isOpen_setOfPred_linearIndependent.preimage
    (continuous_pi fun i => continuous_subtype_val.comp (continuous_apply i))).measurableSet

/-- **`k ≤ d` i.i.d. points of a rotation-invariant law on `𝕊^{d-1}` are
almost surely linearly independent.**

By induction on `k`.  Split the last point off: the product measure on
`(𝕊^{d-1})^{k+1}` is `σ ⊗ σ^{⊗k}` (`measurePreserving_piFinSuccAbove`), and a
tuple `(Y, x)` with `Y` independent is dependent only if `x` lies in the span
of `Y` (`linearIndependent_finSnoc`), a subspace of dimension `k < d`, which
`σ` does not charge (`measure_mem_submodule_eq_zero`). -/
theorem ae_linearIndependent_pi (σ : Measure (SSphere d)) [IsFiniteMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ) :
    ∀ k, k ≤ d → ∀ᵐ X ∂(Measure.pi fun _ : Fin k => σ),
      LinearIndependent ℝ fun i => (X i : EucSpace d) := by
  intro k
  induction k with
  | zero => exact fun _ => Filter.Eventually.of_forall fun _ => linearIndependent_empty_type
  | succ k ih =>
    intro hk
    set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (k + 1) => SSphere d) (Fin.last k)
    have hmp := (measurePreserving_piFinSuccAbove (fun _ : Fin (k + 1) => σ) (Fin.last k)).symm e
    have hS : MeasurableSet
        {X : Fin (k + 1) → SSphere d | ¬LinearIndependent ℝ fun i => (X i : EucSpace d)} :=
      (measurableSet_linearIndependent d (k + 1)).compl
    rw [ae_iff, ← hmp.measure_preimage hS.nullMeasurableSet,
      Measure.prod_apply_symm (e.symm.measurable hS)]
    refine (lintegral_congr_ae ?_).trans (lintegral_zero (μ := Measure.pi fun _ : Fin k => σ))
    filter_upwards [ih (by omega)] with Y hY
    set V := Submodule.span ℝ (Set.range fun i => (Y i : EucSpace d))
    have hV : V ≠ ⊤ := by
      intro h
      have : Module.finrank ℝ V = Fintype.card (Fin k) := finrank_span_eq_card hY
      rw [h, finrank_top, finrank_euclideanSpace_fin, Fintype.card_fin] at this
      omega
    refine measure_mono_null (fun x hx => ?_) (measure_mem_submodule_eq_zero d σ hσ hV)
    simp only [Set.mem_preimage, Set.mem_ofPred_eq] at hx
    by_contra hxV
    refine hx ?_
    have hsnoc : e.symm (x, Y) = Fin.snoc Y x := by
      funext j
      rw [MeasurableEquiv.piFinSuccAbove_symm_apply, Fin.insertNthEquiv_apply, Fin.insertNth_last']
    have : (fun i => (e.symm (x, Y) i : EucSpace d)) =
        Fin.snoc (fun i => (Y i : EucSpace d)) (x : EucSpace d) := by
      rw [hsnoc]
      exact Fin.comp_snoc (fun y : SSphere d => (y : EucSpace d)) Y x
    rw [this, linearIndependent_finSnoc]
    exact ⟨hY, hxV⟩

/-- The hypotheses of `ae_linearIndependent_pi` are satisfiable: the zero
measure is finite and invariant under every `sphereMap`, and `1 ≤ 1`. -/
example : IsFiniteMeasure (0 : Measure (SSphere 1)) ∧
    (∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1, (0 : Measure (SSphere 1)).map (sphereMap 1 U) = 0) ∧
    1 ≤ 1 :=
  ⟨inferInstance, fun _ => Measure.map_zero _, le_rfl⟩

/-- **`n ≤ d` uniform points of `𝕊^{d-1}` are almost surely linearly
independent**, for the uniform law `UniformTuple` of §4. -/
theorem ae_linearIndependent_of_uniformTuple (hnd : n ≤ d) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      ∀ᵐ X ∂P, LinearIndependent ℝ fun i => (X i : EucSpace d) := by
  rintro P ⟨σ, _, hσ, rfl⟩
  exact ae_linearIndependent_pi d σ hσ n hnd

/-- The hypothesis of `ae_linearIndependent_of_uniformTuple` is satisfiable:
`n = d = 1`; the uniform law is quantified over. -/
example : 1 ≤ 1 := le_rfl

/-- **Linearly independent points lie in a common open hemisphere.**

If `x_1, …, x_n` are linearly independent — which forces `n ≤ d` — there is a
unit vector `w` with `⟨x_i, w⟩ > 0` for every `i`.  The map
`u ↦ (⟨x_i, u⟩)_i` from the span of the `x_i` to `ℝ^n` is injective, because a
vector of the span orthogonal to every `x_i` is orthogonal to itself; the span
has dimension `n`, so the map is onto, and the preimage of the all-ones vector
is the required `w` up to normalization.

Source: arXiv:2312.10794v5, §6.1, the paragraph after `thm: d.infty`;
arXiv:2512.01868v4, §4, before `cor: d-ge-n`. -/
theorem exists_common_hemisphere_of_linearIndependent
    (X : Idx n → EucSpace d) (hX : LinearIndependent ℝ X) (hn : 1 ≤ n) :
    ∃ w : SSphere d, ∀ i : Idx n, 0 < inner (𝕜 := ℝ) (X i) ((w : EucSpace d)) := by
  classical
  -- Step 1: a vector `w₀` with `⟨x_i, w₀⟩ = 1` for every `i`.
  have hone : ∃ w₀ : EucSpace d, ∀ i : Idx n, inner (𝕜 := ℝ) (X i) w₀ = 1 := by
    set S := Submodule.span ℝ (Set.range X) with hS
    let g : S →ₗ[ℝ] (Idx n → ℝ) :=
      { toFun := fun u i => inner (𝕜 := ℝ) (X i) (u : EucSpace d)
        map_add' := by intro a b; funext i; simp [inner_add_right]
        map_smul' := by intro c a; funext i; simp [real_inner_smul_right] }
    have hginj : Function.Injective g := by
      rw [← LinearMap.ker_eq_bot, Submodule.eq_bot_iff]
      intro u hu
      have h0 : ∀ i : Idx n, inner (𝕜 := ℝ) (X i) (u : EucSpace d) = 0 :=
        fun i => congrFun (LinearMap.mem_ker.mp hu) i
      have hall : ∀ y ∈ S, inner (𝕜 := ℝ) y (u : EucSpace d) = 0 := by
        intro y hy
        induction hy using Submodule.span_induction with
        | mem x hx => obtain ⟨i, rfl⟩ := hx; exact h0 i
        | zero => simp
        | add a b _ _ ha hb => rw [inner_add_left, ha, hb, add_zero]
        | smul c a _ ha => rw [real_inner_smul_left, ha, mul_zero]
      exact Subtype.ext (inner_self_eq_zero.mp (hall (u : EucSpace d) u.2))
    have hfinS : Module.finrank ℝ S = n := by
      rw [hS, Module.finrank_eq_card_basis (Module.Basis.span hX)]
      simp
    have hsurj : Function.Surjective g :=
      (LinearMap.injective_iff_surjective_of_finrank_eq_finrank
        (by rw [hfinS]; simp)).mp hginj
    obtain ⟨u, hu⟩ := hsurj (fun _ => (1 : ℝ))
    exact ⟨(u : EucSpace d), fun i => congrFun hu i⟩
  obtain ⟨w₀, hw₀⟩ := hone
  -- Step 2: `w₀ ≠ 0`, so it can be normalized.
  have hw₀ne : w₀ ≠ 0 := by
    rintro rfl
    simpa using hw₀ ⟨0, hn⟩
  have hnpos : (0 : ℝ) < ‖w₀‖ := norm_pos_iff.mpr hw₀ne
  refine ⟨⟨‖w₀‖⁻¹ • w₀, ?_⟩, fun i => ?_⟩
  · rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, Real.norm_eq_abs,
      abs_of_pos hnpos, inv_mul_cancel₀ (ne_of_gt hnpos)]
  · show (0 : ℝ) < inner (𝕜 := ℝ) (X i) (‖w₀‖⁻¹ • w₀)
    rw [real_inner_smul_right, hw₀ i, mul_one]
    exact inv_pos.mpr hnpos

/-- The hypotheses of `exists_common_hemisphere_of_linearIndependent` are
satisfiable: the single standard basis vector of `ℝ^1` is a linearly
independent family of one vector. -/
example : LinearIndependent ℝ (fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ∧
    1 ≤ 1 := by
  refine ⟨?_, le_refl 1⟩
  rw [linearIndependent_unique_iff]
  simp

/-- **`n ≤ d` uniform points lie in an open hemisphere almost surely**: there
is `w ∈ 𝕊^{d-1}` with `⟨x_i, w⟩ > 0` for every `i`.  The survey's "weak
version of Wendel's theorem", seen directly: the points are almost surely
linearly independent (`ae_linearIndependent_of_uniformTuple`), and independent
points lie in a common open hemisphere
(`exists_common_hemisphere_of_linearIndependent`).

At `d = n` this is the case `P = 2^{-(n-1)} Σ_{k<n} C(n-1, k) = 1` of `wendel`.
The survey's `n ≥ 1` is not needed: for `n = 0` the claim asks only for a
point of the sphere, and the uniform law, a probability measure on it, gives
one.

Source: arXiv:2312.10794v5, §6.1, the paragraph after `thm: d.infty` and its
footnote; arXiv:2512.01868v4, §4, `cor: d-ge-n` ("they lie in some open
hemisphere almost surely"). -/
theorem ae_exists_openHemisphere (hnd : n ≤ d) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      ∀ᵐ X ∂P, ∃ w : SSphere d, ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) (X i : EucSpace d) (w : EucSpace d) := by
  intro P hP
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · obtain ⟨σ, _, -, -⟩ := hP
    obtain ⟨w⟩ := nonempty_of_isProbabilityMeasure σ
    exact Filter.Eventually.of_forall fun _ => ⟨w, fun i => i.elim0⟩
  filter_upwards [ae_linearIndependent_of_uniformTuple d hnd P hP] with X hX
  exact exists_common_hemisphere_of_linearIndependent d _ hX hn

/-- The hypothesis of `ae_exists_openHemisphere` is satisfiable: `n = d = 1`;
the uniform law is quantified over. -/
example : 1 ≤ 1 := le_rfl

end Perspective
end Transformer
