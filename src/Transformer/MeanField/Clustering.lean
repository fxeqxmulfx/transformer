/-
# Mean-Field Dynamics — clustering & rates (§4 of 2512.01868v4)

* `Theorem thm: clustering_finite`  — global clustering for `d ≥ 3` (revisits
                                       `thm: boumal`),
* `Corollary cor: d-ge-n`           — uniform random init when `d ≥ n`.

`Theorem thm: mfclust` is in `MeanField.GlobalRate`.

`Theorem thm: cone-collapse` is `Perspective.cone_collapse`
(in `Perspective.Section5_ExpRate`) verbatim and is not restated here.

`thm: clustering_finite` is read against the uniform law
`Perspective.UniformTuple`, and is not proved.  What *is* proved is the deterministic core of
`cor: d-ge-n`: linearly independent particles lie in a common open hemisphere,
which is the hypothesis `thm: cone-collapse` runs on, and which `n` points in
dimension `d ≥ n` satisfy almost surely.  The almost-sure half is not
formalized.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap
import Transformer.Perspective.Section2_GradientFlow
import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.Section5_HighD
import Transformer.MeanField.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace MeanField

open MeanField Perspective

variable (d n : ℕ)

/-- **Theorem (thm: clustering_finite).** *Global clustering for `d ≥ 3`.*

(Markdahl–Thunberg–Boumal; Criscitiello–Boumal; Geshkovski et al.)

For both `eq: SA` and `eq: USA` with `n ≥ 2` particles in dimension `d ≥ 3`
and `β ≥ 0`, for almost every initial `(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n` the
trajectories exist globally and converge to a single cluster:

  `lim_{t → ∞} ‖x_i(t) - x_j(t)‖ = 0`.

"Almost every" is with respect to the uniform law on `(𝕊^{d-1})^n`,
`Perspective.UniformTuple`.  Against an arbitrary reference measure the claim
is false: the Dirac mass at an antipodal pair, a stationary configuration of
both dynamics, is a counterexample.  The two dynamics are one conjunction, as
the source states them in one theorem; for each, global existence is the first
half and convergence of every solution the second.

Not proved here.

Source: arXiv:2512.01868v4, §4, `thm:clustering_finite`. -/
theorem global_clustering (β : ℝ) (hd : 3 ≤ d) (hn : 2 ≤ n) (hβ : 0 ≤ β) :
    ∀ P : Measure (SphereTuple d n), Perspective.UniformTuple d n P →
    ∀ᵐ X₀ ∂P,
      ((∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ Perspective.SA d n β X) ∧
        ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
          ∀ i j : Idx n,
            Filter.Tendsto
              (fun t : ℝ => ‖(X t i : EucSpace d) - (X t j : EucSpace d)‖)
              Filter.atTop (nhds 0)) ∧
      ((∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ Perspective.USA d n β X) ∧
        ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.USA d n β X →
          ∀ i j : Idx n,
            Filter.Tendsto
              (fun t : ℝ => ‖(X t i : EucSpace d) - (X t j : EucSpace d)‖)
              Filter.atTop (nhds 0)) := by
  sorry

/-- The hypotheses of `global_clustering` are satisfiable: `d = 3`, `n = 2`,
`β = 0`. -/
example : 3 ≤ 3 ∧ 2 ≤ 2 ∧ (0 : ℝ) ≤ 0 := ⟨le_rfl, le_rfl, le_rfl⟩

/-- **The deterministic core of `cor: d-ge-n`.**

If the particles `x_1, …, x_n` are linearly independent — which forces
`n ≤ d` — then they lie in a common *open* hemisphere: there is a unit vector
`w` with `⟨x_i, w⟩ > 0` for every `i`.

The map `u ↦ (⟨x_i, u⟩)_i` from the span of the `x_i` to `ℝ^n` is injective,
because a vector of the span orthogonal to every `x_i` is orthogonal to itself;
the span has dimension `n`, so the map is onto, and the preimage of the
all-ones vector is the required `w` up to normalization.

This is the hypothesis `thm: cone-collapse` (`Perspective.cone_collapse`) runs
on — so `cor: d-ge-n` follows from it
together with the fact that `n` i.i.d. uniform points in dimension `d ≥ n` are
almost surely independent, which is not formalized.
Source: arXiv:2512.01868v4, §4. -/
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
  have hi₀ : Nonempty (Idx n) := ⟨⟨0, hn⟩⟩
  obtain ⟨i₀⟩ := hi₀
  have hw₀ne : w₀ ≠ 0 := by
    intro h
    have := hw₀ i₀
    rw [h, inner_zero_right] at this
    exact zero_ne_one this
  have hnpos : (0 : ℝ) < ‖w₀‖ := norm_pos_iff.mpr hw₀ne
  refine ⟨⟨‖w₀‖⁻¹ • w₀, ?_⟩, fun i => ?_⟩
  · rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, Real.norm_eq_abs,
      abs_of_pos hnpos, inv_mul_cancel₀ (ne_of_gt hnpos)]
  · show (0 : ℝ) < inner (𝕜 := ℝ) (X i) (‖w₀‖⁻¹ • w₀)
    rw [real_inner_smul_right, hw₀ i, mul_one]
    exact inv_pos.mpr hnpos

/-- The hypotheses are satisfiable: the single standard basis vector of `ℝ^1`
is a linearly independent family of one vector. -/
example : LinearIndependent ℝ (fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ∧
    1 ≤ 1 := by
  refine ⟨?_, le_refl 1⟩
  rw [linearIndependent_unique_iff]
  simp

end MeanField
end Transformer
