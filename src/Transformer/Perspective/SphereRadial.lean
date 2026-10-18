/-
# The sphere flows as slices of a flow on `(ℝ^d ∖ 0)^n`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

A vector field `V` on configurations `(𝕊^{d-1})^n` extends to tuples of vectors
of `ℝ^d` as `F(x)_i = ‖x_i‖ V(x/‖x‖)_i` (`radialField`).  The extension has the
flow of `V` on every slice `{(r_i u_i)_i : u ∈ (𝕊^{d-1})^n}`, `r_i > 0`:

* `X` solves `u̇ = V(u)` exactly when `t ↦ (r_i X(t)_i)_i` solves `ẋ = F(x)`
  (`hasDerivAt_radialField_iff`);
* if `V` is tangent, `⟨u_i, V(u)_i⟩ = 0`, every solution of `ẋ = F(x)` keeps the
  norms `‖x_i‖` (`norm_eq_of_radialField`); one that starts from nonzero vectors
  stays on its slice, and its directions solve `u̇ = V(u)`
  (`hasDerivAt_tupleNormalize_of_radialField`).

The fields of `USA` and `SA` are tangent, so both flows are the slices of their
radial extensions (`usa_iff_radialField`, `usa_of_radialField`, and the same for
`SA`): the bridge M5.3 of `landscape_plan.md`, in the vector space `(ℝ^d)^n`.
`Perspective.SphereRadialGradient` writes the extension of `USA` as
`n⁻¹ ‖x_i‖² ∇_i(𝖤_β ∘ normalize)`; `Perspective.SphereNull` matches the null sets.

Source: arXiv:2312.10794v5, §2 (`eq: USA`, `eq: SA`), and Appendix A, where the
flows are gradient flows on the manifold `(𝕊^{d-1})^n`.
-/

import Transformer.Perspective.SAField
import Transformer.Perspective.UniformLipschitz

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The radial extension of a field of configurations:**
`F(x)_i = ‖x_i‖ V(x/‖x‖)_i`. -/
noncomputable def radialField (hd : 1 ≤ d) (V : SphereTuple d n → Idx n → EucSpace d)
    (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  ‖x i‖ • V (tupleNormalize hd x) i

/-- Radial projection undoes a positive rescaling of a configuration. -/
theorem tupleNormalize_smul (hd : 1 ≤ d) {r : Idx n → ℝ} (hr : ∀ i, 0 < r i)
    (X : SphereTuple d n) : tupleNormalize hd (fun i => r i • (X i : EucSpace d)) = X := by
  funext i
  apply Subtype.ext
  have hX : ‖(X i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X i).2
  have hne : r i • (X i : EucSpace d) ≠ 0 :=
    smul_ne_zero (hr i).ne' (ne_zero_of_mem_unit_sphere (X i))
  show (sphereNormalize hd (r i • (X i : EucSpace d)) : EucSpace d) = X i
  rw [coe_sphereNormalize hd hne, norm_smul, hX, Real.norm_eq_abs, abs_of_pos (hr i), mul_one,
    smul_smul, inv_mul_cancel₀ (hr i).ne', one_smul]

/-- On the slice of radii `r`, the radial extension is `V` scaled by `r`. -/
theorem radialField_smul (hd : 1 ≤ d) (V : SphereTuple d n → Idx n → EucSpace d)
    {r : Idx n → ℝ} (hr : ∀ i, 0 < r i) (X : SphereTuple d n) (i : Idx n) :
    radialField hd V (fun j => r j • (X j : EucSpace d)) i = r i • V X i := by
  rw [radialField, tupleNormalize_smul hd hr, norm_smul, mem_sphere_zero_iff_norm.mp (X i).2,
    Real.norm_eq_abs, abs_of_pos (hr i), mul_one]

/-- **The sphere flow is the slice of its radial extension:** `X` solves
`u̇ = V(u)` exactly when `t ↦ (r_i X(t)_i)_i` solves `ẋ = F(x)`. -/
theorem hasDerivAt_radialField_iff (hd : 1 ≤ d) (V : SphereTuple d n → Idx n → EucSpace d)
    {r : Idx n → ℝ} (hr : ∀ i, 0 < r i) (X : ℝ → SphereTuple d n) :
    (∀ t i, HasDerivAt (fun s => (X s i : EucSpace d)) (V (X t) i) t) ↔
      ∀ t i, HasDerivAt (fun s => r i • (X s i : EucSpace d))
        (radialField hd V (fun j => r j • (X t j : EucSpace d)) i) t := by
  simp_rw [radialField_smul hd V hr]
  refine ⟨fun h t i => (h t i).const_smul (r i), fun h t i => ?_⟩
  convert (h t i).const_smul (r i)⁻¹ using 1
  · funext s
    simp [smul_smul, inv_mul_cancel₀ (hr i).ne']
  · simp [smul_smul, inv_mul_cancel₀ (hr i).ne']

/-- **Tangent fields keep the norms:** if `⟨u_i, V(u)_i⟩ = 0`, every solution of
`ẋ = F(x)` has `‖x_i(t)‖ = ‖x_i(0)‖`, since `d/dt ‖x_i‖² = 2 ‖x_i‖ ⟨x_i, V(u)_i⟩`. -/
theorem norm_eq_of_radialField (hd : 1 ≤ d) {V : SphereTuple d n → Idx n → EucSpace d}
    (hV : ∀ U i, inner (𝕜 := ℝ) (U i : EucSpace d) (V U i) = 0)
    {x : ℝ → Idx n → EucSpace d}
    (hx : ∀ t i, HasDerivAt (fun s => x s i) (radialField hd V (x t) i) t) (t : ℝ) (i : Idx n) :
    ‖x t i‖ = ‖x 0 i‖ := by
  have hderiv : ∀ s, HasDerivAt (fun s => ‖x s i‖ ^ 2) 0 s := by
    intro s
    convert (hx s i).norm_sq using 1
    rw [radialField, real_inner_smul_right]
    by_cases h0 : x s i = 0
    · simp [h0]
    · have h := hV (tupleNormalize hd (x s)) i
      rw [show (tupleNormalize hd (x s) i : EucSpace d) = ‖x s i‖⁻¹ • x s i from
        coe_sphereNormalize hd h0, real_inner_smul_left, mul_eq_zero] at h
      rcases h with h | h
      · exact absurd h (inv_ne_zero (norm_ne_zero_iff.2 h0))
      · rw [h, mul_zero, mul_zero]
  have h := is_const_of_deriv_eq_zero (fun s => (hderiv s).differentiableAt)
    (fun s => (hderiv s).deriv) t 0
  exact (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).1 h

/-- **Off the slices nothing new happens:** if `V` is tangent, the directions of a
solution of `ẋ = F(x)` from nonzero vectors solve `u̇ = V(u)`. -/
theorem hasDerivAt_tupleNormalize_of_radialField (hd : 1 ≤ d)
    {V : SphereTuple d n → Idx n → EucSpace d}
    (hV : ∀ U i, inner (𝕜 := ℝ) (U i : EucSpace d) (V U i) = 0) {x : ℝ → Idx n → EucSpace d}
    (hx : ∀ t i, HasDerivAt (fun s => x s i) (radialField hd V (x t) i) t)
    (h0 : ∀ i, x 0 i ≠ 0) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => (tupleNormalize hd (x s) i : EucSpace d))
      (V (tupleNormalize hd (x t)) i) t := by
  have hn : ∀ s, ‖x s i‖ = ‖x 0 i‖ := fun s => norm_eq_of_radialField hd hV hx s i
  have h0' : ‖x 0 i‖ ≠ 0 := norm_ne_zero_iff.2 (h0 i)
  have hfun : (fun s => (tupleNormalize hd (x s) i : EucSpace d)) =
      fun s => ‖x 0 i‖⁻¹ • x s i := by
    funext s
    rw [← hn s]
    exact coe_sphereNormalize hd (norm_ne_zero_iff.1 (by rw [hn s]; exact h0'))
  rw [hfun]
  convert (hx t i).const_smul ‖x 0 i‖⁻¹ using 1
  rw [radialField, smul_smul, hn t, inv_mul_cancel₀ h0', one_smul]

/-- The hypotheses of the theorems above hold for the zero field, a constant
solution from the first basis vector of `ℝ³`, and the radii `1`. -/
example : (∀ i : Idx 2, (0 : ℝ) < (fun _ => 1) i) ∧
    (∀ (U : SphereTuple 3 2) i, inner (𝕜 := ℝ) (U i : EucSpace 3) ((fun _ _ => 0) U i) = 0) ∧
    (∀ t i, HasDerivAt (fun _ : ℝ => (fun _ : Idx 2 => (basePoint 2 : EucSpace 3)) i)
      (radialField (by norm_num) (fun _ _ => 0) (fun _ => (basePoint 2 : EucSpace 3)) i) t) ∧
    ∀ i : Idx 2, (fun _ : Idx 2 => (basePoint 2 : EucSpace 3)) i ≠ 0 :=
  ⟨fun _ => one_pos, fun _ _ => inner_zero_right _, fun t _ => by
    simpa [radialField] using hasDerivAt_const t (basePoint 2 : EucSpace 3),
    fun _ => ne_zero_of_mem_unit_sphere (basePoint 2)⟩

/-- The field of `USA` on configurations:
`u ↦ (Proj_{u_i}(n⁻¹ Σ_j e^{β⟨u_i, u_j⟩} u_j))_i`. -/
noncomputable def usaField (β : ℝ) (U : SphereTuple d n) (i : Idx n) : EucSpace d :=
  proj d (U i) (((n : ℝ)⁻¹) • ∑ j : Idx n,
    Real.exp (β * inner (𝕜 := ℝ) (U i : EucSpace d) (U j : EucSpace d)) • (U j : EucSpace d))

/-- The field of `USA` is tangent to the spheres. -/
theorem inner_usaField (β : ℝ) (U : SphereTuple d n) (i : Idx n) :
    inner (𝕜 := ℝ) (U i : EucSpace d) (usaField β U i) = 0 :=
  inner_proj_eq_zero (mem_sphere_zero_iff_norm.mp (U i).2) _

/-- The field of `SA` is tangent to the spheres. -/
theorem inner_saField (β : ℝ) (U : SphereTuple d n) (i : Idx n) :
    inner (𝕜 := ℝ) (U i : EucSpace d) (saField d n β (fun j => (U j : EucSpace d)) i) = 0 :=
  inner_proj_eq_zero (mem_sphere_zero_iff_norm.mp (U i).2) _

/-- **`USA` is the slice of its radial extension** at all radii `r_i > 0`. -/
theorem usa_iff_radialField (hd : 1 ≤ d) (β : ℝ) {r : Idx n → ℝ} (hr : ∀ i, 0 < r i)
    (X : ℝ → SphereTuple d n) :
    USA d n β X ↔ ∀ t i, HasDerivAt (fun s => r i • (X s i : EucSpace d))
      (radialField hd (usaField β) (fun j => r j • (X t j : EucSpace d)) i) t :=
  hasDerivAt_radialField_iff hd (usaField β) hr X

/-- **`SA` is the slice of its radial extension** at all radii `r_i > 0`. -/
theorem sa_iff_radialField (hd : 1 ≤ d) (β : ℝ) {r : Idx n → ℝ} (hr : ∀ i, 0 < r i)
    (X : ℝ → SphereTuple d n) :
    SA d n β X ↔ ∀ t i, HasDerivAt (fun s => r i • (X s i : EucSpace d))
      (radialField hd (fun U => saField d n β fun j => (U j : EucSpace d))
        (fun j => r j • (X t j : EucSpace d)) i) t :=
  (SA_iff d n β X).trans
    (hasDerivAt_radialField_iff hd (fun U => saField d n β fun j => (U j : EucSpace d)) hr X)

/-- **The radial extension of `USA` has no other flows:** a solution from nonzero
vectors keeps its norms, and its directions solve `USA`. -/
theorem usa_of_radialField (hd : 1 ≤ d) (β : ℝ) {x : ℝ → Idx n → EucSpace d}
    (hx : ∀ t i, HasDerivAt (fun s => x s i) (radialField hd (usaField β) (x t) i) t)
    (h0 : ∀ i, x 0 i ≠ 0) :
    USA d n β (fun t => tupleNormalize hd (x t)) ∧ ∀ t i, ‖x t i‖ = ‖x 0 i‖ :=
  ⟨hasDerivAt_tupleNormalize_of_radialField hd (inner_usaField β) hx h0,
    norm_eq_of_radialField hd (inner_usaField β) hx⟩

/-- **The radial extension of `SA` has no other flows:** a solution from nonzero
vectors keeps its norms, and its directions solve `SA`. -/
theorem sa_of_radialField (hd : 1 ≤ d) (β : ℝ) {x : ℝ → Idx n → EucSpace d}
    (hx : ∀ t i, HasDerivAt (fun s => x s i)
      (radialField hd (fun U => saField d n β fun j => (U j : EucSpace d)) (x t) i) t)
    (h0 : ∀ i, x 0 i ≠ 0) :
    SA d n β (fun t => tupleNormalize hd (x t)) ∧ ∀ t i, ‖x t i‖ = ‖x 0 i‖ :=
  ⟨(SA_iff d n β _).2 (hasDerivAt_tupleNormalize_of_radialField hd (inner_saField β) hx h0),
    norm_eq_of_radialField hd (inner_saField β) hx⟩

/-- The hypotheses of the four theorems above hold for one token at the base point
of `𝕊²`, which neither flow moves, and the radius `1`. -/
example : (∀ i : Idx 1, (0 : ℝ) < (fun _ => 1) i) ∧
    (∀ t i, HasDerivAt (fun _ : ℝ => (fun _ : Idx 1 => (basePoint 2 : EucSpace 3)) i)
      (radialField (by norm_num) (usaField 1) (fun _ => (basePoint 2 : EucSpace 3)) i) t) ∧
    (∀ t i, HasDerivAt (fun _ : ℝ => (fun _ : Idx 1 => (basePoint 2 : EucSpace 3)) i)
      (radialField (by norm_num) (fun U => saField 3 1 1 fun j => (U j : EucSpace 3))
        (fun _ => (basePoint 2 : EucSpace 3)) i) t) ∧
    ∀ i : Idx 1, (fun _ : Idx 1 => (basePoint 2 : EucSpace 3)) i ≠ 0 := by
  have hu : ∀ (U : SphereTuple 3 1) j, ‖(U j : EucSpace 3)‖ = 1 :=
    fun U j => mem_sphere_zero_iff_norm.mp (U j).2
  refine ⟨fun _ => one_pos, fun t i => ?_, fun t i => ?_,
    fun _ => ne_zero_of_mem_unit_sphere (basePoint 2)⟩
  · obtain rfl : i = 0 := Subsingleton.elim i 0
    convert hasDerivAt_const t (basePoint 2 : EucSpace 3) using 1
    rw [radialField, usaField, Fin.sum_univ_one, smul_smul, proj_smul_self (hu _ 0), smul_zero]
  · convert hasDerivAt_const t (basePoint 2 : EucSpace 3) using 1
    rw [radialField, saField_of_one 3 1 (hu _) i, smul_zero]

end Perspective
end Transformer
