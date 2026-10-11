/-
# Rotation equivariance of the actual spherical attention flow

ArXiv:2312.10794v5, §2, `SA`; arXiv:2411.04551v3, §5,
the rotation argument in `prop:generic-discrete`.

A common orthogonal transformation preserves all attention scores and
commutes with the weighted average and tangent projection. Applying it
to a trajectory therefore gives a trajectory from the rotated input.
Uniqueness of the proved global flow gives its equivariance.

These are identities of the original dynamics, for every real temperature.
The sphere action is continuous and measurable; no rotation property is
assumed of the trajectory chosen by the existence theorem.
-/

import Transformer.Perspective.Section2_FlowContinuity
import Transformer.Perspective.SphereInvariant

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perspective

variable {d n : ℕ}

/-- Apply a common orthogonal transformation to every spherical token.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
def sphereTupleMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (X : SphereTuple d n) : SphereTuple d n := fun i => sphereMap d U (X i)

/-- The spherical rotation action preserves distances.
Source: arXiv:2411.04551v3, §5, its action of `O(d)`. -/
theorem isometry_sphereMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Isometry (sphereMap d U) := by
  apply Isometry.of_dist_eq
  intro x y
  simp only [Subtype.dist_eq, sphereMap, LinearIsometryEquiv.dist_map]

/-- The common rotation preserves the maximum token distance.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem isometry_sphereTupleMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Isometry (sphereTupleMap (n := n) U) :=
  (isometry_sphereMap U).postcomp_pi

/-- The rotation action on tuples is continuous.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem continuous_sphereTupleMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Continuous (sphereTupleMap (n := n) U) :=
  (isometry_sphereTupleMap U).continuous

/-- The rotation action on tuples is measurable for the Borel laws.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem measurable_sphereTupleMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Measurable (sphereTupleMap (n := n) U) :=
  (continuous_sphereTupleMap U).measurable

/-- The inverse orthogonal transformation recovers every token.
Source: arXiv:2411.04551v3, §5, the common rotation of the inputs. -/
theorem sphereTupleMap_symm_apply (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (X : SphereTuple d n) : sphereTupleMap U.symm (sphereTupleMap U X) = X := by
  funext i
  apply Subtype.ext
  exact U.symm_apply_apply (X i)

/-- Applying a rotation after its inverse also recovers the tuple.
Source: arXiv:2411.04551v3, §5, the action of `O(d)`. -/
theorem sphereTupleMap_apply_symm (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (X : SphereTuple d n) : sphereTupleMap U (sphereTupleMap U.symm X) = X := by
  funext i
  apply Subtype.ext
  exact U.apply_symm_apply (X i)

/-- The identity orthogonal transformation fixes the input tuple.
Source: arXiv:2411.04551v3, §5, the action of `O(d)`. -/
theorem sphereTupleMap_refl (X : SphereTuple d n) :
    sphereTupleMap (LinearIsometryEquiv.refl ℝ (EucSpace d)) X = X := by
  funext i
  apply Subtype.ext
  rfl

/-- The action respects composition of the ambient orthogonal maps.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem sphereTupleMap_trans (U V : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (X : SphereTuple d n) :
    sphereTupleMap (U.trans V) X = sphereTupleMap V (sphereTupleMap U X) := by
  funext i
  apply Subtype.ext
  rfl

/-- A common orthogonal transformation does not identify distinct inputs.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem injective_sphereTupleMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Function.Injective (sphereTupleMap (n := n) U) := by
  intro X Y h
  have heq := congrArg (sphereTupleMap U.symm) h
  simpa only [sphereTupleMap_symm_apply] using heq

/-- A common rotation leaves each attention score unchanged.
Source: arXiv:2312.10794v5, §2, the inner products in `SA`. -/
theorem saScore_rotate (β : ℝ) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (Y : Idx n → EucSpace d) (i : Idx n) :
    saScore d n β (fun j => U (Y j)) i = saScore d n β Y i := by
  funext j
  simp only [saScore, LinearIsometryEquiv.inner_map_map]

/-- The tangent-projection formula commutes with an orthogonal map,
including in the ambient space. Source: arXiv:2312.10794v5, §2, `SA`. -/
theorem proj_rotate (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (x y : EucSpace d) :
    proj d (U x) (U y) = U (proj d x y) := by
  simp only [proj, LinearIsometryEquiv.inner_map_map, map_sub, map_smul]

/-- The original attention vector field is rotation equivariant.
Source: arXiv:2312.10794v5, §2, `SA`; arXiv:2411.04551v3, §5,
`prop:generic-discrete`. -/
theorem saField_rotate (β : ℝ) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (Y : Idx n → EucSpace d) (i : Idx n) :
    saField d n β (fun j => U (Y j)) i = U (saField d n β Y i) := by
  have hsum : U (∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • Y j) =
      ∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • U (Y j) := by
    rw [map_sum]
    simp only [map_smul]
  rw [saField, saScore_rotate, ← hsum, proj_rotate]
  rfl

/-- Rotating an actual solution gives an actual solution of the same ODE.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`;
the underlying equation is `SA` in arXiv:2312.10794v5, §2. -/
theorem SA_sphereTupleMap (β : ℝ) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (X : ℝ → SphereTuple d n) (hX : SA d n β X) :
    SA d n β (fun t => sphereTupleMap U (X t)) := by
  apply (SA_iff d n β _).mpr
  intro t i
  have hderiv := U.hasFDerivAt.comp_hasDerivAt t ((SA_iff d n β X).mp hX t i)
  simpa only [Function.comp_def, sphereTupleMap, sphereMap, saField_rotate,
    LinearIsometryEquiv.coe_coe''] using hderiv

/-- The solution hypothesis is realized by nonzero stationary unit tokens,
for every temperature, before any rotation is applied. -/
example (β : ℝ) (x : SSphere 1) : SA 1 1 β (fun _ _ => x) :=
  SA_const_consensus 1 1 one_pos β x

/-- **The actual global attention flow is rotation equivariant.**
This supplies the deterministic-flow identity used in the atomlessness
argument of arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem saFlow_sphereTupleMap (β : ℝ) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (X₀ : SphereTuple d n) (t : ℝ) :
    saFlow β (sphereTupleMap U X₀) t = sphereTupleMap U (saFlow β X₀ t) := by
  have heq := SA_unique β (saFlow β (sphereTupleMap U X₀))
    (fun s => sphereTupleMap U (saFlow β X₀ s))
    (saFlow_spec β (sphereTupleMap U X₀)).2
    (SA_sphereTupleMap β U _ (saFlow_spec β X₀).2)
    (by rw [saFlow_zero, saFlow_zero])
  exact congrFun heq t

end Transformer.Perspective
