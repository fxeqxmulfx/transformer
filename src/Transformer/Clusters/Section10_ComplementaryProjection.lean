/-
# The two projections in the paranormal splitting

ArXiv:2305.05465v6, §5, `d:goodmulti`, and §10, the opening of the proof
of `t:multiplicity`. The manuscript projects onto the leading eigenspace
parallel to its invariant complement. The splitting constructs this
projection as a continuous linear map. Its complementary projection is
the identity minus the leading one, and it commutes with the value map.

These are projections along the specified direct sum. Orthogonality of
the summands is not required for any statement below.
-/

import Transformer.Clusters.Section10_ProjectedDynamics
import Mathlib.LinearAlgebra.Projection

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {d : ℕ}

/-- The direct-sum splitting constructs the actual continuous projection
onto the leading space along its complement. Source: arXiv:2305.05465v6,
§10, the definitions of `π_F` and `π_G` in the first paragraph. -/
theorem exists_isProjOnto (F G : Submodule ℝ (EucSpace d)) (hFG : IsCompl F G) :
    ∃ proj : ParamMatrix d, IsProjOnto proj F G := by
  let proj : ParamMatrix d := LinearMap.toContinuousLinearMap (F.projection G hFG)
  refine ⟨proj, ?_, ?_, ?_⟩
  · intro x
    change ((F.projectionOnto G hFG x : F) : EucSpace d) ∈ F
    exact (F.projectionOnto G hFG x).property
  · intro x hx
    change ((F.projectionOnto G hFG x : F) : EucSpace d) = x
    rw [Submodule.projectionOnto_apply_of_mem_left hFG hx]
  · intro x hx
    change ((F.projectionOnto G hFG x : F) : EucSpace d) = 0
    rw [(Submodule.projectionOnto_apply_eq_zero_iff hFG).mpr hx]
    rfl

/-- The direct-sum hypothesis is realized by the full space and its zero
complement. Source: arXiv:2305.05465v6, §5, `d:goodmulti`. -/
example : IsCompl (⊤ : Submodule ℝ (EucSpace d)) ⊥ := isCompl_top_bot

/-- Subtracting the leading projection leaves precisely a vector in the
complement. Source: arXiv:2305.05465v6, §10, the `F ⊕ G` decomposition. -/
theorem IsProjOnto.mem_complement (proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (hproj : IsProjOnto proj F G)
    (hFG : IsCompl F G) (x : EucSpace d) : x - proj x ∈ G := by
  obtain ⟨u, v, hu, hv, huv⟩ :=
    Submodule.codisjoint_iff_exists_add_eq.mp hFG.codisjoint x
  rw [← huv, map_add, hproj.2.1 u hu, hproj.2.2 v hv, add_zero]
  simpa only [add_sub_cancel_left] using hv

/-- The projection and complement hypotheses coexist in the identity
case. Source: arXiv:2305.05465v6, §5, `d:goodmulti`. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsCompl (⊤ : Submodule ℝ (EucSpace d)) ⊥ :=
  ⟨isProjOnto_one d, isCompl_top_bot⟩

/-- Applying the leading projection twice has the same value as applying
it once. Source: arXiv:2305.05465v6, §10, the specified projection `π_F`. -/
theorem IsProjOnto.apply_twice (proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (hproj : IsProjOnto proj F G)
    (x : EucSpace d) : proj (proj x) = proj x := by
  exact hproj.2.1 (proj x) (hproj.1 x)

/-- The idempotence input is realized by the full-space projection.
Source: arXiv:2305.05465v6, §10, `π_F`. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ := isProjOnto_one d

/-- The leading and complementary parts reconstruct the original vector.
Source: arXiv:2305.05465v6, §10, the decomposition along `F ⊕ G`. -/
theorem projected_add_complement (proj : ParamMatrix d) (x : EucSpace d) :
    proj x + (1 - proj) x = x := by
  simp only [sub_apply, one_apply_eq_self]
  abel

/-- The complementary component has zero leading projection.
Source: arXiv:2305.05465v6, §10, the projection `π_G` parallel to `F`. -/
theorem IsProjOnto.apply_complement_eq_zero (proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (hproj : IsProjOnto proj F G)
    (x : EucSpace d) : proj ((1 - proj) x) = 0 := by
  simp only [sub_apply, one_apply_eq_self, map_sub]
  rw [hproj.apply_twice proj F G, sub_self]

/-- The complete projection input holds for the identity projection.
Source: arXiv:2305.05465v6, §10, `π_G`. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ := isProjOnto_one d

/-- The complementary projection is the identity minus the leading one.
Source: arXiv:2305.05465v6, §10, the specified projection `π_G`. -/
theorem IsProjOnto.complement (proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (hproj : IsProjOnto proj F G)
    (hFG : IsCompl F G) : IsProjOnto (1 - proj) G F := by
  refine ⟨?_, ?_, ?_⟩
  · intro x
    simpa only [sub_apply, one_apply_eq_self] using
      hproj.mem_complement proj F G hFG x
  · intro x hx
    simp only [sub_apply, one_apply_eq_self, hproj.2.2 x hx, sub_zero]
  · intro x hx
    simp only [sub_apply, one_apply_eq_self, hproj.2.1 x hx, sub_self]

/-- The full inputs for the complementary projection hold with `F` the
whole space. Source: arXiv:2305.05465v6, §10, `π_G`. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsCompl (⊤ : Submodule ℝ (EucSpace d)) ⊥ :=
  ⟨isProjOnto_one d, isCompl_top_bot⟩

/-- The value map on a leading projected vector is multiplication by its
positive eigenvalue. Source: arXiv:2305.05465v6, §5, `d:goodmulti` (ii). -/
theorem IsProjOnto.value_apply (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) : V (proj x) = lam • proj x := by
  exact hV.2.2.2.2.1 (proj x) (hproj.1 x)

/-- Projection and paranormality hold together for the identity value
map. Source: arXiv:2305.05465v6, §5, `d:goodmulti` (ii). -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

/-- The leading projection commutes with the paranormal value map.
Source: arXiv:2305.05465v6, §10, the projected ODE, using the invariance
of both summands in `d:goodmulti`. -/
theorem IsProjOnto.apply_value_eq_value_apply (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) : proj (V x) = V (proj x) := by
  rw [hproj.apply_value proj V F G lam hV, hproj.value_apply proj V F G lam hV]

/-- The full commutation inputs are realized by the identity map.
Source: arXiv:2305.05465v6, §10, the projected ODE. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

/-- The complementary projection also commutes with the value map,
so the adapted metric can bound its actual token dynamics.
Source: arXiv:2305.05465v6, §9, `e:notdi`, and §10, `π_G`. -/
theorem IsProjOnto.complement_apply_value (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) : (1 - proj) (V x) = V ((1 - proj) x) := by
  simp only [sub_apply, one_apply_eq_self, map_sub]
  rw [hproj.apply_value_eq_value_apply proj V F G lam hV]

/-- All complementary commutation inputs coexist for a good triple.
Source: arXiv:2305.05465v6, §5, `d:goodmulti`, and §10, `π_G`. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

end Transformer.Clusters
