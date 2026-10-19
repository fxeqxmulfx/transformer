module

public import Transformer.Clusters.Section10_ProjHull

/-!
# The projected hull shrinks along the actual dynamics

ArXiv:2305.05465v6, §10, opening of the proof of `t:multiplicity`.
The projection onto the leading eigenspace parallel to its invariant
complement satisfies `π_F V = λ π_F`. Applying it to `e:Rres` gives a
consensus equation with positive speed `λ` and nonnegative attention
weights. At a token maximizing any linear functional, its projected
derivative is nonpositive. Separation then proves nesting of the
projected convex hull itself.

No nesting of the full, unprojected hull is assumed, and no commutation
of the attention root with the projection is used. The conclusion follows
from the actual rescaled dynamics and the paranormal value-map splitting.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- The leading projection intertwines the value map with multiplication
by `λ`. Source: arXiv:2305.05465v6, §10, projected equation after the first
paragraph, and §5, `d:goodmulti` (ii). The complement is invariant. -/
theorem IsProjOnto.apply_value (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam) (x : EucSpace d) :
    proj (V x) = lam • proj x := by
  obtain ⟨u, v, hu, hv, huv⟩ :=
    Submodule.codisjoint_iff_exists_add_eq.mp hV.2.2.1.codisjoint x
  calc
    proj (V x) = proj (V u) + proj (V v) := by rw [← huv, map_add, map_add]
    _ = lam • u := by
      rw [hV.2.2.2.2.1 u hu, map_smul, hproj.2.1 u hu,
        hproj.2.2 (V v) (hV.2.1 v hv), add_zero]
    _ = lam • proj x := by
      rw [← huv, map_add, hproj.2.1 u hu, hproj.2.2 v hv, add_zero]

/-- The projection and full splitting hypotheses coexist for the identity
value map. Source: arXiv:2305.05465v6, §5, `d:goodmulti` (ii). -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

/-- Applying the leading projection to the actual ODE gives a consensus
drift with speed `λ`. Source: arXiv:2305.05465v6, §10, the first projected
equation in the proof of `t:multiplicity`, and §3, `e:Rres`. -/
theorem RescaledDynamics.hasDerivAt_projected (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z)
    (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => proj (Z s i))
      (∑ j : Idx n, attentionMatrix Q K (fun l => expTime V t (Z t l)) i j •
        (lam • (proj (Z t j) - proj (Z t i)))) t := by
  have h := (proj.hasFDerivAt (x := Z t i)).comp_hasDerivAt t (hZ t i)
  simpa only [Function.comp_def, map_sum, map_smul,
    hproj.apply_value proj V F G lam hV, map_sub, smul_sub] using h

/-- Every dynamical and projection input holds for two stationary unit
tokens. Source: arXiv:2305.05465v6, §3, `e:Rres`, and §5, `d:goodmulti`. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

/-- Projection commutes with taking the finite token hull.
Source: arXiv:2305.05465v6, §10, the hull `π_F(Cx({z_i(t)}))`. -/
theorem image_tokenHull_eq_tokenHull (proj : ParamMatrix d) (Z : Idx n → EucSpace d) :
    proj '' tokenHull Z = tokenHull (fun i => proj (Z i)) := by
  change proj '' convexHull ℝ (Set.range Z) = convexHull ℝ (Set.range (proj ∘ Z))
  rw [Set.range_comp]
  exact proj.toLinearMap.image_convexHull _

/-- **The projected hull is nested along every actual solution.**
Source: arXiv:2305.05465v6, §10, opening paragraph of `t:multiplicity`.
The positive leading eigenvalue makes every maximizing projected token
have a nonpositive derivative under a separating linear functional. -/
theorem RescaledDynamics.tokenHull_projected_antitone (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z)
    {s t : ℝ} (hst : s ≤ t) :
    tokenHull (fun i => proj (Z t i)) ⊆ tokenHull (fun i => proj (Z s i)) := by
  rcases isEmpty_or_nonempty (Idx n) with hn | hn
  · simp [tokenHull, Set.range_eq_empty]
  refine convexHull_min ?_ (convex_convexHull ℝ _)
  rintro _ ⟨j, rfl⟩
  by_contra hj
  obtain ⟨f, u, hf, hu⟩ := geometric_hahn_banach_closed_point (convex_convexHull ℝ _)
    (isCompact_tokenHull (fun i => proj (Z s i))).isClosed hj
  have hM := antitone_sup'_of_hasDerivAt (fun i t => f (proj (Z t i))) _
    (fun i t => (f.hasFDerivAt (x := proj (Z t i))).comp_hasDerivAt t
      (hZ.hasDerivAt_projected Q K V proj F G lam hV hproj Z t i))
    (fun i t hi => ?_) hst
  · have h1 := Finset.le_sup' (fun i => f (proj (Z t i))) (Finset.mem_univ j)
    have h2 : (Finset.univ.sup' Finset.univ_nonempty fun i => f (proj (Z s i))) < u :=
      (Finset.sup'_lt_iff _).2 fun i _ => hf _ (mem_tokenHull _ i)
    exact (hu.trans_le (h1.trans hM)).not_gt h2
  · simp only [map_sum, map_smul, map_sub, smul_eq_mul]
    apply Finset.sum_nonpos
    intro j _
    exact mul_nonpos_of_nonneg_of_nonpos (attentionMatrix_nonneg _ _ _ i j)
      (mul_nonpos_of_nonneg_of_nonpos hV.2.2.2.1.le (sub_nonpos.mpr (hi j)))

/-- The full inputs, including the ordered times, hold for an actual
stationary solution. Source: arXiv:2305.05465v6, §10, the projected hull. -/
example (z : EucSpace 1) : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1 (fun _ (_ : Idx 2) => z) ∧ (0 : ℝ) ≤ 1 :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 z,
    zero_le_one⟩

/-- The paper's image notation inherits nesting from the actual projected
token dynamics. Source: arXiv:2305.05465v6, §10, opening of `t:multiplicity`. -/
theorem RescaledDynamics.image_tokenHull_antitone (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z)
    {s t : ℝ} (hst : s ≤ t) : proj '' tokenHull (Z t) ⊆ proj '' tokenHull (Z s) := by
  rw [image_tokenHull_eq_tokenHull, image_tokenHull_eq_tokenHull]
  exact hZ.tokenHull_projected_antitone Q K V proj F G lam hV hproj Z hst

/-- All image-hull inputs hold in the identity-value case with coincident
tokens. Source: arXiv:2305.05465v6, §10, `t:multiplicity`. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1 (fun _ (_ : Idx 2) => 0) ∧ (0 : ℝ) ≤ 1 :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _,
    zero_le_one⟩

/-- Every future projected token lies in the initial projected hull.
Source: arXiv:2305.05465v6, §10, nesting of `π_F(Cx({z_i(t)}))`.
This confines the leading component even when the other components grow. -/
theorem RescaledDynamics.mem_projected_tokenHull_zero (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z)
    (i : Idx n) {t : ℝ} (ht : 0 ≤ t) :
    proj (Z t i) ∈ tokenHull (fun j => proj (Z 0 j)) :=
  hZ.tokenHull_projected_antitone Q K V proj F G lam hV hproj Z ht (mem_tokenHull _ i)

/-- The ODE, splitting, projection and nonnegative time all hold for
stationary unit tokens. Source: arXiv:2305.05465v6, §10, the projected hull. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) ∧ (0 : ℝ) ≤ 0 :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _, le_rfl⟩

end Transformer.Clusters
