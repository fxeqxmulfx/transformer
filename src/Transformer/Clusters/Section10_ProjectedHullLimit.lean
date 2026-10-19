module

public import Transformer.Clusters.Section10_ProjectedDynamics
public import Transformer.Clusters.Section8_HullLimit

/-!
# The projected hull converges to a genuine polytope

ArXiv:2305.05465v6, §10, the first paragraph of `t:multiplicity`.
Nesting of the projected hull follows from the actual rescaled dynamics.
Compactness of the initial projected token tuple and finite-hull continuity
then identify its limiting hull as the intersection of all future hulls.
Its generators are lifted into the leading subspace `F`, so the resulting
polytope has the same relative ambient space as the manuscript.

For positive token count the generators are nonempty and number at most
`n`. Set equality identifies the actual limit independently of the
real-valued Hausdorff convention. Candidate finiteness and the convergence
of individual projected tokens are separate later steps.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- A polytope generated inside `F` is the ambient hull of the images of
its generators. Source: arXiv:2305.05465v6, §10, the polytope `𝒦 ⊂ F`. -/
theorem polytopeIn_eq_convexHull_image (F : Submodule ℝ (EucSpace d)) (S : Finset F) :
    polytopeIn F S = convexHull ℝ (((↑) : F → EucSpace d) '' (S : Set F)) :=
  F.subtype.image_convexHull _

/-- Seeing the polytope in the ambient space retains its membership in
`F`. Source: arXiv:2305.05465v6, §10, `𝒦 ⊂ F`. -/
theorem polytopeIn_subset_submodule (F : Submodule ℝ (EucSpace d)) (S : Finset F) :
    polytopeIn F S ⊆ (F : Set (EucSpace d)) := by
  rintro _ ⟨x, -, rfl⟩
  exact x.property

/-- A finite set contained in `F` can be expressed by generators of type
`F`, without increasing its cardinality. Source: arXiv:2305.05465v6,
§10, expressing the limiting polytope in the leading subspace. -/
theorem exists_finset_submodule_image (F : Submodule ℝ (EucSpace d))
    (S : Finset (EucSpace d)) (hS : (S : Set (EucSpace d)) ⊆ F) :
    ∃ T : Finset F, T.card ≤ S.card ∧
      ((↑) : F → EucSpace d) '' (T : Set F) = (S : Set (EucSpace d)) := by
  classical
  let lift : S → F := fun x => ⟨x, hS x.property⟩
  refine ⟨S.attach.image lift, ?_, ?_⟩
  · exact (Finset.card_image_le).trans_eq Finset.card_attach
  · ext x
    constructor
    · rintro ⟨y, hy, rfl⟩
      obtain ⟨z, -, rfl⟩ := Finset.mem_image.mp hy
      exact z.property
    · intro hx
      exact ⟨lift ⟨x, hx⟩, Finset.mem_image.mpr
        ⟨⟨x, hx⟩, Finset.mem_attach _ _, rfl⟩, rfl⟩

/-- The containment input is satisfied by an actual singleton in the full
leading subspace. Source: arXiv:2305.05465v6, §10, `𝒦 ⊂ F`. -/
example (z : EucSpace 1) : (({z} : Finset (EucSpace 1)) : Set (EucSpace 1)) ⊆
    (⊤ : Submodule ℝ (EucSpace 1)) := fun _ _ => Submodule.mem_top

/-- The intersection of all future projected hulls is compact, even
before its polytope description is known. Source: arXiv:2305.05465v6,
§10, the limiting projected hull. -/
theorem isCompact_future_image_tokenHull (proj : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) :
    IsCompact {x | ∀ t : ℝ, 0 ≤ t → x ∈ proj '' tokenHull (Z t)} := by
  simpa only [image_tokenHull_eq_tokenHull] using
    isCompact_future_tokenHull (fun t i => proj (Z t i))

/-- The future projected-hull intersection remains in the leading
subspace. Source: arXiv:2305.05465v6, §10, `𝒦 ⊂ F`. -/
theorem future_image_tokenHull_subset_submodule (proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) :
    {x | ∀ t : ℝ, 0 ≤ t → x ∈ proj '' tokenHull (Z t)} ⊆ F :=
  fun _ hx => image_tokenHull_subset proj hproj (Z 0) (hx 0 le_rfl)

/-- The full projection hypothesis is realized by the identity map.
Source: arXiv:2305.05465v6, §10, `π_F` parallel to `G`. -/
example : IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ := isProjOnto_one 1

/-- **The actual projected hull has a nonempty polytope limit in `F`.**
Source: arXiv:2305.05465v6, §10, opening paragraph of `t:multiplicity`.
Its at most `n` generators describe the future-hull intersection exactly.
The source's positive-definiteness assumption is unnecessary for this
geometric step; positive token count records its implicit nonemptiness. -/
theorem RescaledDynamics.exists_projected_hull_polytope_limit
    (Q K V proj : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (hn : 0 < n) :
    ∃ S : Finset F, S.Nonempty ∧ S.card ≤ n ∧
      polytopeIn F S = {x | ∀ t : ℝ, 0 ≤ t → x ∈ proj '' tokenHull (Z t)} ∧
      Tendsto (fun t => Metric.hausdorffDist (proj '' tokenHull (Z t)) (polytopeIn F S))
        atTop (nhds 0) := by
  classical
  obtain ⟨S, hSne, hcard, hHull, hHaus⟩ :=
    exists_polytope_limit_of_tokenHull_antitone (fun t i => proj (Z t i)) hn
      (fun s t _ hst => hZ.tokenHull_projected_antitone Q K V proj F G lam hV hproj Z hst)
  have hSF : (S : Set (EucSpace d)) ⊆ F := by
    intro x hx
    have hfuture := hHull ▸ subset_convexHull ℝ (S : Set (EucSpace d)) hx
    have hx0 := hfuture 0 le_rfl
    rw [← image_tokenHull_eq_tokenHull] at hx0
    exact image_tokenHull_subset proj hproj (Z 0) hx0
  obtain ⟨T, hTcard, hTimage⟩ := exists_finset_submodule_image F S hSF
  have hpoly : polytopeIn F T = convexHull ℝ (S : Set (EucSpace d)) := by
    rw [polytopeIn_eq_convexHull_image, hTimage]
  have hTne : T.Nonempty := by
    obtain ⟨x, hx⟩ := hSne
    have himage : x ∈ ((↑) : F → EucSpace d) '' (T : Set F) := by
      rw [hTimage]
      exact hx
    obtain ⟨y, hy, -⟩ := himage
    exact ⟨y, hy⟩
  refine ⟨T, hTne, hTcard.trans hcard, ?_, ?_⟩
  · exact hpoly.trans (by simpa only [image_tokenHull_eq_tokenHull] using hHull)
  · simpa only [image_tokenHull_eq_tokenHull, hpoly] using hHaus

/-- All inputs of the nonempty projected-polytope result are realized by
two stationary unit tokens. Source: arXiv:2305.05465v6, §5, `d:goodmulti`,
and §10, the limiting hull. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) ∧ (0 : ℕ) < 2 :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _, by norm_num⟩

/-- A single positive radius bounds every projected token for all future
times. Source: arXiv:2305.05465v6, §10, compactness of the nested projected
hull. No bound on the complementary rescaled coordinates is assumed. -/
theorem RescaledDynamics.exists_bound_projected (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ R : ℝ, 0 < R ∧ ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖proj (Z t i)‖ ≤ R := by
  obtain ⟨R, hR⟩ := (isCompact_tokenHull (fun i => proj (Z 0 i))).isBounded.subset_closedBall 0
  refine ⟨|R| + 1, by positivity, ?_⟩
  intro i t ht
  have hbound : ‖proj (Z t i)‖ ≤ R := by
    simpa using hR (hZ.mem_projected_tokenHull_zero Q K V proj F G lam hV hproj Z i ht)
  exact hbound.trans ((le_abs_self R).trans (by linarith))

/-- The ODE, splitting and projection inputs hold for a stationary
nonzero configuration. Source: arXiv:2305.05465v6, §10, the projected hull. -/
example (z : EucSpace 1) : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1 (fun _ (_ : Idx 2) => z) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 z⟩

end Transformer.Clusters
