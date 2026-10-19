module

public import Transformer.Interpolation.Section2_HullLevels
public import Mathlib.Analysis.Convex.Join
public import Mathlib.Analysis.LocallyConvex.Separation

/-!
# Convex neighbourhoods of supports with separated hulls

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`: "when we act on
the `i`-th measure, all the other measures remain invariant", and the
geodesic convex hulls of the final supports stay apart. Nonempty compact
sets of the sphere with disjoint geodesic hulls are separated through
the origin, with a margin, by one linear functional. The finite
intersections of the open half-spaces these functionals give are open
neighbourhoods of the sets: pairwise disjoint, closed under the geodesic
hull, inside an open half-space unless they are the whole sphere, and
with pairwise disjoint hulls of their closures.
-/

@[expose] public section

open Real Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- **Separation of hulls.** Nonempty compact sets of the sphere with
disjoint geodesic hulls are separated through the origin with a margin.
Source auxiliary: arXiv:2411.04551v3, §2.2, the disjoint hulls of
`prop: compression`. -/
theorem exists_separating_functional (A B : Set (SSphere d)) (hA : IsCompact A)
    (hB : IsCompact B) (hAne : A.Nonempty) (hBne : B.Nonempty)
    (hdisj : Disjoint (convG d A) (convG d B)) :
    ∃ (f : StrongDual ℝ (EucSpace d)) (c : ℝ), 0 < c ∧ (∀ x ∈ A, c ≤ f x) ∧
      ∀ x ∈ B, f x ≤ -c := by
  set SA : Set (EucSpace d) := (↑) '' A
  set SB : Set (EucSpace d) := (↑) '' B
  obtain ⟨a₀, ha₀⟩ := hAne
  obtain ⟨b₀, hb₀⟩ := hBne
  have hA0 : (0 : EucSpace d) ∉ convexHull ℝ SA := fun h =>
    disjoint_left.mp hdisj (Or.inl h) (subset_convG d B hb₀)
  have hB0 : (0 : EucSpace d) ∉ convexHull ℝ SB := fun h =>
    disjoint_right.mp hdisj (Or.inl h) (subset_convG d A ha₀)
  have hS : (0 : EucSpace d) ∉ convexHull ℝ (SA ∪ -SB) := by
    rw [convexHull_union ⟨_, mem_image_of_mem _ ha₀⟩ ⟨-(b₀ : EucSpace d), show _ ∈ -SB by
      rw [mem_neg, neg_neg]
      exact mem_image_of_mem _ hb₀⟩, mem_convexJoin]
    rintro ⟨a, ha, b', hb', s, t, hs, ht, hst, h0⟩
    rw [convexHull_neg, mem_neg] at hb'
    have hab : s • a = t • -b' := by
      rw [smul_neg, eq_neg_iff_add_eq_zero]
      exact h0
    have ha_ne : a ≠ 0 := fun h => hA0 (h ▸ ha)
    have hs0 : 0 < s := by
      rcases hs.lt_or_eq with h | h
      · exact h
      · rw [← h, zero_smul] at hab
        rw [← h, zero_add] at hst
        rw [hst, one_smul] at hab
        exact absurd (hab ▸ hb') hB0
    have hna : 0 < ‖a‖ := norm_pos_iff.mpr ha_ne
    let x : SSphere d := ⟨‖a‖⁻¹ • a, mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm ha_ne)⟩
    have hxA : x ∈ convG d A := Or.inr ⟨a, ha, ‖a‖⁻¹, inv_pos.mpr hna, rfl⟩
    have ht0 : 0 < t := by
      rcases ht.lt_or_eq with h | h
      · exact h
      · rw [← h, zero_smul, smul_eq_zero] at hab
        exact absurd (hab.resolve_left hs0.ne') ha_ne
    have hxB : x ∈ convG d B := by
      refine Or.inr ⟨-b', hb', ‖a‖⁻¹ * (t / s), by positivity, ?_⟩
      change ‖a‖⁻¹ • a = (‖a‖⁻¹ * (t / s)) • -b'
      rw [show ‖a‖⁻¹ * (t / s) = ‖a‖⁻¹ * s⁻¹ * t by ring, mul_smul, ← hab, smul_smul,
        mul_assoc, inv_mul_cancel₀ hs0.ne', mul_one]
    exact disjoint_left.mp hdisj hxA hxB
  have hcpt : IsCompact (convexHull ℝ (SA ∪ -SB)) :=
    isCompact_convexHull_of_isCompact ((hA.image continuous_subtype_val).union
      (hB.image continuous_subtype_val).neg)
  obtain ⟨f, u, hfu, hu0⟩ :=
    geometric_hahn_banach_closed_point (convex_convexHull ℝ _) hcpt.isClosed hS
  rw [map_zero] at hu0
  refine ⟨-f, -u, neg_pos.mpr hu0, fun x hx => ?_, fun x hx => ?_⟩
  · have h := hfu _ (subset_convexHull ℝ _ (Or.inl (mem_image_of_mem _ hx)))
    rw [neg_apply]
    linarith
  · have h := hfu (-(x : EucSpace d)) (subset_convexHull ℝ _ (Or.inr (show -(x : EucSpace d) ∈ -SB by
      rw [mem_neg, neg_neg]
      exact mem_image_of_mem _ hx)))
    rw [map_neg] at h
    rw [neg_apply]
    linarith

/-- **Separated convex regions.** Nonempty compact sets of the sphere
with pairwise disjoint geodesic hulls have open neighbourhoods that are
pairwise disjoint, closed under the geodesic hull, inside an open
half-space unless they are the whole sphere, and whose closures have
pairwise disjoint geodesic hulls. Source auxiliary: arXiv:2411.04551v3,
§2.2, proof of `prop: compression`, the regions where the phases acting
on one measure live. -/
theorem exists_separated_regions {N : ℕ} (K : Fin N → Set (SSphere d))
    (hK : ∀ i, IsCompact (K i)) (hKne : ∀ i, (K i).Nonempty)
    (hdisj : ∀ i j, i ≠ j → Disjoint (convG d (K i)) (convG d (K j))) :
    ∃ U : Fin N → Set (SSphere d), (∀ i, IsOpen (U i)) ∧ (∀ i, K i ⊆ U i) ∧
      (∀ i j, i ≠ j → Disjoint (U i) (U j)) ∧ (∀ i, ∀ A ⊆ U i, convG d A ⊆ U i) ∧
      (∀ i, U i = univ ∨ ∃ f : StrongDual ℝ (EucSpace d), ∀ x ∈ U i, 0 < f x) ∧
      ∀ i j, i ≠ j → Disjoint (convG d (closure (U i))) (convG d (closure (U j))) := by
  have hsep (i j : Fin N) (hij : i ≠ j) := exists_separating_functional (K i) (K j) (hK i)
    (hK j) (hKne i) (hKne j) (hdisj i j hij)
  choose f c hc hfi hfj using hsep
  have hcont (g : StrongDual ℝ (EucSpace d)) : Continuous fun x : SSphere d => g x :=
    g.continuous.comp continuous_subtype_val
  let U (i : Fin N) : Set (SSphere d) := ⋂ j, ⋂ h : i ≠ j,
    {x | c i j h / 2 < f i j h x} ∩ {x | f j i (Ne.symm h) x < -(c j i (Ne.symm h) / 2)}
  have hmem {i : Fin N} {x : SSphere d} : x ∈ U i ↔ ∀ j (h : i ≠ j),
      c i j h / 2 < f i j h x ∧ f j i (Ne.symm h) x < -(c j i (Ne.symm h) / 2) := by
    simp only [U, mem_iInter, mem_inter_iff, mem_ofPred_eq]
  refine ⟨U, fun i => ?_, fun i x hx => ?_, fun i j hij => ?_, fun i A hA => ?_, fun i => ?_,
    fun i j hij => ?_⟩
  · exact isOpen_iInter_of_finite fun j => isOpen_iInter_of_finite fun h =>
      (isOpen_lt continuous_const (hcont _)).inter (isOpen_lt (hcont _) continuous_const)
  · refine hmem.mpr fun j h => ⟨?_, ?_⟩
    · linarith [hfi i j h x hx, hc i j h]
    · linarith [hfj j i (Ne.symm h) x hx, hc j i (Ne.symm h)]
  · refine disjoint_left.mpr fun x hxi hxj => ?_
    have h1 := (hmem.mp hxi j hij).1
    have h2 := (hmem.mp hxj i (Ne.symm hij)).2
    linarith [hc i j hij]
  · intro x hx
    refine hmem.mpr fun j h => ⟨?_, ?_⟩
    · exact convG_subset_level_lt (f i j h) _ (half_pos (hc i j h)) A
        (fun y hy => (hmem.mp (hA hy) j h).1) x hx
    · have hneg := convG_subset_level_lt (-f j i (Ne.symm h)) _ (half_pos (hc j i (Ne.symm h))) A
        (fun y hy => by rw [neg_apply]; linarith [(hmem.mp (hA hy) j h).2]) x hx
      rw [neg_apply] at hneg
      linarith
  · by_cases hN : ∃ j, i ≠ j
    · obtain ⟨j, h⟩ := hN
      exact Or.inr ⟨f i j h, fun x hx => (half_pos (hc i j h)).trans (hmem.mp hx j h).1⟩
    · exact Or.inl (eq_univ_of_forall fun x => hmem.mpr fun j h => (hN ⟨j, h⟩).elim)
  · have hi : closure (U i) ⊆ {x | c i j hij / 2 ≤ f i j hij x} :=
      closure_minimal (fun x hx => (hmem.mp hx j hij).1.le) (isClosed_le continuous_const (hcont _))
    have hj : closure (U j) ⊆ {x | c i j hij / 2 ≤ (-f i j hij) x} := by
      refine closure_minimal (fun x hx => ?_) (isClosed_le continuous_const (hcont _))
      have h2 := (hmem.mp hx i (Ne.symm hij)).2
      change c i j hij / 2 ≤ (-f i j hij) x
      rw [neg_apply]
      linarith
    refine disjoint_left.mpr fun x hxi hxj => ?_
    have h1 := convG_subset_level (f i j hij) _ (half_pos (hc i j hij)) _ (fun y hy => hi hy) x hxi
    have h2 := convG_subset_level (-f i j hij) _ (half_pos (hc i j hij)) _ (fun y hy => hj hy) x hxj
    rw [neg_apply] at h2
    linarith [hc i j hij]

/-- The hypotheses of `exists_separating_functional` and
`exists_separated_regions` are satisfiable: two antipodal points of the
circle. -/
example (p : SSphere 2) : IsCompact ({p} : Set (SSphere 2)) ∧ ({p} : Set (SSphere 2)).Nonempty ∧
    Disjoint (convG 2 {p}) (convG 2 {(⟨-(p : EucSpace 2), by simp⟩ : SSphere 2)}) := by
  refine ⟨isCompact_singleton, singleton_nonempty p, ?_⟩
  rw [convG_singleton, convG_singleton, disjoint_singleton]
  intro h
  have hv : (p : EucSpace 2) = -(p : EucSpace 2) := congrArg Subtype.val h
  have h2 : (2 : ℝ) • (p : EucSpace 2) = 0 := by
    rw [two_smul]
    nth_rewrite 2 [hv]
    exact add_neg_cancel _
  have h0 : (p : EucSpace 2) = 0 := (smul_eq_zero.mp h2).resolve_left two_ne_zero
  have h1 := mem_sphere_zero_iff_norm.mp p.2
  rw [h0, norm_zero] at h1
  exact zero_ne_one h1

end Transformer.Interpolation
