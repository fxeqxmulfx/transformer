module

public import Transformer.Interpolation.Section3_TargetCaps
public import Mathlib.Analysis.Convex.Caratheodory
public import Mathlib.LinearAlgebra.AffineSpace.FiniteDimensional

/-!
# Compact hulls and levels of linear functionals on geodesic hulls

ArXiv:2411.04551v3, §1.5, the geodesic convex hull `conv_g`, as used in
§2.2, proof of `prop: compression`. In finite dimension the convex hull
of a compact set is compact, by Carathéodory's theorem. A linear
functional that exceeds a positive level on a set of the sphere exceeds
it on the geodesic hull of the set: the convex hull lies in the
half-space and in the unit ball, and the radial projection scales by a
factor at least one.
-/

@[expose] public section

open Real Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- In finite dimension the convex hull of a compact set is compact: by
Carathéodory's theorem it is the image of finitely many products of a
simplex with powers of the set. Source auxiliary: arXiv:2411.04551v3,
§1.5, the geodesic hull of a compact support. -/
theorem isCompact_convexHull_of_isCompact {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {K : Set E} (hK : IsCompact K) : IsCompact (convexHull ℝ K) := by
  classical
  let W (m : ℕ) : Set (Fin m → ℝ) := {w | (∀ i, 0 ≤ w i) ∧ ∑ i, w i = 1}
  let D (m : ℕ) : Set ((Fin m → ℝ) × (Fin m → E)) := W m ×ˢ Set.univ.pi fun _ => K
  let Φ (m : ℕ) : (Fin m → ℝ) × (Fin m → E) → E := fun p => ∑ i, p.1 i • p.2 i
  have hΦ (m : ℕ) : Continuous (Φ m) := continuous_finsetSum _ fun i _ =>
    ((continuous_apply i).comp continuous_fst).smul ((continuous_apply i).comp continuous_snd)
  have hW (m : ℕ) : IsCompact (W m) := by
    have hWc : IsClosed (W m) := by
      have h : W m = (⋂ i, {w : Fin m → ℝ | 0 ≤ w i}) ∩ {w | ∑ i, w i = 1} := by
        ext w
        simp only [W, mem_ofPred_eq, mem_inter_iff, mem_iInter]
      rw [h]
      exact (isClosed_iInter fun i => isClosed_le continuous_const (continuous_apply i)).inter
        (isClosed_eq (continuous_finsetSum _ fun i _ => continuous_apply i) continuous_const)
    refine (isCompact_univ_pi fun _ : Fin m =>
      isCompact_Icc (a := (0 : ℝ)) (b := 1)).of_isClosed_subset hWc fun w hw i _ => ⟨hw.1 i, ?_⟩
    rw [← hw.2]
    exact Finset.single_le_sum (fun j _ => hw.1 j) (Finset.mem_univ i)
  have hD (m : ℕ) : IsCompact (D m) := (hW m).prod (isCompact_univ_pi fun _ => hK)
  have heq : convexHull ℝ K =
      ⋃ m ∈ Finset.range (Module.finrank ℝ E + 2), Φ m '' D m := by
    apply Subset.antisymm
    · intro x hx
      obtain ⟨ι, _, z, w, hzK, hind, hw, hw1, hx⟩ := eq_pos_convex_span_of_mem_convexHull hx
      have hcard : Fintype.card ι < Module.finrank ℝ E + 2 :=
        Nat.lt_succ_of_le ((AffineIndependent.card_le_finrank_succ hind).trans
          (Nat.succ_le_succ (Submodule.finrank_le _)))
      let e := (Fintype.equivFin ι).symm
      refine mem_biUnion (Finset.mem_range.mpr hcard)
        ⟨(fun j => w (e j), fun j => z (e j)), ⟨⟨fun j => (hw _).le, ?_⟩,
          fun j _ => hzK ⟨_, rfl⟩⟩, ?_⟩
      · rw [Equiv.sum_comp e w]
        exact hw1
      · change ∑ j, w (e j) • z (e j) = x
        rw [Equiv.sum_comp e (fun i => w i • z i)]
        exact hx
    · refine iUnion₂_subset fun m _ => ?_
      rintro _ ⟨⟨w, z⟩, ⟨⟨hw0, hw1⟩, hz⟩, rfl⟩
      exact (convex_convexHull ℝ K).sum_mem (fun i _ => hw0 i) hw1
        fun i _ => subset_convexHull ℝ K (hz i (mem_univ i))
  rw [heq]
  exact (Finset.range _).isCompact_biUnion fun m _ => (hD m).image (hΦ m)

/-- The geodesic hull of a set on which a linear functional exceeds a
positive level lies there too. Source auxiliary: arXiv:2411.04551v3,
§1.5, the definition of `conv_g`. -/
theorem convG_subset_level (f : StrongDual ℝ (EucSpace d)) (a : ℝ) (ha : 0 < a)
    (A : Set (SSphere d)) (hA : ∀ x ∈ A, a ≤ f x) : ∀ x ∈ convG d A, a ≤ f x := by
  have hlin : IsLinearMap ℝ f := ⟨f.map_add, f.map_smul⟩
  have hcap : convexHull ℝ ((↑) '' A : Set (EucSpace d)) ⊆ {v | a ≤ f v} :=
    convexHull_min (by rintro _ ⟨x, hx, rfl⟩; exact hA x hx) (convex_halfSpace_ge hlin a)
  have hnorm : convexHull ℝ ((↑) '' A : Set (EucSpace d)) ⊆ Metric.closedBall 0 1 :=
    convexHull_min (by
      rintro _ ⟨x, -, rfl⟩
      exact mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp x.2).le)
      (convex_closedBall 0 1)
  rintro x (hzero | ⟨v, hv, c, hc, hx⟩)
  · have h0 : a ≤ f 0 := hcap hzero
    rw [map_zero] at h0
    exact absurd h0 (not_le.mpr ha)
  · have hvn : ‖v‖ ≤ 1 := mem_closedBall_zero_iff.mp (hnorm hv)
    have hxn := mem_sphere_zero_iff_norm.mp x.2
    rw [hx, norm_smul, Real.norm_eq_abs, abs_of_pos hc] at hxn
    have hc1 : 1 ≤ c := by nlinarith [mul_le_mul_of_nonneg_left hvn hc.le]
    have hfv : a ≤ f v := hcap hv
    rw [hx, map_smul, smul_eq_mul]
    nlinarith

/-- The strict version of `convG_subset_level`. Source auxiliary:
arXiv:2411.04551v3, §1.5. -/
theorem convG_subset_level_lt (f : StrongDual ℝ (EucSpace d)) (a : ℝ) (ha : 0 < a)
    (A : Set (SSphere d)) (hA : ∀ x ∈ A, a < f x) : ∀ x ∈ convG d A, a < f x := by
  intro x hx
  have hlin : IsLinearMap ℝ f := ⟨f.map_add, f.map_smul⟩
  have hcap : convexHull ℝ ((↑) '' A : Set (EucSpace d)) ⊆ {v | a < f v} :=
    convexHull_min (by rintro _ ⟨x, hx, rfl⟩; exact hA x hx) (convex_halfSpace_gt hlin a)
  have hnorm : convexHull ℝ ((↑) '' A : Set (EucSpace d)) ⊆ Metric.closedBall 0 1 :=
    convexHull_min (by
      rintro _ ⟨x, -, rfl⟩
      exact mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp x.2).le)
      (convex_closedBall 0 1)
  rcases hx with hzero | ⟨v, hv, c, hc, hx⟩
  · have h0 : a < f 0 := hcap hzero
    rw [map_zero] at h0
    exact absurd h0 (not_lt.mpr ha.le)
  · have hvn : ‖v‖ ≤ 1 := mem_closedBall_zero_iff.mp (hnorm hv)
    have hxn := mem_sphere_zero_iff_norm.mp x.2
    rw [hx, norm_smul, Real.norm_eq_abs, abs_of_pos hc] at hxn
    have hc1 : 1 ≤ c := by nlinarith [mul_le_mul_of_nonneg_left hvn hc.le]
    have hfv : a < f v := hcap hv
    rw [hx, map_smul, smul_eq_mul]
    nlinarith

/-- The hypotheses of `isCompact_convexHull_of_isCompact`,
`convG_subset_level` and `convG_subset_level_lt` are satisfiable: a
point of the circle, the inner product with it, and the levels `1` and
`1 / 2`. -/
example (p : SSphere 2) : IsCompact ({(p : EucSpace 2)} : Set (EucSpace 2)) ∧ (0 : ℝ) < 1 ∧
    (∀ x ∈ ({p} : Set (SSphere 2)), (1 : ℝ) ≤ innerSL ℝ (p : EucSpace 2) x) ∧
    ∀ x ∈ ({p} : Set (SSphere 2)), (1 / 2 : ℝ) < innerSL ℝ (p : EucSpace 2) x := by
  have hp : innerSL ℝ (p : EucSpace 2) p = 1 := by
    rw [innerSL_apply_apply, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2, one_pow]
  refine ⟨isCompact_singleton, one_pos, fun x hx => ?_, fun x hx => ?_⟩
  · rw [mem_singleton_iff.mp hx, hp]
  · rw [mem_singleton_iff.mp hx, hp]
    norm_num

end Transformer.Interpolation
