/-
# The convex-hull form of Wendel's hemisphere event

For a nonempty finite sample on the sphere, lying in a common open hemisphere
is equivalent to the origin lying outside the convex hull of the sample.
The reverse implication uses strict separation of a point and a compact
convex hull.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.UniformHemisphere
import Transformer.Perspective.WendelOneDimBasic
import Mathlib.Analysis.Convex.Topology
import Mathlib.Analysis.Convex.Combination
import Mathlib.Analysis.LocallyConvex.Separation
import Mathlib.Analysis.InnerProductSpace.Dual

open MeasureTheory

namespace Transformer.Perspective

/-- A nonempty finite spherical sample lies in a common open hemisphere if
and only if the origin is outside its convex hull.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem exists_openHemisphere_iff_zero_notMem_convexHull (d n : ℕ) (hn : 1 ≤ n)
    (X : SphereTuple d n) :
    (∃ w : SSphere d, ∀ i : Idx n,
      0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))) ↔
      (0 : EucSpace d) ∉ convexHull ℝ (Set.range fun i : Idx n => (X i : EucSpace d)) := by
  constructor
  · rintro ⟨w, hw⟩ hzero
    have hconv : Convex ℝ {v : EucSpace d |
        0 < inner (𝕜 := ℝ) v ((w : EucSpace d))} := by
      intro x hx y hy a b ha hb hab
      change 0 < inner (𝕜 := ℝ) (a • x + b • y) ((w : EucSpace d))
      rw [inner_add_left, real_inner_smul_left, real_inner_smul_left]
      rcases lt_or_eq_of_le ha with ha' | ha'
      · exact add_pos_of_pos_of_nonneg (mul_pos ha' hx) (mul_nonneg hb hy.le)
      · subst a
        have hb' : 0 < b := by linarith
        simpa using mul_pos hb' hy
    have hsubset : convexHull ℝ (Set.range fun i : Idx n => (X i : EucSpace d))
        ⊆ {v : EucSpace d | 0 < inner (𝕜 := ℝ) v ((w : EucSpace d))} :=
      convexHull_min (fun v hv => by obtain ⟨i, rfl⟩ := hv; exact hw i) hconv
    have hpos := hsubset hzero
    simp at hpos
  · intro hzero
    have hfinite : (Set.range fun i : Idx n => (X i : EucSpace d)).Finite :=
      Set.finite_range _
    obtain ⟨f, u, hfu, hsep⟩ :=
      geometric_hahn_banach_point_closed
        (convex_convexHull ℝ _) (hfinite.isClosed_convexHull ℝ) hzero
    let w₀ : EucSpace d := (InnerProductSpace.toDual ℝ (EucSpace d)).symm f
    have hfw (v : EucSpace d) : f v = inner (𝕜 := ℝ) v w₀ := by
      change f v = inner (𝕜 := ℝ) v ((InnerProductSpace.toDual ℝ (EucSpace d)).symm f)
      rw [real_inner_comm]
      exact (InnerProductSpace.toDual_symm_apply (x := v) (y := f)).symm
    have hu : 0 < u := by simpa using hfu
    have hpos (i : Idx n) : 0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) w₀ := by
      have hi : (X i : EucSpace d) ∈ convexHull ℝ
          (Set.range fun j : Idx n => (X j : EucSpace d)) :=
        subset_convexHull ℝ _ ⟨i, rfl⟩
      rw [← hfw]
      exact hu.trans (hsep _ hi)
    have hwne : w₀ ≠ 0 := by
      intro h
      have := hpos ⟨0, hn⟩
      simp [h] at this
    have hnorm : 0 < ‖w₀‖ := norm_pos_iff.mpr hwne
    refine ⟨⟨‖w₀‖⁻¹ • w₀, ?_⟩, fun i => ?_⟩
    · rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, Real.norm_eq_abs,
        abs_of_pos hnorm, inv_mul_cancel₀ (ne_of_gt hnorm)]
    · show (0 : ℝ) < inner (𝕜 := ℝ) ((X i : EucSpace d)) (‖w₀‖⁻¹ • w₀)
      rw [real_inner_smul_right]
      exact mul_pos (inv_pos.mpr hnorm) (hpos i)

/-- The origin belongs to the hull precisely when the sample admits a
nonnegative affine dependence. This is the finite coefficient form needed
to count the sign patterns that fail Wendel's hemisphere event.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem zero_mem_convexHull_iff_nonneg_relation (d n : ℕ) (X : SphereTuple d n) :
    (0 : EucSpace d) ∈ convexHull ℝ
      (Set.range fun i : Idx n => (X i : EucSpace d)) ↔
      ∃ c : Idx n → ℝ, (∀ i, 0 ≤ c i) ∧ ∑ i, c i = 1 ∧
        ∑ i, c i • (X i : EucSpace d) = 0 := by
  classical
  constructor
  · intro h
    rw [convexHull_range_eq_exists_affineCombination] at h
    obtain ⟨s, w, hw, hsum, hcomb⟩ := h
    let c : Idx n → ℝ := fun i => if i ∈ s then w i else 0
    refine ⟨c, ?_, ?_, ?_⟩
    · intro i
      by_cases hi : i ∈ s
      · simp [c, hi, hw i hi]
      · simp [c, hi]
    · simpa only [c, Fintype.sum_ite_mem] using hsum
    · rw [Finset.affineCombination_eq_linear_combination s _ _ hsum] at hcomb
      simpa only [c, ite_smul, zero_smul, Fintype.sum_ite_mem] using hcomb
  · rintro ⟨c, hc, hsum, hcomb⟩
    exact mem_convexHull_of_exists_fintype c (fun i : Idx n => (X i : EucSpace d))
      hc hsum (fun i => Set.mem_range_self i) hcomb

/-- The hypotheses are satisfiable at `d = n = 1`: the positive point lies
in its own open hemisphere and its singleton hull excludes the origin.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
example :
    (∃ w : SSphere 1, ∀ i : Idx 1,
      0 < inner (𝕜 := ℝ) (((fun _ => eOne) i : SSphere 1) : EucSpace 1)
        ((w : EucSpace 1))) ∧
    (0 : EucSpace 1) ∉ convexHull ℝ
      (Set.range fun i : Idx 1 => (((fun _ => eOne) i : SSphere 1) : EucSpace 1)) := by
  have h : ∃ w : SSphere 1, ∀ i : Idx 1,
      0 < inner (𝕜 := ℝ) (((fun _ => eOne) i : SSphere 1) : EucSpace 1)
        ((w : EucSpace 1)) := by
    refine ⟨eOne, fun i => ?_⟩
    rw [inner_eOne_eOne]
    norm_num
  exact ⟨h, (exists_openHemisphere_iff_zero_notMem_convexHull 1 1 le_rfl _).mp h⟩

end Transformer.Perspective
