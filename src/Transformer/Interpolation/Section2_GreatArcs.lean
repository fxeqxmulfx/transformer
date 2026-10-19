module

public import Transformer.Interpolation.Basic
public import Mathlib.Topology.UniformSpace.HeineCantor

/-!
# Great circle arcs between points off each other's line

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Step 3: "we can
find a sequence of open balls" from a packing ball to the anchor ball,
consecutive ones meeting. Here the chain runs along the great circle arc
from a start point to an end point off its line, the normalized chord
point `(1 - t) a + t b`, which never vanishes. The arc is continuous,
lies in the geodesic hull and in the plane of its end points, and equally
spaced parameters give chains of points with small steps.
-/

@[expose] public section

open Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The chord from a unit vector `a` to a point `b` off its line misses
the origin. Source auxiliary: arXiv:2411.04551v3, §2.2, Step 3. -/
theorem chord_ne_zero (a b : SSphere d)
    (hab : ∀ α : ℝ, (b : EucSpace d) ≠ α • (a : EucSpace d)) (t : ℝ) :
    (1 - t) • (a : EucSpace d) + t • (b : EucSpace d) ≠ 0 := by
  intro h
  by_cases ht : t = 0
  · rw [ht, sub_zero, one_smul, zero_smul, add_zero] at h
    simpa [h] using mem_sphere_zero_iff_norm.mp a.2
  · apply hab (-(1 - t) / t)
    have htb : t • (b : EucSpace d) = -((1 - t) • (a : EucSpace d)) :=
      eq_neg_of_add_eq_zero_right h
    calc (b : EucSpace d) = t⁻¹ • (t • (b : EucSpace d)) := by
          rw [smul_smul, inv_mul_cancel₀ ht, one_smul]
      _ = (-(1 - t) / t) • (a : EucSpace d) := by
          rw [htb, smul_neg, smul_smul, ← neg_smul]
          congr 1
          ring

/-- The point of the great circle arc from `a` to `b` at parameter `t`:
the normalized chord point `(1 - t) a + t b`. -/
noncomputable def arcPoint (a b : EucSpace d) (t : ℝ) : EucSpace d :=
  ‖(1 - t) • a + t • b‖⁻¹ • ((1 - t) • a + t • b)

/-- The arc point is a unit vector. -/
theorem norm_arcPoint (a b : SSphere d)
    (hab : ∀ α : ℝ, (b : EucSpace d) ≠ α • (a : EucSpace d)) (t : ℝ) :
    ‖arcPoint (a : EucSpace d) b t‖ = 1 := by
  rw [arcPoint, norm_smul, norm_inv, norm_norm,
    inv_mul_cancel₀ (norm_ne_zero_iff.mpr (chord_ne_zero a b hab t))]

/-- The great circle arc from `a` to `b` as a curve on the sphere.
Source auxiliary: arXiv:2411.04551v3, §2.2, Step 3. -/
noncomputable def arcSphere (a b : SSphere d)
    (hab : ∀ α : ℝ, (b : EucSpace d) ≠ α • (a : EucSpace d)) (t : ℝ) : SSphere d :=
  ⟨arcPoint (a : EucSpace d) b t, mem_sphere_zero_iff_norm.mpr (norm_arcPoint a b hab t)⟩

variable (a b : SSphere d) (hab : ∀ α : ℝ, (b : EucSpace d) ≠ α • (a : EucSpace d))

/-- The arc starts at `a`. -/
theorem arcSphere_zero : arcSphere a b hab 0 = a := by
  apply Subtype.ext
  change arcPoint _ _ 0 = _
  rw [arcPoint, sub_zero, one_smul, zero_smul, add_zero, mem_sphere_zero_iff_norm.mp a.2,
    inv_one, one_smul]

/-- The arc ends at `b`. -/
theorem arcSphere_one : arcSphere a b hab 1 = b := by
  apply Subtype.ext
  change arcPoint _ _ 1 = _
  rw [arcPoint, sub_self, zero_smul, zero_add, one_smul, mem_sphere_zero_iff_norm.mp b.2,
    inv_one, one_smul]

/-- The arc is continuous. -/
theorem continuous_arcSphere : Continuous (arcSphere a b hab) := by
  have hc : Continuous fun t : ℝ => (1 - t) • (a : EucSpace d) + t • (b : EucSpace d) := by
    fun_prop
  exact ((hc.norm.fun_inv₀ fun t => norm_ne_zero_iff.mpr (chord_ne_zero a b hab t)).smul
    hc).subtype_mk _

/-- For parameters in `[0, 1]` the arc lies in the geodesic hull of its
end points. Source auxiliary: arXiv:2411.04551v3, §2.2, Step 3, the
chains inside the piece. -/
theorem arcSphere_mem_convG (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
    arcSphere a b hab t ∈ convG d {a, b} := by
  refine Or.inr ⟨(1 - t) • (a : EucSpace d) + t • (b : EucSpace d), ?_,
    ‖(1 - t) • (a : EucSpace d) + t • (b : EucSpace d)‖⁻¹,
    inv_pos.mpr (norm_pos_iff.mpr (chord_ne_zero a b hab t)), rfl⟩
  refine segment_subset_convexHull (mem_image_of_mem _ (mem_insert a {b}))
    (mem_image_of_mem _ (mem_insert_of_mem a rfl)) ⟨1 - t, t, ?_, ht.1, by ring, rfl⟩
  linarith [ht.2]

/-- The arc misses every point off the plane of its end points. Source
auxiliary: arXiv:2411.04551v3, §2.2, Step 3. -/
theorem arcSphere_ne_of_avoid (m : SSphere d)
    (hm : ∀ α β : ℝ, (m : EucSpace d) ≠ α • (a : EucSpace d) + β • (b : EucSpace d)) (t : ℝ) :
    arcSphere a b hab t ≠ m := by
  intro h
  apply hm (‖(1 - t) • (a : EucSpace d) + t • (b : EucSpace d)‖⁻¹ * (1 - t))
    (‖(1 - t) • (a : EucSpace d) + t • (b : EucSpace d)‖⁻¹ * t)
  rw [← h]
  change arcPoint (a : EucSpace d) b t = _
  rw [arcPoint, mul_smul, mul_smul, ← smul_add]

/-- **Arc chains.** Equally spaced points of the arc have steps below any
positive bound once the spacing is fine. Source auxiliary:
arXiv:2411.04551v3, §2.2, Step 3, `𝓑_{ℓ,n} ∩ 𝓑_{ℓ+1,n} ≠ ∅`. -/
theorem exists_arc_chain (R : ℝ) (hR : 0 < R) :
    ∃ n : ℕ, 0 < n ∧ ∀ l < n, dist (arcSphere a b hab (((l + 1 : ℕ) : ℝ) / n))
      (arcSphere a b hab ((l : ℝ) / n)) < R := by
  have hu := (isCompact_Icc (a := (0 : ℝ)) (b := 1)).uniformContinuousOn_of_continuous
    (continuous_arcSphere a b hab).continuousOn
  obtain ⟨δ, hδ, hδR⟩ := Metric.uniformContinuousOn_iff.mp hu R hR
  obtain ⟨m, hm⟩ := exists_nat_one_div_lt hδ
  have hm0 : (0 : ℝ) < m + 1 := by positivity
  refine ⟨m + 1, m.succ_pos, fun l hl => hδR _ ⟨by positivity, ?_⟩ _ ⟨by positivity, ?_⟩ ?_⟩
  · push_cast
    rw [div_le_one hm0]
    exact_mod_cast hl
  · push_cast
    rw [div_le_one hm0]
    exact_mod_cast hl.le
  · push_cast
    rw [Real.dist_eq, ← sub_div, add_sub_cancel_left, abs_of_pos (by positivity)]
    exact hm

/-- The hypotheses of the arc lemmas are satisfiable: two orthogonal unit
vectors of the plane. -/
example : ∃ a b : SSphere 2, ∀ α : ℝ, (b : EucSpace 2) ≠ α • (a : EucSpace 2) := by
  refine ⟨⟨EuclideanSpace.single 0 1, by simp⟩, ⟨EuclideanSpace.single 1 1, by simp⟩,
    fun α h => ?_⟩
  have h1 := congrArg (fun v : EucSpace 2 => v 1) h
  simp at h1

end Transformer.Interpolation
