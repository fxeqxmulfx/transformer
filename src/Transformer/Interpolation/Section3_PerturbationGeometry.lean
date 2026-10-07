/-
# A continuous sphere map cannot move points only inside a plane

Geometry behind the counterexample to Part 1 of `lem: perturbation` in
arXiv:2411.04551v3, §3, `eq: identity.flow`. The source requires the terminal
flow to fix every point outside the union of two geodesic convex hulls.
If both supports lie in a proper plane through the origin, in the positive
quadrant, their hulls lie in that plane too. Its complement in the sphere
is dense, so continuity forces the whole terminal flow to be the identity.

The proof uses explicit normalized perturbations in the first coordinate;
it requires neither a flow existence theorem nor an unproved PDE input.
`Section3_PerturbationFalse` supplies two different probability measures
meeting all the source's equal-barycenter hypotheses in this plane.
-/

import Transformer.Interpolation.Basic
import Transformer.Interpolation.Clustering
import Transformer.Interpolation.HypPropagationFalse
import Mathlib.Analysis.SpecificLimits.Basic

open scoped Topology
open Filter MeasureTheory

namespace Transformer.Interpolation

/-- The proper plane `4x₀ = 3x₁` used for the §3 perturbation counterexample.
Positive points on this plane exist: `(3p/5, 4p/5, q)` lies on the unit
sphere whenever `p² + q² = 1`, with all coordinates positive when `p,q > 0`.
Source: arXiv:2411.04551v3, §3, `lem: perturbation`, Part 1. -/
def perturbationPlane : Set (EucSpace 3) :=
  {x | 4 * x 0 = 3 * x 1}

/-- Any continuous sphere map fixing the complement of `4x₀ = 3x₁` fixes
the whole sphere. For a point in the plane, add `1/(n+1)` to its first
coordinate and normalize. These unit vectors stay outside the plane and
converge to the original point. Continuity and uniqueness of the limit
then force that point to be fixed as well.

This is the obstruction to `eq: identity.flow` when the geodesic hulls
have empty interior in the sphere. Source: arXiv:2411.04551v3, §3. -/
theorem continuous_eq_id_of_fix_off_plane (Φ : SSphere 3 → SSphere 3)
    (hΦ : Continuous Φ)
    (hfix : ∀ x : SSphere 3, (x : EucSpace 3) ∉ perturbationPlane → Φ x = x) :
    Φ = id := by
  funext x
  by_cases hx : (x : EucSpace 3) ∈ perturbationPlane
  · have hxnorm : ‖(x : EucSpace 3)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
    let a : ℕ → ℝ := fun n => 1 / ((n : ℝ) + 1)
    have ha : ∀ n, 0 < a n := fun n => by dsimp [a]; positivity
    let u : ℕ → EucSpace 3 := fun n => (x : EucSpace 3) + a n • !₂[1, 0, 0]
    have hu : ∀ n, u n ≠ 0 := by
      intro n h
      have h0 := congrArg (fun v : EucSpace 3 => v 0) h
      have h1 := congrArg (fun v : EucSpace 3 => v 1) h
      simp [u] at h0 h1
      have hx' : 4 * (x : EucSpace 3) 0 = 3 * (x : EucSpace 3) 1 := hx
      have := ha n
      linarith
    let z : ℕ → SSphere 3 := fun n =>
      ⟨‖u n‖⁻¹ • u n, by
        rw [mem_sphere_zero_iff_norm, norm_smul, Real.norm_eq_abs,
          abs_of_pos (inv_pos.mpr (norm_pos_iff.mpr (hu n))), inv_mul_cancel₀]
        exact norm_ne_zero_iff.mpr (hu n)⟩
    have hz : ∀ n, (z n : EucSpace 3) ∉ perturbationPlane := by
      intro n h
      change 4 * (‖u n‖⁻¹ * ((x : EucSpace 3) 0 + a n * 1)) =
        3 * (‖u n‖⁻¹ * ((x : EucSpace 3) 1 + a n * 0)) at h
      have hx' : 4 * (x : EucSpace 3) 0 = 3 * (x : EucSpace 3) 1 := hx
      have hn : 0 < ‖u n‖⁻¹ := inv_pos.mpr (norm_pos_iff.mpr (hu n))
      have := ha n
      nlinarith
    have ha_lim : Tendsto a atTop (𝓝 0) := tendsto_one_div_add_atTop_nhds_zero_nat
    have hu_lim : Tendsto u atTop (𝓝 (x : EucSpace 3)) := by
      simpa [u] using tendsto_const_nhds.add (ha_lim.smul
        (tendsto_const_nhds (x := (!₂[1, 0, 0] : EucSpace 3))))
    have hz_lim : Tendsto z atTop (𝓝 x) := by
      apply tendsto_subtype_rng.mpr
      simpa [z, hxnorm] using (hu_lim.norm.inv₀ (by simp [hxnorm])).smul hu_lim
    have hlim := hΦ.continuousAt.tendsto.comp hz_lim
    have heq : Φ ∘ z = z := funext (fun n => hfix (z n) (hz n))
    rw [heq] at hlim
    exact tendsto_nhds_unique hlim hz_lim
  · exact hfix x hx

/-- A set in the plane with positive first coordinate has its geodesic
convex hull in that same plane. Convexity preserves the strict positive
coordinate, excluding the origin; radial projection then preserves the
homogeneous plane equation.

The positive-coordinate hypothesis matters because the source defines
`conv_g A` to be the whole sphere when `0 ∈ conv A`.
Source: arXiv:2411.04551v3, §1.5 and §3, `eq: identity.flow`. -/
theorem convG_subset_perturbationPlane {A : Set (SSphere 3)}
    (hplane : ∀ x ∈ A, (x : EucSpace 3) ∈ perturbationPlane)
    (hpos : ∀ x ∈ A, 0 < (x : EucSpace 3) 0) :
    ∀ x ∈ convG 3 A, (x : EucSpace 3) ∈ perturbationPlane := by
  have hp : Convex ℝ perturbationPlane := by
    have hl : IsLinearMap ℝ (fun x : EucSpace 3 => 4 * x 0 - 3 * x 1) :=
      ⟨by intros; simp; ring, by intros; simp; ring⟩
    simpa [perturbationPlane, sub_eq_zero] using convex_hyperplane hl 0
  have hc : convexHull ℝ ((↑) '' A : Set (EucSpace 3)) ⊆ perturbationPlane :=
    convexHull_min (by rintro _ ⟨x, hx, rfl⟩; exact hplane x hx) hp
  have hposc : convexHull ℝ ((↑) '' A : Set (EucSpace 3)) ⊆
      {x | 0 < x 0} :=
    convexHull_min (by rintro _ ⟨x, hx, rfl⟩; exact hpos x hx)
      (convex_halfSpace_gt (f := fun x : EucSpace 3 => x 0)
        ⟨by intros; rfl, by intros; rfl⟩ 0)
  intro x hx
  rcases hx with hz | ⟨v, hv, c, _, hx⟩
  · exact False.elim (lt_irrefl (0 : ℝ) (hposc hz))
  · have hvp : 4 * v 0 = 3 * v 1 := hc hv
    change 4 * (x : EucSpace 3) 0 = 3 * (x : EucSpace 3) 1
    rw [hx]
    change 4 * (c * v 0) = 3 * (c * v 1)
    linear_combination c * hvp

/-- Fixing points outside the two geodesic hulls already forces a continuous
map to be the identity when both sets lie in this plane and have positive
first coordinate. The maps in Part 1 of the source are Lipschitz, so they
meet the continuity hypothesis.
Source: arXiv:2411.04551v3, §3, `lem: perturbation`, `eq: identity.flow`. -/
theorem eq_id_of_fix_outside_plane_hulls (A B : Set (SSphere 3))
    (hA : ∀ x ∈ A, (x : EucSpace 3) ∈ perturbationPlane)
    (hB : ∀ x ∈ B, (x : EucSpace 3) ∈ perturbationPlane)
    (hApos : ∀ x ∈ A, 0 < (x : EucSpace 3) 0)
    (hBpos : ∀ x ∈ B, 0 < (x : EucSpace 3) 0)
    (Φ : SSphere 3 → SSphere 3) (hΦ : Continuous Φ)
    (hfix : ∀ x, x ∉ convG 3 A ∪ convG 3 B → Φ x = x) : Φ = id := by
  apply continuous_eq_id_of_fix_off_plane Φ hΦ
  intro x hx
  apply hfix x
  rintro (hxA | hxB)
  · exact hx (convG_subset_perturbationPlane hA hApos x hxA)
  · exact hx (convG_subset_perturbationPlane hB hBpos x hxB)

/-- All geometric hypotheses above have a nonempty witness: take both sets
to be the singleton `{(3/5, 4/5, 0)}` and the map to be the identity. This
also witnesses the stronger off-plane fixing hypothesis. -/
example : ∃ (A : Set (SSphere 3)) (Φ : SSphere 3 → SSphere 3), A.Nonempty ∧
    (∀ x ∈ A, (x : EucSpace 3) ∈ perturbationPlane) ∧
    (∀ x ∈ A, 0 < (x : EucSpace 3) 0) ∧ Continuous Φ ∧
    (∀ x : SSphere 3, (x : EucSpace 3) ∉ perturbationPlane → Φ x = x) ∧
    (∀ x, x ∉ convG 3 A ∪ convG 3 A → Φ x = x) := by
  let p : SSphere 3 := ⟨!₂[3 / 5, 4 / 5, 0], by
    norm_num [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_three]⟩
  refine ⟨{p}, id, Set.singleton_nonempty p, ?_, ?_, continuous_id,
    fun _ _ => rfl, fun _ _ => rfl⟩
  · rintro x rfl
    norm_num [perturbationPlane, p]
  · rintro x rfl
    norm_num [p]

/-- The even two-atom law is supported on its two atoms: their finite
complement is an open null set. Used for the §3 atomic perturbation witnesses.
Source: arXiv:2411.04551v3, §3, `lem: perturbation`. -/
theorem mem_of_mem_support_halfDirac {d : ℕ} {x y z : SSphere d}
    (hz : z ∈ (halfDirac d x y : Measure (SSphere d)).support) : z = x ∨ z = y := by
  by_contra hne
  push Not at hne
  refine Measure.notMem_support_iff_exists.mpr ⟨{x, y}ᶜ, ?_, ?_⟩ hz
  · exact ((Set.toFinite _).isClosed).isOpen_compl.mem_nhds (by simp [hne.1, hne.2])
  · simp

/-- The barycenter of `½δ_x + ½δ_y` is `½x + ½y`.
Source: arXiv:2411.04551v3, §3, `lem: perturbation`. -/
theorem barycenter_halfDirac {d : ℕ} (x y : SSphere d) :
    barycenter d (halfDirac d x y) = (2⁻¹ : ℝ) • (x : EucSpace d) + (2⁻¹ : ℝ) • (y : EucSpace d) := by
  rw [barycenter, coe_halfDirac, integral_add_measure, integral_smul_measure,
    integral_smul_measure, integral_dirac, integral_dirac]
  · simp
  · exact (integrable_dirac (by simp)).smul_measure (by simp)
  · exact (integrable_dirac (by simp)).smul_measure (by simp)

/-- The support hypothesis is inhabited because this law has total mass one. -/
example : ∃ z : SSphere 1, z ∈
    (halfDirac 1 (basePoint 0) (basePoint 0) : Measure (SSphere 1)).support := by
  apply Measure.nonempty_support
  intro h
  have hm : (halfDirac 1 (basePoint 0) (basePoint 0) : Measure (SSphere 1)) Set.univ = 1 :=
    measure_univ
  simp [h] at hm

end Transformer.Interpolation
