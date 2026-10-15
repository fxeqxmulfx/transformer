/-
# Same-sign original forces within one circular cluster

ArXiv:2411.04990v2, §5, the critical-force split in the proof
of `thm:fixed_centers`.
When the current token is outside a cluster's radius but within
one peak plus that radius of its center, every token in the
cluster exerts a force with the center force's sign, provided
the radius is at most half the actual interaction peak.

We prove this directly from the original signed sine forces.
The proof uses representatives relative to the cluster center
and checks that their differences stay in the first semicircle.
The original force is unchanged by these representatives, so
arbitrary lifted angles and configurations crossing the cut are
included. A nonnegative weighted sum of these aligned forces
cannot cancel the original center force of mass at least one.
This supplies the signed lower bound in the source's induction.
-/

import Transformer.Causal.Section5_NarrowWindow
import Transformer.Causal.Section5_WeightedInteraction

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {ι : Type*}

/-- A displacement outside a smaller central interval and its displacement from a point of that interval have aligned sine signs before the antipodal cut.
Source: arXiv:2411.04990v2, §5, the same-sign assertion in the force split for `thm:fixed_centers`, with the semicircle bound explicit. -/
theorem sin_mul_sin_sub_nonneg (x y : ℝ) (hy : |y| ≤ |x|) (hpi : |x| + |y| ≤ Real.pi) :
    0 ≤ Real.sin x * Real.sin (x - y) := by
  by_cases hx : 0 ≤ x
  · rw [abs_of_nonneg hx] at hy hpi
    have hsx := Real.sin_nonneg_of_mem_Icc ⟨hx, by linarith [abs_nonneg y]⟩
    have hsy := Real.sin_nonneg_of_mem_Icc (show x - y ∈ Icc 0 Real.pi from
      ⟨by linarith [le_abs_self y], by linarith [neg_le_abs y]⟩)
    exact mul_nonneg hsx hsy
  · have hx0 : x < 0 := lt_of_not_ge hx
    rw [abs_of_neg hx0] at hy hpi
    have hsx := Real.sin_nonpos_of_nonpos_of_neg_pi_le hx0.le (by linarith [abs_nonneg y])
    have hsy := Real.sin_nonpos_of_nonpos_of_neg_pi_le
      (show x - y ≤ 0 by linarith [neg_le_abs y]) (show -Real.pi ≤ x - y by linarith [le_abs_self y])
    exact mul_nonneg_of_nonpos_of_nonpos hsx hsy

/-- The actual original force inherits the signed semicircle alignment, since its original exponential factors are positive.
Source: arXiv:2411.04990v2, §5, the aligned original `h` terms in the critical-force split for `thm:fixed_centers`. -/
theorem h_pot_mul_sub_nonneg (β x y : ℝ) (hy : |y| ≤ |x|) (hpi : |x| + |y| ≤ Real.pi) :
    0 ≤ h_pot β x * h_pot β (x - y) := by
  have h := mul_nonneg (mul_pos (Real.exp_pos (β * (Real.cos x - 1)))
    (Real.exp_pos (β * (Real.cos (x - y) - 1)))).le (sin_mul_sin_sub_nonneg x y hy hpi)
  unfold h_pot
  nlinarith only [h]

/-- Taking separate actual angle representatives preserves the original force at their difference.
Source: arXiv:2411.04990v2, Appendix B and §5, periodicity of the original interaction in the circular force split. -/
theorem h_pot_sub_toReal (β x y : ℝ) :
    h_pot β ((x : Real.Angle).toReal - (y : Real.Angle).toReal) = h_pot β (x - y) := by
  simp only [h_pot, Real.cos_sub, Real.sin_sub, Real.Angle.cos_toReal,
    Real.Angle.sin_toReal, Real.Angle.cos_coe, Real.Angle.sin_coe]

/-- Every original token in a narrow cluster has its force aligned with the center force at a current token just outside that cluster.
Source: arXiv:2411.04990v2, §5, the same-sign step in `thm:fixed_centers`; the half-peak radius is explicit and can be derived from its original parameters. -/
theorem h_pot_circular_cluster_alignment (β : ℝ) (hβ : 0 < β) (e θ u v : ℝ)
    (hnarrow : e ≤ interactionPeak β / 2) (hout : e ≤ circDist (u - θ))
    (hu : circDist (u - θ) ≤ interactionPeak β + e) (hv : circDist (v - θ) ≤ e) :
    0 ≤ h_pot β (u - θ) * h_pot β (u - v) := by
  let x := ((u - θ : ℝ) : Real.Angle).toReal
  let y := ((v - θ : ℝ) : Real.Angle).toReal
  have hx : |x| = circDist (u - θ) := rfl
  have hy : |y| = circDist (v - θ) := rfl
  have hsmall : |y| ≤ |x| := by rw [hx, hy]; exact hv.trans hout
  have hpi : |x| + |y| ≤ Real.pi := by
    rw [hx, hy]
    linarith [hnarrow, hu, hv, (interactionPeak_properties β hβ).2.1]
  have h := h_pot_mul_sub_nonneg β x y hsmall hpi
  have hcenter : h_pot β x = h_pot β (u - θ) := by
    simp only [x, h_pot, Real.Angle.cos_toReal, Real.Angle.sin_toReal, Real.Angle.cos_coe, Real.Angle.sin_coe]
  have hother : h_pot β (x - y) = h_pot β (u - v) := by
    rw [show h_pot β (x - y) = h_pot β ((u - θ) - (v - θ)) from h_pot_sub_toReal β (u - θ) (v - θ)]
    congr 1
    ring
  rwa [hcenter, hother] at h

/-- A central mass at least one and any aligned residual force have total magnitude at least the unweighted central force.
Source: arXiv:2411.04990v2, §5, the lower bound on the cluster side of the critical-force equation in `thm:fixed_centers`. -/
theorem abs_core_force_le_of_alignment (a F T : ℝ) (ha : 1 ≤ a) (hFT : 0 ≤ F * T) :
    |F| ≤ |a * F + T| := by
  have ha0 : 0 ≤ a := by linarith
  have ha2 : 0 ≤ a ^ 2 - 1 := by nlinarith
  have h1 := mul_nonneg ha2 (sq_nonneg F)
  have h2 := mul_nonneg ha0 hFT
  have hs : F ^ 2 ≤ (a * F + T) ^ 2 := by nlinarith only [h1, h2, sq_nonneg T]
  have hleft := sq_abs F
  have hright := sq_abs (a * F + T)
  nlinarith only [hs, hleft, hright, abs_nonneg F, abs_nonneg (a * F + T)]

/-- The original positively weighted same-cluster force cannot cancel an original center of mass at least one.
Source: arXiv:2411.04990v2, §5, the cluster-side force lower bound in `thm:fixed_centers`, retaining every original mass and sign. -/
theorem weighted_h_pot_aligned_core_lower (s : Finset ι) (a z : ι → ℝ) (ha : ∀ j ∈ s, 0 ≤ a j)
    (β θ u A : ℝ) (hA : 1 ≤ A)
    (halign : ∀ j ∈ s, 0 ≤ h_pot β (u - θ) * h_pot β (u - z j)) :
    |h_pot β (u - θ)| ≤ |A * h_pot β (u - θ) + ∑ j ∈ s, a j * h_pot β (u - z j)| := by
  apply abs_core_force_le_of_alignment A _ _ hA
  rw [Finset.mul_sum]
  apply Finset.sum_nonneg
  intro j hj
  have h := mul_nonneg (ha j hj) (halign j hj)
  nlinarith only [h]

/-- At a current token outside a narrow cluster, the full original weighted cluster force has magnitude at least its original center force.
Source: arXiv:2411.04990v2, §5, the signed cluster-side lower bound in `thm:fixed_centers`, now with its sufficient narrow-radius geometry proved. -/
theorem weighted_h_pot_narrow_core_lower (s : Finset ι) (a z : ι → ℝ) (ha : ∀ j ∈ s, 0 ≤ a j)
    (β : ℝ) (hβ : 0 < β) (e θ u A : ℝ) (hA : 1 ≤ A)
    (hnarrow : e ≤ interactionPeak β / 2) (hout : e ≤ circDist (u - θ))
    (hu : circDist (u - θ) ≤ interactionPeak β + e) (hcluster : ∀ j ∈ s, circDist (z j - θ) ≤ e) :
    |h_pot β (u - θ)| ≤ |A * h_pot β (u - θ) + ∑ j ∈ s, a j * h_pot β (u - z j)| :=
  weighted_h_pot_aligned_core_lower s a z ha β θ u A hA
    (fun j hj => h_pot_circular_cluster_alignment β hβ e θ u (z j) hnarrow hout hu (hcluster j hj))

/-- Genuine nonzero displacements on both sides of the center jointly satisfy the original signed and circular cluster-force hypotheses.
Source: arXiv:2411.04990v2, §5, the narrow-cluster force comparison for `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ interactionPeak 1 / 4 ≤ interactionPeak 1 / 2 ∧
    interactionPeak 1 / 4 ≤ circDist (interactionPeak 1 / 2 - 0) ∧
    circDist (interactionPeak 1 / 2 - 0) ≤ interactionPeak 1 + interactionPeak 1 / 4 ∧
    circDist (-interactionPeak 1 / 4 - 0) ≤ interactionPeak 1 / 4 ∧
    |(-interactionPeak 1 / 4 : ℝ)| ≤ |interactionPeak 1 / 2| ∧
    |(interactionPeak 1 / 2 : ℝ)| + |(-interactionPeak 1 / 4 : ℝ)| ≤ Real.pi := by
  have ht := interactionPeak_properties 1 one_pos
  have hpos : 0 ≤ interactionPeak 1 / 2 := by linarith [ht.1]
  have hquarter : 0 ≤ interactionPeak 1 / 4 := by linarith [ht.1]
  have hhalf := circDist_eq_self (interactionPeak 1 / 2) ⟨hpos, by linarith [ht.2.1, Real.pi_pos]⟩
  have hq := circDist_eq_self (interactionPeak 1 / 4) ⟨hquarter, by linarith [ht.2.1, Real.pi_pos]⟩
  rw [sub_zero, sub_zero, neg_div, circDist_neg, hhalf, hq, abs_neg, abs_of_nonneg hpos, abs_of_nonneg hquarter]
  exact ⟨one_pos, by linarith [ht.1], by linarith [ht.1], by linarith [ht.1], le_rfl,
    by linarith [ht.1], by linarith [ht.2.1]⟩

/-- A positive original central force and an actual positive same-cluster force jointly realize the nonnegative weighted-sum and unit central mass premises.
Source: arXiv:2411.04990v2, §5, the original cluster force sum in `thm:fixed_centers`. -/
example : (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ (1 : ℝ) * 1 ∧
    (∀ j ∈ (Finset.univ : Finset (Idx 1)), (0 : ℝ) ≤ (fun _ => 1 : Idx 1 → ℝ) j) ∧
    (∀ j ∈ (Finset.univ : Finset (Idx 1)),
      0 ≤ h_pot 1 (interactionPeak 1 / 2 - 0) * h_pot 1 (interactionPeak 1 / 2 - (fun _ : Idx 1 => 0) j)) ∧
    ∀ j ∈ (Finset.univ : Finset (Idx 1)), circDist ((fun _ : Idx 1 => 0) j - 0) ≤ interactionPeak 1 / 4 := by
  have ht := (interactionPeak_properties 1 one_pos).1
  refine ⟨le_rfl, by norm_num, fun _ _ => zero_le_one, fun _ _ => mul_self_nonneg _, ?_⟩
  intro j hj
  rw [sub_self, circDist_zero]
  linarith

end Transformer.Causal
