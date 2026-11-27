module

public import Transformer.Landscape.SaddleAvoidance.Descent

/-!
# Clipped momentum on a fixed objective

The velocity is the Argand Polyak buffer, with global Euclidean clipping
before the buffer update (ARGAND.md, Optimizer and WSD). The descent quantity includes the clipping scale
at the current weights. This lets it cover the part of a run where clipping
is active as well as the eventual unclipped steps.
-/

@[expose] public section

open scoped InnerProductSpace NNReal

namespace Transformer.Optimization

variable {E : Type*} [NormedAddCommGroup E]

/-- The positive denominator of global Euclidean clipping at one. -/
def clipScale (g : E) : ℝ := max ‖g‖ 1

theorem one_le_clipScale (g : E) : 1 ≤ clipScale g := le_max_right _ _

variable [InnerProductSpace ℝ E]

theorem norm_clippedDirection_le (g : E) : ‖(clipScale g)⁻¹ • g‖ ≤ 1 := by
  have h : 0 < clipScale g := zero_lt_one.trans_le (one_le_clipScale g)
  rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr h.le), ← div_eq_inv_mul]
  exact (div_le_one h).2 (le_max_left _ _)

/-- Unit clipping bounds the native momentum buffer at ten. -/
theorem norm_fullMomentum_velocity_le (g v : E) (hv : ‖v‖ ≤ 10) :
    ‖(9 / 10 : ℝ) • v + (clipScale g)⁻¹ • g‖ ≤ 10 := by
  refine (norm_add_le _ _).trans ?_
  rw [norm_smul, Real.norm_of_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10)]
  linarith [norm_clippedDirection_le g]

variable [CompleteSpace E]

/-- Coupling the kinetic term to the clipping scale gives an energy for
ClippedMomentum's velocity on the full gradient of a fixed objective. -/
noncomputable def fullMomentumEnergy (f : E → ℝ) (η : ℝ) (x v : E) : ℝ :=
  f x + η / 2 * clipScale (gradient f x) * ‖v‖ ^ 2

theorem le_fullMomentumEnergy (f : E → ℝ) {η : ℝ} (hη : 0 ≤ η) (x v : E) :
    f x ≤ fullMomentumEnergy f η x v := by
  have h := (zero_le_one : (0 : ℝ) ≤ 1).trans (one_le_clipScale (gradient f x))
  exact le_add_of_nonneg_right (by positivity)

/-- For the native momentum coefficient, the energy decreases by a
positive multiple of both the old and the new squared velocities.
The smoothness bound is only needed on the segment of the step. -/
theorem fullMomentumEnergy_step {f : E → ℝ} {L : ℝ≥0} {s : Set E}
    (hf : Differentiable ℝ f) (hL : LipschitzOnWith L (gradient f) s) (hs : Convex ℝ s)
    {η : ℝ} (hη : 0 < η) (hηL : (L : ℝ) * η ≤ 1 / 240)
    {x y v u : E} (hx : x ∈ s) (hy : y ∈ s) (hu : ‖u‖ ≤ 10)
    (hvel : u = (9 / 10 : ℝ) • v + (clipScale (gradient f x))⁻¹ • gradient f x)
    (hpos : y = x - η • u) :
    fullMomentumEnergy f η y u + η / 40 * (‖v‖ ^ 2 + ‖u‖ ^ 2) ≤
      fullMomentumEnergy f η x v := by
  let q := clipScale (gradient f x)
  let r := clipScale (gradient f y)
  have hq : 1 ≤ q := one_le_clipScale _
  have hqpos : 0 < q := zero_lt_one.trans_le hq
  have hg : gradient f x = q • (u - (9 / 10 : ℝ) • v) := by
    rw [hvel, add_sub_cancel_left, smul_smul, mul_inv_cancel₀ hqpos.ne', one_smul]
  have hd : y - x = -(η • u) := by rw [hpos, sub_sub_cancel_left]
  have hdist : ‖y - x‖ = η * ‖u‖ := by
    rw [hd, norm_neg, norm_smul, Real.norm_of_nonneg hη.le]
  have hbound : r ≤ q + (L : ℝ) * (η * ‖u‖) := by
    have h := hL.norm_sub_le hy hx
    rw [hdist] at h
    apply max_le
    · have hsum := norm_add_le (gradient f y - gradient f x) (gradient f x)
      rw [sub_add_cancel] at hsum
      have hg_le : ‖gradient f x‖ ≤ q := le_max_left _ _
      linarith
    · exact hq.trans (le_add_of_nonneg_right (by positivity))
  have hr : r ≤ q + 10 * ((L : ℝ) * η) := by
    have h := mul_le_mul_of_nonneg_left hu
      (mul_nonneg (show 0 ≤ (L : ℝ) from L.coe_nonneg) hη.le)
    linarith [show (L : ℝ) * (η * ‖u‖) = (L : ℝ) * η * ‖u‖ by ring]
  have hiv : ⟪v, u⟫_ℝ ≤ (‖v‖ ^ 2 + ‖u‖ ^ 2) / 2 := by
    nlinarith [real_inner_le_norm v u, sq_nonneg (‖v‖ - ‖u‖)]
  have hloss := Landscape.SaddleAvoidance.le_add_inner_add_mul_norm_sq_of_segment (x := x) (y := y) hf
    L.coe_nonneg (fun z hz => hL.norm_sub_le (hs.segment_subset hx hy hz) hx)
  rw [hdist, hd, inner_neg_right, real_inner_smul_right, hg,
    real_inner_smul_left, inner_sub_left, real_inner_smul_left, real_inner_self_eq_norm_sq] at hloss
  have hi := mul_le_mul_of_nonneg_left hiv
    (show 0 ≤ η * q * (9 / 10) by positivity)
  have hkin := mul_le_mul_of_nonneg_left hr (show 0 ≤ η / 2 * ‖u‖ ^ 2 by positivity)
  have hvd := mul_le_mul_of_nonneg_right hq (show 0 ≤ η / 20 * (‖v‖ ^ 2 + ‖u‖ ^ 2) by positivity)
  have hdamp := mul_le_mul_of_nonneg_right hηL (show 0 ≤ 6 * η * ‖u‖ ^ 2 by positivity)
  have hout : 0 ≤ η / 40 * ‖v‖ ^ 2 := by positivity
  change f y + η / 2 * r * ‖u‖ ^ 2 + _ ≤ f x + η / 2 * q * ‖v‖ ^ 2
  nlinarith only [hloss, hi, hkin, hvd, hdamp, hout]

example : ‖(9 / 10 : ℝ) • (10 : ℝ) + (clipScale (2 : ℝ))⁻¹ • 2‖ ≤ 10 :=
  norm_fullMomentum_velocity_le 2 10 (by norm_num)

end Transformer.Optimization
