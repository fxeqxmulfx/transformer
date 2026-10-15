/-
# The actual two-token overlap and its exponential decay

ArXiv:2411.04990v2, §4, proof of `thm1`, base case `k=2`.
The original leader is fixed and the actual second-token self term
vanishes. Its leader overlap therefore satisfies exactly
`r' = w * (1 - r²)`, with the original positive causal weight `w`.

The spherical overlap stays in `[-1,1]` and is nondecreasing.
Outside the original initial antipode, its initial value exceeds
`-1`. The actual uniform weight floor then supplies the positive
rate `floor * (1 + r(0))` in the scalar Grönwall estimate.
No score, temperature, original normalization, or original initial
condition is replaced. All statements concern actual CSA solutions.
-/

import Transformer.Causal.Section4_ParticleDrift
import Transformer.Causal.Section4_WeightFloor
import Transformer.Perspective.Gronwall

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {d : ℕ}

/-- Every actual unit-token overlap lies in the original spherical interval.
Source: arXiv:2411.04990v2, §4, the scalar overlaps in the proof of `thm1`. -/
theorem csaSphereOverlap_bounds (x y : SSphere d) :
    -1 ≤ inner ℝ (x : EucSpace d) (y : EucSpace d) ∧
      inner ℝ (x : EucSpace d) (y : EucSpace d) ≤ 1 := by
  apply abs_le.mp
  simpa only [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2, mul_one]
    using abs_real_inner_le_norm (x : EucSpace d) (y : EucSpace d)

/-- The only original spherical point with overlap `-1` is the original antipode.
Source: arXiv:2411.04990v2, §4, the exceptional two-token initialization in `thm1`. -/
theorem csaSphereOverlap_gt_neg_one (x y : SSphere d) (hxy : x ≠ Perspective.antipode d y) :
    -1 < inner ℝ (x : EucSpace d) (y : EucSpace d) := by
  have hn : inner ℝ (x : EucSpace d) (y : EucSpace d) ≠ -1 := by
    intro h
    apply hxy
    apply Subtype.ext
    exact (inner_eq_neg_one_iff_of_norm_eq_one
      (mem_sphere_zero_iff_norm.mp x.2) (mem_sphere_zero_iff_norm.mp y.2)).mp h
  exact lt_of_le_of_ne (csaSphereOverlap_bounds x y).1 hn.symm

/-- The original squared distance to the leader is exactly twice the original overlap defect.
Source: arXiv:2411.04990v2, §4, the passage from leader overlaps to spherical clustering. -/
theorem csaSphere_dist_sq (x y : SSphere d) :
    ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 =
      2 * (1 - inner ℝ (x : EucSpace d) (y : EucSpace d)) := by
  rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2]
  ring

/-- The actual second original token has precisely the scalar logistic overlap derivative.
Source: arXiv:2411.04990v2, §4, the base case `k=2` in the proof of `thm1`. -/
theorem hasDerivAt_csaSecondOverlap (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d 2)
    (hX : CSA d 2 β Q K (ContinuousLinearMap.id ℝ _) X) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ (X s 1 : EucSpace d) (X 0 0 : EucSpace d))
      (csaTokenWeight β Q K (fun j => (X t j : EucSpace d)) 1 0 *
        (1 - inner ℝ (X t 1 : EucSpace d) (X 0 0 : EucSpace d) ^ 2)) t := by
  have h := hasDerivAt_csaOverlap_id β Q K X hX 1 t
  have ha : inner ℝ (X 0 0 : EucSpace d) (X 0 0 : EucSpace d) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (X 0 0).2, one_pow]
  have hx : inner ℝ (X t 1 : EucSpace d) (X t 1 : EucSpace d) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (X t 1).2, one_pow]
  apply h.congr_deriv
  change (∑ j : Idx 2, csaTokenWeight β Q K (fun l => (X t l : EucSpace d)) 1 j *
      (inner ℝ (X t j : EucSpace d) (X 0 0 : EucSpace d) -
        inner ℝ (X t 1 : EucSpace d) (X t j : EucSpace d) *
          inner ℝ (X t 1 : EucSpace d) (X 0 0 : EucSpace d))) = _
  rw [Fin.sum_univ_two]
  simp only [csaLeader_constant β Q K X hX t, ha, hx,
    one_mul, sub_self, mul_zero, add_zero, pow_two]

/-- The actual second-token leader overlap is nondecreasing for every original real time.
Source: arXiv:2411.04990v2, §4, the two-token base case of `thm1`. -/
theorem csaSecondOverlap_monotone (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d 2)
    (hX : CSA d 2 β Q K (ContinuousLinearMap.id ℝ _) X) :
    Monotone (fun t => inner ℝ (X t 1 : EucSpace d) (X 0 0 : EucSpace d)) := by
  apply monotone_of_deriv_nonneg (fun t => (hasDerivAt_csaSecondOverlap β Q K X hX t).differentiableAt)
  intro t
  rw [(hasDerivAt_csaSecondOverlap β Q K X hX t).deriv]
  apply mul_nonneg (csaTokenWeight_nonneg β Q K _ 1 0)
  have hr := csaSphereOverlap_bounds (X t 1) (X 0 0)
  nlinarith [mul_nonneg (sub_nonneg.mpr hr.2) (by linarith :
    0 ≤ 1 + inner ℝ (X t 1 : EucSpace d) (X 0 0 : EucSpace d))]

/-- The original second-token overlap defect has an explicit exponential upper bound.
Source: arXiv:2411.04990v2, §4, the positive two-token drift proving the base case of `thm1`.
The rate is the actual original weight floor times the original initial antipodal margin. -/
theorem csaSecondOverlap_decay (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d 2)
    (hX : CSA d 2 β Q K (ContinuousLinearMap.id ℝ _) X) (t : ℝ) (ht : 0 ≤ t) :
    1 - inner ℝ (X t 1 : EucSpace d) (X 0 0 : EucSpace d) ≤
      (1 - inner ℝ (X 0 1 : EucSpace d) (X 0 0 : EucSpace d)) *
        Real.exp (-(csaWeightFloor β Q K 2 *
          (1 + inner ℝ (X 0 1 : EucSpace d) (X 0 0 : EucSpace d))) * t) := by
  have hd : ∀ s : ℝ, 0 ≤ s → ∃ c : ℝ,
      HasDerivAt (fun s => inner ℝ (X s 1 : EucSpace d) (X 0 0 : EucSpace d)) c s ∧
      (csaWeightFloor β Q K 2 * (1 + inner ℝ (X 0 1 : EucSpace d) (X 0 0 : EucSpace d))) *
        (1 - inner ℝ (X s 1 : EucSpace d) (X 0 0 : EucSpace d)) ≤ c := by
    intro s hs
    refine ⟨_, hasDerivAt_csaSecondOverlap β Q K X hX s, ?_⟩
    have hm := csaSecondOverlap_monotone β Q K X hX hs
    have hr := csaSphereOverlap_bounds (X s 1) (X 0 0)
    have hf := csaWeightFloor_le β Q K (fun j => (X s j : EucSpace d))
      (fun j => mem_sphere_zero_iff_norm.mp (X s j).2) (1 : Idx 2) 0 (by norm_num)
    have hprod : 0 ≤ 1 - inner ℝ (X s 1 : EucSpace d) (X 0 0 : EucSpace d) ^ 2 := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hr.2) (by linarith :
        0 ≤ 1 + inner ℝ (X s 1 : EucSpace d) (X 0 0 : EucSpace d))]
    calc
      _ ≤ (csaWeightFloor β Q K 2 *
          (1 + inner ℝ (X s 1 : EucSpace d) (X 0 0 : EucSpace d))) *
            (1 - inner ℝ (X s 1 : EucSpace d) (X 0 0 : EucSpace d)) :=
        mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left (by linarith) (csaWeightFloor_pos β Q K (by norm_num : 0 < 2)).le)
          (sub_nonneg.mpr hr.2)
      _ = csaWeightFloor β Q K 2 *
          (1 - inner ℝ (X s 1 : EucSpace d) (X 0 0 : EucSpace d) ^ 2) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_right hf hprod
  simpa only [sub_zero, zero_sub, mul_neg, neg_mul] using
    Perspective.decay_of_deriv_ge _ 0 _ hd t ht

/-- Away from the original initial antipode, the actual two-token decay rate is strictly positive.
Source: arXiv:2411.04990v2, §4, the two-token base case of `thm1`. -/
theorem csaSecondOverlap_rate_pos (β : ℝ) (Q K : ParamMatrix d) (X₀ : SphereTuple d 2)
    (hanti : X₀ 1 ≠ Perspective.antipode d (X₀ 0)) :
    0 < csaWeightFloor β Q K 2 * (1 + inner ℝ (X₀ 1 : EucSpace d) (X₀ 0 : EucSpace d)) := by
  apply mul_pos (csaWeightFloor_pos β Q K (by norm_num : 0 < 2))
  have h := csaSphereOverlap_gt_neg_one (X₀ 1) (X₀ 0) hanti
  linarith

/-- The original antipodal margin, CSA dynamics, and nonnegative forward time have a joint nonzero witness.
Source: arXiv:2411.04990v2, §4, the original consensus equilibria in the proof of `thm1`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
      (basePoint 1 : SSphere 2) ≠ Perspective.antipode 2 (basePoint 1) ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨CSA_const_consensus β Q K (basePoint 1), ?_, by norm_num⟩
  intro h
  have hv := congrArg Subtype.val h
  have hn := mem_sphere_zero_iff_norm.mp (basePoint 1).2
  change (basePoint 1 : EucSpace 2) = -(basePoint 1 : EucSpace 2) at hv
  have hip : inner ℝ (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, hn, one_pow]
  have hp := congrArg (fun z => inner ℝ (basePoint 1 : EucSpace 2) z) hv
  rw [inner_neg_right, hip] at hp
  norm_num at hp

end Transformer.Causal
