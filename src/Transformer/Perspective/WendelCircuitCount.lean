/-
# Counting the two bad sign patterns of a minimal circuit

For a general-position sample of `d + 1` vectors, exactly two coordinatewise
sign choices place the origin in its convex hull. This is Wendel's formula at
the first sample size beyond full dimension.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelCircuitSigns

namespace Transformer.Perspective

/-- Flipping precisely the vectors with negative circuit coefficient makes a
strictly positive relation, hence puts zero in the convex hull.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem zero_mem_convexHull_flipSigns_negative_mask (d : ℕ)
    (X : SphereTuple d (d + 1)) (c : Idx (d + 1) → ℝ)
    (hc : ∀ i, c i ≠ 0)
    (hrel : ∑ i, c i • (X i : EucSpace d) = 0) :
    (0 : EucSpace d) ∈ convexHull ℝ
      (Set.range fun i : Idx (d + 1) =>
        (flipSigns d (d + 1) (circuitNegativeMask c) X i : EucSpace d)) := by
  classical
  let A : ℝ := ∑ i : Idx (d + 1), |c i|
  have hA : 0 < A := by
    apply Fintype.sum_pos
    rw [Pi.lt_def]
    refine ⟨fun i => abs_nonneg _, ⟨⟨0, by omega⟩, ?_⟩⟩
    exact abs_pos.mpr (hc _)
  let a : Idx (d + 1) → ℝ := fun i => A⁻¹ * |c i|
  have hweighted : ∑ i, |c i| •
      (flipSigns d (d + 1) (circuitNegativeMask c) X i : EucSpace d) = 0 := by
    calc
      _ = ∑ i, c i • (X i : EucSpace d) := by
        apply Finset.sum_congr rfl
        intro i _
        rw [flipSigns_coe]
        by_cases hci : c i < 0
        · simp [circuitNegativeMask, hci, abs_of_neg hci]
        · have hpos : 0 < c i := by
            rcases lt_or_gt_of_ne (hc i) with hn | hp
            · exact False.elim (hci hn)
            · exact hp
          simp [circuitNegativeMask, hci, abs_of_pos hpos]
      _ = 0 := hrel
  apply (zero_mem_convexHull_iff_nonneg_relation d (d + 1)
    (flipSigns d (d + 1) (circuitNegativeMask c) X)).mpr
  refine ⟨a, ?_, ?_, ?_⟩
  · intro i
    exact mul_nonneg (inv_nonneg.mpr hA.le) (abs_nonneg _)
  · change (∑ i : Idx (d + 1), A⁻¹ * |c i|) = 1
    rw [← Finset.mul_sum]
    exact inv_mul_cancel₀ hA.ne'
  · change (∑ i : Idx (d + 1), (A⁻¹ * |c i|) •
      (flipSigns d (d + 1) (circuitNegativeMask c) X i : EucSpace d)) = 0
    simp_rw [mul_smul]
    rw [← Finset.smul_sum, hweighted, smul_zero]

/-- Flipping precisely the vectors with positive circuit coefficient gives
the opposite strictly positive relation and also puts zero in the hull.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem zero_mem_convexHull_flipSigns_positive_mask (d : ℕ)
    (X : SphereTuple d (d + 1)) (c : Idx (d + 1) → ℝ)
    (hc : ∀ i, c i ≠ 0)
    (hrel : ∑ i, c i • (X i : EucSpace d) = 0) :
    (0 : EucSpace d) ∈ convexHull ℝ
      (Set.range fun i : Idx (d + 1) =>
        (flipSigns d (d + 1) (circuitPositiveMask c) X i : EucSpace d)) := by
  classical
  let A : ℝ := ∑ i : Idx (d + 1), |c i|
  have hA : 0 < A := by
    apply Fintype.sum_pos
    rw [Pi.lt_def]
    refine ⟨fun i => abs_nonneg _, ⟨⟨0, by omega⟩, ?_⟩⟩
    exact abs_pos.mpr (hc _)
  let a : Idx (d + 1) → ℝ := fun i => A⁻¹ * |c i|
  have hweighted : ∑ i, |c i| •
      (flipSigns d (d + 1) (circuitPositiveMask c) X i : EucSpace d) = 0 := by
    calc
      _ = ∑ i, -(c i • (X i : EucSpace d)) := by
        apply Finset.sum_congr rfl
        intro i _
        rw [flipSigns_coe]
        by_cases hci : 0 < c i
        · simp [circuitPositiveMask, hci, abs_of_pos hci]
        · have hneg : c i < 0 := by
            rcases lt_or_gt_of_ne (hc i) with hn | hp
            · exact hn
            · exact False.elim (hci hp)
          simp [circuitPositiveMask, hci, abs_of_neg hneg]
      _ = 0 := by rw [Finset.sum_neg_distrib, hrel, neg_zero]
  apply (zero_mem_convexHull_iff_nonneg_relation d (d + 1)
    (flipSigns d (d + 1) (circuitPositiveMask c) X)).mpr
  refine ⟨a, ?_, ?_, ?_⟩
  · intro i
    exact mul_nonneg (inv_nonneg.mpr hA.le) (abs_nonneg _)
  · change (∑ i : Idx (d + 1), A⁻¹ * |c i|) = 1
    rw [← Finset.mul_sum]
    exact inv_mul_cancel₀ hA.ne'
  · change (∑ i : Idx (d + 1), (A⁻¹ * |c i|) •
      (flipSigns d (d + 1) (circuitPositiveMask c) X i : EucSpace d)) = 0
    simp_rw [mul_smul]
    rw [← Finset.smul_sum, hweighted, smul_zero]

/-- Exactly the two masks read from the circuit relation put zero in the
flipped sample's convex hull.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem zero_mem_convexHull_flipSigns_iff_mask (d : ℕ)
    (X : SphereTuple d (d + 1))
    (hgen : ∀ I : Finset (Idx (d + 1)), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (X i : EucSpace d))
    (c : Idx (d + 1) → ℝ) (hc : ∀ i, c i ≠ 0)
    (hrel : ∑ i, c i • (X i : EucSpace d) = 0)
    (mask : Idx (d + 1) → Bool) :
    (0 : EucSpace d) ∈ convexHull ℝ
      (Set.range fun i : Idx (d + 1) =>
        (flipSigns d (d + 1) mask X i : EucSpace d)) ↔
      mask = circuitNegativeMask c ∨ mask = circuitPositiveMask c := by
  constructor
  · exact zero_mem_convexHull_flipSigns_imp_mask d X hgen c hc hrel mask
  · rintro (rfl | rfl)
    · exact zero_mem_convexHull_flipSigns_negative_mask d X c hc hrel
    · exact zero_mem_convexHull_flipSigns_positive_mask d X c hc hrel

end Transformer.Perspective
