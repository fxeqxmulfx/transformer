/-
# Wendel's count at `n = d + 1`

The unique circuit of `d + 1` points in `d` dimensions has exactly two sign
orientations that put the origin in the convex hull. Every other orientation
lies in a common open hemisphere.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelCircuitCount

open MeasureTheory

namespace Transformer.Perspective

/-- The `d + 1`-point general-position sample has exactly `2^(d+1) - 2`
successful sign patterns.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem hemisphereSignCount_circuit (d : ℕ) (X : SphereTuple d (d + 1))
    (hgen : ∀ I : Finset (Idx (d + 1)), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (X i : EucSpace d)) :
    hemisphereSignCount d (d + 1) X = 2 ^ (d + 1) - 2 := by
  classical
  obtain ⟨c, hc, hrel⟩ := exists_full_support_relation d X hgen
  have hmaskne : circuitNegativeMask c ≠ circuitPositiveMask c := by
    intro h
    let i : Idx (d + 1) := ⟨0, by omega⟩
    have hi := congrFun h i
    rcases lt_or_gt_of_ne (hc i) with hneg | hpos
    · have hnpos : ¬0 < c i := not_lt.mpr hneg.le
      simp [circuitNegativeMask, circuitPositiveMask, hneg, hnpos] at hi
    · have hnneg : ¬c i < 0 := not_lt.mpr hpos.le
      simp [circuitNegativeMask, circuitPositiveMask, hpos, hnneg] at hi
  have hfilter :
      (Finset.univ.filter fun mask : Idx (d + 1) → Bool =>
        ∃ w : SSphere d, ∀ i : Idx (d + 1),
          0 < inner (𝕜 := ℝ) ((flipSigns d (d + 1) mask X i : EucSpace d))
            ((w : EucSpace d))) =
        Finset.univ \ {circuitNegativeMask c, circuitPositiveMask c} := by
    ext mask
    have heq := exists_openHemisphere_iff_zero_notMem_convexHull d (d + 1)
      (by omega : 1 ≤ d + 1) (flipSigns d (d + 1) mask X)
    have hbad := zero_mem_convexHull_flipSigns_iff_mask d X hgen c hc hrel mask
    simpa only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_sdiff,
      Finset.mem_insert, Finset.mem_singleton] using heq.trans (not_congr hbad)
  change (Finset.univ.filter fun mask : Idx (d + 1) → Bool =>
      ∃ w : SSphere d, ∀ i : Idx (d + 1),
        0 < inner (𝕜 := ℝ) ((flipSigns d (d + 1) mask X i : EucSpace d))
          ((w : EucSpace d))).card = _
  rw [hfilter, Finset.card_sdiff]
  simp [hmaskne, Fintype.card_bool]

/-- **Wendel's theorem at `n = d + 1`.** The sample has exactly two bad sign
patterns, so its hemisphere probability is `1 - 2^{-d}`.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem wendel_circuit (d : ℕ) (hd : 1 ≤ d) :
    ∀ P : Measure (SphereTuple d (d + 1)), UniformTuple d (d + 1) P →
      (P {X : SphereTuple d (d + 1) | ∃ w : SSphere d, ∀ i : Idx (d + 1),
        0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))}).toReal =
          (∑ k ∈ Finset.range d, (((d + 1 - 1).choose k) : ℝ)) /
            2 ^ (d + 1 - 1) := by
  intro P hP
  have hsum : (∑ k ∈ Finset.range d, d.choose k) + 1 = 2 ^ d := by
    simpa [Finset.sum_range_succ] using Nat.sum_range_choose d
  have hn : 1 ≤ d + 1 := hd.trans (Nat.le_succ d)
  apply wendel_of_ae_sign_count d (d + 1) hn P hP
  filter_upwards [ae_linearGeneralPosition d (d + 1) P hP] with X hX
  rw [hemisphereSignCount_circuit d X hX]
  simp only [Nat.add_sub_cancel_right]
  rw [pow_succ]
  omega

/-- The dimension and sample-size hypotheses of `wendel_circuit` are
satisfiable at `d = 1`, `n = 2`. -/
example : 1 ≤ 1 ∧ (1 : ℕ) + 1 = 2 := ⟨le_rfl, rfl⟩

end Transformer.Perspective
