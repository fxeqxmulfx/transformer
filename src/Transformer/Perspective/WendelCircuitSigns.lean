/-
# Sign patterns of a minimal linear circuit

For `d + 1` points in `d` dimensions, the unique linear relation controls
which coordinatewise sign choices put the origin in the sample's convex hull.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelCircuit
import Transformer.Perspective.WendelConvexHull
import Transformer.Perspective.WendelSignAverage

namespace Transformer.Perspective

/-- Coordinatewise reflection acts by multiplication by `-1` on the ambient
vector. Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem flipSigns_coe (d n : ℕ) (mask : Idx n → Bool)
    (X : SphereTuple d n) (i : Idx n) :
    (flipSigns d n mask X i : EucSpace d) =
      if mask i then -(X i : EucSpace d) else (X i : EucSpace d) := by
  cases h : mask i <;> simp [flipSigns, sphereMap, h]

/-- The two candidates for a sign pattern whose flipped sample has the origin
in its convex hull: orient every coefficient of the circuit to be positive,
or orient every coefficient to be negative.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def circuitNegativeMask {d : ℕ} (c : Idx (d + 1) → ℝ) :
    Idx (d + 1) → Bool := fun i => decide (c i < 0)

noncomputable def circuitPositiveMask {d : ℕ} (c : Idx (d + 1) → ℝ) :
    Idx (d + 1) → Bool := fun i => decide (0 < c i)

/-- No other sign pattern can put the origin in the convex hull of a
`d + 1`-point sample in general position.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem zero_mem_convexHull_flipSigns_imp_mask (d : ℕ) (X : SphereTuple d (d + 1))
    (hgen : ∀ I : Finset (Idx (d + 1)), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (X i : EucSpace d))
    (c : Idx (d + 1) → ℝ) (hc : ∀ i, c i ≠ 0)
    (hrel : ∑ i, c i • (X i : EucSpace d) = 0)
    (mask : Idx (d + 1) → Bool)
    (hzero : (0 : EucSpace d) ∈ convexHull ℝ
      (Set.range fun i : Idx (d + 1) => (flipSigns d (d + 1) mask X i : EucSpace d))) :
    mask = circuitNegativeMask c ∨ mask = circuitPositiveMask c := by
  classical
  obtain ⟨a, ha, hasum, harel⟩ :=
    (zero_mem_convexHull_iff_nonneg_relation d (d + 1) (flipSigns d (d + 1) mask X)).mp hzero
  let b : Idx (d + 1) → ℝ := fun i => if mask i then -a i else a i
  have hbrel : ∑ i, b i • (X i : EucSpace d) = 0 := by
    calc
      _ = ∑ i, a i • (flipSigns d (d + 1) mask X i : EucSpace d) := by
        apply Finset.sum_congr rfl
        intro i _
        rw [flipSigns_coe]
        cases h : mask i <;> simp [b, h]
      _ = 0 := harel
  obtain ⟨t, ht⟩ := relation_eq_smul_of_full_support d X hgen c hc hrel b hbrel
  have htn : t ≠ 0 := by
    intro ht0
    have hazero : ∀ i, a i = 0 := by
      intro i
      have hi := congrFun ht i
      simp only [b, ht0, zero_mul] at hi
      cases h : mask i <;> simpa [h] using hi
    simp [hazero] at hasum
  rcases lt_or_gt_of_ne htn with htneg | htpos
  · right
    funext i
    have hi := congrFun ht i
    by_cases hpos : 0 < c i
    · have hm : mask i = true := by
        cases h : mask i with
        | true => rfl
        | false =>
          simp [b, h] at hi
          have htc := mul_neg_of_neg_of_pos htneg hpos
          linarith [ha i]
      simp [circuitPositiveMask, hpos, hm]
    · have hneg : c i < 0 := by
        rcases lt_or_gt_of_ne (hc i) with hn | hp
        · exact hn
        · exact False.elim (hpos hp)
      have hm : mask i = false := by
        cases h : mask i with
        | false => rfl
        | true =>
          simp [b, h] at hi
          have htc := mul_pos_of_neg_of_neg htneg hneg
          linarith [ha i]
      simp [circuitPositiveMask, hpos, hm]
  · left
    funext i
    have hi := congrFun ht i
    by_cases hneg : c i < 0
    · have hm : mask i = true := by
        cases h : mask i with
        | true => rfl
        | false =>
          simp [b, h] at hi
          have htc := mul_neg_of_pos_of_neg htpos hneg
          linarith [ha i]
      simp [circuitNegativeMask, hneg, hm]
    · have hpos : 0 < c i := by
        rcases lt_or_gt_of_ne (hc i) with hn | hp
        · exact False.elim (hneg hn)
        · exact hp
      have hm : mask i = false := by
        cases h : mask i with
        | false => rfl
        | true =>
          simp [b, h] at hi
          have htc := mul_pos htpos hpos
          linarith [ha i]
      simp [circuitNegativeMask, hneg, hm]

end Transformer.Perspective
