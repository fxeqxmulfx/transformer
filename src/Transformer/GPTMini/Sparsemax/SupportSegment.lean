import Transformer.GPTMini.Sparsemax.QKPrefixError

/-!
# Sparsemax is affine between scores with the same active support

Derived from arXiv:1602.02068v2, §2.2, Proposition 1, equation
`sparsemax_closedform`. For two score rows with the same zero pattern,
the convex combination of their thresholds certifies the convex
combination of their actual causal simplex projections. No support set
is supplied as a training target, and the projection is not redefined.

This proves an exact segment identity, including inactive threshold ties.
It permits one shared matrix segment to transport several row readouts
simultaneously when the upstream normalized scores are affine on that
segment. The identity alone gives no guarantee that arbitrary endpoints
have the same support, nor that arbitrary QKNorm paths are affine.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- The positive part is affine when both endpoints agree on being zero.
Source: the positive-part formula in §2.2 of arXiv:1602.02068v2,
applied to a convex segment rather than to an assumed derivative. -/
theorem positivePart_segment (x y t : ℝ) (ht : 0 ≤ t) (hu : t ≤ 1)
    (hz : max x 0 = 0 ↔ max y 0 = 0) :
    max ((1 - t) * x + t * y) 0 = (1 - t) * max x 0 + t * max y 0 := by
  have hc : 0 ≤ 1 - t := by linarith
  by_cases hx : max x 0 = 0
  · have hy := hz.mp hx
    have hxn : x ≤ 0 := le_trans (le_max_left x 0) (le_of_eq hx)
    have hyn : y ≤ 0 := le_trans (le_max_left y 0) (le_of_eq hy)
    have hm : (1 - t) * x + t * y ≤ 0 := by
      nlinarith [mul_nonneg hc (neg_nonneg.mpr hxn), mul_nonneg ht (neg_nonneg.mpr hyn)]
    rw [max_eq_right hm, hx, hy]
    ring
  · have hy : max y 0 ≠ 0 := fun h => hx (hz.mpr h)
    have hxp : 0 ≤ x := by
      by_contra hn
      exact hx (max_eq_right (le_of_not_ge hn))
    have hyp : 0 ≤ y := by
      by_contra hn
      exact hy (max_eq_right (le_of_not_ge hn))
    have hm : 0 ≤ (1 - t) * x + t * y :=
      add_nonneg (mul_nonneg hc hxp) (mul_nonneg ht hyp)
    rw [max_eq_left hm, max_eq_left hxp, max_eq_left hyp]

/-- Inactive endpoints satisfy every positive-part segment premise.
Source context: arXiv:1602.02068v2, §2.2, threshold cells. -/
example : max ((1 - (1 / 2 : ℝ)) * (-1) + (1 / 2) * (-2)) 0 =
    (1 - (1 / 2 : ℝ)) * max (-1) 0 + (1 / 2) * max (-2) 0 :=
  positivePart_segment _ _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Active endpoints also satisfy the same segment premises.
Source context: arXiv:1602.02068v2, §2.2, threshold cells. -/
example : max ((1 - (1 / 2 : ℝ)) * 1 + (1 / 2) * 2) 0 =
    (1 - (1 / 2 : ℝ)) * max 1 0 + (1 / 2) * max 2 0 :=
  positivePart_segment _ _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Matching endpoint supports give an exact affine sparsemax segment.
Source: a derived cell property of §2.2 of arXiv:1602.02068v2,
equation `sparsemax_closedform`; causally forbidden entries remain zero. -/
theorem sparseWeights_segment {T : ℕ} (start stop : Fin T → ℝ) (i : Fin T) (t : ℝ)
    (ht : 0 ≤ t) (hu : t ≤ 1)
    (hs : ∀ j, sparseWeights start i j = 0 ↔ sparseWeights stop i j = 0) :
    sparseWeights ((1 - t) • start + t • stop) i =
      (1 - t) • sparseWeights start i + t • sparseWeights stop i := by
  classical
  obtain ⟨leftThreshold, hl⟩ := sparseWeights_exists_threshold start i
  obtain ⟨rightThreshold, hr⟩ := sparseWeights_exists_threshold stop i
  have hclosed : thresholdWeights ((1 - t) • start + t • stop) i
      ((1 - t) * leftThreshold + t * rightThreshold) =
      (1 - t) • thresholdWeights start i leftThreshold +
        t • thresholdWeights stop i rightThreshold := by
    funext j
    by_cases hj : j ≤ i
    · have hz := hs j
      rw [hl, hr] at hz
      simp only [thresholdWeights, hj, ite_true] at hz
      have hm := positivePart_segment (start j - leftThreshold)
        (stop j - rightThreshold) t ht hu hz
      simp only [thresholdWeights, hj, ite_true, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      have hid : (1 - t) * start j + t * stop j -
          ((1 - t) * leftThreshold + t * rightThreshold) =
          (1 - t) * (start j - leftThreshold) + t * (stop j - rightThreshold) := by ring
      rw [hid]
      exact hm
    · simp only [thresholdWeights, hj, ite_false, Pi.add_apply, Pi.smul_apply,
        smul_eq_mul, mul_zero, add_zero]
  have hm0 := (sparseWeights_spec start i).1.2.1
  have hm1 := (sparseWeights_spec stop i).1.2.1
  rw [hl] at hm0
  rw [hr] at hm1
  have hmass : ∑ j, thresholdWeights ((1 - t) • start + t • stop) i
      ((1 - t) * leftThreshold + t * rightThreshold) j = 1 := by
    rw [hclosed]
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hm0, hm1]
    ring
  rw [← thresholdWeights_eq_sparseWeights _ i _ hmass, hclosed, ← hl, ← hr]

/-- The earlier actual sparse anchor row inhabits the segment hypotheses.
Source context: arXiv:1602.02068v2, §2.2, with identical endpoints allowed. -/
example : sparseWeights
    ((1 - (1 / 2 : ℝ)) • anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) +
      (1 / 2 : ℝ) • anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2 =
    (1 - (1 / 2 : ℝ)) • sparseWeights
      (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2 +
      (1 / 2 : ℝ) • sparseWeights
        (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2 :=
  sparseWeights_segment _ _ _ _ (by norm_num) (by norm_num) (fun _ => Iff.rfl)

/-- Linear value aggregation transports the same sparse segment exactly.
Source: §2.2's derived segment identity and the actual value sum at `73f8a0b`;
the endpoint support equality concerns inferred weights, not routing labels. -/
theorem frozenReadout_segment {T : ℕ} {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (values : Fin T → E) (start stop : Fin T → ℝ) (i : Fin T) (t : ℝ)
    (ht : 0 ≤ t) (hu : t ≤ 1)
    (hs : ∀ j, sparseWeights start i j = 0 ↔ sparseWeights stop i j = 0) :
    frozenValueReadout values (sparseWeights ((1 - t) • start + t • stop) i) =
      (1 - t) • frozenValueReadout values (sparseWeights start i) +
        t • frozenValueReadout values (sparseWeights stop i) := by
  rw [sparseWeights_segment start stop i t ht hu hs, map_add, map_smul, map_smul]

/-- A scalar readout satisfies every support-preserving aggregation premise.
Source context: arXiv:1602.02068v2, §2.2, actual value aggregation at `73f8a0b`. -/
example : frozenValueReadout (fun _ : Fin 1 => (3 : ℝ))
    (sparseWeights ((1 - (1 / 2 : ℝ)) • (fun _ : Fin 1 => 0) +
      (1 / 2 : ℝ) • (fun _ : Fin 1 => 0)) 0) =
      (1 - (1 / 2 : ℝ)) • frozenValueReadout (fun _ : Fin 1 => (3 : ℝ))
        (sparseWeights (fun _ : Fin 1 => 0) 0) +
      (1 / 2 : ℝ) • frozenValueReadout (fun _ : Fin 1 => (3 : ℝ))
        (sparseWeights (fun _ : Fin 1 => 0) 0) :=
  frozenReadout_segment _ _ _ _ _ (by norm_num) (by norm_num) (fun _ => Iff.rfl)

/-- An inactive coordinate remains exactly zero throughout the segment.
Source: the derived cell identity of §2.2 of arXiv:1602.02068v2;
no dense mixture is used to preserve an inactive attention position. -/
theorem sparseWeights_segment_zero {T : ℕ} (start stop : Fin T → ℝ) (i j : Fin T) (t : ℝ)
    (ht : 0 ≤ t) (hu : t ≤ 1)
    (hs : ∀ k, sparseWeights start i k = 0 ↔ sparseWeights stop i k = 0)
    (hz : sparseWeights start i j = 0) :
    sparseWeights ((1 - t) • start + t • stop) i j = 0 := by
  rw [sparseWeights_segment start stop i t ht hu hs]
  simp only [Pi.add_apply, Pi.smul_apply, hz, (hs j).mp hz, smul_zero, add_zero]

/-- The actual visible third zero of the anchor row satisfies all premises.
Source context: arXiv:1602.02068v2, §2.2, support-preserving segments. -/
example : sparseWeights
    ((1 - (1 / 2 : ℝ)) • anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) +
      (1 / 2 : ℝ) • anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2 2 = 0 :=
  sparseWeights_segment_zero _ _ _ _ _ (by norm_num) (by norm_num) (fun _ => Iff.rfl)
    (by norm_num [anchoredScores_sparse_example])

end Transformer.GPTMini.Sparsemax
