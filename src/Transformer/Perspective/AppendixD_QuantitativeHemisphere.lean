/-
# A uniform hemisphere for almost orthogonal initial data

ArXiv:2312.10794v5, Appendix D, Step 3, and §6.1,
`lem: hemisphere.clustering`. Normalizing the sum of the orthogonal
comparison points gives a common direction. Its height at each comparison
point is at least `1/n`. A perturbation of at most `1/(2n)` therefore
retains height at least `1/(2n)` for every original token.

The same height bound controls the initial chart spread independently of
dimension. The chart's inverse normalization transfers its proved width
contraction to the actual pairwise inner products. These explicit estimates
will give constants uniform over the initial configurations in Appendix D's
almost-orthogonality event.
-/

import Transformer.Perspective.ConeLimit

open scoped BigOperators
open Real

namespace Transformer.Perspective

variable {d n : ℕ}

/-- Every orthogonal comparison point has inner product one with their sum.
Source: arXiv:2312.10794v5, Appendix D, Step 3, the common hemisphere. -/
theorem inner_orthogonal_sum (Y : SphereTuple d n)
    (hY : ∀ i j : Idx n, i ≠ j → inner (𝕜 := ℝ) (Y i : EucSpace d) (Y j : EucSpace d) = 0)
    (i : Idx n) : inner (𝕜 := ℝ) (Y i : EucSpace d) (∑ j, (Y j : EucSpace d)) = 1 := by
  classical
  rw [inner_sum, Finset.sum_eq_single i]
  · rw [real_inner_self_eq_norm_sq, norm_coe_tuple, one_pow]
  · intro j hj hji
    exact hY i j hji.symm
  · simp

example : ∀ i j : Idx 2, i ≠ j → inner (𝕜 := ℝ)
    (EuclideanSpace.single i (1 : ℝ) : EucSpace 2) (EuclideanSpace.single j 1) = 0 := by
  intro i j hij
  simp [EuclideanSpace.inner_single_left, hij]

/-- Almost orthogonal data have a common hemisphere with a height floor
depending only on the token count. Source: arXiv:2312.10794v5,
Appendix D, Step 3, `eq: d.large`; this uses the initial comparison frame
directly, instead of a differentiable minimum along the eventual limit. -/
theorem uniform_hemisphere_of_near_orthogonal (hn : 0 < n) (X Y : SphereTuple d n)
    (hY : ∀ i j : Idx n, i ≠ j → inner (𝕜 := ℝ) (Y i : EucSpace d) (Y j : EucSpace d) = 0)
    {δ : ℝ} (hδ : δ ≤ 1 / (2 * (n : ℝ)))
    (hclose : ∀ i : Idx n, ‖(X i : EucSpace d) - (Y i : EucSpace d)‖ ≤ δ) :
    ∃ w : SSphere d, ∀ i : Idx n,
      1 / (2 * (n : ℝ)) ≤ inner (𝕜 := ℝ) (X i : EucSpace d) (w : EucSpace d) := by
  let S : EucSpace d := ∑ j, (Y j : EucSpace d)
  have hS : S ≠ 0 := by
    intro hz
    have h := inner_orthogonal_sum Y hY ⟨0, hn⟩
    change inner (𝕜 := ℝ) (Y ⟨0, hn⟩ : EucSpace d) S = 1 at h
    rw [hz, inner_zero_right] at h
    norm_num at h
  have hSn : ‖S‖ ≤ (n : ℝ) := by
    calc
      _ ≤ ∑ j : Idx n, ‖(Y j : EucSpace d)‖ := norm_sum_le _ _
      _ = n := by simp
  let w : SSphere d := ⟨‖S‖⁻¹ • S,
    mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm (𝕜 := ℝ) hS)⟩
  have hw : ‖(w : EucSpace d)‖ = 1 := norm_eq_of_mem_sphere w
  have hny : 1 / (n : ℝ) ≤ ‖S‖⁻¹ := by
    simpa only [one_div] using one_div_le_one_div_of_le (norm_pos_iff.mpr hS) hSn
  have hfrac : 1 / (n : ℝ) = 2 * (1 / (2 * (n : ℝ))) := by field_simp
  refine ⟨w, fun i => ?_⟩
  have hy : 1 / (n : ℝ) ≤ inner (𝕜 := ℝ) (Y i : EucSpace d) (w : EucSpace d) := by
    change 1 / (n : ℝ) ≤ inner (𝕜 := ℝ) (Y i : EucSpace d) (‖S‖⁻¹ • S)
    rw [inner_smul_right, inner_orthogonal_sum Y hY, mul_one]
    exact hny
  have he := abs_real_inner_le_norm ((X i : EucSpace d) - (Y i : EucSpace d)) (w : EucSpace d)
  rw [hw, mul_one, inner_sub_left] at he
  have hl := (abs_le.mp he).1
  linarith [hclose i]

/-- Two actual unit axes realize every near-orthogonal geometry hypothesis. -/
example : ∃ X Y : SphereTuple 2 2, (0 : ℕ) < 2 ∧
    (∀ i j : Idx 2, i ≠ j → inner (𝕜 := ℝ) (Y i : EucSpace 2) (Y j : EucSpace 2) = 0) ∧
    (0 : ℝ) ≤ 1 / (2 * (2 : ℝ)) ∧
    ∀ i : Idx 2, ‖(X i : EucSpace 2) - (Y i : EucSpace 2)‖ ≤ 0 := by
  let Y : SphereTuple 2 2 := fun i => ⟨EuclideanSpace.single i 1, by simp⟩
  refine ⟨Y, Y, by decide, ?_, by norm_num, fun i => by simp⟩
  intro i j hij
  simp [Y, EuclideanSpace.inner_single_left, hij]

/-- A common height floor bounds the chart spread by `2n²/r`, uniformly
in the ambient dimension. Source: arXiv:2312.10794v5, §6.1,
`lem: hemisphere.clustering`, the chart contraction used in Appendix D. -/
theorem chartSpread_le_of_height (X : ℝ → SphereTuple d n) (w : EucSpace d)
    {r : ℝ} (hr : 0 < r)
    (h0 : ∀ i : Idx n, r ≤ inner (𝕜 := ℝ) (X 0 i : EucSpace d) w) :
    chartSpread w X ≤ 2 * (n : ℝ) ^ 2 / r := by
  have hb (i : Idx n) : ‖hemiChart w (X 0 i : EucSpace d)‖ ≤ r⁻¹ := by
    rw [(hemiChart_normalize (norm_coe_tuple _ _) (hr.trans_le (h0 i))).1]
    exact inv_anti₀ hr (h0 i)
  calc
    _ ≤ ∑ _p : Idx n, ∑ _q : Idx n, (2 / r : ℝ) := by
      apply Finset.sum_le_sum
      intro p hp
      apply Finset.sum_le_sum
      intro q hq
      apply (norm_sub_le _ _).trans
      have h := add_le_add (hb p) (hb q)
      simpa only [div_eq_mul_inv, mul_one, two_mul] using h
    _ = _ := by simp [Finset.sum_const, nsmul_eq_mul]; ring

example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ inner (𝕜 := ℝ)
    ((basePoint 0 : SSphere 1) : EucSpace 1) ((basePoint 0 : SSphere 1) : EucSpace 1) := by
  simp

/-- Inverse normalization transfers chart width to actual sphere distance.
Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`,
the quantitative chart argument applied to Appendix D's configurations. -/
theorem norm_sub_le_two_chart_width (w : SSphere d) (x y : SSphere d)
    (hx : 0 < inner (𝕜 := ℝ) (x : EucSpace d) (w : EucSpace d))
    (hy : 0 < inner (𝕜 := ℝ) (y : EucSpace d) (w : EucSpace d)) :
    ‖(x : EucSpace d) - (y : EucSpace d)‖ ≤
      2 * ‖hemiChart (w : EucSpace d) (x : EucSpace d) -
        hemiChart (w : EucSpace d) (y : EucSpace d)‖ := by
  have hxN := hemiChart_normalize (norm_coe_tuple (fun _ : Idx 1 => x) 0) hx
  have hyN := hemiChart_normalize (norm_coe_tuple (fun _ : Idx 1 => y) 0) hy
  have hu : 1 ≤ ‖hemiChart (w : EucSpace d) (x : EucSpace d)‖ := by
    rw [hxN.1]
    simpa only [inv_one] using (inv_le_inv₀ one_pos hx).mpr
      (height_le_one (norm_eq_of_mem_sphere w) x)
  have hv : 0 < ‖hemiChart (w : EucSpace d) (y : EucSpace d)‖ := by rw [hyN.1]; positivity
  have h := norm_normalize_sub_le hu hv
  rwa [hxN.2, hyN.2] at h

example : (0 : ℝ) < inner (𝕜 := ℝ) ((basePoint 0 : SSphere 1) : EucSpace 1)
    ((basePoint 0 : SSphere 1) : EucSpace 1) := by simp

/-- The actual pairwise inner-product defect is at most twice the chart
width. Source: arXiv:2312.10794v5, Appendix D, `e:ineqsecondpart`,
using the quantitative hemisphere chart in place of the limiting minimum. -/
theorem inner_defect_le_two_chart_width (w : SSphere d) (x y : SSphere d)
    (hx : 0 < inner (𝕜 := ℝ) (x : EucSpace d) (w : EucSpace d))
    (hy : 0 < inner (𝕜 := ℝ) (y : EucSpace d) (w : EucSpace d)) :
    1 - inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) ≤
      2 * ‖hemiChart (w : EucSpace d) (x : EucSpace d) -
        hemiChart (w : EucSpace d) (y : EucSpace d)‖ := by
  calc
    _ = inner (𝕜 := ℝ) (x : EucSpace d) ((x : EucSpace d) - (y : EucSpace d)) := by
      rw [inner_sub_right, real_inner_self_eq_norm_sq, norm_eq_of_mem_sphere, one_pow]
    _ ≤ ‖(x : EucSpace d) - (y : EucSpace d)‖ := by
      simpa only [norm_eq_of_mem_sphere, one_mul] using
        real_inner_le_norm (x : EucSpace d) ((x : EucSpace d) - (y : EucSpace d))
    _ ≤ _ := norm_sub_le_two_chart_width w x y hx hy

example : (0 : ℝ) < inner (𝕜 := ℝ) ((basePoint 0 : SSphere 1) : EucSpace 1)
    ((basePoint 0 : SSphere 1) : EucSpace 1) := by simp

end Transformer.Perspective
