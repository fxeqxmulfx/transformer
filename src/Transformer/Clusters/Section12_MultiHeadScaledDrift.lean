/-
# Exact scaled head scores and bounded original multi-head drift

ArXiv:2305.05465v6, §7, proof of `l:boundedother`, with
§3's discrete rescaling and §12's original separate head sum.
Scaling an original scalar query by s while dividing every anchor
by s preserves its scores when s is nonzero. Its weighted coordinate
average is consequently divided by s. Summing over the actual heads
retains that identity with their own query-key and value coefficients.

The original coordinate equation then identifies this scaled summed
drift with (x_next-x)/(Δt s). A genuinely bounded original coordinate
has an increment of magnitude at most twice its bound. At the full
original scale s=q^k, this gives a scaled drift tending to zero.
These are consequences of the original vector update and actual
boundedness, rather than an assumed limiting scalar equation.
-/

import Transformer.Clusters.Section12_MultiHeadFullGrowth

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {n H : ℕ}

/-- Original scalar head scores are invariant under inverse anchor scaling and direct query scaling.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for each separate head of §12. -/
theorem softmaxWeight_scaled_query (c : ℝ) (y : Idx n → ℝ) (r s : ℝ) (hs : s ≠ 0) (j : Idx n) :
    Perspective.softmaxWeight (fun l => c * (s * r) * (y l / s)) j =
      Perspective.softmaxWeight (fun l => c * r * y l) j := by
  have he : (fun l => c * (s * r) * (y l / s)) = (fun l => c * r * y l) := by
    funext l
    field_simp [hs]
  rw [he]

/-- The actual original attention row has exactly its scaled-query, normalized-anchor scores.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for the original matrices of §12. -/
theorem attentionMatrix_scaled_scores (Q K : ParamMatrix 1) (Y : Idx n → EucSpace 1)
    (i j : Idx n) (s : ℝ) (hs : s ≠ 0) :
    attentionMatrix Q K Y i j = Perspective.softmaxWeight
      (fun l => inner (𝕜 := ℝ) (Q unit1) (K unit1) * (s * Y i 0) * (Y l 0 / s)) j := by
  simpa only [attentionMatrix_eq_one_dim, mul_assoc] using
    (softmaxWeight_scaled_query (inner (𝕜 := ℝ) (Q unit1) (K unit1)) (fun l => Y l 0) (Y i 0) s hs j).symm

/-- Every original head has its exact scaled-query scores at the full original positive layer factor.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for §12's original separate heads. -/
theorem multiHead_attention_rescaled {Δt : ℝ} (hΔt : 0 ≤ Δt)
    (Q K V : Idx H → ParamMatrix 1) (hV : ∀ h, IsPosDefOp (V h))
    (X : ℕ → Idx n → EucSpace 1) (h : Idx H) (k : ℕ) (i j : Idx n) :
    attentionMatrix (Q h) (K h) (X k) i j = Perspective.softmaxWeight
      (fun l => inner (𝕜 := ℝ) (Q h unit1) (K h unit1) *
        (multiHeadLayerFactor Δt V ^ k * X k i 0) * (X k l 0 / multiHeadLayerFactor Δt V ^ k)) j := by
  have hq : 0 < multiHeadLayerFactor Δt V := by
    have he := multiHeadLayerFactor_ge_one hΔt V hV
    linarith
  exact attentionMatrix_scaled_scores (Q h) (K h) (X k) i j _ (pow_pos hq k).ne'

/-- The exact original head average scales by the inverse anchor factor.
Source: arXiv:2305.05465v6, §7, the rescaled mean in `l:boundedother`, for §12's original heads. -/
theorem scalarHeadAverage_div_scale (c : ℝ) (y : Idx n → ℝ) (r s : ℝ) :
    scalarHeadAverage c (fun l => y l / s) (s * r) = scalarHeadAverage c y r / s := by
  by_cases hs : s = 0
  · subst s
    simp [scalarHeadAverage]
  unfold scalarHeadAverage
  simp_rw [softmaxWeight_scaled_query c y r s hs, ← mul_div_assoc]
  rw [← Finset.sum_div]

/-- The original weighted finite head sum retains the exact inverse-factor scaling.
Source: arXiv:2305.05465v6, §12's head sum and §7, the rescaled mean in `l:boundedother`. -/
theorem multiHeadScalarDrift_div_scale (c v : Idx H → ℝ) (y : Idx n → ℝ) (r s : ℝ) :
    multiHeadScalarDrift c v (fun l => y l / s) (s * r) = multiHeadScalarDrift c v y r / s := by
  simp only [multiHeadScalarDrift, scalarHeadAverage_div_scale, ← mul_div_assoc, ← Finset.sum_div]

/-- The original vector equation gives the exact scaled scalar drift at every actual layer.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for §12's actual discrete equation. -/
theorem multiHeadTransformer_rescaled_coord {Δt : ℝ} (hΔt : Δt ≠ 0)
    {Q K V : Idx H → ParamMatrix 1} {X : ℕ → Idx n → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (k : ℕ) (i : Idx n) (s : ℝ) :
    (X (k + 1) i 0 - X k i 0) / (Δt * s) =
      multiHeadScalarDrift (fun h => inner (𝕜 := ℝ) (Q h unit1) (K h unit1))
        (fun h => (V h unit1) 0) (fun j => X k j 0 / s) (s * X k i 0) := by
  rw [multiHeadScalarDrift_div_scale]
  by_cases hs : s = 0
  · subst s
    simp
  have he := multiHeadTransformer_coord hX k i
  field_simp [hΔt, hs]
  nlinarith

/-- Actual bounded original coordinates give an explicit bound for their scaled summed drift.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for §12's original separate heads. -/
theorem multiHeadTransformer_bounded_scaled_drift_le {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} {X : ℕ → Idx n → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (i : Idx n) (R : ℝ) (hR : ∀ k, |X k i 0| ≤ R)
    (k : ℕ) (s : ℝ) (hs : 0 < s) :
    |multiHeadScalarDrift (fun h => inner (𝕜 := ℝ) (Q h unit1) (K h unit1))
      (fun h => (V h unit1) 0) (fun j => X k j 0 / s) (s * X k i 0)| ≤ 2 * R / (Δt * s) := by
  have hi := abs_le.mp (hR k)
  have hn := abs_le.mp (hR (k + 1))
  have hd : |X (k + 1) i 0 - X k i 0| ≤ 2 * R := abs_le.mpr ⟨by linarith, by linarith⟩
  rw [← multiHeadTransformer_rescaled_coord hΔt.ne' hX k i s, abs_div, abs_of_pos (mul_pos hΔt hs)]
  exact div_le_div_of_nonneg_right hd (mul_pos hΔt hs).le

/-- The actual scaled drift of an original bounded token tends to zero at the full original layer scale.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for §12's original equation. -/
theorem multiHeadTransformer_bounded_scaled_drift_zero {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx n → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H)
    (i : Idx n) (hb : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R) :
    Tendsto (fun k => multiHeadScalarDrift (fun a => inner (𝕜 := ℝ) (Q a unit1) (K a unit1))
      (fun a => (V a unit1) 0) (fun j => X k j 0 / multiHeadLayerFactor Δt V ^ k)
      (multiHeadLayerFactor Δt V ^ k * X k i 0)) atTop (nhds 0) := by
  obtain ⟨R, hR⟩ := hb
  let q := multiHeadLayerFactor Δt V
  have hq : 1 < q := multiHeadLayerFactor_gt_one hΔt V hV h
  have hq0 : 0 < q := by linarith
  have he : ∀ k : ℕ, 2 * R / (Δt * q ^ k) = (2 * R / Δt) / q ^ k := by
    intro k
    rw [div_div]
  have hbound : ∀ᶠ k in atTop,
      ‖multiHeadScalarDrift (fun a => inner (𝕜 := ℝ) (Q a unit1) (K a unit1))
        (fun a => (V a unit1) 0) (fun j => X k j 0 / q ^ k) (q ^ k * X k i 0)‖ ≤
      (2 * R / Δt) / q ^ k := by
    apply Eventually.of_forall
    intro k
    rw [Real.norm_eq_abs]
    have hd := multiHeadTransformer_bounded_scaled_drift_le hΔt hX i R hR k (q ^ k) (pow_pos hq0 k)
    rwa [he] at hd
  have ht : Tendsto (fun k : ℕ => (2 * R / Δt) / q ^ k) atTop (nhds 0) :=
    tendsto_const_nhds.div_atTop (tendsto_pow_atTop_atTop_of_one_lt hq)
  exact squeeze_zero_norm' hbound ht

/-- Nonzero original anchors, a nonzero scale, and an actual positive query
jointly realize every static scaled-score premise. -/
example : ∃ (y : Idx 2 → ℝ) (r s : ℝ), y 0 ≠ y 1 ∧ 0 < r ∧ 0 < s ∧ s ≠ 0 := by
  exact ⟨![-1, 2], 1, 2, by norm_num, one_pos, by norm_num, by norm_num⟩

/-- Two genuine original heads and a bounded actual interior token jointly realize
all discrete coordinate, boundedness, step, and full-layer scaling premises. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 3 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ MultiHeadTransformer 1 V V V X ∧
    (∀ k, |X k 1 0| ≤ 0) ∧ (0 : ℝ) < 2 := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  have hV : ∀ h, IsPosDefOp (V h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (V h) (V h) := fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  obtain ⟨X, hX, hz, _, _, _⟩ := exists_multiHead_symmetric_escape one_pos hV hQK 0
  refine ⟨V, X, one_pos, hV, hX, ?_, by norm_num⟩
  intro k
  rw [hz k]
  norm_num

end Transformer.Clusters
