/-
# The leading coordinate has only three possible cluster values

ArXiv:2305.05465v6, §9, the proof of `l:3hyperplanes11` and
`e:defab`. Write `a` and `b` for the limiting minimum and maximum
leading coordinates. The metric's competing gaps are exactly
`c₁₁ f(z_i) (f(z_j) - f(z_i))`, with `c₁₁ > 0`.

The extremal coordinates are attained by tokens at every time.
Thus the vanishing maximal gain bounds both the gap toward the
minimum and the gap toward the maximum. At a cluster value `p`,
these become `c₁₁ p (a-p) ≤ 0` and `c₁₁ p (b-p) ≤ 0`.
Since `a ≤ p ≤ b`, a negative value must be `a`, a positive
value must be `b`, and the only other possibility is zero.

This is a limit-transfer lemma. The actual dynamics supplies every
gain and extreme-coordinate limit in the following convergence module.
-/

import Transformer.Clusters.Section9_LeadingGain
import Transformer.Clusters.Section9_Limits

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d m : ℕ}

/-- The leading metric gap is the scalar leading attention gap.
Source: arXiv:2305.05465v6, §9, `c₁₁` in `e:majorizeweights` and `eq:phistarvar`. -/
theorem leading_inner_gap_eq (Q K A : ParamMatrix d) (φ : EucSpace d)
    (f : EucSpace d →L[ℝ] ℝ) (hform : IsAttentionFormOn Q K A (Submodule.span ℝ {φ}))
    (x y : EucSpace d) :
    inner (𝕜 := ℝ) (A ((f.smulRight φ) x))
      (A ((f.smulRight φ) y - (f.smulRight φ) x)) =
      inner (𝕜 := ℝ) (Q φ) (K φ) * f x * (f y - f x) := by
  have hmem (w : EucSpace d) : (f.smulRight φ) w ∈ Submodule.span ℝ {φ} :=
    Submodule.mem_span_singleton.mpr ⟨f w, rfl⟩
  rw [show (f.smulRight φ) y - (f.smulRight φ) x = (f.smulRight φ) (y - x) by rw [map_sub]]
  rw [← hform _ (hmem x) _ (hmem (y - x))]
  simp only [ContinuousLinearMap.smulRight_apply, map_smul, real_inner_smul_left,
    real_inner_smul_right, map_sub, inner_sub_right]
  ring

example (φ : EucSpace d) : IsAttentionFormOn (1 : ParamMatrix d) 1 1 (Submodule.span ℝ {φ}) :=
  isAttentionFormOn_of_isAttentionRoot 1 1 1 _ (isAttentionRoot_id d)

/-- The maximal leading gain bounds the two scalar gaps toward tokens
attaining the minimum and maximum coordinates.
Source: arXiv:2305.05465v6, §9, `e:defab` and the extreme token in `e:boundj0`. -/
theorem leading_attentionGain_ge_extreme_gaps (Q K A : ParamMatrix d) (φ : EucSpace d)
    (f : EucSpace d →L[ℝ] ℝ) (hform : IsAttentionFormOn Q K A (Submodule.span ℝ {φ}))
    (X : Idx (m + 1) → EucSpace d) (i : Idx (m + 1)) :
    inner (𝕜 := ℝ) (Q φ) (K φ) * f (X i) * (maxCoord f X - f (X i)) ≤
        attentionGain A (fun j => (f.smulRight φ) (X j)) i ∧
      inner (𝕜 := ℝ) (Q φ) (K φ) * f (X i) * (minCoord f X - f (X i)) ≤
        attentionGain A (fun j => (f.smulRight φ) (X j)) i := by
  obtain ⟨jmax, _, hmax⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty (fun j => f (X j))
  obtain ⟨jmin, _, hmin⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty (fun j => f (X j))
  change maxCoord f X = f (X jmax) at hmax
  change minCoord f X = f (X jmin) at hmin
  constructor
  · have h := inner_gap_le_attentionGain_of_mem_hull A
      (fun j => (f.smulRight φ) (X j)) i _ (mem_tokenHull _ jmax)
    simpa only [leading_inner_gap_eq Q K A φ f hform, hmax] using h
  · have h := inner_gap_le_attentionGain_of_mem_hull A
      (fun j => (f.smulRight φ) (X j)) i _ (mem_tokenHull _ jmin)
    simpa only [leading_inner_gap_eq Q K A φ f hform, hmin] using h

example (φ : EucSpace d) : IsAttentionFormOn (1 : ParamMatrix d) 1 1 (Submodule.span ℝ {φ}) :=
  isAttentionFormOn_of_isAttentionRoot 1 1 1 _ (isAttentionRoot_id d)

/-- The limiting minimum cannot exceed the limiting maximum.
Source: arXiv:2305.05465v6, §9, the interval `[a,b]` defined in `e:defab`. -/
theorem minCoord_limit_le_maxCoord_limit (f : EucSpace d →L[ℝ] ℝ)
    (Z : ℝ → Idx (m + 1) → EucSpace d) (a b : ℝ)
    (ha : Tendsto (fun t => minCoord f (Z t)) atTop (nhds a))
    (hb : Tendsto (fun t => maxCoord f (Z t)) atTop (nhds b)) : a ≤ b :=
  le_of_tendsto_of_tendsto ha hb
    (Eventually.of_forall fun t => minCoord_le_maxCoord f (Z t))

example (f : EucSpace d →L[ℝ] ℝ) (z : EucSpace d) :
    Tendsto (fun _ : ℝ => minCoord f (fun _ : Idx (m + 1) => z)) atTop (nhds (f z)) ∧
      Tendsto (fun _ : ℝ => maxCoord f (fun _ : Idx (m + 1) => z)) atTop (nhds (f z)) := by
  simp only [minCoord, maxCoord, Finset.inf'_const, Finset.sup'_const]
  exact ⟨tendsto_const_nhds, tendsto_const_nhds⟩

/-- **Every leading-coordinate cluster value is `0`, `a`, or `b`.**
Source: arXiv:2305.05465v6, §9, the proof of `l:3hyperplanes11`.
The extrema and gain limits are explicit inputs to this analytic lemma;
the following module derives them from the original good-triple ODE. -/
theorem eigenCoord_cluster_eq_zero_or_extreme (Q K A : ParamMatrix d) (φ : EucSpace d)
    (f : EucSpace d →L[ℝ] ℝ) (hform : IsAttentionFormOn Q K A (Submodule.span ℝ {φ}))
    (hpositive : 0 < inner (𝕜 := ℝ) (Q φ) (K φ))
    (Z : ℝ → Idx (m + 1) → EucSpace d) (a b : ℝ)
    (ha : Tendsto (fun t => minCoord f (Z t)) atTop (nhds a))
    (hb : Tendsto (fun t => maxCoord f (Z t)) atTop (nhds b)) (i : Idx (m + 1))
    (hgain : Tendsto (fun t => attentionGain A (fun j => (f.smulRight φ) (Z t j)) i)
      atTop (nhds 0)) (p : ℝ) (hp : MapClusterPt p atTop (fun t => f (Z t i))) :
    p = 0 ∨ p = a ∨ p = b := by
  obtain ⟨τ, hlim, htime⟩ := hp.exists_seq_tendsto
  change Tendsto (fun k => f (Z (τ k) i)) atTop (nhds p) at hlim
  let c := inner (𝕜 := ℝ) (Q φ) (K φ)
  have hmaxlim : Tendsto (fun k => c * f (Z (τ k) i) *
      (maxCoord f (Z (τ k)) - f (Z (τ k) i))) atTop (nhds (c * p * (b - p))) := by
    simpa only [Function.comp_def] using (hlim.const_mul c).mul ((hb.comp htime).sub hlim)
  have hminlim : Tendsto (fun k => c * f (Z (τ k) i) *
      (minCoord f (Z (τ k)) - f (Z (τ k) i))) atTop (nhds (c * p * (a - p))) := by
    simpa only [Function.comp_def] using (hlim.const_mul c).mul ((ha.comp htime).sub hlim)
  have hmax : c * p * (b - p) ≤ 0 := le_of_tendsto_of_tendsto hmaxlim (hgain.comp htime)
    (Eventually.of_forall fun k => (leading_attentionGain_ge_extreme_gaps Q K A φ f hform
      (Z (τ k)) i).1)
  have hmin : c * p * (a - p) ≤ 0 := le_of_tendsto_of_tendsto hminlim (hgain.comp htime)
    (Eventually.of_forall fun k => (leading_attentionGain_ge_extreme_gaps Q K A φ f hform
      (Z (τ k)) i).2)
  have hlow : a ≤ p := le_of_tendsto_of_tendsto (ha.comp htime) hlim
    (Eventually.of_forall fun k => minCoord_le f (Z (τ k)) i)
  have hup : p ≤ b := le_of_tendsto_of_tendsto hlim (hb.comp htime)
    (Eventually.of_forall fun k => le_maxCoord f (Z (τ k)) i)
  have hc : 0 < c := hpositive
  rcases lt_trichotomy p 0 with hnegative | hzero | hpos
  · have hpa : p ≤ a := by
      by_contra hnot
      have hsub : a - p < 0 := by linarith
      have hprod := mul_pos_of_neg_of_neg (mul_neg_of_pos_of_neg hc hnegative) hsub
      linarith
    exact Or.inr (Or.inl (le_antisymm hpa hlow))
  · exact Or.inl hzero
  · have hbp : b ≤ p := by
      by_contra hnot
      have hsub : 0 < b - p := by linarith
      have hprod := mul_pos (mul_pos hc hpos) hsub
      linarith
    exact Or.inr (Or.inr (le_antisymm hup hbp))

/-- Every limit-transfer input coexists for the actual stationary zero
trajectory, a positive coefficient, and its normalized leading coordinate.
Source: arXiv:2305.05465v6, §9, `e:defab` and `l:3hyperplanes11`. -/
example : IsAttentionFormOn (1 : ParamMatrix 1) 1 1
      (Submodule.span ℝ {(EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)}) ∧
    (0 < inner (𝕜 := ℝ) ((1 : ParamMatrix 1) (EuclideanSpace.single 0 (1 : ℝ)))
      ((1 : ParamMatrix 1) (EuclideanSpace.single 0 (1 : ℝ)))) ∧
    Tendsto (fun _ : ℝ => minCoord (EuclideanSpace.proj (𝕜 := ℝ) 0)
      (fun _ : Idx 1 => (0 : EucSpace 1))) atTop (nhds 0) ∧
    Tendsto (fun _ : ℝ => maxCoord (EuclideanSpace.proj (𝕜 := ℝ) 0)
      (fun _ : Idx 1 => (0 : EucSpace 1))) atTop (nhds 0) ∧
    Tendsto (fun _ : ℝ => attentionGain (1 : ParamMatrix 1)
      (fun _ : Idx 1 => ((EuclideanSpace.proj (𝕜 := ℝ) 0).smulRight
        (EuclideanSpace.single 0 (1 : ℝ))) (0 : EucSpace 1)) 0) atTop (nhds 0) ∧
    MapClusterPt (0 : ℝ) atTop (fun _ : ℝ => (EuclideanSpace.proj (𝕜 := ℝ) 0) (0 : EucSpace 1)) := by
  have hzero : Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds 0) := tendsto_const_nhds
  refine ⟨isAttentionFormOn_of_isAttentionRoot 1 1 1 _ (isAttentionRoot_id 1), by simp,
    ?_, ?_, ?_, ?_⟩
  · simpa only [minCoord, map_zero, Finset.inf'_const] using hzero
  · simpa only [maxCoord, map_zero, Finset.sup'_const] using hzero
  · simp [attentionGain]
  · simpa only [map_zero] using hzero.mapClusterPt

end Transformer.Clusters
