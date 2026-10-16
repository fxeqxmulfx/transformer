/-
# Contraction of the full original attention support in a hemisphere chart

ArXiv:2411.04551v3, §2, `prop: targets.atoms`, and §1, `eq: vf`.
Index the actual characteristics by the entire compact initial support.
At each maximal chart-coordinate pair the current probability is
almost everywhere between those two coordinates. The exact consensus
equation and the common kernel floor therefore contract its gap.

The compact-family width theorem gives exponential decay in every
direction. Choosing the direction to be the current chart difference
turns the scalar estimate into a vector distance estimate. All bounds
hold for every original weak solution and arbitrary initial laws; no
finite-particle approximation or atomic mass lower bound is assumed.
-/

import Transformer.Interpolation.Section2_AttentionChartEquation
import Transformer.Perspective.CompactWidth
import Transformer.Perspective.ConeLimit

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every directional chart width of the full actual attention support
decays exponentially. Auxiliary for arXiv:2411.04551v3, §2,
`prop: targets.atoms`, using the original arbitrary weak law. -/
theorem inner_hemiChart_attentionFlow_sub_le (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : SSphere d) (r : ℝ) (hr : 0 < r)
    (h0 : ∀ y ∈ (μ 0 : Measure (SSphere d)).support, r ≤ inner ℝ (w : EucSpace d) (y : EucSpace d))
    (u : EucSpace d) (x z : SSphere d)
    (hx : x ∈ (μ 0 : Measure (SSphere d)).support) (hz : z ∈ (μ 0 : Measure (SSphere d)).support)
    (t : ℝ) (ht : 0 ≤ t) :
    inner ℝ (Perspective.hemiChart w (attentionFlow B μ hμ x t : EucSpace d) -
      Perspective.hemiChart w (attentionFlow B μ hμ z t : EucSpace d)) u ≤
        (2 * r⁻¹ * ‖u‖) * exp (-(exp (-(2 * ‖B‖)) * r) * t) := by
  let S := (μ 0 : Measure (SSphere d)).support
  let : CompactSpace S := (Measure.isClosed_support (μ := (μ 0 : Measure (SSphere d)))).isClosedEmbedding_subtypeVal.compactSpace
  let g (i : S) (s : ℝ) := inner ℝ (Perspective.hemiChart w (attentionFlow B μ hμ i s : EucSpace d)) u
  let g' (i : S) (s : ℝ) := ∫ y : SSphere d,
    attentionChartKernel B (μ s) w (attentionFlow B μ hμ i s) y *
      (inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u - g i s) ∂(μ s : Measure (SSphere d))
  have hheight (i : S) (s : ℝ) (hs : 0 ≤ s) :
      r ≤ inner ℝ (attentionFlow B μ hμ i s : EucSpace d) (w : EucSpace d) := by
    rw [real_inner_comm]
    exact le_height_attentionFlow_of_support B μ hμ w r hr h0 i i.2 s hs
  have hν (s : ℝ) (hs : 0 ≤ s) : ∀ᵐ y : SSphere d ∂(μ s : Measure (SSphere d)),
      r ≤ inner ℝ (y : EucSpace d) (w : EucSpace d) := by
    filter_upwards [Measure.support_mem_ae (μ := (μ s : Measure (SSphere d)))] with y hy
    rw [real_inner_comm]
    exact le_height_attention_support B μ hμ w r hr h0 s hs y hy
  have hp : Continuous (fun q : S × ℝ => (q.2, (q.1 : SSphere d))) :=
    continuous_snd.prodMk (continuous_subtype_val.comp continuous_fst)
  have hX : Continuous (fun q : S × ℝ => (attentionFlow B μ hμ q.1 q.2 : EucSpace d)) :=
    continuous_subtype_val.comp ((continuous_attentionFlow B μ hμ).comp hp)
  have hc : ContinuousOn (fun q : S × ℝ => g q.1 q.2) (univ ×ˢ Ici 0) := by
    have hi : ContinuousOn (fun q : S × ℝ =>
        (inner ℝ (attentionFlow B μ hμ q.1 q.2 : EucSpace d) (w : EucSpace d))⁻¹) (univ ×ˢ Ici 0) :=
      (hX.inner continuous_const).continuousOn.inv₀
        (fun q hq => (hr.trans_le (hheight q.1 q.2 hq.2)).ne')
    exact (hi.smul hX.continuousOn).inner continuousOn_const
  have hd (i : S) (s : ℝ) (hs : 0 ≤ s) : HasDerivAt (g i) (g' i s) s :=
    hasDerivAt_inner_hemiChart_attentionFlow_of_support B μ hμ w u r hr h0 i i.2 s hs
  have hcontract (s : ℝ) (hs : 0 ≤ s) (i k : S)
      (hmax : ∀ j : S, g j s ≤ g i s) (hmin : ∀ j : S, g k s ≤ g j s) :
      g' i s - g' k s ≤ -(exp (-(2 * ‖B‖)) * r) * (g i s - g k s) := by
    have hνne : ∀ᵐ y : SSphere d ∂(μ s : Measure (SSphere d)),
        inner ℝ (y : EucSpace d) (w : EucSpace d) ≠ 0 :=
      (hν s hs).mono fun y hy => (hr.trans_le hy).ne'
    have hkernel (j : S) : ∀ᵐ y : SSphere d ∂(μ s : Measure (SSphere d)),
        exp (-(2 * ‖B‖)) * r ≤ attentionChartKernel B (μ s) w (attentionFlow B μ hμ j s) y :=
      (hν s hs).mono fun y hy => attentionChartKernel_lower B (μ s) w _ y r hr
        (hr.trans_le (hheight j s hs)) hy
    have htop : ∀ᵐ y : SSphere d ∂(μ s : Measure (SSphere d)),
        inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u ≤ g i s := by
      filter_upwards [Measure.support_mem_ae (μ := (μ s : Measure (SSphere d)))] with y hy
      obtain ⟨q, hq, rfl⟩ := attention_support_subset_flow_image B μ hμ s hs hy
      exact hmax ⟨q, hq⟩
    have hbot : ∀ᵐ y : SSphere d ∂(μ s : Measure (SSphere d)),
        g k s ≤ inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u := by
      filter_upwards [Measure.support_mem_ae (μ := (μ s : Measure (SSphere d)))] with y hy
      obtain ⟨q, hq, rfl⟩ := attention_support_subset_flow_image B μ hμ s hs hy
      exact hmin ⟨q, hq⟩
    exact Perspective.consensus_extreme_deriv_sub_le (μ s : Measure (SSphere d))
      (attentionChartKernel B (μ s) w (attentionFlow B μ hμ i s))
      (attentionChartKernel B (μ s) w (attentionFlow B μ hμ k s))
      (fun y : SSphere d => inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u)
      (exp (-(2 * ‖B‖)) * r) (g i s) (g k s) (hkernel i) (hkernel k) htop hbot
      (integrable_inner_hemiChart_of_ae_height (μ s) w u r hr (hν s hs))
      (integrable_attentionChartKernel_gap B (μ s) w u _ (hr.trans_le (hheight i s hs)).ne' hνne)
      (integrable_attentionChartKernel_gap B (μ s) w u _ (hr.trans_le (hheight k s hs)).ne' hνne)
  have hD (i k : S) : g i 0 - g k 0 ≤ 2 * r⁻¹ * ‖u‖ := by
    have hi := norm_hemiChart_attentionFlow_le B μ hμ w r hr h0 i i.2 0 le_rfl
    have hk := norm_hemiChart_attentionFlow_le B μ hμ w r hr h0 k k.2 0 le_rfl
    change inner ℝ (Perspective.hemiChart w (attentionFlow B μ hμ i 0 : EucSpace d)) u -
      inner ℝ (Perspective.hemiChart w (attentionFlow B μ hμ k 0 : EucSpace d)) u ≤ _
    rw [← inner_sub_left]
    calc
      _ ≤ ‖Perspective.hemiChart w (attentionFlow B μ hμ i 0 : EucSpace d) -
          Perspective.hemiChart w (attentionFlow B μ hμ k 0 : EucSpace d)‖ * ‖u‖ := real_inner_le_norm _ _
      _ ≤ (r⁻¹ + r⁻¹) * ‖u‖ := mul_le_mul_of_nonneg_right
        ((norm_sub_le _ _).trans (add_le_add hi hk)) (norm_nonneg u)
      _ = _ := by ring
  have hh := Perspective.compact_width_decay_of_deriv_bounds g g' (exp (-(2 * ‖B‖)) * r)
    (2 * r⁻¹ * ‖u‖) hc hd hcontract hD t ht (⟨x, hx⟩ : S) (⟨z, hz⟩ : S)
  rwa [inner_sub_left]

/-- Chart distances between any two originally supported particles
decay at the same rate. Auxiliary for arXiv:2411.04551v3, §2,
`prop: targets.atoms`, without an atomicity restriction. -/
theorem norm_hemiChart_attentionFlow_sub_le (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : SSphere d) (r : ℝ) (hr : 0 < r)
    (h0 : ∀ y ∈ (μ 0 : Measure (SSphere d)).support, r ≤ inner ℝ (w : EucSpace d) (y : EucSpace d))
    (x z : SSphere d)
    (hx : x ∈ (μ 0 : Measure (SSphere d)).support) (hz : z ∈ (μ 0 : Measure (SSphere d)).support)
    (t : ℝ) (ht : 0 ≤ t) :
    ‖Perspective.hemiChart w (attentionFlow B μ hμ x t : EucSpace d) -
      Perspective.hemiChart w (attentionFlow B μ hμ z t : EucSpace d)‖ ≤
        (2 * r⁻¹) * exp (-(exp (-(2 * ‖B‖)) * r) * t) := by
  let v := Perspective.hemiChart w (attentionFlow B μ hμ x t : EucSpace d) -
    Perspective.hemiChart w (attentionFlow B μ hμ z t : EucSpace d)
  have hh := inner_hemiChart_attentionFlow_sub_le B μ hμ w r hr h0 v x z hx hz t ht
  change inner ℝ v v ≤ _ at hh
  rw [real_inner_self_eq_norm_mul_norm] at hh
  rcases (norm_nonneg v).eq_or_lt with he | he
  · rw [← he]
    positivity
  · nlinarith

/-- A genuine nonzero attention weak law, supported initial particles,
a unit pole and a positive uniform initial margin jointly witness all
contraction hypotheses at a positive time. The directional estimate
allows every ambient direction. -/
example : ∃ (B : ParamMatrix 2) (μ : ℝ → Perspective.ProbSphere 2) (w x z : SSphere 2),
    cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) μ ∧
    (∀ y ∈ (μ 0 : Measure (SSphere 2)).support, (1 : ℝ) ≤ inner ℝ (w : EucSpace 2) (y : EucSpace 2)) ∧
    x ∈ (μ 0 : Measure (SSphere 2)).support ∧ z ∈ (μ 0 : Measure (SSphere 2)).support ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  let B := ContinuousLinearMap.id ℝ (EucSpace 2)
  let p : SSphere 2 := basePoint 1
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  have hθ : pureAttentionParams 2 1 = (fun _ => ⟨B, B, 0, 0, 0⟩) := by
    funext t
    simp only [pureAttentionParams, B, one_smul]
  have hμ : cauchyPB 2 (fun _ => ⟨B, B, 0, 0, 0⟩) μ := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  have hh : inner ℝ (p : EucSpace 2) (p : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  obtain ⟨x, hx⟩ := Measure.nonempty_support (IsProbabilityMeasure.ne_zero (μ 0 : Measure (SSphere 2)))
  refine ⟨B, μ, p, x, x, hμ, ?_, hx, hx, zero_lt_one, zero_le_one⟩
  intro y hy
  rw [eq_of_mem_support_dirac hy, hh]

end Transformer.Interpolation
