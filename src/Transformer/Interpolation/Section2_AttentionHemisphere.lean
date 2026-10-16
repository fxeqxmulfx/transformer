/-
# Preservation of the original supporting hemisphere

ArXiv:2411.04551v3, §2, `prop: targets.atoms`, and §1, `eq: vf`.
The exact characteristic representation carries the support into the
image of its initial compact support. At a particle of minimum height,
the current law is almost everywhere above that same height. The
original projected attention velocity then has nonnegative height
derivative whenever the height is nonnegative.

The compact positive-floor principle preserves every positive initial
height floor along all characteristics from the support, hence on the
entire evolving support. Compactness turns containment in an open
hemisphere into such a uniform positive floor. The associated hemisphere
chart consequently stays uniformly bounded.
-/

import Transformer.Interpolation.Section2_AttentionTransport
import Transformer.Interpolation.Section3_ProjectionTransport
import Transformer.Perspective.CompactPositiveFloor
import Transformer.Perspective.ConeChart

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original weak support lies in the characteristic image of its
initial support. Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem attention_support_subset_flow_image (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (t : ℝ) (ht : 0 ≤ t) : (μ t : Measure (SSphere d)).support ⊆
      (fun x => attentionFlow B μ hμ x t) '' (μ 0 : Measure (SSphere d)).support := by
  rw [cauchyPB_attention_transport_nonneg B μ hμ t ht]
  exact support_map_subset_image _ _ (lipschitzWith_attentionFlow_initial B μ hμ t).continuous

/-- Every particle from the initial support preserves its positive
hemisphere floor. Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem le_height_attentionFlow_of_support (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : EucSpace d) (r₀ : ℝ) (hr₀ : 0 < r₀)
    (h0 : ∀ x ∈ (μ 0 : Measure (SSphere d)).support, r₀ ≤ inner ℝ w (x : EucSpace d))
    (x : SSphere d) (hx : x ∈ (μ 0 : Measure (SSphere d)).support) (t : ℝ) (ht : 0 ≤ t) :
    r₀ ≤ inner ℝ w (attentionFlow B μ hμ x t : EucSpace d) := by
  let S := (μ 0 : Measure (SSphere d)).support
  let : CompactSpace S := (Measure.isClosed_support (μ := (μ 0 : Measure (SSphere d)))).isClosedEmbedding_subtypeVal.compactSpace
  let : Nonempty S := ⟨⟨x, hx⟩⟩
  let f (i : S) (s : ℝ) := inner ℝ w (attentionFlow B μ hμ i s : EucSpace d)
  let f' (i : S) (s : ℝ) := inner ℝ w
    (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩)
      (μ s) s (attentionFlow B μ hμ i s))
  have hp : Continuous (fun q : S × ℝ => (q.2, (q.1 : SSphere d))) :=
    continuous_snd.prodMk (continuous_subtype_val.comp continuous_fst)
  have hX : Continuous (fun q : S × ℝ => (attentionFlow B μ hμ q.1 q.2 : EucSpace d)) :=
    continuous_subtype_val.comp ((continuous_attentionFlow B μ hμ).comp hp)
  have hc : Continuous (fun q : S × ℝ => f q.1 q.2) := continuous_const.inner hX
  have hd (i : S) (s : ℝ) : HasDerivAt (f i) (f' i s) s := by
    simpa only [inner_zero_left, add_zero] using
      (hasDerivAt_const s w).inner ℝ (hasDerivAt_attentionFlow B μ hμ i s)
  have hsign (s : ℝ) (hs : 0 ≤ s) (i : S) (hi : ∀ j : S, f i s ≤ f j s) (hn : 0 ≤ f i s) :
      0 ≤ f' i s := by
    apply inner_attention_height_nonneg B (μ s) (attentionFlow B μ hμ i s) w (f i s) s hn rfl
    filter_upwards [Measure.support_mem_ae (μ := (μ s : Measure (SSphere d)))] with y hy
    obtain ⟨z, hz, rfl⟩ := attention_support_subset_flow_image B μ hμ s hs hy
    exact hi ⟨z, hz⟩
  have hh := Perspective.compact_positive_floor_of_deriv_nonneg f f' hc (fun i s _ => hd i s)
    hsign r₀ hr₀ (fun i => by dsimp only [f]; rw [attentionFlow_zero]; exact h0 i i.2) t ht ⟨x, hx⟩
  exact hh

/-- The whole actual weak support preserves the positive hemisphere
floor. Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem le_height_attention_support (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : EucSpace d) (r₀ : ℝ) (hr₀ : 0 < r₀)
    (h0 : ∀ x ∈ (μ 0 : Measure (SSphere d)).support, r₀ ≤ inner ℝ w (x : EucSpace d))
    (t : ℝ) (ht : 0 ≤ t) (x : SSphere d) (hx : x ∈ (μ t : Measure (SSphere d)).support) :
    r₀ ≤ inner ℝ w (x : EucSpace d) := by
  obtain ⟨z, hz, rfl⟩ := attention_support_subset_flow_image B μ hμ t ht hx
  exact le_height_attentionFlow_of_support B μ hμ w r₀ hr₀ h0 z hz t ht

/-- Compactness supplies a uniform positive margin inside an open
hemisphere. Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem exists_pos_le_support_height (ν : Perspective.ProbSphere d) (w : EucSpace d)
    (hpos : ∀ x ∈ (ν : Measure (SSphere d)).support, 0 < inner ℝ w (x : EucSpace d)) :
    ∃ r : ℝ, 0 < r ∧ ∀ x ∈ (ν : Measure (SSphere d)).support, r ≤ inner ℝ w (x : EucSpace d) := by
  have hc : Continuous (fun x : SSphere d => inner ℝ w (x : EucSpace d)) :=
    continuous_const.inner continuous_subtype_val
  obtain ⟨x, hx, hmin⟩ := (Measure.isClosed_support (μ := (ν : Measure (SSphere d)))).isCompact.exists_isMinOn
    (Measure.nonempty_support (IsProbabilityMeasure.ne_zero (ν : Measure (SSphere d)))) hc.continuousOn
  exact ⟨inner ℝ w (x : EucSpace d), hpos x hx, fun y hy => hmin hy⟩

/-- The chart of every supported original characteristic has norm at
most the reciprocal initial margin. Auxiliary for arXiv:2411.04551v3,
§2, `prop: targets.atoms`, the chart diameter estimate. -/
theorem norm_hemiChart_attentionFlow_le (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : EucSpace d) (r₀ : ℝ) (hr₀ : 0 < r₀)
    (h0 : ∀ x ∈ (μ 0 : Measure (SSphere d)).support, r₀ ≤ inner ℝ w (x : EucSpace d))
    (x : SSphere d) (hx : x ∈ (μ 0 : Measure (SSphere d)).support) (t : ℝ) (ht : 0 ≤ t) :
    ‖Perspective.hemiChart w (attentionFlow B μ hμ x t : EucSpace d)‖ ≤ r₀⁻¹ := by
  have hh := le_height_attentionFlow_of_support B μ hμ w r₀ hr₀ h0 x hx t ht
  have hh' : r₀ ≤ inner ℝ (attentionFlow B μ hμ x t : EucSpace d) w := by
    rwa [real_inner_comm] at hh
  rw [(Perspective.hemiChart_normalize (mem_sphere_zero_iff_norm.mp (attentionFlow B μ hμ x t).2)
    (hr₀.trans_le hh')).1]
  exact (inv_le_inv₀ (hr₀.trans_le hh') hr₀).mpr hh'

/-- An initially supported open hemisphere has one uniform positive
margin for all future supported particles and all future weak supports.
Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem exists_uniform_hemisphere_attention_support (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : SSphere d)
    (hpos : ∀ x ∈ (μ 0 : Measure (SSphere d)).support, 0 < inner ℝ (w : EucSpace d) (x : EucSpace d)) :
    ∃ r : ℝ, 0 < r ∧
      (∀ t : ℝ, 0 ≤ t → ∀ x ∈ (μ t : Measure (SSphere d)).support,
        r ≤ inner ℝ (w : EucSpace d) (x : EucSpace d)) ∧
      (∀ x ∈ (μ 0 : Measure (SSphere d)).support, ∀ t : ℝ, 0 ≤ t →
        r ≤ inner ℝ (w : EucSpace d) (attentionFlow B μ hμ x t : EucSpace d)) := by
  obtain ⟨r, hr, h0⟩ := exists_pos_le_support_height (μ 0) w hpos
  exact ⟨r, hr, (fun t ht x hx => le_height_attention_support B μ hμ w r hr h0 t ht x hx),
    fun x hx t ht => le_height_attentionFlow_of_support B μ hμ w r hr h0 x hx t ht⟩

/-- A nonzero attention matrix and its genuine Dirac weak law jointly
realize the weak equation, a positive uniform support margin, a supported
initial particle and all nonnegative-time hypotheses. -/
example : ∃ (B : ParamMatrix 2) (μ : ℝ → Perspective.ProbSphere 2)
    (w x : SSphere 2),
    cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) μ ∧
    (∀ y ∈ (μ 0 : Measure (SSphere 2)).support, (1 : ℝ) ≤ inner ℝ (w : EucSpace 2) (y : EucSpace 2)) ∧
    (∀ y ∈ (μ 0 : Measure (SSphere 2)).support, 0 < inner ℝ (w : EucSpace 2) (y : EucSpace 2)) ∧
    x ∈ (μ 0 : Measure (SSphere 2)).support ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  let B := ContinuousLinearMap.id ℝ (EucSpace 2)
  let p : SSphere 2 := basePoint 1
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  have hθ : pureAttentionParams 2 1 = (fun _ => ⟨B, B, 0, 0, 0⟩) := by
    funext t
    simp only [pureAttentionParams, B, one_smul]
  have hμ : cauchyPB 2 (fun _ => ⟨B, B, 0, 0, 0⟩) μ := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  have hheight (y : SSphere 2) (hy : y ∈ (μ 0 : Measure (SSphere 2)).support) :
      inner ℝ (p : EucSpace 2) (y : EucSpace 2) = 1 := by
    have he : y = p := eq_of_mem_support_dirac hy
    rw [he, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  obtain ⟨x, hx⟩ := Measure.nonempty_support (IsProbabilityMeasure.ne_zero (μ 0 : Measure (SSphere 2)))
  exact ⟨B, μ, p, x, hμ, (fun y hy => (hheight y hy).ge),
    (fun y hy => by rw [hheight y hy]; exact zero_lt_one), hx, zero_lt_one, zero_le_one⟩

end Transformer.Interpolation
