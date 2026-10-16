/-
# Collapse of the full original attention support

ArXiv:2411.04551v3, §2, `prop: targets.atoms`. The actual chart flow
contracts exponentially from every compactly supported open hemisphere.
Normalizing chart points recovers the original unit particles and
transfers the estimate to their spherical chord distance. Exact weak
transport transfers it to the whole evolving support.

The explicit exponential diameter bound tends to zero. The existing
radius property of W∞ then gives the original declaration's point-mass
approximation, for every actual weak law with the prescribed initial
measure and the original constant matrix parameters.
-/

import Transformer.Interpolation.Section2_AttentionContraction
import Transformer.Interpolation.Section2_DiracApproximation

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Recovering unit points from a common hemisphere chart is Lipschitz.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem dist_sphere_le_two_mul_hemiChart (w x z : SSphere d)
    (hx : 0 < inner ℝ (x : EucSpace d) (w : EucSpace d))
    (hz : 0 < inner ℝ (z : EucSpace d) (w : EucSpace d)) :
    dist x z ≤ 2 * ‖Perspective.hemiChart w (x : EucSpace d) - Perspective.hemiChart w (z : EucSpace d)‖ := by
  have hxn := Perspective.hemiChart_normalize (mem_sphere_zero_iff_norm.mp x.2) hx
  have hzn := Perspective.hemiChart_normalize (mem_sphere_zero_iff_norm.mp z.2) hz
  have hheight : inner ℝ (x : EucSpace d) (w : EucSpace d) ≤ 1 := by
    have hh := real_inner_le_norm (x : EucSpace d) (w : EucSpace d)
    simpa only [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp w.2, mul_one] using hh
  have hnorm : 1 ≤ ‖Perspective.hemiChart w (x : EucSpace d)‖ := by
    rw [hxn.1]
    exact (one_le_inv₀ hx).mpr hheight
  have hnormz : 0 < ‖Perspective.hemiChart w (z : EucSpace d)‖ := by rw [hzn.1]; positivity
  have hh := Perspective.norm_normalize_sub_le hnorm hnormz
  rw [hxn.2, hzn.2] at hh
  simpa only [Subtype.dist_eq, dist_eq_norm] using hh

/-- Distances of any two originally supported actual particles decay
exponentially. Auxiliary for arXiv:2411.04551v3, §2. -/
theorem dist_attentionFlow_of_support_le (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : SSphere d) (r : ℝ) (hr : 0 < r)
    (h0 : ∀ y ∈ (μ 0 : Measure (SSphere d)).support, r ≤ inner ℝ (w : EucSpace d) (y : EucSpace d))
    (x z : SSphere d)
    (hx : x ∈ (μ 0 : Measure (SSphere d)).support) (hz : z ∈ (μ 0 : Measure (SSphere d)).support)
    (t : ℝ) (ht : 0 ≤ t) : dist (attentionFlow B μ hμ x t) (attentionFlow B μ hμ z t) ≤
      (4 * r⁻¹) * exp (-(exp (-(2 * ‖B‖)) * r) * t) := by
  have hxp := hr.trans_le (le_height_attentionFlow_of_support B μ hμ w r hr h0 x hx t ht)
  have hzp := hr.trans_le (le_height_attentionFlow_of_support B μ hμ w r hr h0 z hz t ht)
  rw [real_inner_comm] at hxp hzp
  calc
    _ ≤ 2 * ‖Perspective.hemiChart w (attentionFlow B μ hμ x t : EucSpace d) -
        Perspective.hemiChart w (attentionFlow B μ hμ z t : EucSpace d)‖ :=
      dist_sphere_le_two_mul_hemiChart w _ _ hxp hzp
    _ ≤ 2 * ((2 * r⁻¹) * exp (-(exp (-(2 * ‖B‖)) * r) * t)) :=
      mul_le_mul_of_nonneg_left (norm_hemiChart_attentionFlow_sub_le B μ hμ w r hr h0 x z hx hz t ht) (by norm_num)
    _ = _ := by ring

/-- Explicit diameter bound for the full original weak support, without
atomicity assumptions. Auxiliary for arXiv:2411.04551v3, §2. -/
theorem diam_attention_support_le (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : SSphere d) (r : ℝ) (hr : 0 < r)
    (h0 : ∀ y ∈ (μ 0 : Measure (SSphere d)).support, r ≤ inner ℝ (w : EucSpace d) (y : EucSpace d))
    (t : ℝ) (ht : 0 ≤ t) : Metric.diam ((μ t : Measure (SSphere d)).support) ≤
      (4 * r⁻¹) * exp (-(exp (-(2 * ‖B‖)) * r) * t) := by
  apply Metric.diam_le_of_forall_dist_le (by positivity)
  intro x hx z hz
  obtain ⟨x₀, hx₀, rfl⟩ := attention_support_subset_flow_image B μ hμ t ht hx
  obtain ⟨z₀, hz₀, rfl⟩ := attention_support_subset_flow_image B μ hμ t ht hz
  exact dist_attentionFlow_of_support_le B μ hμ w r hr h0 x₀ z₀ hx₀ hz₀ t ht

/-- The full actual weak support has vanishing diameter in a supported
open hemisphere. Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem diam_attention_support_tendsto_zero (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w : SSphere d) (r : ℝ) (hr : 0 < r)
    (h0 : ∀ y ∈ (μ 0 : Measure (SSphere d)).support, r ≤ inner ℝ (w : EucSpace d) (y : EucSpace d)) :
    Tendsto (fun t : ℝ => Metric.diam ((μ t : Measure (SSphere d)).support)) atTop (𝓝 0) := by
  have hκ : 0 < exp (-(2 * ‖B‖)) * r := mul_pos (exp_pos _) hr
  have he : Tendsto (fun t : ℝ => exp (-(exp (-(2 * ‖B‖)) * r) * t)) atTop (𝓝 0) := by
    simpa only [Function.comp_def, id_eq, neg_mul] using
      tendsto_exp_neg_atTop_nhds_zero.comp ((tendsto_const_mul_atTop_of_pos hκ).mpr tendsto_id)
  have hbound : Tendsto (fun t : ℝ => (4 * r⁻¹) * exp (-(exp (-(2 * ‖B‖)) * r) * t)) atTop (𝓝 0) := by
    simpa only [mul_zero] using he.const_mul (4 * r⁻¹)
  apply squeeze_zero' (Eventually.of_forall fun _ => Metric.diam_nonneg) ?_ hbound
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
  exact diam_attention_support_le B μ hμ w r hr h0 t ht

/-- **Proposition (prop: targets.atoms).**  *Clustering to a single point.*

If `𝐁 ∈ M_{d×d}(ℝ)` and `supp μ_0` is contained in an open hemisphere, then
the solution to `eq: cauchy.pb`–`eq: vf` with `(𝐕, 𝐁, 𝐖) ≡ (I_d, 𝐁, 0)`
satisfies `diam(supp μ(t)) → 0` as `t → ∞`, and for any `ε > 0` there are
`z ∈ 𝕊^{d-1}` and `T > 0` with `W_∞(μ(T), δ_z) ≤ ε`.

Exact characteristic transport, preservation of a positive hemisphere
margin and compact chart consensus prove the diameter conclusion for
every original weak probability law. `exists_dirac_close_of_diam_tendsto`
then gives the W∞ conclusion.

As in the original declaration, this records support diameter collapse
and a spherical approximating point. The source additionally states
geodesic-convex-hull collapse, initial hull membership of the limiting
point, and a logarithmic W₂ hitting-time rate.

Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem clustering_to_atom (d : ℕ)
    (Winf : Measure (SSphere d) → Measure (SSphere d) → ℝ)
    (hWinf : IsWinfToDirac d Winf)
    (B : ParamMatrix d) (μ₀ : Perspective.ProbSphere d)
    (hhemi : ∃ w : SSphere d, ∀ x ∈ (μ₀ : Measure (SSphere d)).support,
      0 < inner (𝕜 := ℝ) (x : EucSpace d) ((w : EucSpace d))) :
    ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ →
      cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ →
        Filter.Tendsto
            (fun t : ℝ => Metric.diam ((μ t : Measure (SSphere d)).support))
            Filter.atTop (nhds 0) ∧
          ∀ ε : ℝ, 0 < ε → ∃ (z : SSphere d) (T : ℝ), 0 < T ∧
            Winf (μ T : Measure (SSphere d)) (Measure.dirac z) ≤ ε := by
  intro μ hμ0 hμ
  obtain ⟨w, hw⟩ := hhemi
  have hpos (y : SSphere d) (hy : y ∈ (μ 0 : Measure (SSphere d)).support) :
      0 < inner ℝ (w : EucSpace d) (y : EucSpace d) := by
    rw [hμ0] at hy
    rw [real_inner_comm]
    exact hw y hy
  obtain ⟨r, hr, h0⟩ := exists_pos_le_support_height (μ 0) w hpos
  have hdiam := diam_attention_support_tendsto_zero B μ hμ w r hr h0
  exact ⟨hdiam, exists_dirac_close_of_diam_tendsto d Winf hWinf μ hdiam⟩

/-- A genuine nonzero attention weak law, its unit pole, supported
particles, positive margin and constrained W∞ jointly witness every
diameter and clustering hypothesis, with the actual initial law. -/
example : ∃ (B : ParamMatrix 1) (μ : ℝ → Perspective.ProbSphere 1) (w x z : SSphere 1),
    IsWinfToDirac 1 (fun _ _ => 0) ∧
    cauchyPB 1 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 1), B, 0, 0, 0⟩) μ ∧
    (∀ y ∈ (μ 0 : Measure (SSphere 1)).support, (1 : ℝ) ≤ inner ℝ (w : EucSpace 1) (y : EucSpace 1)) ∧
    (∀ y ∈ (μ 0 : Measure (SSphere 1)).support, 0 < inner ℝ (y : EucSpace 1) (w : EucSpace 1)) ∧
    x ∈ (μ 0 : Measure (SSphere 1)).support ∧ z ∈ (μ 0 : Measure (SSphere 1)).support ∧
    (0 : ℝ) < inner ℝ (x : EucSpace 1) (w : EucSpace 1) ∧
    (0 : ℝ) < inner ℝ (z : EucSpace 1) (w : EucSpace 1) ∧
    μ 0 = μ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  let B := ContinuousLinearMap.id ℝ (EucSpace 1)
  let p : SSphere 1 := basePoint 0
  let μ : ℝ → Perspective.ProbSphere 1 := fun _ => Perspective.diracProb 1 p
  have hμ : cauchyPB 1 (fun _ => ⟨B, B, 0, 0, 0⟩) μ :=
    cauchyPB_const_diracProb 1 _ p (fun _ => rfl) (fun _ => rfl)
  have hh : inner ℝ (p : EucSpace 1) (p : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  have hheight (y : SSphere 1) (hy : y ∈ (μ 0 : Measure (SSphere 1)).support) :
      inner ℝ (y : EucSpace 1) (p : EucSpace 1) = 1 := by
    rw [eq_of_mem_support_dirac hy, hh]
  obtain ⟨x, hx⟩ := Measure.nonempty_support (IsProbabilityMeasure.ne_zero (μ 0 : Measure (SSphere 1)))
  refine ⟨B, μ, p, x, x, (fun _ _ _ hr _ => hr), hμ, ?_, ?_, hx, hx, ?_, ?_,
    rfl, zero_lt_one, zero_le_one⟩
  · intro y hy
    rw [real_inner_comm, hheight y hy]
  · intro y hy
    rw [hheight y hy]
    exact zero_lt_one
  · rw [hheight x hx]
    exact zero_lt_one
  · rw [hheight x hx]
    exact zero_lt_one

end Transformer.Interpolation
