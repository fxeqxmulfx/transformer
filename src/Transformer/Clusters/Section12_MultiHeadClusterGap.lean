module

public import Transformer.Clusters.Section12_MultiHeadFullGrowth

/-!
# Bounded gaps inside the original escaping multi-head clusters

ArXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, for
§12's original head sum and §3's discrete layer convention.
For a positive query, the proved original reference pull differs from
the largest-token drift by at most n Σ(v_h/c_h)/x_i. The actual gap
to that largest token therefore increases by at most this quantity
times the original step size. Full-factor positive escape makes the
accumulated error a bounded geometric sum.

Consequently, the original unscaled gap stays bounded after a finite
layer and its quotient by the full original layer factor tends to
zero. Every positive escaping token has the normalized maximum's
actual limit. Sign reversal gives the normalized minimum's limit
for every negative escaping token. No common head temperature or
replacement iteration is introduced.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ}

/-- The actual gap from a positive original query to an original maximum has an inverse-query step error.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, for §12's exact head sum. -/
theorem multiHeadTransformer_positive_gap_step {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (k : ℕ) (i M : Idx (m + 1))
    (hi : 0 < X k i 0) (hM : ∀ j, X k j 0 ≤ X k M 0) :
    X (k + 1) M 0 - X (k + 1) i 0 ≤ X k M 0 - X k i 0 +
      Δt * (multiHeadPullCoefficient (m + 1) Q K V / X k i 0) := by
  have hm := multiHeadTransformer_max_step hΔt hV hX k M hM
  have hi' := multiHeadTransformer_reference_pull hΔt hV hQK hX k i M hi
  rw [multiHeadLayerFactor] at hm
  nlinarith

/-- The actual unscaled gap from a positively escaping token to a fixed original maximum stays bounded.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, for §12's original equation. -/
theorem multiHeadTransformer_positive_gap_bounded {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i M : Idx (m + 1))
    (hi : Tendsto (fun k => X k i 0) atTop atTop) (hM : ∀ k j, X k j 0 ≤ X k M 0) :
    ∃ (k₀ : ℕ) (C : ℝ), 0 ≤ C ∧ ∀ l, 0 ≤ X (k₀ + l) M 0 - X (k₀ + l) i 0 ∧
      X (k₀ + l) M 0 - X (k₀ + l) i 0 ≤ C := by
  obtain ⟨k₀, c, hc, hg⟩ := multiHeadTransformer_positive_full_growth hΔt hV hQK hX h i hi
  let q := multiHeadLayerFactor Δt V
  have hq : 1 < q := multiHeadLayerFactor_gt_one hΔt V hV h
  have hq0 : 0 < q := by linarith
  let B := multiHeadPullCoefficient (m + 1) Q K V
  have hB : 0 ≤ B := (multiHead_coefficients_pos (m + 1) Q K V hV hQK h).2
  let r := 1 / q
  have hr : 0 ≤ r := div_nonneg zero_le_one hq0.le
  have hr1 : r < 1 := (div_lt_iff₀ hq0).mpr (by linarith)
  let C₀ := Δt * (B / c)
  have hC₀ : 0 ≤ C₀ := mul_nonneg hΔt.le (div_nonneg hB hc.le)
  let f := fun l => X (k₀ + l) M 0 - X (k₀ + l) i 0
  have hstep : ∀ l, f (l + 1) ≤ f l + C₀ * r ^ l := by
    intro l
    have hden : 0 < c * q ^ l := mul_pos hc (pow_pos hq0 l)
    have hip : 0 < X (k₀ + l) i 0 := hden.trans_le (hg l)
    have hs := multiHeadTransformer_positive_gap_step hΔt.le hV hQK hX (k₀ + l) i M hip (hM (k₀ + l))
    have hd := div_le_div_of_nonneg_left hB hden (hg l)
    have hm := mul_le_mul_of_nonneg_left hd hΔt.le
    have he : Δt * (B / (c * q ^ l)) = C₀ * r ^ l := by
      dsimp [C₀, r]
      simp only [div_eq_mul_inv, mul_inv, ← inv_pow]
      ring
    rw [he] at hm
    change f (l + 1) ≤ f l + Δt * (B / X (k₀ + l) i 0) at hs
    linarith
  let C := f 0 + C₀ / (1 - r)
  have hC : 0 ≤ C := by
    have hf : 0 ≤ f 0 := sub_nonneg.mpr (hM k₀ i)
    have hd := div_nonneg hC₀ (show 0 ≤ 1 - r by linarith)
    dsimp [C]
    linarith
  have hb := discrete_geometric_error_upper f C₀ r hC₀ hr hr1 hstep
  exact ⟨k₀, C, hC, fun l => ⟨sub_nonneg.mpr (hM (k₀ + l) i), hb l⟩⟩

/-- The gap inside an actual positive escaping cluster vanishes under the full original rescaling.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, for §12's original heads. -/
theorem multiHeadTransformer_normalized_positive_gap_zero {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i M : Idx (m + 1))
    (hi : Tendsto (fun k => X k i 0) atTop atTop) (hM : ∀ k j, X k j 0 ≤ X k M 0) :
    Tendsto (fun k => (X k M 0 - X k i 0) / multiHeadLayerFactor Δt V ^ k) atTop (nhds 0) := by
  obtain ⟨k₀, C, _, hg⟩ := multiHeadTransformer_positive_gap_bounded hΔt hV hQK hX h i M hi hM
  let q := multiHeadLayerFactor Δt V
  have hq : 1 < q := multiHeadLayerFactor_gt_one hΔt V hV h
  have hq0 : 0 < q := by linarith
  have hb : ∀ᶠ k in atTop, ‖(X k M 0 - X k i 0) / q ^ k‖ ≤ C / q ^ k := by
    apply eventually_atTop.mpr
    refine ⟨k₀, ?_⟩
    intro k hk
    have he : k₀ + (k - k₀) = k := by omega
    have hs := hg (k - k₀)
    rw [he] at hs
    have hp : 0 < q ^ k := pow_pos hq0 k
    rw [Real.norm_eq_abs, abs_of_nonneg (div_nonneg hs.1 hp.le)]
    exact div_le_div_of_nonneg_right hs.2 hp.le
  have ht : Tendsto (fun k : ℕ => C / q ^ k) atTop (nhds 0) :=
    tendsto_const_nhds.div_atTop (tendsto_pow_atTop_atTop_of_one_lt hq)
  exact squeeze_zero_norm' hb ht

/-- Every original positively escaping token has the actual normalized maximum's limit.
Source: arXiv:2305.05465v6, §7, `l:exactasymptotic`, for §12's original separate heads. -/
theorem multiHeadTransformer_positive_rate_eq_max {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i M : Idx (m + 1))
    (hi : Tendsto (fun k => X k i 0) atTop atTop) (hM : ∀ k j, X k j 0 ≤ X k M 0) {γ : ℝ}
    (hlim : Tendsto (fun k => X k M 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γ)) :
    Tendsto (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γ) := by
  have hd := multiHeadTransformer_normalized_positive_gap_zero hΔt hV hQK hX h i M hi hM
  have hs := hlim.sub hd
  have he : (fun k => X k M 0 / multiHeadLayerFactor Δt V ^ k -
      (X k M 0 - X k i 0) / multiHeadLayerFactor Δt V ^ k) =
      (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) := by funext k; ring
  rw [he, sub_zero] at hs
  exact hs

/-- Every original negatively escaping token has the actual normalized minimum's limit.
Source: arXiv:2305.05465v6, §7, `l:exactasymptotic`, by the original sign symmetry for §12. -/
theorem multiHeadTransformer_negative_rate_eq_min {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i L : Idx (m + 1))
    (hi : Tendsto (fun k => X k i 0) atTop atBot) (hL : ∀ k j, X k L 0 ≤ X k j 0) {γ : ℝ}
    (hlim : Tendsto (fun k => X k L 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γ)) :
    Tendsto (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γ) := by
  have hn : ∀ k j, (-X k j) 0 ≤ (-X k L) 0 := by
    intro k j
    change -(X k j 0) ≤ -(X k L 0)
    linarith [hL k j]
  have hl : Tendsto (fun k => (-X k L) 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (-γ)) := by
    simpa only [PiLp.neg_apply, neg_div] using hlim.neg
  have ht := multiHeadTransformer_positive_rate_eq_max hΔt hV hQK (multiHeadTransformer_neg hX)
    h i L (tendsto_neg_atTop_iff.mpr hi) hn hl
  simpa only [PiLp.neg_apply, neg_div, neg_neg] using ht.neg

/-- Two genuine original positive heads have distinct positive initial tokens
which both escape, with the second remaining the actual maximum. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 2 → EucSpace 1) (γ : ℝ),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (V h) (V h)) ∧
    MultiHeadTransformer 1 V V V X ∧ MultiHeadTransformer 1 V V V (fun k i => -X k i) ∧
    X 0 0 0 < X 0 1 0 ∧ (∀ i, Tendsto (fun k => X k i 0) atTop atTop) ∧
    (∀ i, Tendsto (fun k => (-X k i) 0) atTop atBot) ∧
    (∀ k j, X k 0 0 ≤ X k j 0 ∧ X k j 0 ≤ X k 1 0) ∧
    Tendsto (fun k => X k 1 0 / multiHeadLayerFactor 1 V ^ k) atTop (nhds γ) ∧
    Tendsto (fun k => (-X k 1) 0 / multiHeadLayerFactor 1 V ^ k) atTop (nhds (-γ)) := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  let Y : Idx 2 → EucSpace 1 := ![(5 : ℝ) • unit1, (6 : ℝ) • unit1]
  let X := multiHeadTrajectory 1 V V V Y
  have hV : ∀ h, IsPosDefOp (V h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (V h) (V h) := fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  have hX : MultiHeadTransformer 1 V V V X := multiHeadTrajectory_solution _ _ _ _ _
  have hstep : ∀ i k, (5 : ℝ) ≤ X k i 0 → (2 : ℝ) * X k i 0 ≤ X (k + 1) i 0 := by
    intro i k hk
    have hp : 0 < X k i 0 := by linarith
    have hs := multiHeadTransformer_self_pull zero_le_one hV hQK hX k i hp
    norm_num [multiHeadValueCoefficient, multiHeadPullCoefficient, V, unit1] at hs
    have hd : (4 : ℝ) / X k i 0 ≤ 1 := (div_le_iff₀ hp).mpr (by linarith)
    nlinarith
  have hini : ∀ i, (5 : ℝ) ≤ X 0 i 0 := by
    intro i
    change (5 : ℝ) ≤ Y i 0
    fin_cases i <;> norm_num [Y, unit1]
  have hp : ∀ i, Tendsto (fun k => X k i 0) atTop atTop := by
    intro i
    exact discrete_tendsto_atTop_of_threshold (fun k => X k i 0) (by norm_num : (1 : ℝ) < 2)
      (by norm_num : (0 : ℝ) < 5) (hstep i) 0 (hini i)
  have hf : ∀ k j, X k 0 0 ≤ X k j 0 ∧ X k j 0 ≤ X k 1 0 := by
    intro k j
    have hl0 : X 0 0 0 ≤ X 0 j 0 := by change Y 0 0 ≤ Y j 0; fin_cases j <;> norm_num [Y, unit1]
    have hu0 : X 0 j 0 ≤ X 0 1 0 := by change Y j 0 ≤ Y 1 0; fin_cases j <;> norm_num [Y, unit1]
    have hl := multiHeadTransformer_initial_gap zero_le_one hV hQK hX 0 j hl0 k
    have hu := multiHeadTransformer_initial_gap zero_le_one hV hQK hX j 1 hu0 k
    constructor <;> linarith
  obtain ⟨_, γ, _, ht, _⟩ := multiHeadTransformer_extreme_limits zero_le_one hV hX 0 1
    (fun k j => (hf k j).1) (fun k j => (hf k j).2)
  refine ⟨V, X, γ, one_pos, hV, hQK, hX, multiHeadTransformer_neg hX, ?_, hp,
    fun i => tendsto_neg_atTop_atBot.comp (hp i), hf, ht, ?_⟩
  · change Y 0 0 < Y 1 0; norm_num [Y, unit1]
  · simpa only [PiLp.neg_apply, neg_div] using ht.neg

end Transformer.Clusters
