module

public import Transformer.Clusters.Section12_MultiHeadClusterGap

/-!
# Actual normalized rate configurations for the original multi-head dynamics

ArXiv:2305.05465v6, §7, `l:exactasymptotic`, with §3's discrete
rescaling and §12's original positive separate heads.
The original normalized extreme coordinates have finite limits.
The bounded original coordinates tend to zero after the exact q^k
rescaling. Every positive escaping coordinate has the normalized
maximum's limit, and every negative escaping coordinate has the
normalized minimum's limit. The proved full-factor growth gives
strictly positive and strictly negative rates, respectively.

Choosing the actual limit of every original coordinate gives one
finite rate configuration. Each rate is one of the two extreme rates
or zero, and it vanishes exactly when its original token is bounded.
All limits and sign assertions are derived from the original vector
iteration; no normalized configuration or escape rate is supplied as
a new premise of the existence theorem.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ}

/-- A genuinely bounded scalar sequence vanishes after division by a strictly expanding discrete power.
Source: arXiv:2305.05465v6, §7, the bounded rate zero in `l:exactasymptotic`, in discrete time. -/
theorem bounded_div_power_zero (f : ℕ → ℝ) (q : ℝ) (hq : 1 < q)
    (hb : ∃ R : ℝ, ∀ k, |f k| ≤ R) : Tendsto (fun k => f k / q ^ k) atTop (nhds 0) := by
  obtain ⟨R, hR⟩ := hb
  have hq0 : 0 < q := by linarith
  have ht : Tendsto (fun k : ℕ => R / q ^ k) atTop (nhds 0) :=
    tendsto_const_nhds.div_atTop (tendsto_pow_atTop_atTop_of_one_lt hq)
  apply squeeze_zero_norm' (Eventually.of_forall fun k => ?_) ht
  rw [Real.norm_eq_abs, abs_div, abs_of_pos (pow_pos hq0 k)]
  exact div_le_div_of_nonneg_right (hR k) (pow_pos hq0 k).le

/-- A normalized limit is zero exactly for the actual original bounded tokens.
Source: arXiv:2305.05465v6, §7, `l:exactasymptotic` and `l:unboundedparticles`, for §12. -/
theorem multiHeadTransformer_normalized_zero_iff_bounded {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i : Idx (m + 1)) {γ : ℝ}
    (hlim : Tendsto (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γ)) :
    γ = 0 ↔ ∃ R : ℝ, ∀ k, |X k i 0| ≤ R := by
  constructor
  · intro hz
    obtain ⟨R, _, halt⟩ := multiHeadTransformer_escape_alternatives hΔt hV hQK hX h
    rcases halt i with hb | hp | hn
    · exact ⟨R, hb⟩
    · have hs := multiHeadTransformer_positive_limit_pos hΔt hV hQK hX h i hp hlim
      linarith
    · have hs := multiHeadTransformer_negative_limit_neg hΔt hV hQK hX h i hn hlim
      linarith
  · intro hb
    exact tendsto_nhds_unique hlim (bounded_div_power_zero (fun k => X k i 0)
      (multiHeadLayerFactor Δt V) (multiHeadLayerFactor_gt_one hΔt V hV h) hb)

/-- Nonzero actual normalized limits are exactly the original tokens escaping to an infinity.
Source: arXiv:2305.05465v6, §7, `l:exactasymptotic` and `l:unboundedparticles`, for §12. -/
theorem multiHeadTransformer_normalized_nonzero_iff_escape {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i : Idx (m + 1)) {γ : ℝ}
    (hlim : Tendsto (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γ)) :
    γ ≠ 0 ↔ Tendsto (fun k => X k i 0) atTop atTop ∨ Tendsto (fun k => X k i 0) atTop atBot := by
  constructor
  · intro hne
    obtain ⟨R, _, halt⟩ := multiHeadTransformer_escape_alternatives hΔt hV hQK hX h
    rcases halt i with hb | hp | hn
    · exact (hne ((multiHeadTransformer_normalized_zero_iff_bounded hΔt hV hQK hX h i hlim).mpr ⟨R, hb⟩)).elim
    · exact Or.inl hp
    · exact Or.inr hn
  · intro he hz
    rcases he with hp | hn
    · have hs := multiHeadTransformer_positive_limit_pos hΔt hV hQK hX h i hp hlim
      linarith
    · have hs := multiHeadTransformer_negative_limit_neg hΔt hV hQK hX h i hn hlim
      linarith

/-- The actual original normalized token configuration converges to its signed extreme-or-zero rates.
Source: arXiv:2305.05465v6, §7, `l:exactasymptotic`, for §12's original discrete multi-head equation. -/
theorem multiHeadTransformer_exists_token_rates {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (hdist : ∀ i j, i ≠ j → X 0 i ≠ X 0 j) (h : Idx H) :
    ∃ (L M : Idx (m + 1)) (γL γM : ℝ) (γ : Idx (m + 1) → ℝ),
      (∀ k j, X k L 0 ≤ X k j 0 ∧ X k j 0 ≤ X k M 0) ∧ γL ≤ γM ∧
      (∀ i, Tendsto (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (γ i))) ∧
      (∀ i, γ i = γL ∨ γ i = 0 ∨ γ i = γM) ∧
      (∀ i, Tendsto (fun k => X k i 0) atTop atTop → γ i = γM ∧ 0 < γ i) ∧
      (∀ i, Tendsto (fun k => X k i 0) atTop atBot → γ i = γL ∧ γ i < 0) ∧
      (∀ i, γ i = 0 ↔ ∃ R : ℝ, ∀ k, |X k i 0| ≤ R) := by
  classical
  obtain ⟨L, M, γL, γM, hf, htL, htM, hle⟩ :=
    multiHeadTransformer_exists_extreme_limits hΔt.le hV hQK hX hdist
  obtain ⟨R, _, halt⟩ := multiHeadTransformer_escape_alternatives hΔt hV hQK hX h
  have hex : ∀ i, ∃ a : ℝ, Tendsto (fun k => X k i 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds a) := by
    intro i
    rcases halt i with hb | hp | hn
    · exact ⟨0, bounded_div_power_zero (fun k => X k i 0) (multiHeadLayerFactor Δt V)
        (multiHeadLayerFactor_gt_one hΔt V hV h) ⟨R, hb⟩⟩
    · exact ⟨γM, multiHeadTransformer_positive_rate_eq_max hΔt hV hQK hX h i M hp
        (fun k j => (hf k j).2) htM⟩
    · exact ⟨γL, multiHeadTransformer_negative_rate_eq_min hΔt hV hQK hX h i L hn
        (fun k j => (hf k j).1) htL⟩
  choose γ hγ using hex
  have hpos : ∀ i, Tendsto (fun k => X k i 0) atTop atTop → γ i = γM ∧ 0 < γ i := by
    intro i hi
    have ht := multiHeadTransformer_positive_rate_eq_max hΔt hV hQK hX h i M hi
      (fun k j => (hf k j).2) htM
    exact ⟨tendsto_nhds_unique (hγ i) ht, multiHeadTransformer_positive_limit_pos hΔt hV hQK hX h i hi (hγ i)⟩
  have hneg : ∀ i, Tendsto (fun k => X k i 0) atTop atBot → γ i = γL ∧ γ i < 0 := by
    intro i hi
    have ht := multiHeadTransformer_negative_rate_eq_min hΔt hV hQK hX h i L hi
      (fun k j => (hf k j).1) htL
    exact ⟨tendsto_nhds_unique (hγ i) ht, multiHeadTransformer_negative_limit_neg hΔt hV hQK hX h i hi (hγ i)⟩
  have hz : ∀ i, γ i = 0 ↔ ∃ A : ℝ, ∀ k, |X k i 0| ≤ A :=
    fun i => multiHeadTransformer_normalized_zero_iff_bounded hΔt hV hQK hX h i (hγ i)
  refine ⟨L, M, γL, γM, γ, hf, hle, hγ, ?_, hpos, hneg, hz⟩
  intro i
  rcases halt i with hb | hp | hn
  · exact Or.inr (Or.inl ((hz i).mpr ⟨R, hb⟩))
  · exact Or.inr (Or.inr (hpos i hp).1)
  · exact Or.inl (hneg i hn).1

/-- A nonzero bounded scalar sequence realizes the vanishing normalized-rate premises. -/
example : ∃ f : ℕ → ℝ, (1 : ℝ) < 2 ∧ ∃ R : ℝ, ∀ k, |f k| ≤ R :=
  ⟨fun _ => 1, by norm_num, 1, fun _ => by norm_num⟩

/-- Actual distinct original data with a bounded middle token and both signs of escape
jointly realize all original rate-configuration premises, including a zero normalized limit. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 3 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (V h) (V h)) ∧
    MultiHeadTransformer 1 V V V X ∧ (∀ i j, i ≠ j → X 0 i ≠ X 0 j) ∧
    (∀ k, |X k 1 0| ≤ 0) ∧ Tendsto (fun k => X k 2 0) atTop atTop ∧
    Tendsto (fun k => X k 0 0) atTop atBot ∧
    Tendsto (fun k => X k 1 0 / multiHeadLayerFactor 1 V ^ k) atTop (nhds 0) := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  have hV : ∀ h, IsPosDefOp (V h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (V h) (V h) := fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  obtain ⟨X, hX, hz, hp, hn, hord⟩ := exists_multiHead_symmetric_escape one_pos hV hQK 0
  have hbound : ∀ k, |X k 1 0| ≤ 0 := by intro k; rw [hz k]; norm_num
  refine ⟨V, X, one_pos, hV, hQK, hX, ?_, hbound, hp, hn, ?_⟩
  · intro i j hij he
    have hc := congrArg (fun z : EucSpace 1 => z 0) he
    rcases lt_or_gt_of_ne hij with hlt | hgt
    · have hs := hord i j hlt; linarith
    · have hs := hord j i hgt; linarith
  · exact bounded_div_power_zero (fun k => X k 1 0) (multiHeadLayerFactor 1 V)
      (multiHeadLayerFactor_gt_one one_pos V hV 0) ⟨0, hbound⟩

end Transformer.Clusters
