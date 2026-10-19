module

public import Transformer.Clusters.Section12_MultiHeadInterval
public import Transformer.Clusters.Section12_MultiHeadExtrema
public import Mathlib.Topology.Order.MonotoneConvergence

/-!
# Limits of the original extreme coordinates after exact multi-head rescaling

ArXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, with
§3's discrete rescaling and the original separate heads of §12.
The largest original token has drift at most its own value times
Σv_h. Dividing by the exact original layer factor q=1+Δt Σv_h
makes its coordinate nonincreasing. The smallest normalized coordinate
is nondecreasing, by sign reversal of the original vector equation.

The proved original interval estimate bounds both normalized extreme
sequences. Monotone convergence therefore supplies their actual limits.
The limits are ordered. Strict positivity or negativity of a limit
when its original coordinate escapes is a separate estimate; no such
nonzero-rate assertion is assumed here.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology Set

namespace Transformer.Clusters

variable {m H : ℕ}

/-- The exact original summed drift is at most the maximum times the total original value coefficient.
Source: arXiv:2305.05465v6, §7, the upper average bound in `l:exactasymptotic`, summed over §12's heads. -/
theorem multiHeadScalarDrift_le_max (c v : Idx H → ℝ) (hv : ∀ h, 0 ≤ v h)
    (y : Idx (m + 1) → ℝ) (r : ℝ) (M : Idx (m + 1)) (hM : ∀ j, y j ≤ y M) :
    multiHeadScalarDrift c v y r ≤ (∑ h, v h) * y M := by
  have hs := Finset.sum_le_sum fun h (_ : h ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (scalarHeadAverage_le_max (c h) y r M hM) (hv h)
  rw [← Finset.sum_mul] at hs
  exact hs

/-- The reflected original summed drift is at least the minimum times the total original value coefficient.
Source: arXiv:2305.05465v6, §7, the lower average bound in `l:exactasymptotic`, for §12's head sum. -/
theorem multiHeadScalarDrift_ge_min (c v : Idx H → ℝ) (hv : ∀ h, 0 ≤ v h)
    (y : Idx (m + 1) → ℝ) (r : ℝ) (L : Idx (m + 1)) (hL : ∀ j, y L ≤ y j) :
    (∑ h, v h) * y L ≤ multiHeadScalarDrift c v y r := by
  have hs := multiHeadScalarDrift_le_max c v hv (fun j => -y j) (-r) L (fun j => by linarith [hL j])
  rw [multiHeadScalarDrift_neg, mul_neg] at hs
  linarith

/-- An actual original maximum increases by at most the exact original layer factor.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, for §12's discrete equation. -/
theorem multiHeadTransformer_max_step {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (k : ℕ) (M : Idx (m + 1)) (hM : ∀ j, X k j 0 ≤ X k M 0) :
    X (k + 1) M 0 ≤ multiHeadLayerFactor Δt V * X k M 0 := by
  have hd := multiHeadScalarDrift_le_max
    (fun h => inner (𝕜 := ℝ) (Q h unit1) (K h unit1)) (fun h => (V h unit1) 0)
    (fun h => (value_coordinate_pos (V h) (hV h)).le) (fun j => X k j 0) (X k M 0) M hM
  have hm := mul_le_mul_of_nonneg_left hd hΔt
  rw [multiHeadTransformer_coord hX k M]
  dsimp only [multiHeadLayerFactor, multiHeadValueCoefficient]
  nlinarith

/-- A fixed original maximum is nonincreasing after the exact original discrete rescaling.
Source: arXiv:2305.05465v6, §3, `r:discreterescaling`, and §7, `l:exactasymptotic`, for §12. -/
theorem multiHeadTransformer_normalized_max_antitone {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (M : Idx (m + 1)) (hM : ∀ k j, X k j 0 ≤ X k M 0) :
    Antitone (fun k => X k M 0 / multiHeadLayerFactor Δt V ^ k) := by
  let q := multiHeadLayerFactor Δt V
  have hq : 0 < q := by
    have hge := multiHeadLayerFactor_ge_one hΔt V hV
    change 1 ≤ q at hge
    linarith
  apply antitone_nat_of_succ_le
  intro k
  have hs : X (k + 1) M 0 ≤ q * X k M 0 := multiHeadTransformer_max_step hΔt hV hX k M (hM k)
  apply (div_le_div_iff₀ (pow_pos hq (k + 1)) (pow_pos hq k)).mpr
  have hm := mul_le_mul_of_nonneg_right hs (pow_pos hq k).le
  rw [pow_succ]
  nlinarith

/-- A fixed original minimum is nondecreasing after the exact original discrete rescaling.
Source: arXiv:2305.05465v6, §3, `r:discreterescaling`, and §7, `l:exactasymptotic`, for §12. -/
theorem multiHeadTransformer_normalized_min_monotone {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (L : Idx (m + 1)) (hL : ∀ k j, X k L 0 ≤ X k j 0) :
    Monotone (fun k => X k L 0 / multiHeadLayerFactor Δt V ^ k) := by
  have hn : ∀ k j, (-X k j) 0 ≤ (-X k L) 0 := by
    intro k j
    change -(X k j 0) ≤ -(X k L 0)
    linarith [hL k j]
  have ha := multiHeadTransformer_normalized_max_antitone hΔt hV (multiHeadTransformer_neg hX) L hn
  intro a b hab
  have hs := ha hab
  change -(X b L 0) / multiHeadLayerFactor Δt V ^ b ≤ -(X a L 0) / multiHeadLayerFactor Δt V ^ a at hs
  rw [neg_div, neg_div] at hs
  linarith

/-- The two actual normalized extreme coordinates have ordered finite limits.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, for the exact discrete rescaling of §12. -/
theorem multiHeadTransformer_extreme_limits {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (L M : Idx (m + 1)) (hL : ∀ k j, X k L 0 ≤ X k j 0) (hM : ∀ k j, X k j 0 ≤ X k M 0) :
    ∃ γL γM : ℝ,
      Tendsto (fun k => X k L 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γL) ∧
      Tendsto (fun k => X k M 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γM) ∧ γL ≤ γM := by
  have hini : ∀ i, X 0 i 0 ∈ Icc (X 0 L 0) (X 0 M 0) := fun i => ⟨hL 0 i, hM 0 i⟩
  have hb := multiHeadTransformer_normalized_mem_Icc hΔt hV hX (X 0 L 0) (X 0 M 0) hini
  let fL := fun k => X k L 0 / multiHeadLayerFactor Δt V ^ k
  let fM := fun k => X k M 0 / multiHeadLayerFactor Δt V ^ k
  have hsup : BddAbove (range fL) := by
    refine ⟨X 0 M 0, ?_⟩
    rintro a ⟨k, rfl⟩
    exact (hb k L).2
  have hinf : BddBelow (range fM) := by
    refine ⟨X 0 L 0, ?_⟩
    rintro a ⟨k, rfl⟩
    exact (hb k M).1
  have htL := tendsto_atTop_ciSup (multiHeadTransformer_normalized_min_monotone hΔt hV hX L hL) hsup
  have htM := tendsto_atTop_ciInf (multiHeadTransformer_normalized_max_antitone hΔt hV hX M hM) hinf
  refine ⟨⨆ k, fL k, ⨅ k, fM k, htL, htM, ?_⟩
  apply le_of_tendsto_of_tendsto htL htM
  apply Eventually.of_forall
  intro k
  exact div_le_div_of_nonneg_right (hL k M) (by
    have hge := multiHeadLayerFactor_ge_one hΔt V hV
    have hp : 0 < multiHeadLayerFactor Δt V := by linarith
    exact (pow_pos hp k).le)

/-- Original distinct data admit actual fixed extreme indices and ordered finite normalized limits.
Source: arXiv:2305.05465v6, §7's fixed extreme tokens and `l:exactasymptotic`, for §12's original equation. -/
theorem multiHeadTransformer_exists_extreme_limits {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (hdist : ∀ i j, i ≠ j → X 0 i ≠ X 0 j) :
    ∃ (L M : Idx (m + 1)) (γL γM : ℝ),
      (∀ k j, X k L 0 ≤ X k j 0 ∧ X k j 0 ≤ X k M 0) ∧
      Tendsto (fun k => X k L 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γL) ∧
      Tendsto (fun k => X k M 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds γM) ∧ γL ≤ γM := by
  obtain ⟨L, M, _, _, hf, _, _⟩ := multiHeadTransformer_extreme_data hΔt hV hQK hX hdist
  obtain ⟨γL, γM, htL, htM, hle⟩ := multiHeadTransformer_extreme_limits hΔt hV hX L M
    (fun k j => (hf k j).1) (fun k j => (hf k j).2)
  exact ⟨L, M, γL, γM, hf, htL, htM, hle⟩

/-- Two distinct bounded original scalar values realize both static average-bound hypotheses. -/
example : ∃ (y : Idx 2 → ℝ) (v : Idx 2 → ℝ),
    (∀ h, 0 ≤ v h) ∧ (∀ j, y 0 ≤ y j) ∧ (∀ j, y j ≤ y 1) := by
  refine ⟨![-1, 1], ![1, 2], ?_, ?_, ?_⟩
  · intro h; fin_cases h <;> norm_num
  · intro j; fin_cases j <;> norm_num
  · intro j; fin_cases j <;> norm_num

/-- Two genuine original positive heads and distinct nonzero initial data have actual fixed extreme indices. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 2 → EucSpace 1) (L M : Idx 2),
    (0 : ℝ) ≤ 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (V h) (V h)) ∧
    MultiHeadTransformer 1 V V V X ∧ (∀ i j, i ≠ j → X 0 i ≠ X 0 j) ∧
    (∀ k j, X k L 0 ≤ X k j 0 ∧ X k j 0 ≤ X k M 0) := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  let Y : Idx 2 → EucSpace 1 := ![-unit1, unit1]
  let X := multiHeadTrajectory 1 V V V Y
  have hV : ∀ h, IsPosDefOp (V h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (V h) (V h) :=
    fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  have hX : MultiHeadTransformer 1 V V V X := multiHeadTrajectory_solution _ _ _ _ _
  have hdist : ∀ i j, i ≠ j → X 0 i ≠ X 0 j := by
    intro i j hij he
    have hc := congrArg (fun z : EucSpace 1 => z 0) he
    change Y i 0 = Y j 0 at hc
    fin_cases i <;> fin_cases j <;> norm_num at hij <;> norm_num [Y, unit1] at hc
  obtain ⟨L, M, _, _, hf, _, _⟩ := multiHeadTransformer_extreme_data zero_le_one hV hQK hX hdist
  exact ⟨V, X, L, M, zero_le_one, hV, hQK, hX, hdist, hf⟩

end Transformer.Clusters
