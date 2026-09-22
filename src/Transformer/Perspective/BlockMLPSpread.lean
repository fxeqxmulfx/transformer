/-
# A block with the feed-forward term of parameter-golf does not collapse

**Not a statement of any paper.**  `blockDrive_spread` for a block of
parameter-golf (github.com/openai/parameter-golf,
`records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`,
`Block.forward`) with its own feed-forward term: heads with nonnegative weights
and rows of sum one — softmax attention for any `β`, `Q`, `K`, causal or not,
with rotary embedding, normalised queries and keys, or a gain on the queries —,
values `V_h` that carry `attn_scale` and the output map, the residual weight as
the linear term `D = mix[0] - 1`, the feed-forward term `W₂ σ_α(W₁ x)²` of
`Perspective.BlockMLP`, and injections `z_i`.  The bounds `IsBoundedBlockOn` of
such a block follow from the norms of its weights (`isBoundedBlockOn_sqMLP`),
so what `blockDrive_spread` assumes of the feed-forward term becomes
`‖W₁‖ ≤ K₁`, `‖W₂‖ ≤ K₂` (`sqMLP_block_spread`).

The attention of the record is not of this form: every layer runs XSA
(`XSA_LAST_N = 11`), and the run enables a sigmoid gate on the output of every
head (`SPARSE_ATTN_GATE_ENABLED=1`), so the rows of a head act differently on
different tokens.  It is a drive of `Perspective.BlockXSA` instead, where the
projection of `eq:xsa` makes the common row sum unnecessary and the gate a
factor on the row (`gatedXSA_block_spread`).
-/

import Transformer.Perspective.BlockMLP
import Transformer.Perspective.InjectedAttention

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- **A block with the feed-forward term of parameter-golf is bounded.**  With
heads of nonnegative weights and rows of sum one, values `‖V_h‖ ≤ N`, a linear
term `‖D‖ ≤ L` (the residual weight `mix[0] - 1`), the feed-forward term
`W₂ σ_α(W₁ x)²` with `‖W₁‖ ≤ K₁`, `‖W₂‖ ≤ K₂`, `|α| ≤ 1`, and injections
`‖z_i‖ ≤ Z` on a window, the block has the bounds `IsBoundedBlockOn` there with
`A = 1`, `B = L + K₂ K₁²` and `L_G = L + 2 K₂ K₁²`.

Source: none — posed here; `Block.forward` and `MLP.forward` of parameter-golf. -/
theorem isBoundedBlockOn_sqMLP {n H m : ℕ} {α : ℝ} (hα : |α| ≤ 1)
    {a : ℝ → Idx H → Idx n → Idx n → ℝ} {V : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace d}
    {D : ℝ → EucSpace d →L[ℝ] EucSpace d} {W₁ : ℝ → EucSpace d →L[ℝ] EucSpace m}
    {W₂ : ℝ → EucSpace m →L[ℝ] EucSpace d} {z : ℝ → Idx n → EucSpace d}
    {t₀ t₁ N L K₁ K₂ Z : ℝ}
    (ha : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h i, (∀ j, 0 ≤ a t h i j) ∧ ∑ j, a t h i j = 1)
    (hV : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h, ‖V t h‖ ≤ N) (hD : ∀ t ∈ Set.Ioo t₀ t₁, ‖D t‖ ≤ L)
    (hW₁ : ∀ t ∈ Set.Ioo t₀ t₁, ‖W₁ t‖ ≤ K₁) (hW₂ : ∀ t ∈ Set.Ioo t₀ t₁, ‖W₂ t‖ ≤ K₂)
    (hz : ∀ t ∈ Set.Ioo t₀ t₁, ∀ i, ‖z t i‖ ≤ Z) :
    IsBoundedBlockOn a V (fun t x => D t x + sqMLP α (W₁ t) (W₂ t) x) z t₀ t₁
      1 N (L + K₂ * K₁ ^ 2) (L + 2 * K₂ * K₁ ^ 2) Z where
  row_sum t ht h i i' := by rw [(ha t ht h i).2, (ha t ht h i').2]
  abs_row_sum t ht h i := by
    rw [Finset.sum_congr rfl fun j _ => abs_of_nonneg ((ha t ht h i).1 j), (ha t ht h i).2]
  norm_value := hV
  norm_ff t ht x := norm_add_sqMLP_le hα (hD t ht) (hW₁ t ht) (hW₂ t ht) x
  lipschitz_ff t ht x y := norm_add_sqMLP_sub_le hα (hD t ht) (hW₁ t ht) (hW₂ t ht) x y
  norm_inj := hz

/-- The hypotheses of `isBoundedBlockOn_sqMLP` are satisfiable, the weights by
softmax attention (`eq:P`) for any `β`, `Q`, `K` and configuration, the rest by
the slope `α = 1/2` of parameter-golf and values, linear term, weights and
injections `0`. -/
example (β : ℝ) (Q K : TimeParam 1) (X : ℝ → SphereTuple 1 1) :
    |(1 / 2 : ℝ)| ≤ 1 ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) 1, ∀ (_ : Idx 1) (i : Idx 1),
      (∀ j, 0 ≤ attention 1 1 β Q K X t i j) ∧ ∑ j, attention 1 1 β Q K X t i j = 1) ∧
    ‖(0 : EucSpace 1 →L[ℝ] EucSpace 1)‖ ≤ 0 ∧ ‖(0 : EucSpace 1)‖ ≤ 0 := by
  refine ⟨by rw [abs_of_pos (by norm_num)]; norm_num,
    fun t _ _ i => ⟨fun j => ?_, sum_attention β Q K X t i⟩, by simp, by simp⟩
  exact div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le)

/-- **Theorem (a block with the feed-forward term of parameter-golf does not
collapse).**  Let `w₁`, `w₂` be linearly independent, `H` a number of heads,
`|α| ≤ 1`, `N, L, K₂, Z ≥ 0` and `τ > 0`.  There is `δ > 0` such that along
every flow `ẋ_i = Proj_{x_i}(v_i(t))` driven on a window `[t₀, t₀ + τ]` by a
block whose heads have nonnegative weights with rows of sum one and values
`‖V_h‖ ≤ N`, whose feed-forward part is `D x + W₂ σ_α(W₁ x)²` with `‖D‖ ≤ L`,
`‖W₁‖ ≤ K₁`, `‖W₂‖ ≤ K₂`, and whose injections are bounded by `Z` and satisfy
`z_k - z_l = w₁`, `z_q - z_p = w₂` throughout, some time of the window has two
tokens more than `δ` apart.  `δ` depends on neither the number of tokens nor the
hidden width, the flow, the weights or `t₀`.

Source: none — posed here; `blockDrive_spread` at `A = 1`, `B = L + K₂ K₁²` and
`L_G = L + 2 K₂ K₁²`, through `isBoundedBlockOn_sqMLP`. -/
theorem sqMLP_block_spread {w₁ w₂ : EucSpace d} (hw : LinearIndependent ℝ ![w₁, w₂])
    (H : ℕ) {α N L K₁ K₂ Z τ : ℝ} (hα : |α| ≤ 1) (hN : 0 ≤ N) (hL : 0 ≤ L) (hK₂ : 0 ≤ K₂)
    (hZ : 0 ≤ Z) (hτ : 0 < τ) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (n m : ℕ) (X : ℝ → SphereTuple d n)
      (a : ℝ → Idx H → Idx n → Idx n → ℝ) (V : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace d)
      (D : ℝ → EucSpace d →L[ℝ] EucSpace d) (W₁ : ℝ → EucSpace d →L[ℝ] EucSpace m)
      (W₂ : ℝ → EucSpace m →L[ℝ] EucSpace d) (z : ℝ → Idx n → EucSpace d) (t₀ : ℝ)
      (k l q p : Idx n),
      IsDrivenFlowOn X
        (blockDrive a V (fun t x => D t x + sqMLP α (W₁ t) (W₂ t) x) z X) t₀ (t₀ + τ) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ h i, (∀ j, 0 ≤ a t h i j) ∧ ∑ j, a t h i j = 1) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ h, ‖V t h‖ ≤ N) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ‖D t‖ ≤ L) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ‖W₁ t‖ ≤ K₁) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ‖W₂ t‖ ≤ K₂) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ i, ‖z t i‖ ≤ Z) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), z t k - z t l = w₁ ∧ z t q - z t p = w₂) →
      ∃ t ∈ Set.Icc t₀ (t₀ + τ), ∃ i j : Idx n, δ < ‖(X t i : EucSpace d) - X t j‖ := by
  obtain ⟨δ, hδ, h⟩ := blockDrive_spread hw H zero_le_one hN (B := L + K₂ * K₁ ^ 2)
    (LG := L + 2 * K₂ * K₁ ^ 2) (by positivity) (by positivity) hZ hτ
  exact ⟨δ, hδ, fun n m X a V D W₁ W₂ z t₀ k l q p hX ha hV hD hW₁ hW₂ hz hkl =>
    h n X a V _ z t₀ k l q p hX (isBoundedBlockOn_sqMLP hα ha hV hD hW₁ hW₂ hz) hkl⟩

/-- The hypotheses of `sqMLP_block_spread` are satisfiable, those inside its
conclusion included: the configuration of the first example after
`blockDrive_spread`, with weights `1/3` on every token, `α = 1/2`, `D = 0` and
`W₁ = W₂ = 0`; `N = L = K₁ = K₂ = 0` and `Z = 1`. -/
example : ∃ (X : ℝ → SphereTuple 2 3) (a : ℝ → Idx 1 → Idx 3 → Idx 3 → ℝ)
    (V : ℝ → Idx 1 → EucSpace 2 →L[ℝ] EucSpace 2) (D : ℝ → EucSpace 2 →L[ℝ] EucSpace 2)
    (W₁ : ℝ → EucSpace 2 →L[ℝ] EucSpace 1) (W₂ : ℝ → EucSpace 1 →L[ℝ] EucSpace 2)
    (z : ℝ → Idx 3 → EucSpace 2),
    LinearIndependent ℝ ![(EuclideanSpace.single 0 1 : EucSpace 2), EuclideanSpace.single 1 1] ∧
    |(1 / 2 : ℝ)| ≤ 1 ∧
    IsDrivenFlowOn X
      (blockDrive a V (fun t x => D t x + sqMLP (1 / 2) (W₁ t) (W₂ t) x) z X) 0 (0 + 1) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ h i, (∀ j, 0 ≤ a t h i j) ∧ ∑ j, a t h i j = 1) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ h, ‖V t h‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ‖D t‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ‖W₁ t‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ‖W₂ t‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ i, ‖z t i‖ ≤ 1) ∧
    ∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), z t 0 - z t 1 = EuclideanSpace.single 0 1 ∧
      z t 0 - z t 2 = EuclideanSpace.single 1 1 := by
  refine ⟨fun _ => ![basePoint 1, basePoint 1, ⟨EuclideanSpace.single 1 1, by simp⟩],
    fun _ _ _ _ => 1 / 3, fun _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0,
    fun _ => ![0, -EuclideanSpace.single 0 1, -EuclideanSpace.single 1 1], ?_,
    by rw [abs_of_pos (by norm_num)]; norm_num,
    ⟨fun _ => continuousOn_const, fun t _ i => ?_⟩,
    fun _ _ _ _ => ⟨fun _ => by norm_num, by norm_num⟩, fun _ _ _ => by simp,
    fun _ _ => by simp, fun _ _ => by simp, fun _ _ => by simp, fun _ _ i => ?_,
    fun _ _ => ?_⟩
  · refine linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => ?_) fun i j hij => ?_
    · fin_cases i <;> simp
    · fin_cases i <;> fin_cases j <;> simp_all [EuclideanSpace.inner_single_left]
  · refine (hasDerivAt_const t _).congr_deriv ?_
    fin_cases i <;> simp [blockDrive, sqMLP, proj, basePoint]
  · fin_cases i <;> simp
  · simp

end Perspective
end Transformer
