/-
# The block of the record — gated exclusive heads and its own MLP — does not collapse

**Not a statement of any paper.**  `xsaDrive_spread` for the block of
parameter-golf as the record runs it (github.com/openai/parameter-golf,
`records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`,
`CausalSelfAttention.forward` with `XSA_LAST_N=11`, `SPARSE_ATTN_GATE_ENABLED=1`,
and `Block.forward`): softmax weights `p_{hij}` for any `β`, `Q`, `K`, causal or
not, with rotary embedding, normalised queries and keys or a gain on them; the
exclusive projection on every head; a gate `g_{hi}` per head and token on the
output of each head; values `U_h` and output maps `O_h` carrying `attn_scale`,
the layer's constant scale and the grouping of queries; the residual weight as
the linear term `D = mix[0] - 1`; the feed-forward term `W₂ σ_α(W₁ x)²` of
`Perspective.BlockMLP`; and injections `z_i = mix[1] ⊙ x0_i`.

The gate is the point.  Applied after the projection, it is a factor on the row
of its head (`xsaDrive_gate`), so the block is a `xsaDrive` with weights
`g_{hi} p_{hij}`, and all that is asked of `g` is `|g_{hi}| ≤ Γ`: nothing about
its shape, its input or its smoothness.  `Γ = 1` for the record's
`σ(scale · w · x)` and for `GatedAttn`, `Γ = 2` for `2 σ(·)` of `AttnOutGate`,
`Γ = 1` with no gate at all.  Without the projection this is out of reach:
`IsBoundedBlockOn` needs the rows of a head to share their sum, and a gate that
depends on the token breaks exactly that.
-/

import Transformer.Perspective.BlockXSASpread
import Transformer.Perspective.BlockMLP
import Transformer.Perspective.InjectedAttention
import Mathlib.Analysis.SpecialFunctions.Sigmoid

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **A gate on the output of a head is a factor on its row:** the block that
computes attention, projects out the self value, multiplies by `g_{hi}` and maps
out is `xsaDrive` with the weights `g_{hi} p_{hij}`.

Source: none — posed here; `CausalSelfAttention.forward` of parameter-golf,
where the gate multiplies the output of `_xsa_efficient`, through
`smul_xsaProj`. -/
theorem xsaDrive_gate {e H : ℕ} (g : ℝ → Idx H → Idx n → ℝ)
    (p : ℝ → Idx H → Idx n → Idx n → ℝ) (U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e)
    (O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d) (G : ℝ → EucSpace d → EucSpace d)
    (z : ℝ → Idx n → EucSpace d) (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) :
    xsaDrive (fun t h i j => g t h i * p t h i j) U O G z X t i =
      ∑ h, O t h (g t h i • xsaProj (U t h (X t i)) (∑ j, p t h i j • U t h (X t j)))
        + G t (X t i) + z t i := by
  simp only [xsaDrive, smul_xsaProj, Finset.smul_sum, smul_smul]

/-- **The block of the record is bounded.**  With softmax rows `p_{hij}` of sum
one, a gate `|g_{hi}| ≤ Γ`, values `‖U_h‖ ≤ N`, output maps `‖O_h‖ ≤ K`, a
linear term `‖D‖ ≤ L` (the residual weight `mix[0] - 1`), the feed-forward term
`W₂ σ_α(W₁ x)²` with `‖W₁‖ ≤ K₁`, `‖W₂‖ ≤ K₂`, `|α| ≤ 1`, and injections
`‖z_i‖ ≤ Z` on a window, the block has the bounds `IsBoundedXSABlockOn` there
with `A = Γ`, `B = L + K₂ K₁²` and `L_G = L + 2 K₂ K₁²`.

Source: none — posed here; `CausalSelfAttention.forward`, `Block.forward` and
`MLP.forward` of parameter-golf. -/
theorem isBoundedXSABlockOn_gate_sqMLP {e m H : ℕ} {α : ℝ} (hα : |α| ≤ 1)
    {g : ℝ → Idx H → Idx n → ℝ} {p : ℝ → Idx H → Idx n → Idx n → ℝ}
    {U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e}
    {O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d} {D : ℝ → EucSpace d →L[ℝ] EucSpace d}
    {W₁ : ℝ → EucSpace d →L[ℝ] EucSpace m} {W₂ : ℝ → EucSpace m →L[ℝ] EucSpace d}
    {z : ℝ → Idx n → EucSpace d} {t₀ t₁ Γ N K L K₁ K₂ Z : ℝ}
    (hp : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h i, (∀ j, 0 ≤ p t h i j) ∧ ∑ j, p t h i j = 1)
    (hg : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h i, |g t h i| ≤ Γ)
    (hU : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h, ‖U t h‖ ≤ N) (hO : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h, ‖O t h‖ ≤ K)
    (hD : ∀ t ∈ Set.Ioo t₀ t₁, ‖D t‖ ≤ L) (hW₁ : ∀ t ∈ Set.Ioo t₀ t₁, ‖W₁ t‖ ≤ K₁)
    (hW₂ : ∀ t ∈ Set.Ioo t₀ t₁, ‖W₂ t‖ ≤ K₂) (hz : ∀ t ∈ Set.Ioo t₀ t₁, ∀ i, ‖z t i‖ ≤ Z) :
    IsBoundedXSABlockOn (fun t h i j => g t h i * p t h i j) U O
      (fun t x => D t x + sqMLP α (W₁ t) (W₂ t) x) z t₀ t₁ Γ N K
      (L + K₂ * K₁ ^ 2) (L + 2 * K₂ * K₁ ^ 2) Z where
  abs_row_sum t ht h i :=
    calc ∑ j, |g t h i * p t h i j| = ∑ j, |g t h i| * p t h i j :=
          Finset.sum_congr rfl fun j _ => by
            rw [abs_mul, abs_of_nonneg ((hp t ht h i).1 j)]
      _ = |g t h i| * ∑ j, p t h i j := by rw [Finset.mul_sum]
      _ ≤ Γ := by rw [(hp t ht h i).2, mul_one]; exact hg t ht h i
  norm_value := hU
  norm_out := hO
  norm_ff t ht x := norm_add_sqMLP_le hα (hD t ht) (hW₁ t ht) (hW₂ t ht) x
  lipschitz_ff t ht x y := norm_add_sqMLP_sub_le hα (hD t ht) (hW₁ t ht) (hW₂ t ht) x y
  norm_inj := hz

/-- The hypotheses of `isBoundedXSABlockOn_gate_sqMLP` are satisfiable, the rows
by softmax attention (`eq:P`) for any `β`, `Q`, `K` and configuration, the gate
by a sigmoid of any scores whatever, the rest by the slope `α = 1/2` of
parameter-golf and values, output maps, linear term, weights and injections `0`;
`Γ = 1`. -/
example (β : ℝ) (Q K : TimeParam 1) (X : ℝ → SphereTuple 1 1) (s : ℝ → Idx 1 → Idx 1 → ℝ) :
    |(1 / 2 : ℝ)| ≤ 1 ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) 1, ∀ (_ : Idx 1) (i : Idx 1),
      (∀ j, 0 ≤ attention 1 1 β Q K X t i j) ∧ ∑ j, attention 1 1 β Q K X t i j = 1) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) 1, ∀ (h : Idx 1) (i : Idx 1), |Real.sigmoid (s t h i)| ≤ 1) ∧
    ‖(0 : EucSpace 1 →L[ℝ] EucSpace 1)‖ ≤ 0 ∧ ‖(0 : EucSpace 1)‖ ≤ 0 := by
  refine ⟨by rw [abs_of_pos (by norm_num)]; norm_num,
    fun t _ _ i => ⟨fun j => ?_, sum_attention β Q K X t i⟩, fun _ _ h i => ?_, by simp, by simp⟩
  · exact div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le)
  · rw [abs_of_nonneg (Real.sigmoid_nonneg _)]
    exact Real.sigmoid_le_one _

/-- **Theorem (the block of the record does not collapse).**  Let `w₁`, `w₂` be
linearly independent, `H` a number of heads, `|α| ≤ 1`, `Γ, N, K, L, K₂, Z ≥ 0`
and `τ > 0`.  There is `δ > 0` such that along every flow
`ẋ_i = Proj_{x_i}(v_i(t))` driven on a window `[t₀, t₀ + τ]` by a block whose
heads have softmax rows `p_{hij}` gated by `|g_{hi}| ≤ Γ`, are exclusive, have
values `‖U_h‖ ≤ N` and output maps `‖O_h‖ ≤ K`, whose feed-forward part is
`D x + W₂ σ_α(W₁ x)²` with `‖D‖ ≤ L`, `‖W₁‖ ≤ K₁`, `‖W₂‖ ≤ K₂`, and whose
injections are bounded by `Z` and satisfy `z_k - z_l = w₁`, `z_q - z_r = w₂`
throughout, some time of the window has two tokens more than `δ` apart.  `δ`
depends on neither the number of tokens nor the width of a head or of the hidden
layer, the flow, the weights, the gate or `t₀`.

Source: none — posed here; `xsaDrive_spread` at `A = Γ`, `B = L + K₂ K₁²` and
`L_G = L + 2 K₂ K₁²`, through `isBoundedXSABlockOn_gate_sqMLP`. -/
theorem gatedXSA_block_spread {w₁ w₂ : EucSpace d} (hw : LinearIndependent ℝ ![w₁, w₂])
    (H : ℕ) {α Γ N K L K₁ K₂ Z τ : ℝ} (hα : |α| ≤ 1) (hΓ : 0 ≤ Γ) (hN : 0 ≤ N) (hK : 0 ≤ K)
    (hL : 0 ≤ L) (hK₂ : 0 ≤ K₂) (hZ : 0 ≤ Z) (hτ : 0 < τ) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (n e m : ℕ) (X : ℝ → SphereTuple d n) (g : ℝ → Idx H → Idx n → ℝ)
      (p : ℝ → Idx H → Idx n → Idx n → ℝ) (U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e)
      (O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d) (D : ℝ → EucSpace d →L[ℝ] EucSpace d)
      (W₁ : ℝ → EucSpace d →L[ℝ] EucSpace m) (W₂ : ℝ → EucSpace m →L[ℝ] EucSpace d)
      (z : ℝ → Idx n → EucSpace d) (t₀ : ℝ) (k l q r : Idx n),
      IsDrivenFlowOn X (xsaDrive (fun t h i j => g t h i * p t h i j) U O
        (fun t x => D t x + sqMLP α (W₁ t) (W₂ t) x) z X) t₀ (t₀ + τ) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ h i, (∀ j, 0 ≤ p t h i j) ∧ ∑ j, p t h i j = 1) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ h i, |g t h i| ≤ Γ) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ h, ‖U t h‖ ≤ N) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ h, ‖O t h‖ ≤ K) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ‖D t‖ ≤ L) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ‖W₁ t‖ ≤ K₁) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ‖W₂ t‖ ≤ K₂) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), ∀ i, ‖z t i‖ ≤ Z) →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), z t k - z t l = w₁ ∧ z t q - z t r = w₂) →
      ∃ t ∈ Set.Icc t₀ (t₀ + τ), ∃ i j : Idx n, δ < ‖(X t i : EucSpace d) - X t j‖ := by
  obtain ⟨δ, hδ, h⟩ := xsaDrive_spread hw H hΓ hN hK (B := L + K₂ * K₁ ^ 2)
    (LG := L + 2 * K₂ * K₁ ^ 2) (by positivity) (by positivity) hZ hτ
  exact ⟨δ, hδ, fun n e m X g p U O D W₁ W₂ z t₀ k l q r hX hp hg hU hO hD hW₁ hW₂ hz hkl =>
    h n e X _ U O _ z t₀ k l q r hX
      (isBoundedXSABlockOn_gate_sqMLP hα hp hg hU hO hD hW₁ hW₂ hz) hkl⟩

/-- The hypotheses of `gatedXSA_block_spread` are satisfiable, those inside its
conclusion included: the configuration of the example after `xsaDrive_spread`,
with rows `1/3` on every token, a gate `1/2`, `α = 1/2`, values, output maps,
`D` and `W₁ = W₂ = 0`; `Γ = 1`, `N = K = L = K₁ = K₂ = 0` and `Z = 1`. -/
example : ∃ (X : ℝ → SphereTuple 2 3) (g : ℝ → Idx 1 → Idx 3 → ℝ)
    (p : ℝ → Idx 1 → Idx 3 → Idx 3 → ℝ) (U : ℝ → Idx 1 → EucSpace 2 →L[ℝ] EucSpace 2)
    (O : ℝ → Idx 1 → EucSpace 2 →L[ℝ] EucSpace 2) (D : ℝ → EucSpace 2 →L[ℝ] EucSpace 2)
    (W₁ : ℝ → EucSpace 2 →L[ℝ] EucSpace 1) (W₂ : ℝ → EucSpace 1 →L[ℝ] EucSpace 2)
    (z : ℝ → Idx 3 → EucSpace 2),
    LinearIndependent ℝ ![(EuclideanSpace.single 0 1 : EucSpace 2), EuclideanSpace.single 1 1] ∧
    |(1 / 2 : ℝ)| ≤ 1 ∧
    IsDrivenFlowOn X (xsaDrive (fun t h i j => g t h i * p t h i j) U O
      (fun t x => D t x + sqMLP (1 / 2) (W₁ t) (W₂ t) x) z X) 0 (0 + 1) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ h i, (∀ j, 0 ≤ p t h i j) ∧ ∑ j, p t h i j = 1) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ h i, |g t h i| ≤ 1) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ h, ‖U t h‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ h, ‖O t h‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ‖D t‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ‖W₁ t‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ‖W₂ t‖ ≤ 0) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), ∀ i, ‖z t i‖ ≤ 1) ∧
    ∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), z t 0 - z t 1 = EuclideanSpace.single 0 1 ∧
      z t 0 - z t 2 = EuclideanSpace.single 1 1 := by
  refine ⟨fun _ => ![basePoint 1, basePoint 1, ⟨EuclideanSpace.single 1 1, by simp⟩],
    fun _ _ _ => 1 / 2, fun _ _ _ _ => 1 / 3, fun _ _ => 0, fun _ _ => 0, fun _ => 0,
    fun _ => 0, fun _ => 0,
    fun _ => ![0, -EuclideanSpace.single 0 1, -EuclideanSpace.single 1 1], ?_,
    by rw [abs_of_pos (by norm_num)]; norm_num,
    ⟨fun _ => continuousOn_const, fun t _ i => ?_⟩,
    fun _ _ _ _ => ⟨fun _ => by norm_num, by norm_num⟩,
    fun _ _ _ _ => by rw [abs_of_pos (by norm_num)]; norm_num,
    fun _ _ _ => by simp, fun _ _ _ => by simp, fun _ _ => by simp, fun _ _ => by simp,
    fun _ _ => by simp, fun _ _ i => ?_, fun _ _ => ?_⟩
  · refine linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => ?_) fun i j hij => ?_
    · fin_cases i <;> simp
    · fin_cases i <;> fin_cases j <;> simp_all [EuclideanSpace.inner_single_left]
  · refine (hasDerivAt_const t _).congr_deriv ?_
    fin_cases i <;> simp [xsaDrive, sqMLP, proj, basePoint]
  · fin_cases i <;> simp
  · simp

end Perspective
end Transformer
