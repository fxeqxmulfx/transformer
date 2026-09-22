/-
# A residual block as a driven flow

**Not a statement of any paper.**  The drive of one block of a residual
network,

  `v_i(t) = Σ_h Σ_j a_{hij}(t) V_h(t) x_j + G(t, x_i) + z_i(t)`   (`blockDrive`),

`H` heads with weights `a` and values `V_h`, a feed-forward term `G` applied
token by token, and an injection `z_i`.  Over a window on which the heads'
rows have one common sum, the absolute row sums are at most `A`, `‖V_h‖ ≤ N`,
and `G` is bounded by `B` and `L_G`-Lipschitz on the sphere
(`IsBoundedBlockOn`), the drive is bounded (`norm_blockDrive_le`) and its
differences are `z_k - z_l` up to `(2 H A N + L_G) ε` while the tokens are
`ε`-close (`norm_blockDrive_sub_sub_le`): whatever the tokens share cancels.
So `IsDrivenFlowOn.spread` applies (`Perspective.BlockSpread`).

What falls under this form: softmax attention, causal or not, with any `β`,
`Q`, `K`, rotary embedding or normalised queries and keys (they change `a`
only, and every row still sums to one); several heads, with the output map and
any gain on the attention absorbed into `V_h`; a feed-forward term that is
Lipschitz on the sphere (that of parameter-golf is, `Perspective.BlockMLP`);
a residual weight `mix[0] ≠ 1`, as the token-wise linear term
`(mix[0] - 1) ⊙ x_i` inside `G`.  Every one of `a`, `V`, `G`, `z` is a
function of time, so the blocks of a stack may all differ.
-/

import Transformer.Perspective.DrivenSpread

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The drive of a residual block:**

  `v_i(t) = Σ_h Σ_j a_{hij}(t) V_h(t) x_j + G(t, x_i) + z_i(t)`.

Source: none — posed here; the vector under `Proj_{x_i}` in `eq: albert`
(arXiv:2312.10794v5, §2.3), with the attention weights `eq:P` of each head
replaced by arbitrary weights `a`, the feed-forward term `w σ(a x + b)` by an
arbitrary `G`, and an injection `z_i` added. -/
noncomputable def blockDrive {H : ℕ} (a : ℝ → Idx H → Idx n → Idx n → ℝ)
    (V : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace d) (G : ℝ → EucSpace d → EucSpace d)
    (z : ℝ → Idx n → EucSpace d) (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) :
    EucSpace d :=
  ∑ h, ∑ j, a t h i j • V t h (X t j) + G t (X t i) + z t i

/-- **The bounds of a block on a window** `t₀ < t < t₁`: in every head all rows
of the weights have one sum and absolute sums at most `A`, the values have
operator norm at most `N`, the feed-forward term is bounded by `B` and
`L_G`-Lipschitz on the sphere, and the injections are bounded by `Z`.

Source: none — posed here; for softmax attention (`eq:P`, arXiv:2312.10794v5,
§2.2) every row sums to one. -/
structure IsBoundedBlockOn {H : ℕ} (a : ℝ → Idx H → Idx n → Idx n → ℝ)
    (V : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace d) (G : ℝ → EucSpace d → EucSpace d)
    (z : ℝ → Idx n → EucSpace d) (t₀ t₁ A N B LG Z : ℝ) : Prop where
  row_sum : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h i i', ∑ j, a t h i j = ∑ j, a t h i' j
  abs_row_sum : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h i, ∑ j, |a t h i j| ≤ A
  norm_value : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h, ‖V t h‖ ≤ N
  norm_ff : ∀ t ∈ Set.Ioo t₀ t₁, ∀ x : SSphere d, ‖G t x‖ ≤ B
  lipschitz_ff : ∀ t ∈ Set.Ioo t₀ t₁, ∀ x y : SSphere d,
    ‖G t x - G t y‖ ≤ LG * ‖(x : EucSpace d) - y‖
  norm_inj : ∀ t ∈ Set.Ioo t₀ t₁, ∀ i, ‖z t i‖ ≤ Z

/-- **Two rows of equal sum act alike on nearby vectors:** if `Σ_j b_j = Σ_j c_j`,
both absolute sums are at most `A`, and every `u_j` is within `r` of one `u₀`,
then `‖Σ_j b_j u_j - Σ_j c_j u_j‖ ≤ 2 A r`. -/
theorem norm_sum_smul_sub_sum_smul_le {b c : Idx n → ℝ} {u : Idx n → EucSpace d}
    (u₀ : EucSpace d) {A r : ℝ} (hbc : ∑ j, b j = ∑ j, c j) (hb : ∑ j, |b j| ≤ A)
    (hc : ∑ j, |c j| ≤ A) (hr : 0 ≤ r) (hu : ∀ j, ‖u j - u₀‖ ≤ r) :
    ‖∑ j, b j • u j - ∑ j, c j • u j‖ ≤ 2 * A * r := by
  have hsplit : ∑ j, b j • u j - ∑ j, c j • u j = ∑ j, (b j - c j) • (u j - u₀) := by
    simp only [smul_sub, sub_smul, Finset.sum_sub_distrib, ← Finset.sum_smul, hbc]
    abel
  rw [hsplit]
  calc ‖∑ j, (b j - c j) • (u j - u₀)‖ ≤ ∑ j, ‖(b j - c j) • (u j - u₀)‖ := norm_sum_le _ _
    _ ≤ ∑ j, (|b j| + |c j|) * r := Finset.sum_le_sum fun j _ => by
        rw [norm_smul, Real.norm_eq_abs]
        exact mul_le_mul (abs_sub _ _) (hu j) (norm_nonneg _) (by positivity)
    _ = (∑ j, |b j| + ∑ j, |c j|) * r := by rw [← Finset.sum_add_distrib, Finset.sum_mul]
    _ ≤ 2 * A * r := by nlinarith

/-- The hypotheses of `norm_sum_smul_sub_sum_smul_le` are satisfiable: one
index, `b = c = 1`, `u = u₀ = 0`, `A = 1`, `r = 0`. -/
example : ∑ j : Idx 1, (fun _ => (1 : ℝ)) j = ∑ j : Idx 1, (fun _ => (1 : ℝ)) j ∧
    ∑ j : Idx 1, |(fun _ => (1 : ℝ)) j| ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
    ∀ j : Idx 1, ‖(fun _ => (0 : EucSpace 1)) j - 0‖ ≤ 0 := by
  simp

/-- **The drive of a block is bounded:** `‖v_i(t)‖ ≤ H A N + B + Z`. -/
theorem norm_blockDrive_le {H : ℕ} {a : ℝ → Idx H → Idx n → Idx n → ℝ}
    {V : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace d} {G : ℝ → EucSpace d → EucSpace d}
    {z : ℝ → Idx n → EucSpace d} (X : ℝ → SphereTuple d n) {t A N B Z : ℝ}
    (habs : ∀ h i, ∑ j, |a t h i j| ≤ A) (hV : ∀ h, ‖V t h‖ ≤ N)
    (hG : ∀ x : SSphere d, ‖G t x‖ ≤ B) (hz : ∀ i, ‖z t i‖ ≤ Z) (i : Idx n) :
    ‖blockDrive a V G z X t i‖ ≤ H * (A * N) + B + Z := by
  have hhead : ∀ h, ‖∑ j, a t h i j • V t h (X t j)‖ ≤ A * N := fun h => by
    have hN : 0 ≤ N := (norm_nonneg _).trans (hV h)
    calc ‖∑ j, a t h i j • V t h (X t j)‖ ≤ ∑ j, ‖a t h i j • V t h (X t j)‖ := norm_sum_le _ _
      _ ≤ ∑ j, |a t h i j| * N := Finset.sum_le_sum fun j _ => by
          rw [norm_smul, Real.norm_eq_abs]
          refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
          calc ‖V t h (X t j)‖ ≤ ‖V t h‖ * ‖(X t j : EucSpace d)‖ :=
                ContinuousLinearMap.le_opNorm _ _
            _ ≤ N := by rw [norm_coe_tuple, mul_one]; exact hV h
      _ = (∑ j, |a t h i j|) * N := by rw [Finset.sum_mul]
      _ ≤ A * N := mul_le_mul_of_nonneg_right (habs h i) hN
  have hsum : ‖∑ h, ∑ j, a t h i j • V t h (X t j)‖ ≤ H * (A * N) :=
    calc ‖∑ h, ∑ j, a t h i j • V t h (X t j)‖ ≤ ∑ h, ‖∑ j, a t h i j • V t h (X t j)‖ :=
          norm_sum_le _ _
      _ ≤ ∑ _h : Idx H, A * N := Finset.sum_le_sum fun h _ => hhead h
      _ = H * (A * N) := by simp
  exact norm_add₃_le.trans (add_le_add (add_le_add hsum (hG _)) (hz i))

/-- **What the tokens share cancels in a difference of drives:** if the tokens
are pairwise within `ε`, then
`‖v_k(t) - v_l(t) - (z_k(t) - z_l(t))‖ ≤ (2 H A N + L_G) ε`. -/
theorem norm_blockDrive_sub_sub_le {H : ℕ} {a : ℝ → Idx H → Idx n → Idx n → ℝ}
    {V : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace d} {G : ℝ → EucSpace d → EucSpace d}
    {z : ℝ → Idx n → EucSpace d} {X : ℝ → SphereTuple d n} {t A N LG ε : ℝ}
    (hrow : ∀ h i i', ∑ j, a t h i j = ∑ j, a t h i' j) (habs : ∀ h i, ∑ j, |a t h i j| ≤ A)
    (hV : ∀ h, ‖V t h‖ ≤ N)
    (hG : ∀ x y : SSphere d, ‖G t x - G t y‖ ≤ LG * ‖(x : EucSpace d) - y‖) (hLG : 0 ≤ LG)
    (hε : 0 ≤ ε) (hX : ∀ i j, ‖(X t i : EucSpace d) - X t j‖ ≤ ε) (k l : Idx n) :
    ‖blockDrive a V G z X t k - blockDrive a V G z X t l - (z t k - z t l)‖ ≤
      (2 * H * A * N + LG) * ε := by
  have hsplit : blockDrive a V G z X t k - blockDrive a V G z X t l - (z t k - z t l) =
      ∑ h, (∑ j, a t h k j • V t h (X t j) - ∑ j, a t h l j • V t h (X t j))
        + (G t (X t k) - G t (X t l)) := by
    simp only [blockDrive, Finset.sum_sub_distrib]
    abel
  have hhead : ∀ h, ‖∑ j, a t h k j • V t h (X t j) - ∑ j, a t h l j • V t h (X t j)‖ ≤
      2 * A * (N * ε) := fun h => by
    have hN : 0 ≤ N := (norm_nonneg _).trans (hV h)
    refine norm_sum_smul_sub_sum_smul_le (V t h (X t k)) (hrow h k l) (habs h k) (habs h l)
      (by positivity) fun j => ?_
    rw [← map_sub]
    exact (ContinuousLinearMap.le_opNorm _ _).trans
      (mul_le_mul (hV h) (hX j k) (norm_nonneg _) hN)
  have hsum : ‖∑ h, (∑ j, a t h k j • V t h (X t j) - ∑ j, a t h l j • V t h (X t j))‖ ≤
      H * (2 * A * (N * ε)) :=
    calc ‖∑ h, (∑ j, a t h k j • V t h (X t j) - ∑ j, a t h l j • V t h (X t j))‖
        ≤ ∑ h, ‖∑ j, a t h k j • V t h (X t j) - ∑ j, a t h l j • V t h (X t j)‖ :=
          norm_sum_le _ _
      _ ≤ ∑ _h : Idx H, 2 * A * (N * ε) := Finset.sum_le_sum fun h _ => hhead h
      _ = H * (2 * A * (N * ε)) := by simp
  have hGkl := (hG (X t k) (X t l)).trans (mul_le_mul_of_nonneg_left (hX k l) hLG)
  rw [hsplit]
  calc _ ≤ _ := norm_add_le _ _
    _ ≤ H * (2 * A * (N * ε)) + LG * ε := add_le_add hsum hGkl
    _ = (2 * H * A * N + LG) * ε := by ring

/-- The hypotheses of `norm_blockDrive_le` and `norm_blockDrive_sub_sub_le` are
satisfiable: one head of weight `1` on one token, values `0`, no feed-forward
term, no injection, `A = 1` and `N = B = Z = L_G = ε = 0`. -/
example : ∃ (a : ℝ → Idx 1 → Idx 1 → Idx 1 → ℝ) (V : ℝ → Idx 1 → EucSpace 1 →L[ℝ] EucSpace 1)
    (G : ℝ → EucSpace 1 → EucSpace 1) (z : ℝ → Idx 1 → EucSpace 1) (X : ℝ → SphereTuple 1 1),
    (∀ h i i', ∑ j, a 0 h i j = ∑ j, a 0 h i' j) ∧ (∀ h i, ∑ j, |a 0 h i j| ≤ 1) ∧
    (∀ h, ‖V 0 h‖ ≤ 0) ∧ (∀ x : SSphere 1, ‖G 0 x‖ ≤ 0) ∧
    (∀ x y : SSphere 1, ‖G 0 x - G 0 y‖ ≤ 0 * ‖(x : EucSpace 1) - y‖) ∧
    (∀ i, ‖z 0 i‖ ≤ 0) ∧ ∀ i j, ‖(X 0 i : EucSpace 1) - X 0 j‖ ≤ 0 :=
  ⟨fun _ _ _ _ => 1, fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => basePoint 0, by simp⟩

end Perspective
end Transformer
