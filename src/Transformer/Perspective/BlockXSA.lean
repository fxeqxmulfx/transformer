/-
# A residual block with exclusive self-attention as a driven flow

**Not a statement of any paper.**  The drive of a block whose heads are
exclusive (`eq:xsa`, arXiv:2603.09078v1, §3): every head reads the tokens
through its value map `U_h`, has its own value subtracted off
(`Perspective.XSAProj`), and is mapped back by `O_h`,

  `v_i(t) = Σ_h O_h(t) P^⊥_{U_h x_i} (Σ_j a_{hij}(t) U_h(t) x_j) + G(t, x_i) + z_i(t)`
  (`xsaDrive`),

with a feed-forward term `G` applied token by token and an injection `z_i`.
Over a window on which the absolute row sums are at most `A`, `‖U_h‖ ≤ N`,
`‖O_h‖ ≤ K`, and `G` is bounded by `B` and `L_G`-Lipschitz on the sphere
(`IsBoundedXSABlockOn`), the drive is bounded (`norm_xsaDrive_le`) and its
differences are `z_k - z_l` up to `(2 H K A N + L_G) ε` while the tokens are
`ε`-close (`norm_xsaDrive_sub_sub_le`), so `IsDrivenFlowOn.spread` applies
(`Perspective.BlockXSASpread`).

The projection buys two things over `blockDrive`.  The rows are free: nothing
is asked of the weights but `Σ_j |a_{hij}| ≤ A`, so a gate on the output of a
head — which is a gate on its row, `smul_xsaProj` — needs only to be bounded,
where rows of different sums leave a difference of drives of order one, which
is what the `row_sum` field of `IsBoundedBlockOn` rules out by hypothesis.  And
the estimate is stronger: every head is `O(ε)` by itself near a cluster
(`norm_xsaAttn_le_of_close`), not only in a difference, so an exclusive
attention layer cannot hold a cluster together at first order — what happens
there is decided by `G` and `z` alone.

What falls under this form: multi-head XSA as in Algorithm 1 of
arXiv:2603.09078v1 and `CausalSelfAttention.forward` of parameter-golf
(github.com/openai/parameter-golf,
`records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`),
with grouped queries (the heads of a group share `U_h`), the gain on the
queries, rotary embedding and the normalisation of queries and keys (they
change `a` only), `attn_scale` and the layer's constant scale folded into `O_h`.
-/

import Transformer.Perspective.XSAProj
import Transformer.Perspective.BlockDrive

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The drive of a residual block with exclusive self-attention:**

  `v_i(t) = Σ_h O_h(t) P^⊥_{U_h x_i} (Σ_j a_{hij}(t) U_h(t) x_j) + G(t, x_i) + z_i(t)`,

the values `U_h` mapping into the space of a head and the output maps `O_h`
back.

Source: none — posed here; `blockDrive` with every head replaced by its
exclusive output (`eq:xsa`, arXiv:2603.09078v1, §3; Algorithm 1 for several
heads), as in `CausalSelfAttention.forward` of parameter-golf. -/
noncomputable def xsaDrive {e H : ℕ} (a : ℝ → Idx H → Idx n → Idx n → ℝ)
    (U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e)
    (O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d) (G : ℝ → EucSpace d → EucSpace d)
    (z : ℝ → Idx n → EucSpace d) (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) :
    EucSpace d :=
  ∑ h, O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j))) + G t (X t i) + z t i

/-- **The bounds of an exclusive block on a window** `t₀ < t < t₁`: in every head
the absolute row sums of the weights are at most `A`, the values have operator
norm at most `N` and the output maps at most `K`, the feed-forward term is
bounded by `B` and `L_G`-Lipschitz on the sphere, and the injections are bounded
by `Z`.

Source: none — posed here; `IsBoundedBlockOn` without its `row_sum` field, which
the projection makes unnecessary.  For softmax attention (`eq:P`,
arXiv:2312.10794v5, §2.2) under a gate `g ∈ [0, 1]`, `Σ_j |a_{hij}| = g_{hi} ≤ 1`. -/
structure IsBoundedXSABlockOn {e H : ℕ} (a : ℝ → Idx H → Idx n → Idx n → ℝ)
    (U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e)
    (O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d) (G : ℝ → EucSpace d → EucSpace d)
    (z : ℝ → Idx n → EucSpace d) (t₀ t₁ A N K B LG Z : ℝ) : Prop where
  abs_row_sum : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h i, ∑ j, |a t h i j| ≤ A
  norm_value : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h, ‖U t h‖ ≤ N
  norm_out : ∀ t ∈ Set.Ioo t₀ t₁, ∀ h, ‖O t h‖ ≤ K
  norm_ff : ∀ t ∈ Set.Ioo t₀ t₁, ∀ x : SSphere d, ‖G t x‖ ≤ B
  lipschitz_ff : ∀ t ∈ Set.Ioo t₀ t₁, ∀ x y : SSphere d,
    ‖G t x - G t y‖ ≤ LG * ‖(x : EucSpace d) - y‖
  norm_inj : ∀ t ∈ Set.Ioo t₀ t₁, ∀ i, ‖z t i‖ ≤ Z

/-- **The exclusive attention of a block is bounded:**
`‖Σ_h O_h P^⊥ (Σ_j a_{hij} U_h x_j)‖ ≤ H K A N`. -/
theorem norm_xsaAttn_le {e H : ℕ} {a : ℝ → Idx H → Idx n → Idx n → ℝ}
    {U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e}
    {O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d} (X : ℝ → SphereTuple d n) {t A N K : ℝ}
    (habs : ∀ h i, ∑ j, |a t h i j| ≤ A) (hU : ∀ h, ‖U t h‖ ≤ N) (hO : ∀ h, ‖O t h‖ ≤ K)
    (i : Idx n) :
    ‖∑ h, O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j)))‖ ≤
      H * (K * (A * N)) := by
  have hhead : ∀ h, ‖O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j)))‖ ≤
      K * (A * N) := fun h => by
    have hN : 0 ≤ N := (norm_nonneg _).trans (hU h)
    have hy : ‖∑ j, a t h i j • U t h (X t j)‖ ≤ A * N :=
      calc ‖∑ j, a t h i j • U t h (X t j)‖ ≤ ∑ j, ‖a t h i j • U t h (X t j)‖ := norm_sum_le _ _
        _ ≤ ∑ j, |a t h i j| * N := Finset.sum_le_sum fun j _ => by
            rw [norm_smul, Real.norm_eq_abs]
            refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
            calc ‖U t h (X t j)‖ ≤ ‖U t h‖ * ‖(X t j : EucSpace d)‖ :=
                  ContinuousLinearMap.le_opNorm _ _
              _ ≤ N := by rw [norm_coe_tuple, mul_one]; exact hU h
        _ = (∑ j, |a t h i j|) * N := by rw [Finset.sum_mul]
        _ ≤ A * N := mul_le_mul_of_nonneg_right (habs h i) hN
    exact (ContinuousLinearMap.le_opNorm _ _).trans (mul_le_mul (hO h)
      ((norm_xsaProj_le _ _).trans hy) (norm_nonneg _) ((norm_nonneg _).trans (hO h)))
  calc ‖∑ h, O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j)))‖
      ≤ ∑ h, ‖O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j)))‖ :=
        norm_sum_le _ _
    _ ≤ ∑ _h : Idx H, K * (A * N) := Finset.sum_le_sum fun h _ => hhead h
    _ = H * (K * (A * N)) := by simp

/-- **Near a cluster an exclusive head is small by itself:** if the tokens are
pairwise within `ε`, then `‖Σ_h O_h P^⊥ (Σ_j a_{hij} U_h x_j)‖ ≤ H K A N ε` —
attention contributes nothing to the drive at first order, whatever the rows. -/
theorem norm_xsaAttn_le_of_close {e H : ℕ} {a : ℝ → Idx H → Idx n → Idx n → ℝ}
    {U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e}
    {O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d} {X : ℝ → SphereTuple d n} {t A N K ε : ℝ}
    (habs : ∀ h i, ∑ j, |a t h i j| ≤ A) (hU : ∀ h, ‖U t h‖ ≤ N) (hO : ∀ h, ‖O t h‖ ≤ K)
    (hε : 0 ≤ ε) (hX : ∀ i j, ‖(X t i : EucSpace d) - X t j‖ ≤ ε) (i : Idx n) :
    ‖∑ h, O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j)))‖ ≤
      H * (K * (A * (N * ε))) := by
  have hhead : ∀ h, ‖O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j)))‖ ≤
      K * (A * (N * ε)) := fun h => by
    have hN : 0 ≤ N := (norm_nonneg _).trans (hU h)
    have hy : ‖xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j))‖ ≤ A * (N * ε) :=
      norm_xsaProj_sum_le (U t h (X t i)) (habs h i) (by positivity) fun j => by
        rw [← map_sub]
        exact (ContinuousLinearMap.le_opNorm _ _).trans
          (mul_le_mul (hU h) (hX j i) (norm_nonneg _) hN)
    exact (ContinuousLinearMap.le_opNorm _ _).trans
      (mul_le_mul (hO h) hy (norm_nonneg _) ((norm_nonneg _).trans (hO h)))
  calc ‖∑ h, O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j)))‖
      ≤ ∑ h, ‖O t h (xsaProj (U t h (X t i)) (∑ j, a t h i j • U t h (X t j)))‖ :=
        norm_sum_le _ _
    _ ≤ ∑ _h : Idx H, K * (A * (N * ε)) := Finset.sum_le_sum fun h _ => hhead h
    _ = H * (K * (A * (N * ε))) := by simp

/-- **The drive of an exclusive block is bounded:** `‖v_i(t)‖ ≤ H K A N + B + Z`. -/
theorem norm_xsaDrive_le {e H : ℕ} {a : ℝ → Idx H → Idx n → Idx n → ℝ}
    {U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e}
    {O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d} {G : ℝ → EucSpace d → EucSpace d}
    {z : ℝ → Idx n → EucSpace d} (X : ℝ → SphereTuple d n) {t A N K B Z : ℝ}
    (habs : ∀ h i, ∑ j, |a t h i j| ≤ A) (hU : ∀ h, ‖U t h‖ ≤ N) (hO : ∀ h, ‖O t h‖ ≤ K)
    (hG : ∀ x : SSphere d, ‖G t x‖ ≤ B) (hz : ∀ i, ‖z t i‖ ≤ Z) (i : Idx n) :
    ‖xsaDrive a U O G z X t i‖ ≤ H * (K * (A * N)) + B + Z :=
  norm_add₃_le.trans
    (add_le_add (add_le_add (norm_xsaAttn_le X habs hU hO i) (hG _)) (hz i))

/-- **What the tokens share cancels, and each head with it:** if the tokens are
pairwise within `ε`, then
`‖v_k(t) - v_l(t) - (z_k(t) - z_l(t))‖ ≤ (2 H K A N + L_G) ε`. -/
theorem norm_xsaDrive_sub_sub_le {e H : ℕ} {a : ℝ → Idx H → Idx n → Idx n → ℝ}
    {U : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace e}
    {O : ℝ → Idx H → EucSpace e →L[ℝ] EucSpace d} {G : ℝ → EucSpace d → EucSpace d}
    {z : ℝ → Idx n → EucSpace d} {X : ℝ → SphereTuple d n} {t A N K LG ε : ℝ}
    (habs : ∀ h i, ∑ j, |a t h i j| ≤ A) (hU : ∀ h, ‖U t h‖ ≤ N) (hO : ∀ h, ‖O t h‖ ≤ K)
    (hG : ∀ x y : SSphere d, ‖G t x - G t y‖ ≤ LG * ‖(x : EucSpace d) - y‖) (hLG : 0 ≤ LG)
    (hε : 0 ≤ ε) (hX : ∀ i j, ‖(X t i : EucSpace d) - X t j‖ ≤ ε) (k l : Idx n) :
    ‖xsaDrive a U O G z X t k - xsaDrive a U O G z X t l - (z t k - z t l)‖ ≤
      (2 * H * K * A * N + LG) * ε := by
  have hsplit : xsaDrive a U O G z X t k - xsaDrive a U O G z X t l - (z t k - z t l) =
      (∑ h, O t h (xsaProj (U t h (X t k)) (∑ j, a t h k j • U t h (X t j)))
        - ∑ h, O t h (xsaProj (U t h (X t l)) (∑ j, a t h l j • U t h (X t j))))
        + (G t (X t k) - G t (X t l)) := by
    simp only [xsaDrive]
    abel
  have hGkl := (hG (X t k) (X t l)).trans (mul_le_mul_of_nonneg_left (hX k l) hLG)
  rw [hsplit]
  calc _ ≤ _ := norm_add_le _ _
    _ ≤ H * (K * (A * (N * ε))) + H * (K * (A * (N * ε))) + LG * ε :=
        add_le_add (norm_sub_le_of_le (norm_xsaAttn_le_of_close habs hU hO hε hX k)
          (norm_xsaAttn_le_of_close habs hU hO hε hX l)) hGkl
    _ = (2 * H * K * A * N + LG) * ε := by ring

/-- The hypotheses of `norm_xsaAttn_le`, `norm_xsaAttn_le_of_close`,
`norm_xsaDrive_le` and `norm_xsaDrive_sub_sub_le` are satisfiable: one head of
weight `1` on one token, values and output maps `0`, no feed-forward term, no
injection, `A = 1` and `N = K = B = Z = L_G = ε = 0`. -/
example : ∃ (a : ℝ → Idx 1 → Idx 1 → Idx 1 → ℝ) (U : ℝ → Idx 1 → EucSpace 1 →L[ℝ] EucSpace 1)
    (O : ℝ → Idx 1 → EucSpace 1 →L[ℝ] EucSpace 1) (G : ℝ → EucSpace 1 → EucSpace 1)
    (z : ℝ → Idx 1 → EucSpace 1) (X : ℝ → SphereTuple 1 1),
    (∀ h i, ∑ j, |a 0 h i j| ≤ 1) ∧ (∀ h, ‖U 0 h‖ ≤ 0) ∧ (∀ h, ‖O 0 h‖ ≤ 0) ∧
    (∀ x : SSphere 1, ‖G 0 x‖ ≤ 0) ∧
    (∀ x y : SSphere 1, ‖G 0 x - G 0 y‖ ≤ 0 * ‖(x : EucSpace 1) - y‖) ∧
    (∀ i, ‖z 0 i‖ ≤ 0) ∧ ∀ i j, ‖(X 0 i : EucSpace 1) - X 0 j‖ ≤ 0 :=
  ⟨fun _ _ _ _ => 1, fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => 0,
    fun _ _ => basePoint 0, by simp⟩

end Perspective
end Transformer
