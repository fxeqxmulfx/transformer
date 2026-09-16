/-
# How large an attention head can get

Everything the Pre-LN growth estimate needs about one head of `CausalMHA`,
in the order the head computes it:

  1. the causal weights are non-negative and vanish past the cutoff;
  2. they are within `e^{±2e^α}` of the uniform weight `(i+1)⁻¹`, which is
     the quantitative content of QK-normalization — the scores live in
     `[-e^α, e^α]` no matter how large `q` and `k` are;
  3. the attention output is a convex combination of the values, so it is
     bounded by any uniform bound on them;
  4. the XSA projection subtracts a vector no longer than the input, so it
     at most doubles the norm.

Together: a head is bounded by twice a uniform bound on its values, with no
dependence on the residual stream.
-/

import Transformer.GPTMini.CausalMHA

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-! ### The weights -/

/-- **Attention weights are zero past the causal cutoff.**

For positions `j > i`, the causal attention weight is `0` by construction. -/
theorem causalAttnWeights_zero_above
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) (hij : (i : ℕ) < (j : ℕ)) :
    causalAttnWeights cfg alpha eps q k i j = 0 := by
  unfold causalAttnWeights
  simp [hij]

/-- **Attention weights are non-negative.**

Direct consequence of `Real.exp_pos` and the softmax-style construction. -/
theorem causalAttnWeights_nonneg
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) :
    0 ≤ causalAttnWeights cfg alpha eps q k i j := by
  unfold causalAttnWeights
  split_ifs with hij
  · exact le_refl 0
  · apply div_nonneg
    · exact le_of_lt (Real.exp_pos _)
    · apply Finset.sum_nonneg
      intros j' _
      split_ifs with h
      · exact le_of_lt (Real.exp_pos _)
      · exact le_refl 0

/-- **Two-sided bound on attention weights** (consequence of `QKNorm`):

  `(i+1)⁻¹ · e^{-2 e^α} ≤ a_{i,j}^{(h)} ≤ (i+1)⁻¹ · e^{2 e^α}`

for `j ≤ i`, where `(i + 1)` is the number of unmasked positions. -/
theorem causalAttnWeights_bounds
    {T : ℕ} (alpha eps : ℝ) (heps : 0 ≤ eps)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) (hij : (j : ℕ) ≤ (i : ℕ)) :
    (((i : ℕ) + 1 : ℝ))⁻¹ * Real.exp (-(2 * Real.exp alpha))
      ≤ causalAttnWeights cfg alpha eps q k i j
    ∧
    causalAttnWeights cfg alpha eps q k i j
      ≤ (((i : ℕ) + 1 : ℝ))⁻¹ * Real.exp (2 * Real.exp alpha) := by
  classical
  have hscore : ∀ j' : Fin T, |preScore cfg alpha eps q k i j'| ≤ Real.exp alpha :=
    fun j' => score_bounded alpha eps heps (q i) (k j')
  have hlo : ∀ j' : Fin T,
      Real.exp (-Real.exp alpha) ≤ Real.exp (preScore cfg alpha eps q k i j') :=
    fun j' => Real.exp_le_exp.mpr (neg_le_of_abs_le (hscore j'))
  have hhi : ∀ j' : Fin T,
      Real.exp (preScore cfg alpha eps q k i j') ≤ Real.exp (Real.exp alpha) :=
    fun j' => Real.exp_le_exp.mpr (le_of_abs_le (hscore j'))
  have hfil : (Finset.univ.filter fun j' : Fin T => (j' : ℕ) ≤ (i : ℕ)) = Finset.Iic i := by
    ext j'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Iic]
    exact Fin.le_def.symm
  have hcard : (((Finset.Iic i).card : ℕ) : ℝ) = ((i : ℕ) : ℝ) + 1 := by
    rw [Fin.card_Iic]; push_cast; ring
  set S := ∑ j' : Fin T,
      if (j' : ℕ) ≤ (i : ℕ) then Real.exp (preScore cfg alpha eps q k i j') else 0 with hSdef
  have hSsum : S = ∑ j' ∈ Finset.Iic i, Real.exp (preScore cfg alpha eps q k i j') := by
    rw [hSdef, ← Finset.sum_filter, hfil]
  have hSlo : (((i : ℕ) : ℝ) + 1) * Real.exp (-Real.exp alpha) ≤ S := by
    rw [hSsum]
    have h := Finset.card_nsmul_le_sum (Finset.Iic i)
      (fun j' => Real.exp (preScore cfg alpha eps q k i j'))
      (Real.exp (-Real.exp alpha)) fun j' _ => hlo j'
    rwa [nsmul_eq_mul, hcard] at h
  have hShi : S ≤ (((i : ℕ) : ℝ) + 1) * Real.exp (Real.exp alpha) := by
    rw [hSsum]
    have h := Finset.sum_le_card_nsmul (Finset.Iic i)
      (fun j' => Real.exp (preScore cfg alpha eps q k i j'))
      (Real.exp (Real.exp alpha)) fun j' _ => hhi j'
    rwa [nsmul_eq_mul, hcard] at h
  have hi1 : (0 : ℝ) < ((i : ℕ) : ℝ) + 1 := by positivity
  have hSpos : 0 < S := lt_of_lt_of_le (by positivity) hSlo
  have hw : causalAttnWeights cfg alpha eps q k i j
      = Real.exp (preScore cfg alpha eps q k i j) / S := by
    rw [causalAttnWeights, ite_eq_right (not_lt.mpr hij)]
  constructor
  · rw [hw, le_div_iff₀ hSpos]
    calc (((i : ℕ) : ℝ) + 1)⁻¹ * Real.exp (-(2 * Real.exp alpha)) * S
        ≤ (((i : ℕ) : ℝ) + 1)⁻¹ * Real.exp (-(2 * Real.exp alpha)) *
            ((((i : ℕ) : ℝ) + 1) * Real.exp (Real.exp alpha)) :=
          mul_le_mul_of_nonneg_left hShi (by positivity)
      _ = Real.exp (-(2 * Real.exp alpha)) * Real.exp (Real.exp alpha) *
            ((((i : ℕ) : ℝ) + 1)⁻¹ * (((i : ℕ) : ℝ) + 1)) := by ring
      _ = Real.exp (-Real.exp alpha) := by
          rw [inv_mul_cancel₀ (ne_of_gt hi1), mul_one, ← Real.exp_add]
          congr 1
          ring
      _ ≤ Real.exp (preScore cfg alpha eps q k i j) := hlo j
  · rw [hw, div_le_iff₀ hSpos]
    calc Real.exp (preScore cfg alpha eps q k i j) ≤ Real.exp (Real.exp alpha) := hhi j
      _ = Real.exp (2 * Real.exp alpha) * Real.exp (-Real.exp alpha) *
            ((((i : ℕ) : ℝ) + 1)⁻¹ * (((i : ℕ) : ℝ) + 1)) := by
          rw [inv_mul_cancel₀ (ne_of_gt hi1), mul_one, ← Real.exp_add]
          congr 1
          ring
      _ = (((i : ℕ) : ℝ) + 1)⁻¹ * Real.exp (2 * Real.exp alpha) *
            ((((i : ℕ) : ℝ) + 1) * Real.exp (-Real.exp alpha)) := by ring
      _ ≤ (((i : ℕ) : ℝ) + 1)⁻¹ * Real.exp (2 * Real.exp alpha) * S :=
          mul_le_mul_of_nonneg_left hSlo (by positivity)

/-! ### The head -/

/-- **The attention output is bounded by any uniform bound on the values.**

`y_i = Σ_j a_{i,j} v_j` is a convex combination — the weights are
non-negative by `causalAttnWeights_nonneg` and sum to `1` by
`causalAttnWeights_row_sum`. -/
theorem attnOutput_norm_le
    {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (i : Fin T) (B : ℝ)
    (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖attnOutput cfg alpha eps q k v i‖ ≤ B := by
  calc ‖attnOutput cfg alpha eps q k v i‖
      ≤ ∑ j : Fin T, ‖causalAttnWeights cfg alpha eps q k i j • v j‖ := by
        rw [attnOutput]; exact norm_sum_le _ _
    _ = ∑ j : Fin T, causalAttnWeights cfg alpha eps q k i j * ‖v j‖ := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [norm_smul, Real.norm_eq_abs,
          abs_of_nonneg (causalAttnWeights_nonneg cfg alpha eps q k i j)]
    _ ≤ ∑ j : Fin T, causalAttnWeights cfg alpha eps q k i j * B :=
        Finset.sum_le_sum fun j _ =>
          mul_le_mul_of_nonneg_left (hB j) (causalAttnWeights_nonneg cfg alpha eps q k i j)
    _ = B := by
        rw [← Finset.sum_mul, causalAttnWeights_row_sum, one_mul]

/-- **XSA at most doubles the norm.**

`z_i = y_i - ⟨y_i, v̂_i⟩ v̂_i` and `‖v̂_i‖ ≤ 1`, so the subtracted vector is
no longer than `y_i`. -/
theorem xsaProjection_norm_le
    {T : ℕ} (eps : ℝ) (heps : 0 ≤ eps)
    (v y : Fin T → EucSpace cfg.head_dim) (i : Fin T) :
    ‖xsaProjection cfg eps v y i‖ ≤ 2 * ‖y i‖ := by
  have hunit : ‖normL2 eps (v i)‖ ≤ 1 := normL2_norm_le eps heps (v i)
  have hx : xsaProjection cfg eps v y i
      = y i - (inner (𝕜 := ℝ) (y i) (normL2 eps (v i))) • normL2 eps (v i) := rfl
  have hproj : ‖(inner (𝕜 := ℝ) (y i) (normL2 eps (v i))) • normL2 eps (v i)‖ ≤ ‖y i‖ := by
    rw [norm_smul, Real.norm_eq_abs]
    calc |inner (𝕜 := ℝ) (y i) (normL2 eps (v i))| * ‖normL2 eps (v i)‖
        ≤ (‖y i‖ * ‖normL2 eps (v i)‖) * ‖normL2 eps (v i)‖ :=
          mul_le_mul_of_nonneg_right (abs_real_inner_le_norm _ _) (norm_nonneg _)
      _ ≤ (‖y i‖ * 1) * 1 := by gcongr
      _ = ‖y i‖ := by ring
  calc ‖xsaProjection cfg eps v y i‖
      ≤ ‖y i‖ + ‖(inner (𝕜 := ℝ) (y i) (normL2 eps (v i))) • normL2 eps (v i)‖ := by
        rw [hx]; exact norm_sub_le _ _
    _ ≤ ‖y i‖ + ‖y i‖ := by gcongr
    _ = 2 * ‖y i‖ := by ring

/-- **One head is bounded by twice a uniform bound on its values** — and by
nothing else: the bound does not see `q`, `k`, the positions, or the size of
the residual stream. -/
theorem attentionHead_norm_le
    {T : ℕ} (alpha eps : ℝ) (heps : 0 ≤ eps)
    (q k v : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ)
    (i : Fin T) (B : ℝ) (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖attentionHead cfg alpha eps q k v positions i‖ ≤ 2 * B := by
  classical
  set y := attnOutput cfg alpha eps
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j))
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) v i with hy
  have heq : attentionHead cfg alpha eps q k v positions i
      = xsaProjection cfg eps v (fun i' => if i' = i then y else 0) i := rfl
  have hyB : ‖y‖ ≤ B := by
    rw [hy]; exact attnOutput_norm_le cfg _ _ _ _ _ _ B hB
  have hval : ‖(fun i' : Fin T => if i' = i then y else 0) i‖ = ‖y‖ := by simp
  rw [heq]
  calc ‖xsaProjection cfg eps v (fun i' => if i' = i then y else 0) i‖
      ≤ 2 * ‖(fun i' : Fin T => if i' = i then y else 0) i‖ :=
        xsaProjection_norm_le cfg eps heps v _ i
    _ = 2 * ‖y‖ := by rw [hval]
    _ ≤ 2 * B := by linarith

end GPTMini
end Transformer
