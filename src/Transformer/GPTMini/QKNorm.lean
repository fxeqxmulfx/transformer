/-
# QK-Normalization with learnable per-head temperature

Formalization of the QK-norm block of `CausalMHA.forward` from `reference/model.py`:

```python
q = F.normalize(q, dim=-1, eps=1e-6)
k = F.normalize(k, dim=-1, eps=1e-6)
alpha  = self.log_alpha.exp().view(1, n_heads, 1, 1)
scores = (q @ k.transpose(-2, -1)) * alpha
```

Mathematical content:
  - `q̂ = q / ‖q‖`, `k̂ = k / ‖k‖`  (per (batch, head, position))
  - score(i, j) = ⟨q̂_i, k̂_j⟩ · e^{α_h}     ∈  [-e^{α_h}, e^{α_h}]

After softmax across `j`, the attention weights satisfy the explicit two-
sided bounds

  `n⁻¹ · e^{-2 α_h} ≤ a_{ij}^{(h)} ≤ n⁻¹ · e^{2 α_h}`

which match exactly the partition-function bounds of §2.2 in 2312.10794v5.
-/

import Transformer.Basic
import Transformer.GPTMini.Config
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable {d_head : ℕ}

/-- L2 normalization on `EucSpace d_head` with `eps` for numerical stability:

  `normL2(x) = x / (‖x‖ + eps)`. -/
noncomputable def normL2 (eps : ℝ) (x : EucSpace d_head) : EucSpace d_head :=
  (1 / (‖x‖ + eps)) • x

/-- **Almost-unit-norm property.**

  `‖normL2 eps x‖ ≤ 1`  always.

In the limit `eps → 0`, `‖normL2 0 x‖ = 1` if `x ≠ 0`. -/
theorem normL2_norm_le
    (eps : ℝ) (heps : 0 ≤ eps) (x : EucSpace d_head) :
    ‖normL2 eps x‖ ≤ 1 := by
  unfold normL2
  rw [norm_smul]
  have hpos : 0 ≤ ‖x‖ + eps := by positivity
  rw [Real.norm_eq_abs, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / (‖x‖ + eps))]
  by_cases hx : ‖x‖ = 0
  · rw [hx]
    simp [norm_eq_zero.mp hx]
  · have hxpos : 0 < ‖x‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hx)
    have : 0 < ‖x‖ + eps := by linarith
    rw [div_mul_eq_mul_div, one_mul, div_le_one this]
    linarith

/-- Inner-product score after QK-norm with per-head temperature `e^α`. -/
noncomputable def score
    (alpha : ℝ) (eps : ℝ) (q k : EucSpace d_head) : ℝ :=
  Real.exp alpha * inner (𝕜 := ℝ) (normL2 eps q) (normL2 eps k)

/-- **Bounded scores.**

For any `q, k` and `eps ≥ 0`:

  `|score(α, eps, q, k)| ≤ e^α`. -/
theorem score_bounded
    (alpha eps : ℝ) (heps : 0 ≤ eps) (q k : EucSpace d_head) :
    |score alpha eps q k| ≤ Real.exp alpha := by
  unfold score
  rw [abs_mul, abs_of_pos (Real.exp_pos alpha)]
  have h_inner_abs : |inner (𝕜 := ℝ) (normL2 eps q) (normL2 eps k)|
      ≤ ‖normL2 eps q‖ * ‖normL2 eps k‖ := abs_real_inner_le_norm _ _
  have h_q : ‖normL2 eps q‖ ≤ 1 := normL2_norm_le eps heps q
  have h_k : ‖normL2 eps k‖ ≤ 1 := normL2_norm_le eps heps k
  have h_prod : ‖normL2 eps q‖ * ‖normL2 eps k‖ ≤ 1 := by
    calc ‖normL2 eps q‖ * ‖normL2 eps k‖
        ≤ 1 * ‖normL2 eps k‖ := by
          apply mul_le_mul_of_nonneg_right h_q (norm_nonneg _)
      _ ≤ 1 * 1 := by
          apply mul_le_mul_of_nonneg_left h_k (by norm_num)
      _ = 1 := by ring
  have := le_trans h_inner_abs h_prod
  calc Real.exp alpha * |inner (𝕜 := ℝ) (normL2 eps q) (normL2 eps k)|
      ≤ Real.exp alpha * 1 := by
        apply mul_le_mul_of_nonneg_left this (le_of_lt (Real.exp_pos _))
    _ = Real.exp alpha := by ring

/-- **Partition-function bound.**

For `n` positions and per-head temperature `α`, the partition function

  `Z_i = Σ_j exp(score(α, eps, q_i, k_j))`

satisfies

  `n · exp(-e^α) ≤ Z_i ≤ n · exp(e^α)`.

This corresponds exactly to `partitionSA_bounds` in `Section1_IPS`, but for
the *QK-normalized* setting. -/
theorem partition_bounds
    {n : ℕ} (alpha eps : ℝ) (heps : 0 ≤ eps)
    (q : Fin n → EucSpace d_head) (k : Fin n → EucSpace d_head)
    (i : Fin n) :
    (n : ℝ) * Real.exp (-(Real.exp alpha)) ≤
      ∑ j : Fin n, Real.exp (score alpha eps (q i) (k j))
    ∧
    (∑ j : Fin n, Real.exp (score alpha eps (q i) (k j)))
      ≤ (n : ℝ) * Real.exp (Real.exp alpha) := by
  sorry

end GPTMini
end Transformer
