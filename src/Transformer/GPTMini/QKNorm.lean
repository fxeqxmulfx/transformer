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
  - `q̂ = q / max(‖q‖, eps)`, `k̂ = k / max(‖k‖, eps)`  (per (batch, head,
    position)), which is `F.normalize`: the unit vector once `‖q‖ ≥ eps`
  - score(i, j) = ⟨q̂_i, k̂_j⟩ · e^{α_h}     ∈  [-e^{α_h}, e^{α_h}]

The partition function is therefore between `n e^{-e^{α_h}}` and
`n e^{e^{α_h}}` (`partition_bounds`), and after softmax across `j` the
attention weights satisfy the two-sided bounds

  `n⁻¹ · e^{-2 e^{α_h}} ≤ a_{ij}^{(h)} ≤ n⁻¹ · e^{2 e^{α_h}}`

(`GPTMini.AttentionBounds`, for the causal row `i`, where `n = i + 1`), the
analogue of the partition-function bounds of
§2.2 in 2312.10794v5 with `e^{α_h}` in place of `β`.
-/

import Transformer.Basic
import Transformer.GPTMini.Config
import Transformer.GPTMini.RMSNorm
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable {d_head : ℕ}

/-- L2 normalization on `EucSpace d_head` with `eps` for numerical stability,
as `torch.nn.functional.normalize` computes it:

  `normL2(x) = x / max(‖x‖, eps)`.

So `normL2 eps x` is the unit vector `x / ‖x‖` whenever `‖x‖ ≥ eps`, and
`x / eps` below; at `eps = 0` it is `x / ‖x‖`, and `0` at `x = 0`. -/
noncomputable def normL2 (eps : ℝ) (x : EucSpace d_head) : EucSpace d_head :=
  (1 / max ‖x‖ eps) • x

/-- **Almost-unit-norm property.**

  `‖normL2 eps x‖ ≤ 1`  always.

In the limit `eps → 0`, `‖normL2 0 x‖ = 1` if `x ≠ 0`. -/
theorem normL2_norm_le
    (eps : ℝ) (heps : 0 ≤ eps) (x : EucSpace d_head) :
    ‖normL2 eps x‖ ≤ 1 := by
  unfold normL2
  rw [norm_smul]
  have hpos : 0 ≤ max ‖x‖ eps := le_max_of_le_right heps
  rw [Real.norm_eq_abs, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / max ‖x‖ eps)]
  by_cases hx : ‖x‖ = 0
  · rw [hx]
    simp
  · have hxpos : 0 < ‖x‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hx)
    have : 0 < max ‖x‖ eps := lt_max_of_lt_left hxpos
    rw [div_mul_eq_mul_div, one_mul, div_le_one this]
    exact le_max_left _ _

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
  have hb : ∀ j : Fin n, |score alpha eps (q i) (k j)| ≤ Real.exp alpha :=
    fun j => score_bounded alpha eps heps _ _
  constructor
  · have h1 : ∀ j ∈ Finset.univ, Real.exp (-(Real.exp alpha))
        ≤ Real.exp (score alpha eps (q i) (k j)) := by
      intro j _
      refine Real.exp_le_exp.mpr ?_
      linarith [(abs_le.mp (hb j)).1]
    simpa [Finset.card_univ, nsmul_eq_mul] using
      Finset.card_nsmul_le_sum Finset.univ _ _ h1
  · have h2 : ∀ j ∈ Finset.univ, Real.exp (score alpha eps (q i) (k j))
        ≤ Real.exp (Real.exp alpha) := by
      intro j _
      refine Real.exp_le_exp.mpr ?_
      linarith [(abs_le.mp (hb j)).2]
    simpa [Finset.card_univ, nsmul_eq_mul] using
      Finset.sum_le_card_nsmul Finset.univ _ _ h2

/-- The hypothesis of the three theorems above is satisfiable, and by the
value the implementation uses: `eps = 1e-6` in `CausalMHA.forward`. -/
example : (0 : ℝ) ≤ 1e-6 := by norm_num

/-! ### The RMS parameterization is the same head

`reference/model.py` normalizes `q, k` by their Euclidean norm; the record
stacks of `openai/parameter-golf` (`train_gpt.py`, 2026-09-13) normalize them
by their **RMS** norm, `x / √(mean x²)`, and carry a learned per-head gain.
The two differ by the constant `√d_head`, which the gain absorbs — so the
bounds above cover both parameterizations, and that is a theorem rather than a
remark.  `rmsNorm` is the one already formalized in
`Transformer.GPTMini.RMSNorm`, the block normalization of `reference/model.py`;
the point is that the same map, applied per head to `q` and `k`, is the QK-norm
of this file up to that constant. -/

/-- The score of the RMS parameterization, with its own per-head gain: the
head of `train_gpt.py`, where `q, k` are RMS-normalized rather than
L2-normalized before the dot product. -/
noncomputable def rmsScore (alpha : ℝ) (q k : EucSpace d_head) : ℝ :=
  Real.exp alpha * inner (𝕜 := ℝ) (rmsNorm q) (rmsNorm k)

/-- **The two normalizations differ by `√d_head` and nothing else.**  At `x = 0`
both sides are `0`, and elsewhere the RMS norm is `‖x‖ / √d_head`, so dividing
by it is dividing by `‖x‖` and multiplying by `√d_head`.

Source: `reference/model.py` (`RMSNorm`, `CausalMHA.forward`) against
`openai/parameter-golf`, `train_gpt.py` (`CausalSelfAttention`). -/
theorem rmsNorm_eq_smul_normL2 (x : EucSpace d_head) :
    rmsNorm x = Real.sqrt d_head • normL2 0 x := by
  unfold rmsNorm normL2
  rw [max_eq_left (norm_nonneg x)]
  by_cases hx : ‖x‖ = 0
  · rw [ite_eq_left hx, norm_eq_zero.mp hx, smul_zero, smul_zero]
  · rw [ite_eq_right hx, smul_smul, mul_one_div]

/-- **And the gain absorbs it.**  An RMS-normalized head with gain `α` is the
L2-normalized head with gain `α + log d_head`: the same operator at a shifted
parameter, so `score_bounded` and `partition_bounds` apply to it verbatim, and
the formalized head covers both parameterizations.

Source: `openai/parameter-golf`, `train_gpt.py` (`CausalSelfAttention`), whose
per-head gain multiplies `q` rather than the score — the same thing. -/
theorem rmsScore_eq_score (hd : 0 < d_head) (alpha : ℝ) (q k : EucSpace d_head) :
    rmsScore alpha q k = score (alpha + Real.log d_head) 0 q k := by
  have hdpos : (0 : ℝ) < (d_head : ℝ) := Nat.cast_pos.mpr hd
  have hsq : Real.sqrt d_head * Real.sqrt d_head = (d_head : ℝ) :=
    Real.mul_self_sqrt hdpos.le
  unfold rmsScore score
  rw [rmsNorm_eq_smul_normL2, rmsNorm_eq_smul_normL2, real_inner_smul_left,
    real_inner_smul_right, Real.exp_add, Real.exp_log hdpos,
    ← mul_assoc (Real.sqrt d_head) (Real.sqrt d_head), hsq]
  ring

/-- The hypothesis is satisfiable, and at the head dimension the reference
implementation uses: `d_head = 64`. -/
example : 0 < 64 := by norm_num

end GPTMini
end Transformer
