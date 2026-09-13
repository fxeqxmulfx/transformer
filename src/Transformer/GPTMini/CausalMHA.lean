/-
# Causal Multi-Head Attention with QK-norm and XSA

Formalization of the `CausalMHA.forward` method from `reference/model.py`:

```python
q, k, v = self.qkv(x).chunk(3, dim=-1)
q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
q = F.normalize(q, dim=-1, eps=1e-6)
k = F.normalize(k, dim=-1, eps=1e-6)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
alpha  = self.log_alpha.exp().view(1, n_heads, 1, 1)
scores = (q @ k.transpose(-2, -1)) * alpha
mask   = triu(full((T, T), -inf), diagonal=1)
attn   = softmax(scores + mask, dim=-1)
y      = attn @ v
# XSA
v_hat  = F.normalize(v, dim=-1, eps=1e-6)
z      = y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat
return self.proj(z.transpose(1, 2).reshape(B, T, D))
```

We formalize the forward pass with three components:
  1. **Causal-softmax weights** (with QK-normalized, RoPE-rotated scores)
  2. **Attention output** `y_i = Σ_{j ≤ i} a_{i,j} v_j`
  3. **XSA projection** `z_i = y_i - ⟨y_i, v̂_i⟩ · v̂_i`
followed by the output linear projection `W_o`.
-/

import Transformer.Basic
import Transformer.GPTMini.Config
import Transformer.GPTMini.QKNorm
import Transformer.GPTMini.RoPE
import Mathlib.Analysis.SpecialFunctions.Exp

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-! ### Attention weights -/

/-- **Pre-softmax causal score** for head `h`, query position `i`, key
position `j`, with QK-norm + RoPE-rotated `q, k`.

For `j > i` (causal masked), the score is `-∞` (encoded as
`Option.none` ↦ excluded from softmax).  We formalize this by restricting
the sum domain to `{j : Fin (i.1 + 1)}` in `softmax`.
-/
noncomputable def preScore
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) : ℝ :=
  score alpha eps (q i) (k j)

/-- **Causal-softmax weights.**

For position `i`, `j ≤ i`, weight is

  `a_{i,j}^{(h)} = exp(score_{i,j}) / Σ_{j' ≤ i} exp(score_{i,j'})`,

and zero for `j > i`. -/
noncomputable def causalAttnWeights
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) : ℝ :=
  if (j : ℕ) > (i : ℕ) then 0
  else
    Real.exp (preScore cfg alpha eps q k i j)
      /
    (∑ j' : Fin T,
        if (j' : ℕ) ≤ (i : ℕ) then
          Real.exp (preScore cfg alpha eps q k i j')
        else 0)

/-- **Row-stochastic property** of the causal attention matrix:

  `Σ_j a_{i,j} = 1`  for every `i`. -/
theorem causalAttnWeights_row_sum
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i : Fin T) :
    (∑ j : Fin T, causalAttnWeights cfg alpha eps q k i j) = 1 := by
  -- Denote the denominator by S; each weight is exp(score)/S for j ≤ i
  -- and 0 otherwise.  Then Σ = (Σ_{j≤i} exp) / S = S / S = 1.
  unfold causalAttnWeights
  -- Distribute the denominator out: each summand has the form
  --   (if j > i then 0 else exp(score)) / S
  rw [show (∑ j : Fin T,
            if (j : ℕ) > (i : ℕ) then (0 : ℝ)
            else Real.exp (preScore cfg alpha eps q k i j) /
                  ∑ j' : Fin T,
                    if (j' : ℕ) ≤ (i : ℕ) then
                      Real.exp (preScore cfg alpha eps q k i j')
                    else 0)
        =
       (∑ j : Fin T,
            if (j : ℕ) ≤ (i : ℕ) then
              Real.exp (preScore cfg alpha eps q k i j)
            else 0)
        /
       (∑ j' : Fin T,
            if (j' : ℕ) ≤ (i : ℕ) then
              Real.exp (preScore cfg alpha eps q k i j')
            else 0)
        from ?_]
  · -- Now `S / S = 1`
    have hS_pos : 0 < ∑ j' : Fin T,
                if (j' : ℕ) ≤ (i : ℕ) then
                  Real.exp (preScore cfg alpha eps q k i j')
                else 0 := by
      apply Finset.sum_pos'
      · intros j _
        split_ifs
        · exact le_of_lt (Real.exp_pos _)
        · exact le_refl 0
      · exact ⟨i, Finset.mem_univ _, by
          simp only [le_refl, if_true]; exact Real.exp_pos _⟩
    exact div_self (ne_of_gt hS_pos)
  · -- Show the rewrite: divide-by-S commutes with the conditional
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intros j _
    by_cases hij : (j : ℕ) > (i : ℕ)
    · simp [hij, not_le.mpr hij]
    · have hij_le : (j : ℕ) ≤ (i : ℕ) := not_lt.mp hij
      simp [hij, hij_le]

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
  sorry

/-! ### Attention output and XSA -/

/-- **Pre-XSA attention output** (a single head):

  `y_i = Σ_{j ≤ i} a_{i,j} · v_j`. -/
noncomputable def attnOutput
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (v : Fin T → EucSpace cfg.head_dim)
    (i : Fin T) : EucSpace cfg.head_dim :=
  ∑ j : Fin T, (causalAttnWeights cfg alpha eps q k i j) • v j

/-- **XSA projection.**  After attention, project the output onto the
orthogonal complement of `v_i`:

  `z_i = y_i - ⟨y_i, v̂_i⟩ · v̂_i`,    `v̂_i = v_i / ‖v_i‖`. -/
noncomputable def xsaProjection
    {T : ℕ} (eps : ℝ) (v : Fin T → EucSpace cfg.head_dim)
    (y : Fin T → EucSpace cfg.head_dim) (i : Fin T) : EucSpace cfg.head_dim :=
  let v_hat := normL2 eps (v i)
  y i - (inner (𝕜 := ℝ) (y i) v_hat) • v_hat

/-- **XSA orthogonality (when `v_i` is exactly unit-norm).**

If the normalized self-value `v̂_i := normL2 eps (v i)` satisfies
`‖v̂_i‖² = 1`, then the XSA-projected output is orthogonal to `v̂_i`:

  `⟨z_i, v̂_i⟩ = ⟨y_i, v̂_i⟩ - ⟨y_i, v̂_i⟩ · ‖v̂_i‖² = 0`.

(The hypothesis `‖v̂_i‖² = 1` holds exactly when `eps = 0` and `v_i ≠ 0`;
with `eps > 0` it holds only approximately — the projection is then
approximately orthogonal, with error of order `eps`.) -/
theorem xsaProjection_orthogonal
    {T : ℕ} (eps : ℝ)
    (v y : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (h_unit : ‖normL2 eps (v i)‖^2 = 1) :
    inner (𝕜 := ℝ) (xsaProjection cfg eps v y i) (normL2 eps (v i)) = 0 := by
  unfold xsaProjection
  simp only []
  rw [inner_sub_left, real_inner_smul_left]
  have hsq : inner (𝕜 := ℝ) (normL2 eps (v i)) (normL2 eps (v i))
              = ‖normL2 eps (v i)‖^2 := real_inner_self_eq_norm_sq _
  rw [hsq, h_unit, mul_one, sub_self]

/-! ### Full head -/

/-- **One head of `CausalMHA`** — combines QK-norm, RoPE-rotated `q, k`,
causal attention weights, and XSA. -/
noncomputable def attentionHead
    {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim)
    (positions : Fin T → ℝ)
    (i : Fin T) : EucSpace cfg.head_dim :=
  let q_rope := fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j)
  let k_rope := fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)
  let y := attnOutput cfg alpha eps q_rope k_rope v i
  xsaProjection cfg eps v (fun i' => if i' = i then y else 0) i

end GPTMini
end Transformer
