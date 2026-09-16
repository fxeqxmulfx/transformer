/-
# How far one attention head can move

The head of `reference/model.py` (`CausalMHA.forward`) is a composition of
three maps, and each one is estimated separately here:

  1. `attnOutput` is `Σ_j a_{i,j} v_j`; splitting the difference of two such
     sums as `Σ (a - a') v'` plus `Σ a (v - v')` bounds it by the ℓ¹ movement
     of the weights times a bound on the values, plus a uniform bound on the
     movement of the values (`attnOutput_dist_le`);
  2. the XSA projection `y - ⟨y, û⟩ û` moves by at most `2‖Δy‖ + 2‖y'‖‖Δû‖`
     (`xsaProjection_dist_le`, from the algebraic `norm_proj_sub_proj_le`);
  3. RoPE is an isometry (`applyRope_dist`), so it contributes nothing, and
     `QKNormLipschitz.score_lipschitz` turns a bound on `q, k` into the score
     shift `r` that step 1 needs.

All three are linear in the displacement, `SoftmaxStability` having already
traded the exponential softmax estimate against the trivial one, so what
`Properties.LipschitzConstants` picks up from here is a genuine Lipschitz
constant and not a modulus of continuity on a bounded set.
-/

import Transformer.GPTMini.SoftmaxStability
import Transformer.GPTMini.QKNormLipschitz

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-! ### The attention output -/

/-- **How far the attention output moves.**

With `r` a bound on the score movement at query position `i`, `B` a bound on
the values `v'`, and `M` a bound on the movement of the values,

  `‖y_i - y'_i‖ ≤ 8 r B + M`.

The first term is `SoftmaxStability.causalAttnWeights_l1_le_linear` against
`B`; the second is the convexity of the weights
(`causalAttnWeights_row_sum`).

Source: `reference/model.py` (`CausalMHA.forward`, `y = attn @ v`). -/
theorem attnOutput_dist_le
    {T : ℕ} (alpha eps r B M : ℝ)
    (q k v q' k' v' : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (hclose : ∀ j : Fin T,
      |preScore cfg alpha eps q k i j - preScore cfg alpha eps q' k' i j| ≤ r)
    (hB : ∀ j, ‖v' j‖ ≤ B) (hM : ∀ j, ‖v j - v' j‖ ≤ M) :
    ‖attnOutput cfg alpha eps q k v i - attnOutput cfg alpha eps q' k' v' i‖
      ≤ 8 * r * B + M := by
  have hB0 : 0 ≤ B := (norm_nonneg _).trans (hB i)
  have hdiff : attnOutput cfg alpha eps q k v i - attnOutput cfg alpha eps q' k' v' i
      = (∑ j : Fin T, (causalAttnWeights cfg alpha eps q k i j
            - causalAttnWeights cfg alpha eps q' k' i j) • v' j)
        + ∑ j : Fin T, causalAttnWeights cfg alpha eps q k i j • (v j - v' j) := by
    rw [attnOutput, attnOutput, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by module
  have h1 : ‖∑ j : Fin T, (causalAttnWeights cfg alpha eps q k i j
        - causalAttnWeights cfg alpha eps q' k' i j) • v' j‖
      ≤ 8 * r * B := by
    calc ‖∑ j : Fin T, (causalAttnWeights cfg alpha eps q k i j
          - causalAttnWeights cfg alpha eps q' k' i j) • v' j‖
        ≤ ∑ j : Fin T, ‖(causalAttnWeights cfg alpha eps q k i j
            - causalAttnWeights cfg alpha eps q' k' i j) • v' j‖ := norm_sum_le _ _
      _ = ∑ j : Fin T, |causalAttnWeights cfg alpha eps q k i j
            - causalAttnWeights cfg alpha eps q' k' i j| * ‖v' j‖ :=
          Finset.sum_congr rfl fun j _ => by rw [norm_smul, Real.norm_eq_abs]
      _ ≤ ∑ j : Fin T, |causalAttnWeights cfg alpha eps q k i j
            - causalAttnWeights cfg alpha eps q' k' i j| * B :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hB j) (abs_nonneg _)
      _ = (∑ j : Fin T, |causalAttnWeights cfg alpha eps q k i j
            - causalAttnWeights cfg alpha eps q' k' i j|) * B := by rw [Finset.sum_mul]
      _ ≤ 8 * r * B :=
          mul_le_mul_of_nonneg_right
            (causalAttnWeights_l1_le_linear cfg alpha eps r q k q' k' i hclose) hB0
  have h2 : ‖∑ j : Fin T, causalAttnWeights cfg alpha eps q k i j • (v j - v' j)‖ ≤ M := by
    calc ‖∑ j : Fin T, causalAttnWeights cfg alpha eps q k i j • (v j - v' j)‖
        ≤ ∑ j : Fin T, ‖causalAttnWeights cfg alpha eps q k i j • (v j - v' j)‖ :=
          norm_sum_le _ _
      _ = ∑ j : Fin T, causalAttnWeights cfg alpha eps q k i j * ‖v j - v' j‖ :=
          Finset.sum_congr rfl fun j _ => by
            rw [norm_smul, Real.norm_eq_abs,
              abs_of_nonneg (causalAttnWeights_nonneg cfg alpha eps q k i j)]
      _ ≤ ∑ j : Fin T, causalAttnWeights cfg alpha eps q k i j * M :=
          Finset.sum_le_sum fun j _ =>
            mul_le_mul_of_nonneg_left (hM j) (causalAttnWeights_nonneg cfg alpha eps q k i j)
      _ = M := by rw [← Finset.sum_mul, causalAttnWeights_row_sum, one_mul]
  rw [hdiff]
  exact (norm_add_le _ _).trans (by linarith)

/-- The hypotheses are satisfiable: identical arguments, `r = 0`, `M = 0`. -/
example (cfg : Config) (alpha eps : ℝ)
    (q k v : Fin 3 → EucSpace cfg.head_dim) (i : Fin 3) (B : ℝ) (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖attnOutput cfg alpha eps q k v i - attnOutput cfg alpha eps q k v i‖
      ≤ 8 * 0 * B + 0 :=
  attnOutput_dist_le cfg alpha eps 0 B 0 q k v q k v i (fun _ => by simp) hB
    (fun _ => by simp)

/-! ### The XSA projection -/

/-- **Two rank-one projections applied to two vectors.**

For unit-or-shorter directions `u, u'`,

  `‖(a - ⟨a,u⟩u) - (b - ⟨b,u'⟩u')‖ ≤ 2‖a - b‖ + 2‖b‖‖u - u'‖`,

by the three-term split
`(a-b) - ⟨a-b,u⟩u - ⟨b,u-u'⟩u - ⟨b,u'⟩(u-u')`.

Source: the XSA step of `reference/model.py` (`CausalMHA.forward`). -/
theorem norm_proj_sub_proj_le {d : ℕ} (a b u u' : EucSpace d)
    (hu : ‖u‖ ≤ 1) (hu' : ‖u'‖ ≤ 1) :
    ‖(a - (inner (𝕜 := ℝ) a u) • u) - (b - (inner (𝕜 := ℝ) b u') • u')‖
      ≤ 2 * ‖a - b‖ + 2 * ‖b‖ * ‖u - u'‖ := by
  have hdec : (a - (inner (𝕜 := ℝ) a u) • u) - (b - (inner (𝕜 := ℝ) b u') • u')
      = (a - b) - (inner (𝕜 := ℝ) (a - b) u) • u
        - (inner (𝕜 := ℝ) b (u - u')) • u - (inner (𝕜 := ℝ) b u') • (u - u') := by
    rw [inner_sub_left, inner_sub_right]
    module
  have h1 : ‖(inner (𝕜 := ℝ) (a - b) u) • u‖ ≤ ‖a - b‖ := by
    rw [norm_smul, Real.norm_eq_abs]
    calc |inner (𝕜 := ℝ) (a - b) u| * ‖u‖ ≤ (‖a - b‖ * ‖u‖) * ‖u‖ :=
          mul_le_mul_of_nonneg_right (abs_real_inner_le_norm _ _) (norm_nonneg _)
      _ ≤ (‖a - b‖ * 1) * 1 := by gcongr
      _ = ‖a - b‖ := by ring
  have h2 : ‖(inner (𝕜 := ℝ) b (u - u')) • u‖ ≤ ‖b‖ * ‖u - u'‖ := by
    rw [norm_smul, Real.norm_eq_abs]
    calc |inner (𝕜 := ℝ) b (u - u')| * ‖u‖ ≤ (‖b‖ * ‖u - u'‖) * ‖u‖ :=
          mul_le_mul_of_nonneg_right (abs_real_inner_le_norm _ _) (norm_nonneg _)
      _ ≤ (‖b‖ * ‖u - u'‖) * 1 := by gcongr
      _ = ‖b‖ * ‖u - u'‖ := by ring
  have h3 : ‖(inner (𝕜 := ℝ) b u') • (u - u')‖ ≤ ‖b‖ * ‖u - u'‖ := by
    rw [norm_smul, Real.norm_eq_abs]
    refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
    calc |inner (𝕜 := ℝ) b u'| ≤ ‖b‖ * ‖u'‖ := abs_real_inner_le_norm _ _
      _ ≤ ‖b‖ * 1 := by gcongr
      _ = ‖b‖ := mul_one _
  rw [hdec]
  have t3 := norm_sub_le (a - b) ((inner (𝕜 := ℝ) (a - b) u) • u)
  have t2 := norm_sub_le ((a - b) - (inner (𝕜 := ℝ) (a - b) u) • u)
      ((inner (𝕜 := ℝ) b (u - u')) • u)
  have t1 := norm_sub_le
      ((a - b) - (inner (𝕜 := ℝ) (a - b) u) • u - (inner (𝕜 := ℝ) b (u - u')) • u)
      ((inner (𝕜 := ℝ) b u') • (u - u'))
  linarith

/-- The hypotheses are satisfiable: the zero direction has norm `0 ≤ 1`. -/
example {d : ℕ} (a b : EucSpace d) :
    ‖(a - (inner (𝕜 := ℝ) a (0 : EucSpace d)) • (0 : EucSpace d))
        - (b - (inner (𝕜 := ℝ) b (0 : EucSpace d)) • (0 : EucSpace d))‖
      ≤ 2 * ‖a - b‖ + 2 * ‖b‖ * ‖(0 : EucSpace d) - 0‖ :=
  norm_proj_sub_proj_le a b 0 0 (by simp) (by simp)

/-- **How far the XSA projection moves** when both the attention output and
the self-value direction move.  `normL2` never exceeds `1`, so
`norm_proj_sub_proj_le` applies directly. -/
theorem xsaProjection_dist_le
    {T : ℕ} (eps : ℝ) (heps : 0 ≤ eps)
    (v y v' y' : Fin T → EucSpace cfg.head_dim) (i : Fin T) :
    ‖xsaProjection cfg eps v y i - xsaProjection cfg eps v' y' i‖
      ≤ 2 * ‖y i - y' i‖ + 2 * ‖y' i‖ * ‖normL2 eps (v i) - normL2 eps (v' i)‖ :=
  norm_proj_sub_proj_le (y i) (y' i) (normL2 eps (v i)) (normL2 eps (v' i))
    (normL2_norm_le eps heps (v i)) (normL2_norm_le eps heps (v' i))

/-- The hypothesis is satisfiable: the `eps = 10⁻⁶` of `reference/model.py`. -/
example (cfg : Config) (v y v' y' : Fin 3 → EucSpace cfg.head_dim) (i : Fin 3) :
    ‖xsaProjection cfg 1e-6 v y i - xsaProjection cfg 1e-6 v' y' i‖
      ≤ 2 * ‖y i - y' i‖ + 2 * ‖y' i‖ * ‖normL2 1e-6 (v i) - normL2 1e-6 (v' i)‖ :=
  xsaProjection_dist_le cfg 1e-6 (by norm_num) v y v' y' i

end GPTMini
end Transformer
