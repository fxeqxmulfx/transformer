/-
# Causal sparsemax with the gpt-mini score and XSA projection

New replacement of `CausalMHA.forward` in `experiments/gpt_mini.py`,
restored from commit f11b6e27d3cfe6813a2876bdeec38565fc54258c. RoPE,
QK normalization, the learned temperature, and XSA keep their existing
definitions. Only the attention weight row uses a quadratic simplex
program. Motivation: arXiv:2211.11052v1, §3.1, not its shared positional
matrix or an equivalence to softmax attention.
-/

import Transformer.GPTMini.Convex.Basic
import Transformer.GPTMini.AttentionBounds

open scoped BigOperators

noncomputable section

namespace Transformer.GPTMini.Convex

open Transformer.ConvexRecall

variable (cfg : Config) {T : ℕ}

/-- The original content scores, including RoPE and QK normalization.
Source: `CausalMHA.forward`, before the causal softmax. -/
def headScores (alpha eps : ℝ) (q k : Fin T → EucSpace cfg.head_dim)
    (positions : Fin T → ℝ) (i j : Fin T) : ℝ :=
  Real.exp alpha * inner (𝕜 := ℝ)
    (applyRope cfg.head_dim cfg.rope_theta (positions i) (normL2 eps (q i)))
    (applyRope cfg.head_dim cfg.rope_theta (positions j) (normL2 eps (k j)))

/-- Replacement of `F.softmax(scores + mask, dim=-1)` by a causal
squared-distance projection. Source: `CausalMHA.forward`; new mechanism
motivated by arXiv:2211.11052v1, §3.1. -/
def attentionWeights (alpha eps : ℝ) (q k : Fin T → EucSpace cfg.head_dim)
    (positions : Fin T → ℝ) (i : Fin T) : Fin T → ℝ :=
  sparseWeights (headScores cfg alpha eps q k positions i) i

/-- The value mixture before XSA. Source: `CausalMHA.forward`, `attn @ v`. -/
def attentionOutput (alpha eps : ℝ) (q k v : Fin T → EucSpace cfg.head_dim)
    (positions : Fin T → ℝ) (i : Fin T) : EucSpace cfg.head_dim :=
  ∑ j, attentionWeights cfg alpha eps q k positions i j • v j

/-- The replacement head retains the original epsilon-regularized XSA.
Source: `CausalMHA.forward`, `z = y - (y * v_hat).sum(...) * v_hat`. -/
def attentionHead (alpha eps : ℝ) (q k v : Fin T → EucSpace cfg.head_dim)
    (positions : Fin T → ℝ) (i : Fin T) : EucSpace cfg.head_dim :=
  xsaProjection cfg eps v (attentionOutput cfg alpha eps q k v positions) i

/-- Every row is feasible for its causal simplex. Source: new replacement
of `CausalMHA.forward`; simplex motivation in arXiv:2211.11052v1, §3.1. -/
theorem attentionWeights_mem (alpha eps : ℝ) (q k : Fin T → EucSpace cfg.head_dim)
    (positions : Fin T → ℝ) (i : Fin T) :
    attentionWeights cfg alpha eps q k positions i ∈ simplexOn {j : Fin T | j ≤ i} :=
  (sparseWeights_spec _ i).1

/-- The replacement weights are nonnegative. Source: new replacement of
`CausalMHA.forward`; arXiv:2211.11052v1, §3.1, simplex motivation. -/
theorem attentionWeights_nonneg (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ) (i j : Fin T) :
    0 ≤ attentionWeights cfg alpha eps q k positions i j :=
  (attentionWeights_mem cfg alpha eps q k positions i).1 j

/-- Every replacement row sums to one. Source: new replacement of
`CausalMHA.forward`; arXiv:2211.11052v1, §3.1, simplex motivation. -/
theorem attentionWeights_row_sum (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ) (i : Fin T) :
    (∑ j, attentionWeights cfg alpha eps q k positions i j) = 1 :=
  (attentionWeights_mem cfg alpha eps q k positions i).2.1

/-- Future positions have exactly zero weight. Source: new replacement of
`CausalMHA.forward`, retaining its upper-triangular causal mask. -/
theorem attentionWeights_zero_above (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ)
    (i j : Fin T) (hij : i < j) :
    attentionWeights cfg alpha eps q k positions i j = 0 :=
  (attentionWeights_mem cfg alpha eps q k positions i).2.2 j (not_le.mpr hij)

/-- The causal exclusion premise occurs already at sequence length two.
Source: `CausalMHA.forward`, causal mask. -/
example : ∃ i j : Fin 2, i < j := ⟨0, 1, by decide⟩

/-- The inference objective is convex for every assignment of trainable
Q/K parameters, positions, and temperatures. This quantifies over scores
but asserts convexity in the row weights. Source: new replacement of
`CausalMHA.forward`; arXiv:2211.11052v1, §3.1, simplex motivation. -/
theorem attentionObjective_convex (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ) (i : Fin T) :
    ConvexOn ℝ (simplexOn {j : Fin T | j ≤ i})
      (routingObjective (fun j => -headScores cfg alpha eps q k positions i j / 2)) :=
  routingObjective_convex _ _

/-- The actual head weights minimize squared distance to the original
score row. Source: new replacement of `CausalMHA.forward`; §3.1 motivation. -/
theorem attentionWeights_projection_min (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ)
    (i : Fin T) (a : Fin T → ℝ) (ha : a ∈ simplexOn {j : Fin T | j ≤ i}) :
    (∑ j, (attentionWeights cfg alpha eps q k positions i j -
      headScores cfg alpha eps q k positions i j) ^ 2) ≤
        ∑ j, (a j - headScores cfg alpha eps q k positions i j) ^ 2 := by
  have h := (sparseWeights_spec (headScores cfg alpha eps q k positions i) i).2 a ha
  have hx := routing_projection_identity (headScores cfg alpha eps q k positions i)
    (attentionWeights cfg alpha eps q k positions i)
  have hy := routing_projection_identity (headScores cfg alpha eps q k positions i) a
  change routingObjective _ (attentionWeights cfg alpha eps q k positions i) ≤ _ at h
  linarith

/-- The minimization premise has feasible comparison rows. Source context:
`CausalMHA.forward`, diagonal allowed by the causal mask. -/
example : basis (0 : Fin 1) ∈ simplexOn {j : Fin 1 | j ≤ 0} :=
  basis_mem_simplex _ _ (by change (0 : Fin 1) ≤ 0; exact le_rfl)

/-- A value bound also bounds the mixture, regardless of the learned
scores. Source: `CausalMHA.forward`, `attn @ v`, with replacement weights. -/
theorem attentionOutput_norm_le (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ)
    (i : Fin T) (B : ℝ) (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖attentionOutput cfg alpha eps q k v positions i‖ ≤ B := by
  calc
    _ ≤ ∑ j, ‖attentionWeights cfg alpha eps q k positions i j • v j‖ :=
      norm_sum_le _ _
    _ = ∑ j, attentionWeights cfg alpha eps q k positions i j * ‖v j‖ := by
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [norm_smul, Real.norm_eq_abs,
        abs_of_nonneg (attentionWeights_nonneg cfg alpha eps q k positions i j)]
    _ ≤ ∑ j, attentionWeights cfg alpha eps q k positions i j * B :=
      Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hB j)
        (attentionWeights_nonneg cfg alpha eps q k positions i j)
    _ = B := by rw [← Finset.sum_mul, attentionWeights_row_sum, one_mul]

/-- The original head bound survives the replacement and retains XSA.
Source: `CausalMHA.forward`, value mixture and XSA projection. -/
theorem attentionHead_norm_le (alpha eps : ℝ) (heps : 0 ≤ eps)
    (q k v : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ)
    (i : Fin T) (B : ℝ) (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖attentionHead cfg alpha eps q k v positions i‖ ≤ 2 * B := by
  exact (xsaProjection_norm_le cfg eps heps v _ i).trans
    (mul_le_mul_of_nonneg_left
      (attentionOutput_norm_le cfg alpha eps q k v positions i B hB) (by norm_num))

/-- Positive normalization epsilon and a uniform value bound are
simultaneously satisfiable. Source: `CausalMHA.forward`, zero values. -/
example : (0 : ℝ) ≤ 1 ∧
    ∀ j : Fin 2, ‖(fun _ : Fin 2 => (0 : EucSpace cfg.head_dim)) j‖ ≤ (0 : ℝ) := by
  constructor
  · norm_num
  · intro j; simp

end Transformer.GPTMini.Convex
