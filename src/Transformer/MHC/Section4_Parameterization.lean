module

public import Transformer.MHC.Section4_Sinkhorn

/-!
# mHC gate parameterization

arXiv:2512.24880v2, §4.2: read weights use sigmoid, write weights use
twice sigmoid, and skip logits undergo Sinkhorn normalization. Inputs
here are the logits after the paper's RMSNorm and learned projections.

The learned projection is an affine map on the normalized flattened
stream; its small dynamic scale is separate from its static bias.
Read gates do not sum to one, and their positivity must not be mistaken
for the convex normalization used by Attention Residuals.
Skip-map conservation survives write-back only with an additional
condition on the branch output. The theorems below retain that output
explicitly and give a concrete complete-block amplification example.
-/

@[expose] public section

open scoped BigOperators
open Transformer.Residual

namespace Transformer.MHC

variable {ι : Type*} [Fintype ι]
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Dynamic/static logit decomposition after flattening and RMS-normalizing
the token's entire stream. Here `z` is that normalized vector; `κ` indexes
either read gates, write gates, or flattened skip-matrix entries.
Source: arXiv:2512.24880v2, §4.2, parameterization before manifold projection.
The normalization is outside this affine map, not absorbed into its bias. -/
noncomputable def projectedLogits {κ : Type*} (z : ι → ℝ) (α : ℝ)
    (P : ι → κ → ℝ) (bias : κ → ℝ) (j : κ) : ℝ :=
  α * (∑ k, z k * P k j) + bias j

/-- At zero dynamic scale only the learned static bias remains.
Source: arXiv:2512.24880v2, §4.2, small-scale initialization. -/
theorem projectedLogits_zero_scale {κ : Type*} (z : ι → ℝ)
    (P : ι → κ → ℝ) (bias : κ → ℝ) :
    projectedLogits z 0 P bias = bias := by
  funext j
  simp [projectedLogits]

/-- Sigmoid read gate. Source: arXiv:2512.24880v2, §4.2, `H_pre = σ(logits)`. -/
noncomputable def gate (z : ℝ) : ℝ := 1 / (1 + Real.exp (-z))

/-- Read gates are positive, but are not a probability distribution.
Source: arXiv:2512.24880v2, §4.2, sigmoid parameterization. -/
theorem gate_pos (z : ℝ) : 0 < gate z := by
  unfold gate
  positivity

/-- A read gate is strictly below one for finite logits.
Source: arXiv:2512.24880v2, §4.2. -/
theorem gate_lt_one (z : ℝ) : gate z < 1 := by
  unfold gate
  rw [div_lt_one (by positivity)]
  have h := Real.exp_pos (-z)
  linarith

/-- The write gate's factor two is part of the source's parameterization.
Source: arXiv:2512.24880v2, §4.2, `H_post = 2 σ(logits)`. -/
noncomputable def writeGate (z : ℝ) : ℝ := 2 * gate z

/-- Write weights lie in `(0,2)`, rather than summing to one.
Source: arXiv:2512.24880v2, §4.2. -/
theorem writeGate_bounds (z : ℝ) : 0 < writeGate z ∧ writeGate z < 2 := by
  have hp := gate_pos z
  have hu := gate_lt_one z
  unfold writeGate
  constructor <;> linarith

/-- Apply the actual gate functions with a supplied normalized skip map.
Source: arXiv:2512.24880v2, Eqs. (3), (8). Supplying the map explicitly
allows the ideal and finite normalization models to remain distinct. -/
noncomputable def gatedHyperStep (readLogits writeLogits : ι → ℝ)
    (H : Matrix ι ι ℝ) (F : E → E) (x : ι → E) : ι → E :=
  hyperStep (fun i => gate (readLogits i)) (fun i => writeGate (writeLogits i)) H F x

/-- The paper's practical finite-iteration block, including the exact
factor two on write-back. Source: arXiv:2512.24880v2, §4.2, `t_max=20`.
Implementation epsilon and floating-point rounding are outside this model. -/
noncomputable def finiteMHCStep (readLogits writeLogits : ι → ℝ)
    (skipLogits : Matrix ι ι ℝ) (F : E → E) (x : ι → E) : ι → E :=
  gatedHyperStep readLogits writeLogits (finiteSinkhorn skipLogits 20) F x

/-- A bound on a complete gated block needs a separate branch bound.
Source: derived from arXiv:2512.24880v2, Eq. (3) and §4.2. The upper
bound on the skip map alone does not prove contraction of the full block. -/
theorem gatedHyperStep_norm_le [DecidableEq ι] (a c : ι → ℝ)
    (H : Matrix ι ι ℝ) (hH : H ∈ doublyStochastic ℝ ι) (F : E → E)
    (x : ι → E) (R B : ℝ) (hB : 0 ≤ B) (hx : ∀ i, ‖x i‖ ≤ R)
    (hF : ‖F (weightedSum (fun j => gate (a j)) x)‖ ≤ B) (i : ι) :
    ‖gatedHyperStep a c H F x i‖ ≤ R + 2 * B := by
  have hc := writeGate_bounds (c i)
  calc
    ‖gatedHyperStep a c H F x i‖ ≤ ‖mix H x i‖ +
        ‖writeGate (c i) • F (weightedSum (fun j => gate (a j)) x)‖ := norm_add_le _ _
    _ = ‖mix H x i‖ + writeGate (c i) *
        ‖F (weightedSum (fun j => gate (a j)) x)‖ := by
      rw [norm_smul, Real.norm_of_nonneg hc.1.le]
    _ ≤ R + writeGate (c i) * B :=
      add_le_add (mix_norm_le H hH x R hx i) (mul_le_mul_of_nonneg_left hF hc.1.le)
    _ ≤ R + 2 * B := add_le_add le_rfl (mul_le_mul_of_nonneg_right hc.2.le hB)

/-- All block-bound hypotheses hold jointly for nonzero four-stream input,
zero gate logits, identity skip, and an identity branch. -/
example : (1 : Matrix (Fin 4) (Fin 4) ℝ) ∈ doublyStochastic ℝ (Fin 4) ∧
    (0 : ℝ) ≤ 2 ∧ (∀ i : Fin 4, ‖(fun _ : Fin 4 => (1 : ℝ)) i‖ ≤ 1) ∧
    ‖id (weightedSum (fun _ : Fin 4 => gate 0) (fun _ => (1 : ℝ)))‖ ≤ 2 := by
  refine ⟨(doublyStochastic ℝ (Fin 4)).one_mem, ?_, ?_, ?_⟩
  · norm_num
  · norm_num
  · norm_num [weightedSum, gate]

/-- Write-back changes the stream sum by the total write weight times the
branch output. Conservation of the skip map does not make the full block
conservative. Source: arXiv:2512.24880v2, Eq. (3) and §4.1. -/
theorem gatedHyperStep_sum [DecidableEq ι] (a c : ι → ℝ)
    (H : Matrix ι ι ℝ) (hH : H ∈ doublyStochastic ℝ ι) (F : E → E) (x : ι → E) :
    ∑ i, gatedHyperStep a c H F x i =
      (∑ i, x i) + (∑ i, writeGate (c i)) • F (weightedSum (fun j => gate (a j)) x) := by
  unfold gatedHyperStep
  rw [sum_hyperStep, mix_preserves_sum H hH]

/-- The sum formula's conservation hypothesis holds for an identity skip. -/
example : (1 : Matrix (Fin 4) (Fin 4) ℝ) ∈ doublyStochastic ℝ (Fin 4) :=
  (doublyStochastic ℝ (Fin 4)).one_mem

/-- Even with a doubly stochastic skip map, a gated branch can amplify
the full block. Source: limitation of the skip-only bound in
arXiv:2512.24880v2, §4.1, made explicit using Eq. (3). -/
theorem full_block_can_amplify :
    ∃ (H : Matrix (Fin 1) (Fin 1) ℝ) (x : Fin 1 → ℝ),
      H ∈ doublyStochastic ℝ (Fin 1) ∧
      ‖x 0‖ < ‖gatedHyperStep (fun _ => 0) (fun _ => 0) H (fun v => 100 * v) x 0‖ := by
  refine ⟨1, fun _ => 1, (doublyStochastic ℝ (Fin 1)).one_mem, ?_⟩
  norm_num [gatedHyperStep, hyperStep, mix, weightedSum, gate, writeGate]

/-- Positive read gates are not softmax: four zero logits have total weight
two. Source: arXiv:2512.24880v2, §4.2; relevant to the four-stream GLM
and DeepSeek configurations. In contrast, AttnRes weights sum to one. -/
theorem four_read_gates_not_normalized : ∑ _ : Fin 4, gate 0 = 2 := by
  unfold gate
  norm_num

end Transformer.MHC
