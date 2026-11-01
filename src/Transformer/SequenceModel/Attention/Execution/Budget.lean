module

public import Transformer.SequenceModel.Attention.Execution.Basic

/-!
# Explicit budgets for the recorded source operations

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C4--C6.
These are arithmetic expressions in actual learned matrix norms, dimensions,
positive normalization epsilons, operation error delta and table error tau.
They contain no assumed attention-output or whole-block error constants.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability

/-- Intermediate bounds in forward evaluation order. These fields are numbers, not claims. -/
structure AttentionBudget where
  qkv : ℝ
  qk : ℝ
  rotary : ℝ
  score : ℝ
  average : ℝ
  direction : ℝ
  heads : ℝ
  projected : ℝ
  output : ℝ

/-- Numerical propagation through input RMS, QKV, RMS Q/K, stored RoPE tables,
scaled scores, SDPA, the moving XSA direction, projection and output RMS. -/
def attentionBudget (cfg : Config) (B : BlockParams cfg) (eps xeps δ τ : ℝ) : AttentionBudget :=
  let qkv := ‖B.qkv‖ * δ + δ
  let qk := rmsLip eps * qkv + δ
  let rotary := (2 * τ) * (Real.sqrt (cfg.head_dim : ℝ) + qk) + qk + δ
  let score := (2 * Real.sqrt (cfg.head_dim : ℝ) + rotary) * rotary /
    Real.sqrt (cfg.head_dim : ℝ) + δ
  let average := 8 * score * valueBound cfg B + qkv + δ
  let direction := (2 / xeps) * qkv + δ
  let heads := (1 + (1 + direction) ^ 2) * average +
    valueBound cfg B * (2 + direction) * direction + δ
  let projected := ‖B.projection‖ * (Real.sqrt (cfg.n_heads : ℝ) * heads) + δ
  ⟨qkv, qk, rotary, score, average, direction, heads, projected, rmsLip eps * projected + δ⟩

/-- Bounds for the independent gate/value, SiLU product, output projection and normalization. -/
structure FFNBudget where
  gate : ℝ
  value : ℝ
  product : ℝ
  projected : ℝ
  output : ℝ

/-- The approximate normalized input need not lie exactly on the source ball.
Its displacement and the actual SiLU and product errors are included explicitly. -/
def ffnBudget (cfg : Config) (B : BlockParams cfg) (eps δ : ℝ) : FFNBudget :=
  let gate := ‖B.gate‖ * δ + δ
  let value := ‖B.value‖ * δ + δ
  let Rg := ‖B.gate‖ * Real.sqrt (cfg.d_model : ℝ)
  let Rv := ‖B.value‖ * Real.sqrt (cfg.d_model : ℝ) + value
  let product := Real.sqrt (cfg.d_ff : ℝ) *
    (δ + δ * Rv + Rv * (1 + Rg) * gate + Rg * value)
  let projected := ‖B.output‖ * product + δ
  ⟨gate, value, product, projected, rmsLip eps * projected + δ⟩

/-- New error from all operations of one block, separately from its input-state error. -/
def blockRadius (cfg : Config) (B : BlockParams cfg) (eps xeps δ τ : ℝ) : ℝ :=
  ffnGain cfg B eps * ((attentionBudget cfg B eps xeps δ τ).output + δ) +
    (ffnBudget cfg B eps δ).output + δ

/-- Input embedding/table error and input-normalizer error, then every block's
explicit amplification and new arithmetic error. The recurrence stops at D. -/
def modelBudget (cfg : Config) (θ : ModelParams cfg) (eps xeps δ τ : ℝ) : ℕ → ℝ
  | 0 => rmsLip eps * δ + δ
  | n + 1 => if h : n < cfg.n_layers then
      blockLip cfg (θ.blocks ⟨n, h⟩) eps xeps * modelBudget cfg θ eps xeps δ τ n +
        blockRadius cfg (θ.blocks ⟨n, h⟩) eps xeps δ τ
    else modelBudget cfg θ eps xeps δ τ n

end Transformer.SequenceModel.Attention.Execution
