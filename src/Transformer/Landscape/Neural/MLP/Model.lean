module

public import Transformer.Landscape.Neural.Residual

/-!
# The actual two-hidden-layer SiLU perceptron

Source: `python/src/lab/infrastructure/nn/mlp.py`, `MLP.__init__` and
`MLP.forward`; M9.3 of `landscape_grokking_plan.md` and the matched controls
in `experiments/mod97_representation_controls/experiment.py`.

Every matrix entry and every bias is a separate Euclidean coordinate.
Both hidden affine maps are followed by SiLU; the final map is affine.
The scalar readout takes a singleton output index, and the vector controls
take one index per answer. Inputs are fixed encoded operands. This is an
exact-real model of the architecture, not a floating-point certificate.
-/

@[expose] public section

namespace Transformer.Landscape.Neural.MLP

/-- The three disjoint affine layers, each with its own matrix and bias.
Matrix indices are input-first, transposing PyTorch's storage convention. -/
abbrev Index (D H O : Type*) := ((D × H) ⊕ H) ⊕ ((H × H) ⊕ H) ⊕ ((H × O) ⊕ O)

/-- The complete Euclidean parameter space, including all three biases. -/
abbrev Weights (D H O : Type*) := EuclideanSpace ℝ (Index D H O)

variable {D H O : Type*} [Fintype D] [Fintype H] [Fintype O]

/-- First affine layer's input-to-hidden matrix (M9.3b, `hidden[0]`). -/
def firstMatrix (w : Weights D H O) (i : D) (j : H) : ℝ :=
  w.ofLp (.inl (.inl (i, j)))

/-- First hidden bias; it is trainable, rather than a fixed zero. -/
def firstBias (w : Weights D H O) (j : H) : ℝ := w.ofLp (.inl (.inr j))

/-- Second affine layer's hidden-to-hidden matrix (M9.3b, `hidden[1]`). -/
def secondMatrix (w : Weights D H O) (i j : H) : ℝ :=
  w.ofLp (.inr (.inl (.inl (i, j))))

/-- Second hidden bias, disjoint from the other layer's coordinates. -/
def secondBias (w : Weights D H O) (j : H) : ℝ := w.ofLp (.inr (.inl (.inr j)))

/-- Final hidden-to-answer matrix, with no weight tying (M9.3b, `output`). -/
def outputMatrix (w : Weights D H O) (i : H) (q : O) : ℝ :=
  w.ofLp (.inr (.inr (.inl (i, q))))

/-- The final trainable bias, one coordinate for every output. -/
def outputBias (w : Weights D H O) (q : O) : ℝ := w.ofLp (.inr (.inr (.inr q)))

/-- First actual hidden activation: affine input followed by SiLU. -/
noncomputable def firstHidden (w : Weights D H O) (v : D → ℝ) (j : H) : ℝ :=
  silu (dense (firstMatrix w) v j + firstBias w j)

/-- Second actual hidden activation, retaining its own matrix and bias. -/
noncomputable def secondHidden (w : Weights D H O) (v : D → ℝ) (j : H) : ℝ :=
  silu (dense (secondMatrix w) (firstHidden w v) j + secondBias w j)

/-- Raw affine output; it is neither normalized nor passed through SiLU. -/
noncomputable def prediction (w : Weights D H O) (v : D → ℝ) (q : O) : ℝ :=
  dense (outputMatrix w) (secondHidden w v) q + outputBias w q

/-- Scalar operand encoding from `MLP.encode`, specialized to two operands. -/
noncomputable def scalarInput {p : ℕ} (a b : Fin p) (i : Fin 2) : ℝ :=
  2 * ((if i = 0 then a.val else b.val) : ℝ) / (p - 1 : ℝ) - 1

/-- Concatenated one-hot operand encoding, with the operand position explicit. -/
def oneHotInput {p : ℕ} (a b : Fin p) (i : Fin 2 × Fin p) : ℝ :=
  if i.2 = (if i.1 = 0 then a else b) then 1 else 0

/-- Exact parameter count of both biased hidden layers and the biased
answer projection (M9.3b, fixed hidden width rather than matched capacity). -/
theorem finrank_weights : Module.finrank ℝ (Weights D H O) =
    Fintype.card H ^ 2 +
      (Fintype.card D + Fintype.card O + 2) * Fintype.card H + Fintype.card O := by
  rw [finrank_euclideanSpace]
  simp only [Index, Fintype.card_sum, Fintype.card_prod]
  ring

/-- Joint analyticity in all first-layer parameters; the fixed input is
arbitrary, so this covers both operand encodings (M9.2c). -/
theorem analyticAt_firstHidden (x : Weights D H O) (v : D → ℝ) (j : H) :
    AnalyticAt ℝ (fun w => firstHidden w v j) x := by
  apply analyticAt_silu_comp
  exact (analyticAt_dense (v := fun _ : Weights D H O => v)
    (fun i k => analyticAt_parameter (.inl (.inl (i, k))) x)
    (fun _ => analyticAt_const) j).fun_add
      (analyticAt_parameter (.inl (.inr j)) x)

/-- Second-layer analyticity includes the trained first layer, rather than
assuming frozen features or an analytic output map (M9.2c). -/
theorem analyticAt_secondHidden (x : Weights D H O) (v : D → ℝ) (j : H) :
    AnalyticAt ℝ (fun w => secondHidden w v j) x := by
  apply analyticAt_silu_comp
  exact (analyticAt_dense
    (fun i k => analyticAt_parameter (.inr (.inl (.inl (i, k)))) x)
    (analyticAt_firstHidden x v) j).fun_add
      (analyticAt_parameter (.inr (.inl (.inr j))) x)

/-- Every raw output is analytic in the full Euclidean parameter vector,
including all biases and both feature-learning layers (M9.2c). -/
theorem analyticAt_prediction (x : Weights D H O) (v : D → ℝ) (q : O) :
    AnalyticAt ℝ (fun w => prediction w v q) x := by
  exact (analyticAt_dense
    (fun i k => analyticAt_parameter (.inr (.inr (.inl (i, k)))) x)
    (analyticAt_secondHidden x v) q).fun_add
      (analyticAt_parameter (.inr (.inr (.inr q))) x)

-- The four executable architectures differ in input and output dimensions.
example : Module.finrank ℝ (Weights (Fin 2) (Fin 195) (Fin 1)) = 39001 := by
  rw [finrank_weights]
  norm_num

example : Module.finrank ℝ (Weights (Fin 2 × Fin 97) (Fin 195) (Fin 1)) = 76441 := by
  rw [finrank_weights]
  norm_num

example : Module.finrank ℝ (Weights (Fin 2) (Fin 195) (Fin 97)) = 57817 := by
  rw [finrank_weights]
  norm_num

example : Module.finrank ℝ (Weights (Fin 2 × Fin 97) (Fin 195) (Fin 97)) = 95257 := by
  rw [finrank_weights]
  norm_num

-- Both operand positions and their boundary values use the executable code.
example : scalarInput (0 : Fin 97) 96 0 = -1 ∧ scalarInput (0 : Fin 97) 96 1 = 1 := by
  norm_num [scalarInput]

example : oneHotInput (2 : Fin 97) 5 (0, 2) = 1 ∧
    oneHotInput (2 : Fin 97) 5 (1, 2) = 0 ∧
    oneHotInput (2 : Fin 97) 5 (1, 5) = 1 := by
  norm_num [oneHotInput]

-- The analytic maps stay live in every parameter at nonzero checkpoints.
example : AnalyticAt ℝ (fun w : Weights (Fin 2) (Fin 3) (Fin 1) =>
    firstHidden w (scalarInput (2 : Fin 97) 5) 0)
      (WithLp.toLp 2 fun _ => (1 : ℝ)) := by
  exact analyticAt_firstHidden _ _ _

example : AnalyticAt ℝ (fun w : Weights (Fin 2 × Fin 97) (Fin 3) (Fin 97) =>
    secondHidden w (oneHotInput (2 : Fin 97) 5) 1)
      (WithLp.toLp 2 fun _ => (1 : ℝ)) := by
  exact analyticAt_secondHidden _ _ _

example : AnalyticAt ℝ (fun w : Weights (Fin 2) (Fin 3) (Fin 97) =>
    prediction w (scalarInput (2 : Fin 97) 5) 7)
      (WithLp.toLp 2 fun _ => (1 : ℝ)) := by
  exact analyticAt_prediction _ _ _

end Transformer.Landscape.Neural.MLP
