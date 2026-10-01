/-
# IC-EoT: embedding, residual blocks and readout

arXiv:2603.22095v2, §3.2, Eqs. (13), (14), (23)–(27).
All weight matrices, biases, activations and positional encodings are
actual parameters. The encoder depth is the length of the block list.
-/

import Transformer.ICEoT.Section3_Attention

noncomputable section

namespace Transformer.ICEoT

/-- One residual encoder block, Eqs. (23)–(25), §3.2. -/
structure Block (m h heads ff : ℕ) where
  att : MultiHead m h heads
  first : Fin ff → Fin m → ℝ
  firstBias : Fin ff → ℝ
  second : Fin m → Fin ff → ℝ
  secondBias : Fin m → ℝ
  activation : ℝ → ℝ

/-- Assumption 1 restricted to an encoder block; §3.3. -/
def BlockConditions {m h heads ff : ℕ} (p : Block m h heads ff) : Prop :=
  MultiHeadConditions p.att ∧ Nonnegative p.first ∧ Nonnegative p.second ∧
    ActivationConditions p.activation

/-- The first residual addition in Eq. (23), §3.2. -/
def firstResidual {n m h heads ff : ℕ} (p : Block m h heads ff)
    (H : Sequence (n + 1) m) : Sequence (n + 1) m := H + attention p.att H

/-- The positionwise feed-forward sublayer of Eq. (24), §3.2. -/
def feedForward {n m h heads ff : ℕ} (p : Block m h heads ff)
    (H : Sequence (n + 1) m) : Sequence (n + 1) m :=
  fun ir => affine p.second p.secondBias
    (fun r => p.activation (affine p.first p.firstBias (fun s => H (ir.1, s)) r)) ir.2

/-- Both residual additions and the intervening sublayer; §3.2, Eqs. (23)–(25). -/
def encoderBlock {n m h heads ff : ℕ} (p : Block m h heads ff)
    (H : Sequence (n + 1) m) : Sequence (n + 1) m :=
  firstResidual p H + feedForward p (firstResidual p H)

/-- Parameters of the complete encoder, including the fixed encoding `pos`.
An arbitrary finite feature index permits both full expansion (Eq. (13))
and selective control expansion (Eq. (33)); §3.2, Eqs. (14), (27). -/
structure Encoder (n : ℕ) (I : Type*) (m h heads ff out : ℕ) where
  embedding : Fin m → I → ℝ
  embeddingBias : Fin m → ℝ
  pos : Sequence (n + 1) m
  blocks : List (Block m h heads ff)
  readout : Fin out → Fin m → ℝ
  readoutBias : Fin out → ℝ

/-- The complete structural assumptions, including every block; §3.3,
Assumption 1. Diagonality is enforced by `Head`'s representation. -/
def EncoderConditions {n m h heads ff out : ℕ} {I : Type*}
    (p : Encoder n I m h heads ff out) : Prop :=
  Nonnegative p.embedding ∧ (∀ b ∈ p.blocks, BlockConditions b) ∧ Nonnegative p.readout

/-- The embedding and positional addition in Eq. (14), §3.2. -/
def embed {n m h heads ff out : ℕ} {I : Type*} [Fintype I]
    (p : Encoder n I m h heads ff out) (X : (Fin (n + 1) × I) → ℝ) :
    Sequence (n + 1) m :=
  fun ir => affine p.embedding p.embeddingBias (fun s => X (ir.1, s)) ir.2 + p.pos ir

/-- Applying the `L_e` blocks in their source order; §3.2, Eqs. (23)–(25). -/
def encoderStack {n m h heads ff : ℕ} :
    List (Block m h heads ff) → Sequence (n + 1) m → Sequence (n + 1) m
  | [], H => H
  | b :: bs, H => encoderStack bs (encoderBlock b H)

/-- The actual one-step normalized predictor, with the last token readout;
§3.2, Eqs. (26), (27). -/
def predict {n m h heads ff out : ℕ} {I : Type*} [Fintype I]
    (p : Encoder n I m h heads ff out) (X : (Fin (n + 1) × I) → ℝ) : Fin out → ℝ :=
  affine p.readout p.readoutBias
    (fun r => encoderStack p.blocks (embed p X) (Fin.last n, r))

/-- Affine expansion `[X,-X]`; the Boolean index selects the sign.
§3.2, Eq. (13), and §3.3, Corollary 1. -/
def expand {n d : ℕ} (X : Sequence n d) : (Fin n × (Bool × Fin d)) → ℝ :=
  fun ir => if ir.2.1 then -X (ir.1, ir.2.2) else X (ir.1, ir.2.2)

/-- Prediction on the original, unexpanded sequence; §3.3, Corollary 1. -/
def predictOriginal {n d m h heads ff out : ℕ}
    (p : Encoder n (Bool × Fin d) m h heads ff out) (X : Sequence (n + 1) d) :
    Fin out → ℝ := predict p (expand X)

/-- A nonzero residual-block witness; §3.3, Assumption 1. -/
def unitBlock : Block 1 1 1 1 where
  att := unitAttention
  first := fun _ _ => 1
  firstBias := fun _ => -1
  second := fun _ _ => 1
  secondBias := fun _ => 0
  activation := relu

/-- The block hypotheses are satisfiable without zero weights;
§3.3, Assumption 1. -/
theorem unitBlock_conditions : BlockConditions unitBlock :=
  ⟨unitAttention_conditions, fun _ _ => zero_le_one,
    fun _ _ => zero_le_one, relu_conditions⟩

/-- A concrete encoder of any specified depth; §3.2, Eqs. (14)–(27). -/
def unitEncoder (depth : ℕ) : Encoder 0 (Bool × Fin 1) 1 1 1 1 1 where
  embedding := fun _ _ => 1
  embeddingBias := fun _ => -1
  pos := fun _ => 2
  blocks := List.replicate depth unitBlock
  readout := fun _ _ => 1
  readoutBias := fun _ => -1

/-- Every depth admits actual parameters satisfying Assumption 1; §3.3. -/
theorem unitEncoder_conditions (depth : ℕ) : EncoderConditions (unitEncoder depth) := by
  refine ⟨fun _ _ => zero_le_one, ?_, fun _ _ => zero_le_one⟩
  intro b hb
  have heq : b = unitBlock := List.eq_of_mem_replicate hb
  rw [heq]
  exact unitBlock_conditions

end Transformer.ICEoT
