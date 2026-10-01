/-
# IC-EoT: the actual additive attention equations

arXiv:2603.22095v2, §3.2, Eqs. (15)–(22). A sequence is a real
array indexed by (position, channel). Diagonal matrices are stored by
their diagonals; there are no off-diagonal coefficients to assume away.
The look-back length is `n + 1`, so averaging and the last token are defined.
-/

import Transformer.ICEoT.Section3_Product

noncomputable section

namespace Transformer.ICEoT

/-- A history of `n` tokens with `d` channels; §3.2, Eq. (12). -/
abbrev Sequence (n d : ℕ) := (Fin n × Fin d) → ℝ

/-- Parameters of a single attention head; §3.2, Eqs. (15)–(20).
The paper takes `h = m / nhead`; the same proof applies to any finite width. -/
structure Head (m h : ℕ) where
  query : Fin h → Fin m → ℝ
  key : Fin h → Fin m → ℝ
  latentBias : Fin h → ℝ
  gateScale : Fin h → ℝ
  valueScale : Fin h → ℝ
  gateBias : Fin h → ℝ
  latentActivation : ℝ → ℝ
  gateActivation : ℝ → ℝ

/-- Assumption 1 restricted to one head; §3.3. Bias signs are unrestricted. -/
def HeadConditions {m h : ℕ} (p : Head m h) : Prop :=
  Nonnegative p.query ∧ Nonnegative p.key ∧
    (∀ r, 0 ≤ p.gateScale r) ∧ (∀ r, 0 ≤ p.valueScale r) ∧
    ActivationConditions p.latentActivation ∧ ActivationConditions p.gateActivation

/-- The common pair latent `Z_ij`; §3.2, Eqs. (15)–(17). -/
def pairLatent {n m h : ℕ} (p : Head m h) (H : Sequence n m)
    (i j : Fin n) (r : Fin h) : ℝ :=
  p.latentActivation ((∑ s, p.query r s * H (i, s)) +
    (∑ s, p.key r s * H (j, s)) + p.latentBias r)

/-- The gate branch; §3.2, Eq. (18). -/
def pairGate {n m h : ℕ} (p : Head m h) (H : Sequence n m)
    (i j : Fin n) (r : Fin h) : ℝ :=
  p.gateActivation (p.gateScale r * pairLatent p H i j r + p.gateBias r)

/-- The diagonal value branch, sharing `Z_ij`; §3.2, Eq. (19). -/
def pairValue {n m h : ℕ} (p : Head m h) (H : Sequence n m)
    (i j : Fin n) (r : Fin h) : ℝ :=
  p.valueScale r * pairLatent p H i j r

/-- The Hadamard contribution, with no softmax normalization;
§3.2, Eq. (20). -/
def pairContribution {n m h : ℕ} (p : Head m h) (H : Sequence n m)
    (i j : Fin n) (r : Fin h) : ℝ := pairGate p H i j r * pairValue p H i j r

/-- Averaging over every source position; §3.2, Eq. (21). -/
def headContext {n m h : ℕ} (p : Head m h) (H : Sequence (n + 1) m) :
    Sequence (n + 1) h :=
  fun ir => (∑ j, pairContribution p H ir.1 j ir.2) / (n + 1 : ℝ)

/-- Multi-head parameters and the non-negative projection; §3.2, Eq. (22).
The pair `(head, channel)` is precisely the concatenated channel index. -/
structure MultiHead (m h heads : ℕ) where
  head : Fin heads → Head m h
  projection : Fin m → (Fin heads × Fin h) → ℝ
  bias : Fin m → ℝ

/-- Assumption 1 restricted to the multi-head layer; §3.3. -/
def MultiHeadConditions {m h heads : ℕ} (p : MultiHead m h heads) : Prop :=
  (∀ a, HeadConditions (p.head a)) ∧ Nonnegative p.projection

/-- Concatenation and output projection, with fixed bias;
§3.2, Eq. (22). -/
def attention {n m h heads : ℕ} (p : MultiHead m h heads)
    (H : Sequence (n + 1) m) : Sequence (n + 1) m :=
  fun ir => affine p.projection p.bias
    (fun ar => headContext (p.head ar.1) H (ir.1, ar.2)) ir.2

/-- A concrete nonzero ReLU head witnessing Assumption 1; §3.3. -/
def unitHead : Head 1 1 where
  query := fun _ _ => 1
  key := fun _ _ => 1
  latentBias := fun _ => 0
  gateScale := fun _ => 1
  valueScale := fun _ => 1
  gateBias := fun _ => -1
  latentActivation := relu
  gateActivation := relu

/-- The witness uses positive weights and an admissible negative gate bias;
§3.3, Assumption 1 and Lemma 2. -/
theorem unitHead_conditions : HeadConditions unitHead :=
  ⟨fun _ _ => zero_le_one, fun _ _ => zero_le_one,
    fun _ => zero_le_one, fun _ => zero_le_one, relu_conditions, relu_conditions⟩

/-- A nonzero multi-head witness; §3.3, Assumption 1. -/
def unitAttention : MultiHead 1 1 1 where
  head := fun _ => unitHead
  projection := fun _ _ => 1
  bias := fun _ => -1

/-- Satisfiability of all attention constraints; §3.3, Assumption 1. -/
theorem unitAttention_conditions : MultiHeadConditions unitAttention :=
  ⟨fun _ => unitHead_conditions, fun _ _ => zero_le_one⟩

end Transformer.ICEoT
