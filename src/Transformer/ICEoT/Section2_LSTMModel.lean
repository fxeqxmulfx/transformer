/-
# IC-EoT: the cited IC-LSTM equations

arXiv:2603.22095v2, §2.2.3, Eqs. (4)–(11). This is the actual
one-layer model, sufficient to test the source's claim about every depth.
The LaTeX source places the skip input OUTSIDE the dense activation in
Eq. (10); the model follows that source, not the PDF's ambiguous extraction.
-/

import Transformer.ICEoT.Section2_Recurrent

noncomputable section

namespace Transformer.ICEoT

/-- The strictly positive internal activations required by §2.2.3.
This is stronger than interpreting "strictly non-negative" as `≥ 0`. -/
def PositiveConvexMonotone (φ : ℝ → ℝ) : Prop := ConvexMonotone φ ∧ ∀ x, 0 < φ x

/-- Parameters of Eqs. (4)–(11), §2.2.3, for a single layer.
All four gates share exactly the same input and hidden matrices. -/
structure ICLSTM (d h out : ℕ) where
  input : Fin h → (Bool × Fin d) → ℝ
  hidden : Fin h → Fin h → ℝ
  forgetScale : Fin h → ℝ
  inputScale : Fin h → ℝ
  outputScale : Fin h → ℝ
  candidateScale : Fin h → ℝ
  forgetBias : Fin h → ℝ
  inputBias : Fin h → ℝ
  outputBias : Fin h → ℝ
  candidateBias : Fin h → ℝ
  forgetActivation : ℝ → ℝ
  inputActivation : ℝ → ℝ
  outputActivation : ℝ → ℝ
  candidateActivation : ℝ → ℝ
  hiddenActivation : ℝ → ℝ
  dense : (Bool × Fin d) → Fin h → ℝ
  denseBias : (Bool × Fin d) → ℝ
  denseActivation : ℝ → ℝ
  readout : Fin out → (Bool × Fin d) → ℝ
  readoutBias : Fin out → ℝ
  readoutActivation : ℝ → ℝ

/-- The listed IC-LSTM structural hypotheses, §2.2.3, using strictly
positive internal activations to avoid ambiguity in the source wording. -/
def ICLSTMConditions {d h out : ℕ} (p : ICLSTM d h out) : Prop :=
  Nonnegative p.input ∧ Nonnegative p.hidden ∧ Nonnegative p.dense ∧ Nonnegative p.readout ∧
  (∀ r, 0 ≤ p.forgetScale r ∧ 0 ≤ p.inputScale r ∧ 0 ≤ p.outputScale r ∧
    0 ≤ p.candidateScale r) ∧
  PositiveConvexMonotone p.forgetActivation ∧ PositiveConvexMonotone p.inputActivation ∧
  PositiveConvexMonotone p.outputActivation ∧ PositiveConvexMonotone p.candidateActivation ∧
  PositiveConvexMonotone p.hiddenActivation ∧ PositiveConvexMonotone p.denseActivation ∧
  ConvexMonotone p.readoutActivation

/-- Expansion of the current input in §2.2.3, Eqs. (4)–(7), (10). -/
def lstmExpand {d : ℕ} (x : Fin d → ℝ) : (Bool × Fin d) → ℝ :=
  fun r => if r.1 then -x r.2 else x r.2

/-- One real cell/hidden update, Eqs. (4)–(9), §2.2.3. The pair stores
the cell state first and hidden state second. -/
def lstmStep {d h out : ℕ} (p : ICLSTM d h out) (x : Fin d → ℝ)
    (state : (Fin h → ℝ) × (Fin h → ℝ)) : (Fin h → ℝ) × (Fin h → ℝ) :=
  let shared := affine p.input (fun _ => 0) (lstmExpand x) +
    affine p.hidden (fun _ => 0) state.2
  let forget := fun r => p.forgetActivation (p.forgetScale r * shared r + p.forgetBias r)
  let ingate := fun r => p.inputActivation (p.inputScale r * shared r + p.inputBias r)
  let outgate := fun r => p.outputActivation (p.outputScale r * shared r + p.outputBias r)
  let candidate := fun r => p.candidateActivation (p.candidateScale r * shared r + p.candidateBias r)
  let cell := fun r => forget r * state.1 r + ingate r * candidate r
  (cell, fun r => outgate r * p.hiddenActivation (cell r))

/-- The actual dense skip and output equations, Eqs. (10), (11), §2.2.3. -/
def lstmReadout {d h out : ℕ} (p : ICLSTM d h out) (x : Fin d → ℝ)
    (hidden : Fin h → ℝ) : Fin out → ℝ :=
  let dense := fun r => p.denseActivation (affine p.dense p.denseBias hidden r) + lstmExpand x r
  fun r => p.readoutActivation (affine p.readout p.readoutBias dense r)

/-- A three-step, one-layer IC-LSTM with fixed zero initial states;
§2.2.3, Eqs. (4)–(11). -/
def lstmThreePredict {d h out : ℕ} (p : ICLSTM d h out) (X : (Fin 3 × Fin d) → ℝ) :
    Fin out → ℝ :=
  let s1 := lstmStep p (fun r => X (0, r)) (0, 0)
  let s2 := lstmStep p (fun r => X (1, r)) s1
  let s3 := lstmStep p (fun r => X (2, r)) s2
  lstmReadout p (fun r => X (2, r)) s3.2

/-- A strictly positive convex non-decreasing activation, admissible under
every internal-activation requirement in §2.2.3. -/
def positiveRelu (x : ℝ) : ℝ := max 1 x

/-- Positivity includes every real argument, not only reachable states;
§2.2.3, the listed activation conditions. -/
theorem positiveRelu_conditions : PositiveConvexMonotone positiveRelu := by
  refine ⟨⟨(convexOn_const 1 convex_univ).sup (convexOn_id convex_univ), ?_⟩, ?_⟩
  · intro x y h
    exact max_le_max le_rfl h
  · intro x
    exact lt_of_lt_of_le zero_lt_one (le_max_left 1 x)

/-- A concrete IC-LSTM using shared matrices and non-negative diagonal
scales, satisfying all listed conditions; §2.2.3. The forget gate retains
the cross-time product that is absent from the IC-EoT product lemma. -/
def counterLSTM : ICLSTM 1 1 1 where
  input := fun _ r => if r.1 then 0 else 1
  hidden := fun _ _ => 0
  forgetScale := fun _ => 1
  inputScale := fun _ => 0
  outputScale := fun _ => 0
  candidateScale := fun _ => 0
  forgetBias := fun _ => 0
  inputBias := fun _ => 0
  outputBias := fun _ => 0
  candidateBias := fun _ => 0
  forgetActivation := positiveRelu
  inputActivation := positiveRelu
  outputActivation := positiveRelu
  candidateActivation := positiveRelu
  hiddenActivation := positiveRelu
  dense := fun r _ => if r.1 then 0 else 1
  denseBias := fun _ => 0
  denseActivation := positiveRelu
  readout := fun _ r => if r.1 then 0 else 1
  readoutBias := fun _ => 0
  readoutActivation := id

/-- The actual counterexample meets even strictly positive activation
conditions; §2.2.3, conditions following Eqs. (4)–(11). -/
theorem counterLSTM_conditions : ICLSTMConditions counterLSTM := by
  refine ⟨?_, ?_, ?_, ?_, ?_, positiveRelu_conditions, positiveRelu_conditions,
    positiveRelu_conditions, positiveRelu_conditions, positiveRelu_conditions,
    positiveRelu_conditions, convexOn_id convex_univ, monotone_id⟩
  · intro r s
    simp only [counterLSTM]
    split <;> norm_num
  · exact fun _ _ => le_rfl
  · intro r s
    simp only [counterLSTM]
    split <;> norm_num
  · intro r s
    simp only [counterLSTM]
    split <;> norm_num
  · exact fun _ => ⟨zero_le_one, le_rfl, le_rfl, le_rfl⟩

end Transformer.ICEoT
