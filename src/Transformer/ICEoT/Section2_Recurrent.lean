/-
# IC-EoT: the ICRNN background

arXiv:2603.22095v2, §2.2.2, Eqs. (2), (3). The initial hidden state
and preceding input are fixed. At stage `k` the output uses only inputs
through `k`; indexing a full sequence avoids padding any finite prefix.
-/

import Transformer.ICEoT.Section2_FeedForward

noncomputable section

namespace Transformer.ICEoT

/-- All six weight matrices of Eqs. (2), (3), §2.2.2. -/
structure RNN (I : Type*) (h out : ℕ) where
  input : Fin h → I → ℝ
  hidden : Fin h → Fin h → ℝ
  previousInput : Fin h → I → ℝ
  output : Fin out → Fin h → ℝ
  previousHidden : Fin out → Fin h → ℝ
  directInput : Fin out → I → ℝ
  hiddenActivation : ℝ → ℝ
  outputActivation : ℝ → ℝ

/-- Exactly the weight and activation assumptions stated in §2.2.2. -/
def RNNConditions {I : Type*} {h out : ℕ} (p : RNN I h out) : Prop :=
  Nonnegative p.input ∧ Nonnegative p.hidden ∧ Nonnegative p.previousInput ∧
    Nonnegative p.output ∧ Nonnegative p.previousHidden ∧ Nonnegative p.directInput ∧
    ConvexMonotone p.hiddenActivation ∧ ConvexMonotone p.outputActivation

/-- The preceding input, with a fixed initial value; §2.2.2, Eq. (2). -/
def recurrentPrevious {I : Type*} (X : (ℕ × I) → ℝ) (initial : I → ℝ) (k : ℕ) :
    I → ℝ := if k = 0 then initial else fun r => X (k - 1, r)

/-- The actual hidden recurrence of Eq. (2), §2.2.2. -/
def rnnHidden {I : Type*} [Fintype I] {h out : ℕ} (p : RNN I h out)
    (initialHidden : Fin h → ℝ) (initialInput : I → ℝ) (X : (ℕ × I) → ℝ) :
    ℕ → Fin h → ℝ
  | 0 => initialHidden
  | k + 1 => fun r => p.hiddenActivation
    (affine p.input (fun _ => 0) (fun s => X (k, s)) r +
      affine p.hidden (fun _ => 0) (rnnHidden p initialHidden initialInput X k) r +
      affine p.previousInput (fun _ => 0) (recurrentPrevious X initialInput k) r)

/-- The actual output of Eq. (3), §2.2.2. -/
def rnnOutput {I : Type*} [Fintype I] {h out : ℕ} (p : RNN I h out)
    (initialHidden : Fin h → ℝ) (initialInput : I → ℝ) (X : (ℕ × I) → ℝ)
    (k : ℕ) : Fin out → ℝ := fun r => p.outputActivation
  (affine p.output (fun _ => 0) (rnnHidden p initialHidden initialInput X (k + 1)) r +
    affine p.previousHidden (fun _ => 0) (rnnHidden p initialHidden initialInput X k) r +
    affine p.directInput (fun _ => 0) (fun s => X (k, s)) r)

/-- Convexity of the real ICRNN hidden recurrence; §2.2.2, Eq. (2). -/
theorem rnnHidden_convex {I : Type*} [Fintype I] {h out : ℕ} (p : RNN I h out)
    (initialHidden : Fin h → ℝ) (initialInput : I → ℝ) (hp : RNNConditions p) (k : ℕ) :
    ComponentwiseConvex (fun X => rnnHidden p initialHidden initialInput X k) := by
  induction k with
  | zero => exact convexOn_const _ convex_univ
  | succ k ih =>
    have hi : ComponentwiseConvex (fun X : (ℕ × I) → ℝ => fun s => X (k, s)) :=
      ⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩
    have hprev : ComponentwiseConvex (fun X => recurrentPrevious X initialInput k) := by
      by_cases hk : k = 0
      · simpa [recurrentPrevious, hk, ComponentwiseConvex] using
          (convexOn_const initialInput (convex_univ : Convex ℝ (Set.univ : Set ((ℕ × I) → ℝ))))
      · simp only [recurrentPrevious, hk, ↓reduceIte]
        exact ⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩
    exact convexMonotone_activation _ _ hp.2.2.2.2.2.2.1
      (((affine_convex p.input (fun _ => 0) _ hp.1 hi).add
        (affine_convex p.hidden (fun _ => 0) _ hp.2.1 ih)).add
        (affine_convex p.previousInput (fun _ => 0) _ hp.2.2.1 hprev))

/-- Nonzero weights and ReLU satisfy the hypotheses of the two recurrent
convexity theorems; §2.2.2. -/
def unitRNN : RNN (Bool × Fin 1) 1 1 where
  input := fun _ _ => 1
  hidden := fun _ _ => 1
  previousInput := fun _ _ => 1
  output := fun _ _ => 1
  previousHidden := fun _ _ => 1
  directInput := fun _ _ => 1
  hiddenActivation := relu
  outputActivation := relu

example : RNNConditions unitRNN :=
  ⟨fun _ _ => zero_le_one, fun _ _ => zero_le_one, fun _ _ => zero_le_one,
    fun _ _ => zero_le_one, fun _ _ => zero_le_one, fun _ _ => zero_le_one,
    ⟨relu_conditions.1, relu_conditions.2.1⟩, relu_conditions.1, relu_conditions.2.1⟩

/-- The background ICRNN output-convexity claim with all six matrices;
§2.2.2, Eqs. (2), (3). Activation values need not be non-negative. -/
theorem rnnOutput_convex {I : Type*} [Fintype I] {h out : ℕ} (p : RNN I h out)
    (initialHidden : Fin h → ℝ) (initialInput : I → ℝ) (hp : RNNConditions p) (k : ℕ) :
    ComponentwiseConvex (fun X => rnnOutput p initialHidden initialInput X k) := by
  have hi : ComponentwiseConvex (fun X : (ℕ × I) → ℝ => fun s => X (k, s)) :=
    ⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩
  exact convexMonotone_activation _ _ hp.2.2.2.2.2.2.2
    (((affine_convex p.output (fun _ => 0) _ hp.2.2.2.1
      (rnnHidden_convex p initialHidden initialInput hp (k + 1))).add
      (affine_convex p.previousHidden (fun _ => 0) _ hp.2.2.2.2.1
        (rnnHidden_convex p initialHidden initialInput hp k))).add
      (affine_convex p.directInput (fun _ => 0) _ hp.2.2.2.2.2.1 hi))

example : RNNConditions unitRNN :=
  ⟨fun _ _ => zero_le_one, fun _ _ => zero_le_one, fun _ _ => zero_le_one,
    fun _ _ => zero_le_one, fun _ _ => zero_le_one, fun _ _ => zero_le_one,
    ⟨relu_conditions.1, relu_conditions.2.1⟩, relu_conditions.1, relu_conditions.2.1⟩

/-- Expansion `[u,-u]` of the complete recurrent input sequence;
§2.2.2, following Eq. (3). -/
def expandTime {d : ℕ} (X : (ℕ × Fin d) → ℝ) : (ℕ × (Bool × Fin d)) → ℝ :=
  fun ir => if ir.2.1 then -X (ir.1, ir.2.2) else X (ir.1, ir.2.2)

/-- The expanded recurrent sequence depends affinely on the original;
§2.2.2, following Eq. (3). -/
theorem expandTime_combination {d : ℕ} (X Y : (ℕ × Fin d) → ℝ) (a b : ℝ) :
    expandTime (a • X + b • Y) = a • expandTime X + b • expandTime Y := by
  funext ir
  rcases ir with ⟨i, s, r⟩
  cases s <;> simp [expandTime, add_comm]

/-- Convexity in the original sequence with its negative copy included;
§2.2.2, Eqs. (2), (3). -/
theorem rnnOriginal_convex {d h out : ℕ} (p : RNN (Bool × Fin d) h out)
    (initialHidden : Fin h → ℝ) (initialInput : (Bool × Fin d) → ℝ)
    (hp : RNNConditions p) (k : ℕ) :
    ComponentwiseConvex (fun X => rnnOutput p initialHidden initialInput (expandTime X) k) := by
  refine ⟨convex_univ, ?_⟩
  intro X hX Y hY a b ha hb hab
  dsimp only
  rw [expandTime_combination]
  exact (rnnOutput_convex p initialHidden initialInput hp k).2
    (Set.mem_univ _) (Set.mem_univ _) ha hb hab

example : RNNConditions unitRNN :=
  ⟨fun _ _ => zero_le_one, fun _ _ => zero_le_one, fun _ _ => zero_le_one,
    fun _ _ => zero_le_one, fun _ _ => zero_le_one, fun _ _ => zero_le_one,
    ⟨relu_conditions.1, relu_conditions.2.1⟩, relu_conditions.1, relu_conditions.2.1⟩

end Transformer.ICEoT
