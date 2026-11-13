module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RowCE

/-!
# The actual complex-head minibatch derivative

Source: `review/flat_spectrum.md`, CM33. Each observed input has its own
table row and one consistent raw label. Derive the normalized minibatch CE
direction through the actual complex head, then use that derivative in the
bounded update. Default rows do not occur in the observed training loss.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Matrix
open Landscape.Neural (crossEntropy)

variable {O Z C : Type*} [DecidableEq O] [Fintype C] [DecidableEq C]

/-- The normalized ordinary-CE derivative in observed table coordinates,
CM33. Defaults and unselected rows have zero derivative. -/
noncomputable def batchTableGradient (batch : Finset O) (labels : O → C)
    (T : Matrix (O ⊕ Z) C ℝ) : Matrix (O ⊕ Z) C ℝ :=
  fun s c => match s with
    | Sum.inl o => if o ∈ batch then rowCEGradient (T (Sum.inl o)) (labels o) c /
        (batch.card : ℝ) else 0
    | Sum.inr _ => 0

/-- Full-vocabulary CE averaged over exactly the selected observed
inputs, computed through actual real/imaginary coefficients, CM33. -/
noncomputable def batchHeadLoss [Fintype O] [Fintype Z] [DecidableEq Z]
    (U : Matrix (O ⊕ Z) (O ⊕ Z) ℂ) (batch : Finset O) (labels : O → C)
    (T : Matrix (O ⊕ Z) C ℝ) : ℝ :=
  (∑ o ∈ batch, crossEntropy
    (headLogits (complexFeatures U) (complexCoefficients (tableHead U T)) (Sum.inl o))
    (labels o)) / batch.card

/-- Pairing the complete table derivative with a direction retains
precisely the batch rows and their original normalization, CM33. -/
theorem batchTableGradient_pairing [Fintype O] [Fintype Z]
    (batch : Finset O) (labels : O → C) (T V : Matrix (O ⊕ Z) C ℝ) :
    (∑ s, ∑ c, batchTableGradient batch labels T s c * V s c) =
      (∑ o ∈ batch, ∑ c, rowCEGradient (T (Sum.inl o)) (labels o) c *
        V (Sum.inl o) c) / batch.card := by
  simp only [Fintype.sum_sum_type, batchTableGradient, ite_mul, zero_mul,
    Finset.sum_const_zero, add_zero, Finset.sum_ite_irrel]
  rw [Finset.sum_ite_mem, Finset.univ_inter, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro o ho
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro c hc
  ring

/-- Every directional derivative of the actual normalized complex-head
batch loss is the pairing with `batchTableGradient`, CM33. Unitarity is a
fixed feature premise; no derivative equality is assumed of a learned head. -/
theorem batchHeadLoss_hasDerivAt [Fintype O] [Fintype Z] [DecidableEq Z]
    (U : Matrix (O ⊕ Z) (O ⊕ Z) ℂ) (hU : U ∈ Matrix.unitaryGroup (O ⊕ Z) ℂ)
    (batch : Finset O) (labels : O → C) (T V : Matrix (O ⊕ Z) C ℝ) :
    HasDerivAt (fun a : ℝ => batchHeadLoss U batch labels (T + a • V))
      (∑ s, ∑ c, batchTableGradient batch labels T s c * V s c) 0 := by
  simp_rw [batchHeadLoss, tableHead_logits U hU]
  rw [batchTableGradient_pairing]
  have h := (HasDerivAt.fun_sum (u := batch) (fun o _ =>
    rowCE_hasDerivAt (T (Sum.inl o)) (V (Sum.inl o)) (labels o))).div_const
    (batch.card : ℝ)
  exact h

/-- Project a step of the derived full-vocabulary batch CE and restore
the declared defaults, CM33. Rates are supplied by the optimizer. -/
noncomputable def observedCEStep (A rate : ℝ) (defaults : Z → C → ℝ)
    (batch : Finset O) (labels : O → C) (T : Matrix (O ⊕ Z) C ℝ) :
    Matrix (O ⊕ Z) C ℝ :=
  boundedTableProposal A defaults (T - rate • batchTableGradient batch labels T)

/-- The selected row sees its CE direction scaled by the actual batch
size. An unselected row sees zero rate, CM33. -/
theorem observedCEStep_row (A rate : ℝ) (defaults : Z → C → ℝ)
    (batch : Finset O) (labels : O → C) (T : Matrix (O ⊕ Z) C ℝ) (o : O) :
    observedCEStep A rate defaults batch labels T (Sum.inl o) =
      rowCEStep A (if o ∈ batch then rate / batch.card else 0) (T (Sum.inl o)) (labels o) := by
  funext c
  by_cases ho : o ∈ batch
  · simp only [observedCEStep, boundedTableProposal, Matrix.sub_apply, Matrix.smul_apply,
      smul_eq_mul, batchTableGradient, ho, ite_true, rowCEStep]
    congr 1
    ring
  · simp only [observedCEStep, boundedTableProposal, Matrix.sub_apply, Matrix.smul_apply,
      smul_eq_mul, batchTableGradient, ho, ite_false, rowCEStep, mul_zero, zero_mul]

/-- Every batch CE step stays in the same convex domain, including a
step with a large nonnegative rate, CM33. Feasibility itself holds for any rate. -/
theorem observedCEStep_feasible {A : ℝ} (hA : 0 ≤ A) (rate : ℝ)
    (defaults : Z → C → ℝ) (hd : ∀ z c, |defaults z c| ≤ A)
    (batch : Finset O) (labels : O → C) (T : Matrix (O ⊕ Z) C ℝ) :
    observedCEStep A rate defaults batch labels T ∈ defaultTableDomain A defaults := by
  exact boundedTableProposal_feasible hA defaults hd _

/-- Empty-batch steps are identities on feasible states, CM33. This is
the zero-gradient boundary; native forward mode executes no training step. -/
theorem observedCEStep_empty (A rate : ℝ) (defaults : Z → C → ℝ) (labels : O → C)
    (T : Matrix (O ⊕ Z) C ℝ) (hT : T ∈ defaultTableDomain A defaults) :
    observedCEStep A rate defaults ∅ labels T = T := by
  have hg : batchTableGradient (Z := Z) ∅ labels T = 0 := by
    ext s c
    cases s <;> simp [batchTableGradient]
  rw [observedCEStep, hg, smul_zero, sub_zero]
  exact boundedTableProposal_eq defaults T hT.1 hT.2

example := batchTableGradient_pairing (Z := Bool) ({true} : Finset Bool) id
  (fun _ c => if c then (2 : ℝ) else -1) (fun _ c => if c then 3 else -2)

example := batchHeadLoss_hasDerivAt (categoricalFourierOn (Bool ⊕ Bool))
  (categoricalFourierOn_unitary _) ({true} : Finset Bool) id
  (fun _ c => if c then (2 : ℝ) else -1) (fun _ c => if c then 3 else -2)

example := observedCEStep_row 32 1 (fun (_ : Bool) (c : Bool) => if c then 24 else 0)
  ({true} : Finset Bool) id (fun _ c => if c then 24 else 0) false

example := observedCEStep_feasible (by norm_num : (0 : ℝ) ≤ 32) 1
  (fun (_ : Bool) (c : Bool) => if c then 24 else 0)
  (by intro z c; cases c <;> norm_num) ({true} : Finset Bool) id
  (fun _ c => if c then 24 else 0)

example := observedCEStep_empty 32 1 (fun (_ : Bool) (c : Bool) => if c then 24 else 0) id
  (fun _ c => if c then 24 else 0)
  (by
    constructor
    · intro s c
      cases c <;> norm_num
    · intro z
      rfl)

private def exampleDirection : Matrix (Bool ⊕ Bool) Bool ℝ :=
  fun s c => if s = Sum.inl true ∧ c = false then 1 else 0

/-- A real nonzero derivative through all four complex feature states.
The one-row minibatch has its actual unit normalization. -/
example : HasDerivAt (fun a : ℝ =>
    batchHeadLoss (categoricalFourierOn (Bool ⊕ Bool)) {true} id
      (0 + a • exampleDirection)) (1 / 2) 0 := by
  have hg : (∑ s, ∑ c, batchTableGradient ({true} : Finset Bool) id 0 s c *
      exampleDirection s c) = (1 / 2 : ℝ) := by
    norm_num [batchTableGradient, rowCEGradient, Geometry.probability,
      exampleDirection, Fintype.sum_sum_type]
  rw [← hg]
  exact batchHeadLoss_hasDerivAt (categoricalFourierOn (Bool ⊕ Bool))
    (categoricalFourierOn_unitary _) {true} id 0 exampleDirection

/-- Selecting both observed rows halves the same coordinate's derivative.
This checks full-batch versus minibatch averaging on the actual head. -/
example : HasDerivAt (fun a : ℝ =>
    batchHeadLoss (categoricalFourierOn (Bool ⊕ Bool)) Finset.univ id
      (0 + a • exampleDirection)) (1 / 4) 0 := by
  have hg : (∑ s, ∑ c, batchTableGradient (Finset.univ : Finset Bool) id 0 s c *
      exampleDirection s c) = (1 / 4 : ℝ) := by
    norm_num [batchTableGradient, rowCEGradient, Geometry.probability,
      exampleDirection, Fintype.sum_sum_type]
  rw [← hg]
  exact batchHeadLoss_hasDerivAt (categoricalFourierOn (Bool ⊕ Bool))
    (categoricalFourierOn_unitary _) Finset.univ id 0 exampleDirection

end Transformer.Grokking.FlatSpectrum.ConvexDepth
