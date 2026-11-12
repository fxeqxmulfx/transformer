module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Basic

/-!
# Feasible sharpening of decoded predictions

Source: `review/flat_spectrum.md`, CM11. Make decoded active rows one-hot, keep
inactive padding uniform, then apply one common positive uniform mixture.
The decoded classes remain strict maxima. Measured scaled operator/nuclear
bounds certify feasibility. If observed decoded labels are correct, convexity
of actual ordinary CE bounds the resulting objective excess. This is endpoint
certification, not a claim that rounding is a convex map or improves unseen answers.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open scoped Matrix.Norms.L2Operator

variable {m n C F S : Type*} [Fintype C] [DecidableEq C]

/-- A decoded vertex on active cells, with uniform inactive padding (CM11). -/
noncomputable def decodedTable (active : m → n → Bool) (decision : m → n → C) :
    Table m n C := fun i j => if active i j then pointMass decision i j else uniform i j

/-- Masked decoding retains all probability constraints (CM11). -/
theorem rows_decodedTable [Nonempty C] (active : m → n → Bool) (decision : m → n → C) :
    Rows (decodedTable active decision) := by
  constructor
  · intro i j c
    cases h : active i j
    · simpa [decodedTable, h] using
        (rows_uniform : Rows (uniform : Table m n C)).1 i j c
    · simpa [decodedTable, h] using (rows_pointMass decision).1 i j c
  · intro i j
    cases h : active i j
    · simpa [decodedTable, h] using
        (rows_uniform : Rows (uniform : Table m n C)).2 i j
    · simpa [decodedTable, h] using (rows_pointMass decision).2 i j

/-- Positive mixing leaves the decoded class strictly above every alternative
on active cells. No correctness assumption about unseen labels is used (CM11). -/
theorem decoded_blend_strict (active : m → n → Bool) (decision : m → n → C)
    {α : ℝ} (ha : 0 < α) {i : m} {j : n} (hactive : active i j = true)
    {c : C} (hc : c ≠ decision i j) :
    blend α (decodedTable active decision) i j c <
      blend α (decodedTable active decision) i j (decision i j) := by
  apply (blend_lt_iff ha _ i j c (decision i j)).2
  simp [decodedTable, hactive, pointMass, hc]

/-- Correct observed decoded classes become exact one-hot fits before mixing
(CM11). The hypothesis concerns only the original observations. -/
theorem decoded_fit (active : m → n → Bool) (decision : m → n → C)
    (cell : S → m × n) (labels : S → C)
    (hactive : ∀ s, active (cell s).1 (cell s).2 = true)
    (hlabel : ∀ s, decision (cell s).1 (cell s).2 = labels s) (s : S) :
    decodedTable active decision (cell s).1 (cell s).2 =
      ConvexRecall.targetLogits (labels s) := by
  funext c
  simp [decodedTable, hactive s, pointMass, hlabel s, ConvexRecall.targetLogits]

variable [Fintype S]

/-- Actual CE excess after common mixing is at most the lost mixture mass
times the uniform-to-ideal loss gap. Repetitions remain in the original index S
(CM11). This holds against every probability-valued competitor. -/
theorem decoded_excess_le (active : m → n → Bool) (decision : m → n → C)
    (cell : S → m × n) (labels : S → C)
    (hactive : ∀ s, active (cell s).1 (cell s).2 = true)
    (hlabel : ∀ s, decision (cell s).1 (cell s).2 = labels s)
    {α : ℝ} (ha : 0 ≤ α) (ha1 : α ≤ 1) {Q : Table m n C} (hQ : Rows Q) :
    loss cell labels (blend α (decodedTable active decision)) - loss cell labels Q ≤
      (1 - α) * (loss cell labels uniform - idealLoss labels) := by
  have he := loss_eq_ideal_of_fit cell labels (decoded_fit active decision cell labels hactive hlabel)
  have hb := (loss_convex cell labels).2
    (Set.mem_univ (decodedTable active decision)) (Set.mem_univ uniform)
    ha (sub_nonneg.mpr ha1) (show α + (1 - α) = 1 by ring)
  change loss cell labels (blend α (decodedTable active decision)) ≤ _ at hb
  rw [he] at hb
  simp only [smul_eq_mul] at hb
  have hl := idealLoss_le cell labels hQ
  nlinarith

variable [Fintype m] [Fintype n] [DecidableEq n] [Nonempty C]

/-- Conditional certificate for the native sharpening step: all moment budgets
hold and the ordinary-CE excess is controlled by the same scalar (CM11).
The code is centered, and both norms are checked after decoding, not before it. -/
theorem certified_decoded_blend (active : m → n → Bool) (decision : m → n → C)
    (φ : F → C → ℂ) (R B : F → ℝ) (cell : S → m × n) (labels : S → C)
    (hφ : ∀ f, ∑ c, φ f c = 0)
    (hactive : ∀ s, active (cell s).1 (cell s).2 = true)
    (hlabel : ∀ s, decision (cell s).1 (cell s).2 = labels s)
    {α : ℝ} (ha : 0 ≤ α) (ha1 : α ≤ 1)
    (hR : ∀ f, α * ‖moment (φ f) (decodedTable active decision)‖ ≤ R f)
    (hB : ∀ f, α * nuclearNorm (moment (φ f) (decodedTable active decision)) ≤ B f) :
    BudgetFeasible φ R B (blend α (decodedTable active decision)) ∧
      ∀ Q : Table m n C, Rows Q →
        loss cell labels (blend α (decodedTable active decision)) - loss cell labels Q ≤
          (1 - α) * (loss cell labels uniform - idealLoss labels) := by
  refine ⟨budget_blend (rows_decodedTable active decision) hφ ha ha1 hR hB, ?_⟩
  intro Q hQ
  exact decoded_excess_le active decision cell labels hactive hlabel ha ha1 hQ

private def active : Fin 1 → Fin 1 → Bool := fun _ _ => true
private def decision : Fin 1 → Fin 1 → Bool := fun _ _ => true
private def code : Fin 1 → Bool → ℂ := fun _ c => if c then 1 else -1
private def cells : Fin 1 → Fin 1 × Fin 1 := fun _ => (0, 0)

private theorem witness_moment (f : Fin 1) :
    moment (code f) (decodedTable active decision) =
      (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
  ext i j
  fin_cases i
  fin_cases j
  norm_num [moment, decodedTable, active, decision, pointMass, code]

example := decoded_blend_strict active decision (by norm_num : (0 : ℝ) < 1 / 2)
  (i := 0) (j := 0) rfl (c := false) (by decide)

example := decoded_fit active decision cells (fun _ => true) (by intro s; rfl)
  (by intro s; rfl) 0

example := decoded_excess_le active decision cells (fun _ => true) (by intro s; rfl)
  (by intro s; rfl) (α := 1 / 2) (by norm_num) (by norm_num)
  (rows_uniform : Rows (uniform : Table (Fin 1) (Fin 1) Bool))

example := certified_decoded_blend active decision code (fun _ => 1) (fun _ => 1)
  cells (fun _ => true) (by intro f; simp [code]) (by intro s; rfl) (by intro s; rfl)
  (α := 1 / 2) (by norm_num) (by norm_num)
  (by intro f; rw [witness_moment]; norm_num)
  (by intro f; rw [witness_moment, Training.nuclearNorm_one_fin_one]; norm_num)

example := witness_moment 0

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
