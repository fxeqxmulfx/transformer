module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Intermediate
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.PredictionHull

/-!
# Spectral coordinates on fixed input groups

Source: `review/flat_spectrum.md`, CM10. Learn a spectral probability table on fixed
row and column groups, then look up its entries for original inputs. The fixed map
is linear in the learned table; probability constraints and CE convexity survive.
The empirical loss retains the original observation index, including repetitions.
Correctness of a map inferred from observed label agreements is not asserted here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

variable {m n u v C F O : Type*}

/-- Lift a table on fixed groups to the original input domain (CM10). -/
def pullback (row : m → u) (column : n → v) (Q : Table u v C) : Table m n C :=
  fun i j => Q (row i) (column j)

variable [Fintype u] [Fintype v] [DecidableEq v] [Fintype C]

/-- Spectral bounds apply in the grouped coordinates, before fixed lookup (CM10). -/
def quotientFamily (row : m → u) (column : n → v) (φ : F → C → ℂ) (R B : F → ℝ) :
    Set (Table m n C) := pullback row column '' {Q | BudgetFeasible φ R B Q}

/-- Fixed grouping yields a convex class in the original prediction coordinates (CM10).
There is no product of freely trained grouping coefficients and probability entries. -/
theorem quotientFamily_convex (row : m → u) (column : n → v)
    (φ : F → C → ℂ) (R B : F → ℝ) : Convex ℝ (quotientFamily row column φ R B) := by
  rintro P ⟨Q, hQ, rfl⟩ P' ⟨Q', hQ', rfl⟩ a b ha hb hab
  refine ⟨a • Q + b • Q', budget_convex φ R B hQ hQ' ha hb hab, ?_⟩
  ext i j c
  rfl

/-- Every lifted candidate remains a probability-valued prediction (CM10). -/
theorem quotientFamily_rows (row : m → u) (column : n → v)
    (φ : F → C → ℂ) (R B : F → ℝ) {P : Table m n C}
    (hP : P ∈ quotientFamily row column φ R B) : Rows P := by
  obtain ⟨Q, hQ, rfl⟩ := hP
  exact lookup_rows (fun i j => (row i, column j)) hQ.1.1

variable [Fintype O] [DecidableEq C]

omit [Fintype u] [Fintype v] [DecidableEq v] [DecidableEq C] in
/-- Original empirical CE is exactly CE indexed by the original observations on their
grouped cells. Noninjective group maps do not deduplicate or reweight examples (CM10). -/
theorem pullback_loss (row : m → u) (column : n → v) (cell : O → m × n)
    (labels : O → C) (Q : Table u v C) :
    loss cell labels (pullback row column Q) =
      loss (fun o => (row (cell o).1, column (cell o).2)) labels Q := by
  unfold loss pullback
  apply Finset.sum_congr rfl
  intro o _
  rfl

/-- A feasible grouped exact fit is a global minimum of the original ordinary CE on
the grouped class, retaining every original observation (CM10). -/
theorem quotient_fit_minimum (row : m → u) (column : n → v) (φ : F → C → ℂ)
    (R B : F → ℝ) (cell : O → m × n) (labels : O → C) {Q : Table u v C}
    (hQ : BudgetFeasible φ R B Q)
    (hfit : ∀ o, Q (row (cell o).1) (column (cell o).2) = ConvexRecall.targetLogits (labels o)) :
    pullback row column Q ∈ quotientFamily row column φ R B ∧
      IsMinOn (loss cell labels) (quotientFamily row column φ R B) (pullback row column Q) := by
  refine ⟨⟨Q, hQ, rfl⟩, ?_⟩
  intro P hP
  rw [loss_eq_ideal_of_fit (P := pullback row column Q) cell labels hfit]
  exact idealLoss_le cell labels (quotientFamily_rows row column φ R B hP)

private def groupIndex (b : Bool) : Fin 4 := if b then 2 else 0

example := quotientFamily_rows groupIndex groupIndex _ _ _
  (show pullback groupIndex groupIndex intermediateTable ∈
      quotientFamily groupIndex groupIndex intermediateCode (fun _ => √8) (fun _ => 2 * √8) from
    ⟨intermediateTable, intermediate_admitted, rfl⟩)

example := quotient_fit_minimum groupIndex groupIndex _ _ _
  (fun p : Bool × Bool => p) (fun p => !(p.1 && p.2)) intermediate_admitted (by
    rintro ⟨a, b⟩
    funext c
    cases a <;> cases b <;>
      simp [groupIndex, intermediateTable, pointMass, intermediateLabel, ConvexRecall.targetLogits])

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
