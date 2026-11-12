module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Basic

/-!
# A single convex class admitting multiple spectral budgets

Source: `review/flat_spectrum.md`, CM7. Take the convex hull of a fixed collection of
budget-constrained tables. All coordinates are still probabilities and ordinary CE stays
convex. If any candidate budget yields a feasible exact observed fit, that table is also a
global CE minimum over the whole hull. This justifies a training-only certificate search;
if no candidate attains the bound, no claim of solving the hull optimization follows.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open scoped Matrix.Norms.L2Operator

variable {m n C F K : Type*} [Fintype m] [Fintype n] [DecidableEq n] [Fintype C]

/-- Direct probability coordinates in the convex hull of fixed spectral-budget sets
(CM7). The definition does not multiply independently trained mixture weights and tables. -/
def Hull (φ : F → C → ℂ) (R B : K → F → ℝ) : Set (Table m n C) :=
  convexHull ℝ (⋃ k, {P | BudgetFeasible φ (R k) (B k) P})

/-- Allowing these spectral geometries within one convex hull preserves convexity (CM7). -/
theorem hull_convex (φ : F → C → ℂ) (R B : K → F → ℝ) :
    Convex ℝ (Hull (m := m) (n := n) φ R B) := convex_convexHull _ _

/-- Every feasible candidate is a feasible point of the common hull (CM7). -/
theorem hull_member (φ : F → C → ℂ) (R B : K → F → ℝ) {k : K} {P : Table m n C}
    (hP : BudgetFeasible φ (R k) (B k) P) : P ∈ Hull φ R B :=
  subset_convexHull ℝ _ (Set.mem_iUnion.mpr ⟨k, hP⟩)

/-- The whole hull retains the same probability constraints and CE lower bound (CM7). -/
theorem hull_rows (φ : F → C → ℂ) (R B : K → F → ℝ) {P : Table m n C}
    (hP : P ∈ Hull φ R B) : Rows P := by
  apply convexHull_min (t := {Q : Table m n C | Rows Q}) ?_ rows_convex hP
  intro Q hQ
  obtain ⟨k, hk⟩ := Set.mem_iUnion.mp hQ
  exact hk.1.1

variable {S : Type*} [Fintype S]

/-- Ordinary CE is convex on the common model class (CM7). -/
theorem hull_loss_convex (φ : F → C → ℂ) (R B : K → F → ℝ)
    (cell : S → m × n) (labels : S → C) : ConvexOn ℝ (Hull φ R B) (loss cell labels) := by
  refine ⟨hull_convex φ R B, ?_⟩
  intro P _ Q _ a b ha hb hab
  exact (loss_convex cell labels).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

variable [DecidableEq C]

/-- A feasible exact fit from any one budget globally minimizes CE over the full hull
(CM7). It need not minimize a surrogate or use held-out labels for model selection. -/
theorem hull_fit_minimum (φ : F → C → ℂ) (R B : K → F → ℝ) (cell : S → m × n)
    (labels : S → C) {k : K} {P : Table m n C} (hP : BudgetFeasible φ (R k) (B k) P)
    (hfit : ∀ s, P (cell s).1 (cell s).2 = ConvexRecall.targetLogits (labels s)) :
    P ∈ Hull φ R B ∧ IsMinOn (loss cell labels) (Hull φ R B) P := by
  refine ⟨hull_member φ R B hP, ?_⟩
  intro Q hQ
  rw [loss_eq_ideal_of_fit cell labels hfit]
  exact idealLoss_le cell labels (hull_rows φ R B hQ)

private def witness : Table (Fin 1) (Fin 1) Bool := pointMass (fun _ _ => true)
private def code : Fin 1 → Bool → ℂ := ![fun c => if c then 1 else -1]
private def radius (k : Bool) : Fin 1 → ℝ := ![if k then 2 else 1]

private theorem witness_budget : BudgetFeasible code (radius false) (radius false) witness := by
  have hm (f : Fin 1) : moment (code f) witness = (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
    rw [witness, moment_pointMass]
    ext i j
    fin_cases f
    fin_cases i
    fin_cases j
    simp [code]
  refine ⟨⟨rows_pointMass _, ?_⟩, ?_⟩
  · intro f
    rw [hm f]
    fin_cases f
    simp [radius]
  · intro f
    rw [hm f, Training.nuclearNorm_one_fin_one]
    fin_cases f
    norm_num [radius]

example := hull_member code radius radius witness_budget
example := hull_rows code radius radius (hull_member code radius radius witness_budget)
example := hull_fit_minimum code radius radius (fun (_ : Fin 1) => (0, 0)) (fun _ => true)
  witness_budget (by intro s; funext c; simp [witness, pointMass, ConvexRecall.targetLogits])

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
