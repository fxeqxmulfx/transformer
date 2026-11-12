module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Blend
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Loss
public import Transformer.Grokking.FlatSpectrum.NuclearBounds
public import Transformer.Grokking.FlatSpectrum.Training.MaxMargin
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.NormComparison

/-!
# Convex budgets for the largest and total singular values

Source: `review/flat_spectrum.md`, CM5. Add a nuclear-norm upper bound to each shared
complex moment. The largest-singular-value and sum-of-singular-values bounds are both
convex. Uniform feasibility, the observed CE certificate and common-mixture repair survive.
The budgets are fixed parameters, not functions of the unobserved answers.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Matrix Finset
open scoped Matrix.Norms.L2Operator

variable {m n C F : Type*} [Fintype m] [Fintype n] [DecidableEq n] [Fintype C]

/-- Shared probability moments with both operator and nuclear upper bounds (CM5). -/
def BudgetFeasible (φ : F → C → ℂ) (R B : F → ℝ) (P : Table m n C) : Prop :=
  Feasible φ R P ∧ ∀ f, nuclearNorm (moment (φ f) P) ≤ B f

/-- The full-rank control keeps precisely the previous feasible domain (CM5). -/
theorem budget_at_card_iff (φ : F → C → ℂ) (R : F → ℝ) (P : Table m n C) :
    BudgetFeasible φ R (fun f => Fintype.card n * R f) P ↔ Feasible φ R P := by
  constructor
  · exact fun h => h.1
  · intro h
    exact ⟨h, fun f => nuclear_le_card_operator (h.2 f)⟩

/-- The two kinds of singular-value budget preserve joint convexity (CM5). -/
theorem budget_convex (φ : F → C → ℂ) (R B : F → ℝ) :
    Convex ℝ {P : Table m n C | BudgetFeasible φ R B P} := by
  intro P hP Q hQ a b ha hb hab
  refine ⟨feasible_convex φ R hP.1 hQ.1 ha hb hab, ?_⟩
  intro f
  rw [moment_linear]
  calc
    nuclearNorm (a • moment (φ f) P + b • moment (φ f) Q) ≤
        nuclearNorm (a • moment (φ f) P) + nuclearNorm (b • moment (φ f) Q) :=
      nuclearNorm_add_le _ _
    _ = a * nuclearNorm (moment (φ f) P) + b * nuclearNorm (moment (φ f) Q) := by
      rw [nuclearNorm_real_smul_eq_abs, nuclearNorm_real_smul_eq_abs,
        abs_of_nonneg ha, abs_of_nonneg hb]
    _ ≤ a * B f + b * B f := add_le_add (mul_le_mul_of_nonneg_left (hP.2 f) ha)
      (mul_le_mul_of_nonneg_left (hQ.2 f) hb)
    _ = B f := by rw [← add_mul, hab, one_mul]

/-- Centered codes admit uniform probabilities under any nonnegative budgets (CM5). -/
theorem budget_uniform [Nonempty C] {φ : F → C → ℂ} {R B : F → ℝ}
    (hφ : ∀ f, ∑ c, φ f c = 0) (hR : ∀ f, 0 ≤ R f) (hB : ∀ f, 0 ≤ B f) :
    BudgetFeasible φ R B (uniform : Table m n C) := by
  refine ⟨feasible_uniform hφ hR, ?_⟩
  intro f
  rw [moment_uniform_zero (hφ f), nuclearNorm_zero]
  exact hB f

/-- One common mixture can simultaneously enforce both singular-value budgets (CM5). -/
theorem budget_blend [Nonempty C] {φ : F → C → ℂ} {R B : F → ℝ} {P : Table m n C}
    (hP : Rows P) (hφ : ∀ f, ∑ c, φ f c = 0) {α : ℝ} (ha : 0 ≤ α) (ha1 : α ≤ 1)
    (hR : ∀ f, α * ‖moment (φ f) P‖ ≤ R f)
    (hB : ∀ f, α * nuclearNorm (moment (φ f) P) ≤ B f) :
    BudgetFeasible φ R B (blend α P) := by
  refine ⟨feasible_blend hP hφ ha ha1 hR, ?_⟩
  intro f
  rw [moment_blend (hφ f), nuclearNorm_real_smul_eq_abs, abs_of_nonneg ha]
  exact hB f

variable {S : Type*} [Fintype S] [DecidableEq C]

/-- Feasible exact observed fits still minimize ordinary cross-entropy (CM5). -/
theorem budget_fit_minimum (φ : F → C → ℂ) (R B : F → ℝ) (cell : S → m × n)
    (labels : S → C) {P : Table m n C} (hP : BudgetFeasible φ R B P)
    (hfit : ∀ s, P (cell s).1 (cell s).2 = ConvexRecall.targetLogits (labels s)) :
    BudgetFeasible φ R B P ∧ IsMinOn (loss cell labels) {Q | BudgetFeasible φ R B Q} P := by
  refine ⟨hP, ?_⟩
  intro Q hQ
  rw [loss_eq_ideal_of_fit cell labels hfit]
  exact idealLoss_le cell labels hQ.1.1

example := budget_uniform (m := Fin 1) (n := Fin 1)
  (φ := fun (_ : Fin 1) (c : Bool) => if c then Complex.I else -Complex.I)
  (R := fun _ => 1) (B := fun _ => 2) (by intro f; simp)
  (by intro f; norm_num) (by intro f; norm_num)

example := budget_blend (m := Fin 1) (n := Fin 1) (P := uniform)
  (φ := fun (_ : Fin 1) (c : Bool) => if c then Complex.I else -Complex.I)
  (R := fun _ => 1) (B := fun _ => 2) rows_uniform (by intro f; simp)
  (α := 1 / 2) (by norm_num) (by norm_num)
  (by intro f; rw [moment_uniform_zero (by simp), norm_zero]; norm_num)
  (by intro f; rw [moment_uniform_zero (by simp), nuclearNorm_zero]; norm_num)

private def witness : Table (Fin 1) (Fin 1) Bool := pointMass (fun _ _ => true)

private theorem witness_budget :
    BudgetFeasible (fun (_ : Fin 1) (c : Bool) => if c then (1 : ℂ) else -1)
      (fun _ => 1) (fun _ => 1) witness := by
  have hm : moment (fun c : Bool => if c then (1 : ℂ) else -1) witness =
      (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
    rw [witness, moment_pointMass]
    ext i j
    fin_cases i
    fin_cases j
    simp
  refine ⟨⟨rows_pointMass _, ?_⟩, ?_⟩
  · intro f
    rw [hm]
    simp
  · intro f
    rw [hm, Training.nuclearNorm_one_fin_one]

example := budget_fit_minimum _ _ _ (fun (_ : Fin 1) => (0, 0)) (fun _ => true)
  witness_budget (by intro s; funext c; simp [witness, pointMass, ConvexRecall.targetLogits])

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
