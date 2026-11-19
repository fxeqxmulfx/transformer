module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Order
public import Mathlib.Analysis.Normed.Module.RCLike.Real

/-!
# Complete complex readout coordinates and radial updates

Source: `review/convex/moments.md`, CM100. The two complex coefficients of every
class/component remain coordinates. The actual radial rule used by the native
row budget preserves feasibility; no Euclidean-projection or descent claim is made.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout

open Pair

/-- Explicit full readout row capacity, CM100. -/
noncomputable def rowSize (u : State) : ℝ := ‖u.1‖ + ‖u.2‖

/-- The full row capacity is nonnegative, CM100. -/
theorem rowSize_nonneg (u : State) : 0 ≤ rowSize u := add_nonneg (norm_nonneg _) (norm_nonneg _)

/-- Both complete complex coefficients enter the triangle bound, CM100. -/
theorem rowSize_add (u v : State) : rowSize (u + v) ≤ rowSize u + rowSize v := by
  have h₁ := norm_add_le u.1 v.1
  have h₂ := norm_add_le u.2 v.2
  change ‖u.1 + v.1‖ + ‖u.2 + v.2‖ ≤ (‖u.1‖ + ‖u.2‖) + (‖v.1‖ + ‖v.2‖)
  linarith

/-- Real scaling retains both real and imaginary coordinates, CM100. -/
theorem rowSize_smul (a : ℝ) (u : State) : rowSize (a • u) = |a| * rowSize u := by
  change ‖a • u.1‖ + ‖a • u.2‖ = |a| * (‖u.1‖ + ‖u.2‖)
  rw [norm_smul, norm_smul, Real.norm_eq_abs]
  ring

/-- Convex combinations preserve the actual numerical row budget, CM100. -/
theorem rowSize_mix (u v : State) (a b : ℝ) (hu : rowSize u ≤ 1)
    (hv : rowSize v ≤ 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    rowSize (a • u + b • v) ≤ 1 := by
  calc
    rowSize (a • u + b • v) ≤ rowSize (a • u) + rowSize (b • v) := rowSize_add _ _
    _ = a * rowSize u + b * rowSize v := by
      rw [rowSize_smul, rowSize_smul, abs_of_nonneg ha, abs_of_nonneg hb]
    _ ≤ a * 1 + b * 1 := add_le_add (mul_le_mul_of_nonneg_left hu ha)
      (mul_le_mul_of_nonneg_left hv hb)
    _ = 1 := by linarith

/-- Actual radial row rule, CM100; an inside row is unchanged. -/
noncomputable def projectRow (u : State) : State :=
  if rowSize u ≤ 1 then u else (rowSize u)⁻¹ • u

/-- The explicit radial rule satisfies the full row budget, CM100. -/
theorem projectRow_bound (u : State) : rowSize (projectRow u) ≤ 1 := by
  by_cases h : rowSize u ≤ 1
  · simp [projectRow, h]
  · have hp : 0 < rowSize u := lt_trans (by norm_num) (lt_of_not_ge h)
    rw [projectRow, ite_eq_right h, rowSize_smul, abs_of_pos (inv_pos.mpr hp),
      inv_mul_cancel₀ (ne_of_gt hp)]

/-- Feasible rows retain their complete coordinates exactly, CM100. -/
theorem projectRow_eq (u : State) (hu : rowSize u ≤ 1) : projectRow u = u := by
  rw [projectRow, ite_eq_left hu]

variable (J O : Type*)

/-- All original-class/component rows vary jointly in this set, CM100. -/
def rowDomain : Set (J → O → State) := {u | ∀ j c, rowSize (u j c) ≤ 1}

/-- Joint convexity in every real and imaginary readout coordinate, CM100. -/
theorem rowDomain_convex : Convex ℝ (rowDomain J O) := by
  intro u hu v hv a b ha hb hab j c
  exact rowSize_mix (u j c) (v j c) a b (hu j c) (hv j c) ha hb hab

/-- Radially updating every row stays in the complete domain, CM100. -/
theorem project_table_mem (u : J → O → State) :
    (fun j c => projectRow (u j c)) ∈ rowDomain J O :=
  fun j c => projectRow_bound (u j c)

example : rowSize ((1 / 2 : ℝ) • (Complex.I, 0) +
    (1 / 2 : ℝ) • (0, Complex.I)) ≤ 1 :=
  rowSize_mix _ _ _ _ (by simp [rowSize]) (by simp [rowSize])
    (by norm_num) (by norm_num) (by norm_num)

example : projectRow (Complex.I, 0) = (Complex.I, 0) :=
  projectRow_eq _ (by simp [rowSize])

example : (fun (_ : Fin 2) (c : Bool) => if c then (Complex.I, 0)
    else (0, Complex.I)) ∈ rowDomain (Fin 2) Bool := by
  intro j c
  cases c <;> simp [rowSize]

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout
