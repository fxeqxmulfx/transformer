module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality
public import Mathlib.Data.Finset.Lattice.Fold

/-!
# Shared capacity across complete weighted complex readouts

Source: `review/convex/moments.md`, CM104. Every original class and component is
retained. The maximum across classes is the native mass required by a component;
the total budget can move between components without changing learned content.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair
open scoped BigOperators
variable {O : Type*} [Fintype O] [Nonempty O]

/-- Native mass needed for all complete class rows of one component, CM104. -/
noncomputable def blockSize (u : O → State) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun c => rowSize (u c))

/-- Every original class row is bounded by the full block capacity, CM104. -/
theorem row_le_block (u : O → State) (c : O) : rowSize (u c) ≤ blockSize u :=
  Finset.le_sup' (f := fun c => rowSize (u c)) (Finset.mem_univ c)

/-- The full block capacity is nonnegative, CM104. -/
theorem block_nonneg (u : O → State) : 0 ≤ blockSize u := by
  let c : O := Classical.choice inferInstance
  exact (rowSize_nonneg (u c)).trans (row_le_block u c)

/-- A common row bound controls the complete original-class maximum, CM104. -/
theorem block_le (u : O → State) (r : ℝ) (hu : ∀ c, rowSize (u c) ≤ r) :
    blockSize u ≤ r := Finset.sup'_le _ _ (fun c _ => hu c)

/-- A zero block keeps all its stored coordinates and uses no capacity, CM104. -/
theorem block_zero : blockSize (0 : O → State) = 0 := by
  simp [blockSize, rowSize]

/-- Shared capacity obeys the full weighted triangle bound, CM104. -/
theorem block_mix (u v : O → State) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    blockSize (a • u + b • v) ≤ a * blockSize u + b * blockSize v := by
  apply block_le
  intro c
  have hs := rowSize_add (a • u c) (b • v c)
  rw [rowSize_smul, rowSize_smul, abs_of_nonneg ha, abs_of_nonneg hb] at hs
  exact hs.trans (add_le_add (mul_le_mul_of_nonneg_left (row_le_block u c) ha)
    (mul_le_mul_of_nonneg_left (row_le_block v c) hb))

/-- Complete constant class rows have their actual complex row size, CM104. -/
theorem block_const (z : State) : blockSize (fun _ : O => z) = rowSize z := by
  unfold blockSize
  exact Finset.sup'_const _ _

variable {J : Type*} [Fintype J]

/-- Sum of native masses required by every learned component, CM104. -/
noncomputable def size (u : J → O → State) : ℝ := ∑ j, blockSize (u j)

/-- Joint weighted-readout capacity; all components/classes remain variables, CM104. -/
def jointDomain (B : ℝ) : Set (J → O → State) := {u | size u ≤ B}

/-- Complete joint capacity is nonnegative, CM104. -/
theorem size_nonneg (u : J → O → State) : 0 ≤ size u :=
  Finset.sum_nonneg (fun j _ => block_nonneg (u j))

/-- Weighted triangle bound permits capacity to move across components, CM104. -/
theorem size_mix (u v : J → O → State) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    size (a • u + b • v) ≤ a * size u + b * size v := by
  have h := Finset.sum_le_sum (s := Finset.univ) (fun (j : J) _ =>
    block_mix (u j) (v j) a b ha hb)
  simpa only [size, Pi.add_apply, Pi.smul_apply, Finset.sum_add_distrib,
    ← Finset.mul_sum] using h

/-- The complete weighted-readout shared-capacity set is convex, CM104. -/
theorem joint_convex (B : ℝ) : Convex ℝ (jointDomain (J := J) (O := O) B) := by
  intro u hu v hv a b ha hb hab
  have h := size_mix u v a b ha hb
  have h₁ := mul_le_mul_of_nonneg_left hu ha
  have h₂ := mul_le_mul_of_nonneg_left hv hb
  change size (a • u + b • v) ≤ B
  calc
    size (a • u + b • v) ≤ a * size u + b * size v := h
    _ ≤ a * B + b * B := add_le_add h₁ h₂
    _ = B := by rw [← add_mul, hab, one_mul]

/-- Every complete unit row table with nonnegative native weights fits the
shared domain when its total native mass fits, CM104. -/
theorem weighted_mem (B : ℝ) (w : J → ℝ) (u : J → O → State)
    (hw : ∀ j, 0 ≤ w j) (hm : ∑ j, w j ≤ B) (hu : u ∈ rowDomain J O) :
    (fun j c => w j • u j c) ∈ jointDomain B := by
  apply le_trans (Finset.sum_le_sum (s := Finset.univ) (fun (j : J) _ => ?_)) hm
  apply block_le
  intro c
  rw [rowSize_smul, abs_of_nonneg (hw j)]
  simpa only [mul_one] using mul_le_mul_of_nonneg_left (hu j c) (hw j)

example : blockSize (fun _ : Bool => (Complex.I, 0)) ≤ 1 :=
  block_le _ _ (by intro c; simp [rowSize])

example : blockSize ((1 / 2 : ℝ) • (fun _ : Bool => ((Complex.I, 0) : State)) +
    (1 / 2 : ℝ) • (fun _ : Bool => ((0, Complex.I) : State))) ≤
      (1 / 2 : ℝ) * blockSize (fun _ : Bool => (Complex.I, 0)) +
        (1 / 2 : ℝ) * blockSize (fun _ : Bool => (0, Complex.I)) :=
  block_mix _ _ _ _ (by norm_num) (by norm_num)

example : size ((1 / 2 : ℝ) • (fun (_ : Fin 1) (_ : Bool) => ((Complex.I, 0) : State)) +
    (1 / 2 : ℝ) • (fun (_ : Fin 1) (_ : Bool) => ((0, Complex.I) : State))) ≤
      (1 / 2 : ℝ) * size (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0)) +
        (1 / 2 : ℝ) * size (fun (_ : Fin 1) (_ : Bool) => (0, Complex.I)) :=
  size_mix _ _ _ _ (by norm_num) (by norm_num)

example : (fun (_ : Fin 1) (_ : Bool) => (16 : ℝ) • (Complex.I, 0)) ∈
    jointDomain 16 := weighted_mem _ (fun _ => 16) (fun _ _ => (Complex.I, 0))
      (by norm_num) (by simp) (by intro j c; simp [rowSize])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity
