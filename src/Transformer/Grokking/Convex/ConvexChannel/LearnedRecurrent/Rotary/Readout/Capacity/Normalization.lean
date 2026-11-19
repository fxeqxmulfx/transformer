module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Basic

/-!
# Exact normalization of every complete weighted readout block

Source: `review/convex/moments.md`, CM104. Zero blocks and coordinates remain
stored. Normalization reconstructs the original complete rows and does not
divide by a small-value threshold or remove a low-weight component.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair Optimality

/-- A zero full row capacity forces both complex coordinates to zero, CM104. -/
theorem row_zero (u : State) (hu : rowSize u = 0) : u = 0 := by
  have h₁ := norm_nonneg u.1
  have h₂ := norm_nonneg u.2
  unfold rowSize at hu
  apply Prod.ext
  · exact norm_eq_zero.mp (by linarith : ‖u.1‖ = 0)
  · exact norm_eq_zero.mp (by linarith : ‖u.2‖ = 0)

variable {O : Type*} [Fintype O] [Nonempty O]

/-- A zero block forces every original class row to zero, CM104. -/
theorem block_zero_rows (u : O → State) (hu : blockSize u = 0) (c : O) : u c = 0 := by
  apply row_zero
  have h := row_le_block u c
  have hn := rowSize_nonneg (u c)
  linarith

/-- Normalize all original class rows by their complete common capacity, CM104. -/
noncomputable def normalized (u : O → State) (c : O) : State := (blockSize u)⁻¹ • u c

/-- Every normalized class row is a complete admissible unit readout, CM104. -/
theorem normalized_bound (u : O → State) (c : O) : rowSize (normalized u c) ≤ 1 := by
  by_cases h : blockSize u = 0
  · simp [normalized, h, rowSize]
  · unfold normalized
    rw [rowSize_smul, abs_of_nonneg (inv_nonneg.mpr (block_nonneg u))]
    have hm := mul_le_mul_of_nonneg_left (row_le_block u c)
      (inv_nonneg.mpr (block_nonneg u))
    simpa only [inv_mul_cancel₀ h] using hm

/-- Scaling reconstructs every full complex row, including a zero block, CM104. -/
theorem restore (u : O → State) (c : O) : blockSize u • normalized u c = u c := by
  by_cases h : blockSize u = 0
  · rw [h, zero_smul, block_zero_rows u h c]
  · rw [normalized, smul_smul, mul_inv_cancel₀ h, one_smul]

/-- Real scaling of the complete left row has the actual full contraction, CM104. -/
theorem pairing_scale (r : ℝ) (u a : State) : pairing (r • u) a = r * pairing u a := by
  rw [pairing_comm, map_smul, pairing_comm a u, smul_eq_mul]

/-- Every arbitrary complex row is controlled by its complete block mass in
the actual coefficient contraction, CM104. -/
theorem block_pairing_lower (u : O → State) (c : O) (a : State) :
    -(blockSize u * ‖a‖) ≤ pairing (u c) a := by
  have h := mul_le_mul_of_nonneg_left
    (pairing_lower (normalized u c) a (normalized_bound u c)) (block_nonneg u)
  rw [← pairing_scale, restore] at h
  linarith

example : ((0, 0) : State) = 0 := row_zero _ (by simp [rowSize])

example : (0 : Bool → State) true = 0 := block_zero_rows _ block_zero true

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity
