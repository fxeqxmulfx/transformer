module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Parity

/-!
# Exact reuse of complete forward-only minibatch queries

Source: `review/convex/moments.md`, CM88. Central differences of actual ordinary
CE at every complete coordinate can reuse two class-bias queries per record.
A coordinate with response s uses step h/s; zero-response coordinates have
zero loss difference. This is an exact finite-query identity, not a claim that
finite differences equal derivatives or that any optimizer converges.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates

open scoped BigOperators
open LearnedHinge (Prefix)
open Identification

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- Ordinary full-vocabulary CE of the actual reconstructed parameter,
CM88. The target is used only by the loss. -/
noncomputable def recordLoss (r : RealCoordinate O T → ℝ)
    (tokens : Prefix O (T + 1)) (target : O) : ℝ :=
  sequenceLoss tokens target (decode r)

/-- Central difference from two complete parameter queries, CM88. -/
noncomputable def coordinateDifference (r : RealCoordinate O T → ℝ)
    (tokens : Prefix O (T + 1)) (target : O) (z : RealCoordinate O T) (step : ℝ) : ℝ :=
  (recordLoss (shift r z step) tokens target - recordLoss (shift r z (-step)) tokens target) /
    (2 * step)

/-- Exact reuse holds after ordinary cross entropy as well as before it,
CM88. No analytic CE derivative appears in this identity. -/
theorem loss_query_eq_bias (r : RealCoordinate O T → ℝ) (tokens : Prefix O (T + 1))
    (target : O) (z : RealCoordinate O T) (t : ℝ) :
    recordLoss (shift r z t) tokens target =
      recordLoss (shift r (biasCoordinate z.2.1) (t * response tokens z)) tokens target := by
  unfold recordLoss sequenceLoss
  rw [coordinate_query_eq_bias]

/-- A coordinate invisible to this record has exactly zero forward
difference at every step, CM88. Its stored parameter is not removed. -/
theorem difference_zero (r : RealCoordinate O T → ℝ) (tokens : Prefix O (T + 1))
    (target : O) (z : RealCoordinate O T) (step : ℝ) (hs : response tokens z = 0) :
    coordinateDifference r tokens target z step = 0 := by
  unfold coordinateDifference
  rw [loss_query_eq_bias r tokens target z step,
    loss_query_eq_bias r tokens target z (-step), hs]
  simp

/-- Every responding real or imaginary coordinate has the same two loss
queries as its class bias with a rescaled step, CM88. Multiplicity and sign
remain in the resulting difference coefficient. -/
theorem difference_reuse (r : RealCoordinate O T → ℝ) (tokens : Prefix O (T + 1))
    (target : O) (z : RealCoordinate O T) (step : ℝ) (hh : step ≠ 0)
    (hs : response tokens z ≠ 0) :
    coordinateDifference r tokens target z (step / response tokens z) =
      response tokens z * coordinateDifference r tokens target (biasCoordinate z.2.1) step := by
  unfold coordinateDifference
  rw [loss_query_eq_bias r tokens target z, loss_query_eq_bias r tokens target z]
  rw [div_mul_cancel₀ step hs, neg_mul, div_mul_cancel₀ step hs]
  field_simp

/-- Finite-query mean over all indexed supervised records, CM88. A native
update rejects an empty batch; the mathematical empty sum is zero. -/
noncomputable def forwardDirection (r : RealCoordinate O T → ℝ)
    (tokens : Fin N → Prefix O (T + 1)) (target : Fin N → O) (step : ℝ)
    (z : RealCoordinate O T) : ℝ :=
  (N : ℝ)⁻¹ * ∑ i, response (tokens i) z *
    coordinateDifference r (tokens i) (target i) (biasCoordinate z.2.1) step

/-- The reused minibatch direction equals the complete coordinate-query
mean, including zero responses, every duplicate record and the actual batch
length, CM88. There is no selected-coordinate or selected-class estimator. -/
theorem forwardDirection_complete (r : RealCoordinate O T → ℝ)
    (tokens : Fin N → Prefix O (T + 1)) (target : Fin N → O) (step : ℝ) (hh : step ≠ 0)
    (z : RealCoordinate O T) :
    forwardDirection r tokens target step z = (N : ℝ)⁻¹ *
      ∑ i, coordinateDifference r (tokens i) (target i) z (step / response (tokens i) z) := by
  unfold forwardDirection
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  by_cases hs : response (tokens i) z = 0
  · rw [difference_zero r (tokens i) (target i) z _ hs, hs, zero_mul]
  · exact (difference_reuse r (tokens i) (target i) z step hh hs).symm

/-- Nontrivial repeated-pair response used by the query-reuse control,
CM88. The same ordered pair occurs twice. -/
def repeated : Prefix Bool (4 + 1) := ⟨[false, true, false, true, false], by simp⟩

/-- A real weight for the repeated ordered-pair address, CM88. -/
def repeatedCoordinate : RealCoordinate Bool 4 :=
  (false, true, none, some (.inr (⟨0, by decide⟩, false, true)))

/-- The control's complete coordinate response is two, CM88. -/
theorem repeated_response : response repeated repeatedCoordinate = 2 := by
  have h := Parity.repeated_pair_count.1
  change contextRow (past repeated) (some (.inr (⟨0, by decide⟩, false, true))) = 2 at h
  simpa [response, repeatedCoordinate, queryRow] using congrArg Complex.re h

example : coordinateDifference (fun _ => 0) repeated true repeatedCoordinate (1 / 2) =
    2 * coordinateDifference (fun _ => 0) repeated true (biasCoordinate true) 1 := by
  have he : repeatedCoordinate.2.1 = true := rfl
  simpa only [repeated_response, he] using
    difference_reuse (fun _ => 0) repeated true repeatedCoordinate 1 (by norm_num)
      (by rw [repeated_response]; norm_num)

example : coordinateDifference (fun _ => 0) repeated true
    (false, true, some true, none) 1 = 0 := by
  apply difference_zero
  simp [response, queryRow, repeated]

example : forwardDirection (fun _ => 0) (fun _ : Fin 2 => repeated)
    (fun _ => true) 1 repeatedCoordinate = (2 : ℝ)⁻¹ *
      ∑ _ : Fin 2, coordinateDifference (fun _ => 0) repeated true repeatedCoordinate (1 / 2) := by
  simpa only [repeated_response, Nat.cast_ofNat] using
    forwardDirection_complete (fun _ => 0) (fun _ : Fin 2 => repeated)
      (fun _ => true) 1 (by norm_num) repeatedCoordinate

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates
