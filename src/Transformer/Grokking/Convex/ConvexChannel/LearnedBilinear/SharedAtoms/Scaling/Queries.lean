module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling.Basic

/-!
# Complete scaled responses and feasible ordinary-CE queries

Source: `review/convex/moments.md`, CM93. A scaled bias component changes a raw
class score by a(none)*b(none). Reserving capacity must account for this factor.
Every complete coordinate and repeated-record response receives its own positive
separable scale; no class or imaginary component is dropped.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling

open scoped BigOperators
open Identification
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

omit [Fintype O] [DecidableEq O] in
/-- A coordinate change commutes with the full positive geometry, CM93. -/
theorem diagonal_shift (w r : RealCoordinate O T → ℝ) (z : RealCoordinate O T) (t : ℝ) :
    Updates.diagonal w (Updates.shift r z t) =
      Updates.shift (Updates.diagonal w r) z (w z * t) := by
  classical
  funext y
  by_cases hy : y = z
  · subst y
    simp [Updates.diagonal, Updates.shift]
    ring
  · simp [Updates.diagonal, Updates.shift, hy]

/-- Every complete real/imaginary coordinate has its scaled actual
response, CM93. This is the chain rule identity used by the native mean. -/
theorem logit_shift (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (r : RealCoordinate O T → ℝ) (tokens : Prefix O (T + 1)) (z : RealCoordinate O T)
    (t : ℝ) (c : O) :
    sequenceLogits (decode (Updates.diagonal (weight a b) (Updates.shift r z t))) tokens c =
      sequenceLogits (decode (Updates.diagonal (weight a b) r)) tokens c +
        if c = z.2.1 then weight a b z * t * Updates.response tokens z else 0 := by
  rw [diagonal_shift, Updates.logit_shift]

/-- A query is feasible in the changed capacity region, CM93. Its actual
logit step is computed below; it is not the unscaled eta*B step. -/
theorem query_feasible (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (B eta : ℝ) (r : RealCoordinate O T → ℝ) (target : O) (s : ℝ)
    (hr : r ∈ SharedAtoms.domain B) (he : 0 ≤ eta) (he1 : eta ≤ 1) (hs : s ^ 2 ≤ 1) :
    Updates.diagonal (weight a b) (queryPoint B eta r target s) ∈ domain a b B :=
  ⟨_, SharedAtoms.query_feasible B eta r target s hr he he1 hs, rfl⟩

omit [Fintype O] in
/-- Exact complete parameter change for a scaled signed bias query, CM93. -/
theorem query_eq_shift (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (B eta : ℝ) (r : RealCoordinate O T → ℝ) (target : O) (s : ℝ) :
    Updates.diagonal (weight a b) (queryPoint B eta r target s) =
      Updates.shift (Updates.diagonal (weight a b) ((1 - eta) • r))
        (Updates.biasCoordinate target) (eta * B * (a none * b none) * s) := by
  rw [SharedAtoms.query_eq_shift, diagonal_shift]
  congr 1
  unfold weight Updates.biasCoordinate
  ring

/-- Original-class logits of the feasible scaled query, CM93. -/
theorem query_logits (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (B eta : ℝ) (r : RealCoordinate O T → ℝ) (target c : O) (s : ℝ)
    (tokens : Prefix O (T + 1)) :
    sequenceLogits (decode (Updates.diagonal (weight a b) (queryPoint B eta r target s))) tokens c =
      sequenceLogits (decode (Updates.diagonal (weight a b) ((1 - eta) • r))) tokens c +
        if c = target then eta * B * (a none * b none) * s else 0 := by
  rw [query_eq_shift, Updates.logit_shift, Updates.bias_response]
  simp only [Updates.biasCoordinate, mul_one]

/-- The two feasible losses are complete finite CE bias queries at the
correct scaled step, CM93. There is no analytic derivative on this side. -/
theorem feasible_difference (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (B eta : ℝ) (r : RealCoordinate O T → ℝ) (tokens : Prefix O (T + 1)) (target c : O) :
    (Updates.recordLoss (Updates.diagonal (weight a b) (queryPoint B eta r c 1)) tokens target -
      Updates.recordLoss (Updates.diagonal (weight a b) (queryPoint B eta r c (-1))) tokens target) /
      (2 * (eta * B * (a none * b none))) =
        Updates.coordinateDifference (Updates.diagonal (weight a b) ((1 - eta) • r))
          tokens target (Updates.biasCoordinate c) (eta * B * (a none * b none)) := by
  simp only [query_eq_shift, mul_one, mul_neg_one, Updates.coordinateDifference]

/-- Capacity needed for a requested positive logit step, CM93. In
particular a bias response 1/V needs eta=h*V/B, retaining the same CE step. -/
theorem query_fraction (B u h : ℝ) (hB : 0 < B) (hu : 0 < u) (hh : 0 ≤ h)
    (hsmall : h ≤ B * u) :
    0 ≤ h / (B * u) ∧ h / (B * u) ≤ 1 ∧ h / (B * u) * B * u = h := by
  have hp : 0 < B * u := mul_pos hB hu
  refine ⟨div_nonneg hh hp.le, (div_le_one hp).mpr hsmall, ?_⟩
  rw [mul_assoc, div_mul_cancel₀ _ hp.ne']

omit [Fintype O] in
/-- Complete indexed means retain their original counts after geometry
scaling, CM93. This applies to analytic residuals or finite CE differences. -/
theorem pullback_scaled (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (tokens : Fin N → Prefix O (T + 1)) (scalar : Fin N → O → ℝ) (z : RealCoordinate O T) :
    (N : ℝ)⁻¹ * (∑ i, (weight a b z * Updates.response (tokens i) z) * scalar i z.2.1) =
      weight a b z * Updates.pullback tokens scalar z := by
  simp only [Updates.pullback, mul_assoc, ← Finset.mul_sum]
  ring

example : 0 ≤ (1 : ℝ) / (8 * (1 / 4)) ∧ (1 : ℝ) / (8 * (1 / 4)) ≤ 1 ∧
    (1 : ℝ) / (8 * (1 / 4)) * 8 * (1 / 4) = 1 :=
  query_fraction 8 (1 / 4) 1 (by norm_num) (by norm_num) (by norm_num) (by norm_num)

example : Updates.diagonal (weight (fun _ : QueryCoordinate Bool => 1 / 2)
    (fun _ : ContextCoordinate Bool 1 => 1 / 2))
    (queryPoint 8 (1 / 2) 0 true (-1)) ∈ domain (fun _ => 1 / 2) (fun _ => 1 / 2) 8 :=
  query_feasible _ _ 8 (1 / 2) 0 true (-1) (SharedAtoms.zero_mem 8)
    (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling
