module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Support

/-!
# Training-input scaling of complete coordinate updates

Source: `review/convex/moments.md`, CM89. Rare input responses can make plain
coordinate SGD slow. Positive diagonal scaling changes the update geometry,
not the learned family or its ordinary CE. The scale uses only indexed raw
training prefixes, with repeated records and squared pair multiplicities.
No convergence or unseen-answer guarantee is inferred from these identities.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates

open scoped BigOperators
open LearnedHinge (Prefix)
open Identification

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- An input-only positive scale on every complete real coordinate, CM89.
The extra one keeps even unobserved coordinates in the unchanged domain. -/
noncomputable def frequencyScale (tokens : Fin N → Prefix O (T + 1))
    (z : RealCoordinate O T) : ℝ :=
  ((N : ℝ) + 1) / (1 + ∑ i, response (tokens i) z ^ 2)

omit [Fintype O] in
/-- No coordinate is removed by training-input scaling, CM89. -/
theorem frequencyScale_pos (tokens : Fin N → Prefix O (T + 1))
    (z : RealCoordinate O T) : 0 < frequencyScale tokens z := by
  have hs : 0 ≤ ∑ i, response (tokens i) z ^ 2 :=
    Finset.sum_nonneg fun i _ => sq_nonneg _
  unfold frequencyScale
  positivity

omit [Fintype O] in
/-- An invisible coordinate still has strictly positive scale N+1, CM89.
It receives zero update because its complete pullback vanishes. -/
theorem frequencyScale_unseen (tokens : Fin N → Prefix O (T + 1))
    (z : RealCoordinate O T) (hz : ∀ i, response (tokens i) z = 0) :
    frequencyScale tokens z = (N : ℝ) + 1 := by
  simp [frequencyScale, hz]

omit [Fintype O] in
/-- Every class bias keeps unit update scale, CM89. -/
theorem frequencyScale_bias (tokens : Fin N → Prefix O (T + 1)) (c : O) :
    frequencyScale tokens (biasCoordinate c) = 1 := by
  simp only [frequencyScale, bias_response, one_pow, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  have hn : (1 : ℝ) + N ≠ 0 := by positivity
  rw [add_comm (N : ℝ) 1, div_self hn]

omit [Fintype O] in
/-- The scale does not select classes or one complex component, CM89.
The response sign disappears only after squaring its full multiplicity. -/
theorem frequencyScale_same_address (tokens : Fin N → Prefix O (T + 1))
    (b e : Bool) (c a : O) (q : QueryCoordinate O) (d : ContextCoordinate O T) :
    frequencyScale tokens (b, c, q, d) = frequencyScale tokens (e, a, q, d) := by
  cases b <;> cases e <;>
    simp [frequencyScale, response, mul_pow]

omit [Fintype O] [DecidableEq O] in
/-- A diagonal change of the complete real coordinates, CM89. -/
def diagonal (w : RealCoordinate O T → ℝ) :
    (RealCoordinate O T → ℝ) →ₗ[ℝ] (RealCoordinate O T → ℝ) where
  toFun r z := w z * r z
  map_add' r s := by ext z; exact mul_add _ _ _
  map_smul' a r := by ext z; exact mul_left_comm _ _ _

omit [Fintype O] [DecidableEq O] in
/-- Nonzero scaling is invertible on every coordinate, CM89. It imposes no
rank, support or answer restriction on the learned parameter domain. -/
theorem diagonal_inverse (w r : RealCoordinate O T → ℝ) (hw : ∀ z, w z ≠ 0) :
    diagonal (fun z => (w z)⁻¹) (diagonal w r) = r := by
  ext z
  change (w z)⁻¹ * (w z * r z) = r z
  rw [← mul_assoc, inv_mul_cancel₀ (hw z), one_mul]

omit [Fintype O] [DecidableEq O] in
/-- Complete real coordinates decode linearly into the original complex
domain, CM89. Real and imaginary weights remain independently trainable. -/
def decodeLinear : (RealCoordinate O T → ℝ) →ₗ[ℝ] SequenceParameters O T where
  toFun := decode
  map_add' r s := by ext c q d; rfl
  map_smul' a r := by
    funext c q d
    apply Complex.ext <;> simp [decode, Complex.real_smul]

/-- Ordinary CE remains jointly convex after any fixed diagonal change of
coordinates, CM89. Actual inference still recovers both learned encoders. -/
theorem scaled_loss_convex (tokens : Prefix O (T + 1)) (target : O)
    (w : RealCoordinate O T → ℝ) :
    ConvexOn ℝ Set.univ (fun r => recordLoss (diagonal w r) tokens target) := by
  have h := (sequenceLoss_convex tokens target).comp_linearMap
    (decodeLinear.comp (diagonal w))
  simpa only [Set.preimage_univ, Function.comp_def, LinearMap.coe_comp,
    decodeLinear, LinearMap.coe_mk, AddHom.coe_mk, recordLoss] using h

omit [Fintype O] in
/-- Positive input scaling preserves exact zero-support updates, CM89.
The same identity holds for analytic or measured finite-query residuals. -/
theorem scaled_zero_update (r w : RealCoordinate O T → ℝ)
    (tokens : Fin N → Prefix O (T + 1)) (scalar : Fin N → O → ℝ)
    (rate : ℝ) (z : RealCoordinate O T) (hr : r z = 0)
    (hz : ∀ i, response (tokens i) z = 0) :
    r z - rate * w z * pullback tokens scalar z = 0 := by
  rw [hr, pullback_zero tokens scalar z hz, mul_zero, sub_zero]

/-- Scaling preserves the complete mean of every coordinate's actual CE
queries, CM89. All indexed records and their multiplicities remain present. -/
theorem scaled_forward_complete (r w : RealCoordinate O T → ℝ)
    (tokens : Fin N → Prefix O (T + 1)) (target : Fin N → O)
    (step : ℝ) (hh : step ≠ 0) (z : RealCoordinate O T) :
    w z * forwardDirection r tokens target step z =
      (N : ℝ)⁻¹ * ∑ i, w z * coordinateDifference r (tokens i) (target i) z
        (step / response (tokens i) z) := by
  rw [forwardDirection_complete r tokens target step hh, ← Finset.mul_sum]
  ring

example : frequencyScale (fun _ : Fin 2 => repeated) repeatedCoordinate = 1 / 3 := by
  norm_num [frequencyScale, repeated_response, Fin.sum_univ_succ]

example : frequencyScale (fun _ : Fin 2 => repeated)
    (false, true, some true, none) = 3 := by
  rw [frequencyScale_unseen]
  · norm_num
  · intro i; simp [response, queryRow, repeated]

example (r : RealCoordinate Bool 4 → ℝ) :
    diagonal (fun _ => (2 : ℝ)⁻¹) (diagonal (fun _ => 2) r) = r :=
  diagonal_inverse _ _ (by intro z; norm_num)

example : (0 : RealCoordinate Bool 4 → ℝ) (false, true, some true, none) -
    1 * 3 * pullback (fun _ : Fin 2 => repeated) (fun _ _ => 1)
      (false, true, some true, none) = 0 := by
  exact scaled_zero_update 0 (fun _ => 3) _ _ 1 _ rfl
    (by intro i; simp [response, queryRow, repeated])

example : (2 : ℝ) * forwardDirection 0 (fun _ : Fin 2 => repeated) (fun _ => true)
    1 repeatedCoordinate = (2 : ℝ)⁻¹ * ∑ _ : Fin 2,
      2 * coordinateDifference 0 repeated true repeatedCoordinate (1 / 2) := by
  simpa [repeated_response] using scaled_forward_complete 0 (fun _ => 2)
    (fun _ : Fin 2 => repeated) (fun _ => true) 1 (by norm_num) repeatedCoordinate

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates
