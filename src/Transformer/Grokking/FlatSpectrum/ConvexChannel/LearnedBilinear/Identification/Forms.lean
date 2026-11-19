module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Coordinates

/-!
# Complete margin forms and observed comparison indices

Source: `review/convex/moments.md`, CM87. Finite linear forms read every real
coordinate of the learned complex state. Observed indices retain every record
and every competing original object. Self-comparisons have coefficient zero;
extending comparison weights by zero preserves the full double sums.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

open scoped BigOperators
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- Complete real coordinate reading, CM87. -/
noncomputable def form (v : RealCoordinate O T → ℝ) :
    (RealCoordinate O T → ℝ) →ₗ[ℝ] ℝ :=
  ∑ z, v z • LinearMap.proj z

omit [DecidableEq O] in
/-- Evaluation of the finite linear reading, CM87. -/
theorem form_apply (v r : RealCoordinate O T → ℝ) : form v r = ∑ z, v z * r z := by
  simp [form, LinearMap.sum_apply, smul_eq_mul]

omit [DecidableEq O] in
/-- A coordinate probe recovers its coefficient, CM87. -/
theorem form_single [DecidableEq (RealCoordinate O T)]
    (v : RealCoordinate O T → ℝ) (z : RealCoordinate O T) :
    form v (Pi.single z 1) = v z := by
  simp [form_apply, Pi.single_apply, mul_ite]

/-- The finite form reads the actual complex learned margin, CM87. -/
theorem form_encode (tokens : Prefix O (T + 1)) (a b : O)
    (p : SequenceParameters O T) :
    form (coefficient tokens a b) (encode p) = margin p tokens a b := by
  rw [form_apply, margin_expansion]

/-- An arbitrary real vector remains an actual learned parameter, CM87. -/
theorem form_decode (tokens : Prefix O (T + 1)) (a b : O)
    (r : RealCoordinate O T → ℝ) :
    form (coefficient tokens a b) r = margin (decode r) tokens a b := by
  rw [form_apply, margin_decode]

omit [DecidableEq O] in
/-- A coefficient certificate is an identity of complete linear readings,
CM87. Both sums are finite, so no limiting assumption is needed. -/
theorem form_combination {I : Type*} [Fintype I]
    (v : I → RealCoordinate O T → ℝ) (w : I → ℝ) (r : RealCoordinate O T → ℝ) :
    form (fun z => ∑ i, w i * v i z) r = ∑ i, w i * form (v i) r := by
  simp only [form_apply, Finset.sum_mul, Finset.mul_sum, mul_assoc]
  rw [Finset.sum_comm]

/-- Every recorded original-object competitor, including repeated records,
CM87. Only a label's comparison with itself is absent. -/
abbrev ObservedIndex (labels : Fin N → O) := Σ i, {c : O // c ≠ labels i}

/-- Restore the complete record-by-vocabulary weight table, CM87. -/
def extendWeights (labels : Fin N → O) (u : ObservedIndex labels → ℝ) : Fin N → O → ℝ :=
  fun i c => if h : c ≠ labels i then u ⟨i, ⟨c, h⟩⟩ else 0

omit [Fintype O] in
/-- Self-label entries of the extended table vanish, CM87. -/
theorem extend_self (labels : Fin N → O) (u : ObservedIndex labels → ℝ) (i : Fin N) :
    extendWeights labels u i (labels i) = 0 := by
  simp [extendWeights]

omit [Fintype O] in
/-- Every actual competitor's weight is retained, CM87. -/
theorem extend_competitor (labels : Fin N → O) (u : ObservedIndex labels → ℝ)
    (j : ObservedIndex labels) : extendWeights labels u j.1 j.2.1 = u j := by
  simp [extendWeights, j.2.2]

/-- Zero self-comparisons let a full table be summed over observed
competitors, CM87. No record or nonzero comparison is removed. -/
theorem sum_observed (labels : Fin N → O) (f : Fin N → O → ℝ)
    (hself : ∀ i, f i (labels i) = 0) :
    (∑ i, ∑ c, f i c) = ∑ j : ObservedIndex labels, f j.1 j.2.1 := by
  rw [Fintype.sum_sigma]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Fintype.sum_eq_add_sum_subtype_ne (f i) (labels i), hself, zero_add]

/-- The extended table preserves total certificate mass, CM87. -/
theorem sum_extend (labels : Fin N → O) (u : ObservedIndex labels → ℝ) :
    (∑ i, ∑ c, extendWeights labels u i c) = ∑ j, u j := by
  rw [sum_observed labels _ (extend_self labels u)]
  apply Finset.sum_congr rfl
  intro j hj
  exact extend_competitor labels u j

omit [Fintype O] in
/-- Nonnegative competitor weights extend to a nonnegative complete table,
CM87. -/
theorem extend_nonneg (labels : Fin N → O) (u : ObservedIndex labels → ℝ)
    (hu : ∀ j, 0 ≤ u j) : ∀ i c, 0 ≤ extendWeights labels u i c := by
  intro i c
  unfold extendWeights
  split
  · exact hu _
  · exact le_rfl

example (labels : Fin N → O) :
    (∑ _ : Fin N, ∑ _ : O, (0 : ℝ)) =
      ∑ j : ObservedIndex labels, (fun (_ : Fin N) (_ : O) => (0 : ℝ)) j.1 j.2.1 :=
  sum_observed labels (fun _ _ => 0) (by simp)

example (labels : Fin N → O) : ∀ i c, 0 ≤ extendWeights labels (fun _ => 1) i c :=
  extend_nonneg labels (fun _ => 1) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification
