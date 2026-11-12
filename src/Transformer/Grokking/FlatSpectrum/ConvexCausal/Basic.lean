module

public import Transformer.Landscape.Neural.Loss
public import Mathlib.Analysis.Convex.SpecificFunctions.Basic
public import Mathlib.Tactic

/-!
# Calibrated causal probability coordinates

Source: `review/convex_causal.md`, CCa1. `X` is the visible causal prefix (or a
fixed encoding of it), never its answer. All entries of `Q` are joint coordinates.
The output probabilities are `(1-epsilon) Q + epsilon / |C|`; their logarithms
are logits for the existing ordinary full-vocabulary cross-entropy. Positivity
is established before using `exp (log p)`, including at zero entries of `Q`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {X C : Type*} [Fintype C]

/-- Conditional probability rows on visible inputs (CCa1). -/
def Rows (Q : X → C → ℝ) : Prop :=
  (∀ x c, 0 ≤ Q x c) ∧ ∀ x, ∑ c, Q x c = 1

/-- A fixed positive probability floor, not a learned logit multiplier (CCa1). -/
noncomputable def smooth (ε : ℝ) (Q : X → C → ℝ) (x : X) (c : C) : ℝ :=
  (1 - ε) * Q x c + ε / Fintype.card C

/-- The actual ordinary answer loss, including every original sample (CCa1). -/
noncomputable def loss {S : Type*} [Fintype S] (ε : ℝ) (input : S → X)
    (label : S → C) (Q : X → C → ℝ) : ℝ :=
  ∑ s, crossEntropy (fun c => Real.log (smooth ε Q (input s) c)) (label s)

/-- The unfactorized conditional probability domain is convex (CCa1). -/
theorem rows_convex : Convex ℝ {Q : X → C → ℝ | Rows Q} := by
  intro Q hQ R hR a b ha hb hab
  constructor
  · intro x c
    exact add_nonneg (mul_nonneg ha (hQ.1 x c)) (mul_nonneg hb (hR.1 x c))
  · intro x
    change (∑ c, (a * Q x c + b * R x c)) = 1
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hQ.2, hR.2]
    simpa using hab

/-- Smoothing is affine in every coordinate simultaneously (CCa1). -/
theorem smooth_affine (ε : ℝ) (Q R : X → C → ℝ) {a b : ℝ} (hab : a + b = 1)
    (x : X) (c : C) :
    smooth ε (a • Q + b • R) x c = a * smooth ε Q x c + b * smooth ε R x c := by
  change (1 - ε) * (a * Q x c + b * R x c) + ε / Fintype.card C = _
  unfold smooth
  calc
    _ = (1 - ε) * (a * Q x c + b * R x c) + (a + b) * (ε / Fintype.card C) :=
      by rw [hab, one_mul]
    _ = _ := by ring

/-- Every row normalizes over the original complete vocabulary (CCa1). -/
theorem smooth_sum [Nonempty C] {Q : X → C → ℝ} (hQ : Rows Q) (ε : ℝ) (x : X) :
    ∑ c, smooth ε Q x c = 1 := by
  have hn : (Fintype.card C : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  simp only [smooth, Finset.sum_add_distrib, ← Finset.mul_sum, hQ.2, mul_one,
    Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp
  ring

/-- The floor makes all log probabilities finite (CCa1). -/
theorem smooth_pos [Nonempty C] {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (x : X) (c : C) : 0 < smooth ε Q x c := by
  apply add_pos_of_nonneg_of_pos (mul_nonneg (sub_nonneg.mpr hε1) (hQ.1 x c))
  exact div_pos hε (Nat.cast_pos.mpr Fintype.card_pos)

/-- Softmax of the emitted logits is exactly the intended distribution (CCa1). -/
theorem normalized_logits [Nonempty C] {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (x : X) (c : C) :
    Real.exp (Real.log (smooth ε Q x c)) /
      (∑ d, Real.exp (Real.log (smooth ε Q x d))) = smooth ε Q x c := by
  simp_rw [Real.exp_log (smooth_pos hQ hε hε1 x _)]
  rw [smooth_sum hQ, div_one]

/-- Ordinary softmax CE equals normalized negative log likelihood (CCa1). -/
theorem crossEntropy_eq [Nonempty C] {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (x : X) (c : C) :
    crossEntropy (fun d => Real.log (smooth ε Q x d)) c = -Real.log (smooth ε Q x c) := by
  unfold crossEntropy
  simp_rw [Real.exp_log (smooth_pos hQ hε hε1 x _)]
  rw [smooth_sum hQ, Real.log_one, zero_sub]

/-- Ordinary CE is convex despite the nonlinear log-probability readout (CCa1). -/
theorem loss_convex [Nonempty C] {S : Type*} [Fintype S] {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (input : S → X) (label : S → C) :
    ConvexOn ℝ {Q | Rows Q} (loss ε input label) := by
  refine ⟨rows_convex, ?_⟩
  intro Q hQ R hR a b ha hb hab
  have hmix := rows_convex hQ hR ha hb hab
  simp only [loss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  rw [crossEntropy_eq hmix hε hε1, crossEntropy_eq hQ hε hε1,
    crossEntropy_eq hR hε hε1, smooth_affine ε Q R hab]
  exact (strictConcaveOn_log_Ioi.concaveOn.neg).2
    (smooth_pos hQ hε hε1 _ _) (smooth_pos hR hε hε1 _ _) ha hb hab

/-- One-hot deterministic predictions remain valid joint coordinates (CCa1). -/
def point [DecidableEq C] (f : X → C) : X → C → ℝ :=
  fun x c => if c = f x then 1 else 0

/-- Deterministic causal rules supply feasible rows (CCa1). -/
theorem rows_point [DecidableEq C] (f : X → C) : Rows (point f) := by
  constructor
  · intro x c
    simp only [point]
    split_ifs <;> norm_num
  · intro x
    simp [point]

example := smooth_affine (1 / 4) (point (id : Bool → Bool)) (point Bool.not)
  (a := 1 / 3) (b := 2 / 3) (by norm_num) false true

example := smooth_sum (rows_point (id : Bool → Bool)) (1 / 4) true

example := smooth_pos (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) false true

example := normalized_logits (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) true false

example := crossEntropy_eq (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) true true

example := loss_convex (X := Bool) (C := Bool) (ε := 1 / 4)
  (by norm_num) (by norm_num) (id : Bool → Bool) Bool.not

end Transformer.Grokking.FlatSpectrum.ConvexCausal
