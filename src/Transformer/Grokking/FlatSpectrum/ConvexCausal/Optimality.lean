module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Mixture

/-!
# Exact empirical loss and identification of observed answers

Source: `review/convex_causal.md`, CCa4. The positive floor is fixed before
training. A deterministic fit attains the exact CE lower bound; conversely
attaining it forces every observed target mass to one in the unsmoothed
coordinates. These are statements about the repaired ordinary CE, not the
old convention of treating probabilities themselves as logits.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {X C S : Type*} [Fintype C] [Nonempty C] [Fintype S]

/-- Exact per-observation optimum with the specified probability floor (CCa4). -/
noncomputable def ideal (C : Type*) [Fintype C] (ε : ℝ) : ℝ :=
  -Real.log (1 - ε + ε / Fintype.card C)

omit [Nonempty C] in
/-- Every conditional coordinate is at most one (CCa4). -/
theorem coordinate_le_one {Q : X → C → ℝ} (hQ : Rows Q) (x : X) (c : C) : Q x c ≤ 1 := by
  have h := Finset.single_le_sum (fun d _ => hQ.1 x d) (Finset.mem_univ c)
  simpa [hQ.2 x] using h

/-- The best attainable target probability is positive (CCa4). -/
theorem cap_pos {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1) :
    0 < 1 - ε + ε / Fintype.card C := by
  exact add_pos_of_nonneg_of_pos (sub_nonneg.mpr hε1)
    (div_pos hε (Nat.cast_pos.mpr Fintype.card_pos))

/-- Every observation has the same explicit lower bound on ordinary CE (CCa4). -/
theorem ce_lower {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (x : X) (c : C) :
    ideal C ε ≤ crossEntropy (fun d => Real.log (smooth ε Q x d)) c := by
  rw [crossEntropy_eq hQ hε hε1]
  apply neg_le_neg
  apply Real.log_le_log (smooth_pos hQ hε hε1 x c)
  have h := mul_le_mul_of_nonneg_left (coordinate_le_one hQ x c) (sub_nonneg.mpr hε1)
  simpa [smooth] using add_le_add_right h (ε / Fintype.card C)

/-- Equality at an observation forces full unsmoothed target mass (CCa4). -/
theorem mass_one_of_ce_le {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε < 1) (x : X) (c : C)
    (hce : crossEntropy (fun d => Real.log (smooth ε Q x d)) c ≤ ideal C ε) :
    Q x c = 1 := by
  rw [crossEntropy_eq hQ hε hε1.le] at hce
  have hl : Real.log (1 - ε + ε / Fintype.card C) ≤ Real.log (smooth ε Q x c) :=
    neg_le_neg_iff.mp hce
  have hm := (Real.log_le_log_iff (cap_pos (C := C) hε hε1.le)
    (smooth_pos hQ hε hε1.le x c)).mp hl
  have hq := coordinate_le_one hQ x c
  unfold smooth at hm
  nlinarith

/-- Summing keeps repeated samples and gives a global lower bound (CCa4). -/
theorem loss_lower {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (input : S → X) (label : S → C) :
    (∑ _s : S, ideal C ε) ≤ loss ε input label Q :=
  Finset.sum_le_sum fun s _ => ce_lower hQ hε hε1 (input s) (label s)

/-- An exact observed fit actually attains the lower bound (CCa4). -/
theorem loss_eq_ideal {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (input : S → X) (label : S → C)
    (hfit : ∀ s, Q (input s) (label s) = 1) :
    loss ε input label Q = ∑ _s : S, ideal C ε := by
  apply Finset.sum_congr rfl
  intro s _
  rw [crossEntropy_eq hQ hε hε1]
  simp [smooth, hfit s, ideal]

/-- No successful optimization is assumed: a constructed fit is a global CE minimum (CCa4). -/
theorem fit_minimum {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (input : S → X) (label : S → C)
    (hfit : ∀ s, Q (input s) (label s) = 1) :
    IsMinOn (loss ε input label) {R | Rows R} Q := by
  intro R hR
  rw [loss_eq_ideal hQ hε hε1 input label hfit]
  exact loss_lower hR hε hε1 input label

/-- At the lower bound every original sample has full target mass (CCa4). -/
theorem observed_mass {Q : X → C → ℝ} (hQ : Rows Q) {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε < 1) (input : S → X) (label : S → C)
    (hL : loss ε input label Q ≤ ∑ _s : S, ideal C ε) (s : S) :
    Q (input s) (label s) = 1 := by
  have he := le_antisymm (loss_lower hQ hε hε1.le input label) hL
  have ht := (Finset.sum_eq_sum_iff_of_le
    (fun t (_ : t ∈ (Finset.univ : Finset S)) => ce_lower hQ hε hε1.le (input t) (label t))).mp he
  exact mass_one_of_ce_le hQ hε hε1 (input s) (label s) (ht s (Finset.mem_univ s)).ge

omit [Nonempty C] in
/-- Deterministic predictions have a strict correct argmax for any floor below one (CCa4). -/
theorem point_strict [DecidableEq C] (f : X → C) {ε : ℝ} (hε : ε < 1)
    (x : X) (c : C) (hc : c ≠ f x) :
    smooth ε (point f) x c < smooth ε (point f) x (f x) := by
  simp only [smooth, point, hc, ↓reduceIte, mul_zero, mul_one, zero_add]
  linarith

example := coordinate_le_one (rows_point (id : Bool → Bool)) true true
example := cap_pos (C := Bool) (ε := 1 / 4) (by norm_num) (by norm_num)
example := ce_lower (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) false true
example := mass_one_of_ce_le (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) true true (by
    rw [crossEntropy_eq (rows_point _) (by norm_num) (by norm_num)]
    simp [smooth, point, ideal])
example := loss_lower (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) (id : Bool → Bool) id
example := loss_eq_ideal (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) (id : Bool → Bool) id (by intro s; simp [point])
example := fit_minimum (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) (id : Bool → Bool) id (by intro s; simp [point])
example := observed_mass (rows_point (id : Bool → Bool)) (ε := 1 / 4)
  (by norm_num) (by norm_num) (id : Bool → Bool) id
  (loss_eq_ideal (rows_point _) (by norm_num) (by norm_num) _ _
    (by intro s; simp [point])).le true
example := point_strict (id : Bool → Bool) (ε := 1 / 4) (by norm_num) true false (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexCausal
