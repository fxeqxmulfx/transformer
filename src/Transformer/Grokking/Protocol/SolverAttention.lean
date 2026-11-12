module

public import Transformer.Grokking.Protocol.SolverParams

/-!
# The attention of the solver writes the gap of the equation

Source: openai/grok at 3d64b1d8, `grok/transformer.py`, `MultiHeadAttention` and
`AttentionHead`, as `Model.score` and `Model.weight`; arXiv:2201.02177v1 (Power et al.),
appendix *Additional experimental details*. The parameters are those of `SolverParams`.

The rows that the first block reads (`solverRows`) are zero at `<|eos|>` and at `/`, the
feature of the residue in the entry `0` with the position added at the numerator and the
denominator, and `solverTarget` at `=` (`solverRows_zero` to `solverRows_four`).

At `=`, a head that is not the keyed one has the score `0` for every position
(`solver_score_of_ne`) and weighs the five positions by `1/5` (`solver_weight_of_ne`). The
keyed head has for score the entry `1` of the position it looks at (`solver_score_keyed`): the
query reads the ballast of the row at `=`, which the factor `√32 / 10000` of the query turns
into the `√32` that the score divides by. Its weights are the exponentials of those entries
over `3 + keyExp 1 + keyExp 3` (`solver_weight_keyed`), and they do not depend on the
equation.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- The rows that the first block of the solver reads: the embeddings with the positions. -/
noncomputable def solverRows (x : Equation) : Fin 6 → Fin 128 → ℝ :=
  decodeTo solver (relu paperSizes) (read x) 0

lemma solverRows_apply (x : Equation) (t : Fin 6) (i : Fin 128) :
    solverRows x t i = solverEmb (read x t) i + position paperSizes t i := rfl

lemma solverRows_zero (x : Equation) (i : Fin 128) : solverRows x 0 i = 0 := by
  rw [solverRows_apply, show read x 0 = eosToken from rfl, solverEmb, ite_eq_left rfl]
  exact neg_add_cancel _

lemma solverRows_two (x : Equation) (i : Fin 128) : solverRows x 2 i = 0 := by
  rw [solverRows_apply, show read x 2 = divToken from rfl, solverEmb,
    ite_eq_right (by decide : divToken ≠ eosToken), ite_eq_left rfl]
  exact neg_add_cancel _

lemma solverRows_four (x : Equation) (i : Fin 128) : solverRows x 4 i = solverTarget i := by
  rw [solverRows_apply, show read x 4 = eqToken from rfl, solverEmb,
    ite_eq_right (by decide : eqToken ≠ eosToken), ite_eq_right (by decide : eqToken ≠ divToken),
    ite_eq_left rfl]
  exact sub_add_cancel _ _

lemma solverRows_one (x : Equation) (i : Fin 128) :
    solverRows x 1 i = ind 0 i * logFeature (num x) + position paperSizes 1 i := by
  rw [solverRows_apply, show read x 1 = residueToken (numerator x) from rfl,
    solverEmb_residueToken]
  rfl

lemma solverRows_three (x : Equation) (i : Fin 128) :
    solverRows x 3 i = ind 0 i * logFeature (den x) + position paperSizes 3 i := by
  rw [solverRows_apply, show read x 3 = residueToken (denominator x) from rfl,
    solverEmb_residueToken]
  rfl


/-- A sum over the positions up to `=`. -/
lemma sum_Iic_four (f : Fin 6 → ℝ) :
    ∑ j ∈ Finset.Iic (4 : Fin 6), f j = f 0 + f 1 + f 2 + f 3 + f 4 := by
  rw [show Finset.Iic (4 : Fin 6) = {0, 1, 2, 3, 4} from by decide]
  simp only [Finset.sum_insert, Finset.mem_insert, Finset.mem_singleton, Finset.sum_singleton,
    Fin.reduceEq, or_self, not_false_eq_true]
  ring

variable {l : Fin paperSizes.blocks} {h : Fin paperSizes.heads} {r : Fin paperSizes.headWidth}

lemma solver_value_sum (hl : (l : ℕ) = 0) (hh : (h : ℕ) ≤ 1) (hr : (r : ℕ) = 0)
    (z : Fin 128 → ℝ) : ∑ i, solver (.value l h r i) * z i = z 0 := by
  have e : ∀ i : Fin 128, solver (.value l h r i) = ind 0 i := fun i => ite_eq_left ⟨hl, hh, hr⟩
  rw [Finset.sum_congr rfl fun i _ => by rw [e i]]
  exact sum_ind_mul (by norm_num) z

lemma solver_key_sum (hl : (l : ℕ) = 0) (hh : (h : ℕ) = 1) (hr : (r : ℕ) = 0)
    (z : Fin 128 → ℝ) : ∑ i, solver (.key l h r i) * z i = z 1 := by
  have e : ∀ i : Fin 128, solver (.key l h r i) = ind 1 i := fun i => ite_eq_left ⟨hl, hh, hr⟩
  rw [Finset.sum_congr rfl fun i _ => by rw [e i]]
  exact sum_ind_mul (by norm_num) z

lemma solver_query_sum (hl : (l : ℕ) = 0) (hh : (h : ℕ) = 1) (hr : (r : ℕ) = 0)
    (z : Fin 128 → ℝ) :
    ∑ i, solver (.query l h r i) * z i = √(paperSizes.headWidth : ℝ) / ballast * z 3 := by
  have e : ∀ i : Fin 128, solver (.query l h r i) =
      √(paperSizes.headWidth : ℝ) / ballast * ind 3 i := fun i => ite_eq_left ⟨hl, hh, hr⟩
  rw [Finset.sum_congr rfl fun i _ => by rw [e i, mul_assoc], ← Finset.mul_sum]
  exact congrArg _ (sum_ind_mul (by norm_num) z)

lemma solver_query_sum_zero (hc : ¬((l : ℕ) = 0 ∧ (h : ℕ) = 1 ∧ (r : ℕ) = 0))
    (z : Fin 128 → ℝ) : ∑ i, solver (.query l h r i) * z i = 0 := by
  have e : ∀ i : Fin 128, solver (.query l h r i) = 0 := fun i => ite_eq_right hc
  rw [Finset.sum_congr rfl fun i _ => by rw [e i, zero_mul], Finset.sum_const_zero]

/-- A head that is not the keyed one has no query: its scores are zero. -/
lemma solver_score_of_ne (hh : (h : ℕ) ≠ 1) (X : Fin 6 → Fin 128 → ℝ) (t j : Fin 6) :
    score solver X l h t j = 0 := by
  rw [score, Finset.sum_eq_zero fun r _ => by
    rw [solver_query_sum_zero (fun hc => hh hc.2.1), zero_mul], zero_div]

/-- **The score of the keyed head** at `=` is the entry `1` of the position it looks at. -/
lemma solver_score_keyed (hl : (l : ℕ) = 0) (hh : (h : ℕ) = 1) (X : Fin 6 → Fin 128 → ℝ)
    (hX : X 4 3 = ballast) (j : Fin 6) : score solver X l h 4 j = X j 1 := by
  have hw : √(paperSizes.headWidth : ℝ) ≠ 0 := (Real.sqrt_pos.2 (by norm_num)).ne'
  have hb : ballast ≠ 0 := by norm_num [ballast]
  rw [score, Finset.sum_eq_single (⟨0, by decide⟩ : Fin paperSizes.headWidth)
    (fun r _ hr => by
      rw [solver_query_sum_zero (fun hc => hr (Fin.ext hc.2.2)), zero_mul])
    (fun hc => absurd (Finset.mem_univ _) hc),
    solver_query_sum hl hh rfl, solver_key_sum hl hh rfl, hX]
  field_simp


/-- A head without a query weighs the five positions up to `=` equally. -/
lemma solver_weight_of_ne (hh : (h : ℕ) ≠ 1) (X : Fin 6 → Fin 128 → ℝ) {j : Fin 6}
    (hj : j ≤ 4) : weight solver X l h 4 j = 1 / 5 := by
  rw [weight, ite_eq_left hj, sum_Iic_four]
  simp only [solver_score_of_ne hh, Real.exp_zero]
  norm_num

/-- The entries `1` and `3` of the rows: what the keyed head scores, and the ballast. -/
lemma solverRows_key (x : Equation) :
    solverRows x 0 1 = 0 ∧ solverRows x 1 1 = position paperSizes 1 (1 : Fin 128) ∧
      solverRows x 2 1 = 0 ∧ solverRows x 3 1 = position paperSizes 3 (1 : Fin 128) ∧
      solverRows x 4 1 = 0 ∧ solverRows x 4 3 = ballast := by
  have i01 : ind 0 (1 : Fin 128) = 0 := ind_of_ne (by decide)
  have i31 : ind 3 (1 : Fin 128) = 0 := ind_of_ne (by decide)
  have i21 : ind 2 (1 : Fin 128) = 0 := ind_of_ne (by decide)
  have i33 : ind 3 (3 : Fin 128) = 1 := ind_of_eq rfl
  have i23 : ind 2 (3 : Fin 128) = 0 := ind_of_ne (by decide)
  refine ⟨solverRows_zero x 1, ?_, solverRows_two x 1, ?_, ?_, ?_⟩
  · rw [solverRows_one, i01, zero_mul, zero_add]
  · rw [solverRows_three, i01, zero_mul, zero_add]
  · rw [solverRows_four, solverTarget, i31, i21, mul_zero, mul_zero, sub_zero]
  · rw [solverRows_four, solverTarget, i33, i23, mul_one, mul_zero, sub_zero]

/-- **The weights of the keyed head** at `=`. -/
lemma solver_weight_keyed (hl : (l : ℕ) = 0) (hh : (h : ℕ) = 1) (x : Equation) {j : Fin 6}
    (hj : j ≤ 4) :
    weight solver (solverRows x) l h 4 j =
      Real.exp (solverRows x j 1) / (3 + keyExp 1 + keyExp 3) := by
  obtain ⟨r0, r1, r2, r3, r4, r43⟩ := solverRows_key x
  rw [weight, ite_eq_left hj, sum_Iic_four]
  simp only [solver_score_keyed hl hh _ r43]
  congr 1
  rw [r0, r1, r2, r3, r4, Real.exp_zero, keyExp, keyExp]
  ring

/-- The entry `0` of the rows: what the two heads read. -/
lemma solverRows_value (x : Equation) :
    solverRows x 0 0 = 0 ∧
      solverRows x 1 0 = logFeature (num x) + position paperSizes 1 (0 : Fin 128) ∧
      solverRows x 2 0 = 0 ∧
      solverRows x 3 0 = logFeature (den x) + position paperSizes 3 (0 : Fin 128) ∧
      solverRows x 4 0 = 0 := by
  have i00 : ind 0 (0 : Fin 128) = 1 := ind_of_eq rfl
  have i30 : ind 3 (0 : Fin 128) = 0 := ind_of_ne (by decide)
  have i20 : ind 2 (0 : Fin 128) = 0 := ind_of_ne (by decide)
  refine ⟨solverRows_zero x 0, ?_, solverRows_two x 0, ?_, ?_⟩
  · rw [solverRows_one, i00, one_mul]
  · rw [solverRows_three, i00, one_mul]
  · rw [solverRows_four, solverTarget, i30, i20, mul_zero, mul_zero, sub_zero]

/-! The witnesses: the first block, its two heads at work, and the equation `3 / 5 = 20`. -/

example := solver_value_sum (l := ⟨0, by decide⟩) (h := ⟨1, by decide⟩) (r := ⟨0, by decide⟩)
  rfl (by decide) rfl fun _ => 1

example := solver_key_sum (l := ⟨0, by decide⟩) (h := ⟨1, by decide⟩) (r := ⟨0, by decide⟩)
  rfl rfl rfl fun _ => 1

example := solver_query_sum (l := ⟨0, by decide⟩) (h := ⟨1, by decide⟩) (r := ⟨0, by decide⟩)
  rfl rfl rfl fun _ => 1

example := solver_query_sum_zero (l := ⟨1, by decide⟩) (h := ⟨1, by decide⟩)
  (r := ⟨0, by decide⟩) (by decide) fun _ => 1

example := solver_score_of_ne (l := ⟨0, by decide⟩) (h := ⟨0, by decide⟩) (by decide)
  (fun _ _ => 1) 4 1

example := solver_score_keyed (l := ⟨0, by decide⟩) (h := ⟨1, by decide⟩) rfl rfl
  (solverRows (20, 4)) (solverRows_key (20, 4)).2.2.2.2.2 1

example := solver_weight_of_ne (l := ⟨0, by decide⟩) (h := ⟨0, by decide⟩) (by decide)
  (fun _ _ => 1) (j := 1) (by decide)

example := solver_weight_keyed (l := ⟨0, by decide⟩) (h := ⟨1, by decide⟩) rfl rfl (20, 4)
  (j := 1) (by decide)

end Transformer.Grokking.Protocol
