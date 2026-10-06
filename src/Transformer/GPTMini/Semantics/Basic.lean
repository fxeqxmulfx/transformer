import Transformer.GPTMini.AttentionBounds

/-!
# Finite-temperature retrieval in the actual softmax head

Source: CausalMHA.forward in archived gpt_mini.py at f11b6e2, specifically
QKNorm, the causal mask and finite softmax. The routing condition is an
internal Q/K score gap, not a desired final logit. Every unmasked weight
stays positive. Nevertheless the total mass outside a selected visible
position is at most (T - 1) exp(-gap).

The per-head temperature alpha remains arbitrary in every result.
T counts the full context here, so the estimate also holds when some
competitors are masked. Applying it to attentionHead uses the actual
RoPE-rotated queries and keys; no hard attention or rounding is substituted.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- RoPE sends the zero value to zero at every actual position.
Source: apply_rope at f11b6e2, whose rotations preserve the Euclidean norm. -/
theorem rope_zero (d : ℕ) (theta position : ℝ) :
    applyRope d theta position 0 = 0 := by
  apply norm_eq_zero.mp
  rw [applyRope_isometry, norm_zero]

/-- The causal denominator is positive because the query can attend to itself.
Source: CausalMHA.forward at f11b6e2, for arbitrary finite real scores. -/
theorem denominator_pos (cfg : Config) {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (i : Fin T) :
    0 < ∑ j : Fin T, if j.val ≤ i.val then
      Real.exp (preScore cfg alpha eps q k i j) else 0 := by
  apply Finset.sum_pos'
  · intro j hj
    split_ifs
    · exact (Real.exp_pos _).le
    · exact le_rfl
  · exact ⟨i, Finset.mem_univ _, by simp only [le_refl, ite_true]; positivity⟩

/-- A visible position has strictly positive mass at every finite temperature.
Source: the unmodified softmax in CausalMHA.forward at f11b6e2. -/
theorem weight_pos (cfg : Config) {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (i j : Fin T) (hji : j.val ≤ i.val) :
    0 < causalAttnWeights cfg alpha eps q k i j := by
  rw [causalAttnWeights, ite_eq_right (not_lt.mpr hji)]
  exact div_pos (Real.exp_pos _) (denominator_pos cfg alpha eps q k i)

example (cfg : Config) : (0 : Fin 2).val ≤ (1 : Fin 2).val ∧
    0 < causalAttnWeights cfg (T := 2) 0 1 (fun _ => 0) (fun _ => 0) 1 0 :=
  ⟨by decide, weight_pos cfg _ _ _ _ _ _ (by decide)⟩

/-- A competing weight is exponentially small when its internal score is below the selected score.
Source: direct finite-softmax arithmetic in CausalMHA.forward at f11b6e2. -/
theorem other_weight_le (cfg : Config) {T : ℕ} (alpha eps gap : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (i selected j : Fin T)
    (hselected : selected.val ≤ i.val)
    (hgap : j.val ≤ i.val → preScore cfg alpha eps q k i j ≤
      preScore cfg alpha eps q k i selected - gap) :
    causalAttnWeights cfg alpha eps q k i j ≤ Real.exp (-gap) := by
  by_cases hj : j.val ≤ i.val
  · rw [causalAttnWeights, ite_eq_right (not_lt.mpr hj)]
    have hden : Real.exp (preScore cfg alpha eps q k i selected) ≤
        ∑ r : Fin T, if r.val ≤ i.val then
          Real.exp (preScore cfg alpha eps q k i r) else 0 := by
      have h := Finset.single_le_sum (s := Finset.univ)
        (f := fun r : Fin T => if r.val ≤ i.val then
          Real.exp (preScore cfg alpha eps q k i r) else 0)
        (fun r _ => by split_ifs <;> positivity) (Finset.mem_univ selected)
      simpa only [ite_eq_left hselected] using h
    calc _ ≤ Real.exp (preScore cfg alpha eps q k i j) /
          Real.exp (preScore cfg alpha eps q k i selected) :=
        div_le_div_of_nonneg_left (Real.exp_pos _).le (Real.exp_pos _) hden
      _ = Real.exp (preScore cfg alpha eps q k i j -
          preScore cfg alpha eps q k i selected) := (Real.exp_sub _ _).symm
      _ ≤ Real.exp (-gap) := Real.exp_le_exp.mpr (by linarith [hgap hj])
  · rw [causalAttnWeights, ite_eq_left (not_le.mp hj)]
    exact (Real.exp_pos _).le

example (cfg : Config) : (0 : Fin 2).val ≤ (1 : Fin 2).val ∧
    ((1 : Fin 2).val ≤ (1 : Fin 2).val →
      preScore cfg (T := 2) 0 1 (fun _ => 0) (fun _ => 0) 1 1 ≤
        preScore cfg (T := 2) 0 1 (fun _ => 0) (fun _ => 0) 1 0 - 0) := by
  exact ⟨by decide, fun _ => by simp [preScore, score, normL2]⟩

/-- The whole softmax tail has a finite explicit bound, without setting any visible weight to zero.
Source: summing the internal-score estimate over all T - 1 competing positions. -/
theorem tail_mass_le (cfg : Config) {T : ℕ} (alpha eps gap : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim) (i selected : Fin T)
    (hselected : selected.val ≤ i.val)
    (hgap : ∀ j, j ≠ selected → j.val ≤ i.val →
      preScore cfg alpha eps q k i j ≤ preScore cfg alpha eps q k i selected - gap) :
    1 - causalAttnWeights cfg alpha eps q k i selected ≤
      ((T - 1 : ℕ) : ℝ) * Real.exp (-gap) := by
  classical
  have hsum : ∑ j ∈ Finset.univ.erase selected,
      causalAttnWeights cfg alpha eps q k i j =
        1 - causalAttnWeights cfg alpha eps q k i selected := by
    rw [Finset.sum_erase_eq_sub (Finset.mem_univ _), causalAttnWeights_row_sum]
  rw [← hsum]
  have hcard : (Finset.univ.erase selected).card = T - 1 := by simp
  have h := Finset.sum_le_card_nsmul (Finset.univ.erase selected)
    (fun j => causalAttnWeights cfg alpha eps q k i j) (Real.exp (-gap))
    (fun j hj => other_weight_le cfg alpha eps gap q k i selected j hselected
      (hgap j (Finset.mem_erase.mp hj).1))
  simpa only [hcard, nsmul_eq_mul] using h

example (cfg : Config) : (0 : Fin 2).val ≤ (1 : Fin 2).val ∧
    ∀ j : Fin 2, j ≠ 0 → j.val ≤ (1 : Fin 2).val →
      preScore cfg (T := 2) 0 1 (fun _ => 0) (fun _ => 0) 1 j ≤
        preScore cfg (T := 2) 0 1 (fun _ => 0) (fun _ => 0) 1 0 - 0 := by
  exact ⟨by decide, fun _ _ _ => by simp [preScore, score, normL2]⟩

/-- Concentration bounds retrieval error by the value diameter times the omitted mass.
Source: attnOutput in CausalMHA.forward at f11b6e2, before XSA. -/
theorem output_error_le (cfg : Config) {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (i selected : Fin T) (B : ℝ)
    (hvalues : ∀ j, j ≠ selected → ‖v j - v selected‖ ≤ B) :
    ‖attnOutput cfg alpha eps q k v i - v selected‖ ≤
      (1 - causalAttnWeights cfg alpha eps q k i selected) * B := by
  classical
  have he : attnOutput cfg alpha eps q k v i - v selected =
      ∑ j ∈ Finset.univ.erase selected,
        causalAttnWeights cfg alpha eps q k i j • (v j - v selected) := by
    rw [Finset.sum_erase Finset.univ (a := selected) (f := fun j =>
      causalAttnWeights cfg alpha eps q k i j • (v j - v selected)) (by simp)]
    simp only [smul_sub, Finset.sum_sub_distrib, ← Finset.sum_smul,
      causalAttnWeights_row_sum, one_smul, attnOutput]
  rw [he]
  calc _ ≤ ∑ j ∈ Finset.univ.erase selected,
        ‖causalAttnWeights cfg alpha eps q k i j • (v j - v selected)‖ := norm_sum_le _ _
    _ ≤ ∑ j ∈ Finset.univ.erase selected,
        causalAttnWeights cfg alpha eps q k i j * B := by
      apply Finset.sum_le_sum
      intro j hj
      rw [norm_smul, Real.norm_eq_abs,
        abs_of_nonneg (causalAttnWeights_nonneg cfg alpha eps q k i j)]
      exact mul_le_mul_of_nonneg_left (hvalues j (Finset.mem_erase.mp hj).1)
        (causalAttnWeights_nonneg cfg alpha eps q k i j)
    _ = _ := by
      rw [← Finset.sum_mul, Finset.sum_erase_eq_sub (Finset.mem_univ _),
        causalAttnWeights_row_sum]

example (cfg : Config) : ∀ j : Fin 2, j ≠ 0 →
    ‖(0 : EucSpace cfg.head_dim) - 0‖ ≤ (1 : ℝ) := by
  intro j hj
  simp

end Transformer.GPTMini.Semantics
