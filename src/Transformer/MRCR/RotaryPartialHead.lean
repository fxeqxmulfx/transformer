module

public import Transformer.MRCR.RotaryPrevHead
public import Mathlib.NumberTheory.Niven

/-!
# Partial RoPE attends to the token before alone

In no paper. A plane whose frequency lies in `2πℤ` is unrotated at integer distances, like the
planes that partial RoPE leaves out, and adds a constant to the score: the case that
`Transformer.MRCR.exists_one_le_weight_rope` excludes. With one, a positional head attends to the
token before alone at precision `s = 0`. At the frequency `θ₀ = arccos(3/5)` with the
coefficients `3, 4`, and an unrotated plane with the coefficient `−5`, the score is
`5 cos((t − 1) θ₀) − 5` (`ropeScore_partial`): `0` at the distance `1` and below `0` at every
other, since `θ₀` is no rational multiple of `π` by Niven's theorem
(`cos_int_mul_partialAngle_ne_one`). So the weight is `1` at the token before and `0` elsewhere
(`val_round_exp_ropeScore_partial`), and with values in `{0, 1}` the head returns the value of the
token before (`sum_val_round_partial`). LeRoPE (arXiv:2607.10134v1, §3.1, Eq. (5)) can learn the
frequency `2π` (`Transformer.RoPE.LeRoPE.learned_frequency_surjective`), which leaves its plane
unrotated (`ropeScore_two_pi`), so it has the head too (`exists_lerope_partial`).

The head has no margin: far keys come back within any `η` of the token before, so `ReadsPrev`
still fails (`Transformer.MRCR.not_readsPrev_ropeLogit` holds at every frequency, `0` and `2π`
included). The five layers of `Transformer.MRCR.PhraseKeyLogits` read with a margin; a reader
without one, and so the five layers with partial RoPE, are not built here.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.CRASP Transformer.CRASP.Grams

/-- The angle `θ₀ = arccos(3/5)`. -/
noncomputable def partialAngle : ℝ := Real.arccos (3 / 5)

theorem cos_partialAngle : Real.cos partialAngle = 3 / 5 :=
  Real.cos_arccos (by norm_num) (by norm_num)

theorem sin_partialAngle : Real.sin partialAngle = 4 / 5 := by
  rw [partialAngle, Real.sin_arccos, show (1 : ℝ) - (3 / 5) ^ 2 = (4 / 5) ^ 2 by norm_num,
    Real.sqrt_sq (by norm_num)]

/-- **`θ₀` is no rational multiple of `π`**: `cos(n θ₀) ≠ 1` for every integer `n ≠ 0`, by Niven's
theorem, since `cos θ₀ = 3/5` is rational and not in `{−1, −1/2, 0, 1/2, 1}`. -/
theorem cos_int_mul_partialAngle_ne_one {n : ℤ} (hn : n ≠ 0) :
    Real.cos (n * partialAngle) ≠ 1 := by
  intro h
  obtain ⟨m, hm⟩ := (Real.cos_eq_one_iff _).1 h
  have hn' : (n : ℝ) ≠ 0 := Int.cast_ne_zero.2 hn
  have hθ : ∃ r : ℚ, partialAngle = r * Real.pi := ⟨2 * m / n, by
    push_cast
    field_simp
    linarith⟩
  have h5 := niven hθ ⟨3 / 5, by rw [cos_partialAngle]; push_cast; ring⟩
  rw [cos_partialAngle] at h5
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at h5
  norm_num at h5

example := cos_int_mul_partialAngle_ne_one (n := 7) (by norm_num)

/-- The frequencies of the head: `θ₀` and an unrotated plane. -/
noncomputable def partialFreq : Fin 2 → ℝ := ![partialAngle, 0]

/-- **The score of the head**: `3 cos(θ₀ t) + 4 sin(θ₀ t) − 5 = 5 cos((t − 1) θ₀) − 5`. -/
theorem ropeScore_partial (t : ℝ) :
    ropeScore partialFreq ![3, -5] ![4, 0] t = 5 * Real.cos ((t - 1) * partialAngle) - 5 := by
  simp only [ropeScore, Fin.sum_univ_two, partialFreq, Matrix.cons_val_zero, Matrix.cons_val_one,
    zero_mul, Real.cos_zero, Real.sin_zero]
  rw [sub_mul, one_mul, Real.cos_sub, cos_partialAngle, sin_partialAngle, mul_comm partialAngle t]
  ring

/-- **The weight of the head**: at the precision `p ≥ 2`, `s = 0`, it is `1` at the distance `1`
and `0` at every other. -/
theorem val_round_exp_ropeScore_partial {p : ℕ} (hp : 2 ≤ p) (t : ℕ) :
    (Fx.round p 0 (Real.exp (ropeScore partialFreq ![3, -5] ![4, 0] t))).val =
      if t = 1 then 1 else 0 := by
  rw [ropeScore_partial]
  split_ifs with ht
  · subst ht
    simpa using val_round_natCast (p := p) (b := 1) (Nat.one_lt_two_pow (by omega))
  · apply val_round_exp_neg
    have hne := cos_int_mul_partialAngle_ne_one (n := (t : ℤ) - 1) (by omega)
    push_cast at hne
    linarith [lt_of_le_of_ne (Real.cos_le_one _) hne]

example := val_round_exp_ropeScore_partial (p := 2) le_rfl 5

/-- **A term of the numerator**: a value `x ∈ {0, 1}` passes at the distance `1` and nowhere
else. -/
theorem val_round_exp_mul_ropeScore_partial {p : ℕ} (hp : 2 ≤ p) (t : ℕ) {x : ℝ}
    (hx : x = 0 ∨ x = 1) :
    (Fx.round p 0 (Real.exp (ropeScore partialFreq ![3, -5] ![4, 0] t) * x)).val =
      if t = 1 then x else 0 := by
  rcases hx with rfl | rfl
  · simp
  · rw [mul_one, val_round_exp_ropeScore_partial hp]

example := val_round_exp_mul_ropeScore_partial (p := 2) le_rfl 1 (x := 1) (Or.inr rfl)

/-- **The head returns the value before**: at a query `i ≥ 1` the weights of the keys `j ≤ i` sum
to `1`, and with values `v j ∈ {0, 1}` the terms of the numerator sum to `v (i − 1)`. -/
theorem sum_val_round_partial {p : ℕ} (hp : 2 ≤ p) {i : ℕ} (hi : 1 ≤ i) (v : ℕ → ℝ)
    (hv : ∀ j, v j = 0 ∨ v j = 1) :
    ∑ j ∈ range (i + 1), (Fx.round p 0
        (Real.exp (ropeScore partialFreq ![3, -5] ![4, 0] ((i - j : ℕ) : ℝ)))).val = 1 ∧
      ∑ j ∈ range (i + 1), (Fx.round p 0
        (Real.exp (ropeScore partialFreq ![3, -5] ![4, 0] ((i - j : ℕ) : ℝ)) * v j)).val =
        v (i - 1) := by
  constructor
  · rw [sum_eq_single (i - 1) (fun j hj hne => by
        rw [val_round_exp_ropeScore_partial hp, ite_eq_right (by have := mem_range.1 hj; omega)])
      (fun h => absurd (mem_range.2 (by omega)) h),
      val_round_exp_ropeScore_partial hp, ite_eq_left (by omega)]
  · rw [sum_eq_single (i - 1) (fun j hj hne => by
        rw [val_round_exp_mul_ropeScore_partial hp _ (hv j),
          ite_eq_right (by have := mem_range.1 hj; omega)])
      (fun h => absurd (mem_range.2 (by omega)) h),
      val_round_exp_mul_ropeScore_partial hp _ (hv _), ite_eq_left (by omega)]

example := sum_val_round_partial (p := 2) le_rfl (i := 3) (by norm_num)
  (fun j => if j = 2 then 1 else 0) (fun j => by by_cases h : j = 2 <;> simp [h])

/-- **A frequency `2π` leaves its plane unrotated** at integer distances. -/
theorem ropeScore_two_pi (t : ℕ) :
    ropeScore ![partialAngle, 2 * Real.pi] ![3, -5] ![4, 0] t =
      ropeScore partialFreq ![3, -5] ![4, 0] t := by
  have h : Real.cos (2 * Real.pi * t) = 1 := by rw [mul_comm]; exact Real.cos_nat_mul_two_pi t
  simp [ropeScore, partialFreq, Fin.sum_univ_two, h]

/-- **LeRoPE learns the head** (arXiv:2607.10134v1, §3.1, Eq. (5)): from any positive base
frequencies it learns `θ₀` and `2π`, and then the weight of the key at the distance `t` is `1` at
`t = 1` and `0` at every other, at the precision `p ≥ 2`, `s = 0`. -/
theorem exists_lerope_partial {p : ℕ} (hp : 2 ≤ p) {omega : Fin 2 → ℝ}
    (homega : ∀ m, 0 < omega m) :
    ∃ alpha : Fin 2 → ℝ, ∀ t : ℕ, (Fx.round p 0 (Real.exp (ropeScore
      (fun m => RoPE.LeRoPE.learnedFrequency (omega m) (alpha m)) ![3, -5] ![4, 0] t))).val =
        if t = 1 then 1 else 0 := by
  obtain ⟨α₀, h₀⟩ := RoPE.LeRoPE.learned_frequency_surjective (homega 0)
    (Real.arccos_pos.2 (by norm_num) : 0 < partialAngle)
  obtain ⟨α₁, h₁⟩ := RoPE.LeRoPE.learned_frequency_surjective (homega 1) Real.two_pi_pos
  refine ⟨![α₀, α₁], fun t => ?_⟩
  have hf : (fun m => RoPE.LeRoPE.learnedFrequency (omega m) (![α₀, α₁] m)) =
      ![partialAngle, 2 * Real.pi] := by
    funext m
    fin_cases m
    · exact h₀
    · exact h₁
  rw [hf, ropeScore_two_pi, val_round_exp_ropeScore_partial hp]

example := exists_lerope_partial (p := 2) le_rfl (omega := fun _ => 1) fun _ => one_pos

end Transformer.MRCR
