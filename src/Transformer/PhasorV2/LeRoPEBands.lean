module

public import Transformer.PhasorV2.LeRoPE

/-!
# Phasor v2: what LeRoPE's bands can do

Partial RoPE (arXiv:2607.10134v1, §2.3, Eq. (4)) stops the slowest bands, `θ_m = 0`: "Setting
`p = 1` recovers RoPE, while `p = 0` recovers NoPE". LeRoPE's frequencies are positive, so it
cannot stop a band, but at integer positions the frequency `2π` does the same, since
`e^{2πit} = 1` (`turn_two_pi_apply`). So every head with nonnegative frequencies, Partial RoPE's
and NoPE's included, is a LeRoPE head on the integer positions: some scales put the stopped bands
at `2π` and keep the others (`exists_turn_eq`).

The converse fails. A band at the frequency `π` turns by `(−1)^t` at the integer position `t`
(`turn_pi_apply`), so a head with its bands at `π` scores `(−1)^δ` times its unturned score at the
offset `δ` (`score_turn_pi`), while NoPE scores the same at every offset (`score_turn_zero`).
One band at `π` scores `−1` at the offset `1` where NoPE scores `1` (`score_turn_pi_one`).
LeRoPE sees the parity of the offset, NoPE does not; `Transformer.LeRoPEBlock` measures this gap
for whole blocks of Phasor.
-/

@[expose] public section

open scoped ComplexConjugate

namespace Transformer.PhasorV2

variable {p : ℕ}

/-- **At the frequency `0` nothing turns**: NoPE. -/
theorem turn_zero (t : ℝ) (q : Fin p → ℂ) : turn 0 t q = q := by
  funext m
  simp [turn]

/-- **A band at `2π` does not turn at integer positions.** -/
theorem turn_two_pi_apply (θ : Fin p → ℝ) {m : Fin p} (hm : θ m = 2 * Real.pi) (t : ℤ)
    (q : Fin p → ℂ) : turn θ t q m = q m := by
  simp only [turn, hm]
  rw [show ((((t : ℝ) * (2 * Real.pi) : ℝ)) : ℂ) * Complex.I =
      (t : ℂ) * (2 * Real.pi * Complex.I) by push_cast; ring,
    Complex.exp_int_mul_two_pi_mul_I, one_mul]

/-- One band at `2π`, at the position `−3`. -/
example (q : Fin 1 → ℂ) := turn_two_pi_apply (fun _ => 2 * Real.pi) (m := 0) rfl (-3) q

/-- **Every head with nonnegative frequencies is a LeRoPE head at integer positions** (§2.3,
Eq. (4)): for `b > 0` some scales turn every band as the head does at every integer position. The
bands that Partial RoPE stops, `θ_m = 0`, go to `2π`. -/
theorem exists_turn_eq {b : ℝ} (hb : 0 < b) {θ : Fin p → ℝ} (hθ : ∀ m, 0 ≤ θ m) :
    ∃ α : Fin p → ℝ, ∀ (t : ℤ) (q : Fin p → ℂ), turn (leropeFrequency b α) t q = turn θ t q := by
  classical
  obtain ⟨α, hα⟩ :=
    exists_leropeFrequency_eq hb (θ := fun m => if θ m = 0 then 2 * Real.pi else θ m)
      fun m => by
        split_ifs with h
        · positivity
        · exact lt_of_le_of_ne (hθ m) (Ne.symm h)
  refine ⟨α, fun t q => funext fun m => ?_⟩
  rw [hα]
  by_cases h : θ m = 0
  · rw [turn_two_pi_apply _ (by simp [h]) t q]
    simp [turn, h]
  · simp [turn, h]

/-- NoPE, every band stopped, at the main model's base. -/
example := exists_turn_eq (p := 16) (b := 10000) (by norm_num) (θ := 0) fun _ => le_rfl

/-- **NoPE scores the same at every offset.** -/
theorem score_turn_zero (s t : ℝ) (q k : Fin p → ℂ) :
    score (turn 0 s q) (turn 0 t k) = score q k := by
  rw [turn_zero, turn_zero]

/-- **A band at `π` turns by `(−1)^t`** at the integer position `t`. -/
theorem turn_pi_apply (θ : Fin p → ℝ) {m : Fin p} (hm : θ m = Real.pi) (t : ℕ)
    (q : Fin p → ℂ) : turn θ t q m = (-1) ^ t * q m := by
  simp only [turn, hm]
  rw [show (((t : ℝ) * Real.pi : ℝ) : ℂ) * Complex.I = (t : ℂ) * (Real.pi * Complex.I) by
      push_cast; ring,
    Complex.exp_nat_mul, Complex.exp_pi_mul_I]

/-- One band at `π`, at the position `3`. -/
example (q : Fin 1 → ℂ) := turn_pi_apply (fun _ => Real.pi) (m := 0) rfl 3 q

/-- **LeRoPE sees the parity of the offset**: with every band at `π`, the score at the integer
offset `δ` is `(−1)^δ` times the unturned score. -/
theorem score_turn_pi (δ : ℕ) (q k : Fin p → ℂ) :
    score q (turn (fun _ => Real.pi) δ k) = (-1) ^ δ * score q k := by
  have h : turn (fun _ => Real.pi) δ k = fun m => (-1) ^ δ * k m :=
    funext fun m => turn_pi_apply _ rfl δ k
  have hs : ∑ m, conj (q m) * ((-1) ^ δ * k m) =
      (((-1 : ℝ) ^ δ : ℝ) : ℂ) * ∑ m, conj (q m) * k m := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun m _ => ?_
    push_cast
    ring
  rw [h]
  simp only [score]
  rw [hs, Complex.re_ofReal_mul]

/-- **One band at `π` against NoPE**: the query and the key `1` score `1` under both at the
offset `0`, and `−1` against `1` at the offset `1`. -/
theorem score_turn_pi_one :
    score (fun _ : Fin 1 => (1 : ℂ)) (turn (fun _ => Real.pi) (1 : ℕ) fun _ => 1) = -1 ∧
      score (fun _ : Fin 1 => (1 : ℂ)) (turn 0 (1 : ℕ) fun _ => 1) = 1 := by
  refine ⟨?_, ?_⟩
  · rw [score_turn_pi]
    simp [score]
  · rw [turn_zero]
    simp [score]

end Transformer.PhasorV2
