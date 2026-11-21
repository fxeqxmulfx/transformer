module

public import Transformer.Phasor.TwoCopies.Slots

/-!
# Two copies: the embedding and the states of the last position

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

A token `v` of `ℤ/p` sits at the angle `θ_v = 2π v / p` (`angle`), and `ψ(v) = e^{iθ_v}` is the
point `circ p v` of a plane. The embedding of the construction puts `b` on the bias, `1` on the code
`(3, v)` and `ψ(v)` on the output plane (`embC`), so every token has the squared length `b² + 2`
(`sum_embC_sq`) and the RMS treats all tokens alike. The state of the last position, after each
sublayer, has the same shape (`stateC`): the own token's embedding times `k`, the codes of the three
other operands times `a`, the phases `P_c ψ(w_c)` of the two copies and `M ψ(f)` on the output.
Sublayers add and the RMS scales these coefficients (`stateC_add`, `smul_stateC`); the squared
length does not depend on the tokens until the output is written (`sum_stateC_sq`); and the readout
against the embedding of `v` is `k (b² + [v = t₃] + cos(θ_v − θ_{t₃})) + M cos(θ_v − θ_f)`
(`sum_embC_mul_stateC`). Two facts on the angles finish the readout: `ψ(v)ψ(w) = ψ(v + w)`
(`cos_angle_add`, `sin_angle_add`), and `cos(θ_v − θ_f) ≤ cos(2π/p)` for `v ≠ f`
(`cos_angle_sub_le`).
-/

@[expose] public section

open Finset Real

namespace Transformer.Phasor.TwoCopies

variable {p : ℕ}

/-- The angle `2π v / p` of the token `v`: `ψ(v) = e^{2πi v/p}` lies at this angle. -/
noncomputable def angle (p : ℕ) (v : Fin p) : ℝ := 2 * π * (v : ℕ) / p

/-- The coordinate `r` of `ψ(v)` on a plane: `cos θ_v` for `r = 0`, `sin θ_v` for `r = 1`. -/
noncomputable def circ (p : ℕ) (v : Fin p) (r : Fin 2) : ℝ :=
  if r = 0 then cos (angle p v) else sin (angle p v)

theorem sum_circ_mul (v w : Fin p) :
    ∑ r, circ p v r * circ p w r = cos (angle p v - angle p w) := by
  simp [circ, Fin.sum_univ_two, cos_sub]

theorem sum_circ_sq (v : Fin p) : ∑ r, circ p v r ^ 2 = 1 := by
  simp [circ, Fin.sum_univ_two, cos_sq_add_sin_sq]

/-- The embedding's coefficients: `b` on the bias, `1` on the code `(3, v)`, `ψ(v)` on the output
plane, `0` on the phase planes. -/
noncomputable def embC (b : ℝ) (v : Fin p) : Slot p → ℝ
  | .inl _ => b
  | .inr (.inl (j, a)) => if a = v then (if j = 3 then 1 else 0) else 0
  | .inr (.inr (.inl _)) => 0
  | .inr (.inr (.inr r)) => circ p v r

/-- **Every token has the squared length `b² + 2`.** -/
theorem sum_embC_sq (b : ℝ) (v : Fin p) : ∑ s, embC b v s ^ 2 = b ^ 2 + 2 := by
  have hc : ∑ x : Fin 4 × Fin p, embC b v (.inr (.inl x)) ^ 2 = 1 := by
    rw [Fintype.sum_prod_type]
    simp only [embC, ite_pow, zero_pow two_ne_zero, sum_ite_eq', mem_univ, ite_true]
    simp
  have ho : ∑ r : Fin 2, embC b v (.inr (.inr (.inr r))) ^ 2 = 1 := sum_circ_sq v
  rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_sum_type, hc, ho]
  simp only [embC, Fintype.univ_unit, sum_singleton]
  simp
  ring

/-- The coefficients of a state of the last position: `k` times the embedding of the own token
`t 3`, `a` on the codes `(j, t j)` of the other operands `j ≠ 3`, the phases `P c ψ(w c)` of the
copies and `M ψ(f)` added on the output plane. -/
noncomputable def stateC (b k a : ℝ) (t : Fin 4 → Fin p) (P : Fin 2 → ℝ) (w : Fin 2 → Fin p)
    (M : ℝ) (f : Fin p) : Slot p → ℝ
  | .inl _ => k * b
  | .inr (.inl (j, c)) => if c = t j then (if j = 3 then k else a) else 0
  | .inr (.inr (.inl (c, r))) => P c * circ p (w c) r
  | .inr (.inr (.inr r)) => k * circ p (t 3) r + M * circ p f r

variable (b : ℝ) (t : Fin 4 → Fin p) (w : Fin 2 → Fin p) (f : Fin p)

/-- The embedding of the own token times `k` is a state. -/
theorem smul_embC_eq_stateC (k : ℝ) : k • embC b (t 3) = stateC b k 0 t 0 w 0 f := by
  funext s
  rcases s with _ | ⟨j, c⟩ | ⟨c, r⟩ | r <;> simp [embC, stateC]
  split_ifs <;> simp_all

/-- Sublayers add states. -/
theorem stateC_add (k a M k' a' M' : ℝ) (P P' : Fin 2 → ℝ) :
    stateC b k a t P w M f + stateC b k' a' t P' w M' f =
      stateC b (k + k') (a + a') t (P + P') w (M + M') f := by
  funext s
  rcases s with _ | ⟨j, c⟩ | ⟨c, r⟩ | r <;> simp only [stateC, Pi.add_apply]
  · ring
  · split_ifs <;> ring
  · ring
  · ring

/-- The RMS scales states. -/
theorem smul_stateC (κ k a M : ℝ) (P : Fin 2 → ℝ) :
    κ • stateC b k a t P w M f = stateC b (κ * k) (κ * a) t (κ • P) w (κ * M) f := by
  funext s
  rcases s with _ | ⟨j, c⟩ | ⟨c, r⟩ | r <;> simp only [stateC, Pi.smul_apply, smul_eq_mul]
  · ring
  · split_ifs <;> ring
  · ring
  · ring

/-- **Before the output is written, the squared length does not depend on the tokens**:
`k² (b² + 2) + 3a² + P₀² + P₁²`. -/
theorem sum_stateC_sq (k a : ℝ) (P : Fin 2 → ℝ) :
    ∑ s, stateC b k a t P w 0 f s ^ 2 = k ^ 2 * (b ^ 2 + 2) + 3 * a ^ 2 + (P 0 ^ 2 + P 1 ^ 2) := by
  have hP (c : Fin 2) : ∑ r, (P c * circ p (w c) r) ^ 2 = P c ^ 2 := by
    simp only [mul_pow, ← mul_sum, sum_circ_sq, mul_one]
  have hk : ∑ r, (k * circ p (t 3) r) ^ 2 = k ^ 2 := by
    simp only [mul_pow, ← mul_sum, sum_circ_sq, mul_one]
  simp only [Fintype.sum_sum_type, Fintype.sum_prod_type, stateC, Fin.sum_univ_four,
    Fintype.univ_unit, sum_singleton, zero_mul, add_zero, hP, hk, Fin.sum_univ_two, ite_pow,
    zero_pow two_ne_zero, sum_ite_eq', mem_univ, ite_true]
  norm_num
  ring

/-- **The readout of a state against the embedding of `v`.** -/
theorem sum_embC_mul_stateC (v : Fin p) (k a M : ℝ) (P : Fin 2 → ℝ) :
    ∑ s, embC b v s * stateC b k a t P w M f s =
      k * (b ^ 2 + (if v = t 3 then 1 else 0) + cos (angle p v - angle p (t 3))) +
        M * cos (angle p v - angle p f) := by
  have hc : ∑ x : Fin 4 × Fin p, embC b v (.inr (.inl x)) * stateC b k a t P w M f (.inr (.inl x)) =
      if v = t 3 then k else 0 := by
    rw [Fintype.sum_prod_type]
    simp only [embC, stateC, ite_mul, zero_mul, sum_ite_eq', mem_univ, ite_true]
    simp
  have hph : ∑ x : Fin 2 × Fin 2,
      embC b v (.inr (.inr (.inl x))) * stateC b k a t P w M f (.inr (.inr (.inl x))) = 0 := by
    simp [embC]
  have ho : ∑ r : Fin 2,
      embC b v (.inr (.inr (.inr r))) * stateC b k a t P w M f (.inr (.inr (.inr r))) =
      k * cos (angle p v - angle p (t 3)) + M * cos (angle p v - angle p f) := by
    simp only [embC, stateC, mul_add, sum_add_distrib, mul_left_comm _ k, mul_left_comm _ M,
      ← mul_sum, sum_circ_mul]
  rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_sum_type, hc, hph, ho]
  simp only [embC, stateC, Fintype.univ_unit, sum_singleton]
  split_ifs <;> ring

/-- The angle of a sum is the sum of the angles, up to a turn. -/
theorem angle_add (v w : Fin p) :
    angle p (v + w) = angle p v + angle p w - ((((v : ℕ) + w) / p : ℕ) : ℝ) * (2 * π) := by
  have hp : (p : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (Fin.pos v).ne'
  have h := congrArg (Nat.cast : ℕ → ℝ) (Nat.mod_add_div ((v : ℕ) + w) p)
  push_cast at h
  have h' : (((v + w : Fin p) : ℕ) : ℝ) = (v : ℝ) + w - p * ((((v : ℕ) + w) / p : ℕ) : ℝ) := by
    rw [Fin.val_add]
    linarith
  rw [angle, angle, angle, h']
  field_simp

/-- **`ψ(v) ψ(w) = ψ(v + w)`**, the real part. -/
theorem cos_angle_add (v w : Fin p) : cos (angle p (v + w)) = cos (angle p v + angle p w) := by
  rw [angle_add, cos_sub_nat_mul_two_pi]

/-- `ψ(v) ψ(w) = ψ(v + w)`, the imaginary part. -/
theorem sin_angle_add (v w : Fin p) : sin (angle p (v + w)) = sin (angle p v + angle p w) := by
  rw [angle_add, sin_sub_nat_mul_two_pi]

example (v w : Fin 5) : cos (angle 5 (v + w)) = cos (angle 5 v + angle 5 w) := cos_angle_add v w

/-- **Two different tokens are at least `2π/p` apart on the circle**:
`cos(θ_v − θ_w) ≤ cos(2π/p)`. -/
theorem cos_angle_sub_le {v w : Fin p} (h : v ≠ w) :
    cos (angle p v - angle p w) ≤ cos (2 * π / p) := by
  wlog hlt : (w : ℕ) < v generalizing v w
  · rw [← cos_neg, neg_sub]
    exact this h.symm (lt_of_le_of_ne (not_lt.1 hlt) fun e => h (Fin.ext e))
  have hp : (0 : ℝ) < p := Nat.cast_pos.2 (Fin.pos v)
  have hv := v.2
  have hn1 : (1 : ℝ) ≤ ((v : ℕ) - w : ℕ) := by exact_mod_cast (by omega : 1 ≤ (v : ℕ) - w)
  have hnp : (((v : ℕ) - w : ℕ) : ℝ) + 1 ≤ p := by exact_mod_cast (by omega : (v : ℕ) - w + 1 ≤ p)
  have hx : angle p v - angle p w = 2 * π * (((v : ℕ) - w : ℕ) : ℝ) / p := by
    rw [angle, angle, Nat.cast_sub hlt.le]
    ring
  rw [hx]
  have h1 : 2 * π / p ≤ 2 * π * (((v : ℕ) - w : ℕ) : ℝ) / p :=
    div_le_div_of_nonneg_right (by nlinarith [pi_pos]) hp.le
  rcases le_or_gt (2 * π * (((v : ℕ) - w : ℕ) : ℝ) / p) π with h2 | h2
  · exact cos_le_cos_of_nonneg_of_le_pi (by positivity) h2 h1
  · rw [← cos_two_pi_sub]
    refine cos_le_cos_of_nonneg_of_le_pi (by positivity) (by linarith) ?_
    rw [show 2 * π - 2 * π * (((v : ℕ) - w : ℕ) : ℝ) / p =
      2 * π * (p - (((v : ℕ) - w : ℕ) : ℝ)) / p by field_simp]
    exact div_le_div_of_nonneg_right (by nlinarith [pi_pos]) hp.le

/-- On `ℤ/5` the tokens `1` and `3` are at least `2π/5` apart. -/
example : cos (angle 5 1 - angle 5 3) ≤ cos (2 * π / 5) := by
  exact_mod_cast cos_angle_sub_le (p := 5) (by decide : (1 : Fin 5) ≠ 3)

example : ∑ s, embC 1 (0 : Fin 5) s * stateC 1 1 0 (fun _ => 0) 0 (fun _ => 0) 0 0 s =
    1 * (1 ^ 2 + (if (0 : Fin 5) = 0 then 1 else 0) + cos (angle 5 0 - angle 5 0)) +
      0 * cos (angle 5 0 - angle 5 0) :=
  sum_embC_mul_stateC 1 _ _ 0 0 1 0 0 0

end Transformer.Phasor.TwoCopies
