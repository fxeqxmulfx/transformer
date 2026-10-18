/-
# Phase model — the full division table

On the full table of division in a cyclic group of order `n`, only `0` is
orthogonal to every row (`eq_zero_of_forall_inner_divisionRow`): a vector
orthogonal to every row is additive on `ℤ/n`, hence zero.  The phases
`ψ_x = 2πjx/n` of a character keep every relation
(`exists_inner_intVec_divisionRow_characterPhase`), so they are a zero of the
energy `V_S` of every set `S` of rows (`divisionEnergy_characterPhase`), and
up to periods they are the only zeros of the energy `V_full` of the full
table (`divisionEnergy_univ_eq_zero_iff`): where every relation holds,
`x ↦ ψ_x mod 2π` is a homomorphism `ℤ/n → ℝ/2πℤ`, determined by `ψ_1`, with
`n ψ_1 ∈ 2πℤ`.  Every row has norm at most `3`
(`norm_intVec_divisionRow_le`), so the squared norms of the rows of `S` sum
to at most `9 |S|` (`sum_norm_intVec_divisionRow_sq_le`), a bound on the
Lipschitz constant of the gradient of `V_S`.
-/

import Transformer.Landscape.PhaseModel.Periods
import Mathlib.Topology.Instances.AddCircle.Defs

open Real

namespace Transformer.Landscape.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- **Only `0` is orthogonal to every row of the full division table**: such
a vector `h` has `h_{x+y} = h_x + h_y`, so `x ↦ h_x` is a homomorphism from
the finite group `ℤ/n` to `ℝ`, which is zero. -/
theorem eq_zero_of_forall_inner_divisionRow (h : EucSpace n)
    (hh : ∀ p : Fin n × Fin n, ⟪intVec (divisionRow p), h⟫ = 0) : h = 0 := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact PiLp.ext fun i => i.elim0
  have : NeZero n := ⟨hn.ne'⟩
  have hadd : ∀ a b : Fin n, h.ofLp (a + b) = h.ofLp a + h.ofLp b := fun a b => by
    have h1 := hh (a + b, a)
    rw [inner_intVec_divisionRow, add_sub_cancel_left] at h1
    linarith
  let g : Fin n →+ ℝ := AddMonoidHom.mk' (fun x => h.ofLp x) hadd
  refine PiLp.ext fun x => ?_
  have h1 := congrArg g (addOrderOf_nsmul_eq_zero x)
  rw [map_nsmul, map_zero, nsmul_eq_mul] at h1
  exact (mul_eq_zero.1 h1).resolve_left (Nat.cast_ne_zero.2 (addOrderOf_pos x).ne')

/-- **The phases of a character keep every relation of division**:
`2πj (x - y - ((x - y) mod n)) / n` is `0` or `-2πj`. -/
theorem exists_inner_intVec_divisionRow_characterPhase (j : ℤ) (p : Fin n × Fin n) :
    ∃ k : ℤ, ⟪intVec (divisionRow p), characterPhase n j⟫ = k * (2 * π) := by
  obtain ⟨x, y⟩ := p
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 x.pos.ne'
  rw [inner_intVec_divisionRow]
  simp only [characterPhase, WithLp.ofLp_toLp]
  rcases le_or_gt y x with hyx | hxy
  · refine ⟨0, ?_⟩
    rw [(Fin.coe_sub_iff_le).2 hyx, Nat.cast_sub hyx]
    field_simp
    ring
  · refine ⟨-j, ?_⟩
    rw [(Fin.coe_sub_iff_lt).2 hxy, Nat.cast_sub (by omega), Nat.cast_add]
    field_simp
    push_cast
    ring

/-- **The phases of every character are a zero of `V_S`** for every set `S`
of pairs. -/
theorem divisionEnergy_characterPhase (S : Finset (Fin n × Fin n)) (j : ℤ) :
    divisionEnergy S (characterPhase n j) = 0 :=
  (phaseEnergy_eq_zero_iff _ _).2 fun r => exists_inner_intVec_divisionRow_characterPhase j r.1

/-- **The zeros of the energy of the full division table are the phases of
the characters up to periods**: `V_full(ψ) = 0` exactly when
`ψ - (2πjx/n)_x ∈ 2πℤⁿ` for some `j ∈ ℤ`. -/
theorem divisionEnergy_univ_eq_zero_iff (ψ : EucSpace n) :
    divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ = 0 ↔
      ∃ j : ℤ, ψ - characterPhase n j ∈ periods n := by
  constructor
  · intro hψ
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · exact ⟨0, fun i => i.elim0⟩
    have : NeZero n := ⟨hn.ne'⟩
    have hrel := (phaseEnergy_eq_zero_iff _ ψ).1 hψ
    have hT : ∀ a b : Fin n, ((ψ.ofLp (a + b) : ℝ) : AddCircle (2 * π)) =
        (ψ.ofLp a : AddCircle (2 * π)) + (ψ.ofLp b : AddCircle (2 * π)) := by
      intro a b
      obtain ⟨k, hk⟩ := hrel ⟨(a + b, a), Finset.mem_univ _⟩
      rw [inner_intVec_divisionRow, add_sub_cancel_left] at hk
      rw [← AddCircle.coe_add, ← sub_eq_zero, ← AddCircle.coe_sub, AddCircle.coe_eq_zero_iff]
      exact ⟨k, by rw [zsmul_eq_mul]; linarith⟩
    let g : Fin n →+ AddCircle (2 * π) :=
      AddMonoidHom.mk' (fun x => (ψ.ofLp x : AddCircle (2 * π))) hT
    have hg : ∀ x : Fin n, g x = (x : ℕ) • g 1 := fun x => by
      rw [← map_nsmul]
      congr 1
      open Fin.NatCast in rw [nsmul_one, Fin.cast_val_eq_self]
    have hn1 : ((n • ψ.ofLp 1 : ℝ) : AddCircle (2 * π)) = 0 := by
      rw [AddCircle.coe_nsmul]
      change n • g 1 = 0
      rw [← map_nsmul]
      have h1 : n • (1 : Fin n) = 0 := by open Fin.NatCast in rw [nsmul_one, Fin.natCast_self]
      rw [h1, map_zero]
    obtain ⟨j, hj⟩ := (AddCircle.coe_eq_zero_iff (2 * π)).1 hn1
    rw [zsmul_eq_mul, nsmul_eq_mul] at hj
    refine ⟨j, fun x => ?_⟩
    have h2 : ((ψ.ofLp x - (characterPhase n j).ofLp x : ℝ) : AddCircle (2 * π)) = 0 := by
      rw [AddCircle.coe_sub, sub_eq_zero]
      change g x = _
      rw [hg]
      change (x : ℕ) • ((ψ.ofLp 1 : ℝ) : AddCircle (2 * π)) = _
      rw [← AddCircle.coe_nsmul]
      congr 1
      have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hn.ne'
      simp only [characterPhase, WithLp.ofLp_toLp, nsmul_eq_mul]
      field_simp
      linear_combination -(x : ℝ) * hj
    obtain ⟨k, hk⟩ := (AddCircle.coe_eq_zero_iff (2 * π)).1 h2
    exact ⟨k, by rw [WithLp.ofLp_sub, Pi.sub_apply, ← hk, zsmul_eq_mul]⟩
  · rintro ⟨j, hj⟩
    have h := phaseEnergy_add_of_mem_periods
      (fun p : (Finset.univ : Finset (Fin n × Fin n)) => divisionRow p.1) hj (characterPhase n j)
    rw [add_sub_cancel] at h
    rw [divisionEnergy, h]
    exact divisionEnergy_characterPhase _ j

/-- Every row of division has norm at most `3`. -/
theorem norm_intVec_divisionRow_le (p : Fin n × Fin n) : ‖intVec (divisionRow p)‖ ≤ 3 := by
  let e : Fin n → EucSpace n := fun i => PiLp.single 2 i 1
  have he : ∀ i, ‖e i‖ = 1 := fun i => by simp [e]
  have h : intVec (divisionRow p) = e p.1 - e p.2 - e (p.1 - p.2) :=
    PiLp.ext fun i => by simp [e, intVec, divisionRow, Pi.single_apply]
  rw [h]
  linarith [norm_sub_le (e p.1 - e p.2) (e (p.1 - p.2)), norm_sub_le (e p.1) (e p.2), he p.1,
    he p.2, he (p.1 - p.2)]

/-- The squared norms of the rows of a set `S` of pairs sum to at most
`9 |S|`. -/
theorem sum_norm_intVec_divisionRow_sq_le (S : Finset (Fin n × Fin n)) :
    ∑ p : S, ‖intVec (divisionRow p.1)‖ ^ 2 ≤ 9 * S.card := by
  calc ∑ p : S, ‖intVec (divisionRow p.1)‖ ^ 2 ≤ ∑ _p : S, (9 : ℝ) :=
        Finset.sum_le_sum fun p _ => by
          nlinarith [norm_intVec_divisionRow_le p.1, norm_nonneg (intVec (divisionRow p.1))]
    _ = 9 * S.card := by simp [mul_comm]

/-- The hypothesis of `eq_zero_of_forall_inner_divisionRow` holds for
`h = 0`. -/
example (n : ℕ) : ∀ p : Fin n × Fin n, ⟪intVec (divisionRow p), (0 : EucSpace n)⟫ = 0 :=
  fun _ => inner_zero_right _

end Transformer.Landscape.PhaseModel
