/-
# Phase model over a finite abelian group — rows, periods and characters

For an integer row `a`, `⟪a, v⟫ = Σ_x a_x v_x` (`inner_intVec`), a multiple
of `2π` when `v` is a period, so the periods leave a phase energy with
integer rows invariant (`phaseEnergy_add_of_mem_periods`).  The inner product
with the row of `x / y` is `ψ_x - ψ_y - ψ_{x-y}` (`inner_intVec_divisionRow`),
and only `0` is orthogonal to every row of the full table
(`eq_zero_of_forall_inner_divisionRow`): such a vector is a homomorphism
from the finite group `G` to `ℝ`.  A character `χ` has `e^{i arg χ(x)} = χ(x)`
(`exp_characterPhase`), so its phases keep every relation
(`exists_inner_intVec_divisionRow_characterPhase`) and leave the energy of
every set of rows invariant (`divisionEnergy_add_characterPhase`).  Up to
periods they are the only zeros of the energy `V_full` of the full table
(`divisionEnergy_univ_eq_zero_iff`): where every relation holds,
`x ↦ e^{iψ_x}` is a character.  Every row has norm at most `3`
(`norm_intVec_divisionRow_le`).
-/

import Transformer.PhaseModel.Abelian.Defs
import Transformer.PhaseModel.Basic

open Real

namespace Transformer.PhaseModel.Abelian

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- `e^{iθ} = 1` exactly when `θ ∈ 2πℤ`. -/
theorem exp_ofReal_mul_I_eq_one_iff (θ : ℝ) :
    Complex.exp (↑θ * Complex.I) = 1 ↔ ∃ k : ℤ, θ = k * (2 * π) := by
  rw [Complex.exp_eq_one_iff]
  refine exists_congr fun k => ⟨fun h => ?_, fun h => by rw [h]; push_cast; ring⟩
  have h1 : (θ : ℂ) * Complex.I = ((k * (2 * π) : ℝ) : ℂ) * Complex.I := by
    rw [h]
    push_cast
    ring
  exact_mod_cast mul_right_cancel₀ Complex.I_ne_zero h1

variable {G : Type*} [Fintype G]

/-- The inner product with an integer row: `⟪a, v⟫ = Σ_x a_x v_x`. -/
theorem inner_intVec (a : G → ℤ) (v : EuclideanSpace ℝ G) :
    ⟪intVec a, v⟫ = ∑ i, (a i : ℝ) * v.ofLp i := by
  simp [intVec, PiLp.inner_apply, mul_comm]

/-- The inner product of an integer row with a period is a multiple of `2π`. -/
theorem exists_inner_intVec_eq (a : G → ℤ) {v : EuclideanSpace ℝ G} (hv : v ∈ periods G) :
    ∃ k : ℤ, ⟪intVec a, v⟫ = k * (2 * π) := by
  choose k hk using hv
  refine ⟨∑ i, a i * k i, ?_⟩
  rw [inner_intVec]
  push_cast
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun i _ => by rw [hk i]; ring

/-- **The periods leave a phase energy with integer rows invariant.** -/
theorem phaseEnergy_add_of_mem_periods {R : Type*} [Fintype R] (a : R → G → ℤ)
    {v : EuclideanSpace ℝ G} (hv : v ∈ periods G) (ψ : EuclideanSpace ℝ G) :
    phaseEnergy (fun r => intVec (a r)) (ψ + v) = phaseEnergy (fun r => intVec (a r)) ψ :=
  phaseEnergy_add _ (fun r => exists_inner_intVec_eq (a r) hv) ψ

variable [AddCommGroup G]

/-- **The exponential of the phase of a character is the character**:
`e^{i arg χ(x)} = χ(x)`, as `|χ(x)| = 1`. -/
theorem exp_characterPhase (χ : AddChar G ℂ) (x : G) :
    Complex.exp (↑((characterPhase χ).ofLp x) * Complex.I) = χ x := by
  have h := Complex.norm_mul_exp_arg_mul_I (χ x)
  rw [AddChar.norm_apply, Complex.ofReal_one, one_mul] at h
  simpa only [characterPhase, WithLp.ofLp_toLp] using h

omit [Fintype G] in
/-- The trivial character `0`, of value `1`, has the phases `0`. -/
theorem characterPhase_zero : characterPhase (0 : AddChar G ℂ) = 0 :=
  PiLp.ext fun _ => by simp [characterPhase]

variable [DecidableEq G]

/-- The inner product with the row of `x / y` is `ψ_x - ψ_y - ψ_{x-y}`. -/
theorem inner_intVec_divisionRow (p : G × G) (ψ : EuclideanSpace ℝ G) :
    ⟪intVec (divisionRow p), ψ⟫ = ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2) := by
  simp [inner_intVec, divisionRow, Pi.single_apply, sub_mul, Finset.sum_sub_distrib]

/-- **The energy of a set `S` of rows of the division table of `G`** is
`Σ_{(x, y) ∈ S} (1 - cos (ψ_x - ψ_y - ψ_{x-y}))`. -/
theorem divisionEnergy_apply (S : Finset (G × G)) (ψ : EuclideanSpace ℝ G) :
    divisionEnergy S ψ =
      ∑ p ∈ S, (1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2))) := by
  rw [divisionEnergy, phaseEnergy, ← Finset.sum_coe_sort S]
  simp only [inner_intVec_divisionRow]

/-- **Only `0` is orthogonal to every row of the full division table**: such
a vector `h` has `h_{x+y} = h_x + h_y`, so `x ↦ h_x` is a homomorphism from
the finite group `G` to `ℝ`, which is zero. -/
theorem eq_zero_of_forall_inner_divisionRow (h : EuclideanSpace ℝ G)
    (hh : ∀ p : G × G, ⟪intVec (divisionRow p), h⟫ = 0) : h = 0 := by
  have hadd : ∀ a b : G, h.ofLp (a + b) = h.ofLp a + h.ofLp b := fun a b => by
    have h1 := hh (a + b, a)
    rw [inner_intVec_divisionRow, add_sub_cancel_left] at h1
    linarith
  let g : G →+ ℝ := AddMonoidHom.mk' (fun x => h.ofLp x) hadd
  refine PiLp.ext fun x => ?_
  have h1 := congrArg g (addOrderOf_nsmul_eq_zero x)
  rw [map_nsmul, map_zero, nsmul_eq_mul] at h1
  exact (mul_eq_zero.1 h1).resolve_left (Nat.cast_ne_zero.2 (addOrderOf_pos x).ne')

/-- **The phases of a character keep every relation of division**:
`e^{i(arg χ(x) - arg χ(y) - arg χ(x-y))} = χ(x) / χ(y) / χ(x - y) = 1`. -/
theorem exists_inner_intVec_divisionRow_characterPhase (χ : AddChar G ℂ) (p : G × G) :
    ∃ k : ℤ, ⟪intVec (divisionRow p), characterPhase χ⟫ = k * (2 * π) := by
  rw [← exp_ofReal_mul_I_eq_one_iff, inner_intVec_divisionRow, Complex.ofReal_sub,
    Complex.ofReal_sub, sub_mul, sub_mul, Complex.exp_sub, Complex.exp_sub, exp_characterPhase,
    exp_characterPhase, exp_characterPhase, AddChar.map_sub_eq_div]
  have h0 : ∀ x, χ x ≠ 0 := fun x h => by simpa [h] using AddChar.norm_apply χ x
  exact div_self (div_ne_zero (h0 _) (h0 _))

/-- **The phases of every character are a zero of `V_S`** for every set `S`
of pairs. -/
theorem divisionEnergy_characterPhase (S : Finset (G × G)) (χ : AddChar G ℂ) :
    divisionEnergy S (characterPhase χ) = 0 :=
  (phaseEnergy_eq_zero_iff _ _).2 fun r => exists_inner_intVec_divisionRow_characterPhase χ r.1

/-- **The phases of a character leave the energy of every set of rows
invariant**: `V_S(ψ + (arg χ(x))_x) = V_S(ψ)`. -/
theorem divisionEnergy_add_characterPhase (S : Finset (G × G)) (χ : AddChar G ℂ)
    (ψ : EuclideanSpace ℝ G) : divisionEnergy S (ψ + characterPhase χ) = divisionEnergy S ψ :=
  phaseEnergy_add (fun p : S => intVec (divisionRow p.1))
    (fun r => exists_inner_intVec_divisionRow_characterPhase χ r.1) ψ

/-- **The zeros of the energy of the full division table are the phases of
the characters up to periods**: `V_full(ψ) = 0` exactly when
`ψ - (arg χ(x))_x ∈ 2πℤ^G` for some character `χ`.  Where every relation
holds, `x ↦ e^{iψ_x}` is a character. -/
theorem divisionEnergy_univ_eq_zero_iff (ψ : EuclideanSpace ℝ G) :
    divisionEnergy (Finset.univ : Finset (G × G)) ψ = 0 ↔
      ∃ χ : AddChar G ℂ, ψ - characterPhase χ ∈ periods G := by
  constructor
  · intro hψ
    have hrel := (phaseEnergy_eq_zero_iff _ ψ).1 hψ
    have hmul : ∀ a b : G, Complex.exp (↑(ψ.ofLp (a + b)) * Complex.I) =
        Complex.exp (↑(ψ.ofLp a) * Complex.I) * Complex.exp (↑(ψ.ofLp b) * Complex.I) := by
      intro a b
      obtain ⟨k, hk⟩ := hrel ⟨(a + b, a), Finset.mem_univ _⟩
      rw [inner_intVec_divisionRow, add_sub_cancel_left] at hk
      rw [show ψ.ofLp (a + b) = ψ.ofLp a + ψ.ofLp b + k * (2 * π) by linarith,
        exp_ofReal_add_int_mul_two_pi_mul_I, ← Complex.exp_add]
      congr 1
      push_cast
      ring
    have h0 : Complex.exp (↑(ψ.ofLp 0) * Complex.I) = 1 := by
      have h := hmul 0 0
      rw [add_zero] at h
      exact (mul_eq_left₀ (Complex.exp_ne_zero _)).1 h.symm
    let χ : AddChar G ℂ :=
      { toFun := fun x => Complex.exp (↑(ψ.ofLp x) * Complex.I)
        map_zero_eq_one' := h0
        map_add_eq_mul' := hmul }
    refine ⟨χ, fun x => (exp_ofReal_mul_I_eq_one_iff _).1 ?_⟩
    rw [WithLp.ofLp_sub, Pi.sub_apply, Complex.ofReal_sub, sub_mul, Complex.exp_sub,
      exp_characterPhase]
    exact div_self (Complex.exp_ne_zero _)
  · rintro ⟨χ, hχ⟩
    have h := phaseEnergy_add_of_mem_periods
      (fun p : (Finset.univ : Finset (G × G)) => divisionRow p.1) hχ (characterPhase χ)
    rw [add_sub_cancel] at h
    rw [divisionEnergy, h]
    exact divisionEnergy_characterPhase _ χ

/-- Every row of division has norm at most `3`. -/
theorem norm_intVec_divisionRow_le (p : G × G) : ‖intVec (divisionRow p)‖ ≤ 3 := by
  let e : G → EuclideanSpace ℝ G := fun i => PiLp.single 2 i 1
  have he : ∀ i, ‖e i‖ = 1 := fun i => by simp [e]
  have h : intVec (divisionRow p) = e p.1 - e p.2 - e (p.1 - p.2) :=
    PiLp.ext fun i => by simp [e, intVec, divisionRow, Pi.single_apply]
  rw [h]
  linarith [norm_sub_le (e p.1 - e p.2) (e (p.1 - p.2)), norm_sub_le (e p.1) (e p.2), he p.1,
    he p.2, he (p.1 - p.2)]

/-- The squared norms of the rows of a set `S` of pairs sum to at most
`9 |S|`. -/
theorem sum_norm_intVec_divisionRow_sq_le (S : Finset (G × G)) :
    ∑ p : S, ‖intVec (divisionRow p.1)‖ ^ 2 ≤ 9 * S.card := by
  calc ∑ p : S, ‖intVec (divisionRow p.1)‖ ^ 2 ≤ ∑ _p : S, (9 : ℝ) :=
        Finset.sum_le_sum fun p _ => by
          nlinarith [norm_intVec_divisionRow_le p.1, norm_nonneg (intVec (divisionRow p.1))]
    _ = 9 * S.card := by simp [mul_comm]

/-- The hypothesis of `exists_inner_intVec_eq` and
`phaseEnergy_add_of_mem_periods` holds for the point with every coordinate
`2π`, and that of `eq_zero_of_forall_inner_divisionRow` for `h = 0`, here in
the Klein four-group `ℤ/2 × ℤ/2`. -/
example : (WithLp.toLp 2 fun _ : ZMod 2 × ZMod 2 => 2 * π) ∈ periods (ZMod 2 × ZMod 2) ∧
    ∀ p : (ZMod 2 × ZMod 2) × (ZMod 2 × ZMod 2),
      ⟪intVec (divisionRow p), (0 : EuclideanSpace ℝ (ZMod 2 × ZMod 2))⟫ = 0 :=
  ⟨fun _ => ⟨1, by simp⟩, fun _ => inner_zero_right _⟩

end Transformer.PhaseModel.Abelian
