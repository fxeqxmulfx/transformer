module

public import Transformer.Grokking.FlatSpectrum.MulIntermediate

/-!
# Theorem B: the `b/a²` route

Source: `review/flat_spectrum.md`, Theorem B: "The b/a² route is no better. Since
1 + b/a² = f/a³, the only multiplicative characters of 1 + b/a² that the label reveals are the
cubic ones. Those equal χ₃(f)". As `b/a²` needs `a ≠ 0`, the pairs are uniform on
`(ℤ/p)ˣ × ℤ/p`. There `χ(1 + b/a²) = χ(f) χ³(a⁻¹)`, and each fiber `f = c` meets every
`a ≠ 0` once, so `E[χ(1 + b/a²) | f] = χ(f) E[χ³(a⁻¹)]`: it is `χ(f)` if `χ³ = 1` and `0`
otherwise.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- `1 + b/a² = f/a³` for `a ≠ 0`, `review/flat_spectrum.md`, Theorem B. -/
lemma one_add_div_sq_eq {a : ZMod p} (ha : a ≠ 0) (b : ZMod p) :
    1 + b / a ^ 2 = (a ^ 3 + a * b) / a ^ 3 := by
  field_simp

example := @one_add_div_sq_eq 5 ⟨Nat.prime_five⟩ 1 (by decide) 2

/-- The sum of a nontrivial character over the units is `0`. -/
lemma sum_units_mulChar_eq_zero {ψ : MulChar (ZMod p) ℂ} (hψ : ψ ≠ 1) :
    ∑ u : (ZMod p)ˣ, ψ u = 0 := by
  obtain ⟨v, hv⟩ := MulChar.ne_one_iff.mp hψ
  have h : ∑ u : (ZMod p)ˣ, ψ u = ψ v * ∑ u : (ZMod p)ˣ, ψ u := by
    rw [Finset.mul_sum]
    conv_lhs => rw [← Equiv.sum_comp (Equiv.mulLeft v)]
    simp only [Equiv.coe_mulLeft, Units.val_mul, map_mul]
  have h' : (1 - ψ v) * ∑ u : (ZMod p)ˣ, ψ u = 0 := by linear_combination h
  exact (mul_eq_zero.mp h').resolve_left (sub_ne_zero.mpr (Ne.symm hv))

example := @sum_units_mulChar_eq_zero 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `χ(1 + b/a²) = χ(f) χ³(a⁻¹)` for `a ≠ 0`. -/
lemma mulChar_one_add_div_sq (χ : MulChar (ZMod p) ℂ) (u : (ZMod p)ˣ) (b : ZMod p) :
    χ (1 + b / (u : ZMod p) ^ 2) = χ ((u : ZMod p) ^ 3 + u * b) * (χ ^ 3) ((u⁻¹ : (ZMod p)ˣ) :
      ZMod p) := by
  rw [one_add_div_sq_eq u.ne_zero, MulChar.pow_apply' χ (by norm_num), div_eq_mul_inv, map_mul,
    Units.val_inv_eq_inv_val, ← map_pow, inv_pow]

example := @mulChar_one_add_div_sq 5 ⟨Nat.prime_five⟩ 1 1 2

/-- "Those equal χ₃(f)": for `χ³ = 1` and `a ≠ 0`, `χ(1 + b/a²) = χ(f)`,
`review/flat_spectrum.md`, Theorem B. -/
theorem mulChar_one_add_div_sq_of_pow_three {χ : MulChar (ZMod p) ℂ} (h3 : χ ^ 3 = 1)
    (u : (ZMod p)ˣ) (b : ZMod p) :
    χ (1 + b / (u : ZMod p) ^ 2) = χ ((u : ZMod p) ^ 3 + u * b) := by
  rw [mulChar_one_add_div_sq, h3, MulChar.one_apply_coe, mul_one]

example := @mulChar_one_add_div_sq_of_pow_three 5 ⟨Nat.prime_five⟩ 1 (one_pow 3) 1 2

/-- Each fiber `a³ + a b = c` over `(ℤ/p)ˣ × ℤ/p` meets every `a` once:
`∑_{f = c} G(a) = ∑_a G(a)`. -/
lemma sum_fiber_units (c : ZMod p) (G : (ZMod p)ˣ → ℂ) :
    ∑ y : (ZMod p)ˣ × ZMod p with (y.1 : ZMod p) ^ 3 + y.1 * y.2 = c, G y.1 = ∑ u, G u := by
  have hrow (u : (ZMod p)ˣ) :
      ∑ b : ZMod p, (if (u : ZMod p) ^ 3 + u * b = c then G u else 0) = G u := by
    simp_rw [cube_add_mul_eq_iff u.ne_zero]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [Finset.sum_filter, Fintype.sum_prod_type]
  exact Finset.sum_congr rfl fun u _ => hrow u

example := @sum_fiber_units 5 ⟨Nat.prime_five⟩ 0 fun _ => 1

/-- The fiber `a³ + a b = c` over `(ℤ/p)ˣ × ℤ/p` has `p − 1` points. -/
lemma card_fiber_units (c : ZMod p) :
    (#{y : (ZMod p)ˣ × ZMod p | (y.1 : ZMod p) ^ 3 + y.1 * y.2 = c} : ℂ) = p - 1 := by
  have h := sum_fiber_units c fun _ => (1 : ℂ)
  simp only [Finset.sum_const, nsmul_eq_mul, mul_one, card_univ, ZMod.card_units] at h
  rw [h, Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]

example := @card_fiber_units 5 ⟨Nat.prime_five⟩ 0

/-- Over uniform `(a, b) ∈ (ℤ/p)ˣ × ℤ/p`, `E[χ(1 + b/a²) | f] = χ(f) ∑_a χ³(a⁻¹)/(p − 1)`. -/
lemma condMean_mulChar_one_add_div_sq (χ : MulChar (ZMod p) ℂ) (x : (ZMod p)ˣ × ZMod p) :
    condMean (fun ab : (ZMod p)ˣ × ZMod p => (ab.1 : ZMod p) ^ 3 + ab.1 * ab.2)
      (fun ab => χ (1 + ab.2 / (ab.1 : ZMod p) ^ 2)) x =
      χ ((x.1 : ZMod p) ^ 3 + x.1 * x.2) *
        (∑ u : (ZMod p)ˣ, (χ ^ 3) ((u⁻¹ : (ZMod p)ˣ) : ZMod p)) / (p - 1) := by
  have hsum : ∑ y : (ZMod p)ˣ × ZMod p with
      (y.1 : ZMod p) ^ 3 + y.1 * y.2 = (x.1 : ZMod p) ^ 3 + x.1 * x.2,
      χ (1 + y.2 / (y.1 : ZMod p) ^ 2) = χ ((x.1 : ZMod p) ^ 3 + x.1 * x.2) *
        ∑ u : (ZMod p)ˣ, (χ ^ 3) ((u⁻¹ : (ZMod p)ˣ) : ZMod p) := by
    rw [Finset.sum_congr rfl fun y hy => by
      rw [mulChar_one_add_div_sq, (Finset.mem_filter.mp hy).2], ← Finset.mul_sum]
    congr 1
    exact sum_fiber_units _ fun u => (χ ^ 3) ((u⁻¹ : (ZMod p)ˣ) : ZMod p)
  simp only [condMean]
  rw [hsum, card_fiber_units]

example := @condMean_mulChar_one_add_div_sq 5 ⟨Nat.prime_five⟩ 1 (1, 0)

/-- **The `b/a²` route, cubic characters**, `review/flat_spectrum.md`, Theorem B: "Those equal
χ₃(f)". Over uniform `(a, b) ∈ (ℤ/p)ˣ × ℤ/p`, `E[χ(1 + b/a²) | f] = χ(f)` if `χ³ = 1`. -/
theorem condMean_mulChar_one_add_div_sq_of_pow_three {χ : MulChar (ZMod p) ℂ} (h3 : χ ^ 3 = 1)
    (x : (ZMod p)ˣ × ZMod p) :
    condMean (fun ab : (ZMod p)ˣ × ZMod p => (ab.1 : ZMod p) ^ 3 + ab.1 * ab.2)
      (fun ab => χ (1 + ab.2 / (ab.1 : ZMod p) ^ 2)) x = χ ((x.1 : ZMod p) ^ 3 + x.1 * x.2) := by
  have hp1 : (p : ℂ) - 1 ≠ 0 := by
    rw [sub_ne_zero, Ne, Nat.cast_eq_one]
    exact (Fact.out : p.Prime).one_lt.ne'
  rw [condMean_mulChar_one_add_div_sq]
  simp only [h3, MulChar.one_apply_coe, Finset.sum_const, card_univ, ZMod.card_units,
    nsmul_eq_mul, mul_one, Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]
  field_simp

example := @condMean_mulChar_one_add_div_sq_of_pow_three 5 ⟨Nat.prime_five⟩ 1 (one_pow 3) (1, 0)

/-- **The `b/a²` route, other characters**, `review/flat_spectrum.md`, Theorem B: "the only
multiplicative characters of 1 + b/a² that the label reveals are the cubic ones". Over uniform
`(a, b) ∈ (ℤ/p)ˣ × ℤ/p`, `E[χ(1 + b/a²) | f] = 0` if `χ³ ≠ 1`. -/
theorem condMean_mulChar_one_add_div_sq_of_pow_three_ne {χ : MulChar (ZMod p) ℂ}
    (h3 : χ ^ 3 ≠ 1) (x : (ZMod p)ˣ × ZMod p) :
    condMean (fun ab : (ZMod p)ˣ × ZMod p => (ab.1 : ZMod p) ^ 3 + ab.1 * ab.2)
      (fun ab => χ (1 + ab.2 / (ab.1 : ZMod p) ^ 2)) x = 0 := by
  rw [condMean_mulChar_one_add_div_sq, show ∑ u : (ZMod p)ˣ, (χ ^ 3) ((u⁻¹ : (ZMod p)ˣ) : ZMod p) =
      ∑ u : (ZMod p)ˣ, (χ ^ 3) u from
      Equiv.sum_comp (Equiv.inv (ZMod p)ˣ) fun u : (ZMod p)ˣ => (χ ^ 3) (u : ZMod p),
    sum_units_mulChar_eq_zero h3, mul_zero, zero_div]

/-- The quadratic character is not cubic: `χ³ = χ ≠ 1` for odd `p`. -/
lemma quadraticChar_ringHomComp_pow_three_ne_one (hp : p ≠ 2) :
    (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) ^ 3 ≠ 1 := by
  rw [pow_succ, ((quadraticChar_isQuadratic _).comp _).sq_eq_one, one_mul]
  exact quadraticChar_ringHomComp_ne_one hp

example := @quadraticChar_ringHomComp_pow_three_ne_one 5 ⟨Nat.prime_five⟩ (by decide)

example := @condMean_mulChar_one_add_div_sq_of_pow_three_ne 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_pow_three_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) (1, 0)

end Transformer.Grokking.FlatSpectrum
