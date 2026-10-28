module

public import Transformer.Grokking.FlatSpectrum.MulIntermediate

/-!
# Theorem B for any `P`: the fibers of `P(a) + a b`

Source: `review/flat_spectrum.md`, Theorem B, which takes `P(a) = a³`; here `P` is any function
of `a`. For prime `p` and `a ≠ 0`, `P(a) + a b = c` has the single solution
`b = a⁻¹ (c − P(a))`, where `a b = c − P(a)`; the line `a = 0` lies in the fiber `c = P(0)`.
So the fiber sum of `ψ_r(a b)` is `ψ_r(c) T + p [c = P(0)]` with `T = ∑_{a ≠ 0} ψ(−r P(a))`,
over `p − 1 + p [c = P(0)]` points, and over uniform `(a, b)`
`‖E[ψ_r(a b) | f]‖² = |T|²/p² + |ψ_r(P(0)) T + p|²/(p² (2p − 1))`. The fiber sum of a
multiplicative `χ(a b)` is `∑_{a ≠ 0} χ(c − P(a))`. A bound `B` on the complete sums
`∑_x ψ(r P(x))`, or `∑_a χ(c − P(a))` for `c ≠ P(0)`, bounds both readouts; for `P(a) = a³`
both bounds are `2 √p` (`norm_cubicSum_le`, `norm_sum_mulChar_sub_cube_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- For `a ≠ 0`, `P(a) + a b = c` has the single solution `b = a⁻¹ (c − P(a))`. -/
lemma poly_add_mul_eq_iff (P : ZMod p → ZMod p) {a b c : ZMod p} (ha : a ≠ 0) :
    P a + a * b = c ↔ b = a⁻¹ * (c - P a) := by
  constructor
  · rintro rfl
    field_simp
    ring
  · rintro rfl
    field_simp
    ring

example := @poly_add_mul_eq_iff 5 ⟨Nat.prime_five⟩ (· ^ 3) 1 1 2 (by decide)

/-- The rows `a ≠ 0` of the fiber `P(a) + a b = c`: one point each, with `a b = c − P(a)`. -/
lemma sum_row_poly_add_mul (P : ZMod p → ZMod p) (r c : ZMod p) {a : ZMod p} (ha : a ≠ 0) :
    ∑ b : ZMod p, (if P a + a * b = c then ZMod.stdAddChar (r * (a * b)) else 0) =
      ZMod.stdAddChar (r * c) * ZMod.stdAddChar (-(r * P a)) := by
  simp_rw [poly_add_mul_eq_iff P ha]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [← AddChar.map_add_eq_mul]
  congr 1
  field_simp
  ring

example := @sum_row_poly_add_mul 5 ⟨Nat.prime_five⟩ (· ^ 3) 1 2 1 (by decide)

/-- The fiber sum `∑_{P(a) + a b = c} ψ_r(a b) = ψ_r(c) T + p [c = P(0)]`, with
`T = ∑_{a ≠ 0} ψ(−r P(a))`. -/
lemma sum_fiber_poly_add_mul (P : ZMod p → ZMod p) (r c : ZMod p) :
    ∑ ab : ZMod p × ZMod p with P ab.1 + ab.1 * ab.2 = c, ZMod.stdAddChar (r * (ab.1 * ab.2)) =
      ZMod.stdAddChar (r * c) * ∑ a ∈ univ.erase 0, ZMod.stdAddChar (-(r * P a)) +
        if c = P 0 then (p : ℂ) else 0 := by
  rw [Finset.sum_filter, Fintype.sum_prod_type, ← Finset.add_sum_erase _ _ (mem_univ 0),
    Finset.sum_congr rfl fun a ha => sum_row_poly_add_mul P r c (ne_of_mem_erase ha),
    ← Finset.mul_sum, add_comm]
  congr 1
  simp only [zero_mul, add_zero, mul_zero, AddChar.map_zero_eq_one]
  by_cases h : c = P 0
  · subst h
    simp [ZMod.card]
  · simp [h, Ne.symm h]

example := @sum_fiber_poly_add_mul 5 ⟨Nat.prime_five⟩ (· ^ 3) 1 0

/-- The fiber size `#{P(a) + a b = c} = p − 1 + p [c = P(0)]`. -/
lemma card_fiber_poly_add_mul (P : ZMod p → ZMod p) (c : ZMod p) :
    (#{ab : ZMod p × ZMod p | P ab.1 + ab.1 * ab.2 = c} : ℝ) =
      p - 1 + if c = P 0 then (p : ℝ) else 0 := by
  have h := sum_fiber_poly_add_mul P 0 c
  simp only [zero_mul, AddChar.map_zero_eq_one, Finset.sum_const, nsmul_eq_mul, mul_one,
    neg_zero, one_mul, Finset.card_erase_of_mem (mem_univ _), Finset.card_univ, ZMod.card,
    Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one] at h
  have h' : ((#{ab : ZMod p × ZMod p | P ab.1 + ab.1 * ab.2 = c} : ℝ) : ℂ) =
      (((p : ℝ) - 1 + if c = P 0 then (p : ℝ) else 0 : ℝ) : ℂ) := by
    push_cast
    rw [h]
    split_ifs <;> simp
  exact_mod_cast h'

example := @card_fiber_poly_add_mul 5 ⟨Nat.prime_five⟩ (· ^ 3) 0

/-- **Theorem B for any `P`**, `review/flat_spectrum.md` with `a³` replaced by `P(a)`: over
uniform `(a, b) ∈ (ℤ/p)²` with label `f = P(a) + a b`,
`‖E[ψ_r(a b) | f]‖² = |T|²/p² + |ψ_r(P(0)) T + p|²/(p² (2p − 1))`,
`T = ∑_{a ≠ 0} ψ(−r P(a))`. -/
theorem meanSq_condMean_poly_add_mul (P : ZMod p → ZMod p) (r : ZMod p) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2)
      fun ab => ZMod.stdAddChar (r * (ab.1 * ab.2))) =
      ‖∑ a ∈ univ.erase 0, ZMod.stdAddChar (-(r * P a))‖ ^ 2 / p ^ 2 +
        ‖ZMod.stdAddChar (r * P 0) * ∑ a ∈ univ.erase 0, ZMod.stdAddChar (-(r * P a)) + p‖ ^ 2 /
          (p ^ 2 * (2 * p - 1)) := by
  have hp := (Fact.out : p.Prime).one_lt
  have hp1 : (1 : ℝ) < p := by exact_mod_cast hp
  rw [meanSq, sum_norm_condMean_sq]
  simp_rw [sum_fiber_poly_add_mul, card_fiber_poly_add_mul]
  set T := ∑ a ∈ univ.erase 0, ZMod.stdAddChar (-(r * P a))
  have hrest : ∀ c ∈ univ.erase (P 0),
      ‖ZMod.stdAddChar (r * c) * T + (if c = P 0 then (p : ℂ) else 0)‖ ^ 2 /
        ((p : ℝ) - 1 + if c = P 0 then (p : ℝ) else 0) = ‖T‖ ^ 2 / (p - 1) := by
    intro c hc
    have hψ : ‖(ZMod.stdAddChar (r * c) : ℂ)‖ = 1 := by simp
    simp only [ne_of_mem_erase hc, ite_false, add_zero, norm_mul, hψ, one_mul]
  rw [← Finset.add_sum_erase _ _ (mem_univ (P 0)), Finset.sum_congr rfl hrest,
    Finset.sum_const, Finset.card_erase_of_mem (mem_univ _), Finset.card_univ, ZMod.card,
    nsmul_eq_mul, Nat.cast_sub hp.le, Fintype.card_prod, ZMod.card]
  simp only [↓reduceIte]
  rw [show (p : ℝ) - 1 + p = 2 * p - 1 by ring]
  have h0 : (p : ℝ) ≠ 0 := by positivity
  have h1 : (p : ℝ) - 1 ≠ 0 := by linarith
  have h3 : 2 * (p : ℝ) - 1 ≠ 0 := by linarith
  push_cast
  field_simp
  ring

example := @meanSq_condMean_poly_add_mul 5 ⟨Nat.prime_five⟩ (· ^ 3) 1

/-- Theorem B for any `P` with a bound `|∑_x ψ(r P(x))| ≤ B`:
`‖E[ψ_r(a b) | f]‖² ≤ (B + 1)²/p² + (p + B + 1)²/(p² (2p − 1))`. -/
theorem meanSq_condMean_poly_add_mul_le (P : ZMod p → ZMod p) {r : ZMod p} {B : ℝ}
    (hB : ‖∑ x, ZMod.stdAddChar (r * P x)‖ ≤ B) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2)
      fun ab => ZMod.stdAddChar (r * (ab.1 * ab.2))) ≤
      (B + 1) ^ 2 / p ^ 2 + (p + B + 1) ^ 2 / (p ^ 2 * (2 * p - 1)) := by
  have hp2 : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
  have h1 : ‖(ZMod.stdAddChar (r * P 0) : ℂ)‖ = 1 := by simp
  have hT : ‖∑ a ∈ univ.erase 0, ZMod.stdAddChar (-(r * P a))‖ ≤ B + 1 := by
    have : ∑ a ∈ univ.erase 0, ZMod.stdAddChar (-(r * P a)) =
        star (∑ x, ZMod.stdAddChar (r * P x) - ZMod.stdAddChar (r * P 0)) := by
      rw [← Finset.sum_erase_eq_sub (mem_univ 0), star_sum]
      exact Finset.sum_congr rfl fun a _ => AddChar.map_neg_eq_conj _ _
    rw [this, norm_star]
    exact (norm_sub_le _ _).trans (by rw [h1]; linarith)
  have hA : ‖ZMod.stdAddChar (r * P 0) * ∑ a ∈ univ.erase 0, ZMod.stdAddChar (-(r * P a)) +
      (p : ℂ)‖ ≤ p + B + 1 := by
    refine (norm_add_le _ _).trans ?_
    rw [norm_mul, h1, one_mul, Complex.norm_natCast]
    linarith
  rw [meanSq_condMean_poly_add_mul]
  have h2p : (0 : ℝ) < 2 * p - 1 := by linarith
  gcongr

example := @meanSq_condMean_poly_add_mul_le 7 ⟨Nat.prime_seven⟩ (· ^ 3) 1 _
  (@norm_cubicSum_le 7 ⟨Nat.prime_seven⟩ 1 (by decide))

end Transformer.Grokking.FlatSpectrum
