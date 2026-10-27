module

public import Transformer.Grokking.FlatSpectrum.Kernels
public import Mathlib.FieldTheory.Finite.Basic

/-!
# The cubic sums `S_r = ∑_x ψ(r x³)`

Source: `review/flat_spectrum.md`, Theorem B. For prime `p`: `S_{r h³} = S_r` for `h ≠ 0`
(substitute `x ↦ h x`); `S_{−r} = conj S_r`; `∑_c S_c = p`, since only `x = 0` has
`x³ = 0`; and `∑_c |S_c|² = p #{x³ = y³} ≤ p (3p − 2)`, since `x³ = a` has at most three
roots.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- `S_r = ∑_x ψ(r x³)`, `review/flat_spectrum.md`, Theorem B. -/
noncomputable def cubicSum (r : ZMod p) : ℂ := ∑ x : ZMod p, ZMod.stdAddChar (r * x ^ 3)

/-- `S_0 = p`. -/
lemma cubicSum_zero : cubicSum (0 : ZMod p) = p := by
  simp [cubicSum, ZMod.card]

example := @cubicSum_zero 7 ⟨Nat.prime_seven⟩

/-- `S_{r h³} = S_r` for `h ≠ 0`: substitute `x ↦ h x`. -/
lemma cubicSum_mul_cube (r : ZMod p) {h : ZMod p} (hh : h ≠ 0) :
    cubicSum (r * h ^ 3) = cubicSum r :=
  Fintype.sum_equiv (Equiv.mulLeft₀ h hh) _ _ fun x => by
    rw [Equiv.mulLeft₀_apply]
    congr 1
    ring

example := @cubicSum_mul_cube 7 ⟨Nat.prime_seven⟩ 1 2 (by decide)

/-- `S_{−r} = conj S_r`. -/
lemma cubicSum_neg (r : ZMod p) : cubicSum (-r) = star (cubicSum r) := by
  simp only [cubicSum, star_sum, neg_mul, AddChar.map_neg_eq_conj, Complex.star_def]

example := @cubicSum_neg 7 ⟨Nat.prime_seven⟩ 1

/-- `∑_c S_c = p`: only `x = 0` has `x³ = 0`. -/
lemma sum_cubicSum : ∑ c : ZMod p, cubicSum c = p := by
  simp only [cubicSum]
  rw [Finset.sum_comm]
  simp [sum_stdAddChar_mul]

example := @sum_cubicSum 7 ⟨Nat.prime_seven⟩

/-- `∑_c |S_c|² = p #{(x, y) | x³ = y³}`. -/
lemma sum_norm_cubicSum_sq :
    ∑ c : ZMod p, ‖cubicSum c‖ ^ 2 = p * #{xy : ZMod p × ZMod p | xy.1 ^ 3 = xy.2 ^ 3} := by
  have h (c : ZMod p) : (‖cubicSum c‖ : ℂ) ^ 2 =
      ∑ xy : ZMod p × ZMod p, ZMod.stdAddChar (c * (xy.1 ^ 3 - xy.2 ^ 3)) := by
    rw [← Complex.mul_conj', cubicSum, map_sum, Finset.sum_mul_sum,
      ← Finset.sum_product']
    refine Finset.sum_congr rfl fun xy _ => ?_
    rw [← AddChar.map_neg_eq_conj, ← AddChar.map_add_eq_mul]
    congr 1
    ring
  apply Complex.ofReal_injective
  push_cast
  simp_rw [h]
  rw [Finset.sum_comm]
  simp_rw [sum_stdAddChar_mul, sub_eq_zero]
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul, mul_comm]

example := @sum_norm_cubicSum_sq 7 ⟨Nat.prime_seven⟩

/-- `x³ = a` has at most three roots. -/
lemma card_pow_three_eq_le (a : ZMod p) : #{x : ZMod p | x ^ 3 = a} ≤ 3 := by
  calc #{x : ZMod p | x ^ 3 = a} ≤ #(Polynomial.nthRootsFinset 3 a) := by
        refine Finset.card_le_card fun x hx => ?_
        rw [Polynomial.mem_nthRootsFinset (by norm_num)]
        simpa using hx
    _ ≤ Multiset.card (Polynomial.nthRoots 3 a) := by
        rw [Polynomial.nthRootsFinset_def]
        exact Multiset.toFinset_card_le _
    _ ≤ 3 := Polynomial.card_nthRoots 3 a

example := @card_pow_three_eq_le 7 ⟨Nat.prime_seven⟩ 1

/-- `#{(x, y) | x³ = y³} ≤ 3 p − 2`: `y = 0` forces `x = 0`, and otherwise `x³ = y³` has
at most three roots. -/
lemma card_cube_eq_le : #{xy : ZMod p × ZMod p | xy.1 ^ 3 = xy.2 ^ 3} ≤ 3 * p - 2 := by
  have h1 : #{xy : ZMod p × ZMod p | xy.1 ^ 3 = xy.2 ^ 3} =
      ∑ y : ZMod p, #{x : ZMod p | x ^ 3 = y ^ 3} := by
    rw [Finset.card_filter, Fintype.sum_prod_type, Finset.sum_comm]
    simp only [Finset.card_filter]
  have h0 : #{x : ZMod p | x ^ 3 = (0 : ZMod p) ^ 3} = 1 := by
    simp [pow_eq_zero_iff, Finset.filter_eq']
  have hp := (Fact.out : p.Prime).one_lt
  rw [h1, ← Finset.add_sum_erase _ _ (Finset.mem_univ 0), h0]
  calc 1 + ∑ y ∈ univ.erase 0, #{x : ZMod p | x ^ 3 = y ^ 3}
      ≤ 1 + ∑ _y ∈ univ.erase (0 : ZMod p), 3 := by
        gcongr with y
        exact card_pow_three_eq_le _
    _ = 3 * p - 2 := by
        rw [Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ 0), Finset.card_univ,
          ZMod.card, smul_eq_mul]
        omega

example := @card_cube_eq_le 7 ⟨Nat.prime_seven⟩

end Transformer.Grokking.FlatSpectrum
