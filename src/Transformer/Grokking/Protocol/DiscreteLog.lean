module

public import Transformer.Grokking.Protocol.QuotientTransfer

/-!
# Division modulo 97 as a subtraction of logarithms

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, the
operation `x / y (mod p)` with `p = 97`. The construction is this repository's: it prepares
the parameters of the decoder that answer every equation (`SolverParams`), the joint witness
of the certificate of `QuotientTransfer`.

`5` generates the nonzero residues modulo 97 (`orderOf_five`, `exists_pow_five`). The feature
of a residue (`logFeature`) is its logarithm to the base `5`, from `0` to `95`, and `300` for
the residue `0`. The gap of an equation `a / b = c` (`logGap`) is the feature of `a` less
that of `b`:

* with `c ≠ 0` it is an integer from `-95` to `95` and `5` to the gap is `c`
  (`logGap_of_ne_zero`);
* with `c = 0` it is from `205` to `300` (`logGap_of_eq_zero`).

So the quotient is a function of the gap, an integer of 287 possible values, and `lookup` is
that function as the indicator of each residue (`lookup_logGap`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Grokking.DivisionOrbits

/-- The powers of `5`, with integer exponents, in the field of 97 elements. -/
def powFive (m : ℤ) : ZMod 97 := (5 : ZMod 97) ^ m

/-- **`5` generates the nonzero residues modulo 97**: its order is `96`. -/
theorem orderOf_five : orderOf (5 : ZMod 97) = 96 := by
  refine orderOf_eq_of_pow_and_pow_div_prime (by norm_num) (by decide) fun p hp hd => ?_
  have h : p = 2 ∨ p = 3 := by
    rw [show (96 : ℕ) = 2 ^ 5 * 3 by norm_num] at hd
    rcases (Nat.Prime.dvd_mul hp).1 hd with h | h
    · exact Or.inl ((Nat.prime_dvd_prime_iff_eq hp Nat.prime_two).1 (hp.dvd_of_dvd_pow h))
    · exact Or.inr ((Nat.prime_dvd_prime_iff_eq hp Nat.prime_three).1 h)
  rcases h with rfl | rfl <;> decide

/-- Every nonzero residue is a power of `5` with an exponent below `96`. -/
theorem exists_pow_five {n : ZMod 97} (hn : n ≠ 0) :
    ∃ k : ℕ, k < 96 ∧ (5 : ZMod 97) ^ k = n := by
  let g : (ZMod 97)ˣ := Units.mk0 5 (by decide)
  have hg : orderOf g = 96 := by rw [← orderOf_units]; exact orderOf_five
  have hinj : Function.Injective fun k : Fin 96 => g ^ (k : ℕ) := fun i j h =>
    Fin.ext (pow_injOn_Iio_orderOf (by rw [hg]; exact i.isLt) (by rw [hg]; exact j.isLt) h)
  have hbij := (Fintype.bijective_iff_injective_and_card _).2
    ⟨hinj, by rw [Fintype.card_fin, ZMod.card_units]⟩
  obtain ⟨k, hk⟩ := hbij.surjective (Units.mk0 n hn)
  exact ⟨k, k.isLt, by simpa [g] using congrArg Units.val hk⟩

/-- **The feature of a residue**: its logarithm to the base `5`, from `0` to `95`, and `300`
for the residue `0`. -/
noncomputable def logFeature (n : ZMod 97) : ℤ :=
  if h : n = 0 then 300 else ((Classical.choose (exists_pow_five h) : ℕ) : ℤ)

lemma logFeature_zero : logFeature 0 = 300 := dite_eq_left rfl

lemma logFeature_spec {n : ZMod 97} (hn : n ≠ 0) :
    0 ≤ logFeature n ∧ logFeature n < 96 ∧ powFive (logFeature n) = n := by
  have h := Classical.choose_spec (exists_pow_five hn)
  rw [logFeature, dite_eq_right hn]
  exact ⟨Int.natCast_nonneg _, by exact_mod_cast h.1, by rw [powFive, zpow_natCast]; exact h.2⟩

/-- The numerator of an equation, in the field. -/
def num (x : Equation) : ZMod 97 := numerator x

/-- The denominator of an equation, in the field. -/
def den (x : Equation) : ZMod 97 := denominator x

lemma num_eq_mul (x : Equation) : num x = den x * quotient x := rfl

lemma den_ne_zero (x : Equation) : den x ≠ 0 := fun h => by
  have h1 : (x.2 : ℕ) + 1 = 0 := congrArg Fin.val h
  omega

/-- **The gap of an equation**: the feature of its numerator less that of its denominator. -/
noncomputable def logGap (x : Equation) : ℤ := logFeature (num x) - logFeature (den x)

/-- With a nonzero quotient the gap is between `-95` and `95`, and `5` to the gap is the
quotient. -/
theorem logGap_of_ne_zero {x : Equation} (hx : quotient x ≠ 0) :
    -95 ≤ logGap x ∧ logGap x ≤ 95 ∧ powFive (logGap x) = quotient x := by
  have hn : num x ≠ 0 := fun h => hx ((numerator_eq_zero_iff x).1 h)
  obtain ⟨a0, a1, a2⟩ := logFeature_spec hn
  obtain ⟨b0, b1, b2⟩ := logFeature_spec (den_ne_zero x)
  refine ⟨by unfold logGap; omega, by unfold logGap; omega, ?_⟩
  rw [powFive] at a2 b2 ⊢
  rw [logGap, zpow_sub₀ (by decide), a2, b2, num_eq_mul, mul_div_cancel_left₀ _ (den_ne_zero x)]

/-- With the quotient zero the gap is between `205` and `300`. -/
theorem logGap_of_eq_zero {x : Equation} (hx : quotient x = 0) :
    205 ≤ logGap x ∧ logGap x ≤ 300 := by
  have hn : num x = 0 := (numerator_eq_zero_iff x).2 hx
  obtain ⟨b0, b1, -⟩ := logFeature_spec (den_ne_zero x)
  rw [logGap, hn, logFeature_zero]
  omega

/-- Every gap is between `-95` and `300`. -/
lemma logGap_mem (x : Equation) : -95 ≤ logGap x ∧ logGap x ≤ 300 := by
  by_cases hx : quotient x = 0
  · have := logGap_of_eq_zero hx
    omega
  · have := logGap_of_ne_zero hx
    omega

/-- What the hidden units of the solver compute of a gap `g` for the residue `r`: whether `g`
is from `-95` to `95` with `5` to it `r`, or `r` is `0` and `g` is at least `101`. -/
noncomputable def lookup (g : ℤ) (r : ZMod 97) : ℝ :=
  (if -95 ≤ g ∧ g ≤ 95 ∧ powFive g = r then 1 else 0) + if r = 0 ∧ 101 ≤ g then 1 else 0

/-- **The lookup of an equation's gap marks its quotient.** -/
theorem lookup_logGap (x : Equation) (r : ZMod 97) :
    lookup (logGap x) r = if r = quotient x then 1 else 0 := by
  by_cases hx : quotient x = 0
  · obtain ⟨h0, h1⟩ := logGap_of_eq_zero hx
    have hz : ¬(-95 ≤ logGap x ∧ logGap x ≤ 95 ∧ powFive (logGap x) = r) := fun h => by omega
    rw [lookup, ite_eq_right hz, zero_add, hx]
    exact if_congr (and_iff_left (by omega)) rfl rfl
  · obtain ⟨h0, h1, h2⟩ := logGap_of_ne_zero hx
    have hz : ¬(r = 0 ∧ 101 ≤ logGap x) := fun h => by omega
    rw [lookup, ite_eq_right hz, add_zero, h2]
    exact if_congr ⟨fun h => h.2.2.symm, fun h => ⟨h0, h1, h.symm⟩⟩ rfl rfl

/-! The witnesses: the residue `3`, the equation `3 / 5 = 20` and the equation `0 / 5 = 0`. -/

example := exists_pow_five (n := 3) (by decide)

example := logFeature_spec (n := 3) (by decide)

example := logGap_of_ne_zero (x := (20, 4)) (by decide)

example := logGap_of_eq_zero (x := (0, 4)) rfl

example : lookup (logGap (20, 4)) 20 = 1 := by
  rw [lookup_logGap]
  exact ite_eq_left rfl

end Transformer.Grokking.Protocol
