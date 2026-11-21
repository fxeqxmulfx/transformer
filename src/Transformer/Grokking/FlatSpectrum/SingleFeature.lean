module

public import Mathlib.Algebra.Group.TypeTags.Finite
public import Mathlib.Algebra.Polynomial.Roots
public import Mathlib.FieldTheory.Finite.Basic
public import Mathlib.SetTheory.Cardinal.Finite
public import Mathlib.Tactic.LinearCombination

/-!
# Tasks that are no function of a single feature

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV11. A character turns one
operation into a product, so for `∘` the sum or product of `ℤ/p`, `h(u(a) ∘ v(b))` is a readout of
rank-one characters of one feature, as `a + a b = a (1 + b)` is. Which `w(a) + a b` are not?

For no `w` is it a function of a product in a group of at most `p` elements, on the nonzero rows
(`not_group_feature`). It is a function of a product in `ℤ/p` exactly when `w` is affine on `a ≠ 0`
(`exists_mul_feature_iff`), and `a ^ k` is not, for `2 ≤ k ≤ p − 2` (`pow_not_affine`). So for
`p ≥ 5` the task `a³ + a b` has no such feature in any group or type with zero of at most `p`
elements (`cubic_not_group_feature`, `cubic_not_addGroup_feature`, `cubic_not_mulZero_feature`).
Both bounds are needed (`cubic_mul_feature_three`, `exists_prod_feature`).

This is about one feature, not a trained model: it does not say that a model fails on such a task,
and several features together can compute it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Polynomial

local instance : Fact (Nat.Prime 97) := ⟨by decide⟩

variable {p : ℕ} [Fact p.Prime]

/-- **The column map is a bijection.** Row `a = 1` of `w a + a b` is injective in `b`, so `v` is
injective, and a target of at most `p` elements makes it bijective. -/
theorem bijective_of_feature {M : Type*} [Mul M] [Finite M] (hM : Nat.card M ≤ p)
    {w : ZMod p → ZMod p} {u v : ZMod p → M} {h : M → ZMod p}
    (hf : ∀ a ≠ 0, ∀ b, w a + a * b = h (u a * v b)) : Function.Bijective v := by
  refine Function.Injective.bijective_of_nat_card_le (fun b b' hb => ?_) (by rwa [Nat.card_zmod])
  have hrow := hf 1 one_ne_zero b
  rw [hb, ← hf 1 one_ne_zero b'] at hrow
  simpa using hrow

example := bijective_of_feature (p := 97) (M := ZMod 97) (Nat.card_zmod 97).le
  (w := id) (u := id) (v := (· + 1)) (h := id) fun a _ b => by simp only [id]; ring

/-- **A zero forces an affine term.** If `w a + a b = h (u a * v b)` on the nonzero rows, with the
product in a type with zero of at most `p` elements, then `w` is affine on `a ≠ 0`: the column
`b₀` with `v b₀ = 0` has the same answer `h 0` in every nonzero row. -/
theorem exists_affine_of_mul_feature {M : Type*} [MulZeroClass M] [Finite M]
    (hM : Nat.card M ≤ p) {w : ZMod p → ZMod p} {u v : ZMod p → M} {h : M → ZMod p}
    (hf : ∀ a ≠ 0, ∀ b, w a + a * b = h (u a * v b)) :
    ∃ c d, ∀ a ≠ 0, w a = c + d * a := by
  obtain ⟨b₀, hb₀⟩ := (bijective_of_feature hM hf).2 0
  refine ⟨h 0, -b₀, fun a ha => ?_⟩
  have hrow := hf a ha b₀
  rw [hb₀, mul_zero] at hrow
  linear_combination hrow

example := exists_affine_of_mul_feature (p := 97) (M := ZMod 97) (Nat.card_zmod 97).le
  (w := id) (u := id) (v := (· + 1)) (h := id) fun a _ b => by simp only [id]; ring

/-- **No group carries `w a + a b`.** For `p ≥ 3` and any `w`, the table on the nonzero rows is not
a function of `u a * v b` in a group of at most `p` elements. Passing from row `1` to row `-1`
would be a translation of the group, with no fixed point unless it is the identity. On columns it
is the reflection `b ↦ w (-1) - w 1 - b`, which has a fixed point and is not the identity. -/
theorem not_group_feature (hp : 3 ≤ p) {G : Type*} [Group G] [Finite G]
    (hG : Nat.card G ≤ p) (w : ZMod p → ZMod p) (u v : ZMod p → G) (h : G → ZMod p) :
    ¬ ∀ a ≠ 0, ∀ b, w a + a * b = h (u a * v b) := by
  intro hf
  have : Fact (2 < p) := ⟨hp⟩
  have hm1 : (-1 : ZMod p) ≠ 0 := neg_ne_zero.mpr one_ne_zero
  have h2 : (2 : ZMod p) ≠ 0 := fun h2 =>
    ZMod.neg_one_ne_one (n := p) (by linear_combination -h2)
  obtain ⟨b₀, hb₀⟩ : ∃ b₀, 2 * b₀ = w (-1) - w 1 :=
    ⟨(w (-1) - w 1) / 2, mul_div_cancel₀ _ h2⟩
  obtain ⟨b', hb'⟩ := (bijective_of_feature hG hf).2 ((u (-1))⁻¹ * u 1 * v b₀)
  have hcol : w (-1) + -1 * b' = w 1 + 1 * b₀ := by
    rw [hf _ hm1, hf _ one_ne_zero, hb', mul_assoc, mul_inv_cancel_left]
  have hfix : b' = b₀ := by linear_combination -hcol - hb₀
  rw [hfix] at hb'
  have hu : u (-1) = u 1 := inv_mul_eq_one.mp (mul_eq_right.mp hb'.symm)
  have hrows (b : ZMod p) : w (-1) + -1 * b = w 1 + 1 * b := by
    rw [hf _ hm1, hf _ one_ne_zero, hu]
  exact h2 (by linear_combination hrows 0 - hrows 1)

example := not_group_feature (p := 97) (by norm_num) (G := Multiplicative (ZMod 97))
  (by rw [Nat.card_congr Multiplicative.toAdd, Nat.card_zmod]) id
  (fun a => .ofAdd a) (fun b => .ofAdd b) (fun t => t.toAdd)

/-- `not_group_feature` for a group written additively, `h (u a + v b)`. -/
theorem not_addGroup_feature (hp : 3 ≤ p) {G : Type*} [AddGroup G] [Finite G]
    (hG : Nat.card G ≤ p) (w : ZMod p → ZMod p) (u v : ZMod p → G) (h : G → ZMod p) :
    ¬ ∀ a ≠ 0, ∀ b, w a + a * b = h (u a + v b) := fun hf =>
  not_group_feature hp (G := Multiplicative G) (by rwa [Nat.card_congr Multiplicative.toAdd]) w
    (fun a => .ofAdd (u a)) (fun b => .ofAdd (v b)) (fun t => h t.toAdd) (by simpa using hf)

example := not_addGroup_feature (p := 97) (by norm_num) (G := ZMod 97) (Nat.card_zmod 97).le
  id id id id

/-- **The criterion in `ℤ/p`.** On the nonzero rows, `w a + a b` is a function of one product of
values in `ℤ/p` exactly when `w` is affine there: `c + d a + a b = c + a (b + d)`. -/
theorem exists_mul_feature_iff (w : ZMod p → ZMod p) :
    (∃ u v h : ZMod p → ZMod p, ∀ a ≠ 0, ∀ b, w a + a * b = h (u a * v b)) ↔
      ∃ c d, ∀ a ≠ 0, w a = c + d * a := by
  constructor
  · rintro ⟨u, v, h, hf⟩
    exact exists_affine_of_mul_feature (Nat.card_zmod p).le hf
  · rintro ⟨c, d, hw⟩
    refine ⟨id, (· + d), (c + ·), fun a ha b => ?_⟩
    simp only [id, hw a ha]
    ring

/-- The positive control: `a + a b = a (1 + b)` has the feature the criterion asks for. -/
example : ∃ u v h : ZMod 97 → ZMod 97, ∀ a ≠ 0, ∀ b, a + a * b = h (u a * v b) :=
  (exists_mul_feature_iff fun a => a).mpr ⟨0, 1, fun a _ => by ring⟩

/-- **A power is not affine on the nonzero residues**, for `2 ≤ k ≤ p − 2`: the difference would be
a monic polynomial of degree `k` with `p − 1` roots. The upper bound is needed, since `a ^ (p - 1)`
is the constant `1` on `a ≠ 0`. -/
theorem pow_not_affine {k : ℕ} (hk : 2 ≤ k) (hkp : k + 2 ≤ p) :
    ¬ ∃ c d : ZMod p, ∀ a ≠ 0, a ^ k = c + d * a := by
  rintro ⟨c, d, hcd⟩
  have hlin : (-(C d * X + C c) : (ZMod p)[X]).degree < k :=
    ((degree_neg _).le.trans degree_linear_le).trans_lt (by exact_mod_cast (hk : 1 < k))
  have hmonic := monic_X_pow_add hlin
  have hdeg : (X ^ k + -(C d * X + C c) : (ZMod p)[X]).natDegree ≤ k := by
    refine (natDegree_add_le _ _).trans (max_le (natDegree_X_pow_le k) ?_)
    rw [natDegree_neg]
    exact natDegree_linear_le.trans (by omega)
  have hroots : (Finset.univ.erase (0 : ZMod p)).val ⊆ (X ^ k + -(C d * X + C c)).roots := by
    intro a ha
    rw [mem_roots hmonic.ne_zero, IsRoot.def]
    simp only [eval_add, eval_neg, eval_pow, eval_X, eval_mul, eval_C]
    rw [hcd a (Finset.ne_of_mem_erase (Finset.mem_def.mpr ha))]
    ring
  have hcard := (card_le_degree_of_subset_roots hroots).trans hdeg
  rw [Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, ZMod.card] at hcard
  omega

example := pow_not_affine (p := 97) (k := 3) (by norm_num) (by norm_num)

/-- **`a ^ k + a b` has no feature with a zero**, for `2 ≤ k ≤ p − 2`, in a type with zero of at
most `p` elements. For `k = 2` this is `a² + a b`. -/
theorem pow_add_mul_not_mulZero_feature {k : ℕ} (hk : 2 ≤ k) (hkp : k + 2 ≤ p)
    {M : Type*} [MulZeroClass M] [Finite M] (hM : Nat.card M ≤ p)
    (u v : ZMod p → M) (h : M → ZMod p) :
    ¬ ∀ a ≠ 0, ∀ b, a ^ k + a * b = h (u a * v b) := fun hf =>
  pow_not_affine hk hkp (exists_affine_of_mul_feature hM hf)

example := pow_add_mul_not_mulZero_feature (p := 97) (k := 2) le_rfl (by norm_num)
  (M := ZMod 97) (Nat.card_zmod 97).le id id id

/-- **`a³ + a b` in a group.** It is no function of `u a * v b` in a group of at most `p` elements.
The proof needs only `p ≥ 3`. -/
theorem cubic_not_group_feature (hp : 3 ≤ p) {G : Type*} [Group G] [Finite G]
    (hG : Nat.card G ≤ p) (u v : ZMod p → G) (h : G → ZMod p) :
    ¬ ∀ a b : ZMod p, a ^ 3 + a * b = h (u a * v b) := fun hf =>
  not_group_feature hp hG (· ^ 3) u v h fun a _ b => hf a b

example := cubic_not_group_feature (p := 97) (by norm_num) (G := Multiplicative (ZMod 97))
  (by rw [Nat.card_congr Multiplicative.toAdd, Nat.card_zmod])
  (fun a => .ofAdd a) (fun b => .ofAdd b) (fun t => t.toAdd)

/-- **`a³ + a b` in an additive group**: no function of `u a + v b`, for example in `ℤ/p`. -/
theorem cubic_not_addGroup_feature (hp : 3 ≤ p) {G : Type*} [AddGroup G] [Finite G]
    (hG : Nat.card G ≤ p) (u v : ZMod p → G) (h : G → ZMod p) :
    ¬ ∀ a b : ZMod p, a ^ 3 + a * b = h (u a + v b) := fun hf =>
  not_addGroup_feature hp hG (· ^ 3) u v h fun a _ b => hf a b

example := cubic_not_addGroup_feature (p := 97) (by norm_num) (G := ZMod 97)
  (Nat.card_zmod 97).le id id id

/-- **`a³ + a b` with a zero.** For `p ≥ 5` it is no function of `u a * v b` in a type with zero of
at most `p` elements, for example `ℤ/p` with multiplication. -/
theorem cubic_not_mulZero_feature (hp : 5 ≤ p) {M : Type*} [MulZeroClass M] [Finite M]
    (hM : Nat.card M ≤ p) (u v : ZMod p → M) (h : M → ZMod p) :
    ¬ ∀ a b : ZMod p, a ^ 3 + a * b = h (u a * v b) := fun hf =>
  pow_add_mul_not_mulZero_feature (by norm_num) hp hM u v h fun a _ b => hf a b

example := cubic_not_mulZero_feature (p := 97) (by norm_num) (M := ZMod 97)
  (Nat.card_zmod 97).le id id id

/-- The bound `p ≥ 5` of `cubic_not_mulZero_feature` is needed: at `p = 3` the cube is the identity,
so `a³ + a b = a (b + 1)`. -/
theorem cubic_mul_feature_three :
    ∃ u v h : ZMod 3 → ZMod 3, ∀ a b, a ^ 3 + a * b = h (u a * v b) :=
  ⟨id, (· + 1), id, fun a b => by simp only [id, ZMod.pow_card]; ring⟩

/-- The bound on the size of the group is needed: every table is a function of `u a + v b` in
`ℤ/n × ℤ/n`, which has `n²` elements. -/
theorem exists_prod_feature {n : ℕ} (f : ZMod n → ZMod n → ZMod n) :
    ∃ (u v : ZMod n → ZMod n × ZMod n) (h : ZMod n × ZMod n → ZMod n),
      ∀ a b, f a b = h (u a + v b) :=
  ⟨fun a => (a, 0), fun b => (0, b), fun t => f t.1 t.2, by simp⟩

end Transformer.Grokking.FlatSpectrum
