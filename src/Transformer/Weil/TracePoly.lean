module

public import Mathlib.FieldTheory.Finite.Trace
public import Transformer.Weil.HasseDeriv

/-!
# The trace polynomial and its Hasse derivatives

Over a field of characteristic `p`, the trace polynomial of `f` is
`T_ν(f) = ∑_{k < ν} f^{p^k}`. On a finite field `F` with `p^ν` elements its value at `x` is
`Tr_{F/𝔽_p}(f(x))`, and for `ν ≥ 1` and `m = deg f ≥ 1` it has degree `m p^{ν − 1}`.

Its Hasse derivatives have low degree: a coefficient of `E^{(b)}(f^{p^k})` at `t` is
`binom(t + b, b)` times a coefficient of `f^{p^k}` at `t + b`, which vanishes unless
`p^k ∣ t + b`; then `binom(t + b, b) = 0` in characteristic `p` unless `p^k ∣ b`, and
`t + b ≤ m p^k ≤ m b`. So `deg E^{(b)}(T_ν(f)) ≤ (m − 1) b` for `b ≥ 1`, and on a level set
`{T_ν(f) = c}` every `E^{(b)}(T_ν(f)^i)` agrees with a polynomial of degree `≤ (m − 1) b`. This
is what Stepanov's method needs for the curve `y^p − y = f(x)`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- The trace polynomial `T_ν(f) = ∑_{k < ν} f^{p^k}`. -/
noncomputable def tracePoly {R : Type*} [CommSemiring R] (p ν : ℕ) (f : R[X]) : R[X] :=
  ∑ k ∈ range ν, f ^ p ^ k

section CharP

variable {F : Type*} [Field F] (p : ℕ) [Fact p.Prime] [CharP F p]

/-- `binom(n, b) = 0` in characteristic `p` if `p^k ∣ n` and `p^k ∤ b`: `(X + 1)^{p^k j}` is
`expand_{p^k}((X + 1)^j)`, which has no term `X^b` with `p^k ∤ b`. -/
theorem choose_eq_zero_of_pow_dvd {k n b : ℕ} (hn : p ^ k ∣ n) (hb : ¬p ^ k ∣ b) :
    (n.choose b : F) = 0 := by
  obtain ⟨j, rfl⟩ := hn
  have h : ((X + 1 : F[X]) ^ (p ^ k * j)) = expand F (p ^ k) ((X + 1) ^ j) := by
    rw [pow_mul, add_pow_char_pow, one_pow, map_pow, map_add, expand_X, map_one]
  have := congrArg (coeff · b) h
  simp only [coeff_X_add_one_pow, coeff_expand (pow_pos (Fact.out : p.Prime).pos k)] at this
  rwa [ite_eq_right hb] at this

example := choose_eq_zero_of_pow_dvd (F := ZMod 2) 2 (k := 1) (n := 2) (b := 1) dvd_rfl
  (by decide)

/-- In characteristic `p`, a nonzero coefficient of `f^{p^k}` sits at a multiple of `p^k`:
`f^{p^{k+1}} = Frob(expand_p(f^{p^k}))`. -/
theorem pow_dvd_of_coeff_pow_char_pow_ne_zero (f : F[X]) (k : ℕ) {n : ℕ}
    (h : (f ^ p ^ k).coeff n ≠ 0) : p ^ k ∣ n := by
  induction k generalizing n with
  | zero => simp
  | succ k ih =>
    rw [pow_succ, pow_mul, ← map_frobenius_expand, coeff_map,
      coeff_expand (Fact.out : p.Prime).pos] at h
    split_ifs at h with hpn
    · obtain ⟨t, rfl⟩ := hpn
      rw [Nat.mul_div_cancel_left t (Fact.out : p.Prime).pos] at h
      have ht : (f ^ p ^ k).coeff t ≠ 0 := fun h0 => h (by rw [h0, map_zero])
      rw [pow_succ, mul_comm]
      exact Nat.mul_dvd_mul_left p (ih ht)
    · exact absurd (map_zero _) h

example := pow_dvd_of_coeff_pow_char_pow_ne_zero (F := ZMod 2) 2 X 0 (n := 1) (by simp)

/-- In characteristic `p`, `deg E^{(b)}(f^{p^k}) ≤ (m − 1) b` for `b ≥ 1`, `m = deg f`. -/
theorem natDegree_hasseDeriv_pow_char_pow_le (f : F[X]) (k : ℕ) {b : ℕ} (hb : 1 ≤ b) :
    (hasseDeriv b (f ^ p ^ k)).natDegree ≤ (f.natDegree - 1) * b := by
  refine natDegree_le_iff_coeff_eq_zero.mpr fun t ht => ?_
  rw [hasseDeriv_coeff]
  by_cases h0 : (f ^ p ^ k).coeff (t + b) = 0
  · rw [h0, mul_zero]
  have hdvd := pow_dvd_of_coeff_pow_char_pow_ne_zero p f k h0
  by_cases hkb : p ^ k ∣ b
  · have hle : t + b ≤ p ^ k * f.natDegree := (le_natDegree_of_ne_zero h0).trans natDegree_pow_le
    have h4 : p ^ k * f.natDegree ≤ f.natDegree * b := by
      rw [mul_comm]
      exact Nat.mul_le_mul_left _ (Nat.le_of_dvd hb hkb)
    rw [Nat.sub_one_mul] at ht
    omega
  · rw [choose_eq_zero_of_pow_dvd p hdvd hkb, zero_mul]

example := natDegree_hasseDeriv_pow_char_pow_le (F := ZMod 2) 2 X 1 (b := 1) le_rfl

/-- `deg E^{(b)}(T_ν(f)) ≤ (m − 1) b` for `b ≥ 1`. -/
theorem natDegree_hasseDeriv_tracePoly_le (ν : ℕ) (f : F[X]) {b : ℕ} (hb : 1 ≤ b) :
    (hasseDeriv b (tracePoly p ν f)).natDegree ≤ (f.natDegree - 1) * b := by
  rw [tracePoly, map_sum]
  exact natDegree_sum_le_of_forall_le _ _ fun k _ => natDegree_hasseDeriv_pow_char_pow_le p f k hb

example := natDegree_hasseDeriv_tracePoly_le (F := ZMod 2) 2 1 X (b := 1) le_rfl

end CharP

/-- `deg T_ν(f) = m p^{ν − 1}` for `ν ≥ 1` and `m = deg f ≥ 1`: the last term `f^{p^{ν−1}}`
has the largest degree. -/
theorem natDegree_tracePoly {F : Type*} [Field F] {p : ℕ} (hp : 2 ≤ p) {ν : ℕ} (hν : 1 ≤ ν)
    {f : F[X]} (hf : 1 ≤ f.natDegree) :
    (tracePoly p ν f).natDegree = f.natDegree * p ^ (ν - 1) := by
  obtain ⟨μ, rfl⟩ : ∃ μ, ν = μ + 1 := ⟨ν - 1, by omega⟩
  rw [tracePoly, sum_range_succ, Nat.add_sub_cancel, natDegree_add_eq_right_of_natDegree_lt,
    natDegree_pow, mul_comm]
  rw [natDegree_pow]
  refine lt_of_le_of_lt (natDegree_sum_le_of_forall_le _ _ (n := p ^ μ * f.natDegree - 1)
    fun k hk => ?_) (by have := Nat.one_le_pow μ p (by omega); have := Nat.mul_le_mul this hf; omega)
  rw [natDegree_pow]
  have h1 : p ^ k < p ^ μ := Nat.pow_lt_pow_right (by omega) (mem_range.mp hk)
  have h2 := Nat.mul_lt_mul_of_pos_right h1 (by omega : 0 < f.natDegree)
  omega

example := natDegree_tracePoly (F := ZMod 2) le_rfl (ν := 1) le_rfl (f := X) (by simp)

/-- On a finite field `F` over `𝔽_p` of dimension `ν`, `T_ν(f)(x) = Tr_{F/𝔽_p}(f(x))`. -/
theorem eval_tracePoly {F : Type*} [Field F] [Finite F] {p : ℕ} [Fact p.Prime]
    [Algebra (ZMod p) F] (f : F[X]) (x : F) :
    (tracePoly p (Module.finrank (ZMod p) F) f).eval x =
      algebraMap (ZMod p) F (Algebra.trace (ZMod p) F (f.eval x)) := by
  rw [FiniteField.algebraMap_trace_eq_sum_pow, tracePoly, eval_finsetSum, Nat.card_zmod]
  simp

example := eval_tracePoly (F := ZMod 2) (p := 2) X 1

/-- If `deg E^{(b)} g ≤ d b` for every `b ≥ 1`, then on `{g = c}` each `E^{(b)}(g^i)` agrees
with a polynomial of degree `≤ d b`: by Leibniz, `E^{(b)}(g · g^i)` is a sum of
`E^{(a)} g · E^{(b − a)}(g^i)`, and `E^{(0)} g = g` takes the value `c`. -/
theorem exists_hasseDeriv_pow_eq_on {F : Type*} [Field F] {g : F[X]} {d : ℕ}
    (hg : ∀ b, 1 ≤ b → (hasseDeriv b g).natDegree ≤ d * b) (c : F) (i : ℕ) :
    ∀ b, ∃ V : F[X], V.natDegree ≤ d * b ∧
      ∀ x, g.eval x = c → (hasseDeriv b (g ^ i)).eval x = V.eval x := by
  induction i with
  | zero =>
    refine fun b => ⟨hasseDeriv b 1, ?_, fun x _ => by rw [pow_zero]⟩
    exact (natDegree_hasseDeriv_le _ _).trans (by simp)
  | succ i ih =>
    choose V hVd hV using ih
    intro b
    refine ⟨∑ ab ∈ antidiagonal b, (if ab.1 = 0 then C c else hasseDeriv ab.1 g) * V ab.2,
      natDegree_sum_le_of_forall_le _ _ fun ab hab => natDegree_mul_le.trans ?_, fun x hx => ?_⟩
    · rw [mem_antidiagonal] at hab
      have h1 : (if ab.1 = 0 then C c else hasseDeriv ab.1 g).natDegree ≤ d * ab.1 := by
        split_ifs with h
        · simp
        · exact hg _ (Nat.pos_of_ne_zero h)
      have := hVd ab.2
      rw [← hab, mul_add]
      omega
    · rw [pow_succ', hasseDeriv_mul, eval_finsetSum, eval_finsetSum]
      refine sum_congr rfl fun ab _ => ?_
      rw [eval_mul, eval_mul, hV ab.2 x hx]
      split_ifs with h
      · rw [h, hasseDeriv_zero, LinearMap.id_apply, eval_C, hx]
      · rfl

example := exists_hasseDeriv_pow_eq_on (F := ZMod 2) (g := X) (d := 0)
  (fun b hb => (natDegree_hasseDeriv_le _ _).trans (by rw [natDegree_X]; omega)) 0 1 1

end Transformer.Weil
