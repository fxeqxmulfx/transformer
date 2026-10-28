module

public import Mathlib.Algebra.CharP.Algebra
public import Mathlib.Analysis.Complex.Norm
public import Mathlib.Analysis.Normed.Ring.Finite
public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
public import Mathlib.FieldTheory.Finiteness
public import Mathlib.NumberTheory.LegendreSymbol.AddCharacter
public import Mathlib.Tactic.NormNum.Prime
public import Transformer.Weil.TraceStepanov

/-!
# Exponential sums `∑_x ψ(Tr f(x))` by Stepanov's method

Let `F` have `q = p^ν ≥ 5` elements and `f ∈ F[X]` have degree `m ≥ 1` prime to `p`. With
`s = ⌊√q⌋`, the parameters `ℓ = s`, `K = p^{ν − 1}`, `J = ⌊s/p⌋ + m + 1` satisfy Stepanov's
condition, and his bound gives `#{x : T_ν(f)(x) = c} ≤ q/p + 4 m √q` for every `c`. As
`T_ν(f)(x) = Tr(f(x))`, every fiber `N_c` of `x ↦ Tr(f(x))` has at most `E = q/p + 4 m √q`
points. For a nontrivial additive character `ψ` of `𝔽_p`, `∑_c ψ(c) = 0`, so
`∑_x ψ(Tr f(x)) = ∑_c (N_c − E) ψ(c)` and `|∑_x ψ(Tr f(x))| ≤ ∑_c (E − N_c) = p E − q`, which is
`4 p m √q`. This is the bound `O(√q)` for the curve `y^p − y = f(x)`, with the constant `4 p m`
in place of Weil's `m − 1`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- Stepanov's dimension count for `ℓ = s`, `K = q`, `J = ⌊s/p⌋ + m + 1`, where `s² ≤ p q` and
`2 s < p q`: `ℓ (K + J + (m − 1)(ℓ − 1)) < p J K`. -/
theorem trace_params {p q s m : ℕ} (hp : 1 ≤ p) (hm : 1 ≤ m) (hs : 1 ≤ s)
    (hsq : s * s ≤ p * q) (h2 : 2 * s < p * q) :
    s * (q + (s / p + m + 1) + (m - 1) * (s - 1)) < p * (s / p + m + 1) * q := by
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  obtain ⟨s', rfl⟩ : ∃ s', s = s' + 1 := ⟨s - 1, by omega⟩
  obtain ⟨t, ht⟩ : ∃ t, (s' + 1) / p = t := ⟨_, rfl⟩
  have ha : p * t ≤ s' + 1 := by
    rw [← ht]
    exact Nat.mul_div_le _ _
  have hb : s' + 1 < p * t + p := by
    have := Nat.lt_mul_div_succ (s' + 1) (by omega : 0 < p)
    rwa [ht, mul_add, mul_one] at this
  have hi : t * (s' + 1) ≤ q := by
    refine Nat.le_of_mul_le_mul_left ?_ (by omega : 0 < p)
    calc p * (t * (s' + 1)) = p * t * (s' + 1) := by ring
      _ ≤ (s' + 1) * (s' + 1) := Nat.mul_le_mul_right _ ha
      _ ≤ p * q := hsq
  have hii : q * (s' + 2) ≤ q * (p * t + p) := Nat.mul_le_mul_left q (by omega)
  have hiii : m' * ((s' + 1) * (s' + 1)) ≤ m' * (p * q) := Nat.mul_le_mul_left m' hsq
  rw [ht, Nat.add_sub_cancel, Nat.add_sub_cancel]
  nlinarith [hi, hii, hiii, h2]

example := trace_params (p := 2) (q := 4) (s := 2) (m := 1) (by norm_num) le_rfl (by norm_num)
  (by norm_num) (by norm_num)

section Count

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F] {p : ℕ} [Fact p.Prime] [CharP F p]

/-- **Stepanov's count for `T_ν(f) = c`**: for `|F| = p^ν ≥ 5` and `f` of degree `m ≥ 1` prime
to `p`, `#{x : T_ν(f)(x) = c} ≤ |F|/p + 4 m √|F|`. -/
theorem card_tracePoly_eq_le {ν : ℕ} (hcard : Fintype.card F = p ^ ν) (h5 : 5 ≤ Fintype.card F)
    {f : F[X]} (hm : 1 ≤ f.natDegree) (hpm : ¬p ∣ f.natDegree) (c : F) :
    (#{x | (tracePoly p ν f).eval x = c} : ℝ) ≤
      Fintype.card F / p + 4 * f.natDegree * √(Fintype.card F) := by
  have hp := (Fact.out : p.Prime)
  obtain ⟨Q, hQdef⟩ : ∃ Q, Fintype.card F = Q := ⟨_, rfl⟩
  obtain ⟨q, hqdef⟩ : ∃ q, p ^ (ν - 1) = q := ⟨_, rfl⟩
  obtain ⟨m, hmdef⟩ : ∃ m, f.natDegree = m := ⟨_, rfl⟩
  obtain ⟨s, hsdef⟩ : ∃ s, Q.sqrt = s := ⟨_, rfl⟩
  have hm' : 1 ≤ m := hmdef ▸ hm
  have hν : 1 ≤ ν := Nat.pos_of_ne_zero fun h0 => by
    rw [h0, pow_zero] at hcard
    omega
  have hQ : Q = p * q := by
    rw [← hQdef, hcard, ← hqdef, ← pow_succ']
    congr 1
    omega
  rw [hQdef] at h5
  have hs1 : 1 ≤ s := by
    rw [← hsdef]
    exact Nat.sqrt_pos.mpr (by omega)
  have hss : s * s ≤ Q := by
    rw [← hsdef]
    exact Nat.sqrt_le Q
  have h2 : 2 * s < Q := by
    rcases Nat.lt_or_ge s 3 with h | h
    · omega
    · nlinarith
  have hst := mul_card_tracePoly_eq_le hcard hm hpm c (ℓ := s) (J := s / p + m + 1) (K := q)
    (by rw [hQdef, ← hsdef]; exact Nat.sqrt_le_self Q) hqdef.ge
    (by rw [hmdef]; exact trace_params hp.one_lt.le hm' hs1 (hQ ▸ hss) (hQ ▸ h2))
  obtain ⟨N, hNdef⟩ : ∃ N, #{x | (tracePoly p ν f).eval x = c} = N := ⟨_, rfl⟩
  rw [hNdef, hmdef, hqdef, hQdef, Nat.add_sub_cancel] at hst
  have h1 : Q * (s / p) ≤ s * q := by
    rw [hQ]
    calc p * q * (s / p) = q * (p * (s / p)) := by ring
      _ ≤ q * s := Nat.mul_le_mul_left q (Nat.mul_div_le s p)
      _ = s * q := mul_comm _ _
  have e1 : (p - 1) * (m * q) + m * q = m * Q := by
    rw [← add_one_mul, Nat.sub_add_cancel hp.one_lt.le, hQ]
    ring
  have e2 : q - 1 ≤ m * q := (Nat.sub_le q 1).trans (Nat.le_mul_of_pos_left q (by omega))
  have hN : s * N ≤ s * q + 2 * m * Q := by nlinarith [hst, h1, e1, e2]
  have hsR : (1 : ℝ) ≤ s := by exact_mod_cast hs1
  have hsqrt : √(Q : ℝ) ≤ s + 1 := by
    rw [← hsdef]
    exact Real.real_sqrt_le_nat_sqrt_succ
  have hQs : (Q : ℝ) ≤ 2 * s * √Q := by
    have h0 := Real.sq_sqrt (Nat.cast_nonneg Q : (0 : ℝ) ≤ Q)
    nlinarith [mul_nonneg (Real.sqrt_nonneg (Q : ℝ)) (sub_nonneg.mpr hsqrt),
      mul_nonneg (Real.sqrt_nonneg (Q : ℝ)) (sub_nonneg.mpr hsR)]
  have hNR : (s : ℝ) * N ≤ s * q + 2 * m * Q := by exact_mod_cast hN
  have hqR : (Q : ℝ) / p = q := by
    rw [hQ, Nat.cast_mul, mul_div_cancel_left₀ _ (by exact_mod_cast hp.ne_zero)]
  rw [hNdef, hQdef, hmdef, hqR]
  refine le_of_mul_le_mul_left ?_ (by linarith : (0 : ℝ) < s)
  nlinarith [mul_le_mul_of_nonneg_left hQs (Nat.cast_nonneg (α := ℝ) m)]

end Count

/-- If every fiber of `T : α → G` has at most `E` points, then for a nontrivial additive
character `ψ` of the finite group `G`, `‖∑_x ψ(T x)‖ ≤ |G| E − |α|`: as `∑_c ψ(c) = 0`,
`∑_x ψ(T x) = ∑_c (N_c − E) ψ(c)`, and `∑_c |N_c − E| = ∑_c (E − N_c)`. -/
theorem norm_sum_addChar_le_of_card_fiber_le {α G : Type*} [Fintype α] [AddCommGroup G]
    [Fintype G] [DecidableEq G] {ψ : AddChar G ℂ} (hψ : ψ ≠ 1) (T : α → G) {E : ℝ}
    (hE : ∀ c, (#{x | T x = c} : ℝ) ≤ E) :
    ‖∑ x, ψ (T x)‖ ≤ Fintype.card G * E - Fintype.card α := by
  have h1 : ∑ x, ψ (T x) = ∑ c, ((#{x | T x = c} : ℕ) : ℂ) * ψ c := by
    rw [← Finset.sum_fiberwise' univ T ψ]
    refine sum_congr rfl fun c _ => ?_
    rw [sum_const, nsmul_eq_mul]
  have h2 : ∑ c, (E : ℂ) * ψ c = 0 := by
    rw [← mul_sum, AddChar.sum_eq_zero_of_ne_one hψ, mul_zero]
  have h3 : ∑ x, ψ (T x) = ∑ c, ((((#{x | T x = c} : ℕ) : ℝ) - E : ℝ) : ℂ) * ψ c := by
    simp only [Complex.ofReal_sub, Complex.ofReal_natCast, sub_mul, sum_sub_distrib, h2,
      sub_zero, h1]
  have h4 : ∑ c, ((#{x | T x = c} : ℕ) : ℝ) = Fintype.card α := by
    rw [← Nat.cast_sum, ← card_eq_sum_card_fiberwise (f := T) (s := univ) (t := univ)
      (fun x _ => by simp), card_univ]
  rw [h3]
  calc ‖∑ c, ((((#{x | T x = c} : ℕ) : ℝ) - E : ℝ) : ℂ) * ψ c‖
      ≤ ∑ c, ‖((((#{x | T x = c} : ℕ) : ℝ) - E : ℝ) : ℂ) * ψ c‖ := norm_sum_le _ _
    _ = ∑ c, (E - ((#{x | T x = c} : ℕ) : ℝ)) := by
      refine sum_congr rfl fun c _ => ?_
      rw [norm_mul, AddChar.norm_apply, mul_one, Complex.norm_real, Real.norm_eq_abs,
        abs_of_nonpos (sub_nonpos.mpr (hE c)), neg_sub]
    _ = Fintype.card G * E - Fintype.card α := by
      rw [sum_sub_distrib, sum_const, card_univ, nsmul_eq_mul, h4]

/-- **The exponential sum `∑_x ψ(Tr f(x))`**: for a finite field `F` over `𝔽_p` with `|F| ≥ 5`,
`f ∈ F[X]` of degree `m ≥ 1` prime to `p` and `ψ` a nontrivial additive character of `𝔽_p`,
`‖∑_{x ∈ F} ψ(Tr_{F/𝔽_p}(f(x)))‖ ≤ 4 p m √|F|`. -/
theorem norm_sum_addChar_trace_le {F : Type*} [Field F] [Fintype F] {p : ℕ} [Fact p.Prime]
    [Algebra (ZMod p) F] {ψ : AddChar (ZMod p) ℂ} (hψ : ψ ≠ 1) {f : F[X]}
    (hm : 1 ≤ f.natDegree) (hpm : ¬p ∣ f.natDegree) (h5 : 5 ≤ Fintype.card F) :
    ‖∑ x, ψ (Algebra.trace (ZMod p) F (f.eval x))‖ ≤ 4 * p * f.natDegree * √(Fintype.card F) := by
  classical
  have : CharP F p := (Algebra.charP_iff (ZMod p) F p).mp inferInstance
  have hcard : Fintype.card F = p ^ Module.finrank (ZMod p) F := by
    rw [Module.card_eq_pow_finrank (K := ZMod p), ZMod.card]
  have hE (c : ZMod p) : (#{x | Algebra.trace (ZMod p) F (f.eval x) = c} : ℝ) ≤
      Fintype.card F / p + 4 * f.natDegree * √(Fintype.card F) := by
    have he : #{x | Algebra.trace (ZMod p) F (f.eval x) = c} =
        #{x | (tracePoly p (Module.finrank (ZMod p) F) f).eval x = algebraMap (ZMod p) F c} := by
      congr 1
      refine filter_congr fun x _ => ?_
      rw [eval_tracePoly, (algebraMap (ZMod p) F).injective.eq_iff]
    rw [he]
    exact card_tracePoly_eq_le hcard h5 hm hpm _
  refine (norm_sum_addChar_le_of_card_fiber_le hψ _ hE).trans_eq ?_
  have hp : (p : ℝ) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero
  rw [ZMod.card]
  field_simp
  ring

section Examples

local instance fact_prime_5 : Fact (Nat.Prime 5) := ⟨by norm_num⟩

example := card_tracePoly_eq_le (F := ZMod 5) (p := 5) (ν := 1) (by rw [ZMod.card]; norm_num)
  (by rw [ZMod.card]) (f := X) (by rw [natDegree_X]) (by rw [natDegree_X]; norm_num) 0

example := norm_sum_addChar_le_of_card_fiber_le (α := ZMod 5) (ψ := ZMod.stdAddChar)
  (AddChar.ne_one_iff.mpr ⟨1, fun h => one_ne_zero (ZMod.injective_stdAddChar
    (h.trans (AddChar.map_zero_eq_one _).symm))⟩) id (E := 1) (fun c => by simp [filter_eq'])

example := norm_sum_addChar_trace_le (F := ZMod 5) (p := 5) (ψ := ZMod.stdAddChar)
  (AddChar.ne_one_iff.mpr ⟨1, fun h => one_ne_zero (ZMod.injective_stdAddChar
    (h.trans (AddChar.map_zero_eq_one _).symm))⟩) (f := X) (by rw [natDegree_X])
  (by rw [natDegree_X]; norm_num) (by rw [ZMod.card])

end Examples

end Transformer.Weil
