module

public import Transformer.Grokking.FlatSpectrum.Intermediate
public import Transformer.Grokking.FlatSpectrum.MulCharKernel
public import Mathlib.Tactic.NormNum.Prime

/-!
# The exact numbers among the note's numerical checks

Source: `review/flat_spectrum.md`. Theorem B, numerical check: "The maximum over r is 0.044 at
p = 97 (bound 0.053) … and (a+b)² with v = a+b gives ½." The bound is Theorem B's formula with
`|S_r| ≤ 2 √p`; at `p = 97` it is `(84681 + 1156 √97)/1815937`, between `0.0528` and `0.053`.
For `f = (a + b)²` the fiber through `v = a + b` is `{a + b = ±v}`, so
`E[ψ_r(a + b) | f] = (ψ_r(v) + ψ_r(−v))/2`, and `‖E[ψ_r(a + b) | f]‖² = ½` for odd `p` and
`r ≠ 0`. Readout-free version: "a fraction ρ of the table pins down at most r ≤ (1−√(1−ρ))p.
With half the data that is 28.4 at p = 97 … At 80% data it rises to 0.55p." As
`r (2p − r) = p² − (p − r)²`, `r (2p − r) ≤ ρ p²` iff `r ≤ (1 − √(1 − ρ)) p` for `r ≤ p`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

/-- The bound of Theorem B at `p = 97`, `(2√97 + 1)²/97² + (96 + 2√97)²/(97² · 193)`, is
between `0.0528` and `0.053`: the source's "bound 0.053". -/
theorem cube_bound_97 :
    0.0528 < (2 * √97 + 1) ^ 2 / 97 ^ 2 + (97 - 1 + 2 * √97) ^ 2 / (97 ^ 2 * (2 * 97 - 1)) ∧
      (2 * √97 + 1) ^ 2 / 97 ^ 2 + (97 - 1 + 2 * √97) ^ 2 / (97 ^ 2 * (2 * 97 - 1)) < 0.053 := by
  have hs : √(97 : ℝ) ^ 2 = 97 := Real.sq_sqrt (by norm_num)
  have hlo : (9.8488 : ℝ) < √97 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hhi : √(97 : ℝ) < 9.8489 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have h1 : (2 * √97 + 1) ^ 2 = 389 + 4 * √(97 : ℝ) := by linear_combination 4 * hs
  have h2 : ((97 : ℝ) - 1 + 2 * √97) ^ 2 = 9604 + 384 * √97 := by linear_combination 4 * hs
  have key : (389 + 4 * √(97 : ℝ)) / 97 ^ 2 + (9604 + 384 * √97) / (97 ^ 2 * (2 * 97 - 1)) =
      (84681 + 1156 * √97) / 1815937 := by
    field_simp
    ring
  rw [h1, h2, key]
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    linarith
  · rw [div_lt_iff₀ (by norm_num)]
    linarith

/-- Source: `review/flat_spectrum.md`, Theorem B, numerical check: at `p = 97` and `r ≠ 0`,
`‖E[ψ_r(a b) | a³ + a b]‖² < 0.053`. -/
theorem meanSq_condMean_cube_add_mul_lt_97 [Fact (Nat.Prime 97)] {r : ZMod 97} (hr : r ≠ 0) :
    meanSq (condMean (fun ab : ZMod 97 × ZMod 97 => ab.1 ^ 3 + ab.1 * ab.2)
      fun ab => ZMod.stdAddChar (r * (ab.1 * ab.2))) < 0.053 := by
  have h := meanSq_condMean_cube_add_mul_le_sqrt hr
  simp only [Nat.cast_ofNat] at h
  exact h.trans_lt cube_bound_97.2

example := @meanSq_condMean_cube_add_mul_lt_97 ⟨by norm_num⟩ 1 (by decide)

variable {p : ℕ} [Fact p.Prime]

/-- `∑_{(a, b)} H(a + b) = p ∑_v H(v)`: for each `b`, `a ↦ a + b` is a bijection. -/
lemma sum_prod_add {M : Type*} [AddCommMonoid M] (H : ZMod p → M) :
    ∑ ab : ZMod p × ZMod p, H (ab.1 + ab.2) = p • ∑ v, H v := by
  have h (b : ZMod p) : ∑ a, H (a + b) = ∑ v, H v := Equiv.sum_comp (Equiv.addRight b) H
  simp only [Fintype.sum_prod_type_right, h, Finset.sum_const, Finset.card_univ, ZMod.card]

example := @sum_prod_add 5 ⟨Nat.prime_five⟩ ℕ _ fun _ => 1

/-- For `f = (a + b)²` and odd `p`, the fiber through `v = a + b` is `{a + b = ±v}`, so
`E[ψ_r(a + b) | f] = (ψ_r(v) + ψ_r(−v))/2`; at `v = 0` both sides are `1`. -/
lemma condMean_sq_add (hp : p ≠ 2) (r : ZMod p) (x : ZMod p × ZMod p) :
    condMean (fun ab : ZMod p × ZMod p => (ab.1 + ab.2) ^ 2)
      (fun ab => ZMod.stdAddChar (r * (ab.1 + ab.2))) x =
      (ZMod.stdAddChar (r * (x.1 + x.2)) + ZMod.stdAddChar (-(r * (x.1 + x.2)))) / 2 := by
  have hp0 : (p : ℂ) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero
  have hsum : ∑ y : ZMod p × ZMod p, (if (y.1 + y.2) ^ 2 = (x.1 + x.2) ^ 2 then
      ZMod.stdAddChar (r * (y.1 + y.2)) else 0) =
      p • ∑ u : ZMod p, if u ^ 2 = (x.1 + x.2) ^ 2 then ZMod.stdAddChar (r * u) else 0 :=
    sum_prod_add fun u => if u ^ 2 = (x.1 + x.2) ^ 2 then ZMod.stdAddChar (r * u) else 0
  have hcard : ∑ y : ZMod p × ZMod p, (if (y.1 + y.2) ^ 2 = (x.1 + x.2) ^ 2 then (1 : ℂ)
      else 0) = p • ∑ u : ZMod p, if u ^ 2 = (x.1 + x.2) ^ 2 then (1 : ℂ) else 0 :=
    sum_prod_add fun u => if u ^ 2 = (x.1 + x.2) ^ 2 then (1 : ℂ) else 0
  simp only [condMean, Finset.sum_filter, Finset.natCast_card_filter]
  rw [hsum, hcard, nsmul_eq_mul, nsmul_eq_mul]
  by_cases hv : x.1 + x.2 = 0
  · have h0 (u : ZMod p) : u ^ 2 = (x.1 + x.2) ^ 2 ↔ u = 0 := by
      rw [hv, zero_pow two_ne_zero, pow_eq_zero_iff two_ne_zero]
    simp only [h0, Finset.sum_ite_eq', Finset.mem_univ, ite_true, mul_one]
    rw [hv, mul_zero, neg_zero, AddChar.map_zero_eq_one]
    norm_num
  · rw [sum_ite_sq_eq hp hv, sum_ite_sq_eq hp hv, mul_neg]
    field_simp
    ring

example := @condMean_sq_add 5 ⟨Nat.prime_five⟩ (by decide) 1 (1, 2)

/-- `|ψ(t) + ψ(−t)|² = 2 + ψ(2t) + ψ(−2t)`, as complex numbers. -/
lemma norm_stdAddChar_add_neg_sq (t : ZMod p) :
    ((‖ZMod.stdAddChar t + ZMod.stdAddChar (-t)‖ ^ 2 : ℝ) : ℂ) =
      2 + ZMod.stdAddChar (2 * t) + ZMod.stdAddChar (-(2 * t)) := by
  have hmul (a b : ZMod p) : ZMod.stdAddChar a * ZMod.stdAddChar b = ZMod.stdAddChar (a + b) :=
    (AddChar.map_add_eq_mul _ a b).symm
  rw [Complex.sq_norm, ← Complex.mul_conj, map_add, ← AddChar.map_neg_eq_conj,
    ← AddChar.map_neg_eq_conj, neg_neg]
  calc (ZMod.stdAddChar t + ZMod.stdAddChar (-t)) * (ZMod.stdAddChar (-t) + ZMod.stdAddChar t)
      = ZMod.stdAddChar t * ZMod.stdAddChar (-t) + ZMod.stdAddChar (-t) * ZMod.stdAddChar t +
          ZMod.stdAddChar t * ZMod.stdAddChar t +
          ZMod.stdAddChar (-t) * ZMod.stdAddChar (-t) := by ring
    _ = 2 + ZMod.stdAddChar (2 * t) + ZMod.stdAddChar (-(2 * t)) := by
        rw [hmul, hmul, hmul, hmul, add_neg_cancel, neg_add_cancel, AddChar.map_zero_eq_one,
          ← two_mul, ← neg_add, ← two_mul]
        ring

example := @norm_stdAddChar_add_neg_sq 5 ⟨Nat.prime_five⟩ 1

/-- Source: `review/flat_spectrum.md`, Theorem B, numerical check: "(a+b)² with v = a+b gives ½".
For odd prime `p` and `r ≠ 0`, `‖E[ψ_r(a + b) | (a + b)²]‖² = ½` exactly. -/
theorem meanSq_condMean_sq_add (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => (ab.1 + ab.2) ^ 2)
      fun ab => ZMod.stdAddChar (r * (ab.1 + ab.2))) = 1 / 2 := by
  have h2r : 2 * r ≠ 0 := mul_ne_zero (Ring.two_ne_zero (by rwa [ZMod.ringChar_zmod_n])) hr
  have hp0 : (p : ℝ) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero
  have hv : ∑ v : ZMod p, ‖(ZMod.stdAddChar (r * v) + ZMod.stdAddChar (-(r * v))) / 2‖ ^ 2 =
      p / 2 := by
    have hpt (v : ZMod p) :
        ((‖(ZMod.stdAddChar (r * v) + ZMod.stdAddChar (-(r * v))) / 2‖ ^ 2 : ℝ) : ℂ) =
          (2 + ZMod.stdAddChar (v * (2 * r)) + ZMod.stdAddChar (v * -(2 * r))) / 4 := by
      rw [norm_div, div_pow, Complex.ofReal_div, norm_stdAddChar_add_neg_sq,
        show 2 * (r * v) = v * (2 * r) by ring, show -(v * (2 * r)) = v * -(2 * r) by ring]
      norm_num
    apply Complex.ofReal_injective
    rw [Complex.ofReal_sum, Finset.sum_congr rfl fun v _ => hpt v, ← Finset.sum_div,
      Finset.sum_add_distrib, Finset.sum_add_distrib, sum_stdAddChar_mul, sum_stdAddChar_mul,
      ite_eq_right h2r, ite_eq_right (neg_ne_zero.mpr h2r)]
    simp only [Finset.sum_const, Finset.card_univ, ZMod.card, nsmul_eq_mul, add_zero,
      Complex.ofReal_div, Complex.ofReal_natCast, Complex.ofReal_ofNat]
    ring
  simp only [meanSq, condMean_sq_add hp]
  rw [sum_prod_add fun v => ‖(ZMod.stdAddChar (r * v) + ZMod.stdAddChar (-(r * v))) / 2‖ ^ 2,
    hv, Fintype.card_prod, ZMod.card, nsmul_eq_mul]
  push_cast
  field_simp

example := @meanSq_condMean_sq_add 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide)

/-- The counting rule of the readout-free version, `review/flat_spectrum.md`: "a rank-r p×p
matrix has r(2p−r) parameters, so a fraction ρ of the table pins down at most
r ≤ (1−√(1−ρ))p". Its arithmetic: for `0 ≤ p`, `r ≤ p` and `ρ ≤ 1`, `r (2p − r) ≤ ρ p²` iff
`r ≤ (1 − √(1 − ρ)) p`. The parameter count itself is not formalized. -/
theorem mul_two_mul_sub_le_iff {p ρ r : ℝ} (hp : 0 ≤ p) (hρ : ρ ≤ 1) (hr : r ≤ p) :
    r * (2 * p - r) ≤ ρ * p ^ 2 ↔ r ≤ (1 - √(1 - ρ)) * p := by
  have hpr : 0 ≤ p - r := by linarith
  have hs : √(1 - ρ) ^ 2 = 1 - ρ := Real.sq_sqrt (by linarith)
  have hs0 : 0 ≤ √(1 - ρ) := Real.sqrt_nonneg _
  constructor
  · intro h
    have h2 : (√(1 - ρ) * p) ^ 2 ≤ (p - r) ^ 2 := by rw [mul_pow, hs]; nlinarith
    have h3 : √(1 - ρ) * p ≤ p - r := (pow_le_pow_iff_left₀ (by positivity) hpr two_ne_zero).mp h2
    linarith
  · intro h
    have h2 : √(1 - ρ) * p ≤ p - r := by linarith
    have h3 : (√(1 - ρ) * p) ^ 2 ≤ (p - r) ^ 2 := pow_le_pow_left₀ (by positivity) h2 2
    rw [mul_pow, hs] at h3
    nlinarith

example := mul_two_mul_sub_le_iff (p := 97) (ρ := 1 / 2) (r := 28) (by norm_num) (by norm_num)
  (by norm_num)

/-- "With half the data that is 28.4 at p = 97": `(1 − √(1 − 1/2)) · 97` is between `28.4`
and `28.45`. -/
theorem counting_rule_half_97 :
    28.4 < (1 - √(1 - 1 / 2)) * 97 ∧ (1 - √(1 - 1 / 2)) * 97 < (28.45 : ℝ) := by
  have hlo : (0.7071 : ℝ) < √(1 - 1 / 2) := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hhi : √(1 - 1 / 2 : ℝ) < 0.70711 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  constructor <;> linarith

/-- "At 80% data it rises to 0.55p": `1 − √(1 − 0.8)` is between `0.5525` and `0.553`, the
torch check's "0.553p". -/
theorem counting_rule_eighty : 0.5525 < 1 - √(1 - 0.8) ∧ 1 - √(1 - 0.8) < (0.553 : ℝ) := by
  have hlo : (0.447 : ℝ) < √(1 - 0.8) := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hhi : √(1 - 0.8 : ℝ) < 0.4475 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  constructor <;> linarith

/-- The torch check, `review/flat_spectrum.md`: "At rank 49 the rule says 0.76". Rank `49` at
`p = 97` needs `ρ ≥ 49 (2 · 97 − 49)/97² = 7105/9409`, between `0.755` and `0.756`. -/
theorem counting_rule_rank_49 :
    (49 : ℝ) * (2 * 97 - 49) / 97 ^ 2 = 7105 / 9409 ∧ 0.755 < (7105 / 9409 : ℝ) ∧
      (7105 / 9409 : ℝ) < 0.756 := by
  norm_num

end Transformer.Grokking.FlatSpectrum
