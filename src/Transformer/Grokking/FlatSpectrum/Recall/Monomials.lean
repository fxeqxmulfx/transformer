module

public import Mathlib.Algebra.MvPolynomial.Degrees
public import Mathlib.Algebra.Order.Antidiag.Pi
public import Mathlib.Analysis.Complex.ExponentialBounds
public import Mathlib.Data.Nat.Choose.Sum
public import Mathlib.Data.Sym.Card

/-!
# The monomials of bounded total degree, counted

Source: `review/flat_spectrum.md`, *Polynomial readouts and two layers* (added), item PR3.

A monomial of total degree at most `N` in `D` variables is a tuple of `D + 1` naturals with sum
`N`, the last one the slack (`monomialExponents`). There are `C(D + N, D)` of them, stars and
bars (`card_monomialExponents`), and a polynomial of total degree at most `N` is a combination
of them (`eval_eq_sum_monomialExponents`).

Three inequalities turn the count into a linear bound. One term of the binomial expansion of
`(t + 1)^{D + N}` is `C(D + N, D) t^N` (`choose_mul_pow_le`). `(n + 1)^n ≤ 3 n^n`, as
`(1 + 1/n)^n ≤ e` (`succ_pow_le_three_mul_pow`). And `128^m ≤ 9^D 81^m` forces `m ≤ 5 D`, as
`2^35 > 3^22` (`le_five_mul_of_pow_le`). So `2^m ≤ C(D + 2m, D)` forces `m ≤ 5 D`
(`le_five_mul_of_le_choose`), and `2^m ≤ C(D + Δ m, D)` forces `4^m ≤ 3^m (2Δ + 1)^{2D}`
(`four_pow_le_of_le_choose`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open MvPolynomial

variable {D : ℕ}

/-- The exponents of the monomials of total degree at most `N` in `D` variables: the tuples of
`D + 1` naturals with sum `N`, the last one the slack `N − ∑_i e_i`. -/
abbrev monomialExponents (D N : ℕ) : Finset (Fin (D + 1) → ℕ) :=
  (Finset.univ : Finset (Fin (D + 1))).piAntidiag N

/-- **Stars and bars** (`review/flat_spectrum.md`, PR3): there are `C(D + N, D)` monomials of
total degree at most `N` in `D` variables. -/
lemma card_monomialExponents (D N : ℕ) :
    Fintype.card ↥(monomialExponents D N) = (D + N).choose D := by
  rw [Fintype.card_coe, monomialExponents, ← Finset.map_sym_eq_piAntidiag, Finset.card_map,
    Finset.sym_univ, Finset.card_univ, Sym.card_sym_eq_choose, Fintype.card_fin,
    show D + 1 + N - 1 = D + N by omega, Nat.choose_symm_add]

/-- **A polynomial of total degree at most `N` is a combination of the monomials of total
degree at most `N`** (`review/flat_spectrum.md`, PR3). -/
lemma eval_eq_sum_monomialExponents {N : ℕ} {p : MvPolynomial (Fin D) ℝ}
    (hp : p.totalDegree ≤ N) (x : Fin D → ℝ) :
    eval x p = ∑ e : ↥(monomialExponents D N),
      p.coeff (Finsupp.equivFunOnFinite.symm fun i => e.1 i.castSucc) *
        ∏ i, x i ^ e.1 i.castSucc := by
  classical
  let ρ : (Fin (D + 1) → ℕ) → (Fin D →₀ ℕ) := fun e =>
    Finsupp.equivFunOnFinite.symm fun i => e i.castSucc
  have hρ : Set.InjOn ρ (monomialExponents D N : Set (Fin (D + 1) → ℕ)) := by
    intro e he e' he' h
    have hc (i : Fin D) : e i.castSucc = e' i.castSucc := DFunLike.congr_fun h i
    have hs := (Finset.mem_piAntidiag.1 (Finset.mem_coe.1 he)).1
    have hs' := (Finset.mem_piAntidiag.1 (Finset.mem_coe.1 he')).1
    rw [Fin.sum_univ_castSucc] at hs hs'
    have hsum : ∑ i : Fin D, e i.castSucc = ∑ i : Fin D, e' i.castSucc :=
      Finset.sum_congr rfl fun i _ => hc i
    funext j
    refine Fin.lastCases ?_ (fun i => hc i) j
    omega
  calc eval x p = ∑ d ∈ p.support, p.coeff d * ∏ i, x i ^ d i := eval_eq' x p
    _ = ∑ d ∈ (monomialExponents D N).image ρ, p.coeff d * ∏ i, x i ^ d i := by
        refine Finset.sum_subset (fun d hd => ?_) fun d _ hd => ?_
        · have hdeg : ∑ i, d i ≤ N := by
            rw [← Finsupp.sum_fintype d (fun _ e => e) fun _ => rfl]
            exact (le_totalDegree hd).trans hp
          refine Finset.mem_image.2
            ⟨Fin.snoc (α := fun _ => ℕ) (fun i => d i) (N - ∑ i, d i), ?_, ?_⟩
          · refine Finset.mem_piAntidiag.2 ⟨?_, fun _ _ => Finset.mem_univ _⟩
            rw [Fin.sum_univ_castSucc]
            simp only [Fin.snoc_castSucc, Fin.snoc_last]
            omega
          · ext i
            simp [ρ]
        · rw [notMem_support_iff.1 hd, zero_mul]
    _ = ∑ e ∈ monomialExponents D N, p.coeff (ρ e) * ∏ i, x i ^ (ρ e) i := Finset.sum_image hρ
    _ = ∑ e : ↥(monomialExponents D N), p.coeff (ρ e.1) * ∏ i, x i ^ (ρ e.1) i :=
        (Finset.sum_coe_sort _ _).symm

/-- `1 + x₀ x₁` has degree two: it is a combination of the `C(4, 2) = 6` monomials of degree at
most two. -/
example (x : Fin 2 → ℝ) := eval_eq_sum_monomialExponents
  (p := (1 : MvPolynomial (Fin 2) ℝ) + X 0 * X 1) (N := 2)
  ((totalDegree_add _ _).trans (max_le (by simp) ((totalDegree_mul _ _).trans (by simp)))) x

/-- **One term of the binomial expansion**: `C(D + N, D) t^N ≤ (t + 1)^{D + N}`. -/
lemma choose_mul_pow_le (D N t : ℕ) : (D + N).choose D * t ^ N ≤ (t + 1) ^ (D + N) := by
  have h := add_pow t 1 (D + N)
  simp only [Nat.cast_id, one_pow, mul_one] at h
  rw [h, Nat.choose_symm_add, mul_comm]
  exact Finset.single_le_sum (f := fun k => t ^ k * (D + N).choose k)
    (fun _ _ => Nat.zero_le _) (Finset.mem_range.2 (by omega))

/-- **`(n + 1)^n ≤ 3 n^n`**, as `(1 + 1/n)^n ≤ e < 3`. -/
lemma succ_pow_le_three_mul_pow (n : ℕ) : (n + 1) ^ n ≤ 3 * n ^ n := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · norm_num
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have h1 : ((n : ℝ) + 1) / n ≤ Real.exp (1 / n) := by
    have h := Real.add_one_le_exp (1 / (n : ℝ))
    rwa [show (1 : ℝ) / n + 1 = (n + 1) / n by rw [add_div, div_self hn'.ne', add_comm]] at h
  have h2 : (((n : ℝ) + 1) / n) ^ n ≤ Real.exp 1 :=
    (pow_le_pow_left₀ (by positivity) h1 n).trans_eq
      (by rw [← Real.exp_nat_mul, mul_one_div_cancel hn'.ne'])
  rw [div_pow, div_le_iff₀ (by positivity)] at h2
  exact_mod_cast h2.trans (mul_le_mul_of_nonneg_right Real.exp_one_lt_three.le (by positivity))

/-- **`128^m ≤ 9^D 81^m` forces `m ≤ 5 D`**, as `128^5 = 2^35 > 3^22 = 9 · 81^5`. -/
lemma le_five_mul_of_pow_le {m D : ℕ} (h : 128 ^ m ≤ 9 ^ D * 81 ^ m) : m ≤ 5 * D := by
  by_contra hlt
  obtain ⟨k, rfl⟩ : ∃ k, m = 5 * D + (k + 1) := ⟨m - 5 * D - 1, by omega⟩
  have h1 : 9 ^ D * 81 ^ (5 * D) ≤ 128 ^ (5 * D) := by
    rw [pow_mul, pow_mul, ← mul_pow]
    exact Nat.pow_le_pow_left (by norm_num) D
  have h2 : (81 : ℕ) ^ (k + 1) < 128 ^ (k + 1) :=
    Nat.pow_lt_pow_left (by norm_num) (Nat.succ_ne_zero k)
  have h3 : 9 ^ D * 81 ^ (5 * D + (k + 1)) < 128 ^ (5 * D + (k + 1)) := by
    rw [pow_add 81, pow_add 128, ← mul_assoc]
    exact Nat.mul_lt_mul_of_le_of_lt h1 h2 (by positivity)
  omega

/-- **The count at degree two is linear**: `2^m ≤ C(D + 2m, D)` forces `m ≤ 5 D`, by the
binomial term at `t = 8`. -/
lemma le_five_mul_of_le_choose {m D : ℕ} (h : 2 ^ m ≤ (D + 2 * m).choose D) : m ≤ 5 * D := by
  refine le_five_mul_of_pow_le ?_
  calc 128 ^ m = 2 ^ m * 8 ^ (2 * m) := by norm_num [pow_mul, ← mul_pow]
    _ ≤ (D + 2 * m).choose D * 8 ^ (2 * m) := Nat.mul_le_mul_right _ h
    _ ≤ (8 + 1) ^ (D + 2 * m) := choose_mul_pow_le D (2 * m) 8
    _ = 9 ^ D * 81 ^ m := by norm_num [pow_add, pow_mul]

/-- **The count at degree `Δ` is linear**: `2^m ≤ C(D + Δ m, D)` forces
`4^m ≤ 3^m (2Δ + 1)^{2D}`, by the binomial term at `t = 2Δ` and `(t + 1)^t ≤ 3 t^t`. -/
lemma four_pow_le_of_le_choose {m D Δ : ℕ} (h : 2 ^ m ≤ (D + Δ * m).choose D) :
    4 ^ m ≤ 3 ^ m * (2 * Δ + 1) ^ (2 * D) := by
  have hpos : 0 < (2 * Δ) ^ (2 * Δ * m) := by
    rcases Nat.eq_zero_or_pos Δ with rfl | hΔ
    · simp
    · exact Nat.pow_pos (by omega)
  have h1 : (2 ^ m * (2 * Δ) ^ (Δ * m)) ^ 2 ≤ ((2 * Δ + 1) ^ (D + Δ * m)) ^ 2 :=
    Nat.pow_le_pow_left
      ((Nat.mul_le_mul_right _ h).trans (choose_mul_pow_le D (Δ * m) (2 * Δ))) 2
  have h2 : (2 * Δ + 1) ^ (2 * Δ * m) ≤ 3 ^ m * (2 * Δ) ^ (2 * Δ * m) := by
    rw [pow_mul (2 * Δ + 1) (2 * Δ) m, pow_mul (2 * Δ) (2 * Δ) m, ← mul_pow]
    exact Nat.pow_le_pow_left (succ_pow_le_three_mul_pow (2 * Δ)) m
  refine Nat.le_of_mul_le_mul_right (c := (2 * Δ) ^ (2 * Δ * m)) ?_ hpos
  calc 4 ^ m * (2 * Δ) ^ (2 * Δ * m) = (2 ^ m * (2 * Δ) ^ (Δ * m)) ^ 2 := by
        rw [show (4 : ℕ) = 2 ^ 2 from rfl, ← pow_mul]
        ring
    _ ≤ ((2 * Δ + 1) ^ (D + Δ * m)) ^ 2 := h1
    _ = (2 * Δ + 1) ^ (2 * D) * (2 * Δ + 1) ^ (2 * Δ * m) := by ring
    _ ≤ (2 * Δ + 1) ^ (2 * D) * (3 ^ m * (2 * Δ) ^ (2 * Δ * m)) := Nat.mul_le_mul_left _ h2
    _ = 3 ^ m * (2 * Δ + 1) ^ (2 * D) * (2 * Δ) ^ (2 * Δ * m) := by ring

/-- `m = 5`, `D = 1`: `128^5 ≤ 9 · 81^5` fails, and `128^4 ≤ 9 · 81^4` holds. -/
example : (4 : ℕ) ≤ 5 * 1 := le_five_mul_of_pow_le (by norm_num)

/-- One number and one binding: `2 ≤ C(1 + 2, 1) = 3` and `2 ≤ C(1 + 1, 1) = 2`. -/
example : (1 : ℕ) ≤ 5 * 1 := le_five_mul_of_le_choose (by decide)

example : 4 ^ 1 ≤ 3 ^ 1 * (2 * 1 + 1) ^ (2 * 1) :=
  four_pow_le_of_le_choose (m := 1) (D := 1) (Δ := 1) (by decide)

example : (3 + 1) ^ 3 ≤ 3 * 3 ^ 3 := succ_pow_le_three_mul_pow 3

example : (2 + 3).choose 2 * 8 ^ 3 ≤ (8 + 1) ^ (2 + 3) := choose_mul_pow_le 2 3 8

example : Fintype.card ↥(monomialExponents 2 2) = 6 := by
  rw [card_monomialExponents]
  decide

end Transformer.Grokking.FlatSpectrum.Recall
