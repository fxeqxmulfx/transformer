module

public import Transformer.Weil.HasseDeriv

/-!
# Stepanov's independence lemma

Let `q = |F|` be odd and `c = (q − 1)/2`. If `∑_{j < J} (r_j + s_j f^c) X^{q j} = 0` with
`deg r_j, deg s_j < K` and `m + 2K ≤ q + 1`, `m = deg f`, then with `A = ∑ r_j X^{qj}` and
`B = ∑ s_j X^{qj}`, `A = −f^c B` gives `f A² = f^q B² = f(X^q) B²`. Both sides are values at
`Y = X^q` of `Φ = f(X) R(X, Y)² − f(Y) S(X, Y)²`, `R = ∑ r_j Y^j`, `S = ∑ s_j Y^j`, whose
coefficients in `Y` have degree `< q` in `X`; such a `Φ` vanishes at `Y = X^q` only if it is `0`
(base-`q` digits). So every `r_j` and `s_j` is `0` as soon as `f(X) R² = f(Y) S²` forces
`S = 0`. It does for odd `m`, the degrees in `Y` being `2 deg_Y R` and `m + 2 deg_Y S`, and for
`f` with a root of odd multiplicity (`IndependenceRoot`).
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

section Digits

variable {R : Type*} [CommSemiring R]

/-- A polynomial `Φ` in `Y` over `R[X]` whose coefficients have degree `< n` is `0` if it
vanishes at `Y = X^n`: the coefficient of `X^{n k + i}`, `i < n`, in `Φ(X^n)` is the `i`-th
coefficient of the `k`-th coefficient of `Φ`. -/
theorem eq_zero_of_eval_X_pow_eq_zero {n : ℕ} {Φ : R[X][X]}
    (hdeg : ∀ k, (Φ.coeff k).natDegree < n) (h : Φ.eval (X ^ n) = 0) : Φ = 0 := by
  refine Polynomial.ext fun k => Polynomial.ext fun i => ?_
  rw [coeff_zero, coeff_zero]
  rcases lt_or_ge i n with hi | hi
  · have hc := congrArg (coeff · (n * k + i)) h
    simp only [eval_eq_sum_range, finsetSum_coeff, coeff_zero, ← pow_mul,
      coeff_mul_X_pow'] at hc
    rw [Finset.sum_eq_single k] at hc
    · rwa [ite_eq_left (by omega), show n * k + i - n * k = i by omega] at hc
    · intro t _ htk
      split_ifs with ht
      · apply coeff_eq_zero_of_natDegree_lt
        have := hdeg t
        rcases lt_or_gt_of_ne htk with htk | htk
        · have := Nat.mul_le_mul_left n (Nat.succ_le_of_lt htk)
          rw [Nat.mul_succ] at this
          omega
        · have := Nat.mul_le_mul_left n (Nat.succ_le_of_lt htk)
          rw [Nat.mul_succ] at this
          omega
      · rfl
    · intro hk
      rw [Finset.mem_range, not_lt] at hk
      rw [ite_eq_left (by omega), coeff_eq_zero_of_natDegree_lt (p := Φ) (n := k) (by omega),
        coeff_zero]
  · exact coeff_eq_zero_of_natDegree_lt ((hdeg k).trans_le hi)

example := eq_zero_of_eval_X_pow_eq_zero (R := ℤ) (n := 1) (Φ := 0) (by simp) (by simp)

/-- Coefficients of degree `≤ d` and `≤ e` give a product with coefficients of degree
`≤ d + e`. -/
theorem natDegree_coeff_mul_le {P Q : R[X][X]} {d e : ℕ} (hP : ∀ k, (P.coeff k).natDegree ≤ d)
    (hQ : ∀ k, (Q.coeff k).natDegree ≤ e) (k : ℕ) : ((P * Q).coeff k).natDegree ≤ d + e := by
  rw [coeff_mul]
  exact natDegree_sum_le_of_forall_le _ _ fun x _ =>
    natDegree_mul_le.trans (add_le_add (hP _) (hQ _))

example := natDegree_coeff_mul_le (P := (0 : ℤ[X][X])) (Q := 0) (d := 0) (e := 0)
  (by simp) (by simp) 0

/-- The coefficients of `∑_j C (r_j) Y^j` are the `r_j`. -/
theorem coeff_sum_C_mul_X_pow {J : ℕ} (r : Fin J → R[X]) (j : Fin J) :
    (∑ i, C (r i) * X ^ (i : ℕ)).coeff j = r j := by
  simp only [finsetSum_coeff, coeff_C_mul_X_pow, Fin.val_inj]
  simp

example := coeff_sum_C_mul_X_pow (R := ℤ) (J := 1) (fun _ => X) 0

end Digits

variable {F : Type*} [Field F] [Fintype F]

/-- The first half of Stepanov's independence lemma: if
`∑_{j < J} (r_j + s_j f^{(q−1)/2}) X^{q j} = 0` with `deg r_j, deg s_j < K`, `K ≥ 1` and
`m + 2K ≤ q + 1`, then `f(X) R(X, Y)² = f(Y) S(X, Y)²` for `R = ∑ r_j Y^j`, `S = ∑ s_j Y^j`. -/
theorem C_mul_sq_eq_map_C_mul_sq {f : F[X]} (hq : Odd (Fintype.card F)) {J K : ℕ} (hK0 : 0 < K)
    (hK : f.natDegree + 2 * K ≤ Fintype.card F + 1) {r s : Fin J → F[X]}
    (hr : ∀ j, (r j).natDegree < K) (hs : ∀ j, (s j).natDegree < K)
    (h : ∑ j, (r j + s j * f ^ (Fintype.card F / 2)) * X ^ (Fintype.card F * (j : ℕ)) = 0) :
    C f * (∑ j, C (r j) * X ^ (j : ℕ)) ^ 2 = f.map C * (∑ j, C (s j) * X ^ (j : ℕ)) ^ 2 := by
  set q := Fintype.card F
  set Rb : F[X][X] := ∑ j, C (r j) * X ^ (j : ℕ)
  set Sb : F[X][X] := ∑ j, C (s j) * X ^ (j : ℕ)
  have hev (t : Fin J → F[X]) : (∑ j, C (t j) * X ^ (j : ℕ)).eval (X ^ q) =
      ∑ j, t j * X ^ (q * (j : ℕ)) := by
    simp [eval_finsetSum, pow_mul]
  have hA : ∑ j, r j * X ^ (q * (j : ℕ)) = -(f ^ (q / 2) * ∑ j, s j * X ^ (q * (j : ℕ))) := by
    rw [eq_neg_iff_add_eq_zero, ← h, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have key : (C f * Rb ^ 2 - f.map C * Sb ^ 2).eval (X ^ q) = 0 := by
    have hmap : (f.map C).eval (X ^ q) = f ^ q := by
      rw [eval_map, show f.eval₂ C (X ^ q) = f.comp (X ^ q) from rfl, ← expand_eq_comp_X_pow,
        FiniteField.Polynomial.expand_card]
    rw [eval_sub, eval_mul, eval_mul, eval_pow, eval_pow, eval_C, hmap, hev, hev, hA,
      show f ^ q = f * (f ^ (q / 2)) ^ 2 by
        rw [← pow_mul, ← pow_succ']; congr 1; obtain ⟨t, ht⟩ := hq; omega]
    ring
  have hcoeff (t : Fin J → F[X]) (ht : ∀ j, (t j).natDegree < K) (k : ℕ) :
      ((∑ j, C (t j) * X ^ (j : ℕ)).coeff k).natDegree ≤ K - 1 := by
    rw [finsetSum_coeff]
    refine natDegree_sum_le_of_forall_le _ _ fun j _ => ?_
    rw [coeff_C_mul_X_pow]
    split_ifs
    · exact Nat.le_sub_one_of_lt (ht j)
    · simp
  have hRc : ∀ k, (Rb.coeff k).natDegree ≤ K - 1 := hcoeff r hr
  have hSc : ∀ k, (Sb.coeff k).natDegree ≤ K - 1 := hcoeff s hs
  have hdeg : ∀ k, ((C f * Rb ^ 2 - f.map C * Sb ^ 2).coeff k).natDegree < q := by
    intro k
    rw [coeff_sub]
    refine (natDegree_sub_le _ _).trans_lt (max_lt ?_ ?_)
    · rw [coeff_C_mul, sq]
      refine natDegree_mul_le.trans_lt ?_
      have := natDegree_coeff_mul_le hRc hRc k
      omega
    · rw [sq, ← mul_assoc]
      have h1 := natDegree_coeff_mul_le (P := f.map C) (d := 0)
        (fun i => by rw [coeff_map]; simp) hSc
      have := natDegree_coeff_mul_le h1 hSc k
      omega
  exact sub_eq_zero.mp (eq_zero_of_eval_X_pow_eq_zero hdeg key)

example := C_mul_sq_eq_map_C_mul_sq (F := ZMod 3) (f := X) (by rw [ZMod.card]; exact ⟨1, rfl⟩)
  (J := 0) (K := 1) one_pos (by simp) (r := 0) (s := 0) (by simp) (by simp) (by simp)

omit [Fintype F] in
/-- `f(X) R(X, Y)² = f(Y) S(X, Y)²` forces `S = 0` for `f` of odd degree: if `S ≠ 0`, the
degrees in `Y` are `2 deg_Y R` and `m + 2 deg_Y S`, of different parity. -/
theorem eq_zero_of_C_mul_sq_eq_of_odd {f : F[X]} (hf : Odd f.natDegree) {R S : F[X][X]}
    (h : C f * R ^ 2 = f.map C * S ^ 2) : S = 0 := by
  have hf0 : f ≠ 0 := fun h0 => by simp [h0] at hf
  have hmap0 : f.map C ≠ 0 := (Polynomial.map_ne_zero_iff C_injective).mpr hf0
  by_contra hS
  have hR : R ≠ 0 := by
    rintro hR0
    rw [hR0, zero_pow two_ne_zero, mul_zero] at h
    exact (mul_ne_zero hmap0 (pow_ne_zero 2 hS)) h.symm
  have := congrArg natDegree h
  rw [natDegree_C_mul hf0, natDegree_pow, natDegree_mul hmap0 (pow_ne_zero 2 hS),
    natDegree_pow, natDegree_map_eq_of_injective C_injective] at this
  obtain ⟨t, ht⟩ := hf
  omega

example := eq_zero_of_C_mul_sq_eq_of_odd (F := ZMod 3) (f := X) (by simp) (R := 0) (S := 0)
  (by simp)

/-- **Stepanov's independence lemma**: let `q = |F|` be odd and suppose `f(X) R² = f(Y) S²`
forces `S = 0`. If `∑_{j < J} (r_j + s_j f^{(q−1)/2}) X^{q j} = 0` with `deg r_j, deg s_j < K`
and `m + 2K ≤ q + 1`, then all `r_j` and `s_j` are `0`. -/
theorem eq_zero_of_sum_add_mul_pow_eq_zero_of_sq {f : F[X]}
    (hsq : ∀ R S : F[X][X], C f * R ^ 2 = f.map C * S ^ 2 → S = 0)
    (hq : Odd (Fintype.card F)) {J K : ℕ} (hK : f.natDegree + 2 * K ≤ Fintype.card F + 1)
    {r s : Fin J → F[X]} (hr : ∀ j, (r j).natDegree < K) (hs : ∀ j, (s j).natDegree < K)
    (h : ∑ j, (r j + s j * f ^ (Fintype.card F / 2)) * X ^ (Fintype.card F * (j : ℕ)) = 0) :
    r = 0 ∧ s = 0 := by
  rcases Nat.eq_zero_or_pos K with hK0 | hK0
  · exact ⟨funext fun j => absurd (hr j) (by omega), funext fun j => absurd (hs j) (by omega)⟩
  have hf0 : f ≠ 0 := by
    rintro rfl
    exact one_ne_zero (hsq 0 1 (by simp))
  have heq := C_mul_sq_eq_map_C_mul_sq hq hK0 hK hr hs h
  have hSb := hsq _ _ heq
  rw [hSb, zero_pow two_ne_zero, mul_zero] at heq
  have hRb := pow_eq_zero_iff two_ne_zero |>.mp
    ((mul_eq_zero.mp heq).resolve_left (C_ne_zero.mpr hf0))
  refine ⟨funext fun j => ?_, funext fun j => ?_⟩
  · rw [Pi.zero_apply, ← coeff_sum_C_mul_X_pow r j, hRb, coeff_zero]
  · rw [Pi.zero_apply, ← coeff_sum_C_mul_X_pow s j, hSb, coeff_zero]

example := eq_zero_of_sum_add_mul_pow_eq_zero_of_sq (F := ZMod 3) (f := X)
  (fun _ _ => eq_zero_of_C_mul_sq_eq_of_odd (by simp)) (by rw [ZMod.card]; exact ⟨1, rfl⟩)
  (J := 0) (K := 1) (by simp) (r := 0) (s := 0) (by simp) (by simp) (by simp)

/-- **Stepanov's independence lemma** for `f` of odd degree `m` over `F` with `q = |F|` odd:
if `∑_{j < J} (r_j + s_j f^{(q−1)/2}) X^{q j} = 0` with `deg r_j, deg s_j < K` and
`m + 2K ≤ q + 1`, then all `r_j` and `s_j` are `0`. -/
theorem eq_zero_of_sum_add_mul_pow_eq_zero {f : F[X]} (hf : Odd f.natDegree)
    (hq : Odd (Fintype.card F)) {J K : ℕ} (hK : f.natDegree + 2 * K ≤ Fintype.card F + 1)
    {r s : Fin J → F[X]} (hr : ∀ j, (r j).natDegree < K) (hs : ∀ j, (s j).natDegree < K)
    (h : ∑ j, (r j + s j * f ^ (Fintype.card F / 2)) * X ^ (Fintype.card F * (j : ℕ)) = 0) :
    r = 0 ∧ s = 0 :=
  eq_zero_of_sum_add_mul_pow_eq_zero_of_sq (fun _ _ => eq_zero_of_C_mul_sq_eq_of_odd hf) hq hK
    hr hs h

example := eq_zero_of_sum_add_mul_pow_eq_zero (F := ZMod 3) (f := X) (by simp)
  (by rw [ZMod.card]; exact ⟨1, rfl⟩) (J := 0) (K := 1) (by simp) (r := 0) (s := 0) (by simp)
  (by simp) (by simp)

end Transformer.Weil
