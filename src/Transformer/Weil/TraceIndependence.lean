module

public import Transformer.Weil.Independence

/-!
# Stepanov's independence lemma for the trace polynomial

Let `g` have degree `m q` with `p` prime to `m`, and `Q = p q`. If
`∑_{i < p} ∑_{j < J} r_{ij} g^i X^{Q j} = 0` with `deg r_{ij} < K ≤ q`, then every `r_{ij}` is
`0`. With `h_i = ∑_j r_{ij} X^{Q j}`, a nonzero `h_i` has degree `Q a + e` with `e < K`, so
`deg(h_i g^i) = q (p a + i m) + e` with `e < q`: its quotient by `q` is `i m` modulo `p`, which
determines `i`. So the nonzero terms `h_i g^i` have distinct degrees and cannot cancel; every
`h_i` is `0`, and then every `r_{ij}` is (base-`Q` digits). For the trace polynomial
`g = T_ν(f)`, `q = p^{ν − 1}` and `m = deg f`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {F : Type*} [Field F]

/-- A nonzero `h = ∑_j r_j X^{Q j}` with `deg r_j < K ≤ Q` has `deg h mod Q < K`: at a degree
`N` with `N mod Q ≥ K`, every term `r_j X^{Q j}` has coefficient `0`. -/
theorem natDegree_mod_lt_of_sum_mul_X_pow {Q J K : ℕ} (hKQ : K ≤ Q) {r : Fin J → F[X]}
    (hr : ∀ j, (r j).natDegree < K) (h : ∑ j, r j * X ^ (Q * (j : ℕ)) ≠ 0) :
    (∑ j, r j * X ^ (Q * (j : ℕ))).natDegree % Q < K := by
  set N := (∑ j, r j * X ^ (Q * (j : ℕ))).natDegree
  by_contra hN
  rw [not_lt] at hN
  have hc : (∑ j, r j * X ^ (Q * (j : ℕ))).coeff N = 0 := by
    rw [finsetSum_coeff]
    refine sum_eq_zero fun j _ => ?_
    rw [coeff_mul_X_pow']
    split_ifs with hj
    · have hrj := hr j
      have hQ : 0 < Q := by omega
      have hdm := Nat.div_add_mod N Q
      have hjN : Q * j ≤ Q * (N / Q) :=
        Nat.mul_le_mul_left Q ((Nat.le_div_iff_mul_le hQ).mpr (by rw [mul_comm]; exact hj))
      exact coeff_eq_zero_of_natDegree_lt (by omega)
    · rfl
  exact h (leadingCoeff_eq_zero.mp hc)

example := natDegree_mod_lt_of_sum_mul_X_pow (F := ZMod 2) (Q := 2) (J := 1) (K := 1)
  (by norm_num) (r := fun _ => 1) (fun _ => by simp) (by simp)

/-- **Independence for the trace polynomial**: if `deg g = m q`, `p` is prime to `m ≥ 1` and
`∑_{i < p} ∑_{j < J} r_{ij} g^i X^{p q j} = 0` with `deg r_{ij} < K ≤ q`, then `r = 0`. -/
theorem eq_zero_of_sum_mul_pow_mul_X_pow_eq_zero {g : F[X]} {p m q J K : ℕ}
    (hpm : Nat.Coprime p m) (hm : 1 ≤ m) (hg : g.natDegree = m * q) (hK : K ≤ q)
    {r : Fin p → Fin J → F[X]} (hr : ∀ i j, (r i j).natDegree < K)
    (h : ∑ i, ∑ j, r i j * g ^ (i : ℕ) * X ^ (p * q * (j : ℕ)) = 0) : r = 0 := by
  rcases Nat.eq_zero_or_pos K with hK0 | hK0
  · exact funext fun i => funext fun j => absurd (hr i j) (by omega)
  rcases Nat.eq_zero_or_pos p with hp0 | hp
  · subst hp0
    exact funext fun i => i.elim0
  have hq : 0 < q := by omega
  have hKQ : K ≤ p * q := hK.trans (Nat.le_mul_of_pos_left q hp)
  set H : Fin p → F[X] := fun i => ∑ j, r i j * X ^ (p * q * (j : ℕ)) with hH
  have hsum : ∑ i, H i * g ^ (i : ℕ) = 0 := by
    rw [← h]
    refine sum_congr rfl fun i _ => ?_
    rw [hH, sum_mul]
    exact sum_congr rfl fun j _ => by ring
  have hg0 : g ≠ 0 := by
    rintro rfl
    rw [natDegree_zero] at hg
    have := Nat.mul_pos hm hq
    omega
  have key (N i : ℕ) (hN : N % (p * q) < K) :
      (N + i * (m * q)) / q % p = i * m % p := by
    have hdm := Nat.div_add_mod N (p * q)
    have e : N + i * (m * q) = q * (p * (N / (p * q)) + i * m) + N % (p * q) := by
      calc N + i * (m * q) = (p * q * (N / (p * q)) + N % (p * q)) + i * (m * q) := by
            rw [hdm]
        _ = q * (p * (N / (p * q)) + i * m) + N % (p * q) := by ring
    rw [e, Nat.mul_add_div hq, Nat.div_eq_of_lt (by omega : N % (p * q) < q), add_zero,
      Nat.mul_add_mod]
  have hdeg (i : Fin p) (hi : H i ≠ 0) :
      (H i * g ^ (i : ℕ)).natDegree = (H i).natDegree + i * (m * q) := by
    rw [natDegree_mul hi (pow_ne_zero _ hg0), natDegree_pow, hg]
  have hdist (i i' : Fin p) (hi : H i ≠ 0) (hi' : H i' ≠ 0)
      (heq : (H i * g ^ (i : ℕ)).natDegree = (H i' * g ^ (i' : ℕ)).natDegree) : i = i' := by
    rw [hdeg i hi, hdeg i' hi'] at heq
    have h1 := key _ i (natDegree_mod_lt_of_sum_mul_X_pow hKQ (hr i) hi)
    have h2 := key _ i' (natDegree_mod_lt_of_sum_mul_X_pow hKQ (hr i') hi')
    rw [heq] at h1
    have hmod : (i : ℕ) * m ≡ i' * m [MOD p] := h1.symm.trans h2
    exact Fin.ext (Nat.ModEq.eq_of_lt_of_lt (Nat.ModEq.cancel_right_of_coprime hpm hmod) i.2 i'.2)
  have hall (i : Fin p) : H i * g ^ (i : ℕ) = 0 := by
    have hd := degree_sum_eq_of_disjoint (fun i => H i * g ^ (i : ℕ)) univ ?_
    · rw [hsum, degree_zero] at hd
      exact degree_eq_bot.mp ((Finset.sup_eq_bot_iff _ _).mp hd.symm i (mem_univ i))
    · intro i hi i' hi' hne
      simp only [Function.onFun, Function.comp_apply]
      intro heq
      exact hne (hdist i i' (left_ne_zero_of_mul hi.2) (left_ne_zero_of_mul hi'.2)
        (natDegree_eq_of_degree_eq heq))
  funext i j
  have hHi : H i = 0 := (mul_eq_zero.mp (hall i)).resolve_right (pow_ne_zero _ hg0)
  have hΦ := eq_zero_of_eval_X_pow_eq_zero (n := p * q) (Φ := ∑ j, C (r i j) * X ^ (j : ℕ))
    (fun k => by
      rw [finsetSum_coeff]
      refine (natDegree_sum_le_of_forall_le _ _ (n := K - 1) fun j _ => ?_).trans_lt
        (by omega)
      rw [coeff_C_mul_X_pow]
      split_ifs
      · exact Nat.le_sub_one_of_lt (hr i j)
      · simp)
    (by rw [← hHi]; simp [hH, eval_finsetSum, pow_mul])
  rw [Pi.zero_apply, Pi.zero_apply, ← coeff_sum_C_mul_X_pow (r i) j, hΦ, coeff_zero]

example := eq_zero_of_sum_mul_pow_mul_X_pow_eq_zero (F := ZMod 2) (g := X) (p := 2) (m := 1)
  (q := 1) (J := 1) (K := 1) (by norm_num) le_rfl (by simp) le_rfl (r := 0) (fun _ _ => by simp)
  (by simp)

end Transformer.Weil
