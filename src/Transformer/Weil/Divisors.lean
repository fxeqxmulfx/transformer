module

public import Mathlib.Algebra.Polynomial.FieldDivision
public import Mathlib.RingTheory.UniqueFactorizationDomain.NormalizedFactors
public import Transformer.Weil.LFunction

/-!
# Prime-power divisors of monic polynomials over a finite field

For monic `g` of degree `≤ N` over a finite field `K`, the prime powers `P^e ∣ g`, `P` monic
irreducible and `1 ≤ e ≤ N`, count the degree: `∑_{P, e : P^e ∣ g} deg P = deg g`. Indeed
`P^e ∣ g` exactly for `e ≤ v_P(g)`, the multiplicity of `P` among the normalized factors of `g`,
and `∑_P v_P(g) deg P = deg g`.

For `P` monic of degree `k`, `h ↦ P^e h` maps the monics of degree `n − k e` onto those of
degree `n` divisible by `P^e`. So for the multiplicative `λ(g) = ψ(∑_i f(β_i))`,
`∑_{deg g = n, P^e ∣ g} λ(g) = λ(P)^e c_{n − k e}` if `k e ≤ n`, and `0` otherwise.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset UniqueFactorizationMonoid

variable {K : Type*} [Field K]

theorem rootSum_pow (f : K[X]) {P : K[X]} (hP : P ≠ 0) (e : ℕ) :
    rootSum f (P ^ e) = e • rootSum f P := by
  induction e with
  | zero => simp [rootSum_one]
  | succ e ih => rw [pow_succ, rootSum_mul f (pow_ne_zero e hP) hP, ih, succ_nsmul]

example := rootSum_pow (K := ℚ) X X_ne_zero 2

/-- `P^e ∣ g` exactly for `e ≤ v_P(g)`, the multiplicity of `P` among the normalized factors of
`g`. -/
theorem pow_dvd_iff_le_count [DecidableEq K] [DecidableEq K[X]] {P g : K[X]} (hP : P.Monic)
    (hPi : Irreducible P) (hg : g ≠ 0) (e : ℕ) :
    P ^ e ∣ g ↔ e ≤ (normalizedFactors g).count P := by
  rw [Multiset.le_count_iff_replicate_le]
  constructor
  · intro h
    have h' := (dvd_iff_normalizedFactors_le_normalizedFactors (pow_ne_zero e hPi.ne_zero) hg).mp h
    rwa [normalizedFactors_pow, normalizedFactors_irreducible hPi, hP.normalize_eq_self,
      Multiset.nsmul_singleton] at h'
  · intro h
    exact (Multiset.prod_replicate e P ▸ Multiset.prod_dvd_prod_of_le h).trans
      (prod_normalizedFactors hg).dvd

example := pow_dvd_iff_le_count (K := ZMod 2) monic_X irreducible_X X_ne_zero 1

variable [Fintype K]

open Classical in
/-- The monic irreducible polynomials of degree at most `N`. -/
noncomputable def irreducibles (K : Type*) [Field K] [Fintype K] (N : ℕ) : Finset K[X] :=
  ((range (N + 1)).biUnion (monics K)).filter Irreducible

theorem mem_irreducibles {N : ℕ} {P : K[X]} :
    P ∈ irreducibles K N ↔ P.Monic ∧ P.natDegree ≤ N ∧ Irreducible P := by
  simp only [irreducibles, mem_filter, mem_biUnion, mem_range, mem_monics]
  constructor
  · rintro ⟨⟨k, hk, hP, rfl⟩, hi⟩
    exact ⟨hP, by omega, hi⟩
  · rintro ⟨hP, hN, hi⟩
    exact ⟨⟨_, by omega, hP, rfl⟩, hi⟩

open Classical in
/-- **Degree from prime powers**: for monic `g` of degree `≤ N`,
`∑_{P monic irreducible, deg P ≤ N} ∑_{1 ≤ e ≤ N} [P^e ∣ g] deg P = deg g`. -/
theorem sum_irreducibles_pow_dvd {g : K[X]} (hg : g.Monic) {N : ℕ}
    (hN : g.natDegree ≤ N) :
    ∑ P ∈ irreducibles K N, ∑ e ∈ Icc 1 N, (if P ^ e ∣ g then P.natDegree else 0) =
      g.natDegree := by
  classical
  have hg0 := hg.ne_zero
  have hFm (Q : K[X]) (hQ : Q ∈ normalizedFactors g) : Q.Monic ∧ Irreducible Q := by
    have hi := irreducible_of_normalized_factor Q hQ
    refine ⟨?_, hi⟩
    rw [← normalize_normalized_factor Q hQ]
    exact monic_normalize hi.ne_zero
  have hprod : (normalizedFactors g).prod = g := by
    have hm := monic_multiset_prod_of_monic (normalizedFactors g) id fun Q hQ => (hFm Q hQ).1
    rw [Multiset.map_id] at hm
    exact eq_of_monic_of_associated hm hg (prod_normalizedFactors hg0)
  have hcount (P : K[X]) (hP : P ∈ irreducibles K N) :
      ∑ e ∈ Icc 1 N, (if P ^ e ∣ g then P.natDegree else 0) =
        (normalizedFactors g).count P * P.natDegree := by
    obtain ⟨hPm, -, hPi⟩ := mem_irreducibles.mp hP
    have hcN : (normalizedFactors g).count P ≤ N := by
      have h1 := natDegree_le_of_dvd ((pow_dvd_iff_le_count hPm hPi hg0 _).mpr le_rfl) hg0
      rw [natDegree_pow] at h1
      have hk := hPi.natDegree_pos
      nlinarith
    rw [← sum_filter, sum_const, smul_eq_mul]
    congr 1
    rw [show (Icc 1 N).filter (fun e => P ^ e ∣ g) = Icc 1 ((normalizedFactors g).count P) by
      ext e
      simp only [mem_filter, mem_Icc, pow_dvd_iff_le_count hPm hPi hg0]
      omega, Nat.card_Icc, Nat.add_sub_cancel]
  rw [sum_congr rfl hcount]
  have hsub : (normalizedFactors g).toFinset ⊆ irreducibles K N := fun Q hQ => by
    rw [Multiset.mem_toFinset] at hQ
    obtain ⟨hQm, hQi⟩ := hFm Q hQ
    exact mem_irreducibles.mpr
      ⟨hQm, (natDegree_le_of_dvd (dvd_of_mem_normalizedFactors hQ) hg0).trans hN, hQi⟩
  rw [← sum_subset hsub fun Q _ hQ => by
    rw [Multiset.count_eq_zero.mpr (fun h => hQ (Multiset.mem_toFinset.mpr h)), zero_mul]]
  conv_rhs => rw [← hprod, natDegree_multiset_prod_of_monic _ fun Q hQ => (hFm Q hQ).1,
    sum_multiset_map_count]
  rfl

example := sum_irreducibles_pow_dvd (K := ZMod 2) monic_X (N := 1) natDegree_X.le

open Classical in
/-- For `P` monic of degree `k`: `∑_{deg g = n, P^e ∣ g} λ(g) = λ(P)^e c_{n − k e}` if
`k e ≤ n`, and `0` otherwise. -/
theorem sum_monics_pow_dvd (ψ : AddChar K ℂ) (f : K[X]) {P : K[X]} (hP : P.Monic) (e n : ℕ) :
    ∑ g ∈ monics K n, (if P ^ e ∣ g then ψ (rootSum f g) else 0) =
      if P.natDegree * e ≤ n then ψ (rootSum f P) ^ e * lCoeff ψ f (n - P.natDegree * e)
      else 0 := by
  classical
  rw [← sum_filter]
  have hPe : (P ^ e).Monic := hP.pow e
  have hdeg : (P ^ e).natDegree = P.natDegree * e := by rw [natDegree_pow, mul_comm]
  split_ifs with hn
  · have hmap : (monics K n).filter (fun g => P ^ e ∣ g) =
        (monics K (n - P.natDegree * e)).map ⟨(P ^ e * ·), mul_right_injective₀ hPe.ne_zero⟩ := by
      ext g
      simp only [mem_filter, mem_map, mem_monics, Function.Embedding.coeFn_mk]
      constructor
      · rintro ⟨⟨hg, hgn⟩, h, rfl⟩
        have hh := hPe.of_mul_monic_left hg
        refine ⟨h, ⟨hh, ?_⟩, rfl⟩
        rw [natDegree_mul hPe.ne_zero hh.ne_zero, hdeg] at hgn
        omega
      · rintro ⟨h, ⟨hh, hhn⟩, rfl⟩
        refine ⟨⟨hPe.mul hh, ?_⟩, dvd_mul_right _ _⟩
        rw [natDegree_mul hPe.ne_zero hh.ne_zero, hdeg, hhn]
        omega
    rw [hmap, sum_map, lCoeff, mul_sum]
    refine sum_congr rfl fun h hh => ?_
    rw [Function.Embedding.coeFn_mk, rootSum_mul f hPe.ne_zero (mem_monics.mp hh).1.ne_zero,
      AddChar.map_add_eq_mul, rootSum_pow f hP.ne_zero, AddChar.map_nsmul_eq_pow]
  · refine sum_eq_zero fun g hg => ?_
    obtain ⟨hg', hdvd⟩ := mem_filter.mp hg
    obtain ⟨hgm, hgn⟩ := mem_monics.mp hg'
    have h1 := natDegree_le_of_dvd hdvd hgm.ne_zero
    rw [hdeg, hgn] at h1
    exact absurd h1 hn

example := sum_monics_pow_dvd (K := ZMod 2) 1 X monic_X 1 1

end Transformer.Weil
