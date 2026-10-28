module

public import Transformer.Weil.Divisors

/-!
# Newton's identity for an `L`-function over a finite field

Let `w` be a weight multiplicative on the monic polynomials and
`S_d = ∑_{P monic irreducible, e ≥ 1, e deg P = d} deg P · w(P)^e`. Then the coefficients
`c_n = ∑_{deg g = n} w(g)` satisfy `n c_n = ∑_{1 ≤ d ≤ n} S_d c_{n − d}`: this is
`T L′(T) = (∑_d S_d T^d) L(T)` for the Euler product `L(T) = ∏_P (1 − w(P) T^{deg P})⁻¹`.
Proof: `n = ∑_{P^e ∣ g} deg P` for every monic `g` of degree `n`, so
`n c_n = ∑_{P, e} deg P ∑_{deg g = n, P^e ∣ g} w(g) = ∑_{e deg P ≤ n} deg P w(P)^e c_{n − e deg P}`,
and grouping by `d = e deg P` gives the right side.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K] [Fintype K]

/-- `S_d = ∑_{P monic irreducible, e ≥ 1, e deg P = d} deg P · w(P)^e`, the coefficients of
`T L′/L`. -/
noncomputable def lPowerSum (w : K[X] → ℂ) (d : ℕ) : ℂ :=
  ∑ P ∈ irreducibles K d, ∑ e ∈ Icc 1 d,
    if P.natDegree * e = d then (P.natDegree : ℂ) * w P ^ e else 0

/-- The sum defining `S_d` may run over `deg P ≤ N` and `e ≤ N` for any `N ≥ d`. -/
theorem lPowerSum_eq_of_le (w : K[X] → ℂ) {d N : ℕ} (hd : d ≤ N) :
    lPowerSum w d = ∑ P ∈ irreducibles K N, ∑ e ∈ Icc 1 N,
      if P.natDegree * e = d then (P.natDegree : ℂ) * w P ^ e else 0 := by
  have hsub : irreducibles K d ⊆ irreducibles K N := fun P hP => by
    obtain ⟨h1, h2, h3⟩ := mem_irreducibles.mp hP
    exact mem_irreducibles.mpr ⟨h1, h2.trans hd, h3⟩
  have hzero (P : K[X]) (hPN : P ∈ irreducibles K N) (hPd : P ∉ irreducibles K d) :
      ∑ e ∈ Icc 1 N, (if P.natDegree * e = d then (P.natDegree : ℂ) * w P ^ e else 0) = 0 := by
    obtain ⟨h1, -, h3⟩ := mem_irreducibles.mp hPN
    have hk : d < P.natDegree := by
      by_contra hk
      exact hPd (mem_irreducibles.mpr ⟨h1, by omega, h3⟩)
    refine sum_eq_zero fun e he => ite_eq_right fun h => ?_
    have := Nat.le_mul_of_pos_right P.natDegree (mem_Icc.mp he).1
    omega
  rw [lPowerSum, ← sum_subset hsub hzero]
  refine sum_congr rfl fun P hP =>
    sum_subset (Icc_subset_Icc le_rfl hd) fun e he hed => ite_eq_right fun h => ?_
  have hk := (mem_irreducibles.mp hP).2.2.natDegree_pos
  simp only [mem_Icc] at he hed
  have := Nat.le_mul_of_pos_left e hk
  omega

example := lPowerSum_eq_of_le (K := ZMod 2) (addWeight 1 X) (d := 1) (N := 2) one_le_two

open Classical in
/-- **Newton's identity for the `L`-function**: `n c_n = ∑_{1 ≤ d ≤ n} S_d c_{n − d}`. -/
theorem natCast_mul_lCoeff {w : K[X] → ℂ} (hw : IsMonicMul w) (n : ℕ) :
    (n : ℂ) * lCoeff w n = ∑ d ∈ Icc 1 n, lPowerSum w d * lCoeff w (n - d) := by
  set T : K[X] → ℕ → ℂ := fun P e => (P.natDegree : ℂ) *
    (if P.natDegree * e ≤ n then w P ^ e * lCoeff w (n - P.natDegree * e) else 0) with hT
  have hL : (n : ℂ) * lCoeff w n = ∑ P ∈ irreducibles K n, ∑ e ∈ Icc 1 n, T P e := by
    rw [lCoeff, mul_sum]
    calc ∑ g ∈ monics K n, (n : ℂ) * w g
        = ∑ g ∈ monics K n, ∑ P ∈ irreducibles K n, ∑ e ∈ Icc 1 n,
            (P.natDegree : ℂ) * (if P ^ e ∣ g then w g else 0) := by
          refine sum_congr rfl fun g hg => ?_
          obtain ⟨hgm, hgn⟩ := mem_monics.mp hg
          have h := congrArg (Nat.cast (R := ℂ)) (sum_irreducibles_pow_dvd hgm hgn.le)
          rw [hgn] at h
          push_cast at h
          rw [← h, sum_mul]
          refine sum_congr rfl fun P _ => ?_
          rw [sum_mul]
          refine sum_congr rfl fun e _ => ?_
          split_ifs <;> simp
      _ = ∑ P ∈ irreducibles K n, ∑ e ∈ Icc 1 n, ∑ g ∈ monics K n,
            (P.natDegree : ℂ) * (if P ^ e ∣ g then w g else 0) := by
          rw [sum_comm]
          exact sum_congr rfl fun P _ => sum_comm
      _ = _ := by
          refine sum_congr rfl fun P hP => sum_congr rfl fun e _ => ?_
          rw [← mul_sum, sum_monics_pow_dvd hw (mem_irreducibles.mp hP).1]
  have hR : ∑ d ∈ Icc 1 n, lPowerSum w d * lCoeff w (n - d) =
      ∑ P ∈ irreducibles K n, ∑ e ∈ Icc 1 n, T P e := by
    calc ∑ d ∈ Icc 1 n, lPowerSum w d * lCoeff w (n - d)
        = ∑ d ∈ Icc 1 n, ∑ P ∈ irreducibles K n, ∑ e ∈ Icc 1 n,
            (if P.natDegree * e = d then (P.natDegree : ℂ) * w P ^ e else 0) *
              lCoeff w (n - d) := by
          refine sum_congr rfl fun d hd => ?_
          rw [lPowerSum_eq_of_le w (mem_Icc.mp hd).2, sum_mul]
          exact sum_congr rfl fun P _ => sum_mul _ _ _
      _ = ∑ P ∈ irreducibles K n, ∑ e ∈ Icc 1 n, ∑ d ∈ Icc 1 n,
            (if P.natDegree * e = d then (P.natDegree : ℂ) * w P ^ e else 0) *
              lCoeff w (n - d) := by
          rw [sum_comm]
          exact sum_congr rfl fun P _ => sum_comm
      _ = _ := by
          refine sum_congr rfl fun P hP => sum_congr rfl fun e he => ?_
          have hke : 1 ≤ P.natDegree * e :=
            Nat.mul_pos (mem_irreducibles.mp hP).2.2.natDegree_pos (mem_Icc.mp he).1
          simp only [ite_mul, zero_mul]
          rw [sum_ite_eq]
          simp only [hT, mem_Icc]
          split_ifs with h1 h2 h2
          · ring
          · exact absurd h1.2 h2
          · exact absurd ⟨hke, h2⟩ h1
          · ring
  rw [hL, hR]

example := natCast_mul_lCoeff (isMonicMul_addWeight (K := ZMod 2) 1 X) 2

end Transformer.Weil
