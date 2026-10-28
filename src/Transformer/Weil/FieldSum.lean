module

public import Mathlib.FieldTheory.Finite.GaloisField
public import Transformer.Weil.FrobeniusOrbit
public import Transformer.Weil.NewtonIdentity

/-!
# `S_d` is the character sum over the extension of degree `d`

For a finite extension `L` of degree `d` of the finite field `K`,
`∑_{x ∈ L} ψ(Tr_{L/K} f(x)) = S_d = ∑_{e deg P = d} deg P · λ(P)^e`. Group the `x` by their
minimal polynomial `P`, monic irreducible of degree `k ∣ d`:

* the `x` with minimal polynomial `P` are the `k` conjugates `x^{q^i}`, `i < k`, and each has
  `ψ(Tr f(x)) = λ(P)^{d/k}`;
* every monic irreducible `P` of degree `k ∣ d` is a minimal polynomial: `K[X]/P` has degree `k`
  over `K`, so it embeds in `L`.

So the fiber of `P` contributes `k λ(P)^{d/k}` if `k ∣ d` and `0` otherwise, which is the term of
`P` in `S_d`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- `∑_{1 ≤ e ≤ d} [k e = d] g(e) = [k ∣ d] g(d/k)` for `k, d ≥ 1`. -/
theorem sum_Icc_ite_mul_eq {M : Type*} [AddCommMonoid M] {k d : ℕ} (hk : 0 < k) (hd : 0 < d)
    (g : ℕ → M) :
    ∑ e ∈ Icc 1 d, (if k * e = d then g e else 0) = if k ∣ d then g (d / k) else 0 := by
  split_ifs with h
  · obtain ⟨c, rfl⟩ := h
    have hc : 0 < c := Nat.pos_of_ne_zero fun h0 => by simp [h0] at hd
    rw [Nat.mul_div_cancel_left c hk, sum_eq_single c
      (fun e _ he => ite_eq_right_iff.mpr fun h => absurd (Nat.eq_of_mul_eq_mul_left hk h) he)
      (fun h => absurd (mem_Icc.mpr ⟨hc, Nat.le_mul_of_pos_left c hk⟩) h)]
    simp
  · exact sum_eq_zero fun e _ => ite_eq_right_iff.mpr fun he => absurd ⟨e, he.symm⟩ h

example := sum_Icc_ite_mul_eq (M := ℕ) one_pos one_pos id

variable {K L : Type*} [Field K] [Fintype K] [Field L] [Algebra K L] [Finite L]

/-- The elements with the minimal polynomial of `x` are its conjugates `x^{q^i}`, `i < k`. -/
theorem minpoly_eq_iff (x y : L) :
    minpoly K y = minpoly K x ↔ ∃ i < (minpoly K x).natDegree, x ^ Fintype.card K ^ i = y := by
  have hx := IsIntegral.of_finite K x
  constructor
  · intro h
    have h0 : ((minpoly K x).map (algebraMap K L)).eval y = 0 := by
      rw [eval_map_algebraMap, ← h, minpoly.aeval]
    rw [map_minpoly_eq_prod, eval_prod, prod_eq_zero_iff] at h0
    obtain ⟨i, hi, h0⟩ := h0
    rw [eval_sub, eval_X, eval_C, sub_eq_zero] at h0
    exact ⟨i, mem_range.mp hi, h0.symm⟩
  · rintro ⟨i, -, rfl⟩
    refine (minpoly.eq_of_irreducible_of_monic (minpoly.irreducible hx) ?_ (minpoly.monic hx)).symm
    rw [aeval_pow_card_pow, minpoly.aeval, zero_pow (pow_ne_zero _ Fintype.card_ne_zero)]

omit [Fintype K] in
/-- A monic irreducible `P` of degree dividing `[L : K]` is the minimal polynomial of some
`x ∈ L`: `K[X]/P` has degree `deg P` over `K`, so it embeds in `L`. -/
theorem exists_minpoly_eq {P : K[X]} (hP : P.Monic) (hPi : Irreducible P)
    (hd : P.natDegree ∣ Module.finrank K L) : ∃ x : L, minpoly K x = P := by
  have := Fact.mk hPi
  have hfin : Module.finrank K (AdjoinRoot P) = P.natDegree := by
    rw [(AdjoinRoot.powerBasis hPi.ne_zero).finrank, AdjoinRoot.powerBasis_dim]
  obtain ⟨φ⟩ := FiniteField.nonempty_algHom_of_finrank_dvd (F := K) (K := AdjoinRoot P) (L := L)
    (by rw [hfin]; exact hd)
  refine ⟨φ (AdjoinRoot.root P), (minpoly.eq_of_irreducible_of_monic hPi ?_ hP).symm⟩
  rw [aeval_algHom_apply, AdjoinRoot.aeval_eq, AdjoinRoot.mk_self, map_zero]

example := exists_minpoly_eq (K := ZMod 2) (L := ZMod 2) monic_X irreducible_X
  (by rw [natDegree_X]; exact one_dvd _)

/-- **`S_d` is the character sum over `L`**: `∑_{x ∈ L} ψ(Tr_{L/K} f(x)) = S_d` for
`d = [L : K]`. -/
theorem sum_addChar_trace_aeval [Fintype L] (ψ : AddChar K ℂ) (f : K[X]) :
    ∑ x : L, ψ (Algebra.trace K L (aeval x f)) = lPowerSum ψ f (Module.finrank K L) := by
  classical
  have hd0 : 0 < Module.finrank K L := Module.finrank_pos
  have hmaps (x : L) (_ : x ∈ (univ : Finset L)) :
      minpoly K x ∈ irreducibles K (Module.finrank K L) :=
    mem_irreducibles.mpr ⟨minpoly.monic (IsIntegral.of_finite K x), minpoly.natDegree_le x,
      minpoly.irreducible (IsIntegral.of_finite K x)⟩
  rw [← sum_fiberwise_of_maps_to hmaps, lPowerSum]
  refine sum_congr rfl fun P hP => ?_
  obtain ⟨hPm, -, hPi⟩ := mem_irreducibles.mp hP
  rw [sum_Icc_ite_mul_eq hPi.natDegree_pos hd0]
  by_cases hex : ∃ x : L, minpoly K x = P
  · obtain ⟨x, rfl⟩ := hex
    have hconst (y : L) (hy : y ∈ univ.filter fun y : L => minpoly K y = minpoly K x) :
        ψ (Algebra.trace K L (aeval y f)) =
          ψ (rootSum f (minpoly K x)) ^ (Module.finrank K L / (minpoly K x).natDegree) := by
      rw [addChar_trace_aeval, (mem_filter.mp hy).2]
    have hfib : (univ.filter fun y : L => minpoly K y = minpoly K x) =
        (range (minpoly K x).natDegree).image (fun i => x ^ Fintype.card K ^ i) := by
      ext y
      simp only [mem_filter, mem_univ, true_and, mem_image, mem_range, minpoly_eq_iff]
    rw [ite_eq_left (minpoly.degree_dvd (IsIntegral.of_finite K x)), sum_eq_card_nsmul hconst,
      hfib, card_image_of_injOn (pow_card_pow_injOn x), card_range, nsmul_eq_mul]
  · rw [ite_eq_right fun h => hex (exists_minpoly_eq hPm hPi h)]
    exact sum_eq_zero fun y hy => absurd ⟨y, (mem_filter.mp hy).2⟩ hex

end Transformer.Weil
