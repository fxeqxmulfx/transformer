module

public import Mathlib.FieldTheory.Perfect
public import Mathlib.FieldTheory.SplittingField.Construction
public import Transformer.Weil.Independence

/-!
# `f(X) R² = f(Y) S²` for `f` with a root of odd multiplicity

If `f ∈ F[X]` has a root `β` of odd multiplicity in an extension `E` of `F`, then
`f(X) R(X, Y)² = f(Y) S(X, Y)²` forces `S = 0`: over `E[X]`, the root `Y = β` has multiplicity
`2 μ(R)` on the left, since `f(X)` is a nonzero constant in `Y`, and `μ_β(f) + 2 μ(S)` on the
right. With `Independence` this gives Stepanov's independence lemma for every such `f`, of any
degree. Over a perfect field `K`, `f = P^e Q` with `P` irreducible, `e` odd and `P ∤ Q` has such
a root over every extension `L` of `K`: a root of `P`, a simple root since `P` is separable,
and not a root of `Q`, since `P` and `Q` are coprime.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial

/-- `f(X) R(X, Y)² = f(Y) S(X, Y)²` forces `S = 0` when `f` has a root `β` of odd multiplicity
in an extension `E`: at `Y = β` the multiplicities are `2 μ(R)` and `μ_β(f) + 2 μ(S)`. -/
theorem eq_zero_of_C_mul_sq_eq_of_rootMultiplicity {F E : Type*} [Field F] [Field E]
    [Algebra F E] {f : F[X]} {β : E} (hβ : Odd (rootMultiplicity β (f.map (algebraMap F E))))
    {R S : F[X][X]} (h : C f * R ^ 2 = f.map C * S ^ 2) : S = 0 := by
  have hf0 : f ≠ 0 := by
    rintro rfl
    rw [Polynomial.map_zero, rootMultiplicity_zero] at hβ
    exact Nat.not_odd_zero hβ
  by_contra hS
  set φ : F[X] →+* E[X] := mapRingHom (algebraMap F E)
  have hφ : Function.Injective φ := Polynomial.map_injective _ (algebraMap F E).injective
  set fE := f.map (algebraMap F E)
  have hfE : fE ≠ 0 := (Polynomial.map_ne_zero_iff (algebraMap F E).injective).mpr hf0
  have hcomp : φ.comp C = (C : E →+* E[X]).comp (algebraMap F E) :=
    RingHom.ext fun x => by simp [φ]
  have h' := congrArg (Polynomial.map φ) h
  rw [Polynomial.map_mul, Polynomial.map_mul, Polynomial.map_pow, Polynomial.map_pow, map_C,
    Polynomial.map_map, hcomp, ← Polynomial.map_map] at h'
  change C fE * (R.map φ) ^ 2 = fE.map C * (S.map φ) ^ 2 at h'
  have hSE : S.map φ ≠ 0 := (Polynomial.map_ne_zero_iff hφ).mpr hS
  have hrhs : fE.map C * (S.map φ) ^ 2 ≠ 0 :=
    mul_ne_zero ((Polynomial.map_ne_zero_iff C_injective).mpr hfE) (pow_ne_zero 2 hSE)
  have hlhs : C fE * (R.map φ) ^ 2 ≠ 0 := h' ▸ hrhs
  have hRE : R.map φ ≠ 0 := fun h0 => hlhs (by rw [h0]; ring)
  have hm := congrArg (rootMultiplicity (C β)) h'
  rw [rootMultiplicity_mul hlhs, rootMultiplicity_mul hrhs, rootMultiplicity_C, sq, sq,
    rootMultiplicity_mul (mul_ne_zero hRE hRE), rootMultiplicity_mul (mul_ne_zero hSE hSE),
    ← eq_rootMultiplicity_map C_injective] at hm
  obtain ⟨k, hk⟩ := hβ
  omega

example := eq_zero_of_C_mul_sq_eq_of_rootMultiplicity (F := ZMod 3) (E := ZMod 3) (f := X)
  (β := 0) (by
    rw [Polynomial.map_X, show (X : (ZMod 3)[X]) = X - C 0 by simp, rootMultiplicity_X_sub_C_self]
    exact odd_one)
  (R := 0) (S := 0) (by simp)

/-- For `f = P^e Q` over a perfect field `K`, `P` irreducible, `e` odd and `P ∤ Q`, and any
extension `L` of `K`, `f` has a root of multiplicity `e` in the splitting field of `P` over
`L`. -/
theorem exists_odd_rootMultiplicity {K L : Type*} [Field K] [PerfectField K] [Field L]
    [Algebra K L] {f P Q : K[X]} {e : ℕ} (hP : Irreducible P) (he : Odd e) (hPQ : ¬P ∣ Q)
    (hf : f = P ^ e * Q) : ∃ β : (P.map (algebraMap K L)).SplittingField,
      Odd (rootMultiplicity β ((f.map (algebraMap K L)).map (algebraMap L _))) := by
  classical
  set E := (P.map (algebraMap K L)).SplittingField
  set ι : K →+* E := (algebraMap L E).comp (algebraMap K L)
  have hmap (g : K[X]) : (g.map (algebraMap K L)).map (algebraMap L E) = g.map ι :=
    Polynomial.map_map _ _ _
  have hdeg : degree ((P.map (algebraMap K L)).map (algebraMap L E)) ≠ 0 := by
    rw [degree_map, degree_map, degree_eq_natDegree hP.ne_zero]
    exact_mod_cast hP.natDegree_pos.ne'
  obtain ⟨β, hβ⟩ := (SplittingField.splits (P.map (algebraMap K L))).exists_eval_eq_zero hdeg
  refine ⟨β, ?_⟩
  rw [hmap] at hβ ⊢
  have hPι : P.map ι ≠ 0 := (Polynomial.map_ne_zero_iff ι.injective).mpr hP.ne_zero
  have hQι : Q.map ι ≠ 0 :=
    (Polynomial.map_ne_zero_iff ι.injective).mpr fun h0 => hPQ (h0 ▸ dvd_zero P)
  have h1 : rootMultiplicity β (P.map ι) = 1 :=
    le_antisymm (rootMultiplicity_le_one_of_separable
      (PerfectField.separable_of_irreducible hP).map β) ((rootMultiplicity_pos hPι).mpr hβ)
  have h0 : rootMultiplicity β (Q.map ι) = 0 := by
    refine rootMultiplicity_eq_zero fun hQβ => ?_
    obtain ⟨a, b, hab⟩ := ((hP.coprime_iff_not_dvd).mpr hPQ).map (mapRingHom ι)
    have := congrArg (eval β) hab
    rw [eval_add, eval_mul, eval_mul, coe_mapRingHom, hβ, hQβ.eq_zero,
      mul_zero, mul_zero, add_zero, eval_one] at this
    exact zero_ne_one this
  rw [hf, Polynomial.map_mul, Polynomial.map_pow,
    rootMultiplicity_mul (mul_ne_zero (pow_ne_zero _ hPι) hQι), ← count_roots, roots_pow,
    Multiset.count_nsmul, count_roots, h1, h0, mul_one, add_zero]
  exact he

example := exists_odd_rootMultiplicity (K := ZMod 3) (L := ZMod 3) (f := X) (P := X) (Q := 1)
  (e := 1) irreducible_X odd_one (by rw [X_dvd_iff]; simp) (by simp)

end Transformer.Weil
