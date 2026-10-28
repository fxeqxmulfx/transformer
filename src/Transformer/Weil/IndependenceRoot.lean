module

public import Mathlib.FieldTheory.Perfect
public import Mathlib.FieldTheory.SplittingField.Construction
public import Mathlib.RingTheory.UniqueFactorizationDomain.Multiplicity
public import Transformer.Weil.Independence

/-!
# `f(X) R² = f(Y) S²` for `f` with a root of odd multiplicity

If `f ∈ F[X]` has a root `β` of odd multiplicity in an extension `E` of `F`, then
`f(X) R(X, Y)² = f(Y) S(X, Y)²` forces `S = 0`: over `E[X]`, the root `Y = β` has multiplicity
`2 μ(R)` on the left, since `f(X)` is a nonzero constant in `Y`, and `μ_β(f) + 2 μ(S)` on the
right. With `Independence` this gives Stepanov's independence lemma for every such `f`, of any
degree. Over a perfect field `K`, `f = P^e Q` with `P` irreducible, `e` odd and `P ∤ Q` has such
a root over every extension `L` of `K`: a root of `P`, a simple root since `P` is separable,
and not a root of `Q`, since `P` and `Q` are coprime. Every `f` that is not a constant times a
square has this form: split off the highest power `P^e` of an irreducible factor, and if `e` is
even, recurse on the rest.
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

/-- A polynomial of odd degree is not `c g²`. -/
theorem ne_C_mul_sq_of_odd {K : Type*} [Field K] {f : K[X]} (hf : Odd f.natDegree) (c : K)
    (g : K[X]) : f ≠ C c * g ^ 2 := by
  rintro rfl
  by_cases hc : c = 0
  · rw [hc, C_0, zero_mul, natDegree_zero] at hf
    exact Nat.not_odd_zero hf
  · rw [natDegree_C_mul hc, natDegree_pow] at hf
    exact Nat.not_odd_iff_even.mpr (even_two_mul _) hf

example := ne_C_mul_sq_of_odd (K := ZMod 3) (f := X) (by rw [natDegree_X]; exact odd_one) 1 1

/-- A polynomial with a root of odd multiplicity is not `c g²`. -/
theorem ne_C_mul_sq_of_rootMultiplicity {K : Type*} [Field K] {f : K[X]} {a : K}
    (ha : Odd (rootMultiplicity a f)) (c : K) (g : K[X]) : f ≠ C c * g ^ 2 := by
  rintro rfl
  by_cases h0 : C c * g ^ 2 = 0
  · rw [h0, rootMultiplicity_zero] at ha
    exact Nat.not_odd_zero ha
  · have hg : g ≠ 0 := fun hg => h0 (by rw [hg]; ring)
    rw [rootMultiplicity_mul h0, rootMultiplicity_C, sq, rootMultiplicity_mul (mul_ne_zero hg hg),
      zero_add] at ha
    exact Nat.not_odd_iff_even.mpr ⟨_, rfl⟩ ha

example := ne_C_mul_sq_of_rootMultiplicity (K := ZMod 3) (f := X) (a := 0) (by
  rw [show (X : (ZMod 3)[X]) = X - C 0 by simp, rootMultiplicity_X_sub_C_self]
  exact odd_one) 1 1

/-- **The odd part of `f`**: if `f` is not `c g²`, then `f = P^e Q` with `P` monic irreducible,
`e` odd and `P ∤ Q`. -/
theorem exists_monic_irreducible_odd_pow_mul {K : Type*} [Field K] {f : K[X]}
    (hf : ∀ (c : K) (g : K[X]), f ≠ C c * g ^ 2) :
    ∃ P Q : K[X], ∃ e : ℕ, P.Monic ∧ Irreducible P ∧ Odd e ∧ ¬P ∣ Q ∧ f = P ^ e * Q := by
  classical
  obtain ⟨n, hn⟩ : ∃ n, f.natDegree = n := ⟨_, rfl⟩
  induction n using Nat.strong_induction_on generalizing f with
  | _ n ih =>
  have hf0 : f ≠ 0 := fun h => hf 0 1 (by rw [h, C_0, zero_mul])
  have hu : ¬IsUnit f := fun h => by
    obtain ⟨c, -, rfl⟩ := Polynomial.isUnit_iff.mp h
    exact hf c 1 (by rw [one_pow, mul_one])
  obtain ⟨P₀, hP₀, hP₀f⟩ := WfDvdMonoid.exists_irreducible_factor hu hf0
  have hP : Irreducible (normalize P₀) := (normalize_associated P₀).symm.irreducible hP₀
  have hPm : (normalize P₀).Monic := monic_normalize hP₀.ne_zero
  obtain ⟨e, Q, hPQ, hfeq⟩ := WfDvdMonoid.max_power_factor hf0 hP
  rcases Nat.even_or_odd e with ⟨k, rfl⟩ | he
  · have hQ0 : Q ≠ 0 := fun h => hf0 (by rw [hfeq, h, mul_zero])
    have hk : k ≠ 0 := fun h => hPQ (by
      rw [h, add_zero, pow_zero, one_mul] at hfeq
      exact hfeq ▸ normalize_dvd_iff.mpr hP₀f)
    have hdeg : Q.natDegree < n := by
      have := hP.natDegree_pos
      rw [← hn, hfeq, natDegree_mul (pow_ne_zero _ hP.ne_zero) hQ0, natDegree_pow]
      nlinarith [Nat.pos_of_ne_zero hk]
    have hQ (c : K) (g : K[X]) : Q ≠ C c * g ^ 2 := fun h =>
      hf c (normalize P₀ ^ k * g) (by rw [hfeq, h]; ring)
    obtain ⟨P, Q', e, hPm', hP', he, hPQ', hQeq⟩ := ih _ hdeg hQ rfl
    refine ⟨P, normalize P₀ ^ (k + k) * Q', e, hPm', hP', he, fun hdvd => ?_,
      by rw [hfeq, hQeq]; ring⟩
    rcases hP'.prime.dvd_or_dvd hdvd with h | h
    · refine hPQ ((hP'.dvd_symm hP (hP'.prime.dvd_of_dvd_pow h)).trans ?_)
      exact (dvd_pow_self P (Nat.pos_iff_ne_zero.mp he.pos)).trans (Dvd.intro _ hQeq.symm)
    · exact hPQ' h
  · exact ⟨normalize P₀, Q, e, hPm, hP, he, hPQ, hfeq⟩

example := exists_monic_irreducible_odd_pow_mul (K := ZMod 3) (f := X)
  (ne_C_mul_sq_of_odd (by rw [natDegree_X]; exact odd_one))

/-- Over a perfect field `K`, if `f` is not `c g²`, then over every extension `L`,
`f(X) R(X, Y)² = f(Y) S(X, Y)²` forces `S = 0`. -/
theorem eq_zero_of_C_mul_sq_eq_of_ne_C_mul_sq {K L : Type*} [Field K] [PerfectField K]
    [Field L] [Algebra K L] {f : K[X]}
    (hf : ∀ (c : K) (g : K[X]), f ≠ C c * g ^ 2) {R S : L[X][X]}
    (h : C (f.map (algebraMap K L)) * R ^ 2 = (f.map (algebraMap K L)).map C * S ^ 2) :
    S = 0 := by
  obtain ⟨P, Q, e, -, hP, he, hPQ, hfeq⟩ := exists_monic_irreducible_odd_pow_mul hf
  obtain ⟨β, hβ⟩ := exists_odd_rootMultiplicity (L := L) hP he hPQ hfeq
  exact eq_zero_of_C_mul_sq_eq_of_rootMultiplicity hβ h

example := eq_zero_of_C_mul_sq_eq_of_ne_C_mul_sq (K := ZMod 3) (L := ZMod 3) (f := X)
  (ne_C_mul_sq_of_odd (by rw [natDegree_X]; exact odd_one)) (R := 0) (S := 0) (by simp)

end Transformer.Weil
