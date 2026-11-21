module

public import Mathlib.Algebra.Ring.GeomSum
public import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
public import Mathlib.FieldTheory.Minpoly.IsIntegrallyClosed
public import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
public import Mathlib.RingTheory.Polynomial.Cyclotomic.Roots
public import Mathlib.RingTheory.RootsOfUnity.Complex

/-!
# A determinant of roots of unity, certified in a finite field

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. To show that a
kernel `[ψ(f)]` is invertible, write its entries as integer polynomials at `ζ = ψ(1)`. If the
determinant `P(ζ)` vanished, the cyclotomic polynomial `Φ_p`, the minimal polynomial of `ζ`, would
divide `P`. In `ZMod q` an element `z ≠ 1` with `z ^ p = 1` is a root of `Φ_p`, so `P(z) = 0`, and
the same matrix at `z` has no inverse. So one inverse over `ZMod q`, which the kernel can check,
proves that the complex matrix is invertible (`det_aeval_ne_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix Polynomial

variable {p q : ℕ} [Fact p.Prime] [Fact q.Prime] {n : Type*} [Fintype n] [DecidableEq n]

omit [Fact q.Prime] in
/-- `ψ(1)` is a primitive `p`-th root of unity. -/
lemma isPrimitiveRoot_stdAddChar_one : IsPrimitiveRoot (ZMod.stdAddChar (1 : ZMod p) : ℂ) p := by
  have h := ZMod.stdAddChar_coe (N := p) 1
  simp only [Int.cast_one, mul_one] at h
  rw [h]
  exact Complex.isPrimitiveRoot_exp p (Fact.out : p.Prime).ne_zero

omit [Fact q.Prime] in
/-- `ψ(e) = ψ(1) ^ e`. -/
lemma stdAddChar_natCast (e : ℕ) :
    (ZMod.stdAddChar (e : ZMod p) : ℂ) = (ZMod.stdAddChar (1 : ZMod p) : ℂ) ^ e := by
  rw [← AddChar.map_nsmul_eq_pow, nsmul_one]

/-- An element `z ≠ 1` of `ZMod q` with `z ^ p = 1` is a root of `Φ_p`. -/
lemma eval₂_cyclotomic_eq_zero {z : ZMod q} (hz : z ≠ 1) (hzp : z ^ p = 1) :
    (cyclotomic p ℤ).eval₂ (Int.castRingHom (ZMod q)) z = 0 := by
  rw [cyclotomic_prime, ← coe_eval₂RingHom, map_sum]
  simp only [map_pow, coe_eval₂RingHom, eval₂_X]
  have h := mul_geom_sum z p
  rw [hzp, sub_self] at h
  exact (mul_eq_zero.1 h).resolve_left (sub_ne_zero.2 hz)

example : (cyclotomic 2 ℤ).eval₂ (Int.castRingHom (ZMod 3)) 2 = 0 :=
  eval₂_cyclotomic_eq_zero (p := 2) (by decide) (by decide)

/-- **A matrix of integer polynomials at `ψ(1)` is invertible if it is invertible at a `p`-th root
of unity `z ≠ 1` of a finite field.** -/
theorem det_aeval_ne_zero (A : Matrix n n ℤ[X]) {z : ZMod q} (hz : z ≠ 1) (hzp : z ^ p = 1)
    (N : Matrix n n (ZMod q)) (hN : A.map (eval₂ (Int.castRingHom (ZMod q)) z) * N = 1) :
    (A.map (aeval (ZMod.stdAddChar (1 : ZMod p) : ℂ))).det ≠ 0 := by
  intro h0
  have hζ := isPrimitiveRoot_stdAddChar_one (p := p)
  have hpos : 0 < p := (Fact.out : p.Prime).pos
  have hP : aeval (ZMod.stdAddChar (1 : ZMod p) : ℂ) A.det = 0 := by
    rw [← h0]
    exact AlgHom.map_det _ A
  obtain ⟨R, hR⟩ : cyclotomic p ℤ ∣ A.det := by
    rw [cyclotomic_eq_minpoly hζ hpos]
    exact minpoly.isIntegrallyClosed_dvd (hζ.isIntegral hpos) hP
  have hdet : (A.map (eval₂ (Int.castRingHom (ZMod q)) z)).det = 0 := by
    have hmap := RingHom.map_det (eval₂RingHom (Int.castRingHom (ZMod q)) z) A
    have hzero : eval₂RingHom (Int.castRingHom (ZMod q)) z A.det = 0 := by
      rw [hR, map_mul, coe_eval₂RingHom, eval₂_cyclotomic_eq_zero hz hzp, zero_mul]
    rw [hzero] at hmap
    exact hmap.symm
  have h1 := congrArg Matrix.det hN
  rw [det_mul, hdet, zero_mul, det_one] at h1
  exact zero_ne_one h1

example : ((1 : Matrix (Fin 1) (Fin 1) ℤ[X]).map
    (aeval (ZMod.stdAddChar (1 : ZMod 2) : ℂ))).det ≠ 0 :=
  det_aeval_ne_zero (q := 3) (z := 2) _ (by decide) (by decide) 1 (by
    rw [Matrix.mul_one, Matrix.map_one _ (by simp) (by simp)])

end Transformer.Grokking.FlatSpectrum.TablePart
