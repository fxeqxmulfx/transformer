module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6BlindDet

/-!
# PT6: the eight equations of a block of the certificate of `⌊|a − b| / 2⌋`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Four columns of the
certificate hold four unknowns, and `Kᴴ Y = Yᴴ K` gives eight linear equations on them and their
conjugates. The matrix of the system has powers of `ζ` for entries, so its inverse over `ZMod 389`
proves it invertible, and the unknowns vanish (`blindSys_solve`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Matrix Polynomial

attribute [local instance] fact_prime_97

/-- The system on `y₀, y₁, y₂, y₃` and their conjugates, with `w = ζ` and `v = ζ̄`. -/
def blindSys {R : Type*} [CommRing R] (w v : R) : Matrix (Fin 8) (Fin 8) R :=
  !![v ^ 2, v ^ 4, 0, 0, -w ^ 2, -w ^ 4, 0, 0;
     v ^ 8, v ^ 5, 0, 0, -w ^ 8, -w ^ 5, 0, 0;
     0, 0, v ^ 5, v ^ 8, 0, 0, -w ^ 5, -w ^ 8;
     0, 0, v ^ 10, v ^ 3, 0, 0, -w ^ 10, -w ^ 3;
     0, 0, v ^ 3, v ^ 10, -1, -w ^ 2, 0, 0;
     1, v ^ 2, 0, 0, 0, 0, -w ^ 3, -w ^ 10;
     0, 0, v ^ 13, 1, -w ^ 5, -w ^ 3, 0, 0;
     v ^ 5, v ^ 3, 0, 0, 0, 0, -w ^ 13, -1]

lemma blindSys_map {R S F : Type*} [CommRing R] [CommRing S] [FunLike F R S]
    [RingHomClass F R S] (φ : F) (w v : R) :
    (blindSys w v).map φ = blindSys (φ w) (φ v) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [blindSys]

/-- The inverse of `blindSys 256 (256 ^ 96)` over `ZMod 389`. -/
def blindW : Matrix (Fin 8) (Fin 8) (ZMod 389) :=
  !![205, 133, 188, 293, 20, 20, 76, 76;
     174, 35, 302, 0, 150, 150, 20, 20;
     205, 120, 297, 157, 265, 257, 13, 272;
     64, 249, 285, 19, 378, 152, 309, 331;
     88, 380, 343, 241, 4, 4, 213, 213;
     263, 277, 144, 112, 164, 164, 187, 187;
     137, 65, 122, 4, 19, 374, 315, 277;
     141, 111, 179, 334, 2, 228, 119, 96]

theorem blindSys_mul_blindW : blindSys (256 : ZMod 389) (256 ^ 96) * blindW = 1 := by
  decide +kernel

/-- **The system of the block is invertible at `ζ`.** -/
theorem det_blindSys_ne_zero :
    (blindSys (ZMod.stdAddChar (1 : ZMod 97) : ℂ)
      ((ZMod.stdAddChar (1 : ZMod 97) : ℂ) ^ 96)).det ≠ 0 := by
  have hN : (blindSys (X : ℤ[X]) (X ^ 96)).map (eval₂ (Int.castRingHom (ZMod 389)) 256) *
      blindW = 1 := by
    rw [← coe_eval₂RingHom, blindSys_map]
    simp only [coe_eval₂RingHom, eval₂_X, eval₂_X_pow]
    exact blindSys_mul_blindW
  have h := det_aeval_ne_zero (p := 97) (q := 389) (z := 256) _ (by decide) (by decide +kernel)
    blindW hN
  rw [blindSys_map] at h
  simpa using h

/-- **The eight equations force the first two unknowns to vanish.** -/
theorem blindSys_solve {w v y₀ y₁ y₂ y₃ z₀ z₁ z₂ z₃ : ℂ} (hdet : (blindSys w v).det ≠ 0)
    (e1 : v ^ 2 * y₀ + v ^ 4 * y₁ = z₀ * w ^ 2 + z₁ * w ^ 4)
    (e2 : v ^ 8 * y₀ + v ^ 5 * y₁ = z₀ * w ^ 8 + z₁ * w ^ 5)
    (e3 : v ^ 5 * y₂ + v ^ 8 * y₃ = z₂ * w ^ 5 + z₃ * w ^ 8)
    (e4 : v ^ 10 * y₂ + v ^ 3 * y₃ = z₂ * w ^ 10 + z₃ * w ^ 3)
    (e5 : v ^ 3 * y₂ + v ^ 10 * y₃ = z₀ * w ^ 0 + z₁ * w ^ 2)
    (e6 : v ^ 0 * y₀ + v ^ 2 * y₁ = z₂ * w ^ 3 + z₃ * w ^ 10)
    (e7 : v ^ 13 * y₂ + v ^ 0 * y₃ = z₀ * w ^ 5 + z₁ * w ^ 3)
    (e8 : v ^ 5 * y₀ + v ^ 3 * y₁ = z₂ * w ^ 13 + z₃ * w ^ 0) : y₀ = 0 ∧ y₁ = 0 := by
  have hx : blindSys w v *ᵥ ![y₀, y₁, y₂, y₃, z₀, z₁, z₂, z₃] = 0 := by
    ext r
    fin_cases r <;> simp [blindSys, mulVec, dotProduct, Fin.sum_univ_succ]
    · linear_combination e1
    · linear_combination e2
    · linear_combination e3
    · linear_combination e4
    · linear_combination e5
    · linear_combination e6
    · linear_combination e7
    · linear_combination e8
  have h0 := Matrix.eq_zero_of_mulVec_eq_zero hdet hx
  exact ⟨congrFun h0 0, congrFun h0 1⟩

example : (0 : ℂ) = 0 ∧ (0 : ℂ) = 0 :=
  blindSys_solve (y₂ := 0) (y₃ := 0) (z₀ := 0) (z₁ := 0) (z₂ := 0) (z₃ := 0)
    det_blindSys_ne_zero (by simp) (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
    (by simp)

end Transformer.Grokking.FlatSpectrum.TablePart.E6
