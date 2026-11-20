module

public import Mathlib.Analysis.Complex.Basic
public import Mathlib.Data.Matrix.Block
public import Mathlib.LinearAlgebra.Complex.Module

/-!
# Complex-linear projections among the real ones

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`). The selected model of
`research/main_model/README.txt` uses bias-free complex-linear projections, stored for
real GEMMs as `[[A,-B],[B,A]]` (`research/main_model/phase_score_native/src/layout.rs`,
`matrix`): `W_Q`, `W_K`, `W_V`, `W_O`, `W_g`, `W_u` and `W_F`. The restriction is stated in
`research/complex_transformer/README.md`: "The actual restriction is `Wz`, whereas a
general real projection is `Wz+V conjugate(z)`."

The complex structure is a real-linear map `J` with `J (J z) = -z`, multiplication by
`i`; the model realizes it per head on its half-split coordinates as
`(re, im) ↦ (-im, re)`. A real-linear map `L` is complex-linear when
`L (J z) = J' (L z)`. Every real-linear map splits into a complex-linear part
`(L - J' L J)/2` and a conjugate-linear part `(L + J' L J)/2`, the `V conjugate(z)` term
(`complexPart_add_antilinearPart`, `complexPart_comm`, `antilinearPart_anticomm`), and the
splitting is unique (`complexPart_unique`). The block matrices `[[A,-B],[B,A]]` are
exactly the real matrices that commute with `[[0,-1],[1,0]]` (`fromBlocks_comm_iff`).
-/

@[expose] public section

namespace Transformer.ComplexProjection

variable {V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]

/-- The complex-linear part `(L z - J' (L (J z)))/2` of a real-linear map. -/
noncomputable def complexPart (J : V →ₗ[ℝ] V) (J' : W →ₗ[ℝ] W) (L : V →ₗ[ℝ] W) :
    V →ₗ[ℝ] W :=
  (2 : ℝ)⁻¹ • (L - J' ∘ₗ L ∘ₗ J)

/-- The conjugate-linear part `(L z + J' (L (J z)))/2` of a real-linear map: the
`V conjugate(z)` term of `research/complex_transformer/README.md`. -/
noncomputable def antilinearPart (J : V →ₗ[ℝ] V) (J' : W →ₗ[ℝ] W) (L : V →ₗ[ℝ] W) :
    V →ₗ[ℝ] W :=
  (2 : ℝ)⁻¹ • (L + J' ∘ₗ L ∘ₗ J)

theorem complexPart_apply (J : V →ₗ[ℝ] V) (J' : W →ₗ[ℝ] W) (L : V →ₗ[ℝ] W) (z : V) :
    complexPart J J' L z = (2 : ℝ)⁻¹ • (L z - J' (L (J z))) :=
  rfl

theorem antilinearPart_apply (J : V →ₗ[ℝ] V) (J' : W →ₗ[ℝ] W) (L : V →ₗ[ℝ] W) (z : V) :
    antilinearPart J J' L z = (2 : ℝ)⁻¹ • (L z + J' (L (J z))) :=
  rfl

/-- Every real projection is its complex-linear part plus its conjugate-linear part:
`Wz + V conjugate(z)`. -/
theorem complexPart_add_antilinearPart (J : V →ₗ[ℝ] V) (J' : W →ₗ[ℝ] W) (L : V →ₗ[ℝ] W) :
    complexPart J J' L + antilinearPart J J' L = L := by
  ext z
  simp only [LinearMap.add_apply, complexPart_apply, antilinearPart_apply]
  module

variable {J : V →ₗ[ℝ] V} {J' : W →ₗ[ℝ] W}

/-- The complex part commutes with the complex structures. -/
theorem complexPart_comm (L : V →ₗ[ℝ] W) (hJ : ∀ z, J (J z) = -z)
    (hJ' : ∀ x, J' (J' x) = -x) (z : V) :
    complexPart J J' L (J z) = J' (complexPart J J' L z) := by
  simp only [complexPart_apply, map_smul, map_sub, hJ, hJ', map_neg]
  module

/-- The conjugate-linear part anticommutes with them. -/
theorem antilinearPart_anticomm (L : V →ₗ[ℝ] W) (hJ : ∀ z, J (J z) = -z)
    (hJ' : ∀ x, J' (J' x) = -x) (z : V) :
    antilinearPart J J' L (J z) = -J' (antilinearPart J J' L z) := by
  simp only [antilinearPart_apply, map_smul, map_add, hJ, hJ', map_neg]
  module

/-- A complex-linear map is its own complex part. -/
theorem complexPart_eq_self {L : V →ₗ[ℝ] W} (hJ' : ∀ x, J' (J' x) = -x)
    (hL : ∀ z, L (J z) = J' (L z)) : complexPart J J' L = L := by
  ext z
  simp only [complexPart_apply, hL, hJ']
  module

/-- A complex-linear map has no conjugate-linear part. -/
theorem antilinearPart_eq_zero {L : V →ₗ[ℝ] W} (hJ' : ∀ x, J' (J' x) = -x)
    (hL : ∀ z, L (J z) = J' (L z)) : antilinearPart J J' L = 0 := by
  ext z
  simp only [antilinearPart_apply, hL, hJ', add_neg_cancel, smul_zero, LinearMap.zero_apply]

/-- A conjugate-linear map is its own conjugate-linear part. -/
theorem antilinearPart_eq_self {L : V →ₗ[ℝ] W} (hJ' : ∀ x, J' (J' x) = -x)
    (hL : ∀ z, L (J z) = -J' (L z)) : antilinearPart J J' L = L := by
  ext z
  simp only [antilinearPart_apply, hL, map_neg, hJ', neg_neg]
  module

/-- The splitting is unique: a complex-linear `P` plus a conjugate-linear `A` has
complex part `P` and conjugate-linear part `A`. -/
theorem complexPart_unique (hJ' : ∀ x, J' (J' x) = -x) {P A : V →ₗ[ℝ] W}
    (hP : ∀ z, P (J z) = J' (P z)) (hA : ∀ z, A (J z) = -J' (A z)) :
    complexPart J J' (P + A) = P ∧ antilinearPart J J' (P + A) = A := by
  constructor <;> ext z <;>
    simp only [complexPart_apply, antilinearPart_apply, LinearMap.add_apply, hP, hA, map_add,
      map_neg, hJ', neg_neg] <;> module

/-- Multiplication by `i` on `ℂ` is a complex structure. -/
theorem mulLeft_I_mulLeft_I (z : ℂ) :
    LinearMap.mulLeft ℝ Complex.I (LinearMap.mulLeft ℝ Complex.I z) = -z := by
  simp only [LinearMap.mulLeft_apply, ← mul_assoc, Complex.I_mul_I, neg_one_mul]

/-- Conjugation on `ℂ` is all conjugate-linear part. -/
example : antilinearPart (LinearMap.mulLeft ℝ Complex.I) (LinearMap.mulLeft ℝ Complex.I)
    Complex.conjAe.toLinearMap = Complex.conjAe.toLinearMap :=
  antilinearPart_eq_self mulLeft_I_mulLeft_I fun z => by simp

/-- Multiplication by a complex number is all complex part. -/
example (c : ℂ) : complexPart (LinearMap.mulLeft ℝ Complex.I) (LinearMap.mulLeft ℝ Complex.I)
    (LinearMap.mulLeft ℝ c) = LinearMap.mulLeft ℝ c :=
  complexPart_eq_self mulLeft_I_mulLeft_I fun z => by simp only [LinearMap.mulLeft_apply]; ring

/-- `c z + conj z` splits into `c z` and `conj z`. -/
example (c : ℂ) :
    complexPart (LinearMap.mulLeft ℝ Complex.I) (LinearMap.mulLeft ℝ Complex.I)
        (LinearMap.mulLeft ℝ c + Complex.conjAe.toLinearMap) = LinearMap.mulLeft ℝ c ∧
      antilinearPart (LinearMap.mulLeft ℝ Complex.I) (LinearMap.mulLeft ℝ Complex.I)
        (LinearMap.mulLeft ℝ c + Complex.conjAe.toLinearMap) = Complex.conjAe.toLinearMap :=
  complexPart_unique mulLeft_I_mulLeft_I (fun z => by simp only [LinearMap.mulLeft_apply]; ring)
    fun z => by simp

open Matrix in
/-- The storage `[[A,-B],[B,A]]` of `layout.rs`: a real block matrix commutes with the
complex structures `[[0,-1],[1,0]]` exactly when its blocks have this form. -/
theorem fromBlocks_comm_iff {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m]
    [DecidableEq n] (P Q R S : Matrix m n ℝ) :
    fromBlocks P Q R S * fromBlocks (0 : Matrix n n ℝ) (-1) 1 0 =
        fromBlocks (0 : Matrix m m ℝ) (-1) 1 0 * fromBlocks P Q R S ↔ S = P ∧ Q = -R := by
  simp only [fromBlocks_multiply, Matrix.mul_zero, Matrix.zero_mul, Matrix.mul_one,
    Matrix.one_mul, Matrix.mul_neg, Matrix.neg_mul, add_zero, zero_add, fromBlocks_inj, neg_inj]
  constructor
  · rintro ⟨h1, -, h3, -⟩
    exact ⟨h3, h1⟩
  · rintro ⟨rfl, rfl⟩
    exact ⟨rfl, rfl, rfl, rfl⟩

example (A B : Matrix (Fin 2) (Fin 3) ℝ) :
    Matrix.fromBlocks A (-B) B A * Matrix.fromBlocks (0 : Matrix (Fin 3) (Fin 3) ℝ) (-1) 1 0 =
      Matrix.fromBlocks (0 : Matrix (Fin 2) (Fin 2) ℝ) (-1) 1 0 * Matrix.fromBlocks A (-B) B A :=
  (fromBlocks_comm_iff A (-B) B A).2 ⟨rfl, rfl⟩

end Transformer.ComplexProjection
