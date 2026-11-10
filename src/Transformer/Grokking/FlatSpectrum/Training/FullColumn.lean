module

public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
public import Mathlib.LinearAlgebra.Matrix.ToLin
public import Transformer.Grokking.FlatSpectrum.Training.Lifted

/-!
# Independent columns at width `min(m, n)`: the certificate holds

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD10.
Let `M W = 0` for the stacked factors `W = [U; V]` and `M = [[λ, G], [Gᴴ, λ]]` with `λ > 0`:
`G V = −λ U` and `Gᴴ U = −λ V`. A vector killed by `U` is then killed by `V`, so if the columns
of `W` are independent, so are those of `U`, and of `V`. If moreover the width is at least the
number of rows of `U`, then `U` is square and onto (`surjective_mulVec_of_injective`), and
`G Gᴴ U = λ² U` gives `‖Gᴴ a‖ = λ ‖a‖` for every `a`
(`frobeniusSq_conjTranspose_mul_of_surjective`); the same holds for `V` and `G`. Either way
`⟨G, a bᴴ⟩² ≤ λ² ‖a‖² ‖b‖²`, which makes `M` positive semidefinite
(`lift_posSemidef_of_pairing_sq_le`). So at width at least `min(m, n)` a stationary point with
independent columns carries the certificate of WD2 (`lift_posSemidef_of_injective`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open scoped ComplexOrder

variable {m n k p q : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype p] [Fintype q]

/-- A matrix with independent columns and at least as many columns as rows is onto. -/
lemma surjective_mulVec_of_injective {S : Matrix p k ℂ} (hinj : Function.Injective S.mulVec)
    (hcard : Fintype.card p ≤ Fintype.card k) : Function.Surjective S.mulVec := by
  have hinj' : Function.Injective S.mulVecLin := hinj
  have hle := LinearMap.finrank_le_finrank_of_injective hinj'
  rw [Module.finrank_fintype_fun_eq_card, Module.finrank_fintype_fun_eq_card] at hle
  exact (LinearMap.injective_iff_surjective_of_finrank_eq_finrank (by
    rw [Module.finrank_fintype_fun_eq_card, Module.finrank_fintype_fun_eq_card]
    exact le_antisymm hle hcard)).mp hinj'

example := surjective_mulVec_of_injective (S := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (fun a b h => by simpa using h) le_rfl

/-- If `P T = −λ S` and `Pᴴ S = −λ T` with `S` onto, then `P Pᴴ = λ²`: `‖Pᴴ A‖² = λ² ‖A‖²` for
every one-column `A`. -/
lemma frobeniusSq_conjTranspose_mul_of_surjective {lam : ℝ} {P : Matrix p q ℂ}
    {S : Matrix p k ℂ} {T : Matrix q k ℂ} (h1 : P * T + lam • S = 0)
    (h2 : Pᴴ * S + lam • T = 0) (hsurj : Function.Surjective S.mulVec) (A : Matrix p Unit ℂ) :
    frobeniusSq (Pᴴ * A) = lam ^ 2 * frobeniusSq A := by
  obtain ⟨c, hc⟩ := hsurj fun i => A i ()
  have hA : A = S * replicateCol Unit c := by
    rw [← replicateCol_mulVec, hc]
    rfl
  have e1 : Pᴴ * S = -(lam • T) := eq_neg_of_add_eq_zero_left h2
  have e2 : P * T = -(lam • S) := eq_neg_of_add_eq_zero_left h1
  have key : P * (Pᴴ * A) = lam ^ 2 • A := by
    rw [hA, ← Matrix.mul_assoc Pᴴ, e1, Matrix.neg_mul, Matrix.smul_mul, Matrix.mul_neg,
      Matrix.mul_smul, ← Matrix.mul_assoc, e2, Matrix.neg_mul, Matrix.smul_mul, smul_neg,
      neg_neg, smul_smul, sq]
  rw [← pairing_self, pairing_mul_left, conjTranspose_conjTranspose, key, pairing_smul_right,
    pairing_self]

/-- `P = S = 1`, `T = −1`, `λ = 1`. -/
example := frobeniusSq_conjTranspose_mul_of_surjective (lam := 1)
  (P := (1 : Matrix Unit Unit ℂ)) (S := 1) (T := -1) (by simp) (by simp)
  (fun y => ⟨y, by simp⟩) 1

variable [DecidableEq m] [DecidableEq n]

/-- **The lifted gradient is positive semidefinite when the operator norm of `G` is at most
`λ`** (`review/flat_spectrum.md`, item WD10), in the form `⟨G, A Bᴴ⟩² ≤ λ² ‖A‖² ‖B‖²` for
one-column `A`, `B`. -/
theorem lift_posSemidef_of_pairing_sq_le {lam : ℝ} (hlam : 0 ≤ lam) {G : Matrix m n ℂ}
    (h : ∀ (A : Matrix m Unit ℂ) (B : Matrix n Unit ℂ),
      pairing G (A * Bᴴ) ^ 2 ≤ lam ^ 2 * (frobeniusSq A * frobeniusSq B)) :
    (lift lam G).PosSemidef := by
  refine posSemidef_of_pairing_nonneg (lift_isHermitian lam G) fun W => ?_
  obtain ⟨A, B, rfl⟩ := exists_eq_stack W
  rw [pairing_stack_lift]
  have ha := frobeniusSq_nonneg A
  have hb := frobeniusSq_nonneg B
  have hp : |pairing G (A * Bᴴ)| ≤ lam * ((frobeniusSq A + frobeniusSq B) / 2) :=
    abs_le_of_sq_le_sq ((h A B).trans (by
      nlinarith [mul_nonneg (sq_nonneg lam) (sq_nonneg (frobeniusSq A - frobeniusSq B))]))
      (by positivity)
  have h2 := neg_abs_le (pairing G (A * Bᴴ))
  linarith

/-- `G = 1` and `λ = 1`: Cauchy–Schwarz. -/
example := lift_posSemidef_of_pairing_sq_le (lam := 1) zero_le_one
  (G := (1 : Matrix (Fin 2) (Fin 2) ℂ)) fun A B => by
    rw [pairing_comm, pairing_mul_right, conjTranspose_conjTranspose, Matrix.one_mul, one_pow,
      one_mul]
    exact pairing_sq_le A B

/-- **A stationary point with independent columns at width `min(m, n)` carries the
certificate** (`review/flat_spectrum.md`, item WD10): let `M [U; V] = 0` with `λ > 0`, let the
columns of `[U; V]` be independent and let the width be at least the number of rows of `U` or
of `V`. Then `M = [[λ, G], [Gᴴ, λ]]` is positive semidefinite. -/
theorem lift_posSemidef_of_injective {lam : ℝ} (hlam : 0 < lam) {G : Matrix m n ℂ}
    {U : Matrix m k ℂ} {V : Matrix n k ℂ} (hstat : lift lam G * stack U V = 0)
    (hinj : Function.Injective (stack U V).mulVec)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    (lift lam G).PosSemidef := by
  rw [lift_mul_stack, stack] at hstat
  obtain ⟨h1, h2⟩ := fromRows_inj (hstat.trans (fromRows_zero (R := ℂ)).symm)
  have hW : ∀ z : k → ℂ, U *ᵥ z = 0 → V *ᵥ z = 0 → z = 0 := fun z hU hV => by
    refine hinj ?_
    rw [mulVec_zero, stack, fromRows_mulVec, hU, hV, Sum.elim_zero_zero]
  have hUV : ∀ z : k → ℂ, U *ᵥ z = 0 → V *ᵥ z = 0 := fun z hz => by
    have h := congrArg (· *ᵥ z) h2
    simp only [add_mulVec, ← mulVec_mulVec, hz, mulVec_zero, zero_add, smul_mulVec,
      zero_mulVec] at h
    exact (smul_eq_zero.mp h).resolve_left hlam.ne'
  have hVU : ∀ z : k → ℂ, V *ᵥ z = 0 → U *ᵥ z = 0 := fun z hz => by
    have h := congrArg (· *ᵥ z) h1
    simp only [add_mulVec, ← mulVec_mulVec, hz, mulVec_zero, zero_add, smul_mulVec,
      zero_mulVec] at h
    exact (smul_eq_zero.mp h).resolve_left hlam.ne'
  refine lift_posSemidef_of_pairing_sq_le hlam.le fun A B => ?_
  have hp : pairing G (A * Bᴴ) = pairing (G * B) A := by
    rw [pairing_comm, pairing_mul_right, conjTranspose_conjTranspose, pairing_comm]
  rcases hcard with hc | hc
  · have hsurj : Function.Surjective U.mulVec := surjective_mulVec_of_injective (fun a b hab =>
      sub_eq_zero.mp (hW _ (by rw [mulVec_sub, hab, sub_self])
        (hUV _ (by rw [mulVec_sub, hab, sub_self])))) hc
    rw [hp, pairing_mul_left]
    calc pairing B (Gᴴ * A) ^ 2 ≤ frobeniusSq B * frobeniusSq (Gᴴ * A) := pairing_sq_le _ _
      _ = lam ^ 2 * (frobeniusSq A * frobeniusSq B) := by
          rw [frobeniusSq_conjTranspose_mul_of_surjective h1 h2 hsurj A]
          ring
  · have hsurj : Function.Surjective V.mulVec := surjective_mulVec_of_injective (fun a b hab =>
      sub_eq_zero.mp (hW _ (hVU _ (by rw [mulVec_sub, hab, sub_self]))
        (by rw [mulVec_sub, hab, sub_self]))) hc
    have hB := frobeniusSq_conjTranspose_mul_of_surjective (P := Gᴴ) h2
      (by rwa [conjTranspose_conjTranspose]) hsurj B
    rw [conjTranspose_conjTranspose] at hB
    rw [hp]
    calc pairing (G * B) A ^ 2 ≤ frobeniusSq (G * B) * frobeniusSq A := pairing_sq_le _ _
      _ = lam ^ 2 * (frobeniusSq A * frobeniusSq B) := by
          rw [hB]
          ring

/-- `G = −1`, `U = V = 1` of size `1 × 1` and `λ = 1`: `M = [[1, −1], [−1, 1]]` and
`W = [1; 1]`. -/
example := lift_posSemidef_of_injective (lam := 1) one_pos
  (G := (-1 : Matrix (Fin 1) (Fin 1) ℂ)) (U := 1) (V := 1)
  (by rw [lift_mul_stack, ← stack_zero]; congr 1 <;> simp)
  (fun a b h => by
    rw [stack, fromRows_mulVec, fromRows_mulVec, one_mulVec, one_mulVec] at h
    exact funext fun i => congrFun h (Sum.inl i))
  (Or.inl le_rfl)

end Transformer.Grokking.FlatSpectrum.Training
