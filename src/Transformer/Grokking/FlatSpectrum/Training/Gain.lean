module

public import Transformer.Grokking.FlatSpectrum.Training.Basic

/-!
# One step of a perturbed linear recursion gains on an indefinite quadratic form

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD4, the
algebra of its proof. `S` is Hermitian and invertible, `T = a − η S` with `a ≥ 1`, `η ≥ 0` and
`⟨W, T W⟩ ≥ 0`, and `Q Z = −⟨Z, S Z⟩`. One step `Z ← (T − η E) Z` of the recursion with a
perturbation `E` gives, with `W = S Z`,

    Q ((T − η E) Z) = a² Q Z + a η ‖W‖² + η ⟨W, T W⟩ + 2 η ⟨T W, E Z⟩ − η² ⟨E Z, S E Z⟩

(`gain_identity`). As `‖Z‖ ≤ ‖S⁻¹‖ ‖W‖`, the last two terms are at most `η ‖W‖²` once `‖E‖` is
small, so `Q ((T − η E) Z) ≥ a² Q Z` (`gain_step`).

The lemmas `gainWitness_*` are a `1 × 1` witness of the hypotheses, with a negative direction:
`S = S⁻¹ = −1`, `a = 2`, `η = 1`, so `T = 3`, with `‖T‖ = 3`, `‖S‖ = ‖S⁻¹‖ = 1` and `E = 0`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {p q : Type*} [Fintype p] [Fintype q] [DecidableEq p]

/-- The quadratic form `−⟨Z, S Z⟩` after one step `Z ← (T − η E) Z`, for `T = a − η S`. -/
lemma gain_identity {S : Matrix p p ℂ} (hS : S.IsHermitian) (a η : ℝ) (E : Matrix p p ℂ)
    (Z : Matrix p q ℂ) :
    -pairing ((a • 1 - η • S - η • E) * Z) (S * ((a • 1 - η • S - η • E) * Z)) =
      a ^ 2 * -pairing Z (S * Z) + a * η * frobeniusSq (S * Z) +
        η * pairing (S * Z) ((a • 1 - η • S) * (S * Z)) +
        2 * η * pairing ((a • 1 - η • S) * (S * Z)) (E * Z) -
        η ^ 2 * pairing (E * Z) (S * (E * Z)) := by
  have hsym (A B : Matrix p q ℂ) : pairing A (S * B) = pairing (S * A) B := by
    rw [pairing_mul_left, hS.eq]
  have hcomm : S * (a • 1 - η • S) = (a • 1 - η • S) * S := by
    simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one,
      Matrix.one_mul]
  generalize hT : a • (1 : Matrix p p ℂ) - η • S = T at hcomm ⊢
  have hSTZ : S * (T * Z) = T * (S * Z) := by
    rw [← Matrix.mul_assoc, hcomm, Matrix.mul_assoc]
  have hTmul (A : Matrix p q ℂ) : T * A = a • A - η • (S * A) := by
    rw [← hT, Matrix.sub_mul, Matrix.smul_mul, Matrix.one_mul, Matrix.smul_mul]
  have hZ' : (T - η • E) * Z = T * Z - η • (E * Z) := by
    rw [Matrix.sub_mul, Matrix.smul_mul]
  have hSZ' : S * ((T - η • E) * Z) = T * (S * Z) - η • (S * (E * Z)) := by
    rw [hZ', Matrix.mul_sub, Matrix.mul_smul, hSTZ]
  have h1 : pairing (T * Z) (S * (E * Z)) = pairing (T * (S * Z)) (E * Z) := by
    rw [hsym, hSTZ]
  have h2 : pairing (E * Z) (T * (S * Z)) = pairing (T * (S * Z)) (E * Z) := pairing_comm _ _
  have h3 : pairing Z (T * (S * Z)) = a * pairing Z (S * Z) - η * frobeniusSq (S * Z) := by
    rw [hTmul (S * Z), pairing_sub_right, pairing_smul_right, pairing_smul_right, hsym Z (S * Z),
      pairing_self]
  have h4 : pairing (T * Z) (T * (S * Z)) =
      a * pairing Z (T * (S * Z)) - η * pairing (S * Z) (T * (S * Z)) := by
    rw [hTmul Z, pairing_sub_left, pairing_smul_left, pairing_smul_left]
  rw [hSZ', hZ']
  simp only [pairing_sub_left, pairing_sub_right, pairing_smul_left, pairing_smul_right]
  rw [h1, h2, h4, h3]
  ring

variable {S R E : Matrix p p ℂ} {a η τ σ s ε : ℝ}

/-- **One-step gain** (`review/flat_spectrum.md`, item WD4): for `T = a − η S` with `a ≥ 1`,
`η ≥ 0` and `⟨W, T W⟩ ≥ 0`, and a perturbation `E` with
`2 ‖T‖ ‖E‖ ‖S⁻¹‖ + η ‖S‖ ‖E‖² ‖S⁻¹‖² ≤ 1`, the step `Z ← (T − η E) Z` multiplies
`−⟨Z, S Z⟩` by at least `a²`. -/
theorem gain_step (hS : S.IsHermitian) (hR : R * S = 1) (ha : 1 ≤ a) (hη : 0 ≤ η)
    (hT : ∀ W : Matrix p q ℂ, 0 ≤ pairing W ((a • 1 - η • S) * W)) (hτ0 : 0 ≤ τ) (hσ0 : 0 ≤ σ)
    (hs0 : 0 ≤ s) (hε0 : 0 ≤ ε) (hτ : frobeniusSq (a • (1 : Matrix p p ℂ) - η • S) ≤ τ ^ 2)
    (hσ : frobeniusSq R ≤ σ ^ 2) (hs : frobeniusSq S ≤ s ^ 2) (hE : frobeniusSq E ≤ ε ^ 2)
    (hsmall : 2 * τ * ε * σ + η * s * ε ^ 2 * σ ^ 2 ≤ 1) (Z : Matrix p q ℂ) :
    a ^ 2 * -pairing Z (S * Z) ≤
      -pairing ((a • 1 - η • S - η • E) * Z) (S * ((a • 1 - η • S - η • E) * Z)) := by
  rw [gain_identity hS]
  have ht := hT (S * Z)
  generalize a • (1 : Matrix p p ℂ) - η • S = T at hτ ht ⊢
  generalize hW : S * Z = W at ht ⊢
  have hw0 := frobeniusSq_nonneg W
  have hZ : frobeniusSq Z ≤ σ ^ 2 * frobeniusSq W := by
    have h : Z = R * W := by rw [← hW, ← Matrix.mul_assoc, hR, Matrix.one_mul]
    calc frobeniusSq Z = frobeniusSq (R * W) := by rw [← h]
      _ ≤ frobeniusSq R * frobeniusSq W := frobeniusSq_mul_le R W
      _ ≤ σ ^ 2 * frobeniusSq W := mul_le_mul_of_nonneg_right hσ hw0
  have hTW : frobeniusSq (T * W) ≤ τ ^ 2 * frobeniusSq W :=
    (frobeniusSq_mul_le T W).trans (mul_le_mul_of_nonneg_right hτ hw0)
  have hEZ : frobeniusSq (E * Z) ≤ ε ^ 2 * (σ ^ 2 * frobeniusSq W) :=
    (frobeniusSq_mul_le E Z).trans (mul_le_mul hE hZ (frobeniusSq_nonneg Z) (sq_nonneg ε))
  have hSEZ : frobeniusSq (S * (E * Z)) ≤ s ^ 2 * (ε ^ 2 * (σ ^ 2 * frobeniusSq W)) :=
    (frobeniusSq_mul_le S (E * Z)).trans
      (mul_le_mul hs hEZ (frobeniusSq_nonneg _) (sq_nonneg s))
  have hx : |pairing (T * W) (E * Z)| ≤ τ * ε * σ * frobeniusSq W := by
    refine abs_pairing_le (by positivity) ?_
    calc frobeniusSq (T * W) * frobeniusSq (E * Z)
        ≤ τ ^ 2 * frobeniusSq W * (ε ^ 2 * (σ ^ 2 * frobeniusSq W)) :=
          mul_le_mul hTW hEZ (frobeniusSq_nonneg _) (by positivity)
      _ = (τ * ε * σ * frobeniusSq W) ^ 2 := by ring
  have hy : |pairing (E * Z) (S * (E * Z))| ≤ s * ε ^ 2 * σ ^ 2 * frobeniusSq W := by
    refine abs_pairing_le (by positivity) ?_
    calc frobeniusSq (E * Z) * frobeniusSq (S * (E * Z))
        ≤ ε ^ 2 * (σ ^ 2 * frobeniusSq W) * (s ^ 2 * (ε ^ 2 * (σ ^ 2 * frobeniusSq W))) :=
          mul_le_mul hEZ hSEZ (frobeniusSq_nonneg _) (by positivity)
      _ = (s * ε ^ 2 * σ ^ 2 * frobeniusSq W) ^ 2 := by ring
  have hx' := neg_abs_le (pairing (T * W) (E * Z))
  have hy' := le_abs_self (pairing (E * Z) (S * (E * Z)))
  have e1 : 0 ≤ (a - 1) * η * frobeniusSq W := mul_nonneg (mul_nonneg (by linarith) hη) hw0
  have e2 : 0 ≤ η * pairing W (T * W) := mul_nonneg hη ht
  have e3 : 0 ≤ 2 * η * (τ * ε * σ * frobeniusSq W + pairing (T * W) (E * Z)) :=
    mul_nonneg (by positivity) (by linarith)
  have e4 : 0 ≤ η ^ 2 * (s * ε ^ 2 * σ ^ 2 * frobeniusSq W - pairing (E * Z) (S * (E * Z))) :=
    mul_nonneg (sq_nonneg η) (by linarith)
  have e5 : 0 ≤ η * frobeniusSq W * (1 - (2 * τ * ε * σ + η * s * ε ^ 2 * σ ^ 2)) :=
    mul_nonneg (mul_nonneg hη hw0) (by linarith)
  linarith

lemma gainWitness_step :
    (2 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ) - (1 : ℝ) • (-1) = (3 : ℝ) • 1 := by module

lemma gainWitness_hermitian : (-1 : Matrix (Fin 1) (Fin 1) ℂ).IsHermitian := by
  simp [IsHermitian]

lemma gainWitness_inverse : (-1 : Matrix (Fin 1) (Fin 1) ℂ) * -1 = 1 := by simp

lemma gainWitness_nonneg (W : Matrix (Fin 1) (Fin 1) ℂ) :
    0 ≤ pairing W (((2 : ℝ) • 1 - (1 : ℝ) • (-1)) * W) := by
  rw [gainWitness_step, Matrix.smul_mul, Matrix.one_mul, pairing_smul_right, pairing_self]
  exact mul_nonneg (by norm_num) (frobeniusSq_nonneg W)

lemma gainWitness_bound :
    frobeniusSq ((2 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ) - (1 : ℝ) • (-1)) ≤ (3 : ℝ) ^ 2 := by
  rw [gainWitness_step, frobeniusSq_smul]
  simp [frobeniusSq]

lemma gainWitness_one : frobeniusSq (-1 : Matrix (Fin 1) (Fin 1) ℂ) ≤ (1 : ℝ) ^ 2 := by
  simp [frobeniusSq]

lemma gainWitness_zero : frobeniusSq (0 : Matrix (Fin 1) (Fin 1) ℂ) ≤ (0 : ℝ) ^ 2 := by
  simp [frobeniusSq]

example := gain_identity gainWitness_hermitian 2 1 (0 : Matrix (Fin 1) (Fin 1) ℂ)
  (1 : Matrix (Fin 1) (Fin 1) ℂ)

example := gain_step gainWitness_hermitian gainWitness_inverse one_le_two zero_le_one
  gainWitness_nonneg (by norm_num) zero_le_one zero_le_one le_rfl gainWitness_bound
  gainWitness_one gainWitness_one gainWitness_zero (by norm_num) 1

end Transformer.Grokking.FlatSpectrum.Training
