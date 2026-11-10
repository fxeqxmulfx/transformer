module

public import Transformer.Grokking.FlatSpectrum.Training.Gain

/-!
# Bounded solutions of a perturbed linear recursion have no negative start

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD4, the
algebra of its proof. With `S` Hermitian and invertible, `T = a − η S`, `⟨W, T W⟩ ≥ 0` and
`Q Z = −⟨Z, S Z⟩` as in `gain_step`, a solution of `Z_{j+1} = (T − η F_j) Z_j` with small
`F_j` has `Q Z_j ≥ a^{2j} Q Z_0` (`gain_iterate`). As `Q Z_j ≤ ‖S‖ ‖Z_j‖²`, a bounded solution
with `a > 1` has `Q Z_0 ≤ 0` (`pairing_nonneg_of_bounded`). The smallness needed of the `F_j`
depends only on `S`, `a` and `η` (`exists_pos_forall_pairing_nonneg`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {p q : Type*} [Fintype p] [Fintype q] [DecidableEq p]

variable {S R : Matrix p p ℂ} {a η τ σ s ε : ℝ}

/-- **The gain along the recursion** (`review/flat_spectrum.md`, item WD4): if
`Z_{j+1} = (T − η E_j) Z_j` with every `E_j` small as in `gain_step`, then
`−⟨Z_j, S Z_j⟩ ≥ a^{2j} (−⟨Z_0, S Z_0⟩)`. -/
theorem gain_iterate (hS : S.IsHermitian) (hR : R * S = 1) (ha : 1 ≤ a) (hη : 0 ≤ η)
    (hT : ∀ W : Matrix p q ℂ, 0 ≤ pairing W ((a • 1 - η • S) * W)) (hτ0 : 0 ≤ τ) (hσ0 : 0 ≤ σ)
    (hs0 : 0 ≤ s) (hε0 : 0 ≤ ε) (hτ : frobeniusSq (a • (1 : Matrix p p ℂ) - η • S) ≤ τ ^ 2)
    (hσ : frobeniusSq R ≤ σ ^ 2) (hs : frobeniusSq S ≤ s ^ 2)
    (hsmall : 2 * τ * ε * σ + η * s * ε ^ 2 * σ ^ 2 ≤ 1) {F : ℕ → Matrix p p ℂ}
    (hF : ∀ j, frobeniusSq (F j) ≤ ε ^ 2) {Z : ℕ → Matrix p q ℂ}
    (hZ : ∀ j, Z (j + 1) = (a • 1 - η • S - η • F j) * Z j) (j : ℕ) :
    (a ^ 2) ^ j * -pairing (Z 0) (S * Z 0) ≤ -pairing (Z j) (S * Z j) := by
  induction j with
  | zero => simp
  | succ j ih =>
    calc (a ^ 2) ^ (j + 1) * -pairing (Z 0) (S * Z 0)
        = a ^ 2 * ((a ^ 2) ^ j * -pairing (Z 0) (S * Z 0)) := by ring
      _ ≤ a ^ 2 * -pairing (Z j) (S * Z j) := mul_le_mul_of_nonneg_left ih (sq_nonneg a)
      _ ≤ -pairing (Z (j + 1)) (S * Z (j + 1)) := by
          rw [hZ j]
          exact gain_step hS hR ha hη hT hτ0 hσ0 hs0 hε0 hτ hσ hs (hF j) hsmall (Z j)

/-- **A bounded solution has no negative start** (`review/flat_spectrum.md`, item WD4): under
the hypotheses of `gain_iterate` with `a > 1`, a bounded solution has `⟨Z_0, S Z_0⟩ ≥ 0`.
Otherwise `−⟨Z_j, S Z_j⟩ ≤ ‖S‖ ‖Z_j‖²` would grow like `a^{2j}`. -/
theorem pairing_nonneg_of_bounded (hS : S.IsHermitian) (hR : R * S = 1) (ha : 1 < a)
    (hη : 0 ≤ η) (hT : ∀ W : Matrix p q ℂ, 0 ≤ pairing W ((a • 1 - η • S) * W)) (hτ0 : 0 ≤ τ)
    (hσ0 : 0 ≤ σ) (hs0 : 0 ≤ s) (hε0 : 0 ≤ ε)
    (hτ : frobeniusSq (a • (1 : Matrix p p ℂ) - η • S) ≤ τ ^ 2) (hσ : frobeniusSq R ≤ σ ^ 2)
    (hs : frobeniusSq S ≤ s ^ 2) (hsmall : 2 * τ * ε * σ + η * s * ε ^ 2 * σ ^ 2 ≤ 1)
    {F : ℕ → Matrix p p ℂ} (hF : ∀ j, frobeniusSq (F j) ≤ ε ^ 2) {Z : ℕ → Matrix p q ℂ}
    (hZ : ∀ j, Z (j + 1) = (a • 1 - η • S - η • F j) * Z j) {B : ℝ}
    (hB : ∀ j, frobeniusSq (Z j) ≤ B) : 0 ≤ pairing (Z 0) (S * Z 0) := by
  by_contra hneg
  have hQ : 0 < -pairing (Z 0) (S * Z 0) := by linarith [not_le.mp hneg]
  obtain ⟨j, hj⟩ := pow_unbounded_of_one_lt (s * B / -pairing (Z 0) (S * Z 0))
    (one_lt_pow₀ ha two_ne_zero)
  have h1 := (div_lt_iff₀ hQ).mp hj
  have h2 := gain_iterate hS hR ha.le hη hT hτ0 hσ0 hs0 hε0 hτ hσ hs hsmall hF hZ j
  have h3 : |pairing (Z j) (S * Z j)| ≤ s * frobeniusSq (Z j) := by
    refine abs_pairing_le (mul_nonneg hs0 (frobeniusSq_nonneg _)) ?_
    calc frobeniusSq (Z j) * frobeniusSq (S * Z j)
        ≤ frobeniusSq (Z j) * (s ^ 2 * frobeniusSq (Z j)) :=
          mul_le_mul_of_nonneg_left ((frobeniusSq_mul_le S (Z j)).trans
            (mul_le_mul_of_nonneg_right hs (frobeniusSq_nonneg _))) (frobeniusSq_nonneg _)
      _ = (s * frobeniusSq (Z j)) ^ 2 := by ring
  have h4 := neg_abs_le (pairing (Z j) (S * Z j))
  have h5 := mul_le_mul_of_nonneg_left (hB j) hs0
  linarith

example := gain_iterate gainWitness_hermitian gainWitness_inverse one_le_two zero_le_one
  gainWitness_nonneg (by norm_num) zero_le_one zero_le_one le_rfl gainWitness_bound
  gainWitness_one gainWitness_one (by norm_num) (F := fun _ => 0) (fun _ => gainWitness_zero)
  (Z := fun _ => 0) (fun _ => by simp) 0

/-- The recursion `Z ← 3 Z` of the witness is bounded only from `Z_0 = 0`. -/
example := pairing_nonneg_of_bounded gainWitness_hermitian gainWitness_inverse one_lt_two
  zero_le_one gainWitness_nonneg (by norm_num) zero_le_one zero_le_one le_rfl gainWitness_bound
  gainWitness_one gainWitness_one (by norm_num) (F := fun _ => 0) (fun _ => gainWitness_zero)
  (Z := fun _ => 0) (fun _ => by simp) (B := 0) (fun _ => by simp [frobeniusSq])

/-- For nonnegative constants some `ε > 0` has `2 τ ε σ + η s ε² σ² ≤ 1`. -/
lemma exists_pos_small {τ σ s η : ℝ} (hτ : 0 ≤ τ) (hσ : 0 ≤ σ) (hs : 0 ≤ s) (hη : 0 ≤ η) :
    ∃ ε : ℝ, 0 < ε ∧ 2 * τ * ε * σ + η * s * ε ^ 2 * σ ^ 2 ≤ 1 := by
  have hX : 0 ≤ 2 * τ * σ + η * s * σ ^ 2 := by positivity
  obtain ⟨ε, hε⟩ : ∃ ε : ℝ, ε = 1 / (1 + (2 * τ * σ + η * s * σ ^ 2)) := ⟨_, rfl⟩
  have hε0 : 0 < ε := by rw [hε]; positivity
  have hεD : ε * (1 + (2 * τ * σ + η * s * σ ^ 2)) = 1 := by
    rw [hε]
    exact one_div_mul_cancel (by positivity)
  have hε1 : ε ≤ 1 := by nlinarith [mul_nonneg hε0.le hX]
  have h1 : 0 ≤ ε - ε ^ 2 := by nlinarith [mul_nonneg hε0.le (sub_nonneg.mpr hε1)]
  have h2 : 0 ≤ η * s * σ ^ 2 * (ε - ε ^ 2) := mul_nonneg (by positivity) h1
  exact ⟨ε, hε0, by nlinarith⟩

example := exists_pos_small (τ := 1) (σ := 1) (s := 1) (η := 1) zero_le_one zero_le_one
  zero_le_one zero_le_one

/-- **Bounded solutions near an invertible Hermitian `S`** (`review/flat_spectrum.md`, item
WD4): for `T = a − η S` with `a > 1`, `η > 0` and `⟨Z, T Z⟩ ≥ 0`, there is `ε > 0` such that
every bounded solution of `Z_{j+1} = (T − η F_j) Z_j` with all `‖F_j‖_S2 ≤ ε` has
`⟨Z_0, S Z_0⟩ ≥ 0`. -/
theorem exists_pos_forall_pairing_nonneg {S : Matrix p p ℂ} (hS : S.IsHermitian)
    (hdet : S.det ≠ 0) {a η : ℝ} (ha : 1 < a) (hη : 0 < η)
    (hT : ∀ Z : Matrix p q ℂ, 0 ≤ pairing Z ((a • 1 - η • S) * Z)) :
    ∃ ε : ℝ, 0 < ε ∧ ∀ (F : ℕ → Matrix p p ℂ) (Z : ℕ → Matrix p q ℂ) (B : ℝ),
      (∀ j, frobeniusSq (F j) ≤ ε ^ 2) → (∀ j, Z (j + 1) = (a • 1 - η • S - η • F j) * Z j) →
      (∀ j, frobeniusSq (Z j) ≤ B) → 0 ≤ pairing (Z 0) (S * Z 0) := by
  have hR : S⁻¹ * S = 1 := nonsing_inv_mul S (isUnit_iff_ne_zero.mpr hdet)
  obtain ⟨τ, hτ0, hτ⟩ : ∃ τ : ℝ, 0 ≤ τ ∧ frobeniusSq (a • (1 : Matrix p p ℂ) - η • S) ≤ τ ^ 2 :=
    ⟨_, Real.sqrt_nonneg _, (Real.sq_sqrt (frobeniusSq_nonneg _)).ge⟩
  obtain ⟨σ, hσ0, hσ⟩ : ∃ σ : ℝ, 0 ≤ σ ∧ frobeniusSq S⁻¹ ≤ σ ^ 2 :=
    ⟨_, Real.sqrt_nonneg _, (Real.sq_sqrt (frobeniusSq_nonneg _)).ge⟩
  obtain ⟨s, hs0, hs⟩ : ∃ s : ℝ, 0 ≤ s ∧ frobeniusSq S ≤ s ^ 2 :=
    ⟨_, Real.sqrt_nonneg _, (Real.sq_sqrt (frobeniusSq_nonneg _)).ge⟩
  obtain ⟨ε, hε0, hsmall⟩ := exists_pos_small hτ0 hσ0 hs0 hη.le
  exact ⟨ε, hε0, fun F Z B hF hZ hB =>
    pairing_nonneg_of_bounded hS hR ha hη.le hT hτ0 hσ0 hs0 hε0.le hτ hσ hs hsmall hF hZ hB⟩


example := exists_pos_forall_pairing_nonneg (q := Fin 1) gainWitness_hermitian (by simp)
  one_lt_two one_pos gainWitness_nonneg

end Transformer.Grokking.FlatSpectrum.Training
