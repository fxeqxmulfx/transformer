module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityOriginRun

/-!
# The identity kernel: the run from equal factors

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD16,
statement 7; the item "Width" of WD11.

For the squared loss on all cells of the identity of any size:

* a step from a pair `(U, U)` gives the pair `(U', U')` with
  `U' = (1 + η (1 − λ)) U − η U Uᴴ U` (`symStep`, `decayStep_identity_self`), so the run from
  `V₀ = U₀` has `V_t = U_t` at every step (`decayRun_identity_self`);
* `U' = M U` with `M = (1 + η (1 − λ)) − η U Uᴴ`, so `U' U'ᴴ = M (U Uᴴ) M`, and what commutes
  with `U₀ U₀ᴴ` commutes with `U_t U_tᴴ` at every step (`commute_iterate_symStep`);
* a Hermitian idempotent `E` that commutes with `U₀ U₀ᴴ` passes through the run:
  `E U_t` is the run from `E U₀` (`mul_iterate_symStep`), and `‖E U₀‖ ≤ ‖U₀‖`
  (`frobeniusSq_mul_le_of_idempotent`).

The witness of the examples is `shearFactor`, the matrix `[[1, 1], [0, 1]]`, which is invertible
and not normal.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m] {η lam : ℝ}
  {U₀ U : Matrix m k ℂ}

local notation "G₁" => fun X : Matrix m m ℂ => mask Set.univ (X - 1)

/-- The step of a pair of equal factors on the identity (`review/flat_spectrum.md`, item WD16,
statement 7): `U ↦ (1 + η (1 − λ)) U − η U Uᴴ U`. -/
noncomputable def symStep (η lam : ℝ) (U : Matrix m k ℂ) : Matrix m k ℂ :=
  (1 + η * (1 - lam)) • U - η • (U * Uᴴ * U)

/-- **A step from equal factors gives equal factors** (`review/flat_spectrum.md`, item WD16,
statement 7). -/
lemma decayStep_identity_self (η lam : ℝ) (U : Matrix m k ℂ) :
    decayStep G₁ η lam (U, U) = (symStep η lam U, symStep η lam U) := by
  simp only [decayStep, mask_univ, symStep]
  rw [conjTranspose_sub, conjTranspose_mul, conjTranspose_conjTranspose, conjTranspose_one,
    Matrix.sub_mul, Matrix.one_mul]
  refine Prod.ext ?_ ?_ <;> (simp only; module)

/-- **The run from equal factors has equal factors at every step** (`review/flat_spectrum.md`,
item WD16, statement 7). -/
lemma decayRun_identity_self (η lam : ℝ) (U₀ : Matrix m k ℂ) (t : ℕ) :
    decayRun G₁ η lam U₀ U₀ t = ((symStep η lam)^[t] U₀, (symStep η lam)^[t] U₀) := by
  induction t with
  | zero => rfl
  | succ t ih => rw [decayRun_succ, ih, decayStep_identity_self, Function.iterate_succ_apply']

omit [DecidableEq m] in
/-- A Hermitian idempotent that commutes with `U Uᴴ` passes through the step. -/
lemma mul_symStep {E : Matrix m m ℂ} (hE : E * E = E) (hH : Eᴴ = E)
    (hc : Commute E (U * Uᴴ)) (η lam : ℝ) :
    E * symStep η lam U = symStep η lam (E * U) := by
  have hc' : E * (U * Uᴴ) = U * Uᴴ * E := hc
  have h : E * U * (E * U)ᴴ * (E * U) = E * (U * Uᴴ * U) := by
    rw [conjTranspose_mul, hH]
    calc E * U * (Uᴴ * E) * (E * U) = E * (U * Uᴴ * E) * (E * U) := by
          simp only [Matrix.mul_assoc]
      _ = E * (E * (U * Uᴴ)) * (E * U) := by rw [← hc']
      _ = E * E * (U * Uᴴ * E * U) := by simp only [Matrix.mul_assoc]
      _ = E * E * (E * (U * Uᴴ) * U) := by rw [← hc']
      _ = E * E * E * (U * Uᴴ * U) := by simp only [Matrix.mul_assoc]
      _ = E * (U * Uᴴ * U) := by rw [hE, hE]
  rw [symStep, symStep, h, Matrix.mul_sub, Matrix.mul_smul, Matrix.mul_smul]

/-- What commutes with `U Uᴴ` commutes with the Gram matrix after a step. -/
lemma commute_symStep {A : Matrix m m ℂ} (h : Commute A (U * Uᴴ)) (η lam : ℝ) :
    Commute A (symStep η lam U * (symStep η lam U)ᴴ) := by
  have hM : symStep η lam U = ((1 + η * (1 - lam)) • (1 : Matrix m m ℂ) - η • (U * Uᴴ)) * U := by
    rw [symStep, Matrix.sub_mul, Matrix.smul_mul, Matrix.smul_mul, Matrix.one_mul]
  have hMH : ((1 + η * (1 - lam)) • (1 : Matrix m m ℂ) - η • (U * Uᴴ))ᴴ =
      (1 + η * (1 - lam)) • (1 : Matrix m m ℂ) - η • (U * Uᴴ) := by
    rw [conjTranspose_sub, conjTranspose_smul, conjTranspose_smul, conjTranspose_one,
      conjTranspose_mul, conjTranspose_conjTranspose, star_trivial, star_trivial]
  have hAM : Commute A ((1 + η * (1 - lam)) • (1 : Matrix m m ℂ) - η • (U * Uᴴ)) :=
    ((Commute.one_right A).smul_right _).sub_right (h.smul_right _)
  rw [hM, conjTranspose_mul, hMH, Matrix.mul_assoc, ← Matrix.mul_assoc U]
  exact hAM.mul_right ((h.mul_right hAM))

/-- What commutes with `U₀ U₀ᴴ` commutes with the Gram matrix at every step. -/
lemma commute_iterate_symStep {A : Matrix m m ℂ} (h : Commute A (U₀ * U₀ᴴ)) (η lam : ℝ)
    (t : ℕ) :
    Commute A ((symStep η lam)^[t] U₀ * ((symStep η lam)^[t] U₀)ᴴ) := by
  induction t with
  | zero => exact h
  | succ t ih =>
    rw [Function.iterate_succ_apply']
    exact commute_symStep ih η lam

/-- A Hermitian idempotent that commutes with `U₀ U₀ᴴ` passes through the run. -/
lemma mul_iterate_symStep {E : Matrix m m ℂ} (hE : E * E = E) (hH : Eᴴ = E)
    (hc : Commute E (U₀ * U₀ᴴ)) (η lam : ℝ) (t : ℕ) :
    E * (symStep η lam)^[t] U₀ = (symStep η lam)^[t] (E * U₀) := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ← ih]
    exact mul_symStep hE hH (commute_iterate_symStep hc η lam t) η lam

omit [DecidableEq m] in
/-- A Hermitian idempotent does not increase the norm. -/
lemma frobeniusSq_mul_le_of_idempotent {E : Matrix m m ℂ} (hE : E * E = E) (hH : Eᴴ = E)
    (U : Matrix m k ℂ) : frobeniusSq (E * U) ≤ frobeniusSq U := by
  have hz : pairing (E * U) (U - E * U) = 0 := by
    rw [pairing_mul_left, hH, Matrix.mul_sub, ← Matrix.mul_assoc, hE, sub_self,
      pairing_zero_right]
  have h : frobeniusSq U = frobeniusSq (E * U) + frobeniusSq (U - E * U) := by
    conv_lhs => rw [show U = E * U + (U - E * U) by abel]
    rw [frobeniusSq_add, hz]
    ring
  linarith [frobeniusSq_nonneg (U - E * U)]

/-- The start of the examples: `[[1, 1], [0, 1]]`, which is not normal. -/
def shearFactor : Matrix (Fin 2) (Fin 2) ℂ := !![1, 1; 0, 1]

lemma shearFactor_mul : shearFactor * !![1, -1; 0, 1] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [shearFactor, Matrix.mul_apply, Fin.sum_univ_two]

lemma frobeniusSq_shearFactor : frobeniusSq shearFactor = 3 := by
  simp [frobeniusSq, shearFactor, Fin.sum_univ_two]
  norm_num

example := mul_symStep (E := 1) (U := shearFactor) (Matrix.one_mul 1) conjTranspose_one
  (Commute.one_left _) (1 / 200) (1 / 2)

example := commute_symStep (A := 1) (U := shearFactor) (Commute.one_left _) (1 / 200) (1 / 2)

example := commute_iterate_symStep (A := 1) (U₀ := shearFactor) (Commute.one_left _) (1 / 200)
  (1 / 2) 3

example := mul_iterate_symStep (E := 1) (U₀ := shearFactor) (Matrix.one_mul 1) conjTranspose_one
  (Commute.one_left _) (1 / 200) (1 / 2) 3

example := frobeniusSq_mul_le_of_idempotent (E := 1) (Matrix.one_mul 1) conjTranspose_one
  shearFactor

example := decayRun_identity_self (1 / 200) (1 / 2) shearFactor 3

end Transformer.Grokking.FlatSpectrum.Training.Scope
