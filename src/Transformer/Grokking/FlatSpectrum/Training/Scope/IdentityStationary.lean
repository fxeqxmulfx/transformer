module

public import Transformer.Grokking.FlatSpectrum.Training.SquaredLoss
public import Transformer.Grokking.FlatSpectrum.Training.Scope.Balanced

/-!
# The identity kernel: the stationary points

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD16,
statements 1 and 2; the item "Width" of WD11.

For the squared loss on all cells of the identity of any size, with factors of any width and
`λ > 0`:

* a pair `(U, V)` is stationary exactly when `V = U` and `U Uᴴ U = (1 − λ) U`
  (`stationary_identity_iff`): the two equations `V = U Vᴴ V + λ U` and `U = V Uᴴ U + λ V`
  (`eq_add_of_stationary_identity`) and `Uᴴ U = Vᴴ V` (WD14) give
  `(U − V) (Uᴴ U + 1 + λ) = 0`;
* so the product `X` of a stationary point is positive semidefinite with `X² = (1 − λ) X`
  (`mul_self_of_stationary_identity`): for `λ < 1` it is `1 − λ` times an orthogonal
  projection;
* for `λ ≥ 1` the only stationary point is the origin (`eq_zero_of_stationary_identity`).

The witnesses for `λ = 3/4` are `½ · 1` and `halfDiag`, which is `½ · diag(1, 0)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology
open scoped ComplexOrder

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m] {lam : ℝ} {U V : Matrix m k ℂ}

omit [Fintype m] [DecidableEq m] in
/-- The mask of all cells keeps a matrix. -/
lemma mask_univ (X : Matrix m m ℂ) : mask Set.univ X = X := by
  ext i j
  simp [mask]

/-- The two equations of a stationary point of the identity: `V = U Vᴴ V + λ U` and
`U = V Uᴴ U + λ V`. -/
lemma eq_add_of_stationary_identity
    (hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0) :
    V = U * (Vᴴ * V) + lam • U ∧ U = V * (Uᴴ * U) + lam • V := by
  rw [lift_mul_stack, mask_univ] at hstat
  have h1 : (U * Vᴴ - 1) * V + lam • U = 0 := by
    ext i j
    have := congrFun (congrFun hstat (Sum.inl i)) j
    simpa [stack] using this
  have h2 : (U * Vᴴ - 1)ᴴ * U + lam • V = 0 := by
    ext i j
    have := congrFun (congrFun hstat (Sum.inr i)) j
    simpa [stack] using this
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc, sub_add_eq_add_sub, sub_eq_zero] at h1
  rw [conjTranspose_sub, conjTranspose_mul, conjTranspose_conjTranspose, conjTranspose_one,
    Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc, sub_add_eq_add_sub, sub_eq_zero] at h2
  exact ⟨h1.symm, h2.symm⟩

/-- **A stationary point of the identity has equal factors** (`review/flat_spectrum.md`, item
WD16, statement 1): for `λ > 0`, `V = U` and `U Uᴴ U = (1 − λ) U`. -/
theorem eq_and_mul_of_stationary_identity (hlam : 0 < lam)
    (hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0) :
    V = U ∧ U * Uᴴ * U = (1 - lam) • U := by
  obtain ⟨h1, h2⟩ := eq_add_of_stationary_identity hstat
  rw [← conjTranspose_mul_self_eq_of_stationary hlam.ne' hstat] at h1
  have hD : (U - V) * (Uᴴ * U) + (1 + lam) • (U - V) = 0 := by
    rw [Matrix.sub_mul, eq_sub_of_add_eq h1.symm, eq_sub_of_add_eq h2.symm]
    module
  have hpair : frobeniusSq ((U - V) * Uᴴ) + (1 + lam) * frobeniusSq (U - V) = 0 := by
    have h := congrArg (fun X => pairing X (U - V)) hD
    simp only [pairing_add_left, pairing_smul_left, pairing_zero_left, pairing_self] at h
    rwa [← Matrix.mul_assoc, pairing_mul_right, pairing_self] at h
  have hz : U - V = 0 := eq_zero_of_frobeniusSq_eq_zero (by
    nlinarith [frobeniusSq_nonneg ((U - V) * Uᴴ), frobeniusSq_nonneg (U - V)])
  obtain rfl : V = U := (sub_eq_zero.1 hz).symm
  refine ⟨rfl, ?_⟩
  rw [Matrix.mul_assoc, eq_sub_of_add_eq h2.symm]
  module

/-- **Equal factors with `U Uᴴ U = (1 − λ) U` are stationary** (`review/flat_spectrum.md`,
item WD16, statement 1). -/
theorem stationary_identity_of_mul (h : U * Uᴴ * U = (1 - lam) • U) :
    lift lam (mask Set.univ (U * Uᴴ - 1)) * stack U U = 0 := by
  have hH : (U * Uᴴ - 1)ᴴ = U * Uᴴ - 1 := by
    rw [conjTranspose_sub, conjTranspose_mul, conjTranspose_conjTranspose, conjTranspose_one]
  have h0 : (U * Uᴴ - 1) * U + lam • U = 0 := by
    rw [Matrix.sub_mul, Matrix.one_mul, h]
    module
  rw [lift_mul_stack, mask_univ, hH, h0, stack_zero]

/-- **The stationary points of the identity** (`review/flat_spectrum.md`, item WD16, statement
1), for `λ > 0`: the pairs `(U, U)` with `U Uᴴ U = (1 − λ) U`. -/
theorem stationary_identity_iff (hlam : 0 < lam) :
    lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0 ↔
      V = U ∧ U * Uᴴ * U = (1 - lam) • U := by
  refine ⟨eq_and_mul_of_stationary_identity hlam, ?_⟩
  rintro ⟨rfl, h⟩
  exact stationary_identity_of_mul h

/-- `Re tr (U Uᴴ) = ‖U‖²`. -/
lemma pairing_mul_conjTranspose_one (U : Matrix m k ℂ) :
    pairing (U * Uᴴ) (1 : Matrix m m ℂ) = frobeniusSq U := by
  rw [pairing_mul_right, conjTranspose_conjTranspose, Matrix.one_mul, pairing_self]

omit [DecidableEq m] in
/-- With `U Uᴴ U = (1 − λ) U`: `‖U Uᴴ‖² = (1 − λ) ‖U‖²`. -/
lemma frobeniusSq_mul_conjTranspose_of_mul (h : U * Uᴴ * U = (1 - lam) • U) :
    frobeniusSq (U * Uᴴ) = (1 - lam) * frobeniusSq U := by
  rw [← pairing_self, pairing_mul_right, conjTranspose_conjTranspose, h, pairing_smul_right,
    pairing_self]

/-- **The product of a stationary point of the identity** (`review/flat_spectrum.md`, item
WD16, statement 2), for `λ > 0`: positive semidefinite, with `X² = (1 − λ) X`. -/
theorem mul_self_of_stationary_identity (hlam : 0 < lam)
    (hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0) :
    (U * Vᴴ).PosSemidef ∧ (U * Vᴴ) * (U * Vᴴ) = (1 - lam) • (U * Vᴴ) := by
  obtain ⟨rfl, h⟩ := eq_and_mul_of_stationary_identity hlam hstat
  exact ⟨posSemidef_self_mul_conjTranspose V, by rw [← Matrix.mul_assoc, h, Matrix.smul_mul]⟩

/-- **For `λ ≥ 1` the only stationary point of the identity is the origin**
(`review/flat_spectrum.md`, item WD16, statement 2). -/
theorem eq_zero_of_stationary_identity (hlam : 1 ≤ lam)
    (hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0) : U = 0 ∧ V = 0 := by
  obtain ⟨rfl, h⟩ := eq_and_mul_of_stationary_identity (by linarith) hstat
  have h2 := frobeniusSq_mul_conjTranspose_of_mul h
  have hz : V * Vᴴ = 0 := eq_zero_of_frobeniusSq_eq_zero (by
    nlinarith [frobeniusSq_nonneg (V * Vᴴ), frobeniusSq_nonneg V])
  have hs : frobeniusSq V = 0 := by
    rw [← pairing_mul_conjTranspose_one, hz, pairing_zero_left]
  exact ⟨eq_zero_of_frobeniusSq_eq_zero hs, eq_zero_of_frobeniusSq_eq_zero hs⟩

/-- `(t · 1) (t · 1)ᴴ = t² · 1` for a real `t`. -/
lemma smul_one_mul_conjTranspose (t : ℝ) :
    (t • (1 : Matrix m m ℂ)) * (t • (1 : Matrix m m ℂ))ᴴ = (t * t) • 1 := by
  rw [conjTranspose_smul, conjTranspose_one, star_trivial, Matrix.smul_mul, Matrix.mul_smul,
    Matrix.one_mul, smul_smul]

/-! Witnesses for `λ = 3/4`, where `1 − λ = 1/4`: the pairs `U = V = ½ · 1` and
`U = V = ½ · diag(1, 0)` of the `2 × 2` identity, and the origin for `λ = 1`. -/

/-- `½ · 1` satisfies `U Uᴴ U = ¼ U`. -/
lemma half_smul_one_mul :
    ((1 / 2 : ℝ) • (1 : Matrix m m ℂ)) * ((1 / 2 : ℝ) • (1 : Matrix m m ℂ))ᴴ *
        ((1 / 2 : ℝ) • (1 : Matrix m m ℂ)) =
      (1 - 3 / 4 : ℝ) • ((1 / 2 : ℝ) • (1 : Matrix m m ℂ)) := by
  rw [smul_one_mul_conjTranspose, Matrix.smul_mul, Matrix.one_mul]
  norm_num

/-- `½ · diag(1, 0)`: a factor of a stationary point of the `2 × 2` identity whose product has
rank one. -/
noncomputable def halfDiag : Matrix (Fin 2) (Fin 2) ℂ := (1 / 2 : ℝ) • Matrix.diagonal ![1, 0]

lemma halfDiag_mul : halfDiag * halfDiagᴴ * halfDiag = (1 - 3 / 4 : ℝ) • halfDiag := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [halfDiag, Matrix.mul_apply, Matrix.diagonal, map_ofNat]
  norm_num

example := eq_add_of_stationary_identity
  (stationary_identity_of_mul (half_smul_one_mul (m := Fin 2)))

example := eq_and_mul_of_stationary_identity (lam := 3 / 4) (by norm_num)
  (stationary_identity_of_mul (half_smul_one_mul (m := Fin 2)))

example := (stationary_identity_iff (lam := 3 / 4) (U := halfDiag) (V := halfDiag)
  (by norm_num)).2 ⟨rfl, halfDiag_mul⟩

example := frobeniusSq_mul_conjTranspose_of_mul halfDiag_mul

example := mul_self_of_stationary_identity (lam := 3 / 4) (by norm_num)
  (stationary_identity_of_mul halfDiag_mul)

example := eq_zero_of_stationary_identity (lam := 1) le_rfl
  (stationary_identity_of_mul (U := (0 : Matrix (Fin 2) (Fin 1) ℂ)) (by simp))

end Transformer.Grokking.FlatSpectrum.Training.Scope
