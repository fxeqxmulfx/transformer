module

public import Transformer.Grokking.FlatSpectrum.Training.Certificate
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityStationary
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignRun

/-!
# The identity kernel: the trained product and the decayed minima

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD16,
statements 3 and 4; the item "Width" of WD11.

For the squared loss on all cells of the identity of any size:

* for `0 ≤ λ ≤ 1` the only minimizer of `½ ‖X − 1‖² + λ ‖X‖_S1` is `(1 − λ) · 1`
  (`isNuclearMin_identity_iff`): by `Re tr X ≤ ‖X‖_S1` the value at `X` is at least
  `½ ‖X − (1 − λ) 1‖²` above the value at `(1 − λ) · 1`;
* so the product of a decayed minimum is `(1 − λ) · 1`
  (`mul_conjTranspose_eq_of_isDecayedMin_identity`), and for `λ < 1` no factorization of a
  width below the size of the identity is a decayed minimum
  (`not_isDecayedMin_identity_of_card_lt`);
* a stationary point is a decayed minimum exactly when its product is `(1 − λ) · 1`
  (`isDecayedMin_identity_iff_of_stationary`): there the gradient is `−λ · 1`, and the lifted
  matrix `[[λ, c], [c, λ]]` is positive semidefinite when `|c| ≤ λ`
  (`lift_smul_one_posSemidef`);
* for `λ ≥ 1` the origin is a decayed minimum (`isDecayedMin_identity_zero`).

This extends to every size what WD10 and statement 5 of WD14 state for the `2 × 2` identity.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology
open scoped ComplexOrder

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m] {lam : ℝ} {U V : Matrix m k ℂ}

/-- `[[λ, c], [c, λ]]` is positive semidefinite when `|c| ≤ λ`. -/
lemma lift_smul_one_posSemidef {c : ℝ} (hc : |c| ≤ lam) :
    (lift lam (c • (1 : Matrix m m ℂ))).PosSemidef := by
  refine posSemidef_of_pairing_nonneg (lift_isHermitian lam _) fun W => ?_
  obtain ⟨A, B, rfl⟩ := exists_eq_stack W
  rw [pairing_stack_lift, pairing_smul_left, pairing_one_mul_conjTranspose,
    frobeniusSq_add_frobeniusSq]
  obtain ⟨h1, h2⟩ := abs_le.1 hc
  nlinarith [mul_nonneg (by linarith : 0 ≤ lam + c) (frobeniusSq_nonneg (A + B)),
    mul_nonneg (by linarith : 0 ≤ lam - c) (frobeniusSq_nonneg (A - B))]

/-- **The trained product of the identity is unique** (`review/flat_spectrum.md`, item WD16,
statement 3): for `0 ≤ λ ≤ 1` a minimizer of `½ ‖X − 1‖² + λ ‖X‖_S1` is `(1 − λ) · 1`. -/
theorem eq_smul_one_of_isNuclearMin_identity (h0 : 0 ≤ lam) (h1 : lam ≤ 1) {X : Matrix m m ℂ}
    (hX : IsNuclearMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam X) : X = (1 - lam) • 1 := by
  have hs : √(1 - lam) * √(1 - lam) = 1 - lam := Real.mul_self_sqrt (by linarith)
  have h := hX ((1 - lam) • 1)
  have h2 := mul_le_mul_of_nonneg_left (nuclearNorm_le_of_eq_mul
    (√(1 - lam) • (1 : Matrix m m ℂ)) (√(1 - lam) • (1 : Matrix m m ℂ))
    (by rw [smul_one_mul_conjTranspose, hs])) h0
  have htr : pairing X 1 ≤ nuclearNorm X := by
    have h := norm_trace_le_nuclearNorm (isContraction_one (n := m)) X
    rw [conjTranspose_one, Matrix.one_mul] at h
    have e : pairing X 1 = X.trace.re := by
      rw [pairing, Matrix.mul_one, trace_conjTranspose, Complex.star_def, Complex.conj_re]
    rw [e]
    exact (Complex.re_le_norm _).trans h
  have h3 := mul_le_mul_of_nonneg_left htr h0
  have h5 : frobeniusSq (X - (1 - lam) • (1 : Matrix m m ℂ)) = frobeniusSq X -
      2 * ((1 - lam) * pairing X 1) + (1 - lam) ^ 2 * frobeniusSq (1 : Matrix m m ℂ) := by
    rw [frobeniusSq_sub, pairing_smul_right, frobeniusSq_smul]
  rw [fitLoss, fitLoss, mask_univ, mask_univ,
    show (1 - lam) • (1 : Matrix m m ℂ) - 1 = (-lam) • (1 : Matrix m m ℂ) by module,
    frobeniusSq_smul, frobeniusSq_sub] at h
  rw [frobeniusSq_smul, sq √(1 - lam), hs] at h2
  refine sub_eq_zero.1 (eq_zero_of_frobeniusSq_eq_zero (le_antisymm ?_ (frobeniusSq_nonneg _)))
  rw [h5]
  linarith

/-- **The product of a decayed minimum of the identity** (`review/flat_spectrum.md`, item
WD16, statement 3) is `(1 − λ) · 1`, for `0 ≤ λ ≤ 1` and every width. -/
theorem mul_conjTranspose_eq_of_isDecayedMin_identity (h0 : 0 ≤ lam) (h1 : lam ≤ 1)
    (h : IsDecayedMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam U V) :
    U * Vᴴ = (1 - lam) • 1 :=
  eq_smul_one_of_isNuclearMin_identity h0 h1 (h.isNuclearMin h0)

/-- A product of factors narrower than the identity is not `c · 1` with `c ≠ 0`. -/
lemma mul_conjTranspose_ne_smul_one (hk : Fintype.card k < Fintype.card m) {c : ℝ}
    (hc : c ≠ 0) (U V : Matrix m k ℂ) : U * Vᴴ ≠ c • (1 : Matrix m m ℂ) := fun h => by
  have h1 : U * (c⁻¹ • Vᴴ) = 1 := by
    rw [Matrix.mul_smul, h, smul_smul, inv_mul_cancel₀ hc, one_smul]
  have h2 := Matrix.rank_mul_le_left U (c⁻¹ • Vᴴ)
  rw [h1, Matrix.rank_one] at h2
  exact absurd (h2.trans U.rank_le_card_width) (not_le.2 hk)

/-- **No factorization narrower than the identity is a decayed minimum**
(`review/flat_spectrum.md`, item WD16, statement 3), for `0 ≤ λ < 1`. -/
theorem not_isDecayedMin_identity_of_card_lt (h0 : 0 ≤ lam) (h1 : lam < 1)
    (hk : Fintype.card k < Fintype.card m) (U V : Matrix m k ℂ) :
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam U V := fun h =>
  mul_conjTranspose_ne_smul_one hk (sub_pos.2 h1).ne' U V
    (mul_conjTranspose_eq_of_isDecayedMin_identity h0 h1.le h)

/-- **A stationary point of the identity is a decayed minimum exactly when its product is
`(1 − λ) · 1`** (`review/flat_spectrum.md`, item WD16, statement 4), for `0 ≤ λ ≤ 1`. -/
theorem isDecayedMin_identity_iff_of_stationary (h0 : 0 ≤ lam) (h1 : lam ≤ 1)
    (hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0) :
    IsDecayedMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam U V ↔
      U * Vᴴ = (1 - lam) • 1 := by
  refine ⟨mul_conjTranspose_eq_of_isDecayedMin_identity h0 h1, fun hX => ?_⟩
  refine isDecayedMin_of_posSemidef (G := fun X => mask Set.univ (X - 1))
    (fitLoss_first_order Set.univ 1) hstat ?_
  rw [mask_univ, hX, show (1 - lam) • (1 : Matrix m m ℂ) - 1 = (-lam) • (1 : Matrix m m ℂ) by
    module]
  exact lift_smul_one_posSemidef (by rw [abs_neg, abs_of_nonneg h0])

/-- **The trained product of the identity** (`review/flat_spectrum.md`, item WD16, statement
3), for `0 ≤ λ ≤ 1`: `(1 − λ) · 1` and no other matrix. -/
theorem isNuclearMin_identity_iff (h0 : 0 ≤ lam) (h1 : lam ≤ 1) {X : Matrix m m ℂ} :
    IsNuclearMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam X ↔ X = (1 - lam) • 1 := by
  refine ⟨eq_smul_one_of_isNuclearMin_identity h0 h1, ?_⟩
  rintro rfl
  have hs : √(1 - lam) * √(1 - lam) = 1 - lam := Real.mul_self_sqrt (by linarith)
  have hm : (√(1 - lam) • (1 : Matrix m m ℂ)) * (√(1 - lam) • (1 : Matrix m m ℂ))ᴴ =
      (1 - lam) • 1 := by rw [smul_one_mul_conjTranspose, hs]
  have hstat := stationary_identity_of_mul (lam := lam) (U := √(1 - lam) • (1 : Matrix m m ℂ))
    (by rw [hm, Matrix.smul_mul, Matrix.one_mul])
  rw [← hm]
  exact ((isDecayedMin_identity_iff_of_stationary h0 h1 hstat).2 hm).isNuclearMin h0

/-- **For `λ ≥ 1` the origin is a decayed minimum of the identity**
(`review/flat_spectrum.md`, item WD16, statement 4). -/
theorem isDecayedMin_identity_zero (hlam : 1 ≤ lam) :
    IsDecayedMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam (0 : Matrix m k ℂ) 0 := by
  refine isDecayedMin_of_posSemidef (G := fun X => mask Set.univ (X - 1))
    (fitLoss_first_order Set.univ 1) (stationary_identity_of_mul (by simp)) ?_
  rw [mask_univ, show (0 : Matrix m k ℂ) * (0 : Matrix m k ℂ)ᴴ - 1 =
    (-1 : ℝ) • (1 : Matrix m m ℂ) by simp]
  exact lift_smul_one_posSemidef (by rw [abs_neg, abs_one]; exact hlam)

/-! Witnesses for `λ = 3/4`: the pair `U = V = ½ · 1` of the `2 × 2` identity is a decayed
minimum, and the stationary pair `U = V = ½ · diag(1, 0)` is not one. -/

lemma halfDiag_mul_conjTranspose_ne :
    halfDiag * halfDiagᴴ ≠ (1 - 3 / 4 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := fun h => by
  have := congrFun (congrFun h 1) 1
  simp [halfDiag, Matrix.mul_apply, Matrix.diagonal] at this
  norm_num at this

example := lift_smul_one_posSemidef (m := Fin 2) (lam := 1) (c := -1) (by norm_num)

example : IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (3 / 4)
    ((1 / 2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) ((1 / 2 : ℝ) • 1) :=
  (isDecayedMin_identity_iff_of_stationary (by norm_num) (by norm_num)
    (stationary_identity_of_mul half_smul_one_mul)).2 (by
      rw [smul_one_mul_conjTranspose]
      norm_num)

example : ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (3 / 4)
    halfDiag halfDiag := fun h =>
  halfDiag_mul_conjTranspose_ne
    (mul_conjTranspose_eq_of_isDecayedMin_identity (by norm_num) (by norm_num) h)

example := (isNuclearMin_identity_iff (m := Fin 2) (lam := 3 / 4) (by norm_num)
  (by norm_num)).2 rfl

example := eq_smul_one_of_isNuclearMin_identity (m := Fin 2) (lam := 3 / 4) (by norm_num)
  (by norm_num) ((isNuclearMin_identity_iff (by norm_num) (by norm_num)).2 rfl)

example := mul_conjTranspose_ne_smul_one (m := Fin 2) (k := Fin 1) (by simp)
  (one_ne_zero (α := ℝ)) 0 0

example := not_isDecayedMin_identity_of_card_lt (m := Fin 2) (k := Fin 1) (lam := 1 / 2)
  (by norm_num) (by norm_num) (by simp) 0 0

example := isDecayedMin_identity_zero (m := Fin 2) (k := Fin 1) (lam := 1) le_rfl

end Transformer.Grokking.FlatSpectrum.Training.Scope
