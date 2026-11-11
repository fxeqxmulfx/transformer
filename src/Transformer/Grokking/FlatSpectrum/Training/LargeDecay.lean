module

public import Transformer.Grokking.FlatSpectrum.Training.SeenCells

/-!
# A large weight decay returns the zero product

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE7 (the
multiple `t = 0`), by items WD1 and WD2.
For a convex loss with gradient `G`, the lifted gradient at the zero factorization is
`[[λ, G 0], [(G 0)ᴴ, λ]]`, positive semidefinite when `‖G 0‖ ≤ λ`. So

* if `‖G 0‖ ≤ λ`, the zero factorization of every width is a decayed minimum
  (`isDecayedMin_zero`);
* if `‖G 0‖ ≤ γ < λ`, every decayed minimum has the product `0`
  (`IsDecayedMin.mul_eq_zero_of_lt`): by WD2 at `Y = 0` and the first-order inequality at `0`,
  `λ ‖X‖_S1 ≤ −⟨G 0, X⟩ ≤ γ ‖X‖_S1`.

For the cross-entropy with probes of norm at most `a` this is `γ = 2 a |S|`
(`isDecayedMin_zero_ceLoss`, `IsDecayedMin.mul_eq_zero_of_ceLoss`), and for the squared loss
on `Ω` it is `γ = ‖K‖` (`isDecayedMin_zero_fitLoss`, `IsDecayedMin.mul_eq_zero_of_fitLoss`).

The witnesses are the sample of item XE3, with `2 a |S| = 2 < 3 = λ`, and the `1 × 1` kernel
`1`, with `‖K‖ = 1 < 2 = λ`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m]
  [DecidableEq n] {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} {lam γ : ℝ}
  {U : Matrix m k ℂ} {V : Matrix n k ℂ}

variable (k) in
/-- **With a weight decay `λ ≥ ‖G 0‖` the zero factorization is a decayed minimum**
(`review/flat_spectrum.md`, item XE7 with WD2), at every width. -/
theorem isDecayedMin_zero (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hlam : 0 ≤ lam)
    (hG : frobeniusSq (G 0) ≤ lam ^ 2) :
    IsDecayedMin L lam (0 : Matrix m k ℂ) (0 : Matrix n k ℂ) := by
  refine isDecayedMin_of_posSemidef hconv ?_ ?_
  · rw [stack_zero, Matrix.mul_zero]
  · rw [Matrix.zero_mul]
    exact lift_posSemidef_of_frobeniusSq_le hlam hG

omit [DecidableEq m] in
/-- **With a weight decay `λ > ‖G 0‖` every decayed minimum has the product zero**
(`review/flat_spectrum.md`, item XE7 with WD2). -/
theorem IsDecayedMin.mul_eq_zero_of_lt (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hγ : 0 ≤ γ) (hG : frobeniusSq (G 0) ≤ γ ^ 2) (hlam : γ < lam)
    (h : IsDecayedMin L lam U V) : U * Vᴴ = 0 := by
  have h1 := h.add_nuclearNorm_le (hγ.trans hlam.le) 0
  rw [nuclearNorm_zero, mul_zero, add_zero] at h1
  have h2 := hconv 0 (U * Vᴴ)
  rw [sub_zero] at h2
  have h3 : |pairing (G 0) (U * Vᴴ)| ≤ γ * nuclearNorm (U * Vᴴ) :=
    abs_pairing_le (mul_nonneg hγ (nuclearNorm_nonneg _)) (by
      rw [mul_pow]
      exact mul_le_mul hG (frobeniusSq_le_nuclearNorm_sq _) (frobeniusSq_nonneg _)
        (sq_nonneg _))
  have h4 := neg_abs_le (pairing (G 0) (U * Vᴴ))
  refine eq_zero_of_nuclearNorm_eq_zero (le_antisymm ?_ (nuclearNorm_nonneg _))
  exact le_of_mul_le_mul_left (by linarith) (sub_pos.2 hlam)

variable [Fintype S] [Fintype C] {A : S → C → Matrix m n ℂ} {y : S → C} {a : ℝ}

variable (k) in
/-- For `λ ≥ 2 a |S|` the zero factorization is a trained fit of the cross-entropy. -/
theorem isDecayedMin_zero_ceLoss (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (hlam : 2 * a * Fintype.card S ≤ lam) :
    IsDecayedMin (ceLoss A y) lam (0 : Matrix m k ℂ) (0 : Matrix n k ℂ) :=
  isDecayedMin_zero k (ceLoss_first_order A y) (le_trans (by positivity) hlam)
    ((frobeniusSq_ceGrad_le ha0 ha 0).trans (pow_le_pow_left₀ (by positivity) hlam 2))

omit [DecidableEq m] in
/-- **For `λ > 2 a |S|` every trained fit of the cross-entropy has the product zero**
(`review/flat_spectrum.md`, item XE7 with XE2). -/
theorem IsDecayedMin.mul_eq_zero_of_ceLoss (ha0 : 0 ≤ a)
    (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2) (hlam : 2 * a * Fintype.card S < lam)
    (h : IsDecayedMin (ceLoss A y) lam U V) : U * Vᴴ = 0 :=
  h.mul_eq_zero_of_lt (ceLoss_first_order A y) (by positivity)
    (frobeniusSq_ceGrad_le ha0 ha 0) hlam

variable {Ω : Set (m × n)} {K : Matrix m n ℂ}

omit [Fintype k] [DecidableEq m] [DecidableEq n] in
/-- The gradient of the squared loss at `0` is `−P_Ω K`, of norm at most `‖K‖`. -/
lemma frobeniusSq_mask_zero_sub_le (Ω : Set (m × n)) (K : Matrix m n ℂ) :
    frobeniusSq (mask Ω (0 - K)) ≤ frobeniusSq K :=
  (frobeniusSq_mask_le Ω _).trans_eq (by simp [frobeniusSq])

variable (k) in
/-- For `λ ≥ ‖K‖` the zero factorization is a trained fit of the squared loss on `Ω`. -/
theorem isDecayedMin_zero_fitLoss (hlam : 0 ≤ lam) (hK : frobeniusSq K ≤ lam ^ 2) :
    IsDecayedMin (fitLoss Ω K) lam (0 : Matrix m k ℂ) (0 : Matrix n k ℂ) :=
  isDecayedMin_zero k (G := fun X => mask Ω (X - K)) (fitLoss_first_order Ω K) hlam
    ((frobeniusSq_mask_zero_sub_le Ω K).trans hK)

omit [DecidableEq m] in
/-- **For `λ > ‖K‖` every trained fit of the squared loss on `Ω` has the product zero**
(`review/flat_spectrum.md`, item XE7 with WD7). -/
theorem IsDecayedMin.mul_eq_zero_of_fitLoss (hγ : 0 ≤ γ) (hK : frobeniusSq K ≤ γ ^ 2)
    (hlam : γ < lam) (h : IsDecayedMin (fitLoss Ω K) lam U V) : U * Vᴴ = 0 :=
  h.mul_eq_zero_of_lt (G := fun X => mask Ω (X - K)) (fitLoss_first_order Ω K) hγ
    ((frobeniusSq_mask_zero_sub_le Ω K).trans hK) hlam

/-! The witnesses: the sample of item XE3, with `a = 1` and `|S| = 1`, at `λ = 3`; and the
`1 × 1` kernel `1` with its cell seen, at `λ = 2`. -/

example := isDecayedMin_zero (Fin 1) (ceLoss_first_order signProbe fun _ => true)
  (lam := 3) (by norm_num) ((frobeniusSq_ceGrad_le zero_le_one frobeniusSq_signProbe_le
    0).trans (by norm_num))

example := isDecayedMin_zero_ceLoss (Fin 1) (y := fun _ => true) (lam := 3) zero_le_one
  frobeniusSq_signProbe_le (by norm_num)

/-- Every trained fit of the sample of XE3 at `λ = 3` has the product zero, and one exists. -/
example : ∃ U V : Matrix (Fin 1) (Fin 1) ℂ,
    IsDecayedMin (ceLoss signProbe fun _ => true) 3 U V ∧ U * Vᴴ = 0 := by
  have h := isDecayedMin_zero_ceLoss (Fin 1) (y := fun _ => true) (lam := 3) zero_le_one
    frobeniusSq_signProbe_le (by norm_num)
  exact ⟨0, 0, h, h.mul_eq_zero_of_ceLoss zero_le_one frobeniusSq_signProbe_le (by norm_num)⟩

example : ∃ U V : Matrix (Fin 1) (Fin 1) ℂ,
    IsDecayedMin (ceLoss signProbe fun _ => true) 3 U V ∧ U * Vᴴ = 0 := by
  have h := isDecayedMin_zero_ceLoss (Fin 1) (y := fun _ => true) (lam := 3) zero_le_one
    frobeniusSq_signProbe_le (by norm_num)
  exact ⟨0, 0, h, h.mul_eq_zero_of_lt (γ := 2) (ceLoss_first_order signProbe fun _ => true)
    zero_le_two ((frobeniusSq_ceGrad_le zero_le_one frobeniusSq_signProbe_le 0).trans
      (by norm_num)) (by norm_num)⟩

example := frobeniusSq_mask_zero_sub_le Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)

/-- Every trained fit of the `1 × 1` kernel `1` at `λ = 2` has the product zero, and one
exists. -/
example : ∃ U V : Matrix (Fin 1) (Fin 1) ℂ,
    IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) 2 U V ∧ U * Vᴴ = 0 := by
  have h := isDecayedMin_zero_fitLoss (Fin 1) (Ω := Set.univ)
    (K := (1 : Matrix (Fin 1) (Fin 1) ℂ)) zero_le_two (by simp [frobeniusSq])
  exact ⟨0, 0, h, h.mul_eq_zero_of_fitLoss zero_le_one (by simp [frobeniusSq]) one_lt_two⟩

end Transformer.Grokking.FlatSpectrum.Training
