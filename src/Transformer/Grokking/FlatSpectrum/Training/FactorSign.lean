module

public import Transformer.Grokking.FlatSpectrum.Training.Subsequence

/-!
# The factors of trained fits need not converge, and an optimal fit is not unique

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), items WD7 and
WD11.
The weight-decayed loss does not change under `(U, V) ↦ (−U, −V)`: the product and the two
norms are the same (`decayedLoss_neg`). So

* the sign change of a trained fit is a trained fit (`IsDecayedMin.neg`), and that of an
  optimal fit is an optimal fit (`isOptimalFit_neg`);
* an optimal fit with `U ≠ 0` is not the only one (`exists_isOptimalFit_ne`);
* if trained fits converge to factors with `U ≠ 0`, changing the signs at the odd indices gives
  trained fits for the same weight decays and with the same products whose factors do not
  converge (`exists_seq_isDecayedMin_not_tendsto`).

For the `1 × 1` kernel `1`: `(1, 1)` and `(−1, −1)` are optimal fits (`isOptimalFit_one`), and
some trained fits for `λ_j → 0⁺` have products that converge to the kernel and factors that
do not converge (`exists_seq_isDecayedMin_one_not_tendsto`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] {L : Matrix m n ℂ → ℝ} {lam : ℝ}
  {U : Matrix m k ℂ} {V : Matrix n k ℂ}

lemma frobeniusSq_neg (U : Matrix m k ℂ) : frobeniusSq (-U) = frobeniusSq U := by
  simp [frobeniusSq]

omit [Fintype m] [Fintype k] in
lemma eq_zero_of_neg_eq {U : Matrix m k ℂ} (h : -U = U) : U = 0 := by
  ext i j
  exact CharZero.neg_eq_self_iff.mp (congrFun (congrFun h i) j)

/-- The weight-decayed loss does not change under `(U, V) ↦ (−U, −V)`. -/
lemma decayedLoss_neg (L : Matrix m n ℂ → ℝ) (lam : ℝ) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    decayedLoss L lam (-U) (-V) = decayedLoss L lam U V := by
  rw [decayedLoss, decayedLoss, conjTranspose_neg, Matrix.neg_mul, Matrix.mul_neg, neg_neg,
    frobeniusSq_neg, frobeniusSq_neg]

/-- **The sign change of a trained fit is a trained fit** (`review/flat_spectrum.md`, items
WD7 and WD11), for every loss. -/
theorem IsDecayedMin.neg (h : IsDecayedMin L lam U V) : IsDecayedMin L lam (-U) (-V) :=
  fun r U' V' => (decayedLoss_neg L lam U V).trans_le (h r U' V')

section

variable {S : Set (Matrix m n ℂ)} {r : ℕ} {U : Matrix m (Fin r) ℂ}
  {V : Matrix n (Fin r) ℂ}

/-- **The sign change of an optimal fit is an optimal fit** (`review/flat_spectrum.md`, item
WD11). -/
theorem isOptimalFit_neg (h : IsOptimalFit S U V) : IsOptimalFit S (-U) (-V) :=
  ⟨by rw [conjTranspose_neg, Matrix.neg_mul, Matrix.mul_neg, neg_neg]; exact h.1,
    fun r' U' V' h' => by
      rw [frobeniusSq_neg, frobeniusSq_neg]
      exact h.2 r' U' V' h'⟩

/-- **An optimal fit with a nonzero factor is not the only one** (`review/flat_spectrum.md`,
item WD11). -/
theorem exists_isOptimalFit_ne (h : IsOptimalFit S U V) (hU : U ≠ 0) :
    ∃ (U' : Matrix m (Fin r) ℂ) (V' : Matrix n (Fin r) ℂ), IsOptimalFit S U' V' ∧ U' ≠ U :=
  ⟨-U, -V, isOptimalFit_neg h, fun e => hU (eq_zero_of_neg_eq e)⟩

end

omit [Fintype m] [Fintype k] in
/-- A sequence that converges to `U ≠ 0`, with the signs changed at the odd indices, does not
converge. -/
lemma not_tendsto_ite_even_neg {Us : ℕ → Matrix m k ℂ} (hU : Tendsto Us atTop (𝓝 U))
    (h0 : U ≠ 0) (W : Matrix m k ℂ) :
    ¬ Tendsto (fun j => if Even j then Us j else -Us j) atTop (𝓝 W) := fun hW => by
  have h2 : Tendsto (fun j : ℕ => 2 * j) atTop atTop :=
    tendsto_atTop_mono (fun j => by change j ≤ _; omega) tendsto_id
  have h3 : Tendsto (fun j : ℕ => 2 * j + 1) atTop atTop :=
    tendsto_atTop_mono (fun j => by change j ≤ _; omega) tendsto_id
  have e0 : W = U := tendsto_nhds_unique (hW.comp h2) ((hU.comp h2).congr fun j => by
    simp only [Function.comp_apply, ite_eq_left (even_two_mul j)])
  have e1 : W = -U := tendsto_nhds_unique (hW.comp h3) ((hU.comp h3).neg.congr fun j => by
    simp only [Function.comp_apply, ite_eq_right (Nat.not_even_two_mul_add_one j)])
  exact h0 (eq_zero_of_neg_eq (e1.symm.trans e0))

/-- **The factors of trained fits need not converge** (`review/flat_spectrum.md`, items WD7
and WD11): if trained fits converge to factors with `U ≠ 0`, some trained fits for the same
weight decays and with the same products have factors that do not converge. -/
theorem exists_seq_isDecayedMin_not_tendsto {lams : ℕ → ℝ} {Us : ℕ → Matrix m k ℂ}
    {Vs : ℕ → Matrix n k ℂ} (hmin : ∀ j, IsDecayedMin L (lams j) (Us j) (Vs j))
    (hU : Tendsto Us atTop (𝓝 U)) (h0 : U ≠ 0) :
    ∃ (Us' : ℕ → Matrix m k ℂ) (Vs' : ℕ → Matrix n k ℂ),
      (∀ j, IsDecayedMin L (lams j) (Us' j) (Vs' j)) ∧
      (∀ j, Us' j * (Vs' j)ᴴ = Us j * (Vs j)ᴴ) ∧ ∀ W, ¬ Tendsto Us' atTop (𝓝 W) := by
  refine ⟨fun j => if Even j then Us j else -Us j, fun j => if Even j then Vs j else -Vs j,
    fun j => ?_, fun j => ?_, not_tendsto_ite_even_neg hU h0⟩
  · dsimp only
    split_ifs
    · exact hmin j
    · exact (hmin j).neg
  · dsimp only
    split_ifs
    · rfl
    · rw [conjTranspose_neg, Matrix.neg_mul, Matrix.mul_neg, neg_neg]

/-! The witness: the `1 × 1` kernel `1` seen on its cell, with the shrinking family
`(1 − s_j) · (1 − s_j)` of trained fits for `λ_j = s_j (2 − s_j) → 0⁺`, which converges to
`(1, 1)`. -/

/-- The factorization `1 · 1` is an optimal fit of the `1 × 1` kernel `1`. -/
lemma isOptimalFit_one : IsOptimalFit
    {X : Matrix (Fin 1) (Fin 1) ℂ | ∀ ij ∈ (Set.univ : Set (Fin 1 × Fin 1)),
      X ij.1 ij.2 = (1 : Matrix (Fin 1) (Fin 1) ℂ) ij.1 ij.2}
    (1 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ) :=
  isOptimalFit_of_tendsto shrink_pos tendsto_shrink isDecayedMin_shrink tendsto_shrink_factor
    tendsto_shrink_factor

example := eq_zero_of_neg_eq (U := (0 : Matrix (Fin 1) (Fin 1) ℂ)) neg_zero

example := (isDecayedMin_shrink 1).neg

example := isOptimalFit_neg isOptimalFit_one

/-- The kernel `1` has the two optimal fits `(1, 1)` and `(−1, −1)`. -/
example := exists_isOptimalFit_ne isOptimalFit_one one_ne_zero

example := not_tendsto_ite_even_neg tendsto_shrink_factor one_ne_zero

/-- **For the `1 × 1` kernel `1` the products of some trained fits converge to the kernel
while their factors do not converge** (`review/flat_spectrum.md`, item WD7, statement 3). -/
theorem exists_seq_isDecayedMin_one_not_tendsto :
    ∃ (lams : ℕ → ℝ) (Us Vs : ℕ → Matrix (Fin 1) (Fin 1) ℂ), (∀ j, 0 < lams j) ∧
      Tendsto lams atTop (𝓝 0) ∧
      (∀ j, IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) (lams j) (Us j)
        (Vs j)) ∧
      Tendsto (fun j => Us j * (Vs j)ᴴ) atTop (𝓝 1) ∧ ∀ W, ¬ Tendsto Us atTop (𝓝 W) := by
  obtain ⟨Us, Vs, hmin, hprod, hno⟩ :=
    exists_seq_isDecayedMin_not_tendsto isDecayedMin_shrink tendsto_shrink_factor one_ne_zero
  refine ⟨_, Us, Vs, shrink_pos, tendsto_shrink, hmin, ?_, hno⟩
  have h := (continuous_mul_conjTranspose.tendsto ((1 : Matrix (Fin 1) (Fin 1) ℂ),
    (1 : Matrix (Fin 1) (Fin 1) ℂ))).comp
      (tendsto_shrink_factor.prodMk_nhds tendsto_shrink_factor)
  simp only [conjTranspose_one, Matrix.mul_one] at h
  exact h.congr fun j => (hprod j).symm

end Transformer.Grokking.FlatSpectrum.Training
