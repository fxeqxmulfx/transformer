module

public import Transformer.Grokking.FlatSpectrum.Training.FactorSpace
public import Transformer.Grokking.FlatSpectrum.Training.SquaredLoss

/-!
# The squared loss in a ball

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), items WD6 and
WD8. The run of gradient descent with weight decay is controlled through three bounds on the
loss at the matrices with `‖X‖ ≤ ρ/2`, and it converges when the loss is real-analytic. The
squared loss on `Ω` with `‖K‖ ≤ κ` and gradient `G X = P_Ω (X − K)` has all four:

* `‖G X‖ ≤ ρ/2 + κ` (`frobeniusSq_mask_sub_le`);
* `⟨G X, X⟩ ≥ −κ²/4` (`neg_le_pairing_mask_sub_of_le`);
* `L Y ≤ L X + ⟨G X, Y − X⟩ + ½ ‖Y − X‖²` (`fitLoss_le_add`), from the exact expansion
  `fitLoss_eq_add`;
* it is a polynomial in the real and imaginary parts of the entries (`analyticAt_fitLoss`).

The witness of the examples downstream is the `1 × 1` kernel `1` seen on its cell.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {m n : Type*} [Fintype m] [Fintype n] {Ω : Set (m × n)} {K : Matrix m n ℂ} {κ ρ : ℝ}

/-- For `‖X‖ ≤ ρ/2` and `‖K‖ ≤ κ` the gradient of the squared loss has
`‖P_Ω (X − K)‖ ≤ ρ/2 + κ`. -/
lemma frobeniusSq_mask_sub_le (hρ0 : 0 ≤ ρ) (hκ0 : 0 ≤ κ) (hκ : frobeniusSq K ≤ κ ^ 2)
    (X : Matrix m n ℂ) (hX : frobeniusSq X ≤ (ρ / 2) ^ 2) :
    frobeniusSq (mask Ω (X - K)) ≤ (ρ / 2 + κ) ^ 2 := by
  have hXK : |pairing X K| ≤ ρ / 2 * κ :=
    abs_pairing_le (mul_nonneg (by linarith) hκ0) (by
      calc frobeniusSq X * frobeniusSq K ≤ (ρ / 2) ^ 2 * κ ^ 2 :=
            mul_le_mul hX hκ (frobeniusSq_nonneg _) (sq_nonneg _)
        _ = (ρ / 2 * κ) ^ 2 := by ring)
  have h1 := frobeniusSq_sub X K
  have h2 := neg_abs_le (pairing X K)
  have h3 := frobeniusSq_mask_le Ω (X - K)
  nlinarith

/-- `⟨P_Ω (X − K), X⟩ ≥ −κ²/4` when `‖K‖ ≤ κ`. -/
lemma neg_le_pairing_mask_sub_of_le (hκ : frobeniusSq K ≤ κ ^ 2) (X : Matrix m n ℂ) :
    -(κ ^ 2 / 4) ≤ pairing (mask Ω (X - K)) X := by
  linarith [neg_le_pairing_mask_sub Ω K X]

/-- The squared loss is quadratic: its expansion at `X` is exact. -/
lemma fitLoss_eq_add (Ω : Set (m × n)) (K X Y : Matrix m n ℂ) :
    fitLoss Ω K Y = fitLoss Ω K X + pairing (mask Ω (X - K)) (Y - X) +
      frobeniusSq (mask Ω (Y - X)) / 2 := by
  have e : mask Ω (Y - K) = mask Ω (X - K) + mask Ω (Y - X) := by
    rw [mask_sub Ω Y X, mask_sub Ω Y K, mask_sub Ω X K]
    abel
  rw [fitLoss, fitLoss, e, frobeniusSq_add, ← pairing_mask_left]
  ring

/-- The squared loss lies below its quadratic model with constant `1`:
`L Y ≤ L X + ⟨G X, Y − X⟩ + ½ ‖Y − X‖²`. -/
lemma fitLoss_le_add (Ω : Set (m × n)) (K X Y : Matrix m n ℂ) :
    fitLoss Ω K Y ≤ fitLoss Ω K X + pairing (mask Ω (X - K)) (Y - X) +
      1 / 2 * frobeniusSq (Y - X) := by
  rw [fitLoss_eq_add Ω K X Y]
  linarith [frobeniusSq_mask_le Ω (Y - X)]

/-- **The squared loss is real-analytic in the entries** (`review/flat_spectrum.md`, item
WD8): a polynomial in their real and imaginary parts. -/
theorem analyticAt_fitLoss (Ω : Set (m × n)) (K : Matrix m n ℂ)
    (y₀ : EuclideanSpace ℂ (m × n)) :
    AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => fitLoss Ω K (of fun i j => y (i, j)))
      y₀ := by
  have hentry (a : m) (b : n) : AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) =>
      mask Ω ((of fun i j => y (i, j)) - K) a b) y₀ := by
    by_cases hab : (a, b) ∈ Ω
    · simp only [mask, of_apply, hab, ↓reduceIte, Matrix.sub_apply]
      exact (((EuclideanSpace.proj (a, b)).restrictScalars ℝ).analyticAt y₀).fun_sub
        analyticAt_const
    · simp only [mask, of_apply, hab, ↓reduceIte]
      exact analyticAt_const
  unfold fitLoss frobeniusSq
  exact AnalyticAt.div_const (Finset.analyticAt_fun_sum _ fun a _ =>
    Finset.analyticAt_fun_sum _ fun b _ => analyticAt_norm_sq (hentry a b))

/-! The witness of the examples downstream: the `1 × 1` kernel `1` seen on its cell, `κ = 1`,
`ρ = 2`, the start `U₀ = [1 0]`, `V₀ = [0 1]` with `‖U₀‖² + ‖V₀‖² = 2`. -/

lemma stepWitness_ball : frobeniusSq (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ)
    (0 : Matrix (Fin 1) (Fin 1) ℂ)) + frobeniusSq (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ)
      (1 : Matrix (Fin 1) (Fin 1) ℂ)) ≤ 2 := by
  rw [← frobeniusSq_stack, stack_fromCols_one]
  norm_num [frobeniusSq, Matrix.one_apply]

lemma stepWitness_kernel : frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) ≤ (1 : ℝ) ^ 2 := by
  simp [frobeniusSq]

/-- The gradient bound of the witness: `γ = ρ/2 + κ = 2`. -/
lemma stepWitness_bound (X : Matrix (Fin 1) (Fin 1) ℂ)
    (hX : frobeniusSq X ≤ ((2 : ℝ) / 2) ^ 2) :
    frobeniusSq (mask (Set.univ : Set (Fin 1 × Fin 1)) (X - 1)) ≤ ((2 : ℝ) / 2 + 1) ^ 2 :=
  frobeniusSq_mask_sub_le zero_le_two zero_le_one stepWitness_kernel X hX

/-- The lower bound of the witness: `b = κ²/4 = 1/4`. -/
lemma stepWitness_low (X : Matrix (Fin 1) (Fin 1) ℂ) :
    -((1 : ℝ) ^ 2 / 4) ≤ pairing (mask (Set.univ : Set (Fin 1 × Fin 1)) (X - 1)) X :=
  neg_le_pairing_mask_sub_of_le stepWitness_kernel X

end Transformer.Grokking.FlatSpectrum.Training
