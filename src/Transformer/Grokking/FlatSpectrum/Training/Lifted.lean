module

public import Transformer.Grokking.FlatSpectrum.Training.Basic

/-!
# The lifted form of gradient descent with weight decay

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD1.
With the factors stacked, `W = [U; V]`, and the lifted gradient `M = [[λ, G], [Gᴴ, λ]]`:

* `M W = [G V + λ U; Gᴴ U + λ V]`, the gradient of the weight-decayed loss
  (`lift_mul_stack`);
* a step of gradient descent with weight decay is `W ← (1 − η M) W` (`stack_decayStep`), so a
  run multiplies the start by the matrices `1 − η M_t` (`stack_decayRun_succ`);
* `⟨W, M W⟩ = λ (‖U‖² + ‖V‖²) + 2 ⟨G, U Vᴴ⟩` (`pairing_stack_lift`), the weight decay plus
  twice the first-order term of the loss;
* `2 |⟨G, U Vᴴ⟩| ≤ ‖G‖_S2 (‖U‖² + ‖V‖²)` (`two_mul_abs_pairing_mul_le`), so `M` is positive
  semidefinite when `‖G‖_S2 ≤ λ` (`lift_posSemidef_of_frobeniusSq_le`);
* the start `U₀ = [1 0]`, `V₀ = [0 1]` of width `|m| + |n|` stacks to the identity
  (`stack_fromCols_one`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open scoped ComplexOrder

variable {m n k : Type*}

lemma stack_sub (U U' : Matrix m k ℂ) (V V' : Matrix n k ℂ) :
    stack (U - U') (V - V') = stack U V - stack U' V' := by
  ext (i | i) j <;> simp [stack]

lemma stack_smul (t : ℝ) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    stack (t • U) (t • V) = t • stack U V := by
  ext (i | i) j <;> simp [stack]

lemma stack_zero : stack (0 : Matrix m k ℂ) (0 : Matrix n k ℂ) = 0 := by
  ext (i | i) j <;> simp [stack]

/-- Every matrix with rows `m ⊕ n` is a stack. -/
lemma exists_eq_stack (W : Matrix (m ⊕ n) k ℂ) : ∃ U V, W = stack U V :=
  ⟨W.toRows₁, W.toRows₂, (fromRows_toRows W).symm⟩

/-- The start `U₀ = [1 0]`, `V₀ = [0 1]` of width `|m| + |n|` stacks to the identity. -/
lemma stack_fromCols_one [DecidableEq m] [DecidableEq n] :
    stack (fromCols (1 : Matrix m m ℂ) (0 : Matrix m n ℂ))
      (fromCols (0 : Matrix n m ℂ) (1 : Matrix n n ℂ)) = 1 := by
  rw [stack, fromRows_fromCols_eq_fromBlocks, fromBlocks_one]

lemma lift_isHermitian [DecidableEq m] [DecidableEq n] (lam : ℝ) (G : Matrix m n ℂ) :
    (lift lam G).IsHermitian := by
  refine IsHermitian.fromBlocks ?_ rfl ?_ <;> simp [IsHermitian, conjTranspose_smul]

variable [Fintype m] [Fintype n]

/-- `M W = [G V + λ U; Gᴴ U + λ V]`: the gradient of the weight-decayed loss, stacked. -/
lemma lift_mul_stack [DecidableEq m] [DecidableEq n] (lam : ℝ) (G : Matrix m n ℂ)
    (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    lift lam G * stack U V = stack (G * V + lam • U) (Gᴴ * U + lam • V) := by
  rw [lift, stack, fromBlocks_mul_fromRows, Matrix.smul_mul, Matrix.one_mul, Matrix.smul_mul,
    Matrix.one_mul, add_comm (lam • U)]
  rfl

variable [Fintype k]

/-- The stacked factors pair blockwise. -/
lemma pairing_stack (U U' : Matrix m k ℂ) (V V' : Matrix n k ℂ) :
    pairing (stack U V) (stack U' V') = pairing U U' + pairing V V' := by
  rw [pairing, stack, stack, conjTranspose_fromRows_eq_fromCols_conjTranspose,
    fromCols_mul_fromRows, trace_add, Complex.add_re]
  rfl

/-- `‖[U; V]‖_S2² = ‖U‖_S2² + ‖V‖_S2²`: twice the weight decay. -/
lemma frobeniusSq_stack (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    frobeniusSq (stack U V) = frobeniusSq U + frobeniusSq V := by
  rw [← pairing_self, pairing_stack, pairing_self, pairing_self]

/-- `2 |⟨G, U Vᴴ⟩| ≤ g (‖U‖² + ‖V‖²)` when `‖G‖_S2 ≤ g`. -/
lemma two_mul_abs_pairing_mul_le {G : Matrix m n ℂ} {g : ℝ} (hg : 0 ≤ g)
    (hG : frobeniusSq G ≤ g ^ 2) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    2 * |pairing G (U * Vᴴ)| ≤ g * (frobeniusSq U + frobeniusSq V) := by
  have ha := frobeniusSq_nonneg U
  have hb := frobeniusSq_nonneg V
  have hUV : frobeniusSq (U * Vᴴ) ≤ frobeniusSq U * frobeniusSq V := by
    simpa [frobeniusSq_conjTranspose] using frobeniusSq_mul_le U Vᴴ
  have h : |pairing G (U * Vᴴ)| ≤ g * ((frobeniusSq U + frobeniusSq V) / 2) := by
    refine abs_pairing_le (by positivity) ?_
    calc frobeniusSq G * frobeniusSq (U * Vᴴ) ≤ g ^ 2 * (frobeniusSq U * frobeniusSq V) :=
          mul_le_mul hG hUV (frobeniusSq_nonneg _) (sq_nonneg g)
      _ ≤ (g * ((frobeniusSq U + frobeniusSq V) / 2)) ^ 2 := by
          nlinarith [mul_nonneg (sq_nonneg g) (sq_nonneg (frobeniusSq U - frobeniusSq V))]
  linarith

example := two_mul_abs_pairing_mul_le (G := (1 : Matrix (Fin 1) (Fin 1) ℂ)) zero_le_one
  (by simp [frobeniusSq]) (1 : Matrix (Fin 1) (Fin 1) ℂ) 1

variable [DecidableEq m] [DecidableEq n]

/-- **The step in stacked form** (`review/flat_spectrum.md`, item WD1): gradient descent with
weight decay is `W ← (1 − η M) W`. -/
theorem stack_decayStep (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ) (U : Matrix m k ℂ)
    (V : Matrix n k ℂ) :
    stack (decayStep G η lam (U, V)).1 (decayStep G η lam (U, V)).2 =
      (1 - η • lift lam (G (U * Vᴴ))) * stack U V := by
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, lift_mul_stack, ← stack_smul, ← stack_sub]
  rfl

/-- The run in stacked form: each step multiplies by `1 − η M_t`. -/
theorem stack_decayRun_succ (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ) (U₀ : Matrix m k ℂ)
    (V₀ : Matrix n k ℂ) (t : ℕ) :
    stack (decayRun G η lam U₀ V₀ (t + 1)).1 (decayRun G η lam U₀ V₀ (t + 1)).2 =
      (1 - η • lift lam (G ((decayRun G η lam U₀ V₀ t).1 * (decayRun G η lam U₀ V₀ t).2ᴴ))) *
        stack (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2 := by
  rw [decayRun, Function.iterate_succ_apply', ← decayRun]
  exact stack_decayStep G η lam _ _

/-- **The quadratic form of the lifted gradient** (`review/flat_spectrum.md`, item WD1):
`⟨W, M W⟩ = λ (‖U‖² + ‖V‖²) + 2 ⟨G, U Vᴴ⟩`. -/
theorem pairing_stack_lift (lam : ℝ) (G : Matrix m n ℂ) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    pairing (stack U V) (lift lam G * stack U V) =
      lam * (frobeniusSq U + frobeniusSq V) + 2 * pairing G (U * Vᴴ) := by
  have h1 : pairing U (G * V) = pairing G (U * Vᴴ) := by
    rw [pairing_comm G, pairing_mul_right, conjTranspose_conjTranspose]
  have h2 : pairing V (Gᴴ * U) = pairing G (U * Vᴴ) := by
    rw [← pairing_mul_left, pairing_comm, h1]
  rw [lift_mul_stack, pairing_stack, pairing_add_right, pairing_add_right, pairing_smul_right,
    pairing_smul_right, pairing_self, pairing_self, h1, h2]
  ring

omit [Fintype k] in
/-- **The lifted gradient is positive semidefinite when `‖G‖_S2 ≤ λ`**
(`review/flat_spectrum.md`, item WD1). The sharp condition has the operator norm in place of
`‖·‖_S2`; this one suffices for the examples and the step-size bounds. -/
theorem lift_posSemidef_of_frobeniusSq_le {lam : ℝ} (hlam : 0 ≤ lam) {G : Matrix m n ℂ}
    (hG : frobeniusSq G ≤ lam ^ 2) : (lift lam G).PosSemidef := by
  refine posSemidef_of_pairing_nonneg (lift_isHermitian lam G) fun W => ?_
  obtain ⟨U, V, rfl⟩ := exists_eq_stack W
  rw [pairing_stack_lift]
  have h := two_mul_abs_pairing_mul_le hlam hG U V
  have h2 := neg_abs_le (pairing G (U * Vᴴ))
  linarith

example := lift_posSemidef_of_frobeniusSq_le (lam := 1) zero_le_one
  (G := (-1 : Matrix (Fin 1) (Fin 1) ℂ)) (by simp [frobeniusSq])

omit [Fintype k] in
/-- **Admissible steps** (`review/flat_spectrum.md`, item WD6): the step matrix `1 − η M` is
positive definite when `η (|λ| + ‖G‖_S2) < 1`. -/
theorem one_sub_smul_lift_posDef {η lam g : ℝ} (hη : 0 ≤ η) (hg : 0 ≤ g) {G : Matrix m n ℂ}
    (hG : frobeniusSq G ≤ g ^ 2) (h : η * (|lam| + g) < 1) : (1 - η • lift lam G).PosDef := by
  have hH : (1 - η • lift lam G).IsHermitian := by
    rw [IsHermitian, conjTranspose_sub, conjTranspose_one, conjTranspose_smul,
      (lift_isHermitian lam G).eq, star_trivial]
  refine posDef_of_pairing_pos hH fun W hW => ?_
  obtain ⟨U, V, rfl⟩ := exists_eq_stack W
  have hpos : 0 < frobeniusSq (stack U V) :=
    lt_of_le_of_ne (frobeniusSq_nonneg _) fun h0 => hW (eq_zero_of_frobeniusSq_eq_zero h0.symm)
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, pairing_sub_right, pairing_smul_right,
    pairing_self, pairing_stack_lift]
  rw [frobeniusSq_stack] at hpos ⊢
  have h1 := two_mul_abs_pairing_mul_le hg hG U V
  have h2 := le_abs_self (pairing G (U * Vᴴ))
  have h3 := mul_le_mul_of_nonneg_right (le_abs_self lam) hpos.le
  have h4 : η * (lam * (frobeniusSq U + frobeniusSq V) + 2 * pairing G (U * Vᴴ)) ≤
      η * ((|lam| + g) * (frobeniusSq U + frobeniusSq V)) :=
    mul_le_mul_of_nonneg_left (by linarith) hη
  have h5 := mul_lt_mul_of_pos_right h hpos
  linarith

example := one_sub_smul_lift_posDef (η := 1 / 4) (lam := 1) (g := 1) (by norm_num) zero_le_one
  (G := (-1 : Matrix (Fin 1) (Fin 1) ℂ)) (by simp [frobeniusSq]) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Training
