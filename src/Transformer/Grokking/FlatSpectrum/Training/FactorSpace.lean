module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Transformer.Grokking.FlatSpectrum.Training.Continuity

/-!
# The factors as one Euclidean space

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8.
The Łojasiewicz inequality and the convergence theorem of `Transformer.Landscape.SaddleAvoidance`
live in a real inner product space. The pairs of factors `(U, V)` become one by listing the
entries of `W = [U; V]`: `factors` reads the pair off a vector of
`EuclideanSpace ℂ ((m ⊕ n) × k)` and `flatten` is its inverse. Both are continuous, `factors`
is real-linear (`factors_add`, `factors_smul`), the squared norm is `‖U‖² + ‖V‖²`
(`norm_sq_eq_frobeniusSq_factors`) and the real inner product is `⟨U, U'⟩ + ⟨V, V'⟩`
(`real_inner_eq_pairing_factors`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {m n k : Type*}

/-- The pair of factors `(U, V)` listed by a vector of the factor space. -/
def factors (x : EuclideanSpace ℂ ((m ⊕ n) × k)) : Matrix m k ℂ × Matrix n k ℂ :=
  (of fun i j => x (Sum.inl i, j), of fun i j => x (Sum.inr i, j))

/-- The vector of the factor space that lists the entries of `W = [U; V]`. -/
def flatten (p : Matrix m k ℂ × Matrix n k ℂ) : EuclideanSpace ℂ ((m ⊕ n) × k) :=
  WithLp.toLp 2 fun ij => stack p.1 p.2 ij.1 ij.2

lemma factors_flatten (p : Matrix m k ℂ × Matrix n k ℂ) : factors (flatten p) = p := by
  refine Prod.ext ?_ ?_ <;> ext i j <;> simp [factors, flatten, stack]

lemma flatten_factors (x : EuclideanSpace ℂ ((m ⊕ n) × k)) : flatten (factors x) = x := by
  ext ⟨i | i, j⟩ <;> simp [factors, flatten, stack]

lemma continuous_factors :
    Continuous (factors : EuclideanSpace ℂ ((m ⊕ n) × k) → Matrix m k ℂ × Matrix n k ℂ) :=
  (continuous_matrix fun i j => PiLp.continuous_apply 2 _ (Sum.inl i, j)).prodMk
    (continuous_matrix fun i j => PiLp.continuous_apply 2 _ (Sum.inr i, j))

lemma continuous_flatten :
    Continuous (flatten : Matrix m k ℂ × Matrix n k ℂ → EuclideanSpace ℂ ((m ⊕ n) × k)) :=
  (PiLp.continuous_toLp 2 _).comp (continuous_pi fun ij =>
    (continuous_stack.matrix_elem ij.1 ij.2))

lemma factors_add (x y : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    factors (x + y) = ((factors x).1 + (factors y).1, (factors x).2 + (factors y).2) := by
  refine Prod.ext ?_ ?_ <;> ext i j <;> simp [factors]

lemma factors_sub (x y : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    factors (x - y) = ((factors x).1 - (factors y).1, (factors x).2 - (factors y).2) := by
  refine Prod.ext ?_ ?_ <;> ext i j <;> simp [factors]

lemma factors_smul (t : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    factors (t • x) = (t • (factors x).1, t • (factors x).2) := by
  refine Prod.ext ?_ ?_ <;> ext i j <;> simp [factors]

variable [Fintype m] [Fintype n] [Fintype k]

/-- The squared norm of the factor space is `‖U‖² + ‖V‖²`. -/
lemma norm_sq_eq_frobeniusSq_factors (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    ‖x‖ ^ 2 = frobeniusSq (factors x).1 + frobeniusSq (factors x).2 := by
  rw [EuclideanSpace.norm_sq_eq, Fintype.sum_prod_type, Fintype.sum_sum_type]
  rfl

/-- The real inner product of the factor space is `⟨U, U'⟩ + ⟨V, V'⟩`. -/
lemma real_inner_eq_pairing_factors (x y : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    inner ℝ x y = pairing (factors x).1 (factors y).1 + pairing (factors x).2 (factors y).2 := by
  have h := norm_add_sq_real x y
  rw [norm_sq_eq_frobeniusSq_factors, norm_sq_eq_frobeniusSq_factors,
    norm_sq_eq_frobeniusSq_factors, factors_add, frobeniusSq_add, frobeniusSq_add] at h
  linarith

end Transformer.Grokking.FlatSpectrum.Training
