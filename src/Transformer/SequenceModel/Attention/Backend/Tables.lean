module

public import Transformer.SequenceModel.Attention.Backend.Activation
public import Mathlib.Analysis.Complex.Trigonometric

/-!
# Independent analytic checks of stored RoPE coefficients

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K5;
diff_gpt b98c665b, `rope.py`. The complex exponential series supplies real
and imaginary polynomial references with a rigorous remainder. Stored
cosines and sines are checked independently, without a unit-circle premise
or trusting a floating-point trigonometric reference implementation.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend

open scoped BigOperators

/-- Real and imaginary parts are finite rational polynomials at rational phases. -/
def rotationPolynomial (x : ℝ) (n : ℕ) : ℂ :=
  ∑ m ∈ Finset.range n, ((x : ℂ) * Complex.I) ^ m / m.factorial

/-- K5: one analytic remainder controls both exact sine and cosine at the
specified phase. Larger finite phases require a sufficiently long polynomial. -/
theorem rotation_polynomial_error (x : ℝ) (n : ℕ)
    (hx : |x| / ((n + 1 : ℕ) : ℝ) ≤ 1 / 2) :
    |Real.cos x - (rotationPolynomial x n).re| ≤ |x| ^ n / n.factorial * 2 ∧
    |Real.sin x - (rotationPolynomial x n).im| ≤ |x| ^ n / n.factorial * 2 := by
  have he := Complex.exp_bound' (x := (x : ℂ) * Complex.I) (n := n)
    (by simpa only [norm_mul, Complex.norm_real, Complex.norm_I, mul_one, Real.norm_eq_abs] using hx)
  simp only [norm_mul, Complex.norm_real, Complex.norm_I, mul_one, Real.norm_eq_abs] at he
  have hr := (Complex.abs_re_le_norm
    (Complex.exp ((x : ℂ) * Complex.I) - rotationPolynomial x n)).trans he
  have hi := (Complex.abs_im_le_norm
    (Complex.exp ((x : ℂ) * Complex.I) - rotationPolynomial x n)).trans he
  simpa [Complex.exp_re, Complex.exp_im] using And.intro hr hi

/-- K5: rational residuals around the two polynomial coordinates discharge
the stored-table contract against the real source phase. -/
theorem stored_rotation_error (x c s r : ℝ) (n : ℕ)
    (hx : |x| / ((n + 1 : ℕ) : ℝ) ≤ 1 / 2)
    (hc : |c - (rotationPolynomial x n).re| ≤ r)
    (hs : |s - (rotationPolynomial x n).im| ≤ r) :
    |c - Real.cos x| ≤ r + |x| ^ n / n.factorial * 2 ∧
    |s - Real.sin x| ≤ r + |x| ^ n / n.factorial * 2 := by
  obtain ⟨hcos, hsin⟩ := rotation_polynomial_error x n hx
  constructor
  · exact (abs_sub_le c _ _).trans (add_le_add hc (by rwa [abs_sub_comm]))
  · exact (abs_sub_le s _ _).trans (add_le_add hs (by rwa [abs_sub_comm]))

example := rotation_polynomial_error 1 12 (by norm_num)
example := stored_rotation_error (1 / 2) (7 / 8) (1 / 2) 0 3 (by norm_num)
  (by norm_num [rotationPolynomial, Finset.sum_range_succ, pow_succ, Complex.mul_re, Complex.mul_im])
  (by norm_num [rotationPolynomial, Finset.sum_range_succ, pow_succ, Complex.mul_re, Complex.mul_im])

end Transformer.SequenceModel.Attention.Backend
