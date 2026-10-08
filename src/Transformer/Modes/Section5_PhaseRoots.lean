import Transformer.Modes.Section5_FinitePhase
import Mathlib.Algebra.Polynomial.Roots

/-!
# Quartic zeros and local bounds for the Gaussian phase

The phase in arXiv:2412.09080v3, §5.5, is a normalized linear combination
of `g(u) = u exp (-βu²/2)` and its first derivative. Its second derivative
is the same positive Gaussian factor times a polynomial of degree at
most four. This module identifies that polynomial, proves it is nonzero
for `β > 0` and every direction of max norm one, and covers its zeros by
a finite set of cardinality at most four.

The nonzero proof uses the constant and cubic coefficients. Vanishing
of the polynomial would force both direction coordinates to vanish,
contradicting normalization. Its roots need not be distinct, located
explicitly, or separated uniformly. The cardinality bound counts the
root multiset, and hence also its underlying finite set.

The proof in the source partitions around zeros of the first derivative.
Here we partition instead at the zeros of the second derivative, so the
signed local estimate from `Section5_UnitPhase` applies on each piece.
`Section5_FinitePhase` combines the pieces with a factor of at most
`2 ^ 4 = 16`, giving the constant `160` in the final local estimate.

This proves the estimate for every short interval, with no global sign
assumption. The lower bound is the actual `gaussianPhaseFloor β R`; it
still records the loss as the argument radius grows. The subsequent
Gaussian amplitude estimates must retain that loss when summing over
all intervals. Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`.
-/

open Real MeasureTheory Polynomial
open scoped Interval

namespace Transformer.Modes

/-- The polynomial multiplying the positive Gaussian in `φ''`.
All coefficients come from the displayed derivatives of `g`.
Source: arXiv:2412.09080v3, §5.5, the formulas for `g''` and `g'''`. -/
noncomputable def gaussianPhasePolynomial (β : ℝ) (θ : ℝ × ℝ) : Polynomial ℝ :=
  C (-(β ^ 3) * θ.2) * X ^ 4 + C (β ^ 2 * θ.1) * X ^ 3 +
    C (6 * β ^ 2 * θ.2) * X ^ 2 + C (-3 * β * θ.1) * X + C (-3 * β * θ.2)

/-- The polynomial is exactly the second derivative of the actual phase,
after factoring out its nonzero Gaussian. Source: arXiv:2412.09080v3, §5.5. -/
theorem gaussian_phase_second_derivative_eq (β u : ℝ) (θ : ℝ × ℝ) :
    θ.1 * bigG'' β u 0 + θ.2 * bigG''' β u 0 =
      Real.exp (-(β / 2) * u ^ 2) * (gaussianPhasePolynomial β θ).eval u := by
  simp only [gaussianPhasePolynomial, Polynomial.eval_add, Polynomial.eval_mul,
    Polynomial.eval_pow, Polynomial.eval_C, Polynomial.eval_X, bigG'', bigG''', sub_zero]
  ring

/-- Normalization excludes an identically zero second derivative.
The constant and cubic coefficients already determine both coordinates.
Source: arXiv:2412.09080v3, §5.5, nondegeneracy of the phase. -/
theorem gaussianPhasePolynomial_ne_zero {β : ℝ} (hβ : 0 < β) {θ : ℝ × ℝ}
    (hθ : max |θ.1| |θ.2| = 1) : gaussianPhasePolynomial β θ ≠ 0 := by
  intro hzero
  have h0 := congrArg (fun p : Polynomial ℝ => p.coeff 0) hzero
  have h3 := congrArg (fun p : Polynomial ℝ => p.coeff 3) hzero
  simp only [gaussianPhasePolynomial, Polynomial.coeff_add, Polynomial.coeff_C_mul_X_pow,
    Polynomial.coeff_C_mul_X, Polynomial.coeff_C, Polynomial.coeff_zero] at h0 h3
  norm_num at h0 h3
  have hθ2 : θ.2 = 0 := h0.resolve_left hβ.ne'
  have hθ1 : θ.1 = 0 := h3.resolve_left hβ.ne'
  rw [hθ1, hθ2] at hθ
  norm_num at hθ

example : (0 : ℝ) < 1 ∧ max |(1 : ℝ)| |0| = 1 := by norm_num

/-- The displayed second derivative has degree at most four after its
Gaussian factor is removed. Source: arXiv:2412.09080v3, §5.5. -/
theorem gaussianPhasePolynomial_degree_le (β : ℝ) (θ : ℝ × ℝ) :
    (gaussianPhasePolynomial β θ).natDegree ≤ 4 := by
  unfold gaussianPhasePolynomial
  apply Polynomial.natDegree_add_le_of_degree_le
  · apply Polynomial.natDegree_add_le_of_degree_le
    · apply Polynomial.natDegree_add_le_of_degree_le
      · apply Polynomial.natDegree_add_le_of_degree_le
        · exact Polynomial.natDegree_C_mul_X_pow_le _ _
        · exact (Polynomial.natDegree_C_mul_X_pow_le _ 3).trans (by decide)
      · exact (Polynomial.natDegree_C_mul_X_pow_le _ 2).trans (by decide)
    · simpa only [pow_one] using
        (Polynomial.natDegree_C_mul_X_pow_le (-3 * β * θ.1) 1).trans (by decide : 1 ≤ 4)
  · rw [Polynomial.natDegree_C]
    exact Nat.zero_le _

/-- Polynomial roots are exactly the zeros of the actual second
derivative, because the factored Gaussian is strictly positive.
Source: arXiv:2412.09080v3, §5.5, the derivative formulas. -/
theorem mem_gaussian_phase_roots_iff {β : ℝ} (hβ : 0 < β) {θ : ℝ × ℝ}
    (hθ : max |θ.1| |θ.2| = 1) (u : ℝ) :
    u ∈ (gaussianPhasePolynomial β θ).roots.toFinset ↔
      θ.1 * bigG'' β u 0 + θ.2 * bigG''' β u 0 = 0 := by
  classical
  rw [Multiset.mem_toFinset, Polynomial.mem_roots (gaussianPhasePolynomial_ne_zero hβ hθ)]
  change (gaussianPhasePolynomial β θ).eval u = 0 ↔ _
  rw [gaussian_phase_second_derivative_eq, mul_eq_zero]
  simp only [Real.exp_ne_zero, false_or]

example : (0 : ℝ) < 1 ∧ max |(1 : ℝ)| |0| = 1 := by norm_num

/-- The second derivative has at most four distinct real zeros.
This supplies the finite cover required by `Section5_FinitePhase` from
the actual Gaussian formula. Source: arXiv:2412.09080v3, §5.5. -/
theorem gaussian_phase_second_derivative_zeros {β : ℝ} (hβ : 0 < β) {θ : ℝ × ℝ}
    (hθ : max |θ.1| |θ.2| = 1) :
    ∃ Z : Finset ℝ, Z.card ≤ 4 ∧
      ∀ u : ℝ, θ.1 * bigG'' β u 0 + θ.2 * bigG''' β u 0 = 0 → u ∈ Z := by
  classical
  let P := gaussianPhasePolynomial β θ
  refine ⟨P.roots.toFinset, ?_, ?_⟩
  · exact (Multiset.toFinset_card_le P.roots).trans
      ((Polynomial.card_roots' P).trans (gaussianPhasePolynomial_degree_le β θ))
  · intro u hu
    rw [Multiset.mem_toFinset, Polynomial.mem_roots (gaussianPhasePolynomial_ne_zero hβ hθ)]
    change P.eval u = 0
    rw [gaussian_phase_second_derivative_eq] at hu
    exact (mul_eq_zero.mp hu).resolve_left (Real.exp_ne_zero _)

example : (0 : ℝ) < 1 ∧ max |(0 : ℝ)| |1| = 1 := by norm_num

/-- The local Gaussian phase estimate holds on every interval of length
at most one, even when its second derivative changes sign.
Source: arXiv:2412.09080v3, §5.5, the stationary-phase estimate. -/
theorem gaussian_phase_unit_interval_bound {a b β ρ R : ℝ} {θ : ℝ × ℝ}
    (hab : a ≤ b) (hlen : b - a ≤ 1) (hρ : ρ ≠ 0) (hβ : 0 < β) (hR : 0 ≤ R)
    (hθ : max |θ.1| |θ.2| = 1)
    (harg : ∀ u ∈ Set.uIcc a b, |u| ≤ R) :
    ‖∫ u in a..b, oscillatoryKernel ρ (fun u => θ.1 * bigG β u 0 + θ.2 * bigG' β u 0) u‖ ≤
      160 / Real.sqrt (|ρ| * gaussianPhaseFloor β R) := by
  obtain ⟨Z, hZ, hcover⟩ := gaussian_phase_second_derivative_zeros hβ hθ
  have h := unit_interval_phase_test_finite_zeros Z hab hlen hρ
    (gaussianPhaseFloor_pos hβ hR)
    (fun u _ => hasDerivAt_gaussianPhase β 0 u θ)
    (fun u _ => hasDerivAt_gaussianPhase' β 0 u θ)
    (by unfold bigG'' bigG'''; fun_prop)
    (fun u hu => gaussian_phase_floor hβ hR hθ u 0 (by simpa only [sub_zero] using harg u hu))
    (fun u _ hu => hcover u hu)
  have hpow : (2 : ℝ) ^ Z.card ≤ 16 := by
    exact (pow_le_pow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hZ).trans_eq (by norm_num)
  calc _ ≤ (2 : ℝ) ^ Z.card * (10 / Real.sqrt (|ρ| * gaussianPhaseFloor β R)) := h
    _ ≤ 16 * (10 / Real.sqrt (|ρ| * gaussianPhaseFloor β R)) :=
      mul_le_mul_of_nonneg_right hpow (by positivity)
    _ = _ := by ring

example : (-1 / 2 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) - (-1 / 2) ≤ 1 ∧
    (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ max |(1 : ℝ)| |0| = 1 ∧
    (∀ u ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2), |u| ≤ 1) := by
  refine ⟨by norm_num, by norm_num, one_ne_zero, one_pos, zero_le_one, by norm_num, ?_⟩
  intro u hu
  rw [Set.uIcc_of_le (by norm_num : (-1 / 2 : ℝ) ≤ 1 / 2)] at hu
  apply abs_le.mpr
  constructor <;> linarith [hu.1, hu.2]

end Transformer.Modes
