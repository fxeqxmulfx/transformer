module

public import Transformer.RoPE.Scores

/-!
# Expected score discrimination

Men et al., *Base of RoPE Bounds Context Length*, arXiv:2405.14591v1,
§4.2–4.3 and Appendix A. Theorem 1's calculation needs zero noise/query
cross moments, not just zero mean of the perturbation. We state the exact
integrable coefficient moments used by the proof. The missing implication
from zero mean alone is disproved below. Numerical base tables and their
empirical extrapolation to multilayer networks are not universal theorems.
-/

@[expose] public section

open MeasureTheory
open scoped BigOperators

namespace Transformer.RoPE.BaseBounds

/-- The discrimination kernel `B` of Men et al., Eq. (9). -/
noncomputable def cosineKernel {n : ℕ} (omega : Fin n → ℝ) (distance : ℝ) : ℝ :=
  ∑ i, Real.cos (omega i * distance)

/-- Moment identity used in Theorem 1's calculation. The cosine coefficient of the score
gap has mean `2 sigma^2`, and its sine coefficient has mean zero. These are
the exact covariance conditions that the source's expectation calculation
requires; no independence claim is smuggled into a definition. -/
theorem expected_score_of_moments {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (mu : Measure Ω) (a b : Fin n → Ω → ℝ) (omega : Fin n → ℝ) (sigma distance : ℝ)
    (ha : ∀ i, Integrable (a i) mu) (hb : ∀ i, Integrable (b i) mu)
    (hmeanA : ∀ i, ∫ x, a i x ∂mu = 2 * sigma ^ 2)
    (hmeanB : ∀ i, ∫ x, b i x ∂mu = 0) :
    (∫ x, ∑ i, score (a i x) (b i x) (omega i) distance ∂mu) =
      2 * sigma ^ 2 * cosineKernel omega distance := by
  have hint (i : Fin n) : Integrable
      (fun x => score (a i x) (b i x) (omega i) distance) mu :=
    ((ha i).mul_const _).add ((hb i).mul_const _)
  calc
    (∫ x, ∑ i, score (a i x) (b i x) (omega i) distance ∂mu)
        = ∑ i, ∫ x, score (a i x) (b i x) (omega i) distance ∂mu :=
          integral_finsetSum _ (fun i _ => hint i)
    _ = ∑ i, (2 * sigma ^ 2) * Real.cos (omega i * distance) := by
      apply Finset.sum_congr rfl
      intro i _
      change (∫ x, a i x * Real.cos (omega i * distance) +
        b i x * Real.sin (omega i * distance) ∂mu) = _
      rw [integral_add ((ha i).mul_const _) ((hb i).mul_const _),
        integral_mul_const, integral_mul_const, hmeanA, hmeanB]
      ring
    _ = 2 * sigma ^ 2 * cosineKernel omega distance := by
      rw [cosineKernel, Finset.mul_sum]

example : (∫ x : ℝ, ∑ i : Fin 1,
    score ((fun _ _ => (2 : ℝ)) i x) ((fun _ _ => (0 : ℝ)) i x) 1 Real.pi
      ∂Measure.dirac 0) =
    2 * (1 : ℝ) ^ 2 * cosineKernel (fun _ : Fin 1 => (1 : ℝ)) Real.pi :=
  expected_score_of_moments (Measure.dirac 0) (fun _ _ => 2) (fun _ _ => 0)
    (fun _ => 1) 1 Real.pi (fun _ => integrable_const _) (fun _ => integrable_const _)
    (by intro i; simp) (by intro i; simp)

/-- Eq. (10)'s nonnegative-kernel criterion follows from the expected gap
only at nonzero variance. The source divides by `2 sigma^2`. -/
theorem discrimination_iff {sigma : ℝ} (hsigma : 0 < sigma) {n : ℕ}
    (omega : Fin n → ℝ) (distance : ℝ) :
    0 ≤ 2 * sigma ^ 2 * cosineKernel omega distance ↔ 0 ≤ cosineKernel omega distance := by
  have hpositive : 0 < 2 * sigma ^ 2 := by positivity
  constructor <;> intro h <;> nlinarith

example : 0 ≤ 2 * (1 : ℝ) ^ 2 * cosineKernel (fun _ : Fin 1 => 1) Real.pi ↔
    0 ≤ cosineKernel (fun _ : Fin 1 => 1) Real.pi := discrimination_iff (by norm_num) _ _

/-- Appendix A, correction: a centered perturbation can be correlated with
the query. This counterexample refutes the missing cross-moment implication,
not an independence premise that the source's theorem never states. -/
theorem zero_mean_noise_not_orthogonal :
    ∃ query noise : Bool → ℝ,
      (query false + query true) / 2 = 0 ∧
      (noise false + noise true) / 2 = 0 ∧
      (query false * noise false + query true * noise true) / 2 = -1 := by
  refine ⟨(fun value => if value then 1 else -1),
    (fun value => if value then -1 else 1), ?_⟩
  norm_num

/-- Eq. (9): with one frequency, the expected discrimination kernel reverses
sign at half a period. Phase coverage alone does not prevent this. -/
theorem one_frequency_kernel_reverses :
    cosineKernel (fun _ : Fin 1 => (1 : ℝ)) Real.pi = -1 := by
  simp [cosineKernel]

end Transformer.RoPE.BaseBounds
