module

public import Transformer.Kinetic.Accuracy
public import Transformer.Kinetic.Hardy
public import Transformer.Kinetic.PeriodicGaussian
public import Transformer.Kinetic.UniformPrompt
public import Transformer.Kinetic.Section1_SourceCells
public import Mathlib.Probability.Independence.Basic

/-!
# Kinetic theory for Transformers — the retrieval score

Formalization of `eq:Acc-soft-Fourier`, `eq:soft-accuracy-expansion` and the
closing claim of `thm:U-shape` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*.

The soft accuracy of the needle-in-a-haystack task, its Fourier
representation, and the expansion involving the limiting correction `𝒮_t`.
The source's claimed unique interior minimum of the finite-token accuracy
is refuted by its actual integer source cells. Attainment of a minimum
is proved separately, without weakening the printed claim under its name.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **Equation (eq:Acc-soft).**  The soft accuracy: the periodic Gaussian
mollification of the retrieval indicator of `eq:Acc-def`, which `acc_def`
identifies with the accuracy itself,

  `𝒜_N(t,σ₀) = E Σ_{k∈ℤ} exp(-M²(θ_N(t) - θ_{i_*}(0) - 2πk)²/(2π²))`.

Source: arXiv:2605.09213v1, `eq:Acc-soft`. -/
noncomputable def softAccuracy (P : Measure Ω) (M : ℝ) (N : ℕ) (hN : 0 < N)
    (ϑ : Ω → ℝ → Idx N → ℝ) (t σ₀ : ℝ) : ℝ :=
  ∫ ω, ∑' k : ℤ, Real.exp (-(M ^ 2 / (2 * π ^ 2)) *
    (ϑ ω t (lastIdx N hN) - ϑ ω 0 (sourceIdx N hN σ₀) - 2 * π * k) ^ 2) ∂P

/-- **Equation (eq:Acc-soft-Fourier).**  By Poisson summation, the soft
accuracy is

  `𝒜_N(t,σ₀) = (√(π/2)/M) Σ_{n∈ℤ} e^{-π²n²/(2M²)} E[e^{in(θ_N(t)-θ_{i_*}(0))}]`.

**What the source says and what is changed here.**  The source takes the
`θ_i(t)` to be random variables; that is the hypothesis `hϑ`, that each is
measurable. Without it the identity is false: for a non-measurable
`D = θ_N(t) - θ_{i_*}(0)` with values in `{0, π}` the left side and every odd
Fourier coefficient integrate to `0` by convention, while the even ones do not.

Source: arXiv:2605.09213v1, `eq:Acc-soft-Fourier`. -/
theorem softAccuracy_fourier (P : Measure Ω) (hP : IsProbabilityMeasure P) (M : ℝ) (hM : 0 < M)
    (N : ℕ) (hN : 0 < N) (ϑ : Ω → ℝ → Idx N → ℝ) (hϑ : ∀ s j, Measurable fun ω => ϑ ω s j)
    (t σ₀ : ℝ) :
    (softAccuracy P M N hN ϑ t σ₀ : ℂ) =
      (Real.sqrt (π / 2) / M : ℝ) • ∑' n : ℤ,
        (Real.exp (-(π ^ 2 / (2 * M ^ 2)) * (n : ℝ) ^ 2) : ℝ) •
          ∫ ω, Complex.exp ((n : ℂ) *
            ((ϑ ω t (lastIdx N hN) - ϑ ω 0 (sourceIdx N hN σ₀) : ℝ) : ℂ) * Complex.I) ∂P := by
  set D : Ω → ℝ := fun ω => ϑ ω t (lastIdx N hN) - ϑ ω 0 (sourceIdx N hN σ₀)
  have hD : Measurable D := (hϑ _ _).sub (hϑ _ _)
  set e : ℤ → ℝ := fun n => Real.exp (-(π ^ 2 / (2 * M ^ 2)) * (n : ℝ) ^ 2)
  set F : ℤ → Ω → ℂ := fun n ω => e n • Complex.exp ((n : ℂ) * (D ω : ℂ) * Complex.I)
  have hnorm : ∀ n ω, ‖F n ω‖ = e n := fun n ω => by
    simp only [F, e, norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le]
    rw [show (n : ℂ) * (D ω : ℂ) * Complex.I = ((n * D ω : ℝ) : ℂ) * Complex.I by push_cast; ring,
      Complex.norm_exp_ofReal_mul_I, mul_one]
  have hint : ∀ n, Integrable (F n) P := fun n =>
    Integrable.of_bound ((by fun_prop : Continuous fun x : ℝ =>
      e n • Complex.exp ((n : ℂ) * (x : ℂ) * Complex.I)).measurable.comp hD).aestronglyMeasurable
      (e n) (ae_of_all _ fun ω => (hnorm n ω).le)
  have hsum : Summable fun n => ∫ ω, ‖F n ω‖ ∂P := by
    simp only [hnorm, integral_const, probReal_univ, one_smul]
    exact summable_exp_neg_mul_int_sq (by positivity)
  unfold softAccuracy
  rw [← integral_complex_ofReal]
  simp_rw [tsum_periodicGaussian_eq hM]
  rw [integral_smul]
  change _ • ∫ ω, ∑' n, F n ω ∂P = _
  rw [← integral_tsum_of_summable_integral_norm hint hsum]
  congr 1
  exact tsum_congr fun n => integral_smul _ _

/-- The hypotheses of `softAccuracy_fourier` are satisfiable: one prompt, a
constant trajectory. -/
example : IsProbabilityMeasure (Measure.dirac () : Measure Unit) ∧ (0 : ℝ) < 2 ∧ 0 < 1 ∧
    ∀ (s : ℝ) (j : Idx 1), Measurable fun _ : Unit => (fun (_ : ℝ) (_ : Idx 1) => (0 : ℝ)) s j :=
  ⟨inferInstance, by norm_num, by norm_num, fun _ _ => measurable_const⟩

/-- **Equation (eq:soft-accuracy-expansion).**  From `th:lost` and the Fourier
representation, for iid uniformly distributed prompts the soft accuracy expands as

  `𝒜_N(t,σ₀) = √(π/2)/M + (√(2π)/(MN)) 𝒮_t(σ₀) + O(N^{-1-ζ} σ₀^{-2} M^C e^{Ct})`.

The leading term is the accuracy of a uniformly distributed answer; all the
positional information sits in the `N⁻¹` correction `𝒮_t`, which is
`softCorrection`.

**What the source says and what is changed here.**  The source derives the
expansion "in the specific case of iid uniformly distributed prompts, thus
satisfying `eq:init-conv` with `f_∘ ≡ 1`": there `f ≡ 1`, the cross-correlation
equation becomes the Volterra-Hardy equation, and `𝒮_t` is the sum of its
solutions. This statement used to take an arbitrary profile `f₀` and a solution
`f` of `eq:mfl-lambda` — a binder the conclusion never used — with prompts
satisfying `eq:init-conv` at rate `N^{-δ}`. Both are beyond the source, and the
statement was false with them:

* for a profile other than `f_∘ ≡ 1` the leading term is not `𝒮_t`, which is computed
  for `f ≡ 1`: the mean term `E[e^{inθ_N(t)}] E[e^{-inθ_{i_*}(0)}]` that
  `eq:Acc-soft-Fourier` splits off from the covariance is then in general of order
  one, while the printed leading term does not depend on `f_∘` at all;
* even for `f_∘ ≡ 1`, `eq:init-conv` only puts `E[e^{inθ⁰}]` within `C N^{-δ}` of
  `0`, so that mean term is only `O(N^{-2δ})`, which for `δ < 1/2` is larger than
  `N^{-1-ζ}`.

The prompts are now exactly iid and uniform on `𝕋` (`IsUniformPrompt` and
independence), the source's case, where the mean term is `0` for every `n ≠ 0`.
`eq:init-conv` then holds for every `δ > 0`, so `ζ` is any number `< 1`, and the
constants `δ`, `γ`, `C` of `eq:init-conv` no longer appear.

The source's `th:lost` uses `≲_{ζ,φ}`, so the prefactor `K` may depend on `ζ`.
The exponent `C` of `M^C e^{Ct}` depends on the model, uniformly in `ζ` and `M`.
Thus `C` precedes both parameters and `K` precedes `M`, as in `lost_correlations`.

Not proved here.

Source: arXiv:2605.09213v1, `eq:soft-accuracy-expansion`. -/
theorem soft_accuracy_expansion (lam β : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ζ : ℝ, ζ < 1 →
      ∃ K : ℝ, 0 < K ∧
        ∀ M : ℝ, 2 ≤ M →
        ∀ (P : Measure Ω), IsProbabilityMeasure P →
        ∀ (N : ℕ) (hN : 0 < N) (ϑ : Ω → ℝ → Idx N → ℝ),
          (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
          iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P →
          IsUniformPrompt P N (fun ω => ϑ ω 0) →
        ∀ t ∈ Set.Ici (0 : ℝ), ∀ σ₀ ∈ Set.Ioo (0 : ℝ) 1,
          |softAccuracy P M N hN ϑ t σ₀ -
              (Real.sqrt (π / 2) / M +
                Real.sqrt (2 * π) / (M * N) * softCorrection β lam M t σ₀)|
            ≤ K * (N : ℝ) ^ (-(1 + ζ)) * σ₀⁻¹ ^ 2 * M ^ C * Real.exp (C * t) := by
  sorry

/-- The source's prompt and flow conditions hold for one token, uniformly
distributed on `𝕋` via one period of `ℝ`, that never moves. This witnesses
`soft_accuracy_expansion` and the original finite-accuracy setting. -/
example (lam β : ℝ) : ∃ P : Measure ℝ, IsProbabilityMeasure P ∧
    ∃ (N : ℕ) (_ : 0 < N) (ϑ : ℝ → ℝ → Idx N → ℝ), (∀ ω, IsGPTFlow lam β N (ϑ ω)) ∧
      iIndepFun (fun (j : Idx N) (ω : ℝ) => ϑ ω 0 j) P ∧ IsUniformPrompt P N (fun ω => ϑ ω 0) := by
  obtain ⟨P, hP, hU⟩ := exists_isUniformPrompt
  refine ⟨P, hP, 1, one_pos, fun ω _ _ => ω, fun ω t j => ?_, iIndepFun.of_subsingleton, hU⟩
  have h0 : gptField lam β 1 (fun _ => ω) j = 0 := by
    fin_cases j
    simp [gptField, alibiZ]
  rw [h0]
  exact hasDerivAt_const t ω

/-- **The unique finite-accuracy minimum claimed in `thm:U-shape` is false.**
The original finite-token accuracy never has a unique interior global
minimizer: a larger interior position selects the same token and gives the
same integral. This holds for every prompt law, trajectory and positive size,
including the source's uniform independent regime. The nonnegative lower
threshold can be `1/N` for the paper's valid source range or zero for the interior.

The former weakening to attainment and strict endpoint gaps is removed;
the literal uniqueness claim is refuted. The limiting correction is separate.

Source: arXiv:2605.09213v1, §1.1, `thm:U-shape`, final sentence;
§1, `eq:Acc-def` and `eq:Acc-soft`. -/
theorem soft_accuracy_no_unique_minimum (P : Measure Ω) (M : ℝ) (N : ℕ) (hN : 0 < N)
    (ϑ : Ω → ℝ → Idx N → ℝ) (t : ℝ) :
    ∀ l : ℝ, 0 ≤ l → ¬ ∃! s : ℝ, s ∈ Set.Ioo l 1 ∧ ∀ u ∈ Set.Ioo l 1,
      softAccuracy P M N hN ϑ t s ≤ softAccuracy P M N hN ϑ t u := by
  unfold softAccuracy
  intro l hl
  exact not_unique_minimum_of_sourceIdx_above N hN (fun j => ∫ ω, ∑' k : ℤ,
    Real.exp (-(M ^ 2 / (2 * π ^ 2)) *
      (ϑ ω t (lastIdx N hN) - ϑ ω 0 j - 2 * π * k) ^ 2) ∂P) l hl

/-- Finite-token accuracy attains an interior minimum from its finite nonempty range.
Source: arXiv:2605.09213v1, §1.1, the attainment part of `thm:U-shape`'s final sentence.
The uniqueness clause is refuted above; endpoint gaps are a separate question. -/
theorem soft_accuracy_minimum_attained (P : Measure Ω) (M : ℝ) (N : ℕ) (hN : 0 < N)
    (ϑ : Ω → ℝ → Idx N → ℝ) (t : ℝ) :
    ∃ s ∈ Set.Ioo (0 : ℝ) 1, ∀ u ∈ Set.Ioo (0 : ℝ) 1,
      softAccuracy P M N hN ϑ t s ≤ softAccuracy P M N hN ϑ t u := by
  unfold softAccuracy
  exact exists_minimum_of_sourceIdx N hN (fun j => ∫ ω, ∑' k : ℤ,
    Real.exp (-(M ^ 2 / (2 * π ^ 2)) *
      (ϑ ω t (lastIdx N hN) - ϑ ω 0 j - 2 * π * k) ^ 2) ∂P)

/-- Prompt sizes are genuine and arbitrarily large; the interior threshold
zero is admissible. The preceding uniform-prompt example also has an actual flow.
Source: arXiv:2605.09213v1, §1, `eq:Acc-soft`. -/
example (N : ℕ) : 0 < N + 1 ∧ (0 : ℝ) ≤ 0 := ⟨Nat.zero_lt_succ N, le_rfl⟩

end Kinetic
end Transformer
