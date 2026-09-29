/-
# Kinetic theory for Transformers — the retrieval score

Formalization of `eq:Acc-soft-Fourier`, `eq:soft-accuracy-expansion` and the
closing claim of `thm:U-shape` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*.

The soft accuracy of the needle-in-a-haystack task, its Fourier
representation, and the expansion that turns the U-shape of the correction
`𝒮_t` into the U-shape of the accuracy itself.
-/

import Transformer.Kinetic.Accuracy
import Transformer.Kinetic.Hardy
import Transformer.Kinetic.PeriodicGaussian
import Transformer.Kinetic.UniformPrompt
import Mathlib.Probability.Independence.Basic

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
measurable.  Without it the identity is false: for a non-measurable
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
solutions.  This statement used to take an arbitrary profile `f₀` and a solution
`f` of `eq:mfl-lambda` — a binder the conclusion never used — with prompts
satisfying `eq:init-conv` at rate `N^{-δ}`.  Both are beyond the source, and the
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

**How the two constants are quantified.**  The expansion is read off `th:lost`,
whose `≲_{ζ,φ}` lets the implicit prefactor depend on `ζ`; the exponent `C` of
`M^C e^{Ct}` is explicit and is a constant of the model alone, uniform in the
vocabulary size `M`.  So the prefactor `K` is quantified after `ζ` and before `M`,
and `C` outside both — see the same discussion at `lost_correlations`.

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

/-- The prompt hypotheses of `soft_accuracy_expansion` and `soft_accuracy_uShape` are
satisfiable: one token, uniformly distributed on `𝕋` — the uniform law of one period of
`ℝ` — that never moves. -/
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

/-- **Theorem (thm:U-shape), the consequence for the accuracy.**  In the
homogeneous baseline `f_∘ ≡ 1` — iid uniformly distributed prompts — and under the
smallness condition `eq:affine-smallness`, the soft accuracy `σ₀ ↦ 𝒜_N(t,σ₀)` is
U-shaped with an interior minimum, for all `N` large enough.

**What the source says and what is changed here.**

* The source says "unique interior minimum".  Uniqueness of the minimizer is false at
  finite `N`: the source position is `i_* = ⌊σ₀N⌋`, so `σ₀ ↦ 𝒜_N(t,σ₀)` is constant on
  each of the `N` intervals `[i/N, (i+1)/N)` and its minimum is attained on a whole
  interval, never at one point.  What survives the discretization, and is what U-shaped
  means, is stated instead: the minimum over `(0,1)` is attained at an interior position
  and is strictly better than every position near either end.
* The source's "in the case `f_∘ ≡ 1`" is the hypothesis `IsUniformPrompt`.  This
  statement used to carry no assumption on the law of the prompts, and was false without
  it: if every `θ_j(0)` is the constant `0` then nothing moves (`w_β'(0) = 0`), `𝒜_N` does
  not depend on `σ₀` and no position is strictly better than one near an end.  It also
  carried a solution `f` of `eq:mfl-lambda` for `f_∘ ≡ 1` that the conclusion never used;
  that binder is gone.

Not proved here.

Source: arXiv:2605.09213v1, `thm:U-shape`, final sentence. -/
theorem soft_accuracy_uShape (lam β : ℝ) (hβ : 0 < β) (hlam : 0 < lam) (M : ℝ) (hM : 2 ≤ M)
    (P : Measure Ω) (hP : IsProbabilityMeasure P) (t : ℝ) (ht : 0 < t)
    (hsmall : ∀ n : ℕ, 1 ≤ n →
      t * aCoeff β n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-lam)))) :
    ∃ N₀ : ℕ, ∀ N : ℕ, N₀ ≤ N → ∀ hN : 0 < N, ∀ ϑ : Ω → ℝ → Idx N → ℝ,
      (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
      iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P →
      IsUniformPrompt P N (fun ω => ϑ ω 0) →
      ∃ s ∈ Set.Ioo (0 : ℝ) 1,
        (∀ σ₀ ∈ Set.Ioo (0 : ℝ) 1,
          softAccuracy P M N hN ϑ t s ≤ softAccuracy P M N hN ϑ t σ₀) ∧
        ∃ ε > 0, ∀ σ₀ ∈ Set.Ioo (0 : ℝ) 1, σ₀ < ε ∨ 1 - ε < σ₀ →
          softAccuracy P M N hN ϑ t s < softAccuracy P M N hN ϑ t σ₀ := by
  sorry

/-- The parameter hypotheses of `soft_accuracy_uShape` are satisfiable, at `β = λ = 1`,
`M = 2`: a positive `t` meets `eq:affine-smallness`, as for `u_shape`.  The prompt
hypotheses are witnessed by the example above. -/
example : (0 : ℝ) < 1 ∧ (2 : ℝ) ≤ 2 ∧ ∃ t : ℝ, 0 < t ∧ ∀ n : ℕ, 1 ≤ n →
    t * aCoeff 1 n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-(1 : ℝ)))) :=
  ⟨one_pos, le_rfl, exists_affine_smallness 1 1 one_pos⟩

end Kinetic
end Transformer
