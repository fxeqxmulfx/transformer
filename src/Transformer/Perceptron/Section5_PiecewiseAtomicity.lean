/-
# Finite and uniform-null original support for piecewise-polynomial activations

ArXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, with the
`thm: circle` and `thm: any.d` (i) arguments in §5.2 and §5.3.
The original continuous activation has finitely many actual polynomial
candidates, including its actual breakpoint values. Choosing one at each
original neuron gives finitely many actual analytic branch fields whose
zero sets cover the original stationary support.

No such branch vanishes globally, by the proved original polynomial
inverse and original non-analytic potential obstruction. On the circle,
its zeros in the original compact angular interval are therefore finite.
In every dimension at least two, its original uniform spherical zero set
is null. The finite union gives the original support conclusions.
Continuity suffices for these intermediate statements; the source's
globally Lipschitz condition is retained in `Piecewise`.
-/

import Transformer.Perceptron.Section5_PolynomialBranchObstruction
import Transformer.Perceptron.Section5_PiecewisePolynomialCover
import Transformer.Perceptron.Section5_ReLUAtomicity

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Actual finite polynomial candidates choose an actual matching branch
at every original neuron input. Source: arXiv:2601.21366v2, §5.2,
`rem:piecewise-poly`. Original breakpoints and zero neurons remain covered. -/
theorem exists_polynomial_neuron_branch {d : ℕ} (σ : ℝ → ℝ) (S : Finset (Polynomial ℝ))
    (hS : ∀ s, ∃ P ∈ S, σ s = P.eval s) (a : Idx d → EucSpace d) (x : EucSpace d) :
    ∃ p : Idx d → S, ∀ j, σ (inner (𝕜 := ℝ) (a j) x) = (p j).1.eval (inner (𝕜 := ℝ) (a j) x) := by
  classical
  choose P hP hval using fun j : Idx d => hS (inner (𝕜 := ℝ) (a j) x)
  exact ⟨fun j => ⟨P j, hP j⟩, hval⟩

/-- A stationary original probability on the circle has finite original
support for a continuous piecewise-polynomial activation and an actual
non-analytic original potential. Source: arXiv:2601.21366v2, §5.2,
`rem:piecewise-poly`, extending `thm: circle`.
The source's global Lipschitz assumption implies the continuity used here.
Each actual branch's compact angular zero set is finite by analytic uniqueness. -/
theorem finite_support_of_stationary_piecewise_polynomial_circle (β : ℝ) (hβ : 0 < β)
    (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s) (hσ : Continuous σ)
    (hpp : IsPiecewisePolynomial σ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (hana : ¬ IsAnalyticOnSphere φ ω a) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β σ ω a μ) : (μ : Measure (SSphere 2)).support.Finite := by
  classical
  obtain ⟨S, hS⟩ := exists_finset_polynomial_cover σ hσ hpp
  let ι := Idx 2 → S
  let F : ι → EucSpace 2 → EucSpace 2 := fun p x =>
    proj 2 x (polynomialAttentionVector β ω a (fun j => (p j).1) μ x)
  have hF (p : ι) : AnalyticOnNhd ℝ (F p) univ :=
    analyticOnNhd_projected_polynomialAttentionVector β ω a (fun j => (p j).1) μ
  have hnot (p : ι) : ¬ ∀ x : SSphere 2, F p (x : EucSpace 2) = 0 :=
    not_zero_projected_polynomial_branch_of_nonanalytic le_rfl β hβ φ σ hφ ω a hana μ hμ
      (fun j => (p j).1)
  have hfin (p : ι) : (Icc (-π) π ∩ {θ | F p (circlePoint θ : EucSpace 2) = 0}).Finite := by
    by_contra hInf
    exact hnot p (vector_eq_zero_on_circle_of_infinite_zeros (F p) (hF p) _ hInf
      (fun _ hθ => hθ.1) (fun _ hθ => hθ.2))
  let A : Set ℝ := Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support
  have hsub : A ⊆ ⋃ p : ι, Icc (-π) π ∩ {θ | F p (circlePoint θ : EucSpace 2) = 0} := by
    intro θ hθ
    obtain ⟨p, hp⟩ := exists_polynomial_neuron_branch σ S hS a (circlePoint θ : EucSpace 2)
    refine mem_iUnion.mpr ⟨p, hθ.1, ?_⟩
    change proj 2 (circlePoint θ : EucSpace 2)
      (polynomialAttentionVector β ω a (fun j => (p j).1) μ (circlePoint θ : EucSpace 2)) = 0
    rw [projected_polynomialAttentionVector_eq_energyGrad β σ ω a (fun j => (p j).1) μ _ hp,
      hμ (circlePoint θ) hθ.2, smul_zero]
  have hA : A.Finite := (Set.finite_iUnion hfin).subset hsub
  apply (hA.image circlePoint).subset
  intro x hx
  obtain ⟨θ, hθ, heq⟩ := exists_circlePoint_eq x
  refine ⟨θ, ⟨hθ, ?_⟩, heq⟩
  change circlePoint θ ∈ (μ : Measure (SSphere 2)).support
  rwa [heq]

/-- The original stationary probability has a finite actual Dirac
presentation. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`,
extending `thm: circle`. Its original support was proved finite above;
the actual representation uses the original singleton masses. -/
theorem isFinitelyAtomic_of_stationary_piecewise_polynomial_circle (β : ℝ) (hβ : 0 < β)
    (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s) (hσ : Continuous σ)
    (hpp : IsPiecewisePolynomial σ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (hana : ¬ IsAnalyticOnSphere φ ω a) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β σ ω a μ) : IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_stationary_piecewise_polynomial_circle β hβ φ σ hφ hσ hpp ω a hana μ hμ)

/-- The original stationary support is null for every actual original
uniform spherical law in dimensions at least two, for a continuous
piecewise-polynomial activation and an actual non-analytic potential.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, extending §5.3,
`thm: any.d` (i). Each original analytic branch has a null actual zero set;
the finite branch cover retains all original support points. -/
theorem piecewisePolynomial_measure_support_eq_zero {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (hσ : Continuous σ) (hpp : IsPiecewisePolynomial σ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β σ ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  classical
  intro ν hν
  obtain ⟨S, hS⟩ := exists_finset_polynomial_cover σ hσ hpp
  let ι := Idx d → S
  let F : ι → EucSpace d → EucSpace d := fun p x =>
    proj d x (polynomialAttentionVector β ω a (fun j => (p j).1) μ x)
  have hnull (p : ι) : ν {x : SSphere d | F p (x : EucSpace d) = 0} = 0 := by
    by_contra hne
    have hglobal := vector_eq_zero_on_sphere_of_positive_zero_measure hd (F p)
      (analyticOnNhd_projected_polynomialAttentionVector β ω a (fun j => (p j).1) μ)
      ν hν (pos_iff_ne_zero.mpr hne)
    exact not_zero_projected_polynomial_branch_of_nonanalytic hd β hβ φ σ hφ ω a hana μ hμ
      (fun j => (p j).1) hglobal
  have hsub : (μ : Measure (SSphere d)).support ⊆ ⋃ p : ι, {x : SSphere d | F p (x : EucSpace d) = 0} := by
    intro x hx
    obtain ⟨p, hp⟩ := exists_polynomial_neuron_branch σ S hS a (x : EucSpace d)
    refine mem_iUnion.mpr ⟨p, ?_⟩
    change proj d (x : EucSpace d)
      (polynomialAttentionVector β ω a (fun j => (p j).1) μ (x : EucSpace d)) = 0
    rw [projected_polynomialAttentionVector_eq_energyGrad β σ ω a (fun j => (p j).1) μ _ hp,
      hμ x hx, smul_zero]
  exact measure_mono_null hsub (measure_iUnion_null hnull)

/-- The finite-candidate selection input holds at every actual input for
the original linear activation and the actual singleton polynomial `X`.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
example : ∀ s : ℝ, ∃ P ∈ ({Polynomial.X} : Finset (Polynomial ℝ)), (fun x : ℝ => x) s = P.eval s := by
  intro s
  exact ⟨Polynomial.X, by simp, by simp⟩

/-- Every original stationary, non-analytic, primitive, dimensional and
continuous piecewise-polynomial input holds jointly for the original
pinned ReLU Dirac. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`,
and §3.1, `thm: circle`. The actual uniform law witnesses the conclusion's
uniform-law quantifier too. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    Continuous (fun s : ℝ => max s 0) ∧ IsPiecewisePolynomial (fun s : ℝ => max s 0) ∧
    ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) (Perspective.diracProb 2 (basePoint 1)) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨le_rfl, one_pos, hasDerivAt_reluSq, lipschitzWith_relu.continuous, isPiecewisePolynomial_relu,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis inner_basePoint_secondAxis,
    isStationary_relu_pin 1 0 (basePoint 1), Metastability.isUniformOn_uniformLaw (by norm_num)⟩

end Transformer.Perceptron
