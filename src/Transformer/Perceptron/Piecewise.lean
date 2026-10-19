module

public import Transformer.Perceptron.HigherDim
public import Transformer.Perceptron.TransformPoly
public import Transformer.Perceptron.Section5_PiecewisePolynomialCover
public import Transformer.Perceptron.Section5_PiecewiseAtomicity
public import Transformer.Perceptron.Section5_UniformSupport
public import Transformer.Perceptron.Section5_LinearStationarity

/-!
# Perceptrons and attention's mean-field landscape — piecewise-polynomial
  activations

Formalization of `rem:piecewise-poly` and `rem:genericity-vacuity` of
arXiv:2601.21366v2: `thm: circle` and `thm: any.d` (i) hold for every globally
Lipschitz piecewise-polynomial activation, and the linear activation
`σ(s) = s` is where a non-atomic stationary measure can still hide.

**What the source says and what is carried here.**

* "piecewise polynomial" is `IsPiecewisePolynomial`: finitely many
  breakpoints, and on every open interval avoiding them `σ` agrees with a
  polynomial. It is stated over open *preconnected* sets, which on `ℝ` are
  exactly the intervals, so it says what it should without enumerating the
  pieces in order. The ReLU and the leaky ReLU are instances, with the single
  breakpoint `0`.

* "globally Lipschitz" is `∃ K, LipschitzWith K σ`.

* "The same extension applies to `thm: any.d`" is read as the extension of
  `thm: any.d` (i)'s first alternative — the one that names the ReLU. Its
  two genericity clauses have distinct activation hypotheses: (ii) is the
  real-analytic case and (iii) the ReLU case. Those original statements
  remain in `HigherDim`; this module carries the finite-support and
  null-support consequences under the actual non-analytic potential hypothesis.

* `rem:genericity-vacuity` is carried as the one statement it makes: for
  `σ(s) = s`, a stationary measure whose support has non-empty interior has a
  density with respect to `σ_d` that is a spherical polynomial of degree `≤ 2`
  — via the proved original quadratic kernel inverse in
  `Section5_QuadraticDensity` — and is therefore not finitely atomic.
  `Section5_LinearStationarity` proves this claim, retaining the original
  stationary measure and nonempty support interior. The uniform law is
  quantified over inside the conclusion, as in `thm: any.d`.

The original uniform law now witnesses every hypothesis of
`linear_stationary_polyDensity`, including the nonempty support interior.
`Section5_UniformSupport` proves its actual zero-weight stationarity,
full support, polynomial density and failure of finite atomicity.

The piecewise-polynomial extension is proved through actual polynomial
branches at every neuron, including all breakpoint values. Their actual
primitives give the scaled first variation and its analytic projected field.
A globally vanishing branch forces a polynomial transform, whose proved
inverse gives a nonnegative polynomial density for the same probability.
Full support would then make the original stationary potential analytic.
Thus each branch has finitely many zeros in the compact circle interval,
and a uniform-null spherical zero set in higher dimensions. Their finite
union covers the original support. The original arbitrary primitive and
all neuron weights remain in the final non-analyticity contradiction.
The branch calculus and support argument are in
`Section5_PolynomialBranches` and `Section5_PiecewiseAtomicity`.

Source: arXiv:2601.21366v2, `rem:piecewise-poly`, `rem:genericity-vacuity`.
-/

@[expose] public section

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The extension of `thm: circle` and `thm: any.d` -/

/-- **Remark (rem:piecewise-poly), for `thm: circle`.**  Let `d = 2`, `β > 0`,
and let `σ` be globally Lipschitz and piecewise polynomial — the leaky ReLU,
say. If `v_ϑ` is not real-analytic on `𝕊¹` and `μ ∈ P(𝕊¹)` is stationary for
`E_{β,ϑ}`, then `μ` is purely atomic with finite support.

The proof covers every actual neuron input by finitely many original
polynomial branches, including all breakpoints. Each branch's angular
zero set is finite: an infinite one would force the same original
probability to have a polynomial density and full support, making its
actual original potential analytic. Global Lipschitz continuity is used
to extend the original branch identities to their boundaries.

Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
theorem piecewisePoly_circle_isFinitelyAtomic (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hlip : ∃ K : ℝ≥0, LipschitzWith K σ)
    (hpp : IsPiecewisePolynomial σ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (hana : ¬ IsAnalyticOnSphere φ ω a) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β σ ω a μ) :
    IsFinitelyAtomic μ := by
  obtain ⟨K, hK⟩ := hlip
  exact isFinitelyAtomic_of_stationary_piecewise_polynomial_circle β hβ φ σ hφ hK.continuous
    hpp ω a hana μ hμ

/-- The hypotheses of `piecewisePoly_circle_isFinitelyAtomic` are satisfiable:
the ReLU is itself Lipschitz and piecewise polynomial, so the witness of
`thm: circle` serves here too. -/
example :
    (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
      (∃ K : ℝ≥0, LipschitzWith K (fun s : ℝ => max s 0)) ∧
      IsPiecewisePolynomial (fun s : ℝ => max s 0) ∧
      ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨one_pos, hasDerivAt_reluSq, ⟨1, lipschitzWith_relu⟩, isPiecewisePolynomial_relu,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis
      inner_basePoint_secondAxis,
    isStationary_relu_pin 1 0 (basePoint 1)⟩

/-- **Remark (rem:piecewise-poly), for `thm: any.d` (i).**  In `d ≥ 2`, for a
globally Lipschitz piecewise-polynomial `σ` with a potential that is not
real-analytic, a stationary `μ` has `σ_d`-null support.

Each actual analytic polynomial branch has an original uniform-null
spherical zero set. A positive zero measure would make that branch zero
globally and force full support of the same original probability, which
contradicts the original non-analytic potential. Finitely many such zero
sets cover every actual original support point, including breakpoints.

Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, and §5.3, `thm: any.d` (i). -/
theorem piecewisePoly_any_d_measure_support_eq_zero (d : ℕ) (hd : 2 ≤ d) (β : ℝ)
    (hβ : 0 < β) (φ σ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hlip : ∃ K : ℝ≥0, LipschitzWith K σ) (hpp : IsPiecewisePolynomial σ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β σ ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  obtain ⟨K, hK⟩ := hlip
  exact piecewisePolynomial_measure_support_eq_zero hd β hβ φ σ hφ hK.continuous hpp ω a hana μ hμ

/-- The hypotheses of `piecewisePoly_any_d_measure_support_eq_zero` are
satisfiable at `d = 2`, by the same ReLU witness. -/
example :
    2 ≤ 2 ∧ (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
      (∃ K : ℝ≥0, LipschitzWith K (fun s : ℝ => max s 0)) ∧
      IsPiecewisePolynomial (fun s : ℝ => max s 0) ∧
      ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨le_rfl, one_pos, hasDerivAt_reluSq, ⟨1, lipschitzWith_relu⟩, isPiecewisePolynomial_relu,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis
      inner_basePoint_secondAxis,
    isStationary_relu_pin 1 0 (basePoint 1)⟩

end Perceptron
end Transformer
