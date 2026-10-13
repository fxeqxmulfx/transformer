/-
# Perceptrons and attention's mean-field landscape — piecewise-polynomial
  activations

Formalization of `rem:piecewise-poly` and `rem:genericity-vacuity` of
arXiv:2601.21366v2: `thm: circle` and `thm: any.d` (i) hold for every globally
Lipschitz piecewise-polynomial activation, and the linear activation
`σ(s) = s` is where a non-atomic stationary measure can still hide.

**What the source says and what is carried here.**

* "piecewise polynomial" is `IsPiecewisePolynomial`: finitely many
  breakpoints, and on every open interval avoiding them `σ` agrees with a
  polynomial.  It is stated over open *preconnected* sets, which on `ℝ` are
  exactly the intervals, so it says what it should without enumerating the
  pieces in order.  The ReLU and the leaky ReLU are instances, with the single
  breakpoint `0`.

* "globally Lipschitz" is `∃ K, LipschitzWith K σ`.

* "The same extension applies to `thm: any.d`" is read as the extension of
  `thm: any.d` (i)'s first alternative — the one that names the ReLU.  Its
  other two parts are stated for a real-analytic `σ`, which no
  piecewise-polynomial activation with a genuine breakpoint is.

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

Source: arXiv:2601.21366v2, `rem:piecewise-poly`, `rem:genericity-vacuity`.
-/

import Transformer.Perceptron.HigherDim
import Transformer.Perceptron.TransformPoly
import Transformer.Perceptron.Section5_UniformSupport
import Transformer.Perceptron.Section5_LinearStationarity

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Piecewise-polynomial activations -/

/-- **`σ` is piecewise polynomial**: there are finitely many breakpoints such
that on every open interval avoiding them `σ` is a polynomial.

Source: arXiv:2601.21366v2, `rem:piecewise-poly`. -/
def IsPiecewisePolynomial (σ : ℝ → ℝ) : Prop :=
  ∃ B : Finset ℝ, ∀ I : Set ℝ, IsOpen I → IsPreconnected I → Disjoint I (B : Set ℝ) →
    ∃ P : Polynomial ℝ, ∀ s ∈ I, σ s = P.eval s

/-- The ReLU is piecewise polynomial, with the single breakpoint `0`. -/
theorem isPiecewisePolynomial_relu : IsPiecewisePolynomial (fun s : ℝ => max s 0) := by
  refine ⟨{0}, fun I _ hconn hdisj => ?_⟩
  have h0 : (0 : ℝ) ∉ I := fun h =>
    (Set.disjoint_left.mp hdisj h) (by simp)
  rcases Set.eq_empty_or_nonempty I with hI | ⟨s₀, hs₀⟩
  · exact ⟨0, fun s hs => absurd (hI ▸ hs) (Set.notMem_empty s)⟩
  rcases lt_or_gt_of_ne (show s₀ ≠ 0 from fun h => h0 (h ▸ hs₀)) with hneg | hpos
  · refine ⟨0, fun s hs => ?_⟩
    have hs0 : s < 0 := by
      by_contra hge
      exact h0 (hconn.Icc_subset hs₀ hs ⟨hneg.le, not_lt.mp hge⟩)
    simp [max_eq_right hs0.le]
  · refine ⟨Polynomial.X, fun s hs => ?_⟩
    have hs0 : 0 < s := by
      by_contra hle
      exact h0 (hconn.Icc_subset hs hs₀ ⟨not_lt.mp hle, hpos.le⟩)
    simp [max_eq_left hs0.le]

/-- The ReLU is globally Lipschitz, with constant `1`. -/
theorem lipschitzWith_relu : LipschitzWith 1 (fun s : ℝ => max s 0) :=
  LipschitzWith.id.max_const 0

/-! ### The extension of `thm: circle` and `thm: any.d` -/

/-- **Remark (rem:piecewise-poly), for `thm: circle`.**  Let `d = 2`, `β > 0`,
and let `σ` be globally Lipschitz and piecewise polynomial — the leaky ReLU,
say.  If `v_ϑ` is not real-analytic on `𝕊¹` and `μ ∈ P(𝕊¹)` is stationary for
`E_{β,ϑ}`, then `μ` is purely atomic with finite support.

Not proved here.

Source: arXiv:2601.21366v2, `rem:piecewise-poly`. -/
theorem piecewisePoly_circle_isFinitelyAtomic (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hlip : ∃ K : ℝ≥0, LipschitzWith K σ)
    (hpp : IsPiecewisePolynomial σ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (hana : ¬ IsAnalyticOnSphere φ ω a) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β σ ω a μ) :
    IsFinitelyAtomic μ := by
  sorry

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

Not proved here.

Source: arXiv:2601.21366v2, `rem:piecewise-poly`, `thm: any.d` (i). -/
theorem piecewisePoly_any_d_measure_support_eq_zero (d : ℕ) (hd : 2 ≤ d) (β : ℝ)
    (hβ : 0 < β) (φ σ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hlip : ∃ K : ℝ≥0, LipschitzWith K σ) (hpp : IsPiecewisePolynomial σ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β σ ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  sorry

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
