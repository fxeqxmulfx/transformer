/-
# Measure-to-measure interpolation — Matching discrete measures (Neural ODE part)

Formalization of §4 of arXiv:2411.04551v3:

* `eq: neural.ode.sphere`              — neural-ODE-only flow on the sphere,
* `Proposition prop: interpolation.neural.ode`,
* `Proposition lem: induction.neural.ode`,
* `eq: estimate.neural` — the exponential convergence of the perceptron-only
  flow, `eq: Hartman.Grobman`, is `Transformer.Interpolation.Settling`,
* `eq: sphere.separation`, `eq: geodesic.toll`, `eq: identity.flow.symm`.
-/

import Transformer.Basic
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Clustering

open scoped BigOperators
open Real

namespace Transformer
namespace Interpolation

open Interpolation

variable (d M : ℕ)

/-- The right-hand side of `eq: neural.ode.sphere`,
`Proj_x 𝐖(t) (𝐔(t) x + b(t))_+`. -/
noncomputable def neuralVF
    (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) (t : ℝ) (x : EucSpace d) :
    EucSpace d :=
  proj d x ((W t) (EuclideanSpace.equiv _ ℝ |>.symm
    (fun k => max ((EuclideanSpace.equiv _ ℝ ((U t) x + b t)) k) 0)))

/-- **Equation (eq: neural.ode.sphere).**

  `ẋ^i(t) = Proj_x 𝐖(t) (𝐔(t) x^i(t) + b(t))_+`,

in integrated form, `x^i(t) = x^i(0) + ∫_0^t Proj_x 𝐖(s)(𝐔(s) x^i(s) + b(s))_+ ds`
with an integrable integrand: the parameters are piecewise constant, and at a
switch `x^i` has a right and a left derivative but in general no derivative, as
for `Interpolation.IsCharacteristic`.  The initial condition is imposed by the
statements.

Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
def neuralODESphere
    (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (x : ℝ → Idx M → EucSpace d) : Prop :=
  ∀ t : ℝ, ∀ i : Idx M,
    IntervalIntegrable (fun s => neuralVF d W U b s (x s i)) MeasureTheory.volume 0 t ∧
      x t i = x 0 i + ∫ s in (0 : ℝ)..t, neuralVF d W U b s (x s i)

/-- The parameter curve `(𝐕, 𝐁, 𝐖, 𝐔, b) = (0, 0, 𝐖, 𝐔, b)` of §4, through which
`PiecewiseConstant` applies to `(𝐖, 𝐔, b)`. -/
def neuralParams (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) : TimeParams d :=
  fun t => { V := 0, B := 0, W := W t, U := U t, b := b t }

/-- **Proposition (prop: interpolation.neural.ode).**

For `d ≥ 3` and data `(x_0^i, y^i) ∈ 𝕊^{d-1} × 𝕊^{d-1}` with
`x_0^i ≠ x_0^j`, `y^i ≠ y^j` for `i ≠ j`, and for every `i` some `γ_i ∈ 𝕊^{d-1}`,
`ε_i > 0` with `⟨γ_i, x_0^i - y^i⟩ = 0` and `x_0^j ∉ H_{ε_i}^{γ_i}` for `j ≠ i`:
for any `T > 0` there is a piecewise-constant `θ = (𝐖, 𝐔, b) : [0, T] →
M_{d×d}² × ℝ^d` with at most `6M` switches such that the solution of
`eq: neural.ode.sphere` satisfies `x^i(T) = y^i`, and
`‖θ‖_∞ ≤ C M / (T min_i ε_i)` with `C` depending neither on the data nor on `T`.

`C` is chosen after `d`, which makes the choice of norm on `Θ` immaterial;
`‖θ‖` is the largest of the operator norms of `𝐖`, `𝐔` and the norm of `b`.
`M ≥ 1`, since `min_i ε_i` is taken over the indices.  "The solution" is: one
exists, and every solution ends at the targets.

Not proved here.

Source: arXiv:2411.04551v3, §4, `prop: interpolation.neural.ode`. -/
theorem prop_interpolation_neural_ode (hd : 3 ≤ d) :
    ∃ C : ℝ, 0 < C ∧ ∀ (M : ℕ) (x₀ y γ : Idx M → SSphere d) (ε : Idx M → ℝ), 1 ≤ M →
      (∀ i j : Idx M, i ≠ j → x₀ i ≠ x₀ j) → (∀ i j : Idx M, i ≠ j → y i ≠ y j) →
      (∀ i, 0 < ε i) →
      (∀ i, inner (𝕜 := ℝ) ((γ i : EucSpace d))
        ((x₀ i : EucSpace d) - (y i : EucSpace d)) = 0) →
      (∀ i j : Idx M, i ≠ j → x₀ j ∉ Hε d (γ i) (ε i)) →
      ∀ T : ℝ, 0 < T →
      ∃ (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d),
        PiecewiseConstant d (neuralParams d W U b) T (6 * M + 1) ∧
        (∀ s ∈ Set.Icc (0 : ℝ) T,
          max (max ‖W s‖ ‖U s‖) ‖b s‖ ≤ C * M / (T * ⨅ i, ε i)) ∧
        (∃ x : ℝ → Idx M → EucSpace d,
          (∀ i, x 0 i = (x₀ i : EucSpace d)) ∧ neuralODESphere d M W U b x) ∧
        ∀ x : ℝ → Idx M → EucSpace d,
          (∀ i, x 0 i = (x₀ i : EucSpace d)) →
          neuralODESphere d M W U b x →
          ∀ i, x T i = (y i : EucSpace d) := by
  sorry

/-- The hypotheses of `prop_interpolation_neural_ode` are satisfiable: one
pair, `x_0 = y = basePoint 2` on `𝕊^2`, `ε = 1`, which leaves nothing to
separate. -/
example :
    3 ≤ 3 ∧ 1 ≤ 1 ∧
      (∀ i j : Idx 1, i ≠ j →
        (fun _ => basePoint 2 : Idx 1 → SSphere 3) i
          ≠ (fun _ => basePoint 2 : Idx 1 → SSphere 3) j) ∧
      (∀ i : Idx 1, (0 : ℝ) < (fun _ => 1 : Idx 1 → ℝ) i) ∧
      (∀ i : Idx 1, inner (𝕜 := ℝ) ((basePoint 2 : EucSpace 3))
        (((fun _ => basePoint 2 : Idx 1 → SSphere 3) i : EucSpace 3)
          - ((fun _ => basePoint 2 : Idx 1 → SSphere 3) i : EucSpace 3)) = 0) ∧
      (∀ i j : Idx 1, i ≠ j →
        (fun _ => basePoint 2 : Idx 1 → SSphere 3) j ∉ Hε 3 (basePoint 2) 1) ∧
      (0 : ℝ) < 1 :=
  ⟨le_rfl, le_rfl, fun i j h => absurd (Subsingleton.elim i j) h, fun _ => one_pos,
    fun _ => by simp, fun i j h => absurd (Subsingleton.elim i j) h, one_pos⟩

/-- **Proposition (lem: induction.neural.ode).** *Induction step.*

For `d ≥ 3` and data `(x_0^i, y^i)` with `x_0^i ≠ x_0^j`, `y^i ≠ y^j` for
`i ≠ j`, `x_0^i = y^i` for `i < M`, and some `γ ∈ 𝕊^{d-1}`, `ε > 0` with
`⟨γ, x_0^M - y^M⟩ = 0` and `x_0^i ∉ H_ε^γ` for `i < M`: for any `T > 0` there is
a piecewise-constant `θ = (𝐖, 𝐔, b)` with at most `6` switches such that the
solution of `eq: neural.ode.sphere` satisfies `x^i(T) = y^i`, and
`‖θ‖_∞ ≤ C / (T ε)` with `C` depending neither on the data nor on `T`.

The data is indexed by `Idx (M + 1)`, so the moving pair is `Fin.last M`.  `C`,
the norm and "the solution" are read as in `prop_interpolation_neural_ode`.

Not proved here.

Source: arXiv:2411.04551v3, §4, `lem: induction.neural.ode`. -/
theorem lem_induction_neural_ode (hd : 3 ≤ d) :
    ∃ C : ℝ, 0 < C ∧ ∀ (M : ℕ) (x₀ y : Idx (M + 1) → SSphere d) (γ : SSphere d) (ε : ℝ),
      (∀ i j : Idx (M + 1), i ≠ j → x₀ i ≠ x₀ j) →
      (∀ i j : Idx (M + 1), i ≠ j → y i ≠ y j) →
      (∀ i : Idx (M + 1), i ≠ Fin.last M → x₀ i = y i) → 0 < ε →
      inner (𝕜 := ℝ) ((γ : EucSpace d))
        ((x₀ (Fin.last M) : EucSpace d) - (y (Fin.last M) : EucSpace d)) = 0 →
      (∀ i : Idx (M + 1), i ≠ Fin.last M → x₀ i ∉ Hε d γ ε) →
      ∀ T : ℝ, 0 < T →
      ∃ (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d),
        PiecewiseConstant d (neuralParams d W U b) T 7 ∧
        (∀ s ∈ Set.Icc (0 : ℝ) T, max (max ‖W s‖ ‖U s‖) ‖b s‖ ≤ C / (T * ε)) ∧
        (∃ x : ℝ → Idx (M + 1) → EucSpace d,
          (∀ i, x 0 i = (x₀ i : EucSpace d)) ∧ neuralODESphere d (M + 1) W U b x) ∧
        ∀ x : ℝ → Idx (M + 1) → EucSpace d,
          (∀ i, x 0 i = (x₀ i : EucSpace d)) →
          neuralODESphere d (M + 1) W U b x →
          ∀ i, x T i = (y i : EucSpace d) := by
  sorry

/-- The hypotheses of `lem_induction_neural_ode` are satisfiable: at `M = 0`
there is a single pair, no index beside it, so distinctness, `x_0^i = y^i` and
the separation are vacuous, and `x_0 = y` makes the orthogonality an inner
product with the zero vector. -/
example :
    3 ≤ 3 ∧
      (∀ i j : Idx 1, i ≠ j →
        (fun _ => basePoint 2 : Idx 1 → SSphere 3) i
          ≠ (fun _ => basePoint 2 : Idx 1 → SSphere 3) j) ∧
      (∀ i : Idx 1, i ≠ Fin.last 0 →
        (fun _ => basePoint 2 : Idx 1 → SSphere 3) i
          = (fun _ => basePoint 2 : Idx 1 → SSphere 3) i) ∧
      (0 : ℝ) < 1 ∧
      inner (𝕜 := ℝ) ((basePoint 2 : EucSpace 3))
        (((fun _ => basePoint 2 : Idx 1 → SSphere 3) (Fin.last 0) : EucSpace 3)
          - ((fun _ => basePoint 2 : Idx 1 → SSphere 3) (Fin.last 0) : EucSpace 3))
        = (0 : ℝ) ∧
      (∀ i : Idx 1, i ≠ Fin.last 0 →
        (fun _ => basePoint 2 : Idx 1 → SSphere 3) i ∉ Hε 3 (basePoint 2) 1) ∧
      (0 : ℝ) < 1 :=
  ⟨le_rfl, fun i j h => absurd (Subsingleton.elim i j) h, fun _ _ => rfl, one_pos,
    by simp, fun i hi => absurd (Subsingleton.elim i (Fin.last 0)) hi, one_pos⟩

/-! ### `eq: Hartman.Grobman`

The exponential settling of the perceptron-only flow onto its attractor is
`Transformer.Interpolation.Hartman_Grobman`, in
`Transformer.Interpolation.HartmanGrobman`, proved for the flow
`eq: neural.ode.separation` of Step 2.  The refutation of the form it had
here — which quantified over every path on the sphere, and was false — is in
`Transformer.Interpolation.Settling`.
-/

end Interpolation
end Transformer
