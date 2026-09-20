/-
# The number of modes of a Gaussian KDE — the Kac-Rice formula

§2.1 of arXiv:2412.09080v3, `sec:kac-rice`: upcrossings, the modulus of
continuity, and `thm:kac-rice` itself.

**What the source says and what is carried here.**

* "`Ψ` has an upcrossing of level `u` at `t`" is `Ψ(t) = u` and `Ψ'(t) > 0`;
  `U_u(Ψ, T)` is the cardinality of the set of them in `T`, taken in `ℝ≥0∞` so
  that a path with infinitely many upcrossings counts as infinite.

* The four hypotheses of `thm:kac-rice` are the four fields of
  `IsKacRiceField`.  "Finite variance over `T`" is `MemLp … 2`, a finite second
  moment, which for a probability measure is finite variance.  "The law admits
  a density `p_t`" is written as an equality of measures, `map Ψ(t) ℙ =
  volume.withDensity p_t`, so that `p_t` is an explicit function and the
  conclusion can name it.  "Continuous for `t ∈ T` and `x` in a neighbourhood
  of `u`" is `ContinuousOn` on `T ×ˢ V` for some `V ∈ 𝓝 u`.

* The conclusion `eq:krf` is written with `lintegral`, which needs no
  integrability side condition and is exact when both sides are infinite.

Source: arXiv:2412.09080v3, `thm:kac-rice`, `eq:krf`
(Azaïs-Wschebor, p. 62; Adler-Taylor, §11.1).
-/

import Transformer.Modes.Section1_KDE
import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory Filter Topology Asymptotics

namespace Transformer
namespace Modes

/-! ### Upcrossings -/

/-- **An upcrossing of level `u`.**  `Ψ : ℝ → ℝ` has one at `t` if `Ψ(t) = u`
and `Ψ'(t) > 0`.

Source: arXiv:2412.09080v3, `sec:kac-rice`. -/
def IsUpcrossing (Ψ : ℝ → ℝ) (u t : ℝ) : Prop := Ψ t = u ∧ 0 < deriv Ψ t

/-- The set of upcrossings of `Ψ` at level `u` inside `T`. -/
def upcrossingSet (Ψ : ℝ → ℝ) (u : ℝ) (T : Set ℝ) : Set ℝ := {t ∈ T | IsUpcrossing Ψ u t}

/-- `U_u(Ψ, T)`: the number of upcrossings of `Ψ` at level `u` in `T`, infinite
if there are infinitely many.

Source: arXiv:2412.09080v3, `thm:kac-rice`. -/
noncomputable def upcrossingCount (Ψ : ℝ → ℝ) (u : ℝ) (T : Set ℝ) : ℝ≥0∞ :=
  ((upcrossingSet Ψ u T).encard : ℝ≥0∞)

/-- `𝔼 U_u(Ψ, T)`, the expectation over the randomness of `Ψ`. -/
noncomputable def expectedUpcrossings {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Ψ : Ω → ℝ → ℝ) (u : ℝ) (T : Set ℝ) : ℝ≥0∞ :=
  ∫⁻ ω, upcrossingCount (Ψ ω) u T ∂P

/-- The modulus of continuity of `f`:
`ω(η) = sup_{t, s : |t - s| ≤ η} |f(t) - f(s)|`.

Source: arXiv:2412.09080v3, `thm:kac-rice`, footnote. -/
noncomputable def modulusOfContinuity (f : ℝ → ℝ) (η : ℝ) : ℝ≥0∞ :=
  ⨆ t : ℝ, ⨆ s : ℝ, ⨆ _ : |t - s| ≤ η, ENNReal.ofReal |f t - f s|

/-- A constant function has vanishing modulus of continuity. -/
@[simp] theorem modulusOfContinuity_const (c η : ℝ) :
    modulusOfContinuity (fun _ => c) η = 0 := by
  simp [modulusOfContinuity]

/-! ### The hypotheses of `thm:kac-rice` -/

/-- **The four hypotheses of `thm:kac-rice`** for a random `Ψ : Ω → ℝ → ℝ`, a
level `u` and a compact `T`, with `p1` the density of `Ψ(t)` and `p` the joint
density of `(Ψ(t), Ψ'(t))`.

Source: arXiv:2412.09080v3, `thm:kac-rice`, items 1-4. -/
structure IsKacRiceField {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Ψ : Ω → ℝ → ℝ) (u : ℝ) (T : Set ℝ) (p1 : ℝ → ℝ → ℝ) (p : ℝ → ℝ → ℝ → ℝ) : Prop where
  /-- (1) `Ψ` is a.s. in `𝒞¹(ℝ)`. -/
  contDiff : ∀ᵐ ω ∂P, ContDiff ℝ 1 (Ψ ω)
  /-- (1) `Ψ` has finite variance over `T`. -/
  memLp : ∀ t ∈ T, MemLp (fun ω => Ψ ω t) 2 P
  /-- (1) `Ψ'` has finite variance over `T`. -/
  memLp_deriv : ∀ t ∈ T, MemLp (fun ω => deriv (Ψ ω) t) 2 P
  /-- (2) The law of `Ψ(t)` admits the density `p_t^{[1]}`. -/
  map_eq : ∀ t ∈ T, Measure.map (fun ω => Ψ ω t) P
    = volume.withDensity fun x => ENNReal.ofReal (p1 t x)
  /-- (2) `p_t^{[1]}(x)` is continuous for `t ∈ T` and `x` near `u`. -/
  continuousOn : ∃ V ∈ 𝓝 u, ContinuousOn (fun q : ℝ × ℝ => p1 q.1 q.2) (T ×ˢ V)
  /-- (3) The joint law of `(Ψ(t), Ψ'(t))` admits the density `p_t(x, y)`. -/
  map_joint_eq : ∀ t ∈ T, Measure.map (fun ω => (Ψ ω t, deriv (Ψ ω) t)) P
    = volume.withDensity fun z : ℝ × ℝ => ENNReal.ofReal (p t z.1 z.2)
  /-- (3) `p_t(x, y)` is continuous for `t ∈ T`, `x` near `u` and every `y`. -/
  continuousOn_joint : ∃ V ∈ 𝓝 u, ContinuousOn (fun q : ℝ × ℝ × ℝ => p q.1 q.2.1 q.2.2)
    (T ×ˢ V ×ˢ (Set.univ : Set ℝ))
  /-- (4) `ℙ(ω(η) > ε) = O(η)` as `η ↘ 0⁺`, for the modulus of continuity of
  `Ψ'`. -/
  modulus : ∀ ε > (0 : ℝ),
    (fun η : ℝ => (P {ω | ENNReal.ofReal ε < modulusOfContinuity (deriv (Ψ ω)) η}).toReal)
      =O[𝓝[>] (0 : ℝ)] fun η : ℝ => η

/-! ### The formula -/

/-- **Theorem (thm:kac-rice), the Kac-Rice formula.**  Under the four
hypotheses above, the expected number of upcrossings of level `u` in `T` is

  `𝔼 U_u(Ψ, T) = ∫_T ∫_0^∞ y p_t(u, y) dy dt`.

Not proved here: it is quoted from Azaïs-Wschebor and Adler-Taylor.

Source: arXiv:2412.09080v3, `thm:kac-rice`, `eq:krf`. -/
theorem kacRice {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (Ψ : Ω → ℝ → ℝ) (u : ℝ) (T : Set ℝ) (hT : IsCompact T) (p1 : ℝ → ℝ → ℝ)
    (p : ℝ → ℝ → ℝ → ℝ) (hΨ : IsKacRiceField P Ψ u T p1 p) :
    expectedUpcrossings P Ψ u T
      = ∫⁻ t in T, ∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal (y * p t u y) := by
  sorry
end Modes
end Transformer
