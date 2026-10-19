module

public import Transformer.Modes.Section1_KDE
public import Mathlib.MeasureTheory.Measure.Prod
public import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral

/-!
# The number of modes of a Gaussian KDE — the printed Kac–Rice hypotheses

§2.1 of arXiv:2412.09080v3, `sec:kac-rice`: upcrossings, the modulus of
continuity, and the unchanged hypotheses of `thm:kac-rice`.

**What the source says and what is carried here.**

* "`Ψ` has an upcrossing of level `u` at `t`" is `Ψ(t) = u` and `Ψ'(t) > 0`;
  `U_u(Ψ, T)` is the cardinality of the set of them in `T`, taken in `ℝ≥0∞` so
  that a path with infinitely many upcrossings counts as infinite.

* The four hypotheses of `thm:kac-rice` are the four fields of
  `IsKacRiceField`. "Finite variance over `T`" is `MemLp … 2`, a finite second
  moment, which for a probability measure is finite variance. "The law admits
  a density `p_t`" is written as an equality of measures, `map Ψ(t) ℙ =
  volume.withDensity p_t`, so that `p_t` is an explicit function and the
  conclusion can name it. "Continuous for `t ∈ T` and `x` in a neighbourhood
  of `u`" is `ContinuousOn` on `T ×ˢ V` for some `V ∈ 𝓝 u`.

* The claimed equality `eq:krf` is refuted in `Section2_KacRiceRefutation`.
  Its two sides use `lintegral`, retaining the original nonnegative
  integral and allowing infinite values. The counterexample has finite
  values on both sides, so no extended-real ambiguity causes the failure.

**The printed general claim is false.** The source's fourth condition is
`P(ω(η)>ε)=O(η)` for every `ε>0`. This exact condition is retained below.
The closed theorem `kacRice_counterexample`, together with `not_kacRice`,
proves that all four original hypotheses can hold while `eq:krf` fails.
The unproved universal equality is therefore replaced by that counterexample,
with no stronger hypothesis silently inserted under the paper's name.

The construction uses an independent uniform signed scale `R` in `(-1,1)`
and a uniform complex point `Z` in the open unit disk. Its field is

  `F(t) = R * (2 + re (Z * exp (i*t/R)))`.

The scale-zero sample is a genuine null set under the normalized interval
law. At every other sample in the disk, `2+re(...)` is positive, so the
entire smooth path has the sign of its scale and never meets level zero.
The original expected upcrossing count is consequently zero on `[0,1]`.

The actual field and derivative satisfy uniform almost-sure amplitude
bounds of three and one respectively. Their finite second moments hold
at every real time, and every sample path is smooth on the real line.
The actual derivative modulus is bounded by `η/|R|` on the disk. Hence a
modulus exceeding `ε` implies `|R|<η/ε`, whose original initialization
probability is at most `η/ε`. This proves precisely the printed `O(η)`
condition, without changing its quantifiers or its global time modulus.

For each fixed time, the scale-dependent phase rotation preserves the
actual disk probability law. Fubini therefore identifies the joint law
of the actual value and actual derivative with that of

  `(R,Z) ↦ (R*(2+re Z), -im Z)`.

The full joint density is derived from this pushforward by volume-preserving
complex coordinates, the positive scale stretch, and Tonelli. Integrating
out its latent disk coordinate gives a density bounded by `1/π`. The real
scalar density is its actual derivative-coordinate marginal and is bounded
by `2/π`. Both functions are proved densities on their full observation
spaces, rather than being specified only near the distinguished level.

For `|x|<1`, the scale stretch admits every actual disk point. The joint
density is then independent of `x` and equals the normalized integral of
`1/(2*(2+u))` from `-sqrt(1-y²)` to `sqrt(1-y²)`. Its moving endpoints
are continuous even when the slice vanishes. This proves continuity for
all real derivative coordinates; the actual scalar density is constant
near level zero and has the original required joint time-value continuity.

At level zero the actual joint density is strictly positive for every
derivative coordinate in `(0,1)`. Thus the original right side of `eq:krf`
on `[0,1]` is positive and finite, while the original left side is zero.
`isKacRiceField_riceCounter` verifies the complete hypothesis structure
for this same probability law, field, and derived densities. The final
counterexample also supplies a compact time set of positive volume.

Source: arXiv:2412.09080v3, `thm:kac-rice`, `eq:krf`
(Azaïs-Wschebor, p. 62; Adler-Taylor, §11.1).
-/

@[expose] public section

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

end Modes
end Transformer
