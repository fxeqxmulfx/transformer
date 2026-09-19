/-
# Appendix A — Proof of Theorem (p:beta0), part 2

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes the second half of Appendix A of the survey:

* `e:helpcl`            — the Hessian of `𝖤_0` at a critical point,
* `Lemma lem: yury.lemma` — every non-trivial critical point of `𝖤_0` is a
                            strict saddle,
* `Lemma l:nosaddleconv`  — gradient ascent avoids strict saddles,
* the assembly of `Theorem p:beta0`.

The energy itself, its gradient flow and `eq: taylor` are in
`Perspective.AppendixA_Beta0`; `e:russiantrick` and `IsSkew` are in
`Perspective.RussianTrick`, where the identity is proved.  The statements below
are written out in full and none of them is proved: each is a theorem closed by
`sorry`.  `PerturbationBy` and `SecondDerivE0At` are predicates of their
arguments and stay definitions.
-/

import Transformer.Perspective.AppendixA_Beta0
import Transformer.Perspective.RussianTrick

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The perturbation of `X` used in `e:helpcl`:

  `x_i(t) = e^{tB} x_i` for `i ∈ 𝒮`,   `x_i(t) = x_i` otherwise.

The rotated particles are described by the differential equation
`ẋ_i(t) = B x_i(t)` they solve rather than by the matrix exponential, which is
the same thing for the initial condition `Y 0 = X` and keeps the statement
inside the `HasDerivAt` API used everywhere else here. -/
def PerturbationBy
    (B : ParamMatrix d) (𝒮 : Finset (Idx n))
    (X : SphereTuple d n) (Y : ℝ → SphereTuple d n) : Prop :=
  Y 0 = X ∧
  (∀ i ∈ 𝒮, ∀ t : ℝ,
    HasDerivAt (fun s => (Y s i : EucSpace d)) (B ((Y t i : EucSpace d))) t) ∧
  (∀ i ∉ 𝒮, ∀ t : ℝ, (Y t i : EucSpace d) = (X i : EucSpace d))

/-- `c` is the second derivative of `t ↦ 𝖤_0(Y(t))` at `t = 0`: the energy is
differentiable along the whole curve, and its derivative is again
differentiable at `0`, with derivative `c`. -/
def SecondDerivE0At (Y : ℝ → SphereTuple d n) (c : ℝ) : Prop :=
  ∃ f' : ℝ → ℝ,
    (∀ t : ℝ, HasDerivAt (fun s => E0 d n (Y s)) (f' t) t) ∧ HasDerivAt f' c 0

/-- **Equation (e:helpcl).** *Hessian of `𝖤_0` at a critical point.*

For a skew-symmetric `B`, a subset `𝒮 ⊂ [n]`, and the perturbation
`x_i(t) = e^{tB} x_i` (`i ∈ 𝒮`), `x_i(t) = x_i` (`i ∉ 𝒮`) of a critical point,

  `𝖤_0''(0) = (2/n) Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨B² x_i, x_j⟩`.

The first-order term is absent precisely because `X` is critical, which is why
criticality is a hypothesis rather than decoration.

Not proved here: the second-order expansion is not carried out.

Source: arXiv:2312.10794v5, Appendix A, `e:helpcl`. -/
theorem hessian_at_critical
    (X : SphereTuple d n) (B : ParamMatrix d) (𝒮 : Finset (Idx n))
    (hB : IsSkew d B) (hX : IsCriticalE0 d n X)
    (Y : ℝ → SphereTuple d n) (hY : PerturbationBy d n B 𝒮 X Y) :
    SecondDerivE0At d n Y
      ((2 * (n : ℝ)⁻¹) * ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
        inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d))) := by
  sorry

/-- The hypotheses of `hessian_at_critical` are satisfiable: the zero matrix is
skew, the antipodal pair is a critical point
(`antipodalPair_critical_nonTrivial`), and rotating none of its particles —
`𝒮 = ∅` — leaves the constant curve as the perturbation. -/
example :
    IsSkew 1 0 ∧ IsCriticalE0 1 2 (antipodalPair 1 northPole) ∧
      PerturbationBy 1 2 0 ∅ (antipodalPair 1 northPole)
        (fun _ => antipodalPair 1 northPole) :=
  ⟨fun x y => by simp, antipodalPair_critical_nonTrivial.1,
    rfl, fun i hi => absurd hi (Finset.notMem_empty i), fun _ _ _ => rfl⟩

/-- **Lemma (lem: yury.lemma).** *Every non-trivial critical point of `𝖤_0` is
a strict saddle.*

Strictness is spelled out as: there is a rotation direction `B` and a subset
`𝒮` whose perturbation has *positive* second derivative of the energy at
`t = 0`.  Since `𝖤_0` is being maximized along `e:gradfl`, such a direction
means the critical point is not a local maximum — and hence, by
`no_saddle_convergence`, is reached from a null set of initial data.  In
particular every local maximum of `𝖤_0` is a consensus configuration, hence a
global maximum.

Not proved here: the proof combines `eq: taylor`, `hessian_at_critical` and
`russian_trick`; only the last of the three is proved.

Source: arXiv:2312.10794v5, Appendix A, `lem: yury.lemma`. -/
theorem yury_lemma (X : SphereTuple d n) (hcrit : IsCriticalE0 d n X)
    (hnt : NonTrivialTuple d n X) :
    ∃ (B : ParamMatrix d) (𝒮 : Finset (Idx n)) (Y : ℝ → SphereTuple d n) (c : ℝ),
      IsSkew d B ∧ PerturbationBy d n B 𝒮 X Y ∧ SecondDerivE0At d n Y c ∧ 0 < c := by
  sorry

/-- The hypotheses of `yury_lemma` are satisfiable: the antipodal pair is a
non-trivial critical point. -/
example : IsCriticalE0 1 2 (antipodalPair 1 northPole) ∧
    NonTrivialTuple 1 2 (antipodalPair 1 northPole) :=
  antipodalPair_critical_nonTrivial

/-- **Lemma (l:nosaddleconv).** *No-saddle-convergence lemma.*

On a compact Riemannian manifold the set of initial conditions whose
gradient-ascent trajectory converges to a strict saddle of a smooth `f` has
zero volume (center-stable manifold theorem).  It is stated here in the only
instance the survey uses it: `ℳ = (𝕊^{d-1})^n`, `f = 𝖤_0`, and — by
`YuryLemma` — the strict saddles are the non-trivial critical points.  Volume
is the uniform measure `UniformTuple`.

Not proved here: the center-stable manifold theorem is not available.

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv`. -/
theorem no_saddle_convergence :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      P { X₀ : SphereTuple d n |
          ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ E0GradientAscent d n X ∧
            ∃ Z : SphereTuple d n, IsCriticalE0 d n Z ∧ NonTrivialTuple d n Z ∧
              ∀ i : Idx n,
                Filter.Tendsto
                  (fun t : ℝ => ((X t i : EucSpace d) - (Z i : EucSpace d)))
                  Filter.atTop (nhds 0) } = 0 := by
  sorry

/-- **Theorem (p:beta0)**, in the almost-sure form Appendix A proves: for
`d, n ≥ 2` the set of initial sequences whose `β = 0` trajectory does *not*
converge to a single point is null for the uniform law on `(𝕊^{d-1})^n`.

This is the statement `Perspective.beta0_consensus` should have; the latter
quantifies over every initial sequence, which is false at the exceptional
null set.

Not proved here.  The paper's proof assembles three ingredients, and each of
them is a `sorry` of its own: (i) Łojasiewicz — `𝖤_0` is analytic on a compact
analytic manifold, so every trajectory of `e:gradfl` converges to a critical
point; (ii) `no_saddle_convergence` — the non-trivial critical points are
reached from a null set; (iii) `yury_lemma` — those are exactly the strict
saddles.  The assembly is not recorded as a statement of its own: with the
conclusion already a sorried theorem, the implication would be provable in one
line and would assert nothing.

Source: arXiv:2312.10794v5, §4, `p:beta0`; Appendix A for the proof. -/
theorem almost_sure_consensus_beta0 (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      P { X₀ : SphereTuple d n | ¬ ∃ x_star : SSphere d,
            ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → beta0Dynamics d n X →
              ∀ i : Idx n,
                Filter.Tendsto
                  (fun t : ℝ => ((X t i : EucSpace d) - (x_star : EucSpace d)))
                  Filter.atTop (nhds 0) } = 0 := by
  sorry

/-- The hypotheses of `almost_sure_consensus_beta0` are satisfiable:
`d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

end Perspective
end Transformer
