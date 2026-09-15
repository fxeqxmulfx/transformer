/-
# Measure-to-measure interpolation — Disentangling supports

Formalization of §3 of arXiv:2411.04551v3:

* `Proposition prop: separation`  — disentanglement via attention with `𝐁 ≡ 0`,
* `Lemma lem: first.quadrant`     — transportation to `ℚ_1^{d-1}`,
* `Lemma lem: induction.barycenter`,
* `Lemma lem: perturbation` / `lem: colinearity`.

`eq: identity.flow` is in `Transformer.Interpolation.IdentityFlow`, where it is
proved from the two conditions that make the vector field vanish.

Every statement of this file asserts that *some* parameter curve drives
`eq: cauchy.pb` to a prescribed configuration; none is proved here.  They are
therefore `Prop`-valued definitions, with three weakenings, the same ones as in
`Transformer.Interpolation.Clustering`:

* the geodesic convex hull `conv_g` is replaced by the support of the measure —
  the conclusions are about `supp μ`, not about `conv_g supp μ`;
* the switch counts stated as `O(d · N)` become `K ≤ C * (d * N)` with `C` a
  parameter, so that the uniformity of `C` in `d`, `N` and the data — which one
  statement per `(d, N)` cannot express — has to be read off the quantifier
  order by the reader;
* `𝐁 ≡ 0` is imposed on the parameter curve rather than `eq: average.vf` being
  used in place of `eq: vf`; the two agree, since `exp ⟨0, x'⟩ = 1` turns the
  attention term of `fullVF` into the barycenter.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Clustering

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

variable (d N : ℕ)

/-- **Proposition (prop: separation).** *Disentanglement of overlapping
supports.*

For `T > 0` and `μ_0^i ∈ 𝒫(ℚ_1^{d-1})`, there is a piecewise-constant
`θ : [0, T] → Θ` with `𝐁 ≡ 0` and at most `O(d · N)` switches such that, for
all `i`, the solution `μ^i` of `eq: cauchy.pb`–`eq: average.vf` with data
`μ_0^i` and `θ` satisfies

  `conv_g supp μ^i(T) ∩ conv_g supp μ^j(T) = ∅`  for all `i ≠ j`.

Source: arXiv:2411.04551v3, §3. -/
def Separation (C : ℕ) (T : ℝ) (μ₀ : Idx N → ProbSphere d) : Prop :=
  0 < T →
  (∀ i : Idx N, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d) →
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → ProbSphere d),
      K ≤ C * (d * N) ∧ (∀ s : ℝ, (θ s).B = 0) ∧ PiecewiseConstant d θ T K ∧
      (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
      ∀ i j : Idx N, i ≠ j →
        Disjoint ((μ i T : Measure (SSphere d)).support)
          ((μ j T : Measure (SSphere d)).support)

/-- **Lemma (lem: first.quadrant).** *Transportation to `ℚ_1^{d-1}`.*

If `⋃_i supp μ_0^i ⊊ 𝕊^{d-1}`, there is a piecewise-constant
`𝐖 : [0, T] → M_{d×d}(ℝ)` with at most one switch and `‖𝐖‖_∞ ≤ C/T` (for some
`C = C(N) > 0`) such that, taking `𝐕 ≡ 𝐁 ≡ 𝐔 ≡ 0` and `b ≡ 1`, the solution
`μ^i` satisfies `supp μ^i(T) ⊂ ℚ_1^{d-1}`.

Source: arXiv:2411.04551v3, §3. -/
def FirstQuadrant (T : ℝ) (μ₀ : Idx N → ProbSphere d) : Prop :=
  0 < T →
  (⋃ i : Idx N, (μ₀ i : Measure (SSphere d)).support) ≠ Set.univ →
    ∃ C : ℝ, 0 < C ∧
      ∃ (θ : TimeParams d) (μ : Idx N → ℝ → ProbSphere d),
        PiecewiseConstant d θ T 2 ∧
        (∀ s : ℝ, (θ s).V = 0 ∧ (θ s).B = 0 ∧ (θ s).U = 0 ∧
          (θ s).b = (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1)) ∧
        (∀ s ∈ Set.Icc (0 : ℝ) T, ‖(θ s).W‖ ≤ C / T) ∧
        (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
        ∀ i : Idx N, (μ i T : Measure (SSphere d)).support ⊆ positiveQuadrant d

/-- **Lemma (lem: induction.barycenter).** *Inductive barycenter shrinking.*

If the barycenters `(𝔼_{μ_0^i}[x])_i` are pairwise non-collinear, then for any
`T, ε > 0` and any `ν_0 ∈ 𝒫(ℚ_1^{d-1})` whose barycenter is collinear with
`𝔼_{μ_0^j}[x]`, there is a piecewise-constant `θ : [0, T] → Θ` with at most
`O(d · N)` switches such that

  `conv_g supp ν(T) ∪ conv_g supp μ^j(T) ⊂ B(𝔼_{μ_0^j}[z] / ‖𝔼_{μ_0^j}[z]‖, ε)`,

while `μ^i(T) = μ_0^i` for `i ≠ j`.

Source: arXiv:2411.04551v3, §3. -/
def InductionBarycenter (C : ℕ) (T ε : ℝ) (j : Idx N)
    (μ₀ : Idx N → ProbSphere d) (ν₀ : ProbSphere d) : Prop :=
  0 < T → 0 < ε →
  (∀ i i' : Idx N, i ≠ i' → ∀ γ : ℝ, barycenter d (μ₀ i) ≠ γ • barycenter d (μ₀ i')) →
  (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d →
  (∃ γ : ℝ, barycenter d ν₀ = γ • barycenter d (μ₀ j)) →
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → ProbSphere d) (ν : ℝ → ProbSphere d),
      K ≤ C * (d * N) ∧ PiecewiseConstant d θ T K ∧
      (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
      ν 0 = ν₀ ∧ cauchyPB d θ ν ∧
      (∀ x ∈ (ν T : Measure (SSphere d)).support ∪ (μ j T : Measure (SSphere d)).support,
        ‖(x : EucSpace d) - ‖barycenter d (μ₀ j)‖⁻¹ • barycenter d (μ₀ j)‖ < ε) ∧
      ∀ i : Idx N, i ≠ j → μ i T = μ₀ i

/-- **Lemma (lem: perturbation / lem: colinearity).**

If `𝔼_{μ_0}[x] = γ_1 𝔼_{ν_0}[x]` for some `γ_1 ∈ (0, 1]`, one can choose
`(𝐕, 𝐖, 𝐔, b)` — piecewise constant with at most two switches — so that the
barycenters of the corresponding solutions are no longer collinear.

Source: arXiv:2411.04551v3, §3. -/
def Perturbation (T γ₁ : ℝ) (μ₀ ν₀ : ProbSphere d) : Prop :=
  0 < T → 0 < γ₁ → γ₁ ≤ 1 →
  barycenter d μ₀ = γ₁ • barycenter d ν₀ →
    ∃ (θ : TimeParams d) (μ ν : ℝ → ProbSphere d),
      PiecewiseConstant d θ T 3 ∧ (∀ s : ℝ, (θ s).B = 0) ∧
      μ 0 = μ₀ ∧ ν 0 = ν₀ ∧ cauchyPB d θ μ ∧ cauchyPB d θ ν ∧
      ∀ γ : ℝ, barycenter d (μ T) ≠ γ • barycenter d (ν T)

end Interpolation
end Transformer
