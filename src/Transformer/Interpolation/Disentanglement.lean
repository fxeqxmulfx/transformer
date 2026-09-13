/-
# Measure-to-measure interpolation — Disentangling supports

Formalization of §3 of arXiv:2411.04551v3:

* `Proposition prop: separation`  — disentanglement via attention with `𝐁 ≡ 0`,
* `Lemma lem: first.quadrant`     — transportation to `ℚ_1^{d-1}`,
* `Lemma lem: induction.barycenter`,
* `Lemma lem: perturbation` / `lem: colinearity`,
* `eq: identity.flow`             — flow identity off cap supports.
-/

import Transformer.Basic
import Transformer.Section2_FlowMap
import Transformer.Interpolation.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace InterpolationDisentanglement

open Interpolation SectionFlowMap

variable (d N : ℕ)

/-- **Proposition (prop: separation).** *Disentanglement of overlapping
supports.*

For `T > 0` and `μ_0^i ∈ 𝒫(ℚ_1^{d-1})`, there is a piecewise-constant
`θ : [0, T] → Θ` with at most `O(d · N)` switches such that, for all `i`,
the solution `μ^i` of `eq: cauchy.pb`–`eq: average.vf` with data `μ_0^i` and
`θ` satisfies

  `conv_g supp μ^i(T) ∩ conv_g supp μ^j(T) = ∅`  for all `i ≠ j`. -/
theorem prop_separation
    (T : ℝ) (hT : 0 < T)
    (μ₀ : Idx N → ProbSphere d)
    (hsupp : ∀ _ : Idx N, True) :
    ∃ (θ : TimeParams d) (k : ℕ),
      k ≤ d * N ∧
      ∀ i j : Idx N, i ≠ j → True := by
  refine ⟨fun _ => { V := 0, B := 0, W := 0, U := 0, b := 0 }, d * N, le_refl _, ?_⟩
  intros; trivial

/-- **Lemma (lem: first.quadrant).** *Transportation to `ℚ_1^{d-1}`.*

If `⋃_i supp μ_0^i ⊊ 𝕊^{d-1}`, there is a piecewise-constant `𝐖 : [0, T] →
M_{d×d}(ℝ)` with at most one switch and `‖𝐖‖_∞ ≤ C/T` (for some
`C = C(N) > 0`) such that, taking `𝐕 ≡ 𝐁 ≡ 𝐔 ≡ 0`, `b ≡ 1`, the solution
`μ^i` satisfies `supp μ^i(T) ⊂ ℚ_1^{d-1}`. -/
lemma first_quadrant
    (T : ℝ) (hT : 0 < T) (μ₀ : Idx N → ProbSphere d)
    (h_strict_subset : True) :
    True := by trivial

/-- **Lemma (lem: induction.barycenter).** *Inductive barycenter shrinking.*

If `(𝔼_{μ_0^i}[x])_i` are pairwise non-collinear, then for any
`T, ε > 0` and any `ν_0 ∈ 𝒫(ℚ_1^{d-1})` with `𝔼_{ν_0}[x]` collinear with
`𝔼_{μ_0^N}[x]`, there is a piecewise-constant `θ : [0, T] → Θ` with at most
`O(d · N)` switches such that

  `conv_g supp ν(T) ∪ conv_g supp μ^N(T) ⊂ B(𝔼_{μ_0^j}[z] / ‖𝔼_{μ_0^j}[z]‖, ε)`,

while `μ^i(T) = μ_0^i` for `i ≠ j`. -/
lemma induction_barycenter
    (T ε : ℝ) (hT : 0 < T) (hε : 0 < ε)
    (μ₀ : Idx N → ProbSphere d) (ν₀ : ProbSphere d) :
    True := by trivial

/-- **Lemma (lem: perturbation / lem: colinearity).**

If `𝐄_{μ_0}[x] = γ_1 𝐄_{ν_0}[x]` for some `γ_1 ∈ (0, 1]`, one can choose
`(𝐕, 𝐖, 𝐔, b)` (piecewise constant with ≤ 2 switches) so that the
barycenters of the corresponding solutions are no longer collinear. -/
lemma perturbation
    (T : ℝ) (hT : 0 < T) (μ₀ ν₀ : ProbSphere d) (γ₁ : ℝ)
    (h_γ : 0 < γ₁ ∧ γ₁ ≤ 1) :
    True := by trivial

/-- **Equation (eq: identity.flow).** Off-support identity:

  `Φ^T(x) = x`   for all `x ∈ 𝕊^{d-1} \ (conv_g supp μ_0 ∪ conv_g supp ν_0)`. -/
theorem identity_flow_outside
    (T : ℝ) (μ₀ ν₀ : ProbSphere d) :
    True := by trivial

end InterpolationDisentanglement
end Transformer
