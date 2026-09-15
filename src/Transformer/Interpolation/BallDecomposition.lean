/-
# Measure-to-measure interpolation — Decomposing a measure into equal-mass balls

Formalization of `Claim cl: balls` of arXiv:2411.04551v3, the combinatorial
step in the proof of `thm: main.result`: once the measures `ν^i` have pairwise
separated supports, each of them can be cut by a nested family of balls of a
common radius into `M` shells of mass `1/M`, in such a way that the family
belonging to one measure is invisible to the others.

The survey obtains the separation from the smallness of the diameters
(`eq: smallness.two`) for a small enough `ε_1`; here the separation itself is
the hypothesis, since it is what the claim uses.  Carriers are explicit sets
of full measure rather than topological supports.
-/

import Transformer.Basic
import Transformer.Interpolation.BallTransport

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- **Claim (cl: balls).**

Let `ν^1, …, ν^N ∈ 𝒫(𝕊^{d-1})` be carried by sets `S^i` that are pairwise
`2κ`-separated.  Then there are centres `x_m^i` and radii `r^i > 0` with

  1. `ν^i(B(x_0^i, r^i)) = 1/M` and
     `ν^i(B(x_m^i, r^i) \ B(x_{m-1}^i, r^i)) = 1/M` for `1 ≤ m < M`,
  2. for every `m < M - 1` some `z ∈ B(x_m^i, r^i)` lies outside every later
     ball `B(x_{m'}^i, r^i)`, `m' > m`,
  3. `ν^i(B(x_m^j, r^j)) = 0` whenever `j ≠ i`.

Indices run from `0`, so the survey's `m ∈ [1, M]` is `m < M` here.

Not proved here.

Source: arXiv:2411.04551v3, §5, `cl: balls`. -/
theorem claim_balls
    (N M : ℕ) (ν : Idx N → Perspective.ProbSphere d)
    (S : Idx N → Set (SSphere d)) (κ : ℝ) (hκ : 0 < κ)
    (hcarrier : ∀ i : Idx N, (ν i : Measure (SSphere d)) (S i)ᶜ = 0)
    (hsep : ∀ i j : Idx N, i ≠ j → ∀ x ∈ S i, ∀ y ∈ S j, 2 * κ ≤ dist x y) :
    ∃ (x : Idx N → ℕ → SSphere d) (r : Idx N → ℝ),
      (∀ i : Idx N, 0 < r i) ∧
      (∀ i : Idx N,
        (ν i : Measure (SSphere d)) (Metric.ball (x i 0) (r i))
          = ENNReal.ofReal ((M : ℝ)⁻¹)) ∧
      (∀ i : Idx N, ∀ m : ℕ, 1 ≤ m → m < M →
        (ν i : Measure (SSphere d))
            (Metric.ball (x i m) (r i) \ Metric.ball (x i (m - 1)) (r i))
          = ENNReal.ofReal ((M : ℝ)⁻¹)) ∧
      (∀ i : Idx N, ∀ m : ℕ, m + 1 < M →
        ∃ z ∈ Metric.ball (x i m) (r i),
          ∀ m' : ℕ, m < m' → m' < M → z ∉ Metric.ball (x i m') (r i)) ∧
      ∀ i j : Idx N, i ≠ j → ∀ m : ℕ, m < M →
        (ν i : Measure (SSphere d)) (Metric.ball (x j m) (r j)) = 0 := by
  sorry

/-- The hypotheses of `claim_balls` are satisfiable: a single measure carried
by the whole sphere, where the separation requirement is empty. -/
example :
    (0 : ℝ) < 1 ∧
      (((⟨Measure.dirac basePoint, inferInstance⟩ : Perspective.ProbSphere 1)
          : Measure (SSphere 1))) (Set.univ : Set (SSphere 1))ᶜ = 0 ∧
      ∀ i j : Idx 1, i ≠ j → ∀ x ∈ (Set.univ : Set (SSphere 1)),
        ∀ y ∈ (Set.univ : Set (SSphere 1)), 2 * (1 : ℝ) ≤ dist x y := by
  refine ⟨one_pos, by simp, fun i j hij => ?_⟩
  exact absurd (Subsingleton.elim i j) hij

end Interpolation
end Transformer
