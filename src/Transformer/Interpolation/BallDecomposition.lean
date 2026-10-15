/-
# Measure-to-measure interpolation — Decomposing a measure into equal-mass balls

Formalization of `Claim cl: balls` of arXiv:2411.04551v3, the combinatorial
step in the proof of `thm: main.result`: once the measures `ν^i` have pairwise
separated small supports, each can be cut by a family of equal-radius balls,
whose intersections with its carrier are nested, into
`M` shells of mass `1/M`, in such a way that the family
belonging to one measure is invisible to the others.
The common radii, ordered centers, equal shell masses and parking
points are constructed explicitly from the original probabilities.

Both separation and smallness are needed in the source proof. The earlier
statement kept separation but omitted `eq: smallness.two`,
`eq: boundary.ball` and `eq: bound`. The corrected statement restores
the source's "small enough" support scale before quantifying over the
measures. Carriers are explicit full-measure sets.

The survey's measures are absolutely continuous, and that is not decoration:
`not_exists_ball_of_mass_of_dirac` shows that a Dirac mass admits no ball of
mass `1/M` at all.  The statement here had dropped it; it is restored, in the
form the survey's proof uses — every geodesic sphere is null.
-/

import Transformer.Basic
import Transformer.Interpolation.Section5_BallConstruction

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- **A measure with an atom admits no such decomposition.**

Requirement 1 of `cl: balls` asks for a ball of mass exactly `1/M`.  A Dirac
mass gives every set measure `0` or `1`, so as soon as `M ≥ 2` no ball — of
any centre, of any radius — carries mass `1/M`, and the claim fails for
`ν = δ_{x₀}`.  Absolute continuity is what rules that out: it is what the
source has, and `claim_balls` below carries it in the form its proof uses.

Source: arXiv:2411.04551v3, §5, `cl: balls` (the "Since `ν^i` is absolutely
continuous" of its proof). -/
theorem not_exists_ball_of_mass_of_dirac (M : ℕ) (hM : 2 ≤ M) (x₀ : SSphere d)
    (ν : Perspective.ProbSphere d)
    (hν : (ν : Measure (SSphere d)) = Measure.dirac x₀) :
    ¬ ∃ (y : SSphere d) (r : ℝ), 0 < r ∧
        (ν : Measure (SSphere d)) (Metric.ball y r) = ENNReal.ofReal ((M : ℝ)⁻¹) := by
  rintro ⟨y, r, -, hmass⟩
  have hM2 : (2 : ℝ) ≤ (M : ℝ) := by exact_mod_cast hM
  have hpos : (0 : ℝ) < (M : ℝ)⁻¹ := inv_pos.mpr (by linarith)
  rw [hν] at hmass
  rcases Measure.dirac_apply_eq_zero_or_one (a := x₀) (s := Metric.ball y r) with h | h
  · rw [h, eq_comm, ENNReal.ofReal_eq_zero] at hmass
    linarith
  · rw [h, eq_comm, ENNReal.ofReal_eq_one] at hmass
    have hMne : (M : ℝ) ≠ 0 := by linarith
    have hone : (M : ℝ) * (M : ℝ)⁻¹ = (M : ℝ) * 1 := by rw [hmass]
    rw [mul_inv_cancel₀ hMne, mul_one] at hone
    linarith

/-- The atom obstruction has actual hypotheses at one Dirac and `M = 2`. -/
example : (2 : ℕ) ≤ 2 ∧
    (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)) = Measure.dirac (basePoint 1) :=
  ⟨le_rfl, rfl⟩

/-- **Claim (cl: balls).**

Let `ν^1, …, ν^N ∈ 𝒫_ac(𝕊^{d-1})` be carried by sets `S^i` that are pairwise
`2κ`-separated and sufficiently small. Then there are centers `x_m^i`
and radii `r^i > 0` with

  1. `ν^i(B(x_0^i, r^i)) = 1/M` and
     `ν^i(B(x_m^i, r^i) \ B(x_{m-1}^i, r^i)) = 1/M` for `1 ≤ m < M`,
  2. for every `m < M - 1` some `z ∈ B(x_m^i, r^i)` lies outside every later
     ball `B(x_{m'}^i, r^i)`, `m' > m`,
  3. `ν^i(B(x_m^j, r^j)) = 0` whenever `j ≠ i`.

Indices run from `0`, so the survey's `m ∈ [1, M]` is `m < M` here.

**What the source says and what is changed here.**
The survey's measures are absolutely continuous — the flow maps of
`prop: targets.atoms` that produce them go from `𝒫_ac(𝕊)` to `𝒫_ac(𝕊)`, and
the proof of the claim opens with "Since `ν^i` is absolutely continuous" to
get the continuity of `(s, r) ↦ ν^i(B(γ(s), r))` that its intermediate-value
argument runs on.  That hypothesis was missing from the statement here, and
without it the claim is false, not merely unproved:
`not_exists_ball_of_mass_of_dirac` refutes requirement 1 for a Dirac mass at
every `M ≥ 2`.  It is carried below in the form the proof uses it — every
geodesic sphere is `ν^i`-null, which is what makes
`(s, r) ↦ ν^i(B(γ(s), r))` continuous, and which at `r = 0` is the absence of
atoms that the Dirac counterexample violates.  That form is implied by
absolute continuity and mentions no reference measure.

The source also assumes the support is sufficiently small relative to the
gap: `eq: boundary.ball` and `eq: bound`, following `eq: smallness.two`.
That was omitted here. We restore the claim's existential smallness scale,
choosing `δ = min(κ,1)/1000`; each carrier lies within `δ` of one of its
points. The source's implicit positive number of target atoms is `0 < M`.
Dimension at least two supplies the geodesic's nonzero unit tangent.
The source's surrounding theorem assumes `d ≥ 3`; this geometric
construction also holds at `d = 2`.
All three conclusions are proved for the original measures and actual balls.

Source: arXiv:2411.04551v3, §5, `cl: balls`. -/
theorem claim_balls
    (hd : 2 ≤ d) (N M : ℕ) (hM : 0 < M) (κ : ℝ) (hκ : 0 < κ) :
    ∃ δ : ℝ, 0 < δ ∧
    ∀ (ν : Idx N → Perspective.ProbSphere d) (S : Idx N → Set (SSphere d)),
    (∀ i : Idx N, ∀ y : SSphere d, ∀ r : ℝ,
      (ν i : Measure (SSphere d)) (Metric.sphere y r) = 0) →
    (∀ i : Idx N, (ν i : Measure (SSphere d)) (S i)ᶜ = 0) →
    (∀ i j : Idx N, i ≠ j → ∀ x ∈ S i, ∀ y ∈ S j, 2 * κ ≤ dist x y) →
    (∀ i : Idx N, ∃ p ∈ S i, ∀ x ∈ S i, dist x p ≤ δ) →
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
  exact exists_smallness_equal_mass_balls hd N M hM κ hκ

/-- Every restored input is realized by a nonempty diffuse family
on the circle for every prescribed positive smallness scale. Two
requested masses are retained. Source: arXiv:2411.04551v3, §5, `cl: balls`. -/
example (δ : ℝ) (hδ : 0 < δ) :
    ∃ (ν : Idx 1 → Perspective.ProbSphere 2) (S : Idx 1 → Set (SSphere 2)),
    (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 ∧
      (∀ i : Idx 1, ∀ y : SSphere 2, ∀ r : ℝ,
        (ν i : Measure (SSphere 2)) (Metric.sphere y r) = 0) ∧
      (∀ i : Idx 1, (ν i : Measure (SSphere 2)) (S i)ᶜ = 0) ∧
      (∀ i j : Idx 1, i ≠ j → ∀ x ∈ S i, ∀ y ∈ S j, 2 * (1 : ℝ) ≤ dist x y) ∧
      (∀ i : Idx 1, ∃ p ∈ S i, ∀ x ∈ S i, dist x p ≤ δ) := by
  obtain ⟨ν, hν, hcap⟩ := exists_small_cap_ball_diffuse le_rfl (basePoint 1) δ hδ
  refine ⟨fun _ => ν, fun _ => Metric.closedBall (basePoint 1) δ,
    le_rfl, by decide, one_pos, fun _ => hν, fun _ => hcap, ?_, ?_⟩
  · intro i j hij
    exact (hij (Subsingleton.elim i j)).elim
  · intro i
    exact ⟨basePoint 1, by simp [hδ.le], fun x hx => hx⟩

end Interpolation
end Transformer
