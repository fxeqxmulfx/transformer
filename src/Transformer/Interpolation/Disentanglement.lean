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
`eq: cauchy.pb` to a prescribed configuration; none is proved here, so each is
a theorem closed by `sorry`.  All four carry three weakenings, the same ones as
in `Transformer.Interpolation.Clustering`:

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

Not proved here.

Source: arXiv:2411.04551v3, §3. -/
theorem separation (C : ℕ) (T : ℝ) (μ₀ : Idx N → ProbSphere d) (hT : 0 < T)
    (hsupp : ∀ i : Idx N, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → ProbSphere d),
      K ≤ C * (d * N) ∧ (∀ s : ℝ, (θ s).B = 0) ∧ PiecewiseConstant d θ T K ∧
      (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
      ∀ i j : Idx N, i ≠ j →
        Disjoint ((μ i T : Measure (SSphere d)).support)
          ((μ j T : Measure (SSphere d)).support) := by
  sorry

/-- The hypotheses of `separation` are satisfiable: `T = 1`, `d = 1`, and the
one-element family of Dirac masses at `basePoint 0`, whose support `{+1}` lies
in `ℚ_1^0`. -/
example :
    (0 : ℝ) < 1 ∧
      ∀ i : Idx 1,
        (((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i : Measure (SSphere 1))).support
          ⊆ positiveQuadrant 1 := by
  refine ⟨one_pos, fun _ x hx => ?_⟩
  have hx' : x ∈ (Measure.dirac (basePoint 0)).support := hx
  rw [eq_of_mem_support_dirac hx']
  exact basePoint_mem_positiveQuadrant

/-- **Lemma (lem: first.quadrant).** *Transportation to `ℚ_1^{d-1}`.*

If `⋃_i supp μ_0^i ⊊ 𝕊^{d-1}`, there is a piecewise-constant
`𝐖 : [0, T] → M_{d×d}(ℝ)` with at most one switch and `‖𝐖‖_∞ ≤ C/T` (for some
`C = C(N) > 0`) such that, taking `𝐕 ≡ 𝐁 ≡ 𝐔 ≡ 0` and `b ≡ 1`, the solution
`μ^i` satisfies `supp μ^i(T) ⊂ ℚ_1^{d-1}`.

Not proved here.

Source: arXiv:2411.04551v3, §3. -/
theorem first_quadrant (T : ℝ) (μ₀ : Idx N → ProbSphere d) (hT : 0 < T)
    (hne : (⋃ i : Idx N, (μ₀ i : Measure (SSphere d)).support) ≠ Set.univ) :
    ∃ C : ℝ, 0 < C ∧
      ∃ (θ : TimeParams d) (μ : Idx N → ℝ → ProbSphere d),
        PiecewiseConstant d θ T 2 ∧
        (∀ s : ℝ, (θ s).V = 0 ∧ (θ s).B = 0 ∧ (θ s).U = 0 ∧
          (θ s).b = (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1)) ∧
        (∀ s ∈ Set.Icc (0 : ℝ) T, ‖(θ s).W‖ ≤ C / T) ∧
        (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
        ∀ i : Idx N, (μ i T : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
  sorry

/-- The hypotheses of `first_quadrant` are satisfiable: `T = 1` and, on `𝕊^0`,
the single Dirac mass at `basePoint 0`, whose support misses the antipode and
is therefore not the whole sphere. -/
example :
    (0 : ℝ) < 1 ∧
      (⋃ i : Idx 1,
          (((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i :
            Measure (SSphere 1))).support) ≠ Set.univ := by
  refine ⟨one_pos, fun huniv => ?_⟩
  have hmem : antipode 1 (basePoint 0) ∈ ⋃ i : Idx 1,
      (((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i : Measure (SSphere 1))).support := by
    rw [huniv]
    trivial
  obtain ⟨_, hi⟩ := Set.mem_iUnion.mp hmem
  have hi' : antipode 1 (basePoint 0) ∈ (Measure.dirac (basePoint 0)).support := hi
  exact antipode_ne 1 (basePoint 0) (eq_of_mem_support_dirac hi')

/-- **Lemma (lem: induction.barycenter).** *Inductive barycenter shrinking.*

If the barycenters `(𝔼_{μ_0^i}[x])_i` are pairwise non-collinear, then for any
`T, ε > 0` and any `ν_0 ∈ 𝒫(ℚ_1^{d-1})` whose barycenter is collinear with
`𝔼_{μ_0^j}[x]`, there is a piecewise-constant `θ : [0, T] → Θ` with at most
`O(d · N)` switches such that

  `conv_g supp ν(T) ∪ conv_g supp μ^j(T) ⊂ B(𝔼_{μ_0^j}[z] / ‖𝔼_{μ_0^j}[z]‖, ε)`,

while `μ^i(T) = μ_0^i` for `i ≠ j`.

Not proved here.

Source: arXiv:2411.04551v3, §3. -/
theorem induction_barycenter (C : ℕ) (T ε : ℝ) (j : Idx N)
    (μ₀ : Idx N → ProbSphere d) (ν₀ : ProbSphere d) (hT : 0 < T) (hε : 0 < ε)
    (hgen : ∀ i i' : Idx N, i ≠ i' →
      ∀ γ : ℝ, barycenter d (μ₀ i) ≠ γ • barycenter d (μ₀ i'))
    (hsupp : (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hcol : ∃ γ : ℝ, barycenter d ν₀ = γ • barycenter d (μ₀ j)) :
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → ProbSphere d) (ν : ℝ → ProbSphere d),
      K ≤ C * (d * N) ∧ PiecewiseConstant d θ T K ∧
      (∀ i : Idx N, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
      ν 0 = ν₀ ∧ cauchyPB d θ ν ∧
      (∀ x ∈ (ν T : Measure (SSphere d)).support ∪ (μ j T : Measure (SSphere d)).support,
        ‖(x : EucSpace d) - ‖barycenter d (μ₀ j)‖⁻¹ • barycenter d (μ₀ j)‖ < ε) ∧
      ∀ i : Idx N, i ≠ j → μ i T = μ₀ i := by
  sorry

/-- The hypotheses of `induction_barycenter` are satisfiable on a one-element
family: `N = 1` leaves the non-collinearity of *distinct* barycenters nothing to
check, `ν₀ = μ₀ 0 = δ_{basePoint 0}` is supported in `ℚ_1^0`, and the two
barycenters are collinear with `γ = 1`. -/
example :
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (∀ i i' : Idx 1, i ≠ i' →
        ∀ γ : ℝ, barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i)
          ≠ γ • barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i')) ∧
      ((diracProb 1 (basePoint 0) : ProbSphere 1) : Measure (SSphere 1)).support
        ⊆ positiveQuadrant 1 ∧
      (∃ γ : ℝ, barycenter 1 (diracProb 1 (basePoint 0))
        = γ • barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) 0)) := by
  refine ⟨one_pos, one_pos, fun i i' hne => absurd (Subsingleton.elim i i') hne,
    fun x hx => ?_, 1, (one_smul ℝ _).symm⟩
  have hx' : x ∈ (Measure.dirac (basePoint 0)).support := hx
  rw [eq_of_mem_support_dirac hx']
  exact basePoint_mem_positiveQuadrant

/-- **Lemma (lem: perturbation / lem: colinearity).**

If `𝔼_{μ_0}[x] = γ_1 𝔼_{ν_0}[x]` for some `γ_1 ∈ (0, 1]`, one can choose
`(𝐕, 𝐖, 𝐔, b)` — piecewise constant with at most two switches — so that the
barycenters of the corresponding solutions are no longer collinear.

Not proved here.

Source: arXiv:2411.04551v3, §3. -/
theorem perturbation (T γ₁ : ℝ) (μ₀ ν₀ : ProbSphere d) (hT : 0 < T) (hγ₀ : 0 < γ₁)
    (hγ₁ : γ₁ ≤ 1) (hbary : barycenter d μ₀ = γ₁ • barycenter d ν₀) :
    ∃ (θ : TimeParams d) (μ ν : ℝ → ProbSphere d),
      PiecewiseConstant d θ T 3 ∧ (∀ s : ℝ, (θ s).B = 0) ∧
      μ 0 = μ₀ ∧ ν 0 = ν₀ ∧ cauchyPB d θ μ ∧ cauchyPB d θ ν ∧
      ∀ γ : ℝ, barycenter d (μ T) ≠ γ • barycenter d (ν T) := by
  sorry

/-- The hypotheses of `perturbation` are satisfiable: `T = γ₁ = 1` and
`μ₀ = ν₀ = δ_{basePoint 0}`, whose barycenters coincide. -/
example :
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧
      barycenter 1 (diracProb 1 (basePoint 0))
        = (1 : ℝ) • barycenter 1 (diracProb 1 (basePoint 0)) :=
  ⟨one_pos, one_pos, le_rfl, (one_smul ℝ _).symm⟩

end Interpolation
end Transformer
