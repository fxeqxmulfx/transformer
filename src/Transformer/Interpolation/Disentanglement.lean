module

public import Transformer.Basic
public import Transformer.Perspective.Section2_FlowMap
public import Transformer.Interpolation.Basic
public import Transformer.Interpolation.Clustering
public import Transformer.Interpolation.Section3_TwoContractions
public import Transformer.Interpolation.Section3_QuadrantControlPhases
public import Transformer.Interpolation.Section3_SeparationControls

/-!
# Measure-to-measure interpolation — Disentangling supports

Formalization of §3 of arXiv:2411.04551v3:

* `Proposition prop: separation`  — disentanglement via attention with `𝐁 ≡ 0`,
* `Lemma lem: first.quadrant`     — transportation to `ℚ_1^{d-1}`.

`lem: induction.barycenter` and `lem: perturbation` are in
`Transformer.Interpolation.Perturbation`; `eq: identity.flow` is in
`Transformer.Interpolation.IdentityFlow`. That `prop: separation` is false for
two equal initial measures is `Interpolation.not_separation`.

"The solution `μ^i` of `eq: cauchy.pb` with data `μ_0^i`" is read as: a
solution exists, and every solution satisfies the conclusion. `O(d · N)` is a
constant `C` chosen before `d`, `N` and the data. `𝐁 ≡ 0` is imposed on the
parameter curve rather than `eq: average.vf` being used in place of `eq: vf`;
the two agree, since `exp ⟨0, x'⟩ = 1` turns the attention term of `fullVF`
into the barycenter.
-/

@[expose] public section

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

Two hypotheses are added to the source's.

* The `μ_0^i` are pairwise distinct. As printed the claim is false:
  `μ_0^1 = μ_0^2` have the same solutions, and `conv_g supp μ(T)` is disjoint
  from itself only if it is empty, which a probability measure's support is not
  (`Interpolation.not_separation`). The proof in §3 feeds its measures to
  `lem: perturbation`, which requires "two different measures", so distinctness
  is what the source uses.
* `N ≥ 1`. For `N = 0` the bound `O(d · N)` allows no switch, i.e. no piece of
  positive length, which `PiecewiseConstant` on `[0, T]` with `T > 0` rules
  out; the source's induction starts at `N = 1`.

The proof gives the absolute constant five. Finite small ridge phases
first make the original means non-collinear. Selective contraction into
disjoint quadrant caps then restores all other laws exactly. Realizing
this finite phase list gives the conclusion for every original weak
solution, without the source's false fixed-outside-hulls clause.

Source: arXiv:2411.04551v3, §3, `prop: separation`. -/
theorem separation :
    ∃ C : ℕ, ∀ (d N : ℕ) (T : ℝ) (μ₀ : Idx N → ProbSphere d), 1 ≤ N → 0 < T →
      (∀ i : Idx N, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d) →
      Function.Injective μ₀ →
      ∃ (θ : TimeParams d) (K : ℕ),
        K ≤ C * (d * N) ∧ (∀ s : ℝ, (θ s).B = 0) ∧ PiecewiseConstant d θ T K ∧
        (∀ i : Idx N, ∃ μ : ℝ → ProbSphere d, μ 0 = μ₀ i ∧ cauchyPB d θ μ) ∧
        ∀ i j : Idx N, i ≠ j → ∀ μ ν : ℝ → ProbSphere d,
          μ 0 = μ₀ i → cauchyPB d θ μ → ν 0 = μ₀ j → cauchyPB d θ ν →
          Disjoint (convG d (μ T : Measure (SSphere d)).support)
            (convG d (ν T : Measure (SSphere d)).support) := by
  exact ⟨5, exists_separation_controls⟩

/-- The hypotheses of `separation` are satisfiable: `N = 1`, `T = 1`, `d = 1`,
and the Dirac mass at `basePoint 0`, whose support `{+1}` lies in `ℚ_1^0`. -/
example :
    1 ≤ 1 ∧ (0 : ℝ) < 1 ∧
      (∀ i : Idx 1,
        (((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i : Measure (SSphere 1))).support
          ⊆ positiveQuadrant 1) ∧
      Function.Injective (fun _ : Idx 1 => diracProb 1 (basePoint 0)) := by
  refine ⟨le_rfl, one_pos, fun _ x hx => ?_, fun i i' _ => Subsingleton.elim i i'⟩
  have hx' : x ∈ (Measure.dirac (basePoint 0)).support := hx
  rw [eq_of_mem_support_dirac hx']
  exact basePoint_mem_positiveQuadrant

/-- **Lemma (lem: first.quadrant).** *Transportation to `ℚ_1^{d-1}`.*

If `⋃_i supp μ_0^i ⊊ 𝕊^{d-1}`, there is a piecewise-constant
`𝐖 : [0, T] → M_{d×d}(ℝ)` with at most one switch and `‖𝐖‖_∞ ≤ C/T` (for some
`C = C(N) > 0`) such that, taking `𝐕 ≡ 𝐁 ≡ 𝐔 ≡ 0` and `b ≡ 1`, the solution
`μ^i` satisfies `supp μ^i(T) ⊂ ℚ_1^{d-1}`.

Two deviations from the source, both recorded here.

* `C` depends on the data `μ_0`, not on `N` alone; it is chosen before `T`,
  which is what "by time-rescaling" in the proof gives. With `C` independent
  of the data the claim fails: every admissible flow is a composition of two
  flows of `ẋ = Proj_x w`, `|w| ≤ √d ‖𝐖‖`, so under `‖𝐖‖ ≤ C/T` it is
  bi-Lipschitz with a constant depending on `C` and `d` only; a support that
  misses only a cap of radius `δ` would have its complement, of diameter `δ`,
  mapped onto `𝕊^{d-1} ∖ ℚ_1^{d-1}`, of diameter `2`. This is not formalized.
* `d ≥ 2`. On `𝕊^0 = {±1}` every tangent field vanishes, so
  `μ_0 = δ_{-1}` never reaches `ℚ_1^0 = {+1}`.

The proof uses two explicit rank-one controls, with intrinsic phase
lengths chosen from the compact union of original supports. Each phase
occupies half of the prescribed horizon, giving `C = 2(τ₁+τ₂)`.
Constructed weak solutions establish existence; the original integrated
equation forces every weak solution to follow the same support maps.

Source: arXiv:2411.04551v3, §3, `lem: first.quadrant` and its proof. -/
theorem first_quadrant (hd : 2 ≤ d) (μ₀ : Idx N → ProbSphere d)
    (hne : (⋃ i : Idx N, (μ₀ i : Measure (SSphere d)).support) ≠ Set.univ) :
    ∃ C : ℝ, 0 < C ∧ ∀ T : ℝ, 0 < T →
      ∃ θ : TimeParams d,
        PiecewiseConstant d θ T 2 ∧
        (∀ s : ℝ, (θ s).V = 0 ∧ (θ s).B = 0 ∧ (θ s).U = 0 ∧
          (θ s).b = (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1)) ∧
        (∀ s ∈ Set.Icc (0 : ℝ) T, ‖(θ s).W‖ ≤ C / T) ∧
        (∀ i : Idx N, ∃ μ : ℝ → ProbSphere d, μ 0 = μ₀ i ∧ cauchyPB d θ μ) ∧
        ∀ i : Idx N, ∀ μ : ℝ → ProbSphere d, μ 0 = μ₀ i → cauchyPB d θ μ →
          (μ T : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
  obtain ⟨w₁, w₂, τ₁, τ₂, hτ₁, hτ₂, hflow⟩ :=
    exists_two_projection_quadrant_supports hd μ₀ hne
  let j : Fin d := ⟨0, by omega⟩
  refine ⟨2 * (τ₁ + τ₂), by positivity, ?_⟩
  intro T hT
  refine ⟨twoProjectionControls j w₁ w₂ τ₁ τ₂ T,
    twoProjectionControls_piecewise j w₁ w₂ τ₁ τ₂ T hT, ?_, ?_, ?_, ?_⟩
  · exact fun s => twoProjectionControls_components j w₁ w₂ τ₁ τ₂ T s
  · exact fun s _ => twoProjectionControls_norm j w₁ w₂ τ₁ τ₂ T hτ₁.le hτ₂.le hT s
  · exact fun i => exists_cauchyPB_twoProjectionControls j w₁ w₂ τ₁ τ₂ T hT (μ₀ i)
  · intro i μ hinit hμ x hx
    obtain ⟨y, hy, rfl⟩ := twoProjectionControls_support j w₁ w₂ τ₁ τ₂ T hT μ hμ hx
    obtain ⟨z, hz, rfl⟩ := hy
    apply hflow i z
    rwa [hinit] at hz

/-- The hypotheses of `first_quadrant` are satisfiable: on `𝕊^1`, the single
Dirac mass at `basePoint 1`, whose support misses the antipode and is therefore
not the whole sphere. -/
example :
    2 ≤ 2 ∧
      (⋃ i : Idx 1,
          (((fun _ : Idx 1 => diracProb 2 (basePoint 1)) i :
            Measure (SSphere 2))).support) ≠ Set.univ := by
  refine ⟨le_rfl, fun huniv => ?_⟩
  have hmem : antipode 2 (basePoint 1) ∈ ⋃ i : Idx 1,
      (((fun _ : Idx 1 => diracProb 2 (basePoint 1)) i : Measure (SSphere 2))).support := by
    rw [huniv]
    trivial
  obtain ⟨_, hi⟩ := Set.mem_iUnion.mp hmem
  have hi' : antipode 2 (basePoint 1) ∈ (Measure.dirac (basePoint 1)).support := hi
  exact antipode_ne 2 (basePoint 1) (eq_of_mem_support_dirac hi')

end Interpolation
end Transformer
