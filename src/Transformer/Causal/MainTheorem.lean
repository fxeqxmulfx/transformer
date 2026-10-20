module

public import Transformer.Basic
public import Transformer.Causal.Section4_TwoClusterRefutation
public import Transformer.Causal.Section4_ConsensusInduction
public import Transformer.Causal.Section4_SubspaceLeader
public import Transformer.Causal.Section4_SubspaceTwoTokens
public import Transformer.Causal.SectionA_Scalar
public import Transformer.Perspective.Section2_FlowMap
public import Transformer.Perspective.Section3_SmallBeta
public import Mathlib.Analysis.InnerProductSpace.Projection.Basic

/-!
# Causal attention — Main theorem for `V = I_d` (§4 of 2411.04990v2)

* `Theorem thm1`        — almost-everywhere convergence to a single cluster
                          for `V = I_d` and *arbitrary* `Q, K`,
* `Conjecture thm1.5`   — analogue for `V` diagonalizable with `λ_max > 0`
                          of multiplicity 1,
* `Conjecture thm2`     — analogue for `V` with `λ_max > 0` of multiplicity
                          `≥ 2`,
* `Lemma lemma:scalar`  — the scalar inequality the proof of `thm1` opens
                          with, and its equality cases.

All three statements quantify over "almost every initial configuration" with
respect to the volume measure on `(𝕊^{d-1})^n`. Its null sets are those of
the uniform law, which is read as every `σ` with `Perspective.UniformTuple d n σ`
— it holds for exactly one `σ`. Conjecture `thm1.5` is refuted by a
positive-measure counterexample in `Section4_TwoClusterRefutation`.
The single-cluster theorem is proved by the original prefix induction,
with its conditional antipodal exception null by the actual follower
expansion and Fubini. Subspace convergence is proved for one and two
tokens; the assertion for three or more tokens remains unproved.
The geometric `lemma:scalar` and its equality cases are in `SectionA_Scalar`.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Causal

open Causal Perspective

variable (d n : ℕ)

/-- **Theorem (thm1).** *Single-cluster convergence with `V = I_d`.*

For arbitrary `Q, K`, `β ≥ 0` and `V = I_d`, for almost any
`(x_1(0),…,x_n(0)) ∈ (𝕊^{d-1})^n` the CSA dynamics satisfy

  `∀ k ∈ [n],  lim_{t → ∞} x_k(t) = x_1(0)`.

The leader `x_1` is a fixed point of `eq: csa`: its own row of the causal mask
sees only itself, so the whole cloud is dragged onto where it started.

*`d ≥ 2`.*  The source puts no bound on `d`, and at `d = 1` the statement is
false: `𝕊^0 = {±1}` has no tangent directions, so no token moves, and the
configuration `(1, -1)` has mass `1/4` under the uniform law and never reaches
`x_1(0)`. `2 ≤ d` is added.

The actual global causal flow and its prefix consistency are proved first.
For a fixed converging prefix, the original last token converges to the
leader or its antipode. The actual field expands near the antipode, so
at most one original last initial token has that limit. The original
invariant marginal has zero singleton masses, and Fubini and induction
give the original almost-everywhere conclusion for every actual solution.

Source: arXiv:2411.04990v2, §4, `thm1`. -/
theorem single_cluster (hd : 2 ≤ d) (β : ℝ) (Q K : ParamMatrix d) (hn : 1 ≤ n)
    (hβ : 0 ≤ β) :
    ∀ σ : Measure (SphereTuple d n), Perspective.UniformTuple d n σ →
    ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ →
      Causal.CSA d n β Q K (ContinuousLinearMap.id ℝ (EucSpace d)) X →
        ∀ k : Idx n,
          Filter.Tendsto (fun t : ℝ => X t k) Filter.atTop (nhds (X₀ ⟨0, hn⟩)) := by
  intro σ hσ
  simpa only [abs_of_nonneg hβ] using single_cluster_all_real hd |β| Q K hn σ hσ

/-- The hypotheses of `single_cluster` are satisfiable: the circle, one token
and `β = 0`. -/
example : 2 ≤ 2 ∧ (1 : ℕ) ≤ 1 ∧ (0 : ℝ) ≤ 0 := ⟨le_rfl, le_rfl, le_rfl⟩

/-- **Conjecture (thm1.5) is false.** The source predicts almost-everywhere
convergence to the leading unit eigenvectors for every query and key pair
when the value matrix has distinct positive eigenvalues.
At dimension and token count two, `V=diag(2,1)`, `Q=I`, `K=diag(0,4)`
and temperature one give a nonempty open region of positive uniform
product measure whose follower stays off the leading axis forever.
All original spectral hypotheses and the original causal equation hold,
by `two_cluster_counterexample`. This is a refutation of the source's
universal assertion, with its exact hypotheses and conclusion retained
inside the negation. Source: arXiv:2411.04990v2, §4, `thm1.5`. -/
theorem not_two_cluster :
    ¬ ∀ (d n : ℕ) (β : ℝ) (Q K V : ParamMatrix d) (lam : ℝ) (ξ : SSphere d), 0 ≤ β →
      (∃ (e : Idx d → EucSpace d) (μ : Idx d → ℝ), LinearIndependent ℝ e ∧
        (∀ i, V (e i) = μ i • e i) ∧ Function.Injective μ ∧ (∀ i, 0 < μ i) ∧
        ∀ i, μ i ≤ lam) →
      V (ξ : EucSpace d) = lam • (ξ : EucSpace d) →
      ∀ σ : Measure (SphereTuple d n), Perspective.UniformTuple d n σ →
        ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → CSA d n β Q K V X →
          ∀ k : Idx n,
            Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
              (nhds (ξ : EucSpace d)) ∨
            Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
              (nhds (-(ξ : EucSpace d))) := by
  intro h
  exact not_forall_two_cluster (h 2 2)

/-- **Conjecture (thm2).**  *Single-cluster convergence with `λ_max > 0`,
`dim L ≥ 2`.*

Let `λ > 0`, let `L` be an eigenspace of `V` for `λ` with `dim L ≥ 2`, let `V`
preserve `L^⊥`, and let `⟨V z, z⟩ < λ ‖z‖²` for every `z ∈ L^⊥ \ {0}` — so `λ`
really is the top eigenvalue and everything transverse to `L` decays relative
to it. Then almost every initialization yields convergence of every token to
the normalized `L`-component of `x_1(0)`, which is defined as soon as that
component is nonzero.

The source defines `ξ` in words as "the normalized `L`-component of `x_1(0)`"
and then in symbols as `y_1 := P_{L^⊥}(x_1(0))`, `ξ := y_1/|y_1|`. The
symbols contradict the words and the surrounding discussion (tokens are
attracted *into* `L`); the words are taken, `ξ` is the projection on `L`.
The source's strict transverse inequality is restricted to nonzero `z`;
at `z = 0` both sides are zero, so its printed unrestricted version is impossible.

The first-token limit and the full singleton case are proved from the
original projected equation in `Section4_SubspaceLeader`. The complete
two-token case, including its exceptional-set argument, is proved in
`Section4_SubspaceTwoTokens`. The assertion at `n ≥ 3` remains unproved.

Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem subspace_cluster (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (L : Submodule ℝ (EucSpace d)) (hn : 1 ≤ n)
    (hβ : 0 ≤ β) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hdim : 2 ≤ Module.finrank ℝ L)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner (𝕜 := ℝ) (V z) z < lam * ‖z‖ ^ 2) :
    ∀ σ : Measure (SphereTuple d n), Perspective.UniformTuple d n σ →
    ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Causal.CSA d n β Q K V X →
      L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d) ≠ 0 →
        ∀ k : Idx n,
          Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
            (nhds (‖L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d)‖⁻¹ •
              L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d))) := by
  by_cases hn1 : n = 1
  · subst n
    intro σ
    refine fun _ => Filter.Eventually.of_forall ?_
    intro X₀ X h0 hX hp i
    have hi : (⟨0, hn⟩ : Idx 1) = 0 := Subsingleton.elim _ _
    rw [hi] at hp ⊢
    have hp' : L.starProjection (X 0 0 : EucSpace d) ≠ 0 := by rwa [h0]
    have h := subspace_cluster_singleton β Q K V L lam heig hinv hdecay X hX hp' i
    simpa only [h0] using h
  · by_cases hn2 : n = 2
    · subst n
      have hi : (⟨0, hn⟩ : Idx 2) = 0 := rfl
      simpa only [hi] using
        subspace_cluster_twoTokens β Q K V L lam hlam heig hdim hinv hdecay
    · sorry

/-- The hypotheses of `subspace_cluster` are satisfiable: in the plane, with
`L = ⊤`, `V = I_2` and `λ = 1`. The whole plane has rank `2`, is invariant, and
`⊤ᗮ = ⊥`, so the decay condition holds on an empty set of `z`. -/
example :
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
        (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
      2 ≤ Module.finrank ℝ (⊤ : Submodule ℝ (EucSpace 2)) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
        (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
        inner (𝕜 := ℝ) ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < 1 * ‖z‖ ^ 2) := by
  refine ⟨le_rfl, one_pos, fun v _ => by simp, ?_, fun z hz => hz, ?_⟩
  · rw [finrank_top ℝ (EucSpace 2), finrank_euclideanSpace_fin]
  · intro z hz hz0
    rw [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] at hz
    exact absurd hz hz0


end Causal
end Transformer
