/-
# Measure-to-measure interpolation — Clustering of input data

Formalization of §2 of arXiv:2411.04551v3:

* `Proposition prop: compression`     — clustering to discrete measures,

`Proposition prop: targets.atoms`, clustering to a single point mass, is
`Transformer.Interpolation.AtomClustering`, together with the refutation of the
form it had here.

`prop: compression` asserts that *some* parameter curve drives `eq: cauchy.pb`
to a prescribed target within `ε` in `W_2` (`Interpolation.W2`), keeping the
geodesic convex hulls (`convG`) of the supports apart; it is not proved.

What is proved is the invariance that the proof starts from: the barycenter
of a measure supported in a cap `⟨·, w⟩ ≥ c` lies in that same cap, so a
measure supported in an open hemisphere has a nonzero barycenter — which is
what makes the drift of `eq: average.vf` point into the hemisphere.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Wasserstein
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Measure.Support

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

variable (d N M : ℕ)

/-- The barycenter `∫ x dμ(x) ∈ ℝ^d` of a measure on the sphere — the
attention output of `eq: average.vf`, whose `𝐁 ≡ 0` turns the softmax average
into the plain one. -/
noncomputable def barycenter (μ : ProbSphere d) : EucSpace d :=
  ∫ x, (x : EucSpace d) ∂(μ : Measure (SSphere d))

/-- **A cap is barycenter-invariant.**

If `μ` gives full mass to the cap `{ x : ⟨x, w⟩ ≥ c }`, then its barycenter
satisfies `⟨∫ x dμ, w⟩ ≥ c` as well: the inner product with `w` commutes with
the integral, and the integral of a function bounded below by `c` against a
probability measure is bounded below by `c`.

This is the step the proof of `prop: targets.atoms` opens with, for the cap
`c > 0` cut out by an open hemisphere.  Source: arXiv:2411.04551v3, §2. -/
theorem le_inner_barycenter (μ : ProbSphere d) (w : EucSpace d) (c : ℝ)
    (hint : Integrable (fun x : SSphere d => (x : EucSpace d))
      (μ : Measure (SSphere d)))
    (hcap : ∀ᵐ x : SSphere d ∂(μ : Measure (SSphere d)),
      c ≤ inner (𝕜 := ℝ) (x : EucSpace d) w) :
    c ≤ inner (𝕜 := ℝ) (barycenter d μ) w := by
  have hmono : ∫ _ : SSphere d, c ∂(μ : Measure (SSphere d))
      ≤ ∫ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) w
          ∂(μ : Measure (SSphere d)) :=
    integral_mono_ae (integrable_const c) (hint.inner_const w) hcap
  rw [integral_const, probReal_univ, one_smul] at hmono
  have hswap : ∫ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) w
      ∂(μ : Measure (SSphere d)) = inner (𝕜 := ℝ) (barycenter d μ) w := by
    calc ∫ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) w
          ∂(μ : Measure (SSphere d))
        = ∫ x : SSphere d, inner (𝕜 := ℝ) w (x : EucSpace d)
            ∂(μ : Measure (SSphere d)) :=
          integral_congr_ae (Filter.Eventually.of_forall fun x => real_inner_comm _ _)
      _ = inner (𝕜 := ℝ) w (∫ x : SSphere d, (x : EucSpace d)
            ∂(μ : Measure (SSphere d))) := integral_inner hint w
      _ = inner (𝕜 := ℝ) (barycenter d μ) w := real_inner_comm _ _
  rwa [hswap] at hmono

/-- **A measure supported in an open hemisphere has a nonzero barycenter.**

The hemisphere hypothesis of `prop: targets.atoms` is `⟨x, w⟩ ≥ c` for some
`c > 0`; by `le_inner_barycenter` the barycenter then has positive inner
product with `w`, so it cannot vanish, and `eq: average.vf` has a nonzero
drift.  Source: arXiv:2411.04551v3, §2. -/
theorem barycenter_ne_zero (μ : ProbSphere d) (w : EucSpace d) (c : ℝ) (hc : 0 < c)
    (hint : Integrable (fun x : SSphere d => (x : EucSpace d))
      (μ : Measure (SSphere d)))
    (hcap : ∀ᵐ x : SSphere d ∂(μ : Measure (SSphere d)),
      c ≤ inner (𝕜 := ℝ) (x : EucSpace d) w) :
    barycenter d μ ≠ 0 := by
  intro h
  have hle := le_inner_barycenter d μ w c hint hcap
  rw [h, inner_zero_left] at hle
  linarith

/-- The hypotheses of both are satisfiable, and not by an empty cap: the Dirac
mass at a point `x₀` of the sphere is a probability measure, the coordinate
map is integrable against it, and all of its mass sits in the cap
`⟨·, x₀⟩ ≥ 1`. -/
example (x₀ : SSphere 1) :
    (0 : ℝ) < 1 ∧
      Integrable (fun x : SSphere 1 => (x : EucSpace 1))
        ((⟨Measure.dirac x₀, inferInstance⟩ : ProbSphere 1) : Measure (SSphere 1)) ∧
      ∀ᵐ x : SSphere 1
        ∂((⟨Measure.dirac x₀, inferInstance⟩ : ProbSphere 1) : Measure (SSphere 1)),
          (1 : ℝ) ≤ inner (𝕜 := ℝ) (x : EucSpace 1) ((x₀ : EucSpace 1)) := by
  refine ⟨one_pos, integrable_dirac (by simp), ?_⟩
  rw [show ((⟨Measure.dirac x₀, inferInstance⟩ : ProbSphere 1) : Measure (SSphere 1))
      = Measure.dirac x₀ from rfl, MeasureTheory.ae_dirac_eq, Filter.eventually_pure]
  have hx : ‖(x₀ : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp x₀.2
  rw [real_inner_self_eq_norm_mul_norm, hx, mul_one]

/-- `θ` is piecewise constant on `[0, T]` with at most `K` pieces — hence at
most `K - 1` switches: there are times `0 = t_0 ≤ … ≤ t_K = T` such that `θ`
is constant on each `[t_k, t_{k+1})`. -/
def PiecewiseConstant (θ : TimeParams d) (T : ℝ) (K : ℕ) : Prop :=
  ∃ t : Fin (K + 1) → ℝ, Monotone t ∧ t 0 = 0 ∧ t (Fin.last K) = T ∧
    ∀ k : Fin K, ∀ s s' : ℝ,
      s ∈ Set.Ico (t k.castSucc) (t k.succ) →
        s' ∈ Set.Ico (t k.castSucc) (t k.succ) → θ s = θ s'

/-- A constant parameter curve is piecewise constant with one piece, so the
definition is not vacuous. -/
example (θ₀ : Params d) : PiecewiseConstant d (fun _ => θ₀) 1 1 := by
  refine ⟨fun i => (i.val : ℝ), fun a b hab => ?_, by simp, by simp,
    fun k s s' _ _ => rfl⟩
  exact Nat.cast_le.mpr hab

/-- **Proposition (prop: compression).** *Clustering to discrete measures.*

Let the `μ_0^i` have no atoms and pairwise disjoint geodesic convex hulls of
their supports.  Fix `M ≥ 1`, and targets `μ_1^i = Σ_k α_k^i δ_{x_k^i}` with
`α_k^i ≥ 0`, `Σ_k α_k^i = 1`, `x_k^i ∈ conv_g supp μ_0^i`, and
`x_k^i = x_{k'}^j` only if `(k, i) = (k', j)`.  Then for any `T > 0` and
`ε > 0` there are piecewise-constant `(𝐖, 𝐔, b)` on `[0, T]` such that, with
`𝐕 ≡ 0`, the solution `μ^i` of `eq: cauchy.pb`–`eq: vf` with data `μ_0^i`
satisfies `W_2(μ^i(T), μ_1^i) ≤ ε`, and
`conv_g supp μ^i(T) ∩ conv_g supp μ^j(T) = ∅` for `i ≠ j`.

"The solution" is read as: one exists, and every solution satisfies the
conclusion.  `W_2` is `Interpolation.W2`, `conv_g` is `convG`.  The count of
switches of `rem: nb.disc.clustering` is not part of the proposition and is
not stated here: its ingredients `𝖭_k^i(δ)` and `L_{k,n}^i` are not defined
in this development.

Not proved here.

Source: arXiv:2411.04551v3, §2.2, `prop: compression`. -/
theorem compression
    (μ₀ : Idx N → ProbSphere d) (x : Idx N → Idx M → SSphere d)
    (α : Idx N → Idx M → ℝ) (T ε : ℝ) (hM : 1 ≤ M) (hT : 0 < T) (hε : 0 < ε)
    (hatom : ∀ i : Idx N, ∀ y : SSphere d, (μ₀ i : Measure (SSphere d)) {y} = 0)
    (hdisj : ∀ i j : Idx N, i ≠ j →
      Disjoint (convG d (μ₀ i : Measure (SSphere d)).support)
        (convG d (μ₀ j : Measure (SSphere d)).support))
    (hαnonneg : ∀ i : Idx N, ∀ k : Idx M, 0 ≤ α i k)
    (hαsum : ∀ i : Idx N, ∑ k : Idx M, α i k = 1)
    (hx : ∀ i : Idx N, ∀ k : Idx M, x i k ∈ convG d (μ₀ i : Measure (SSphere d)).support)
    (hxinj : Function.Injective fun p : Idx N × Idx M => x p.1 p.2) :
    ∃ (θ : TimeParams d) (K : ℕ),
      (∀ s : ℝ, (θ s).V = 0) ∧ PiecewiseConstant d θ T K ∧
      (∀ i : Idx N, ∃ μ : ℝ → ProbSphere d, μ 0 = μ₀ i ∧ cauchyPB d θ μ) ∧
      (∀ i : Idx N, ∀ μ : ℝ → ProbSphere d, μ 0 = μ₀ i → cauchyPB d θ μ →
        W2 d (μ T : Measure (SSphere d))
          (∑ k : Idx M, (α i k).toNNReal • Measure.dirac (x i k)) ≤ ε) ∧
      (∀ i j : Idx N, i ≠ j → ∀ μ ν : ℝ → ProbSphere d,
        μ 0 = μ₀ i → cauchyPB d θ μ → ν 0 = μ₀ j → cauchyPB d θ ν →
        Disjoint (convG d (μ T : Measure (SSphere d)).support)
          (convG d (ν T : Measure (SSphere d)).support)) := by
  sorry

/-- The hypotheses of `compression` are satisfiable — but only on an empty
family of initial measures, `N = 0`, where the atomlessness, disjointness and
target conditions have nothing to check.  A witness with `N ≥ 1` would need
an atomless probability measure on the sphere, and none is shown atomless in
this development. -/
example (μ₀ : Idx 0 → ProbSphere 1) (x : Idx 0 → Idx 1 → SSphere 1)
    (α : Idx 0 → Idx 1 → ℝ) :
    1 ≤ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (∀ i : Idx 0, ∀ y : SSphere 1, (μ₀ i : Measure (SSphere 1)) {y} = 0) ∧
      (∀ i j : Idx 0, i ≠ j →
        Disjoint (convG 1 (μ₀ i : Measure (SSphere 1)).support)
          (convG 1 (μ₀ j : Measure (SSphere 1)).support)) ∧
      (∀ i : Idx 0, ∀ k : Idx 1, 0 ≤ α i k) ∧
      (∀ i : Idx 0, ∑ k : Idx 1, α i k = 1) ∧
      (∀ i : Idx 0, ∀ k : Idx 1, x i k ∈ convG 1 (μ₀ i : Measure (SSphere 1)).support) ∧
      Function.Injective fun p : Idx 0 × Idx 1 => x p.1 p.2 :=
  ⟨le_rfl, one_pos, one_pos, fun i => i.elim0, fun i => i.elim0, fun i => i.elim0,
    fun i => i.elim0, fun i => i.elim0, fun p => p.1.elim0⟩

end Interpolation
end Transformer
