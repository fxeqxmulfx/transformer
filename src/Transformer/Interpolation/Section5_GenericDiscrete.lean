module

public import Transformer.Interpolation.Section5_IndependentClusters
public import Transformer.Interpolation.Section5_DiracPersistence
public import Mathlib.MeasureTheory.Constructions.Pi
public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Measure-to-measure interpolation — Generic discrete inputs

Formalization of `Proposition prop:generic-discrete` of arXiv:2411.04551v3:
for `d ≥ 3` and uniformly sampled discrete inputs, the pure-attention flow
`𝐁 ≡ β I_d`, `𝐕 ≡ I_d`, `𝐖 ≡ 0` sends different inputs to different cluster
points almost surely — the `O(1)`-switch regime the survey contrasts with the
`O(d · N)` bound of `prop: compression`.
-/

@[expose] public section

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

/-- The parameters of `prop:generic-discrete`: `𝐕 ≡ I_d`, `𝐁 ≡ β I_d`,
`𝐖 ≡ 0`, so the vector field is pure self-attention. -/
noncomputable def genericParams (d : ℕ) (β : ℝ) : TimeParams d :=
  fun _ =>
    { V := ContinuousLinearMap.id ℝ (EucSpace d)
      B := β • ContinuousLinearMap.id ℝ (EucSpace d)
      W := 0
      U := 0
      b := 0 }

/-- The original generic parameters coincide with the actual attention
parameters used in the particle and weak-solution constructions.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`.
This identifies the existing field without changing its definition. -/
theorem genericParams_eq_pureAttentionParams (d : ℕ) (β : ℝ) :
    genericParams d β = pureAttentionParams d β := rfl

/-- **Proposition (prop:generic-discrete).**

Let `d ≥ 3`, let `ref` be the uniform measure on `(𝕊^{d-1})^n` — identified with
the empirical measures `𝒮_n` — and sample `μ_0^1, …, μ_0^N` i.i.d. from `π`.
For `eq: cauchy.pb` with `𝐁 ≡ β I_d`, `𝐕 ≡ I_d`, `𝐖 ≡ 0`, let `x_*^i` be the
almost surely existing cluster point of `μ^i(t)`. Then

  `ℙ[x_*^i ≠ x_*^j] = 1`   for `i ≠ j`.

The cluster point is carried by a function `f` on configurations, whose
defining property — almost sure weak convergence of the flow to `δ_{f(X)}` —
is an explicit hypothesis, since the source quotes the clustering theorem
from the survey. The distinct-cluster conclusion is proved from this input;
no unproved clustering theorem is invoked.

**What the source says and what is changed here.**  The reference measure is
`Perspective.UniformTuple d n`, the `n`-fold product of a rotation-invariant
Borel probability measure on the sphere: that is the survey's "uniform measure
on `(𝕊^{d-1})^n`", it is unique, so quantifying over every such `ref` names
it, and it is what the survey's proof actually uses — the rotation invariance
is what rules out atoms in the law of the actual limit clusters.
Carried as a free probability measure, as it was, nothing rules them out: a
Dirac `ref` charges one configuration, and for `N ≥ 2` a constant cluster map
then makes `{f(X^i) ≠ f(X^j)}` null instead of full.

One further condition the survey leaves implicit is written out, and it is not
a weakening but a repair: `n ≥ 1`. For `n = 0` the empirical measure of the
empty tuple is the zero measure, which no probability measure equals, so the
cluster hypothesis would hold vacuously.

The proof constructs an actual global particle solution and its empirical
weak solution, to which the specified cluster hypothesis applies. Uniqueness
of the particle equation gives rotation equivariance. The first-token limit
gives almost-everywhere measurability of `f`, and rotation invariance makes
the cluster law atomless. The product law then gives the conclusion.

Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem generic_discrete_distinct
    (d n N : ℕ) (β : ℝ)
    (hd : 3 ≤ d) (hβ : 0 ≤ β) (hn : 1 ≤ n) :
    ∀ ref : Measure (SphereTuple d n), Perspective.UniformTuple d n ref →
    ∀ f : SphereTuple d n → SSphere d,
      (∀ᵐ X : SphereTuple d n ∂ref,
        ∀ μ : ℝ → Perspective.ProbSphere d,
          (μ 0 : Measure (SSphere d)) = Perspective.empiricalMeasure d n X →
          cauchyPB d (genericParams d β) μ →
          Filter.Tendsto μ Filter.atTop
            (nhds (⟨Measure.dirac (f X), inferInstance⟩ :
              Perspective.ProbSphere d))) →
      ∀ i j : Idx N, i ≠ j →
        (Measure.pi fun _ : Idx N => ref)
            { X : Idx N → SphereTuple d n | f (X i) ≠ f (X j) } = 1 := by
  intro ref href f hf i j hij
  have hn' : 0 < n := by omega
  have hactual : ∀ᵐ X ∂ref, Filter.Tendsto
      (fun t => Perspective.empiricalProb hn' (Perspective.saFlow β X t)) Filter.atTop
      (nhds (Perspective.diracProb d (f X))) := by
    filter_upwards [hf] with X hcluster
    let μ : ℝ → Perspective.ProbSphere d :=
      fun t => Perspective.empiricalProb hn' (Perspective.saFlow β X t)
    have h0 : (μ 0 : Measure (SSphere d)) = Perspective.empiricalMeasure d n X := by
      simp only [μ, Perspective.saFlow_zero, Perspective.coe_empiricalProb]
    have hμ : cauchyPB d (genericParams d β) μ := by
      rw [genericParams_eq_pureAttentionParams]
      exact cauchyPB_empirical_of_SA hn' β _ (Perspective.saFlow_spec β X).2
    exact hcluster μ h0 hμ
  exact measure_independent_clusters_ne_one N (by omega) β hβ hn' ref href f hactual i j hij

/-- The original, universally quantified weak-solution clustering condition
holds for every one-token reference law at zero temperature.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`, at `n = 1`, `β = 0`.
This uses the proved persistence of all weak Dirac solutions, rather than
assuming that the particle construction exhausts the weak solutions. -/
theorem generic_zero_one_token_clusters (d : ℕ) (ref : Measure (SphereTuple d 1)) :
    ∀ᵐ X : SphereTuple d 1 ∂ref, ∀ μ : ℝ → Perspective.ProbSphere d,
      (μ 0 : Measure (SSphere d)) = Perspective.empiricalMeasure d 1 X →
      cauchyPB d (genericParams d 0) μ →
      Filter.Tendsto μ Filter.atTop
        (nhds (⟨Measure.dirac (X 0), inferInstance⟩ : Perspective.ProbSphere d)) := by
  rw [genericParams_eq_pureAttentionParams]
  exact zero_one_token_clusters ref

/-- **Every hypothesis of the original proposition has a complete model.**
The reference is the constructed spherical uniform law in dimension three.
The input has one token, the temperature is zero, and there are two distinct
sample indices. The cluster condition concerns every weak solution with
the specified initial law and has been proved without an unproved input. -/
example : 3 ≤ 3 ∧ (0 : ℝ) ≤ 0 ∧ 1 ≤ 1 ∧
    Perspective.UniformTuple 3 1 (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) ∧
    (∀ᵐ X : SphereTuple 3 1 ∂(Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3),
      ∀ μ : ℝ → Perspective.ProbSphere 3,
        (μ 0 : Measure (SSphere 3)) = Perspective.empiricalMeasure 3 1 X →
        cauchyPB 3 (genericParams 3 0) μ →
        Filter.Tendsto μ Filter.atTop
          (nhds (⟨Measure.dirac (X 0), inferInstance⟩ : Perspective.ProbSphere 3))) ∧
    (0 : Idx 2) ≠ 1 := by
  refine ⟨le_rfl, le_rfl, le_rfl, uniformTuple_uniformLaw (by norm_num) 1, ?_, by decide⟩
  exact generic_zero_one_token_clusters 3 _

/-- The original proposition, including its universal weak-solution input,
applies to two concrete independent uniform one-token configurations.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
example :
    (Measure.pi fun _ : Idx 2 => Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3)
      {X : Idx 2 → SphereTuple 3 1 | X 0 0 ≠ X 1 0} = 1 := by
  exact generic_discrete_distinct 3 1 2 0 le_rfl le_rfl le_rfl
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3)
    (uniformTuple_uniformLaw (by norm_num) 1) (fun X => X 0)
    (generic_zero_one_token_clusters 3 _) 0 1 (by decide)

end Interpolation
end Transformer
