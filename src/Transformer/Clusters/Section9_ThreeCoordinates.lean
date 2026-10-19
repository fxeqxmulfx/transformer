module

public import Transformer.Clusters.Section9_ThreeCandidates
public import Transformer.Clusters.Section9_Fj
public import Transformer.Clusters.Section8_LimitPrinciples

/-!
# Convergence to three leading coordinates for an actual good triple

ArXiv:2305.05465v6, §9, the proof of `l:3hyperplanes11`.
The extrema from `l:fj` converge to `a` and `b` in `e:defab`.
The actual score and corrected-energy estimates force every leading
cluster value into `{0,a,b}`. Continuity then gives convergence to
one member of that finite set.

Compact confinement uses the initial coordinate interval. The limiting
interval `[a,b]` need not contain coordinates at finite times, so it is
not used as a confinement assumption.

All estimates are derived from the original good-triple dynamics.
Positivity is needed only along the leading eigenline, and the
spectral complement need not be diagonalizable. The construction
does not assume convergence of any leading coordinate or boundedness
of the full tokens in their complementary directions.
-/

@[expose] public section

open Real Filter Topology

namespace Transformer.Clusters

variable {d m : ℕ}

/-- Each eigencoordinate with nonnegative eigenvalue stays in the
initial minimum--maximum interval at future times.
Source: arXiv:2305.05465v6, §9, `l:fj` and the bounds preceding `e:defab`. -/
theorem RescaledDynamics.mem_initial_eigenCoord_interval (Q K V : ParamMatrix d)
    (f : EucSpace d →L[ℝ] ℝ) (lam : ℝ) (hf : IsEigenFunctional V f lam)
    (hlam : 0 ≤ lam) (Z : ℝ → Idx (m + 1) → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (i : Idx (m + 1)) (t : ℝ) (ht : 0 ≤ t) :
    f (Z t i) ∈ Set.Icc (minCoord f (Z 0)) (maxCoord f (Z 0)) := by
  obtain ⟨hmax, hmin⟩ :=
    maxCoord_antitoneOn_minCoord_monotoneOn Q K V f lam hf hlam Z hZ
  have h0 : (0 : ℝ) ∈ Set.Ici (0 : ℝ) := Set.mem_Ici.mpr le_rfl
  exact ⟨(hmin h0 (Set.mem_Ici.mpr ht) ht).trans (minCoord_le f (Z t) i),
    (le_maxCoord f (Z t) i).trans (hmax h0 (Set.mem_Ici.mpr ht) ht)⟩

example (f : EucSpace d →L[ℝ] ℝ) (z : EucSpace d) :
    IsEigenFunctional (1 : ParamMatrix d) f 1 ∧ (0 : ℝ) ≤ 1 ∧
      RescaledDynamics (n := m + 1) (1 : ParamMatrix d) 1 1 (fun _ _ => z) ∧
      (0 : ℝ) ≤ 0 :=
  ⟨isEigenFunctional_one f, zero_le_one, rescaledDynamics_one_const _ _ z, le_rfl⟩

/-- A good triple constructs a representing metric and gives vanishing
gain for every actual leading token. Source: arXiv:2305.05465v6, §9,
`c₁₁ > 0` and the leading attention gaps in `eq:phistarvar`.
The form is represented only on the leading line, as required by `d:good`. -/
theorem RescaledDynamics.exists_leading_form_with_vanishing_gain
    (Q K V : ParamMatrix d) (lam : ℝ) (φ : EucSpace d)
    (G : Submodule ℝ (EucSpace d)) (hgood : IsGoodTripleWith Q K V lam φ G)
    (f : EucSpace d →L[ℝ] ℝ) (hf : IsEigenFunctional V f lam) (hfφ : f φ = 1)
    (Z : ℝ → Idx (m + 1) → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ A : ParamMatrix d, Function.Injective A ∧
      IsAttentionFormOn Q K A (Submodule.span ℝ {φ}) ∧
      ∀ i : Idx (m + 1),
        Tendsto (fun t => attentionGain A (fun j => (f.smulRight φ) (Z t j)) i)
          atTop (nhds 0) := by
  obtain ⟨A, hA, hform⟩ := exists_injective_attentionFormOn_span_singleton
    Q K φ hgood.2.1 hgood.2.2.2.2.2.2
  refine ⟨A, hA, hform, fun i => ?_⟩
  exact hZ.tendsto_attentionGain_zero_of_leading_form
    Q K V (f.smulRight φ) A (Submodule.span ℝ {φ}) G lam
    (hgood.isParanormalOn Q K V lam φ G)
    (hgood.isProjOnto_smulRight Q K V lam φ G f hf hfφ) hform Z i

example : IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1
      (EuclideanSpace.single 0 (1 : ℝ)) ⊥ ∧
    IsEigenFunctional (1 : ParamMatrix 1) (EuclideanSpace.proj (𝕜 := ℝ) 0) 1 ∧
    (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨isGoodTripleWith_one, isEigenFunctional_one _, by simp,
    rescaledDynamics_one_const 1 1 _⟩

/-- **The actual good-triple ODE has ordered extreme limits and only
three leading cluster values.** Source: arXiv:2305.05465v6, §9,
`e:defab`, `eq:phistarvar`, and the conclusion following `e:movem`.
The representing form and vanishing gain are constructed here; they
are not additional assumptions on the dynamics. -/
theorem RescaledDynamics.exists_extreme_limits_and_leading_clusters
    (Q K V : ParamMatrix d) (lam : ℝ) (φ : EucSpace d)
    (G : Submodule ℝ (EucSpace d)) (hgood : IsGoodTripleWith Q K V lam φ G)
    (f : EucSpace d →L[ℝ] ℝ) (hf : IsEigenFunctional V f lam) (hfφ : f φ = 1)
    (Z : ℝ → Idx (m + 1) → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ a b : ℝ, a ≤ b ∧
      Tendsto (fun t => minCoord f (Z t)) atTop (nhds a) ∧
      Tendsto (fun t => maxCoord f (Z t)) atTop (nhds b) ∧
      ∀ (i : Idx (m + 1)) (p : ℝ), MapClusterPt p atTop (fun t => f (Z t i)) →
        p = 0 ∨ p = a ∨ p = b := by
  obtain ⟨hmax, hmin⟩ :=
    maxCoord_antitoneOn_minCoord_monotoneOn Q K V f lam hf hgood.1.le Z hZ
  obtain ⟨⟨b, hb, _⟩, a, ha, _⟩ := exists_tendsto_maxCoord_minCoord f Z hmax hmin
  have hpositive : 0 < inner (𝕜 := ℝ) (Q φ) (K φ) := hgood.2.2.2.2.2.2
  obtain ⟨A, _, hform, hgain⟩ :=
    hZ.exists_leading_form_with_vanishing_gain Q K V lam φ G hgood f hf hfφ Z
  refine ⟨a, b, minCoord_limit_le_maxCoord_limit f Z a b ha hb, ha, hb, ?_⟩
  intro i p hp
  exact eigenCoord_cluster_eq_zero_or_extreme Q K A φ f hform hpositive Z a b ha hb i
    (hgain i) p hp

/-- A nonzero stationary scalar trajectory satisfies every original
good-triple, eigenfunctional, normalization and ODE hypothesis.
Source: arXiv:2305.05465v6, §9, `l:3hyperplanes11`. -/
example : IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1
      (EuclideanSpace.single 0 (1 : ℝ)) ⊥ ∧
    IsEigenFunctional (1 : ParamMatrix 1) (EuclideanSpace.proj (𝕜 := ℝ) 0) 1 ∧
    (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨isGoodTripleWith_one, isEigenFunctional_one _, by simp,
    rescaledDynamics_one_const 1 1 _⟩

/-- **Every actual leading coordinate converges to one of at most
three reals, with the original hypotheses of the paper.**
Source: arXiv:2305.05465v6, §9, the first paragraph and conclusion of
the proof of `l:3hyperplanes11`. Compactness supplies a cluster value;
continuity and the proved finite classification give convergence.
The compact interval comes from the initial configuration, not from
an assumption that a token already lies between its limiting extrema. -/
theorem RescaledDynamics.exists_three_leading_coordinate_limits
    (Q K V : ParamMatrix d) (lam : ℝ) (φ : EucSpace d)
    (G : Submodule ℝ (EucSpace d)) (hgood : IsGoodTripleWith Q K V lam φ G)
    (f : EucSpace d →L[ℝ] ℝ) (hf : IsEigenFunctional V f lam) (hfφ : f φ = 1)
    (Z : ℝ → Idx (m + 1) → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ S : Finset ℝ, S.card ≤ 3 ∧
      ∀ i : Idx (m + 1), ∃ c ∈ S, Tendsto (fun t => f (Z t i)) atTop (nhds c) := by
  classical
  obtain ⟨a, b, _, _, _, hcluster⟩ :=
    hZ.exists_extreme_limits_and_leading_clusters Q K V lam φ G hgood f hf hfφ Z
  let S : Finset ℝ := {0, a, b}
  refine ⟨S, Finset.card_le_three, fun i => ?_⟩
  apply exists_limit_of_finite_cluster_set (fun t => f (Z t i))
    (f.continuous.comp (hZ.continuous_token Q K V Z i))
    (Set.Icc (minCoord f (Z 0)) (maxCoord f (Z 0))) (S : Set ℝ)
    isCompact_Icc S.finite_toSet
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    exact hZ.mem_initial_eigenCoord_interval Q K V f lam hf hgood.1.le Z i t ht
  · intro p _ hp
    rcases hcluster i p hp with rfl | rfl | rfl <;> simp [S]

/-- Every convergence-theorem hypothesis is realized by the same
actual nonzero stationary scalar example.
Source: arXiv:2305.05465v6, §9, `l:3hyperplanes11`. -/
example : IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1
      (EuclideanSpace.single 0 (1 : ℝ)) ⊥ ∧
    IsEigenFunctional (1 : ParamMatrix 1) (EuclideanSpace.proj (𝕜 := ℝ) 0) 1 ∧
    (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨isGoodTripleWith_one, isEigenFunctional_one _, by simp,
    rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters
