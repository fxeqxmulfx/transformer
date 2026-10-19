module

public import Transformer.Perceptron.Section5_BiasedBranches
public import Transformer.Perceptron.Section5_AffineQuadraticDensity

/-!
# The original affine-quadratic transform of a zero biased branch

ArXiv:2601.21366v2, §3.1, `rem: ext` (i), with §5.2 and §5.3,
`eq:global-Id`. A globally zero original biased branch makes the actual
scaled first variation constant on the original sphere. Expanding its
actual biased coordinate squares gives original constant, linear and
quadratic terms, all computed explicitly below.

The constant is fixed at a supplied actual sphere point. The source's
transverse hyperplane supplies such a point for the later source
statements. The original kernel inverse then gives a nonnegative
polynomial density and full support of that same original probability.

The actual finitely many biased activation patterns cover every original
point. This also supplies the actual infinite-angle and positive-measure
pattern selections used by the source's two biased ReLU conclusions.
No field identity, polynomial representation, inverse or support result
is hidden in a definition or supplied as an unproved conclusion.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The original biased square sum has its actual constant, linear and
quadratic coefficients. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i),
and §5.3, the biased expansion of `eq:global-Id`.
The actual original bias appears in both the constant and linear terms. -/
theorem const_sub_half_weighted_biased_squares_eq {d : ℕ} (C β : ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (b : Idx d → ℝ)
    (J : Finset (Idx d)) (z : EucSpace d) :
    C - β / 2 * (∑ j ∈ J, ω j * (inner (𝕜 := ℝ) (a j) z + b j) ^ 2) =
      (C - β / 2 * ∑ j ∈ J, ω j * b j ^ 2) +
        inner (𝕜 := ℝ) (∑ j ∈ J, (-β * ω j * b j) • a j) z +
        ∑ j ∈ J, (-β / 2 * ω j) * inner (𝕜 := ℝ) (a j) z ^ 2 := by
  have hs : (∑ j ∈ J, (-β * ω j * b j) * inner (𝕜 := ℝ) (a j) z) +
      (∑ j ∈ J, (-β / 2 * ω j) * inner (𝕜 := ℝ) (a j) z ^ 2) =
        β / 2 * (∑ j ∈ J, ω j * b j ^ 2) -
          β / 2 * (∑ j ∈ J, ω j * (inner (𝕜 := ℝ) (a j) z + b j) ^ 2) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [sum_inner]
  simp_rw [real_inner_smul_left]
  linarith

/-- A globally zero original biased branch gives the actual original
affine-quadratic transform. Source: arXiv:2601.21366v2, §5.3,
`eq:global-Id`, with §3.1, `rem: ext` (i).
The original constant is computed at the actual supplied sphere point. -/
theorem exists_affine_add_squares_transform_of_zero_biased_branch {d : ℕ}
    (hd : 2 ≤ d) (β : ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (b : Idx d → ℝ)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) (o : SSphere d)
    (hzero : ∀ x : SSphere d, proj d (x : EucSpace d)
      (biasedAttentionVector β ω a b μ J (x : EucSpace d)) = 0) :
    ∃ C : ℝ, ∀ x : SSphere d, attentionTransform β μ x = C +
      inner (𝕜 := ℝ) (∑ j ∈ J, (-β * ω j * b j) • a j) (x : EucSpace d) +
      ∑ j ∈ J, (-β / 2 * ω j) * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2 := by
  have hconst := eq_on_sphere_of_projected_derivative_zero hd (biasedAttentionPotential β ω a b μ J)
    (biasedAttentionVector β ω a b μ J) (hasFDerivAt_biasedAttentionPotential β ω a b μ J) fun z hz => by
      exact hzero ⟨z, mem_sphere_zero_iff_norm.mpr hz⟩
  refine ⟨biasedAttentionPotential β ω a b μ J (o : EucSpace d) - β / 2 * ∑ j ∈ J, ω j * b j ^ 2, fun x => ?_⟩
  rw [← const_sub_half_weighted_biased_squares_eq]
  have h := hconst x o
  have hW : attentionWeightMap (β • LinearMap.id) μ (x : EucSpace d) = attentionTransform β μ x := by
    simp only [attentionWeightMap, attentionTransform, LinearMap.smul_apply, LinearMap.id_apply, real_inner_smul_right]
  change attentionWeightMap (β • LinearMap.id) μ (x : EucSpace d) + β *
    ((1 / 2 : ℝ) * ∑ j ∈ J, ω j * (inner (𝕜 := ℝ) (a j) (x : EucSpace d) + b j) ^ 2) =
      biasedAttentionPotential β ω a b μ J (o : EucSpace d) at h
  rw [hW] at h
  linarith

/-- A globally zero original biased branch gives full original support
at positive temperature. Source: arXiv:2601.21366v2, §5.3,
the full-support inference after `eq:global-Id`, with §3.1, `rem: ext` (i).
All actual original constant, linear and quadratic terms enter the proved inverse. -/
theorem support_eq_univ_of_zero_projected_biased_branch {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (b : Idx d → ℝ)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) (o : SSphere d)
    (hzero : ∀ x : SSphere d, proj d (x : EucSpace d)
      (biasedAttentionVector β ω a b μ J (x : EucSpace d)) = 0) :
    (μ : Measure (SSphere d)).support = univ := by
  obtain ⟨C, hC⟩ := exists_affine_add_squares_transform_of_zero_biased_branch hd β ω a b μ J o hzero
  exact support_eq_univ_of_affine_add_squares_transform hd J β hβ μ C
    (∑ j ∈ J, (-β * ω j * b j) • a j) (fun j => -β / 2 * ω j) a hC

/-- An actual infinite set of original angles has infinitely many
angles in one actual biased pattern. Source: arXiv:2601.21366v2,
§5.2, the infinite-pattern selection, with §3.1, `rem: ext` (i).
The actual finite cover includes original zero neurons and boundary points. -/
theorem exists_infinite_biased_activation_angles (a : Idx 2 → EucSpace 2) (b : Idx 2 → ℝ)
    (S : Set ℝ) (hS : S.Infinite) :
    ∃ J : Finset (Idx 2), (S ∩ {θ | IsBiasedActivationPattern a b J (circlePoint θ : EucSpace 2)}).Infinite := by
  classical
  by_contra h
  have hf : ∀ J : Finset (Idx 2),
      (S ∩ {θ | IsBiasedActivationPattern a b J (circlePoint θ : EucSpace 2)}).Finite := by
    intro J
    by_contra hJ
    exact h ⟨J, hJ⟩
  apply hS
  apply (finite_iUnion hf).subset
  intro θ hθ
  obtain ⟨J, hJ⟩ := exists_biased_activationPattern a b (circlePoint θ : EucSpace 2)
  exact mem_iUnion.mpr ⟨J, hθ, hJ⟩

/-- One actual biased-pattern intersection has positive original
measure whenever the original set does. Source: arXiv:2601.21366v2,
§5.3, the positive-pattern selection, with §3.1, `rem: ext` (i).
The actual finite cover retains every original point and bias. -/
theorem exists_positive_measure_biased_activationPattern {d : ℕ} (ν : Measure (SSphere d))
    (S : Set (SSphere d)) (hS : 0 < ν S) (a : Idx d → EucSpace d) (b : Idx d → ℝ) :
    ∃ J : Finset (Idx d), 0 < ν (S ∩ {x : SSphere d | IsBiasedActivationPattern a b J (x : EucSpace d)}) := by
  have hcover : (⋃ J : Finset (Idx d), S ∩ {x : SSphere d | IsBiasedActivationPattern a b J (x : EucSpace d)}) = S := by
    ext x
    constructor
    · intro hx
      obtain ⟨J, hx⟩ := mem_iUnion.mp hx
      exact hx.1
    · intro hx
      obtain ⟨J, hJ⟩ := exists_biased_activationPattern a b (x : EucSpace d)
      exact mem_iUnion.mpr ⟨J, hx, hJ⟩
  apply exists_measure_pos_of_not_measure_iUnion_null
  rw [hcover]
  exact hS.ne'

/-- Every zero-branch, positive-temperature, dimensional and positive
original measure input holds jointly for the actual uniform circle law
with zero weights and actual nonzero biases. The infinite-angle input
holds on the actual entire angle line. Source: arXiv:2601.21366v2,
§3.1, `rem: ext` (i), with §5.2 and §5.3. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ x : SSphere 2, proj 2 (x : EucSpace 2)
      (biasedAttentionVector 1 (0 : Idx 2 → ℝ) 0 (fun _ => (-1 : ℝ)) μ ∅ (x : EucSpace 2)) = 0) ∧
    0 < (μ : Measure (SSphere 2)) univ ∧ (univ : Set ℝ).Infinite := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  refine ⟨μ, le_rfl, one_pos, ?_, by simp, Set.infinite_univ⟩
  intro x
  rw [biasedAttentionVector]
  have hv : biasedBranchVector (0 : Idx 2 → ℝ) 0 (fun _ => (-1 : ℝ)) ∅ (x : EucSpace 2) = 0 := by
    simp [biasedBranchVector]
  rw [hv, smul_zero, add_zero, ← attentionGradMap_eq_proj_ambientGradMap]
  exact attentionGradMap_zero_of_uniform 1 μ hν x

end Transformer.Perceptron
