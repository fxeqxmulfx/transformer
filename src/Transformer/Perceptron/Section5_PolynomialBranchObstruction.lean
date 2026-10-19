module

public import Transformer.Perceptron.Section5_PolynomialBranches
public import Transformer.Perceptron.Section5_ReLUObstruction

/-!
# Non-analytic original potentials obstruct every zero polynomial branch

ArXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, and §5.3,
the contradiction after `eq:global-Id`. A projected polynomial branch
that vanishes on the entire original sphere has a constant actual first
variation. Its actual exponential transform is therefore polynomial of
some finite degree, with the original neurons and branch coefficients.

The proved original transform characterization gives a nonnegative
polynomial density for the same original probability. Its proved support
is the whole original sphere. Original stationarity then makes the actual
original perceptron potential analytic, contradicting the source's
non-analyticity hypothesis. The primitive of twice the original activation
is arbitrary and unchanged in this final contradiction.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- A globally zero projected original polynomial branch makes the
actual original transform a spherical polynomial of some finite degree.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, the extension of
`eq:global-Id`. Its original primitive terms have their proved degrees;
constancy follows from their proved actual ambient derivative. -/
theorem exists_polynomial_transform_of_zero_projected_polynomial_branch {d : ℕ}
    (hd : 2 ≤ d) (β : ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (P : Idx d → Polynomial ℝ) (μ : Perspective.ProbSphere d)
    (hzero : ∀ x : SSphere d,
      proj d (x : EucSpace d) (polynomialAttentionVector β ω a P μ (x : EucSpace d)) = 0) :
    ∃ k : ℕ, IsPolyOfDegreeLE k (attentionTransform β μ) := by
  have hconst := eq_on_sphere_of_projected_derivative_zero hd
    (polynomialAttentionPotential β ω a P μ) (polynomialAttentionVector β ω a P μ)
    (hasFDerivAt_polynomialAttentionPotential β ω a P μ) (fun z hz =>
      hzero ⟨z, mem_sphere_zero_iff_norm.mpr hz⟩)
  let o : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  let C := polynomialAttentionPotential β ω a P μ (o : EucSpace d)
  let k := ∑ j, ((P j).natDegree + 1)
  have hdeg (j : Idx d) : (P j).natDegree + 1 ≤ k :=
    Finset.single_le_sum (f := fun j => (P j).natDegree + 1)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ j)
  have hpoly : IsPolyOfDegreeLE k (fun x : SSphere d =>
      C + (-β) * polynomialBranchHalfPotential ω a P (x : EucSpace d)) :=
    isPolyOfDegreeLE_add (isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const C) (Nat.zero_le k))
      (isPolyOfDegreeLE_const_mul (-β) (isPolyOfDegreeLE_sum_polynomialPrimitives ω a P hdeg))
  have heq : attentionTransform β μ = (fun x : SSphere d =>
      C + (-β) * polynomialBranchHalfPotential ω a P (x : EucSpace d)) := by
    funext x
    have h := hconst x o
    change attentionWeightMap (β • LinearMap.id) μ (x : EucSpace d) +
      β * polynomialBranchHalfPotential ω a P (x : EucSpace d) = C at h
    have hW : attentionWeightMap (β • LinearMap.id) μ (x : EucSpace d) = attentionTransform β μ x := by
      simp only [attentionWeightMap, attentionTransform, LinearMap.smul_apply,
        LinearMap.id_apply, real_inner_smul_right]
    rw [hW] at h
    linarith
  exact ⟨k, heq ▸ hpoly⟩

/-- An original probability with an actual polynomial transform has
full original support in dimension at least two.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, the consequence
of `lem: quadpol` (i). The original inverse supplies the actual
nonnegative polynomial density before the support theorem is applied. -/
theorem support_eq_univ_of_polynomial_attentionTransform {d k : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (μ : Perspective.ProbSphere d)
    (hk : IsPolyOfDegreeLE k (attentionTransform β μ)) :
    (μ : Measure (SSphere d)).support = univ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := d) (by omega)
  obtain ⟨g, hg, hnn, heq⟩ :=
    (isPolyOfDegreeLE_attentionTransform_iff β hβ k μ (MeanField.uniformLaw d) hν).mp hk
  exact support_eq_univ_of_nonneg_polynomial_density hd μ (MeanField.uniformLaw d) hν g hg hnn heq

/-- The same original probability has full original support whenever
one original polynomial branch vanishes globally at positive temperature.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, the polynomial
density inference from `lem: quadpol` (i).
Both its actual density and pointwise nonnegativity are proved. -/
theorem support_eq_univ_of_zero_projected_polynomial_branch {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (P : Idx d → Polynomial ℝ) (μ : Perspective.ProbSphere d)
    (hzero : ∀ x : SSphere d,
      proj d (x : EucSpace d) (polynomialAttentionVector β ω a P μ (x : EucSpace d)) = 0) :
    (μ : Measure (SSphere d)).support = univ := by
  obtain ⟨k, hk⟩ := exists_polynomial_transform_of_zero_projected_polynomial_branch hd β ω a P μ hzero
  exact support_eq_univ_of_polynomial_attentionTransform hd β hβ μ hk

/-- The source's original non-analytic potential excludes every globally
zero original polynomial branch at an actual stationary probability.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, and §5.3,
the contradiction after `eq:global-Id`.
The original arbitrary primitive, activation, probability, stationarity
and non-analyticity are all retained in the full-support contradiction. -/
theorem not_zero_projected_polynomial_branch_of_nonanalytic {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β σ ω a μ) (P : Idx d → Polynomial ℝ) :
    ¬ ∀ x : SSphere d, proj d (x : EucSpace d)
      (polynomialAttentionVector β ω a P μ (x : EucSpace d)) = 0 := by
  intro hzero
  have hfull := support_eq_univ_of_zero_projected_polynomial_branch hd β hβ ω a P μ hzero
  exact not_full_support_of_nonanalytic_stationary hd β hβ.ne' φ σ hφ ω a hana μ hμ hfull

/-- All original dimensional, positive-temperature and zero-branch
inputs hold jointly for the constructed original uniform circle law,
actual zero neuron weights and the original polynomial `X`.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, the polynomial
density step. This uses the original nonzero probability mass. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    ∀ x : SSphere 2, proj 2 (x : EucSpace 2)
      (polynomialAttentionVector 1 (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2)
        (fun _ => Polynomial.X) μ (x : EucSpace 2)) = 0 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  refine ⟨μ, le_rfl, one_pos, fun x => ?_⟩
  rw [projected_polynomialAttentionVector_eq_energyGrad 1 (fun s => s) 0 0
    (fun _ => Polynomial.X) μ (x : EucSpace 2) (fun _ => by simp)]
  have hx : x ∈ (μ : Measure (SSphere 2)).support := by
    rw [support_eq_univ_of_uniform μ hν]
    exact mem_univ _
  rw [isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => s) 0 μ hν x hx, smul_zero]

/-- The polynomial-transform support inputs hold jointly for the actual
uniform circle probability and its constant original transform.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    IsPolyOfDegreeLE 0 (attentionTransform 1 μ) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  refine ⟨μ, le_rfl, one_pos, (isPolyOfDegreeLE_zero_iff_constant _).mpr ?_⟩
  exact ⟨attentionTransform 1 μ (basePoint 1), fun x => attentionTransform_eq_of_uniform 1 μ hν x _⟩

/-- Every original non-analytic obstruction input holds jointly for the
actual pinned Dirac and original ReLU primitive and neuron data.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, and §3.1,
`thm: circle`. Its source activation is unchanged. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨le_rfl, one_pos, hasDerivAt_reluSq,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis inner_basePoint_secondAxis,
    isStationary_relu_pin 1 0 (basePoint 1)⟩

end Transformer.Perceptron
