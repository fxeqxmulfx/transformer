/-
# Perceptrons and attention's mean-field landscape — polynomial transforms

Formalization of `lem: quadpol` (i), "in particular", of arXiv:2601.21366v2:
`f^μ` is a polynomial of degree `≤ k` exactly when `μ` has a density with
respect to `σ_d` that is a polynomial of degree `≤ k`.  The transform itself
is `Transform`.

**What the source says and what is carried here.**

* "`f^μ` is a polynomial of degree `k`" is carried as "`f^μ` is the
  restriction to the sphere of a polynomial of total degree `≤ k`".  Degree is
  not well defined for a *function* on the sphere: `Σ_i x_i² = 1` there, so
  every representing polynomial can be raised by two, and only the bound
  survives.  The same reading is used for the density, and the source's `iff`
  is then an equality of the two least such bounds.

* `σ_d` is `Metastability.IsUniformOn`, quantified over inside the
  conclusion, as in `thm: any.d`. The original rotation-invariant law is
  constructed in `Metastability.Section4_UniformLaw` and witnesses the
  theorem jointly with its original uniform probability and density one.

* The source's standing `d ≥ 2` is not assumed, as in `Transform`.  On
  `𝕊^0 = {±1}` every function is `a + b x`, so for `k ≥ 1` both sides hold for
  every `μ`, and for `k = 0` both say `μ({1}) = 1/2`; `𝒫(𝕊^{-1})` is empty.

* The source says that the polynomial is a density of a nonnegative measure.
  The former formalization used only `ν.withDensity (ENNReal.ofReal ∘ g)`;
  this clips a negative polynomial and need not have a polynomial density.
  The explicit `∀ y, 0 ≤ g y` now preserves the source's density meaning.
  It is proved from the signed inverse in the forward direction, then used
  to integrate the actual density values in the converse.

* **What is not carried.**  The `L²(σ_d)` criterion of `lem: quadpol` (i) —
  `μ` has an `L²` density iff `Σ_{j,ℓ} |f̂^μ_{jℓ}/λ_j(β)|² < ∞`, with the
  density given by `eq:degreek` — is not stated.  Writing it down needs the
  spherical harmonics `Y_{jℓ}` and the Funk–Hecke eigenvalues
  `λ_j(β) = Γ(d/2)(2/β)^{d/2-1} I_{j+d/2-1}(β)`, which Mathlib does not have
  and which the source imports as standard; a definition supplying them would
  be the whole difficulty parked in a definition.  Its consequence — the
  polynomial characterization, which is what `rem:genericity-vacuity` uses —
  is `isPolyOfDegreeLE_attentionTransform_iff` below.

The proof uses the original homogeneous kernel moments, the actual closed
polynomial space and its proved spanning by powers of linear functions.
The original kernel therefore restricts to an injective endomorphism of
each finite-dimensional bounded-degree space. Its actual inverse gives a
continuous signed candidate; finite-measure injectivity identifies its
positive and negative parts with the original probability, establishing
nonnegativity and the actual measure equality.

Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
-/

import Transformer.Perceptron.Section5_PolynomialInverse

open scoped ENNReal
open MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-- **Lemma (lem: quadpol) (i), "in particular".**  `f^μ` is a polynomial of
degree `≤ k` if and only if `μ` has a density with respect to `σ_d` that is a
polynomial of degree `≤ k`.

The source's density is made explicitly nonnegative: the former
`ofReal`-only expression allowed clipped sign-changing polynomials.
The original kernel inverse proves this positivity from the original
nonnegative probability; it is not assumed in that construction.

Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i). -/
theorem isPolyOfDegreeLE_attentionTransform_iff (β : ℝ) (hβ : 0 < β) (k : ℕ)
    (μ : Perspective.ProbSphere d) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      (IsPolyOfDegreeLE k (attentionTransform β μ) ↔
        ∃ g : SSphere d → ℝ, IsPolyOfDegreeLE k g ∧ (∀ y, 0 ≤ g y) ∧
          (μ : Measure (SSphere d))
            = ν.withDensity fun y => ENNReal.ofReal (g y)) := by
  intro ν hν
  let _ : IsProbabilityMeasure ν := hν.1
  constructor
  · intro hf
    let f : C(SSphere d, ℝ) := ⟨attentionTransform β μ,
      Perspective.continuous_partitionMu d β hβ.le μ⟩
    obtain ⟨g, hg, hval⟩ := exists_polynomial_kernel_inverse_all_dimensions β hβ ν hν f hf
    obtain ⟨hnn, heq⟩ := nonneg_and_eq_withDensity_of_uniform_transform_eq β hβ
      (μ : Measure (SSphere d)) ν hν g g.continuous (fun x => (hval x).symm)
    exact ⟨g, hg, hnn, heq⟩
  · rintro ⟨g, hg, hnn, heq⟩
    have hc := continuous_of_isPolyOfDegreeLE hg
    let f : C(SSphere d, ℝ) := ⟨g, hc⟩
    have hp := (mem_spherePolynomialSpace_iff _).mp
      (sphereKernelOperator_preserves_polynomials_all_dimensions β ν hν f
        ((mem_spherePolynomialSpace_iff f).mpr hg))
    have hfun : attentionTransform β μ = (sphereKernelOperator β ν f : SSphere d → ℝ) := by
      funext x
      rw [attentionTransform, heq, integral_exp_inner_withDensity_of_nonneg β ν g hc hnn x]
      exact (sphereKernelOperator_apply β ν f x).symm
    rwa [← hfun] at hp

/-- Every original temperature, uniform-law and polynomial-density
condition holds jointly for the original uniform circle probability
and density one. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The explicit nonnegativity condition holds pointwise, and the actual
nonzero probability equality uses the original `withDensity` expression. -/
example : ∃ (μ : Perspective.ProbSphere 2) (ν : Measure (SSphere 2)), (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 ν ∧
    ∃ g : SSphere 2 → ℝ, IsPolyOfDegreeLE 2 g ∧ (∀ y, 0 ≤ g y) ∧
      (μ : Measure (SSphere 2)) = ν.withDensity (fun y => ENNReal.ofReal (g y)) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  refine ⟨μ, MeanField.uniformLaw 2, one_pos, hν, fun _ => 1,
    isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const 1) (Nat.zero_le 2), fun _ => zero_le_one, ?_⟩
  simp [μ]

end Perceptron
end Transformer
