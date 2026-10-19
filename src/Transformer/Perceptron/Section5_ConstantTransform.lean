module

public import Transformer.Perceptron.Section5_UniformInteraction
public import Transformer.Perceptron.Section5_QuadraticPolynomials
public import Mathlib.MeasureTheory.Integral.Prod

/-!
# The original degree-zero polynomial transform characterization

ArXiv:2601.21366v2, §5.1, `lem: quadpol` (i), the polynomial
characterization at degree zero. A probability measure has constant
original exponential transform exactly when it is the original uniform
spherical law, at every nonzero temperature.

The uniform transform's constancy was proved from orthogonal invariance.
Fubini for the actual continuous exponential kernel compares its value
with the actual uniform average of any probability measure's transform.
If the latter transform is constant, those two constants agree. The
proved injectivity theorem then identifies the two original measures.

A polynomial of total degree zero is its actual constant coefficient.
For a constant candidate density, the original probability normalization
forces its actual ENNReal weight to be one, even with the original
`ofReal` convention. Thus the source's full polynomial-density equivalence
is proved at degree zero, without using the open higher-degree statement.
The source fixes positive temperature; nonzero negative temperature also
works by the already proved signed-temperature injectivity theorem.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Perceptron

/-- A spherical polynomial of degree at most zero is an actual constant.
Source: arXiv:2601.21366v2, §5.1, the `k=0` case in `lem: quadpol` (i).
The statement uses the original multivariate polynomial and its coefficient. -/
theorem isPolyOfDegreeLE_zero_iff_constant {d : ℕ} (f : SSphere d → ℝ) :
    IsPolyOfDegreeLE 0 f ↔ ∃ C : ℝ, ∀ x, f x = C := by
  constructor
  · rintro ⟨P, hP, hval⟩
    have hzero : P.totalDegree = 0 := le_antisymm hP (Nat.zero_le _)
    have heq := MvPolynomial.totalDegree_eq_zero_iff_eq_C.mp hzero
    refine ⟨P.coeff 0, fun x => ?_⟩
    have hx := hval x
    rw [heq, MvPolynomial.eval_C] at hx
    exact hx
  · rintro ⟨C, hC⟩
    exact ⟨MvPolynomial.C C, by simp, fun x => by simp [hC x]⟩

/-- The original uniform average of any probability measure's transform
is the original uniform transform's value at every sphere point.
Source: arXiv:2601.21366v2, §5.1, the degree-zero harmonic in `eq:FH-basic`.
The equality uses the original two probability integrals and the same kernel. -/
theorem integral_attentionTransform_eq_uniform_value {d : ℕ} (β : ℝ)
    (μ : Perspective.ProbSphere d) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (x : SSphere d) :
    (∫ z, attentionTransform β μ z ∂ν) =
      attentionTransform β (⟨ν, hν.1⟩ : Perspective.ProbSphere d) x := by
  let _ : IsProbabilityMeasure ν := hν.1
  let σ : Perspective.ProbSphere d := ⟨ν, hν.1⟩
  have hf : Continuous (fun p : SSphere d × SSphere d =>
      Real.exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d))) :=
    Real.continuous_exp.comp (continuous_const.mul
      ((continuous_subtype_val.comp continuous_fst).inner
        (continuous_subtype_val.comp continuous_snd)))
  calc
    _ = ∫ y, ∫ z, Real.exp (β * inner (𝕜 := ℝ) (z : EucSpace d) (y : EucSpace d)) ∂ν
        ∂(μ : Measure (SSphere d)) :=
      integral_integral_swap_of_hasCompactSupport hf (HasCompactSupport.of_compactSpace _)
    _ = ∫ y, attentionTransform β σ y ∂(μ : Measure (SSphere d)) := by
      apply integral_congr_ae
      exact ae_of_all _ fun y => by
        change (∫ z, Real.exp (β * inner (𝕜 := ℝ) (z : EucSpace d) (y : EucSpace d)) ∂ν) =
          (∫ z, Real.exp (β * inner (𝕜 := ℝ) (y : EucSpace d) (z : EucSpace d)) ∂ν)
        simp only [real_inner_comm (y : EucSpace d)]
    _ = attentionTransform β σ x := by
      simp_rw [attentionTransform_eq_of_uniform β σ hν _ x]
      simp

/-- A constant original transform identifies the original measure with
the original uniform law. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i), degree zero. Its constant is recovered by actual uniform
averaging before the proved injectivity theorem is used. -/
theorem eq_uniform_of_constant_attentionTransform {d : ℕ} (β : ℝ) (hβ : β ≠ 0)
    (μ : Perspective.ProbSphere d) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν)
    (hconst : ∃ C : ℝ, ∀ x, attentionTransform β μ x = C) :
    (μ : Measure (SSphere d)) = ν := by
  let _ : IsProbabilityMeasure ν := hν.1
  let σ : Perspective.ProbSphere d := ⟨ν, hν.1⟩
  let x : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  obtain ⟨C, hC⟩ := hconst
  have hvalue : C = attentionTransform β σ x := by
    have havg := integral_attentionTransform_eq_uniform_value β μ ν hν x
    simpa only [hC, integral_const, probReal_univ, one_smul] using havg
  have heq : attentionTransform β μ = attentionTransform β σ := by
    funext y
    exact (hC y).trans (hvalue.trans (attentionTransform_eq_of_uniform β σ hν y x).symm)
  exact congrArg (fun m : Perspective.ProbSphere d => (m : Measure (SSphere d)))
    (injective_attentionTransform β hβ heq)

/-- The original degree-zero transform criterion is equality with the
original uniform law. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i). Every dimension and every original probability are retained. -/
theorem isPolyOfDegreeLE_zero_attentionTransform_iff_eq_uniform {d : ℕ}
    (β : ℝ) (hβ : β ≠ 0) (μ : Perspective.ProbSphere d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    IsPolyOfDegreeLE 0 (attentionTransform β μ) ↔ (μ : Measure (SSphere d)) = ν := by
  constructor
  · intro h
    exact eq_uniform_of_constant_attentionTransform β hβ μ ν hν
      ((isPolyOfDegreeLE_zero_iff_constant _).mp h)
  · intro heq
    have hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d)) := by rwa [heq]
    apply (isPolyOfDegreeLE_zero_iff_constant _).mpr
    let x : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
    exact ⟨attentionTransform β μ x, fun y => attentionTransform_eq_of_uniform β μ hμ y x⟩

/-- The source's original polynomial-density equivalence holds at degree
zero. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), `k=0`.
The actual probability mass forces the constant density's actual weight
to be one; no inverse-transform statement for higher degree is assumed. -/
theorem isPolyOfDegreeLE_zero_attentionTransform_iff_density {d : ℕ}
    (β : ℝ) (hβ : β ≠ 0) (μ : Perspective.ProbSphere d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    IsPolyOfDegreeLE 0 (attentionTransform β μ) ↔
      ∃ g : SSphere d → ℝ, IsPolyOfDegreeLE 0 g ∧
        (μ : Measure (SSphere d)) = ν.withDensity fun y => ENNReal.ofReal (g y) := by
  let _ : IsProbabilityMeasure ν := hν.1
  constructor
  · intro h
    have heq := (isPolyOfDegreeLE_zero_attentionTransform_iff_eq_uniform β hβ μ ν hν).mp h
    exact ⟨fun _ => 1, isPolyOfDegreeLE_const 1, by simpa using heq⟩
  · rintro ⟨g, hg, heq⟩
    obtain ⟨C, hC⟩ := (isPolyOfDegreeLE_zero_iff_constant g).mp hg
    simp_rw [hC] at heq
    rw [withDensity_const] at heq
    have hmass : ENNReal.ofReal C = 1 := by
      have h := congrArg (fun m : Measure (SSphere d) => m univ) heq
      simpa only [measure_univ, Measure.smul_apply, smul_eq_mul, mul_one] using h.symm
    rw [hmass, one_smul] at heq
    exact (isPolyOfDegreeLE_zero_attentionTransform_iff_eq_uniform β hβ μ ν hν).mpr heq

/-- All constant-transform, averaging, nonzero-temperature and original
uniform-law inputs hold jointly for the constructed circle probability.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), `k=0`.
The density is the actual constant one and has original probability mass one. -/
example : ∃ (μ : Perspective.ProbSphere 2) (ν : Measure (SSphere 2)),
    (1 : ℝ) ≠ 0 ∧ Metastability.IsUniformOn 2 ν ∧ (μ : Measure (SSphere 2)) = ν ∧
    (∃ C : ℝ, ∀ x, attentionTransform 1 μ x = C) ∧
    ∃ g : SSphere 2 → ℝ, IsPolyOfDegreeLE 0 g ∧
      (μ : Measure (SSphere 2)) = ν.withDensity fun y => ENNReal.ofReal (g y) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  exact ⟨μ, MeanField.uniformLaw 2, one_ne_zero, hν, rfl,
    ⟨attentionTransform 1 μ (basePoint 1), fun x => attentionTransform_eq_of_uniform 1 μ hν x _⟩,
    fun _ => 1, isPolyOfDegreeLE_const 1, by simp [μ]⟩

end Transformer.Perceptron
