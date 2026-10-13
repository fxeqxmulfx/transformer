/-
# Null support for the actual normalized stationary dynamics

ArXiv:2601.21366v2, §6, `prop: unified.log`, part (a), and
`rem:unifiedlog-B`. The actual support is null for every original
uniform spherical probability law in every dimension at least two.

The finite weak-pattern cover gives a positive-measure pattern if the
original support has positive measure. The actual branch field vanishes
there by the actual stationarity hypothesis. Proved analytic uniqueness
then makes that branch zero on the entire original sphere. Its proved
integrating-factor identity makes the original transform even, and
proved transform injectivity makes the original measure antipodal.

For every support point in the chosen pattern, its actual antipode is
in the complementary pattern and in the actual support. Stationarity
and oddness of that actual complementary field make the first point
a zero of the complementary branch. Its proved null zero set contradicts
the original positive measure, by the signed Gram non-degeneracy.

All zero neurons and activation boundaries remain in the finite cover.
The source assumes a symmetric invertible matrix; only invertibility
is required by this proved argument for the original weighted field.
The source's scalar parameter is positive. The scalar field result
below extends it to every nonzero parameter through the actual
matrix-field identity, retaining the original probability measure.
-/

import Transformer.Perceptron.Section6_BranchObstruction
import Transformer.Perceptron.Section6_AntipodalFields
import Transformer.Perceptron.TransformMap

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The support of an actual normalized stationary probability measure
is null for the original uniform spherical law under non-degeneracy.
Source: arXiv:2601.21366v2, §6, `prop: unified.log`, part (a),
and `rem:unifiedlog-B`. The source additionally assumes symmetry of `B`;
the proved weighted-field argument only needs invertibility for parity.
No analytic continuation, antipodal invariance or null-set conclusion
is included as an input hypothesis. -/
theorem measure_support_eq_zero_of_normalizedStationaryMap {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hB : Function.Bijective B)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere d)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ν (μ : Measure (SSphere d)).support = 0 := by
  classical
  by_contra hne
  obtain ⟨J, hpos⟩ := exists_positive_support_pattern a μ ν (pos_iff_ne_zero.mpr hne)
  have hposZeros : 0 < ν {x : SSphere d |
      normalizedBranchField B ω a μ J (x : EucSpace d) = 0} :=
    hpos.trans_le (measure_mono (fun x hx =>
      normalizedBranchField_zero_on_support B ω a μ hμ J x hx.1 hx.2))
  have hzero : ∀ x : SSphere d,
      normalizedBranchField B ω a μ J (x : EucSpace d) = 0 :=
    vector_eq_zero_on_sphere_of_positive_zero_measure hd _
      (analyticOnNhd_normalizedBranchField B ω a μ J) ν hν hposZeros
  have hanti : (μ : Measure (SSphere d)).map (antipodeMap d) =
      (μ : Measure (SSphere d)) :=
    (even_attentionTransformMap_iff B hB μ).mp
      (even_attentionTransformMap_of_spherical_branch_zero hd B ω a μ J hzero)
  have hnull := measure_complement_branch_zeros_eq_zero hd B ω a hnd μ J hzero ν hν
  have hpatternNull : ν ((μ : Measure (SSphere d)).support ∩
      {x : SSphere d | IsActivationPattern a J (x : EucSpace d)}) = 0 := by
    apply measure_mono_null _ hnull
    intro x hx
    have hxanti := antipode_mem_support_of_invariant (μ : Measure (SSphere d))
      hanti x hx.1
    have hpatanti : IsActivationPattern a Jᶜ (antipodeMap d x : EucSpace d) := by
      simpa only [coe_antipodeMap] using
        isActivationPattern_compl_neg a J (x : EucSpace d) hx.2
    have hz := normalizedBranchField_zero_on_support B ω a μ hμ Jᶜ
      (antipodeMap d x) hxanti hpatanti
    change normalizedBranchField B ω a μ Jᶜ (-(x : EucSpace d)) = 0 at hz
    rw [normalizedBranchField_neg_of_antipodal B ω a μ hanti Jᶜ] at hz
    exact neg_eq_zero.mp hz
  exact hpos.ne' hpatternNull

/-- Scalar normalized stationarity has uniform-null support for every
nonzero interaction parameter. Source: arXiv:2601.21366v2,
§6, `prop: unified.log`, part (a), states this for `β > 0`.
The proved actual matrix-field argument allows `β ≠ 0`, because the
original exponential weight is positive for either sign and `β I` is
invertible. No weights, probability measure or neuron are replaced. -/
theorem measure_support_eq_zero_of_normalizedStationary {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (hnd : IsNonDegenerate ω a) (μ : Perspective.ProbSphere d)
    (hμ : IsNormalizedStationary β (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ν (μ : Measure (SSphere d)).support = 0 := by
  have hB : Function.Bijective (β • LinearMap.id (R := ℝ) (M := EucSpace d)) := by
    change Function.Bijective (fun x : EucSpace d => β • x)
    exact (isUnit_iff_ne_zero.mpr hβ).smul_bijective (β := EucSpace d)
  exact measure_support_eq_zero_of_normalizedStationaryMap hd _ hB ω a hnd μ
    ((isNormalizedStationaryMap_smul_id_iff β hβ (fun s => max s 0) ω a μ).mpr hμ) ν hν

/-- Every matrix null-support hypothesis holds jointly for the original
pinned Dirac, identity matrix and the actual uniform circle law.
The genuine neuron is inactive at the atom and non-degenerate for every
index subset. Source: arXiv:2601.21366v2, §6,
`eq: nondeg.normalized`, `prop: unified.log` and `rem:unifiedlog-B`. -/
example : (2 : ℕ) ≤ 2 ∧
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    IsNonDegenerate (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun s => max s 0) (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨le_rfl, Function.bijective_id, isNonDegenerate_pin,
    isNormalizedStationaryMap_relu_pin 0 (basePoint 1),
    Metastability.isUniformOn_uniformLaw (by norm_num)⟩

/-- The scalar hypotheses hold jointly even at the additional negative
parameter `β = -1`, with the same actual non-degenerate pinned Dirac and
uniform law. Its stationarity is the original drift identity at its atom.
Source: arXiv:2601.21366v2, §6, `prop: unified.log`; the extension from
positive to nonzero `β` is documented in the preceding theorem. -/
example : (2 : ℕ) ≤ 2 ∧ (-1 : ℝ) ≠ 0 ∧
    IsNonDegenerate (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsNormalizedStationary (-1) (fun s => max s 0) (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨le_rfl, by norm_num, isNonDegenerate_pin,
    isNormalizedStationary_relu_pin (-1) 0 (basePoint 1),
    Metastability.isUniformOn_uniformLaw (by norm_num)⟩

/-- Applying the actual matrix theorem to its full witness gives a
uniform-null support of a genuine probability measure, rather than an
empty-support example. Source: arXiv:2601.21366v2, §6,
`prop: unified.log`, part (a), and `rem:unifiedlog-B`.
The inactive neuron pins the atom while the zero second neuron lies on
every activation boundary. Both remain in the original perceptron.
The measured support here is the original Dirac support, without any
replacement by the zero set of an auxiliary branch. -/
example : (MeanField.uniformLaw 2)
    (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support = 0 := by
  exact measure_support_eq_zero_of_normalizedStationaryMap le_rfl
    (LinearMap.id (R := ℝ) (M := EucSpace 2)) Function.bijective_id
    (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
    isNonDegenerate_pin (Perspective.diracProb 2 (basePoint 1))
    (isNormalizedStationaryMap_relu_pin 0 (basePoint 1))
    (MeanField.uniformLaw 2) (Metastability.isUniformOn_uniformLaw (by norm_num))

end Transformer.Perceptron
