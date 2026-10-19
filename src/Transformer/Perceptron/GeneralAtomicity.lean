module

public import Transformer.Perceptron.GeneralAttention
public import Transformer.Perceptron.HigherDim
public import Transformer.Perceptron.Section3_StrictMatrixAtomicity
public import Transformer.Perceptron.Section5_SphericalMapNullSupport
public import Transformer.Perceptron.Section5_ReflectionAtomicity

/-!
# Perceptrons and attention's mean-field landscape — atomicity for a general
  attention matrix

`rem: ext` (ii) of arXiv:2601.21366v2 in its carried forms: `thm: circle`,
`thm: circle.gelu` and `thm: any.d` (i) for the energy `E_B` of
`GeneralAttention.lean`, with a symmetric invertible `B` in place of `β I_d`.

* `rem: ext` (ii) says only "the results above easily extend"; the argument is
  `rem:unifiedlog-B`, which names `thm: circle` and `thm: any.d` and points at
  `rem: general-attention` for the injectivity and parity of `f_B` — both
  already in `Transform.lean`.
* `thm: any.d` (ii) and (iii) are not carried: they are genericity statements
  over `ℝ_{>0} × (ℝ^{d+1})^d`, and the source describes no generic set in the
  space of symmetric invertible matrices to put in place of the `β`.
* The witnesses take `B = id`, where `E_B` is the source's `E_{β,ϑ}` at
  `β = 1` on the nose — `energyMap_id_eq` — so the pinned Diracs of
  `Dirac.lean` serve here too, through `isStationaryMap_id_iff` and
  `isStrictSOPDMap_id_iff`.

Source: arXiv:2601.21366v2, `rem: ext` (ii), `rem:unifiedlog-B`.
-/

@[expose] public section

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

/-- The identity is a symmetric invertible attention matrix: the isotropic
case `β = 1`, and the witness every `B` below is instantiated at. -/
theorem isSymmetric_bijective_id (d : ℕ) :
    (LinearMap.id (R := ℝ) (M := EucSpace d)).IsSymmetric ∧
      Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace d)) :=
  ⟨fun _ _ => rfl, Function.bijective_id⟩

/-- **Remark (rem: ext) (ii), for `thm: circle`.**  With a symmetric invertible
`B` in the interaction energy, a stationary measure for the ReLU perceptron
with a non-analytic potential is still purely atomic with finite support.

The non-degenerate, conformal-matrix and commuting-neuron-reflection cases are proved.
Otherwise both complementary continued fields vanish globally for an
actual infinite support pattern, and its signed Gram matrix is scalar.
Scalar zero is excluded by analyticity of the original potential.
The remaining case without a commuting neuron reflection is unproved; the full claim is retained.
Source: arXiv:2601.21366v2, `rem: ext` (ii), `thm: circle`. -/
theorem ext_circle_isFinitelyAtomic (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (hB : Function.Bijective B) (φ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsStationaryMap B (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ := by
  by_cases hnd : IsNonDegenerate ω a
  · exact isFinitelyAtomic_of_stationaryMap_nondegenerate_circle B hB ω a hnd μ hμ
  by_cases hconf : ∃ (r : ℝ) (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2),
      0 < r ∧ B = r • U.toLinearEquiv.toLinearMap
  · obtain ⟨r, U, hr, rfl⟩ := hconf
    exact isFinitelyAtomic_of_stationaryMap_conformal_circle r hr U φ hφ ω a hana μ hμ
  by_contra hnot
  have hInf : (μ : Measure (SSphere 2)).support.Infinite :=
    fun hf => hnot (isFinitelyAtomic_of_finite_topological_support μ hf)
  obtain ⟨J, c, hpattern, hJ, hC, hscalar⟩ :=
    exists_scalar_support_pattern_of_stationaryMap_infinite_circle B hB ω a μ hμ hInf
  by_cases hc0 : c = 0
  · apply hana
    apply isAnalyticOnSphere_of_zero_signedGram_infinite_pattern φ hφ ω a J _ hpattern
      (fun θ hθ => hθ.1.1) (fun θ hθ => hθ.2)
    intro x
    rw [hscalar, hc0, zero_smul]
  have hframe := two_neuron_geometry_of_nonzero_scalar_signedGram ω a J c hc0 hscalar
  by_cases href : ∃ U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2,
      (∀ x, B (U x) = U (B x)) ∧ U (a 0) = -a 0 ∧ U (a 1) = a 1
  · obtain ⟨U, hcomm, h₀, h₁⟩ := href
    exact hnot (isFinitelyAtomic_of_stationaryMap_commuting_neuron_reflection B hB U hcomm ω a
      h₀ h₁ hframe.2.1 hframe.2.2.1 (hframe.2.2.2.2.2 0) μ hμ)
  sorry

/-- The hypotheses of `ext_circle_isFinitelyAtomic` are satisfiable at
`B = id`, where the pinned ReLU Dirac of `thm: circle`'s own witness is again
stationary — by `isStationaryMap_id_iff`. -/
example :
    (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
      Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
      (∀ s : ℝ, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
      ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun s => max s 0)
        (Pi.single 0 (1 : ℝ)) (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨(isSymmetric_bijective_id 2).1, (isSymmetric_bijective_id 2).2, hasDerivAt_reluSq,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis
      inner_basePoint_secondAxis,
    (isStationaryMap_id_iff _ _ _ _).mpr (isStationary_relu_pin 1 0 (basePoint 1))⟩

/-- **Remark (rem: ext) (ii), for `thm: circle.gelu`.**  With a symmetric
invertible `B` in the interaction energy, a strict SOPD critical point at a
real-analytic `σ` is still purely atomic with finite support.

The proved result strengthens the source by removing matrix invertibility.
The local proof already applies to a continuous actual activation derivative;
the analytic activation hypothesis and actual finite-atomic conclusion
are retained here. Symmetry is used in the genuine energy Hessian.

Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), `thm: circle.gelu`, and §5.2. -/
theorem ext_circle_gelu_isFinitelyAtomic (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hσ : AnalyticOnNhd ℝ σ Set.univ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPDMap B φ σ ω a μ) :
    IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_strictSOPDMap_analytic B hsymm φ σ hφ hσ ω a μ hμ

/-- The hypotheses of `ext_circle_gelu_isFinitelyAtomic` are satisfiable at
`B = id`, where the pinned Dirac of `isStrictSOPD_pin` is again a strict SOPD
critical point — by `isStrictSOPDMap_id_iff`. -/
example :
    (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
      Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
      (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) s) s) ∧
      AnalyticOnNhd ℝ (fun _ : ℝ => (2 : ℝ)⁻¹) Set.univ ∧
      IsStrictSOPDMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun s => s)
        (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨(isSymmetric_bijective_id 2).1, (isSymmetric_bijective_id 2).2,
    fun s => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := s), fun _ _ => analyticAt_const,
    (isStrictSOPDMap_id_iff _ _ _ _ _).mpr (isStrictSOPD_pin 1 0 (basePoint 1))⟩

/-- **Remark (rem: ext) (ii), for `thm: any.d` (i).**  In `d ≥ 2`, with a
symmetric invertible `B` in the interaction energy, the support of a
stationary measure for the ReLU perceptron with a non-analytic potential — or
of a strict SOPD critical point at a real-analytic `σ` — is still
`σ_d`-null.

The analytic strict alternative is proved. The ReLU alternative is proved
for conformal matrices and non-degenerate signed Gram data. Otherwise an
actual positive-measure support pattern gives two globally zero complementary
fields and a scalar signed Gram map; that case is unproved. Both alternatives remain.

Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), `thm: any.d` (i), and §5.3. -/
theorem ext_any_d_measure_support_eq_zero (d : ℕ) (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hsymm : B.IsSymmetric) (hB : Function.Bijective B)
    (φ σ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hcrit : (σ = fun s => max s 0) ∧ ¬ IsAnalyticOnSphere φ ω a ∧
        IsStationaryMap B σ ω a μ ∨
      AnalyticOnNhd ℝ σ Set.univ ∧ IsStrictSOPDMap B φ σ ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  intro ν hν
  rcases hcrit with ⟨rfl, hana, hstat⟩ | ⟨hσ, hstrict⟩
  · by_cases hnd : IsNonDegenerate ω a
    · exact measure_support_eq_zero_of_stationaryMap_nondegenerate hd B hB ω a hnd μ hstat ν hν
    by_cases hconf : ∃ (r : ℝ) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d),
        0 < r ∧ B = r • U.toLinearEquiv.toLinearMap
    · obtain ⟨r, U, hr, rfl⟩ := hconf
      exact measure_support_eq_zero_of_stationaryMap_conformal hd r hr U φ hφ ω a hana μ hstat ν hν
    by_contra hne
    obtain ⟨J, c, hpattern, hJ, hC, hscalar⟩ :=
      exists_scalar_support_pattern_of_stationaryMap_positive_uniform hd B hB ω a μ hstat ν hν
        (pos_iff_ne_zero.mpr hne)
    sorry
  · exact analytic_strictMap_measure_support_eq_zero hd B hsymm φ σ hφ hσ ω a μ hstrict ν hν

/-- The hypotheses of `ext_any_d_measure_support_eq_zero` are satisfiable at
`d = 2` and `B = id`, in the first of the two alternatives. -/
example :
    (2 : ℕ) ≤ 2 ∧ (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
      Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
      (((fun s => max s 0) = fun s : ℝ => max s 0) ∧
        ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
            (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
        IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun s => max s 0)
          (Pi.single 0 (1 : ℝ)) (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
          (Perspective.diracProb 2 (basePoint 1))) :=
  ⟨le_rfl, (isSymmetric_bijective_id 2).1, (isSymmetric_bijective_id 2).2, rfl,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis
      inner_basePoint_secondAxis,
    (isStationaryMap_id_iff _ _ _ _).mpr (isStationary_relu_pin 1 0 (basePoint 1))⟩

/-- The second alternative of `ext_any_d_measure_support_eq_zero` is also
satisfiable at `d = 2` and `B = id`. -/
example :
    AnalyticOnNhd ℝ (fun _ : ℝ => (2 : ℝ)⁻¹) Set.univ ∧
      IsStrictSOPDMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun s => s)
        (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨fun _ _ => analyticAt_const,
    (isStrictSOPDMap_id_iff _ _ _ _ _).mpr (isStrictSOPD_pin 1 0 (basePoint 1))⟩

end Perceptron
end Transformer
