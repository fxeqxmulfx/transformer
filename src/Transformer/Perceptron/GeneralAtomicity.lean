/-
# Perceptrons and attention's mean-field landscape — atomicity for a general
  attention matrix

`rem: ext` (ii) of arXiv:2601.21366v2 in its carried forms: `thm: circle`,
`thm: circle.gelu` and `thm: any.d` (i) for the energy `E_B` of
`GeneralAttention.lean`, with a symmetric invertible `B` in place of `β I_d`.

**What the source says and what is carried here.**

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

import Transformer.Perceptron.GeneralAttention
import Transformer.Perceptron.HigherDim

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

Not proved here.

Source: arXiv:2601.21366v2, `rem: ext` (ii), `thm: circle`. -/
theorem ext_circle_isFinitelyAtomic (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (hB : Function.Bijective B) (φ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsStationaryMap B (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ := by
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

Not proved here.

Source: arXiv:2601.21366v2, `rem: ext` (ii), `thm: circle.gelu`. -/
theorem ext_circle_gelu_isFinitelyAtomic (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (hB : Function.Bijective B) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hσ : AnalyticOnNhd ℝ σ Set.univ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPDMap B φ σ ω a μ) :
    IsFinitelyAtomic μ := by
  sorry

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

Not proved here.

Source: arXiv:2601.21366v2, `rem: ext` (ii), `thm: any.d` (i). -/
theorem ext_any_d_measure_support_eq_zero (d : ℕ) (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hsymm : B.IsSymmetric) (hB : Function.Bijective B)
    (φ σ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hcrit : (σ = fun s => max s 0) ∧ ¬ IsAnalyticOnSphere φ ω a ∧
        IsStationaryMap B σ ω a μ ∨
      AnalyticOnNhd ℝ σ Set.univ ∧ IsStrictSOPDMap B φ σ ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  sorry

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
