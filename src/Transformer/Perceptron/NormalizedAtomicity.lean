/-
# Perceptrons and attention's mean-field landscape — sparsity for normalized
  attention

Formalization of `prop: unified.log` and `rem:unifiedlog-B` of
arXiv:2601.21366v2, §6: under the non-degeneracy condition
`eq: nondeg.normalized` on the perceptron weights, a stationary measure of the
*normalized* dynamics `eq:full-WGF-main` is `σ_d`-null, and in `d = 2` purely
atomic with finitely many atoms.

**What the source says and what is carried here.**

* The proposition's two conclusions are two theorems, as `thm: any.d` (i) and
  `thm: circle` are: `σ_d(supp μ) = 0` in every `d ≥ 2`, and finite atomicity
  in `d = 2`.

* `σ(s) = s_+` is substituted into the statement.  No primitive `φ` appears:
  `eq: fulltrans.stat` is a condition on the drift alone, and the potential
  `v_ϑ` enters only the proof.

* `σ_d` is `Metastability.IsUniformOn`, quantified over inside the conclusion,
  as in `thm: any.d`: the tree never constructs the uniform measure.

* "in particular, `μ` is singular with respect to `σ_d`" is
  `mutuallySingular_of_measure_support_eq_zero` of `HigherDim.lean`, proved
  there from the null-support conclusion, which is the same clause of
  `thm: any.d` (i); it applies verbatim here.

* `rem:unifiedlog-B` replaces `E_β` by `E_B` of `GeneralAttention.lean`, with
  the weight and the field of `NormalizedMap.lean`; its two statements are the
  `Map` versions below.  Its last sentence — that `thm: circle` and
  `thm: any.d` likewise extend to `E_B` — is `rem: ext` (ii), carried in
  `GeneralAtomicity.lean`.

Source: arXiv:2601.21366v2, `prop: unified.log`, `rem:unifiedlog-B`.
-/

import Transformer.Perceptron.NormalizedMap
import Transformer.Perceptron.SignedGram
import Transformer.Perceptron.GeneralAtomicity

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### `prop: unified.log` -/

/-- **Proposition (prop: unified.log), first conclusion.**  Let `d ≥ 2`,
`β > 0` and `σ(s) = s_+`, and let the weights satisfy the non-degeneracy
condition `eq: nondeg.normalized`.  Then the support of a measure stationary
for the normalized dynamics `eq:full-WGF-main` is `σ_d`-null.

Not proved here.

Source: arXiv:2601.21366v2, `prop: unified.log`. -/
theorem unifiedLog_measure_support_eq_zero (d : ℕ) (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere d)
    (hμ : IsNormalizedStationary β (fun s => max s 0) ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  sorry

/-- The hypotheses of `unifiedLog_measure_support_eq_zero` are satisfiable at
`d = 2`, `β = 1`, with the single neuron `a_0 = -basePoint 1` — non-degenerate
by `isNonDegenerate_pin` — and the Dirac mass it pins, which is stationary
because that neuron is inactive at its atom. -/
example :
    2 ≤ 2 ∧ (0 : ℝ) < 1 ∧
      IsNonDegenerate (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsNormalizedStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨le_rfl, one_pos, isNonDegenerate_pin, isNormalizedStationary_relu_pin 1 0 (basePoint 1)⟩

/-- **Proposition (prop: unified.log), second conclusion.**  In `d = 2` the
stationary measures of the normalized dynamics are purely atomic with finite
support.

Not proved here.

Source: arXiv:2601.21366v2, `prop: unified.log`. -/
theorem unifiedLog_isFinitelyAtomic (β : ℝ) (hβ : 0 < β) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationary β (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ := by
  sorry

/-- The hypotheses of `unifiedLog_isFinitelyAtomic` are satisfiable, by the
same witness. -/
example :
    (0 : ℝ) < 1 ∧
      IsNonDegenerate (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsNormalizedStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨one_pos, isNonDegenerate_pin, isNormalizedStationary_relu_pin 1 0 (basePoint 1)⟩

/-! ### `rem:unifiedlog-B` -/

/-- **Remark (rem:unifiedlog-B), first conclusion.**  The same for normalized
attention with a symmetric invertible matrix `B` in place of `β I_d`: the
first variation of `E_B` is strictly positive and real-analytic, which is all
the argument uses.

Not proved here.

Source: arXiv:2601.21366v2, `rem:unifiedlog-B`. -/
theorem unifiedLogMap_measure_support_eq_zero (d : ℕ) (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hsymm : B.IsSymmetric) (hB : Function.Bijective B)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere d)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  sorry

/-- The hypotheses of `unifiedLogMap_measure_support_eq_zero` are satisfiable
at `B = id`, where normalized stationarity is the case `β = 1` — by
`isNormalizedStationaryMap_id_iff` — so the pinned Dirac serves here too. -/
example :
    2 ≤ 2 ∧ (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
      Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
      IsNonDegenerate (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
        (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨le_rfl, (isSymmetric_bijective_id 2).1, (isSymmetric_bijective_id 2).2,
    isNonDegenerate_pin, isNormalizedStationaryMap_relu_pin 0 (basePoint 1)⟩

/-- **Remark (rem:unifiedlog-B), second conclusion.**  In `d = 2`, with a
symmetric invertible `B`, the stationary measures of the normalized dynamics
are purely atomic with finite support.

Not proved here.

Source: arXiv:2601.21366v2, `rem:unifiedlog-B`. -/
theorem unifiedLogMap_isFinitelyAtomic (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (hB : Function.Bijective B) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ := by
  sorry

/-- The hypotheses of `unifiedLogMap_isFinitelyAtomic` are satisfiable, by the
same witness at `B = id`. -/
example :
    (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
      Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
      IsNonDegenerate (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
        (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨(isSymmetric_bijective_id 2).1, (isSymmetric_bijective_id 2).2,
    isNonDegenerate_pin, isNormalizedStationaryMap_relu_pin 0 (basePoint 1)⟩

end Perceptron
end Transformer
