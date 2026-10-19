module

public import Transformer.Perceptron.Section5_PolynomialPrimitives
public import Transformer.Perceptron.Section5_LinearStationarity

/-!
# Actual first-variation fields of the original polynomial branches

ArXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, and §5.3,
`eq:global-Id`. The original activation on a branch is polynomial at
each original neuron input. Its actual primitive gives the corresponding
half-potential, retaining every original weight and coordinate.

The branch's scaled first variation combines that half-potential with
the original exponential interaction weight. Its actual derivative is
the ambient vector below. The projected vector is analytic globally and
equals the original temperature times the actual stationary field wherever
the supplied polynomials match the original activation values.
No global polynomial or analyticity property of the activation is assumed.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The actual half-potential of the original polynomial branch.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
The original relation `φ'=2σ` makes these primitives half-potentials;
their additive constants are absorbed in the branch's constant value. -/
noncomputable def polynomialBranchHalfPotential {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ) (x : EucSpace d) : ℝ :=
  ∑ j, ω j * polynomialPrimitive (P j) (inner (𝕜 := ℝ) (a j) x)

/-- The actual ambient vector of the original polynomial branch.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
Its derivative and actual projected-field identity are proved below. -/
noncomputable def polynomialBranchVector {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ) (x : EucSpace d) : EucSpace d :=
  ∑ j, (ω j * (P j).eval (inner (𝕜 := ℝ) (a j) x)) • a j

/-- The original scaled first variation on the supplied polynomial
branch. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
This copies the original linear-branch normalization, with the actual
polynomial primitives in place of the original squared coordinates. -/
noncomputable def polynomialAttentionPotential {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) : ℝ :=
  attentionWeightMap (β • LinearMap.id) μ x + β * polynomialBranchHalfPotential ω a P x

/-- The actual derivative vector of the original scaled branch variation.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
The original exponential integral, temperature and all neurons are retained. -/
noncomputable def polynomialAttentionVector {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) : EucSpace d :=
  ambientAttentionGradMap (β • LinearMap.id) μ x + β • polynomialBranchVector ω a P x

/-- The original polynomial branch vector represents its actual
half-potential derivative. Source: arXiv:2601.21366v2, §5.2,
`rem:piecewise-poly`, the polynomial first-variation branch. -/
theorem hasFDerivAt_polynomialBranchHalfPotential {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ) (x : EucSpace d) :
    HasFDerivAt (polynomialBranchHalfPotential ω a P)
      (innerSL ℝ (polynomialBranchVector ω a P x)) x := by
  have h := HasFDerivAt.fun_sum (u := Finset.univ) (fun j _ =>
    (hasFDerivAt_polynomialPrimitive_inner (P j) (a j) x).const_mul (ω j))
  apply h.congr_fderiv
  ext v
  simp only [sum_apply, smul_apply,
    innerSL_apply_apply, smul_eq_mul, polynomialBranchVector, sum_inner, real_inner_smul_left,
    mul_assoc]

/-- The actual scaled branch vector represents the original first
variation's actual derivative. Source: arXiv:2601.21366v2, §5.2,
`rem:piecewise-poly`. The proof follows `hasFDerivAt_linearAttentionPotential`
with the proved actual primitive derivative for every original neuron. -/
theorem hasFDerivAt_polynomialAttentionPotential {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    HasFDerivAt (polynomialAttentionPotential β ω a P μ)
      (innerSL ℝ (polynomialAttentionVector β ω a P μ x)) x := by
  have hW := hasFDerivAt_attentionWeightMap (β • LinearMap.id) μ x
  rw [attentionCovectorMap_eq_innerSL_ambientGradMap] at hW
  have h := hW.add ((hasFDerivAt_polynomialBranchHalfPotential ω a P x).const_mul β)
  apply h.congr_fderiv
  ext v
  change inner (𝕜 := ℝ) (ambientAttentionGradMap (β • LinearMap.id) μ x) v +
    β * inner (𝕜 := ℝ) (polynomialBranchVector ω a P x) v =
      inner (𝕜 := ℝ) (ambientAttentionGradMap (β • LinearMap.id) μ x +
        β • polynomialBranchVector ω a P x) v
  rw [inner_add_left, real_inner_smul_left]

/-- The original polynomial branch vector is analytic in the actual
ambient space. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
Every actual original polynomial evaluation and neuron is retained. -/
theorem analyticOnNhd_polynomialBranchVector {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ) :
    AnalyticOnNhd ℝ (polynomialBranchVector ω a P) univ := by
  intro x hx
  unfold polynomialBranchVector
  apply Finset.analyticAt_fun_sum
  intro j _
  exact (analyticAt_const.fun_mul (analyticOnNhd_polynomial_eval_inner (P j) (a j) x hx)).fun_smul
    analyticAt_const

/-- The actual projected polynomial branch field is analytic globally.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, and §5.3,
the analytic continuation leading to `eq:global-Id`.
The projection is its original bilinear formula, as in the linear branch. -/
theorem analyticOnNhd_projected_polynomialAttentionVector {d : ℕ} (β : ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ)
    (μ : Perspective.ProbSphere d) :
    AnalyticOnNhd ℝ (fun x => proj d x (polynomialAttentionVector β ω a P μ x)) univ := by
  intro x hx
  have hD : AnalyticAt ℝ (polynomialAttentionVector β ω a P μ) x :=
    (analyticOnNhd_ambientAttentionGradMap (β • LinearMap.id) μ x hx).fun_add
      (analyticAt_const.fun_smul (analyticOnNhd_polynomialBranchVector ω a P x hx))
  let L : EucSpace d →L[ℝ] (EucSpace d →L[ℝ] ℝ) := innerSL ℝ
  have hi : AnalyticAt ℝ (fun z => inner (𝕜 := ℝ) z (polynomialAttentionVector β ω a P μ z)) x :=
    (L.analyticAt_bilinear (x, polynomialAttentionVector β ω a P μ x)).comp
      (f := fun z => (z, polynomialAttentionVector β ω a P μ z)) (analyticAt_id.prod hD)
  exact hD.fun_sub (hi.fun_smul analyticAt_id)

/-- At actual matching neuron inputs, the original projected polynomial
branch is the original temperature times the actual stationary field.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, and §2.3,
`eq: steady.state`. Agreement is local at the actual point, including
breakpoints, rather than an assumed global polynomial activation. -/
theorem projected_polynomialAttentionVector_eq_energyGrad {d : ℕ} (β : ℝ)
    (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (P : Idx d → Polynomial ℝ)
    (μ : Perspective.ProbSphere d) (x : EucSpace d)
    (hx : ∀ j, σ (inner (𝕜 := ℝ) (a j) x) = (P j).eval (inner (𝕜 := ℝ) (a j) x)) :
    proj d x (polynomialAttentionVector β ω a P μ x) = β • energyGrad β σ ω a μ x := by
  have hp (y v : EucSpace d) : proj d x (y + β • v) = proj d x y + β • proj d x v := by
    simp only [proj, inner_add_right, real_inner_smul_right, add_smul, mul_smul, smul_sub]
    abel
  rw [polynomialAttentionVector, hp, ← attentionGradMap_eq_proj_ambientGradMap, attentionGradMap_smul_id]
  have hv : polynomialBranchVector ω a P x =
      ∑ j, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j := by
    unfold polynomialBranchVector
    simp_rw [← hx]
  rw [hv]
  change β • attentionGrad β μ x + β • drift σ ω a x =
    β • (attentionGrad β μ x + drift σ ω a x)
  rw [smul_add]

/-- All original pointwise matching hypotheses hold jointly for the
actual linear activation, actual polynomial `X` at every original neuron,
and the original circle axis. Source: arXiv:2601.21366v2, §5.2,
`rem:piecewise-poly`. No neuron is removed from the family. -/
example : ∀ j : Idx 2, (fun s : ℝ => s) (inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2)) =
    ((fun _ : Idx 2 => (Polynomial.X : Polynomial ℝ)) j).eval
      (inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2)) := by
  intro j
  simp

end Transformer.Perceptron
