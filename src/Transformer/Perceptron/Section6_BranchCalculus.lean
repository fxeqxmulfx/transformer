/-
# Global analytic quadratic branches in the circle angle

ArXiv:2601.21366v2, §6, proof of `prop: unified.log` and
`eq:log-identity-global`. The branch polynomial is defined at every
angle, independently of whether that angle has the branch's pattern.
Its actual derivative and the original interaction weight are analytic.

The source differentiates the logarithm of the positive weight. Here
the analytic branch force is multiplied by that positive weight:
`F' + F P'`, where `P` is half the branch potential. This has exactly
the same zeros, and avoids choosing a logarithm in the analytic proof.
The quadratic branch and its force have period `π`, because the actual
circle point and its velocity both change sign at the antipode.
-/

import Transformer.Perceptron.Section6_ActivationPatterns
import Transformer.Perceptron.Section6_AnalyticWeight
import Transformer.Perceptron.Transform
import Mathlib.Analysis.Calculus.Deriv.Shift

open Real Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The actual circle point changes sign at angle `θ+π`.
Source: arXiv:2601.21366v2, §6, the antipodal component argument. -/
theorem coe_circlePoint_add_pi (θ : ℝ) :
    (circlePoint (θ + π) : EucSpace 2) = -(circlePoint θ : EucSpace 2) := by
  simp only [circlePoint, greatCircle, Real.cos_add_pi, Real.sin_add_pi, neg_smul]
  abel

/-- The actual sphere points at opposite angles are antipodes.
Source: arXiv:2601.21366v2, §6, `eq:log-identity-global`. -/
theorem circlePoint_add_pi (θ : ℝ) :
    circlePoint (θ + π) = antipodeMap 2 (circlePoint θ) :=
  Subtype.ext (coe_circlePoint_add_pi θ)

/-- The actual velocity changes sign at the antipode as well.
Source: arXiv:2601.21366v2, §6, the two complementary branch gradients. -/
theorem circleVel_add_pi (θ : ℝ) : circleVel (θ + π) = -circleVel θ := by
  simp only [circleVel, Real.cos_add_pi, Real.sin_add_pi, neg_neg, neg_smul]
  abel

/-- Inner products with the actual circle point are globally analytic.
Source: arXiv:2601.21366v2, §6, the quadratic branch polynomials. -/
theorem analyticOnNhd_inner_circlePoint (v : EucSpace 2) :
    AnalyticOnNhd ℝ (fun θ => inner (𝕜 := ℝ) v (circlePoint θ : EucSpace 2)) univ := by
  intro θ _
  exact ((innerSL ℝ v).analyticAt _).comp
    (analyticOnNhd_greatCircle ((basePoint 1 : SSphere 2) : EucSpace 2) secondAxis θ (mem_univ θ))

/-- Inner products with the actual circle velocity are globally analytic.
Source: arXiv:2601.21366v2, §6, the derivatives of the quadratic branches. -/
theorem analyticOnNhd_inner_circleVel (v : EucSpace 2) :
    AnalyticOnNhd ℝ (fun θ => inner (𝕜 := ℝ) v (circleVel θ)) univ := by
  intro θ _
  exact ((innerSL ℝ v).analyticAt _).comp
    ((Real.analyticAt_sin.neg.smul analyticAt_const).add
      (Real.analyticAt_cos.smul analyticAt_const))

/-- The actual half quadratic potential is analytic at every angle.
Source: arXiv:2601.21366v2, §6, the globally continued branch of `H_log`. -/
theorem analyticOnNhd_circleBranchHalfPotential (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2)) :
    AnalyticOnNhd ℝ (circleBranchHalfPotential ω a J) univ := by
  intro θ _
  exact analyticAt_const.mul (J.analyticAt_fun_sum (fun j _ =>
    analyticAt_const.mul ((analyticOnNhd_inner_circlePoint (a j) θ (mem_univ θ)).pow 2)))

/-- Its actual branch force is globally analytic.
Source: arXiv:2601.21366v2, §6, the globally continued branch gradient. -/
theorem analyticOnNhd_circleBranchForce (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2)) :
    AnalyticOnNhd ℝ (circleBranchForce ω a J) univ := by
  intro θ _
  exact J.analyticAt_fun_sum (fun j _ => analyticAt_const.mul
    ((analyticOnNhd_inner_circlePoint (a j) θ (mem_univ θ)).mul
      (analyticOnNhd_inner_circleVel (a j) θ (mem_univ θ))))

/-- The total angular derivative agrees with the actual branch force.
Source: arXiv:2601.21366v2, §6, the half potential in `H_log`.
This also identifies the derivative used in global continuation. -/
theorem deriv_circleBranchHalfPotential (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2)) (θ : ℝ) :
    deriv (circleBranchHalfPotential ω a J) θ = circleBranchForce ω a J θ :=
  (hasDerivAt_circleBranchHalfPotential ω a J θ).deriv

/-- The half quadratic branch is even on the actual sphere.
Source: arXiv:2601.21366v2, §6, evenness in `eq:log-identity-global`. -/
theorem periodic_circleBranchHalfPotential (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2)) :
    Function.Periodic (circleBranchHalfPotential ω a J) π := by
  intro θ
  simp only [circleBranchHalfPotential, coe_circlePoint_add_pi, inner_neg_right, neg_sq]

/-- The actual polynomial branch force also has period `π`.
Source: arXiv:2601.21366v2, §6, the antipodal branch-gradient comparison. -/
theorem periodic_circleBranchForce (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2)) :
    Function.Periodic (circleBranchForce ω a J) π := by
  intro θ
  simp only [circleBranchForce, coe_circlePoint_add_pi, circleVel_add_pi,
    inner_neg_right, neg_mul_neg]

/-- The globally defined branch gradient multiplied by the actual positive
weight. Source: arXiv:2601.21366v2, §6, `g_log` and its global continuation.
Multiplication by `F>0` gives `F'+F P'` with the same zero set. -/
noncomputable def circleNormalizedBranchForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2)) (θ : ℝ) : ℝ :=
  deriv (circleAngularWeightMap B μ) θ +
    circleAngularWeightMap B μ θ * circleBranchForce ω a J θ

/-- The actual globally continued normalized branch force is analytic.
Source: arXiv:2601.21366v2, §6, the identity theorem for the branch gradient. -/
theorem analyticOnNhd_circleNormalizedBranchForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2)) :
    AnalyticOnNhd ℝ (circleNormalizedBranchForce B ω a μ J) univ := by
  intro θ hθ
  exact (analyticOnNhd_deriv_circleAngularWeightMap B μ θ hθ).add
    ((analyticOnNhd_circleAngularWeightMap B μ θ hθ).mul
      (analyticOnNhd_circleBranchForce ω a J θ hθ))

/-- Differentiation preserves the period on the real line, including at
non-differentiable points under Mathlib's total derivative convention.
Source: arXiv:2601.21366v2, §6, the even weight's antipodal gradients.
Proof adapted from this repository's `Transformer.Kinetic.periodic_deriv`. -/
theorem periodic_deriv_real {f : ℝ → ℝ} {c : ℝ} (h : Function.Periodic f c) :
    Function.Periodic (deriv f) c := by
  intro θ
  have heq : (fun r => f (r + c)) = f := funext h
  have hd := deriv_comp_add_const f c θ
  rw [heq] at hd
  exact hd.symm

/-- The periodicity hypothesis is realized by every actual quadratic
branch; the same branch has the actual derivative stated above.
Source: arXiv:2601.21366v2, §6, the even quadratic branch in
`eq:log-identity-global`. The neurons need no activation assumption. -/
example (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2)) :
    Function.Periodic (circleBranchHalfPotential ω a J) π ∧
      (∀ θ, HasDerivAt (circleBranchHalfPotential ω a J)
        (circleBranchForce ω a J θ) θ) :=
  ⟨periodic_circleBranchHalfPotential ω a J,
    hasDerivAt_circleBranchHalfPotential ω a J⟩

end Transformer.Perceptron
