/-
# Actual perceptron parameters with a transverse barycenter variation

Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2.
For two spherical probability laws in an open hemisphere and a scale
`γ≠1`, a single actual ReLU neuron distinguishes their transverse
barycenter velocities. The attention parameters are exactly zero.

The tangent projection contributes the positive weight
`1-⟨x,p⟩²`, where `p` is a unit vector perpendicular to the hemisphere
axis. The preceding weighted ridge theorem supplies an affine test;
the matrices below realize that test in the original `fullVF`.

This proves the instantaneous parameter-selection step. Constructing
weak solutions and turning the transverse variation into a uniform
finite-time separation are further steps in the original theorem.
-/

import Transformer.Interpolation.Section3_ReLURidges

open Real MeasureTheory Filter Set
namespace Transformer.Interpolation

/-- The spherical projection has strictly positive transverse weight
inside a hemisphere whose axis is perpendicular to the unit test vector.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem projection_weight_pos {d : ℕ} (x : SSphere d) (p m : EucSpace d)
    (hp : ‖p‖ = 1) (hpm : inner ℝ p m = 0) (hxm : 0 < inner ℝ (x : EucSpace d) m) :
    0 < 1 - inner ℝ (x : EucSpace d) p ^ 2 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hi := norm_inner_le_norm (𝕜 := ℝ) (x : EucSpace d) p
  rw [Real.norm_eq_abs, hx, hp, mul_one] at hi
  have hs : inner ℝ (x : EucSpace d) p ^ 2 ≤ 1 := by
    have hb := abs_le.mp hi
    nlinarith [mul_nonneg (sub_nonneg.mpr hb.2) (show 0 ≤ 1 + inner ℝ (x : EucSpace d) p by linarith)]
  have hne : inner ℝ (x : EucSpace d) p ^ 2 ≠ 1 := by
    intro heq
    rcases sq_eq_one_iff.mp heq with h | h
    · have hx' := (inner_eq_one_iff_of_norm_eq_one hx hp).mp h
      rw [hx', hpm] at hxm
      exact (lt_irrefl 0) hxm
    · have hx' := (inner_eq_neg_one_iff_of_norm_eq_one hx hp).mp h
      rw [hx', inner_neg_left, hpm, neg_zero] at hxm
      exact (lt_irrefl 0) hxm
  exact sub_pos.mpr (lt_of_le_of_ne hs hne)

/-- A ridge separates the transverse flux integrals at a scale `γ≠1`.
The weighted measures cannot agree, because recovering their original
masses would force `γ=1`. Auxiliary for arXiv:2411.04551v3, §3.2. -/
theorem exists_transverse_relu_variation {d : ℕ} (μ ν : Perspective.ProbSphere d)
    (γ : ℝ) (hγ : 0 ≤ γ) (hγne : γ ≠ 1) (p m : EucSpace d)
    (hp : ‖p‖ = 1) (hpm : inner ℝ p m = 0)
    (hμ : ∀ᵐ (x : SSphere d) ∂(μ : Measure (SSphere d)), 0 < inner ℝ (x : EucSpace d) m)
    (hν : ∀ᵐ (x : SSphere d) ∂(ν : Measure (SSphere d)), 0 < inner ℝ (x : EucSpace d) m) :
    ∃ (u : EucSpace d) (b : ℝ),
      (∫ x, (1 - inner ℝ (x : EucSpace d) p ^ 2) *
        max (b - inner ℝ (x : EucSpace d) u) 0 ∂(μ : Measure (SSphere d))) ≠
      γ * ∫ x, (1 - inner ℝ (x : EucSpace d) p ^ 2) *
        max (b - inner ℝ (x : EucSpace d) u) 0 ∂(ν : Measure (SSphere d)) := by
  let νs : Measure (SSphere d) := ENNReal.ofReal γ • (ν : Measure (SSphere d))
  have : IsFiniteMeasure νs := ⟨by simp [νs]⟩
  have hn : (μ : Measure (SSphere d)) ≠ νs := by
    intro heq
    have h := congrArg (fun ξ : Measure (SSphere d) => ξ Set.univ) heq
    have hg : ENNReal.ofReal γ = 1 := by simpa [νs] using h.symm
    apply hγne
    have hr := congrArg ENNReal.toReal hg
    simpa [ENNReal.toReal_ofReal hγ] using hr
  have hposμ : ∀ᵐ (x : SSphere d) ∂(μ : Measure (SSphere d)), 0 < 1 - inner ℝ (x : EucSpace d) p ^ 2 :=
    hμ.mono fun x hx => projection_weight_pos x p m hp hpm hx
  have hposν : ∀ᵐ (x : SSphere d) ∂(ν : Measure (SSphere d)), 0 < 1 - inner ℝ (x : EucSpace d) p ^ 2 :=
    hν.mono fun x hx => projection_weight_pos x p m hp hpm hx
  obtain ⟨u, b, hub⟩ := exists_positive_weighted_sphere_relu_ridge
    (μ : Measure (SSphere d)) νs (fun x => 1 - inner ℝ (x : EucSpace d) p ^ 2)
    (by fun_prop) hposμ (Measure.ae_smul_measure hposν _) hn
  refine ⟨u, b, ?_⟩
  simpa only [νs, integral_smul_measure, ENNReal.toReal_ofReal hγ, smul_eq_mul] using hub


/-- One actual neuron with output direction `p`, activation
`max(b-⟨x,u⟩,0)`, and attention switched off.
Source: arXiv:2411.04551v3, §1, `eq: vf`, and §3.2, `lem: perturbation`. -/
noncomputable def reluRidgeParams {d : ℕ} (i : Fin d) (p u : EucSpace d) (b : ℝ) : Params d :=
  { V := 0
    B := 0
    W := (EuclideanSpace.proj i).smulRight p
    U := (-innerSL ℝ u).smulRight (EuclideanSpace.single i 1)
    b := b • EuclideanSpace.single i 1 }

/-- The original network field at the explicit matrices equals the
selected ridge times the actual spherical projection.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem fullVF_reluRidgeParams {d : ℕ} (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (μ : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d (fun _ => reluRidgeParams i p u b) μ t x =
      max (b - inner ℝ x u) 0 • proj d x p := by
  have hrelu : (EuclideanSpace.equiv (Fin d) ℝ).symm
      (fun k => max ((EuclideanSpace.equiv (Fin d) ℝ
        ((reluRidgeParams i p u b).U x + (reluRidgeParams i p u b).b)) k) 0) =
      max (b - inner ℝ x u) 0 • EuclideanSpace.single i 1 := by
    ext k
    by_cases hk : k = i
    · subst k
      simp [reluRidgeParams, real_inner_comm x u]
      congr 1
      ring
    · simp [reluRidgeParams, hk]
  have hV : (reluRidgeParams i p u b).V = 0 := rfl
  simp only [fullVF, hV, zero_apply, zero_add]
  change proj d x ((reluRidgeParams i p u b).W _) = _
  rw [hrelu]
  simp only [map_smul]
  have hW : (reluRidgeParams i p u b).W (EuclideanSpace.single i 1) = p := by
    simp [reluRidgeParams]
  rw [hW, proj, proj, inner_smul_right, smul_sub, smul_smul]

/-- Testing the original field against its unit output direction gives
the positive projection weight used in ridge selection.
Source: arXiv:2411.04551v3, §1, `eq: vf`, and §3.2, `lem: perturbation`. -/
theorem inner_fullVF_reluRidgeParams {d : ℕ} (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (hp : ‖p‖ = 1) (μ : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    inner ℝ p (fullVF d (fun _ => reluRidgeParams i p u b) μ t x) =
      (1 - inner ℝ x p ^ 2) * max (b - inner ℝ x u) 0 := by
  rw [fullVF_reluRidgeParams, real_inner_smul_right, proj, inner_sub_right,
    real_inner_self_eq_norm_sq, hp, one_pow, real_inner_smul_right, real_inner_comm p x]
  ring


/-- Produce actual parameters with zero attention and different scaled
transverse barycenter velocities for the two original probability laws.
No flow or separation conclusion is assumed. Auxiliary instantaneous step
for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem exists_transverse_perceptron_params {d : ℕ} (i : Fin d) (μ ν : Perspective.ProbSphere d)
    (γ : ℝ) (hγ : 0 ≤ γ) (hγne : γ ≠ 1) (p m : EucSpace d)
    (hp : ‖p‖ = 1) (hpm : inner ℝ p m = 0)
    (hμ : ∀ᵐ (x : SSphere d) ∂(μ : Measure (SSphere d)), 0 < inner ℝ (x : EucSpace d) m)
    (hν : ∀ᵐ (x : SSphere d) ∂(ν : Measure (SSphere d)), 0 < inner ℝ (x : EucSpace d) m) :
    ∃ θ : Params d, θ.V = 0 ∧ θ.B = 0 ∧
      (∫ x : SSphere d, inner ℝ p (fullVF d (fun _ => θ) μ 0 (x : EucSpace d))
        ∂(μ : Measure (SSphere d))) ≠
      γ * ∫ x : SSphere d, inner ℝ p (fullVF d (fun _ => θ) ν 0 (x : EucSpace d))
        ∂(ν : Measure (SSphere d)) := by
  obtain ⟨u, b, hub⟩ := exists_transverse_relu_variation μ ν γ hγ hγne p m hp hpm hμ hν
  refine ⟨reluRidgeParams i p u b, rfl, rfl, ?_⟩
  simp_rw [inner_fullVF_reluRidgeParams i p u b hp]
  exact hub

/-- A genuine hemisphere atom, transverse unit vector, and scale in `(0,1)`
jointly realize all geometric and numerical inputs of parameter selection.
The same probability law may be used on both sides because `γ≠1` already
distinguishes their scaled masses in this auxiliary velocity statement. -/
example :
    ‖(EuclideanSpace.single (1 : Fin 2) (1 : ℝ))‖ = 1 ∧
      inner ℝ (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) (basePoint 1 : EucSpace 2) = 0 ∧
      (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≠ 1 ∧
      ∀ᵐ (x : SSphere 2) ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)),
        0 < inner ℝ (x : EucSpace 2) (basePoint 1 : EucSpace 2) := by
  refine ⟨by simp, by simp [basePoint, EuclideanSpace.inner_single_left],
    by norm_num, by norm_num, ?_⟩
  change ∀ᵐ (x : SSphere 2) ∂Measure.dirac (basePoint 1),
    0 < inner ℝ (x : EucSpace 2) (basePoint 1 : EucSpace 2)
  simp [ae_dirac_eq, basePoint]

end Transformer.Interpolation
