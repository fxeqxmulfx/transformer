/-
# §10 — Measure-to-measure approximation

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §10.

The survey cites, without stating it, "the first universal approximation results for Transformers,
viewed as measure-to-measure maps".  The repository had written it as a theorem: the time-`T` flow
maps of the mean-field dynamics with time-dependent `Q, K, V` approximate every continuous
self-map of `𝒫(𝕊^{d-1})`.  That is false, and it is refuted here at `β = 0`, where the reason is
visible without any well-posedness theory: attention is uniform, a symmetric measure has barycenter
`0`, and the field then vanishes for every `Q, K, V`, so the measure does not move.  For `β > 0`
the claim fails for the same reason that a flow acts by push-forward: a Dirac mass stays a Dirac
mass, and a constant map onto the uniform measure cannot be approximated.  That argument needs the
solution theory the survey only quotes; the refutation below does not.
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-- The mean-field vector field of `eq: SA.QKV`, i.e. `eq: vfSd` with general
time-dependent `Q, K, V`:

  `𝒳_t[μ](x) = Proj_x ( (∫ exp(β ⟨Q_t x, K_t y⟩) dμ(y))⁻¹
                          ∫ exp(β ⟨Q_t x, K_t y⟩) V_t y dμ(y) )`.

`vectorField` is this one at `Q = K = V = I_d`. -/
noncomputable def vectorFieldQKV
    (β : ℝ) (Q K V : TimeParam d) (t : ℝ) (μ : ProbSphere d) (x : EucSpace d) :
    EucSpace d :=
  proj d x
    ((∫ y, Real.exp (β * inner (𝕜 := ℝ) (Q t x) (K t (y : EucSpace d)))
        ∂(μ : Measure (SSphere d)))⁻¹ •
      ∫ y, Real.exp (β * inner (𝕜 := ℝ) (Q t x) (K t (y : EucSpace d)))
            • V t (y : EucSpace d) ∂(μ : Measure (SSphere d)))

/-- **Measure-to-measure universal approximation, as formerly stated, is false.**

The claim was: for every `β`, every continuous `Φ : 𝒫(𝕊^{d-1}) → 𝒫(𝕊^{d-1})` and every `ε > 0`
there are `T > 0` and time-dependent `Q, K, V` such that every solution `m` of the continuity
equation driven by `vectorFieldQKV` satisfies, for all bounded `1`-Lipschitz `φ` (the
bounded-Lipschitz distance), `|∫ φ d m(T) - ∫ φ d Φ(m(0))| < ε`.

At `β = 0` the weights are constant, the attention average is `V_t` applied to the barycenter of
the measure, and the barycenter of `μ₀ = (δ_{x₀} + δ_{-x₀})/2` is `0`.  So the constant curve
`m ≡ μ₀` solves the equation for **every** choice of `T, Q, K, V`, and `Φ ≡ δ_{x₀}`, which is
continuous, must be matched at `μ₀` by `μ₀` itself; with `φ(x) = sin ⟨x₀, x⟩` the two sides differ
by `sin 1 > 1/2`.  This holds in every dimension `d + 1 ≥ 1`.

The survey states no such theorem: §10 only refers to work on universal approximation "viewed as
measure-to-measure maps".  A flow map acts on measures by push-forward, so it cannot send a Dirac
mass to a diffuse measure, which an arbitrary continuous `Φ` may do.

Source: arXiv:2312.10794v5, §10, the paragraph citing Agrachev–Sarychev and Furuya et al. -/
theorem not_universal_approximation_measure :
    ¬ ∀ (β : ℝ) (Φ : ProbSphere (d + 1) → ProbSphere (d + 1)), Continuous Φ →
      ∀ ε : ℝ, 0 < ε →
      ∃ (T : ℝ) (Q K V : TimeParam (d + 1)), 0 < T ∧
      ∀ (μ : ProbSphere (d + 1)) (m : ℝ → ProbSphere (d + 1)),
        m 0 = μ →
        auxCE (d + 1) m (fun t x => vectorFieldQKV (d + 1) β Q K V t (m t) x) →
        ∀ φ : EucSpace (d + 1) → ℝ, LipschitzWith 1 φ → (∀ x : EucSpace (d + 1), |φ x| ≤ 1) →
          |(∫ x, φ (x : EucSpace (d + 1)) ∂(m T : Measure (SSphere (d + 1))))
              - ∫ x, φ (x : EucSpace (d + 1))
                  ∂((Φ μ : ProbSphere (d + 1)) : Measure (SSphere (d + 1)))| < ε := by
  intro h
  set x₀ : SSphere (d + 1) := basePoint d with hx₀
  have hx₀n : ‖(x₀ : EucSpace (d + 1))‖ = 1 := mem_sphere_zero_iff_norm.mp x₀.2
  set x₁ : SSphere (d + 1) :=
    ⟨-(x₀ : EucSpace (d + 1)), mem_sphere_zero_iff_norm.mpr (by rw [norm_neg, hx₀n])⟩ with hx₁
  set ν : Measure (SSphere (d + 1)) :=
    (2⁻¹ : ℝ≥0∞) • Measure.dirac x₀ + (2⁻¹ : ℝ≥0∞) • Measure.dirac x₁ with hν
  have hνprob : IsProbabilityMeasure ν :=
    ⟨by simp [hν, ENNReal.inv_two_add_inv_two]⟩
  set μ₀ : ProbSphere (d + 1) := ⟨ν, hνprob⟩ with hμ₀
  -- integrals against the symmetric mixture
  have hint : ∀ (E : Type) [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
      (g : SSphere (d + 1) → E), ∫ y, g y ∂ν = (1 / 2 : ℝ) • g x₀ + (1 / 2 : ℝ) • g x₁ := by
    intro E _ _ _ g
    have hi : ∀ a, MeasureTheory.Integrable g (Measure.dirac a) := fun a =>
      integrable_dirac enorm_lt_top
    rw [hν, integral_add_measure ((hi x₀).smul_measure (by simp)) ((hi x₁).smul_measure (by simp)),
      integral_smul_measure, integral_smul_measure, integral_dirac, integral_dirac]
    simp [ENNReal.toReal_inv]
  -- the field vanishes at `β = 0` along `μ₀`, whatever `Q, K, V`
  have hz : ∀ (Q K V : TimeParam (d + 1)) (t : ℝ) (x : EucSpace (d + 1)),
      vectorFieldQKV (d + 1) 0 Q K V t μ₀ x = 0 := by
    intro Q K V t x
    have hone : ∫ y, (1 : ℝ) ∂ν = 1 := by simp
    have hbar : ∫ y, V t (y : EucSpace (d + 1)) ∂ν = 0 := by
      rw [hint _ (fun y => V t (y : EucSpace (d + 1)))]
      simp [hx₁, map_neg]
    show proj (d + 1) x ((∫ y, Real.exp (0 * inner (𝕜 := ℝ) (Q t x) (K t (y : EucSpace (d + 1))))
        ∂ν)⁻¹ • ∫ y, Real.exp (0 * inner (𝕜 := ℝ) (Q t x) (K t (y : EucSpace (d + 1))))
          • V t (y : EucSpace (d + 1)) ∂ν) = 0
    simp only [zero_mul, Real.exp_zero, one_smul, hone, hbar, inv_one, smul_zero]
    simp [proj]
  have hsol : ∀ Q K V : TimeParam (d + 1), auxCE (d + 1) (fun _ => μ₀)
      (fun t x => vectorFieldQKV (d + 1) 0 Q K V t ((fun _ : ℝ => μ₀) t) x) := by
    intro Q K V φ _ t
    have h0 : ∫ x : SSphere (d + 1), inner (𝕜 := ℝ) (gradient φ (x : EucSpace (d + 1)))
        (vectorFieldQKV (d + 1) 0 Q K V t μ₀ x) ∂(μ₀ : Measure (SSphere (d + 1))) = 0 := by
      simp [hz]
    show HasDerivAt (fun _ : ℝ => ∫ x, φ (x : EucSpace (d + 1)) ∂(μ₀ : Measure (SSphere (d + 1))))
      (∫ x : SSphere (d + 1), inner (𝕜 := ℝ) (gradient φ (x : EucSpace (d + 1)))
        (vectorFieldQKV (d + 1) 0 Q K V t μ₀ x) ∂(μ₀ : Measure (SSphere (d + 1)))) t
    rw [h0]
    exact hasDerivAt_const t _
  obtain ⟨T, Q, K, V, -, hall⟩ :=
    h 0 (fun _ => diracProb (d + 1) x₀) continuous_const (1 / 2) (by norm_num)
  -- the test function `sin ⟨x₀, ·⟩`
  obtain ⟨φ, hφ⟩ : ∃ φ : EucSpace (d + 1) → ℝ,
      φ = fun x => Real.sin (inner (𝕜 := ℝ) (x₀ : EucSpace (d + 1)) x) := ⟨_, rfl⟩
  have hinner : LipschitzWith 1
      (fun x : EucSpace (d + 1) => inner (𝕜 := ℝ) (x₀ : EucSpace (d + 1)) x) := by
    refine LipschitzWith.of_dist_le_mul fun x y => ?_
    rw [Real.dist_eq, dist_eq_norm, NNReal.coe_one, one_mul, ← inner_sub_right]
    calc |inner (𝕜 := ℝ) (x₀ : EucSpace (d + 1)) (x - y)| ≤ ‖(x₀ : EucSpace (d + 1))‖ * ‖x - y‖ :=
          abs_real_inner_le_norm _ _
      _ = ‖x - y‖ := by rw [hx₀n, one_mul]
  have hφL : LipschitzWith 1 φ := by
    rw [hφ]
    simpa [Function.comp_def] using Real.lipschitzWith_sin.comp hinner
  have hφb : ∀ x, |φ x| ≤ 1 := by
    intro x
    rw [hφ]
    exact Real.abs_sin_le_one _
  have key := hall μ₀ (fun _ => μ₀) rfl (hsol Q K V) φ hφL hφb
  have hφ0 : φ (x₀ : EucSpace (d + 1)) = Real.sin 1 := by
    simp only [hφ, real_inner_self_eq_norm_mul_norm, hx₀n, mul_one]
  have hφ1 : φ (x₁ : EucSpace (d + 1)) = -Real.sin 1 := by
    simp only [hφ, hx₁, inner_neg_right, real_inner_self_eq_norm_mul_norm, hx₀n, mul_one,
      Real.sin_neg]
  have e1 : ∫ x, φ (x : EucSpace (d + 1)) ∂(μ₀ : Measure (SSphere (d + 1))) = 0 := by
    show ∫ x, φ (x : EucSpace (d + 1)) ∂ν = 0
    rw [hint ℝ (fun x => φ (x : EucSpace (d + 1))), smul_eq_mul, smul_eq_mul, hφ0, hφ1]
    ring
  have e2 : ∫ x, φ (x : EucSpace (d + 1))
      ∂((diracProb (d + 1) x₀ : ProbSphere (d + 1)) : Measure (SSphere (d + 1))) = Real.sin 1 := by
    show ∫ x, φ (x : EucSpace (d + 1)) ∂(Measure.dirac x₀) = _
    rw [integral_dirac, hφ0]
  simp only [e1, e2, zero_sub, abs_neg] at key
  have hsin : 1 - 1 ^ 3 / 6 < Real.sin 1 := Real.sin_gt_sub_cube one_pos
  rw [abs_of_pos (by linarith : 0 < Real.sin 1)] at key
  linarith

end Perspective
end Transformer
