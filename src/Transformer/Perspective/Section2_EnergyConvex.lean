/-
# §3.2 — The interaction energy is strictly convex

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §3.2: the analytic groundwork of
the minimiser half of `prop: existence.uniqueness.energy`
(`Perspective.Section2_EnergyMin`).

For `β > 0` and a continuous `V : 𝕊^{d-1} → ℝ`, write
`F(μ) = 𝖤_β[μ] + ∫ V dμ`.

* `F` is continuous for the weak topology on `𝒫(𝕊^{d-1})`, which is compact
  (Prokhorov), so `F` has a global minimiser
  (`exists_isMin_interactionEnergy_add`).
* `F` is strictly convex along midpoints.  The moments
  `m_I(μ) = ∫ x_{I 0} ⋯ x_{I (k-1)} dμ(x)` and `∫ V dμ` are linear in `μ`, and
  `𝖤_β` is a positive series of squares of moments
  (`hasSum_interactionEnergy`), so with `ν = (μ₀ + μ₁)/2`
  `(F(μ₀) + F(μ₁))/2 - F(ν) = (8β)⁻¹ Σ_k β^k/k! Σ_I (m_I(μ₀) - m_I(μ₁))²`
  (`hasSum_interactionEnergy_midProb`).  Two minimisers make the left side
  `≤ 0`, hence are equal (`eq_of_hasSum_sq_moment_sub_nonpos`), and the
  minimiser is unique (`existsUnique_isMin_interactionEnergy_add`).

The survey expands `e^{βt}` in Gegenbauer polynomials and cites Bilyk–Dai
(Prop. 2.2) for the uniqueness of the minimiser once the coefficients are
positive.  Here the expansion is in monomials, whose coefficients `β^k/k!` are
visibly positive, and Mathlib's Stone–Weierstrass theorem replaces the
completeness of spherical harmonics (`SphereMoments`).
-/

import Transformer.Perspective.PositiveDefinite
import Mathlib.MeasureTheory.Measure.Prokhorov
import Mathlib.MeasureTheory.Measure.FiniteMeasureProd

open scoped BigOperators Nat ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-! ### Continuity and existence -/

/-- `(x, x') ↦ e^{β⟨x, x'⟩}` is continuous on `𝕊^{d-1} × 𝕊^{d-1}`. -/
theorem continuous_expInner_prod (β : ℝ) :
    Continuous fun p : SSphere d × SSphere d =>
      exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)) :=
  (continuous_const.mul continuous_inner_sphere).rexp

/-- `𝖤_β[μ] = (2β)⁻¹ ∫ e^{β⟨x, x'⟩} d(μ ⊗ μ)(x, x')`. -/
theorem interactionEnergy_eq_integral_prod (β : ℝ) (μ : ProbSphere d) :
    interactionEnergy d β μ = (2 * β)⁻¹ * ∫ p : SSphere d × SSphere d,
      exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d))
        ∂((μ : Measure (SSphere d)).prod μ) := by
  rw [interactionEnergy, integral_prod _ (integrable_of_continuous_compact
    (continuous_expInner_prod β) _)]

/-- **`𝖤_β` is continuous** on `𝒫(𝕊^{d-1})`, for the topology of weak
convergence: `μ ↦ μ ⊗ μ` is continuous, and so is integration against a
continuous function on a compact space. -/
theorem continuous_interactionEnergy (β : ℝ) : Continuous (interactionEnergy d β) := by
  have h : Continuous fun μ : ProbSphere d => ∫ p : SSphere d × SSphere d,
      exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d))
        ∂((μ : Measure (SSphere d)).prod μ) :=
    (ProbabilityMeasure.continuous_integral_continuousMap
      (⟨_, continuous_expInner_prod β⟩ : C(SSphere d × SSphere d, ℝ))).comp
      (ProbabilityMeasure.continuous_prod.comp (continuous_id.prodMk continuous_id))
  rw [show interactionEnergy d β = _ from funext (interactionEnergy_eq_integral_prod β)]
  exact continuous_const.mul h

/-- **`𝖤_β + ∫ V` has a global minimiser** on `𝒫(𝕊^{d-1})`, for continuous
`V`: a continuous function on a compact space attains its minimum. -/
theorem exists_isMin_interactionEnergy_add (β : ℝ) (hd : 1 ≤ d) {V : SSphere d → ℝ}
    (hV : Continuous V) :
    ∃ μ₀ : ProbSphere d, ∀ μ : ProbSphere d,
      interactionEnergy d β μ₀ + ∫ x, V x ∂(μ₀ : Measure (SSphere d)) ≤
        interactionEnergy d β μ + ∫ x, V x ∂(μ : Measure (SSphere d)) := by
  have hF : Continuous fun μ : ProbSphere d =>
      interactionEnergy d β μ + ∫ x, V x ∂(μ : Measure (SSphere d)) :=
    (continuous_interactionEnergy β).add
      (ProbabilityMeasure.continuous_integral_continuousMap (⟨V, hV⟩ : C(SSphere d, ℝ)))
  obtain ⟨μ₀, -, hμ₀⟩ := isCompact_univ.exists_isMinOn
    ⟨diracProb d ⟨EuclideanSpace.single ⟨0, hd⟩ 1, by simp⟩, Set.mem_univ _⟩ hF.continuousOn
  exact ⟨μ₀, fun μ => isMinOn_iff.1 hμ₀ μ (Set.mem_univ μ)⟩

/-- The hypotheses of `exists_isMin_interactionEnergy_add` are satisfiable:
`d = 1` and `V = 0`. -/
example : 1 ≤ 1 ∧ Continuous (fun _ : SSphere 1 => (0 : ℝ)) := ⟨le_rfl, continuous_const⟩

/-! ### Strict convexity along midpoints -/

/-- The midpoint `(μ₀ + μ₁)/2` of two probability measures on the sphere. -/
noncomputable def midProb (μ₀ μ₁ : ProbSphere d) : ProbSphere d :=
  ⟨(2 : ℝ≥0∞)⁻¹ • ((μ₀ : Measure (SSphere d)) + μ₁),
    ⟨by simp [ENNReal.inv_two_add_inv_two]⟩⟩

@[simp] theorem coe_midProb (μ₀ μ₁ : ProbSphere d) :
    (midProb μ₀ μ₁ : Measure (SSphere d)) =
      (2 : ℝ≥0∞)⁻¹ • ((μ₀ : Measure (SSphere d)) + μ₁) := rfl

/-- `∫ f d(μ₀ + μ₁)/2 = (∫ f dμ₀ + ∫ f dμ₁)/2`, for continuous `f`. -/
theorem integral_midProb (μ₀ μ₁ : ProbSphere d) {f : SSphere d → ℝ} (hf : Continuous f) :
    ∫ x, f x ∂(midProb μ₀ μ₁ : Measure (SSphere d)) =
      (∫ x, f x ∂(μ₀ : Measure (SSphere d)) + ∫ x, f x ∂(μ₁ : Measure (SSphere d))) / 2 := by
  rw [coe_midProb, integral_smul_measure, integral_add_measure
    (integrable_of_continuous_compact hf _) (integrable_of_continuous_compact hf _)]
  simp [div_eq_inv_mul]

/-- The hypothesis of `integral_midProb` is satisfiable: `f = 0`. -/
example : Continuous (fun _ : SSphere 1 => (0 : ℝ)) := continuous_const

/-- The moments of the midpoint are the midpoints of the moments. -/
@[simp] theorem integral_mono_midProb (μ₀ μ₁ : ProbSphere d) {k : ℕ} (I : Fin k → Fin d) :
    ∫ x, mono I (x : EucSpace d) ∂(midProb μ₀ μ₁ : Measure (SSphere d)) =
      (∫ x, mono I (x : EucSpace d) ∂(μ₀ : Measure (SSphere d)) +
        ∫ x, mono I (x : EucSpace d) ∂(μ₁ : Measure (SSphere d))) / 2 :=
  integral_midProb μ₀ μ₁ ((continuous_mono I).comp continuous_subtype_val)

/-- **`𝖤_β` is strictly convex along midpoints:**
`(𝖤_β[μ₀] + 𝖤_β[μ₁])/2 - 𝖤_β[(μ₀+μ₁)/2] = Σ_k (8β)⁻¹ β^k/k! Σ_I (m_I(μ₀) - m_I(μ₁))²`,
termwise `a²/2 + b²/2 - ((a+b)/2)² = (a-b)²/4` in `hasSum_interactionEnergy`. -/
theorem hasSum_interactionEnergy_midProb (β : ℝ) (μ₀ μ₁ : ProbSphere d) :
    HasSum (fun k : ℕ => (8 * β)⁻¹ * (β ^ k / k ! * ∑ I : Fin k → Fin d,
        ((∫ x, mono I (x : EucSpace d) ∂(μ₀ : Measure (SSphere d))) -
          ∫ x, mono I (x : EucSpace d) ∂(μ₁ : Measure (SSphere d))) ^ 2))
      ((interactionEnergy d β μ₀ + interactionEnergy d β μ₁) / 2 -
        interactionEnergy d β (midProb μ₀ μ₁)) := by
  convert (((hasSum_interactionEnergy β μ₀).add (hasSum_interactionEnergy β μ₁)).div_const 2).sub
    (hasSum_interactionEnergy β (midProb μ₀ μ₁)) using 1
  funext k
  simp only [integral_mono_midProb, Finset.mul_sum, ← Finset.sum_add_distrib, Finset.sum_div,
    ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun I _ => by ring

/-! ### Uniqueness -/

/-- **Two global minimisers of `𝖤_β + ∫ V` are equal**, for `β > 0` and
continuous `V`: at the midpoint `ν` of two minimisers the energy is at least
theirs, so the moment series of `hasSum_interactionEnergy_midProb`, whose terms
are nonnegative, sums to `≤ 0`. -/
theorem eq_of_isMin_interactionEnergy_add (β : ℝ) (hβ : 0 < β) {V : SSphere d → ℝ}
    (hV : Continuous V) {μ₀ μ₁ : ProbSphere d}
    (h₀ : ∀ μ : ProbSphere d, interactionEnergy d β μ₀ + ∫ x, V x ∂(μ₀ : Measure (SSphere d)) ≤
      interactionEnergy d β μ + ∫ x, V x ∂(μ : Measure (SSphere d)))
    (h₁ : ∀ μ : ProbSphere d, interactionEnergy d β μ₁ + ∫ x, V x ∂(μ₁ : Measure (SSphere d)) ≤
      interactionEnergy d β μ + ∫ x, V x ∂(μ : Measure (SSphere d))) :
    μ₀ = μ₁ := by
  have hν₀ := h₀ (midProb μ₀ μ₁)
  have hν₁ := h₁ (midProb μ₀ μ₁)
  rw [integral_midProb μ₀ μ₁ hV] at hν₀ hν₁
  have h := (hasSum_interactionEnergy_midProb β μ₀ μ₁).mul_left (8 * β)
  simp only [← mul_assoc, mul_inv_cancel₀ (by positivity : (8 * β : ℝ) ≠ 0), one_mul] at h
  exact ProbabilityMeasure.toMeasure_injective (eq_of_hasSum_sq_moment_sub_nonpos β hβ _ _
    (mul_nonpos_of_nonneg_of_nonpos (by positivity) (by linarith)) h)

/-- The hypotheses of `eq_of_isMin_interactionEnergy_add` are satisfiable:
`β = 1`, `V = 0` and `μ₀ = μ₁` a minimiser of `𝖤_1`, which exists by
`exists_isMin_interactionEnergy_add`. -/
example : ∃ μ₀ μ₁ : ProbSphere 1, (0 : ℝ) < 1 ∧ Continuous (fun _ : SSphere 1 => (0 : ℝ)) ∧
    (∀ μ : ProbSphere 1, interactionEnergy 1 1 μ₀ + ∫ _, (0 : ℝ) ∂(μ₀ : Measure (SSphere 1)) ≤
      interactionEnergy 1 1 μ + ∫ _, (0 : ℝ) ∂(μ : Measure (SSphere 1))) ∧
    (∀ μ : ProbSphere 1, interactionEnergy 1 1 μ₁ + ∫ _, (0 : ℝ) ∂(μ₁ : Measure (SSphere 1)) ≤
      interactionEnergy 1 1 μ + ∫ _, (0 : ℝ) ∂(μ : Measure (SSphere 1))) := by
  obtain ⟨μ₀, h₀⟩ := exists_isMin_interactionEnergy_add 1 le_rfl
    (continuous_const : Continuous fun _ : SSphere 1 => (0 : ℝ))
  exact ⟨μ₀, μ₀, one_pos, continuous_const, h₀, h₀⟩

/-- **`𝖤_β + ∫ V` has exactly one global minimiser** on `𝒫(𝕊^{d-1})`, for
`β > 0`, `d ≥ 1` and continuous `V`.

Source: arXiv:2312.10794v5, §3.2, `prop: existence.uniqueness.energy`, the
case `V = 0`. -/
theorem existsUnique_isMin_interactionEnergy_add (β : ℝ) (hβ : 0 < β) (hd : 1 ≤ d)
    {V : SSphere d → ℝ} (hV : Continuous V) :
    ∃! μ₀ : ProbSphere d, ∀ μ : ProbSphere d,
      interactionEnergy d β μ₀ + ∫ x, V x ∂(μ₀ : Measure (SSphere d)) ≤
        interactionEnergy d β μ + ∫ x, V x ∂(μ : Measure (SSphere d)) := by
  obtain ⟨μ₀, h₀⟩ := exists_isMin_interactionEnergy_add β hd hV
  exact ⟨μ₀, h₀, fun μ₁ h₁ => eq_of_isMin_interactionEnergy_add β hβ hV h₁ h₀⟩

/-- The hypotheses of `existsUnique_isMin_interactionEnergy_add` are
satisfiable: `β = 1`, `d = 1` and `V = 0`. -/
example : (0 : ℝ) < 1 ∧ 1 ≤ 1 ∧ Continuous (fun _ : SSphere 1 => (0 : ℝ)) :=
  ⟨one_pos, le_rfl, continuous_const⟩

end Perspective
end Transformer
