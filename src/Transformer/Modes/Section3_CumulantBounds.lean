/-
# The number of modes of a Gaussian KDE — third cumulants

The order-three case of `lem:eta`, arXiv:2412.09080v3, §3.1.  For the
standardized law the cumulant is a centred mixed moment, so its absolute
value is bounded by the third Euclidean moment with constant one.  The
bounded KDE summands also have the exponential moments needed to identify
the derivatives of the cumulant generating function with those moments.
-/

import Transformer.Modes.Section3_CumulantMoment
import Transformer.Modes.Section3_Standardized

open Real MeasureTheory Filter ProbabilityTheory

namespace Transformer.Modes

/-- Every continuous function of the bounded standardized KDE summand is
integrable.  Source: arXiv:2412.09080v3, §2.2, `eq:Yi`, and §3.1. -/
theorem integrable_lawY_of_continuous {β : ℝ} (hβ : 0 < β) (t : ℝ)
    {f : ℝ × ℝ → ℝ} (hf : Continuous f) : Integrable f (lawY β t) := by
  let L : ℝ × ℝ → ℝ := fun p => f (whiten (sigmaFst β t) (sigmaCov β t)
    (sigmaSnd β t) (p.1 - meanG β t, p.2 - meanG' β t))
  have hL : Continuous L := by dsimp [L, whiten]; fun_prop
  obtain ⟨C, hC⟩ := (isCompact_Icc
    (a := ((-(1 + 2 / β), -2) : ℝ × ℝ)) (b := (1 + 2 / β, 2))
    ).exists_bound_of_continuousOn hL.continuousOn
  change Integrable f ((gaussianReal 0 1).map (singleY β t))
  rw [integrable_map_measure hf.aestronglyMeasurable
    (measurable_singleY β t).aemeasurable]
  refine Integrable.of_bound
    (hf.measurable.comp (measurable_singleY β t)).aestronglyMeasurable C ?_
  refine ae_of_all _ fun x => ?_
  have h1 := abs_le.1 (abs_bigG_le hβ t x)
  have h2 := abs_le.1 (abs_bigG'_le hβ t x)
  exact hC (bigG β t x, bigG' β t x)
    ⟨⟨h1.1, h2.1⟩, ⟨h1.2, h2.2⟩⟩

example : (0 : ℝ) < 1 ∧ Continuous (fun z : ℝ × ℝ => z.1 ^ 3) :=
  ⟨one_pos, by fun_prop⟩

/-- All Euclidean moments of the standardized KDE summand are finite.
This ensures that `η_s` is the expectation in the source, for every
positive bandwidth before taking a regime limit.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`, and its proof in §5.3. -/
theorem integrable_eucl_pow_lawY {β : ℝ} (hβ : 0 < β) (t : ℝ) (s : ℕ) :
    Integrable (fun z => eucl z ^ s) (lawY β t) :=
  integrable_lawY_of_continuous hβ t (by unfold eucl; fun_prop)

example : (0 : ℝ) < 1 := one_pos

/-- The single-sample law has exponential moments, as required by the
cumulant identity after `eq:psi`.  In fact all real parameters are allowed
because the summand is bounded.  Source: arXiv:2412.09080v3, §3.1. -/
theorem hasExpMoments_lawY {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    HasExpMoments (lawY β t) := by
  refine ⟨1, one_pos, ?_⟩
  intro u v _ _
  exact integrable_lawY_of_continuous hβ t (by fun_prop)

example : (0 : ℝ) < 1 := one_pos

/-- A fourth-degree polynomial majorizes the cube of the Euclidean norm.
This supplies integrability for the third-moment bound in `lem:eta`.
Source: arXiv:2412.09080v3, §3.1. -/
theorem eucl_cube_le_quartic (z : ℝ × ℝ) :
    eucl z ^ 3 ≤ 1 + (z.1 ^ 4 + 2 * (z.1 ^ 2 * z.2 ^ 2) + z.2 ^ 4) := by
  have hr : 0 ≤ eucl z := Real.sqrt_nonneg _
  have hs : eucl z ^ 2 = z.1 ^ 2 + z.2 ^ 2 := Real.sq_sqrt (by positivity)
  have h4 : eucl z ^ 4 = z.1 ^ 4 + 2 * (z.1 ^ 2 * z.2 ^ 2) + z.2 ^ 4 := by
    calc eucl z ^ 4 = (eucl z ^ 2) ^ 2 := by ring
      _ = _ := by
        rw [hs]
        ring
  rw [← h4]
  by_cases h : eucl z ≤ 1
  · have h3 := pow_le_pow_left₀ hr h 3
    nlinarith [sq_nonneg (eucl z ^ 2)]
  · have h3 : 0 ≤ eucl z ^ 3 := by positivity
    have hlarge : 0 ≤ eucl z - 1 := by linarith
    nlinarith [mul_nonneg hlarge h3]

/-- Exponential moments imply that the third Euclidean moment is finite.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`. -/
theorem integrable_eucl_cube {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    (hexp : HasExpMoments μ) : Integrable (fun z => eucl z ^ 3) μ := by
  obtain ⟨ε, hε, hE⟩ := hexp
  have h40 : Integrable (fun z : ℝ × ℝ => z.1 ^ 4) μ := by
    simpa using integrable_pow_mul_pow hE hε 4 0
  have h22 := integrable_pow_mul_pow hE hε 2 2
  have h04 : Integrable (fun z : ℝ × ℝ => z.2 ^ 4) μ := by
    simpa using integrable_pow_mul_pow hE hε 0 4
  have hp := (integrable_const (1 : ℝ)).add
    ((h40.add (h22.const_mul 2)).add h04)
  have hmeas : AEStronglyMeasurable (fun z => eucl z ^ 3) μ := by
    unfold eucl
    fun_prop
  refine hp.mono' hmeas (ae_of_all _ fun z => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (by unfold eucl; positivity)]
  exact eucl_cube_le_quartic z

example : HasExpMoments stdGauss2 :=
  ⟨1, one_pos, hasExpMomentsOn_stdGauss2⟩

/-- A third mixed monomial is bounded by the cube of the Euclidean norm.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`, order three. -/
theorem abs_mixed_cube_le (z : ℝ × ℝ) (k : ℕ) (hk : k ≤ 3) :
    |z.1 ^ k * z.2 ^ (3 - k)| ≤ eucl z ^ 3 := by
  have h1 : |z.1| ≤ eucl z := by
    rw [eucl, ← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_le_sqrt (by nlinarith)
  have h2 : |z.2| ≤ eucl z := by
    rw [eucl, ← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_le_sqrt (by nlinarith)
  have hr : 0 ≤ eucl z := Real.sqrt_nonneg _
  rw [abs_mul, abs_pow, abs_pow]
  calc |z.1| ^ k * |z.2| ^ (3 - k) ≤ eucl z ^ k * eucl z ^ (3 - k) := by
        gcongr
    _ = eucl z ^ 3 := by rw [← pow_add, Nat.add_sub_of_le hk]

example : (2 : ℕ) ≤ 3 := by norm_num

/-- **`lem:eta` at order three, for a standardized law:** each third
cumulant is bounded in absolute value by the third Euclidean moment.
The source gives an unspecified constant; centring gives constant one.
Exponential moments justify the derivative definition of the cumulant.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`, the identity after `eq:psi`. -/
theorem abs_cumulant_three_le {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hexp : HasExpMoments μ) (k : ℕ) (hk : k ≤ 3) :
    |cumulantOf μ k (3 - k)| ≤ ∫ z, eucl z ^ 3 ∂μ := by
  rw [cumulant_three_eq_integral_hermite μ hμ hexp k hk]
  obtain ⟨ε, hε, hE⟩ := hexp
  rw [integral_hermite3_eq hE hε hμ k hk]
  refine abs_integral_le_integral_abs.trans ?_
  exact integral_mono (integrable_pow_mul_pow hE hε k (3 - k)).abs
    (integrable_eucl_cube ⟨ε, hε, hE⟩) fun z => abs_mixed_cube_le z k hk

example : IsStandardized stdGauss2 ∧ HasExpMoments stdGauss2 ∧ (1 : ℕ) ≤ 3 := by
  exact ⟨isStandardized_stdGauss2,
    ⟨1, one_pos, hasExpMomentsOn_stdGauss2⟩, by norm_num⟩

/-- The third cumulants of `Y(t)` are bounded by `η₃`, with constant one.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`. -/
theorem abs_cumulant_lawY_three_le {β : ℝ} (hβ : 0 < β) (t : ℝ)
    (k : ℕ) (hk : k ≤ 3) : |cumulantOf (lawY β t) k (3 - k)| ≤ etaMoment β t 3 :=
  abs_cumulant_three_le (isStandardized_lawY hβ t) (hasExpMoments_lawY hβ t) k hk

example : (0 : ℝ) < 1 ∧ (3 : ℕ) ≤ 3 := ⟨one_pos, le_rfl⟩

end Transformer.Modes
