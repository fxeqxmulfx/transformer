/-
# The original exponential transform on finite spherical measures

ArXiv:2601.21366v2, §5.1, `lem: quadpol`, injectivity.
The source restricts to probability measures. The already proved moment
argument also identifies arbitrary finite nonnegative measures: its
monomial expansions and moment-determinacy step require only finiteness.

The one place using probability normalization was integration of the
uniform series majorant. For the actual finite measure that integral is
its actual total real mass times the majorant. This summable mass-scaled
bound proves the same actual termwise integration identity.

The original two-measure exponential expansion then holds with the
original, unnormalized moments of both finite measures. Equality of
their original transforms makes the actual quadratic difference zero;
the proved positive moment-square criterion identifies the measures.
Neither measure is normalized or replaced in the statement or proof.

This finite-measure extension is needed when comparing actual positive
and negative parts of a polynomial-density candidate in the remaining
inverse-transform proof. No polynomial inverse is assumed here.
-/

import Transformer.Perspective.PositiveDefinite

open scoped BigOperators Nat ENNReal NNReal
open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Termwise integration with an actual finite measure retains its
actual total-mass factor in the summable bound. Source:
arXiv:2601.21366v2, §5.1, the expansion in the proof of `lem: quadpol`.
This generalizes `Perspective.hasSum_integral_of_abs_le`, whose measure
was a probability, using the same original continuous functions. -/
theorem hasSum_integral_of_abs_le_finite {X : Type*} [TopologicalSpace X] [CompactSpace X]
    [MeasurableSpace X] [OpensMeasurableSpace X] (ρ : Measure X) [IsFiniteMeasure ρ]
    {F : ℕ → X → ℝ} {G : X → ℝ} {u : ℕ → ℝ} (hF : ∀ k, Continuous (F k))
    (hu : Summable u) (hle : ∀ k x, |F k x| ≤ u k)
    (hG : ∀ x, HasSum (fun k => F k x) (G x)) :
    HasSum (fun k => ∫ x, F k x ∂ρ) (∫ x, G x ∂ρ) := by
  have h := hasSum_integral_of_summable_integral_norm (μ := ρ)
    (fun k => Perspective.integrable_of_continuous_compact (hF k) ρ)
    ((hu.mul_left (ρ.real univ)).of_nonneg_of_le
      (fun k => integral_nonneg fun _ => norm_nonneg _) fun k => ?_)
  · rwa [show (fun x => ∑' k, F k x) = G from funext fun x => (hG x).tsum_eq] at h
  · calc
      ∫ x, ‖F k x‖ ∂ρ ≤ ∫ _x, u k ∂ρ :=
        integral_mono_of_nonneg (ae_of_all _ fun _ => norm_nonneg _) (integrable_const _)
          (ae_of_all _ fun x => by simpa only [Real.norm_eq_abs] using hle k x)
      _ = ρ.real univ * u k := by rw [integral_const, smul_eq_mul]

/-- The original exponential interaction has its actual moment series
for arbitrary finite original measures. Source: arXiv:2601.21366v2,
§5.1, `lem: quadpol`, injectivity. This copies the proved probability
argument of `Perspective.hasSum_integral_exp_inner`, with the actual
mass-scaled summability bound above; all moments remain unnormalized. -/
theorem hasSum_integral_exp_inner_finite {d : ℕ} (β : ℝ) (a b : Measure (SSphere d))
    [IsFiniteMeasure a] [IsFiniteMeasure b] :
    HasSum (fun k : ℕ => β ^ k / k ! * ∑ I : Fin k → Fin d,
        (∫ x : SSphere d, Perspective.mono I (x : EucSpace d) ∂a) *
          ∫ y : SSphere d, Perspective.mono I (y : EucSpace d) ∂b)
      (∫ x : SSphere d, ∫ y : SSphere d,
        exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂b ∂a) := by
  have hK := Perspective.continuous_inner_sphere (d := d)
  have h := hasSum_integral_of_abs_le_finite (a.prod b)
    (F := fun k p => β ^ k / k ! * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) ^ k)
    (G := fun p => exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)))
    (u := fun k => |β| ^ k / k !) (fun k => continuous_const.mul (hK.pow k))
    (Real.summable_pow_div_factorial |β|) (fun k p => ?_)
    (fun p => Perspective.hasSum_exp_mul β _)
  · have hint : Integrable (fun p : SSphere d × SSphere d =>
        exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d))) (a.prod b) :=
      Perspective.integrable_of_continuous_compact (continuous_const.mul hK).rexp _
    rw [← integral_prod _ hint]
    convert h using 2 with k
    rw [integral_const_mul, Perspective.integral_prod_inner_pow_eq_sum]
  · rw [abs_mul, abs_pow, abs_div, abs_pow, Nat.abs_cast]
    refine mul_le_of_le_one_right (by positivity) (pow_le_one₀ (abs_nonneg _) ?_)
    exact Perspective.abs_inner_sphere_le_one p.1 p.2

/-- Equality of the original transforms identifies arbitrary finite
original nonnegative measures, at positive temperature. Source:
arXiv:2601.21366v2, §5.1, `lem: quadpol`, injectivity.
The source's probabilities are included; the proved moment argument from
`Perspective.eq_of_integral_exp_inner_eq` needs only the finite extension
of its original integral series, not a normalization of either measure. -/
theorem eq_of_integral_exp_inner_eq_finite {d : ℕ} (β : ℝ) (hβ : 0 < β)
    (μ₁ μ₂ : Measure (SSphere d)) [IsFiniteMeasure μ₁] [IsFiniteMeasure μ₂]
    (h : ∀ x : SSphere d,
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ₁ =
        ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ₂) :
    μ₁ = μ₂ := by
  have hT : ∀ a : Measure (SSphere d),
      ∫ x, ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ₁ ∂a =
        ∫ x, ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ₂ ∂a :=
    fun a => congrArg (fun f : SSphere d → ℝ => ∫ x, f x ∂a) (funext h)
  have h1 := (hasSum_integral_exp_inner_finite β μ₁ μ₁).sub
    (hasSum_integral_exp_inner_finite β μ₁ μ₂)
  have h2 := (hasSum_integral_exp_inner_finite β μ₂ μ₁).sub
    (hasSum_integral_exp_inner_finite β μ₂ μ₂)
  rw [hT, sub_self] at h1 h2
  have h3 := h1.sub h2
  rw [sub_self] at h3
  apply Perspective.eq_of_hasSum_sq_moment_sub_nonpos β hβ μ₁ μ₂ le_rfl
  convert h3 using 1
  funext k
  simp only [← mul_sub, ← Finset.sum_sub_distrib]
  congr 1
  exact Finset.sum_congr rfl fun I _ => by ring

/-- Every actual nonzero finite spherical measure has a strictly positive
original transform at every original sphere point. Source:
arXiv:2601.21366v2, §5.1, the exponential kernel in `lem: quadpol`.
Finiteness makes the original continuous kernel integrable, and its
strictly positive original support has the full nonzero measure mass. -/
theorem integral_exp_inner_pos_of_finite_nonzero {d : ℕ} (β : ℝ)
    (μ : Measure (SSphere d)) [IsFiniteMeasure μ] (hμ : μ ≠ 0) (x : SSphere d) :
    0 < ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ := by
  have hf : Continuous (fun y : SSphere d =>
      exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) :=
    Real.continuous_exp.comp (continuous_const.mul (continuous_const.inner continuous_subtype_val))
  apply (integral_pos_iff_support_of_nonneg (fun y => (exp_pos _).le)
    (Perspective.integrable_of_continuous_compact hf μ)).mpr
  have hs : Function.support (fun y : SSphere d =>
      exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) = univ := by
    ext y
    simp only [Function.mem_support, mem_univ, iff_true]
    exact (exp_pos _).ne'
  rw [hs]
  exact Measure.measure_univ_pos.mpr hμ

/-- All finite-series inputs hold jointly for the actual mass-two law
on a point, the nonzero exponential series and its actual positive terms.
Source: arXiv:2601.21366v2, §5.1, the exponential expansion for `lem: quadpol`.
The total original mass is two, so the witness exercises the finite extension. -/
example : IsFiniteMeasure ((2 : ℝ≥0) • Measure.dirac () : Measure Unit) ∧
    (∀ k : ℕ, Continuous (fun _ : Unit => (1 : ℝ) ^ k / k !)) ∧
    Summable (fun k : ℕ => (1 : ℝ) ^ k / k !) ∧
    (∀ k : ℕ, ∀ _x : Unit, |(1 : ℝ) ^ k / k !| ≤ (1 : ℝ) ^ k / k !) ∧
    ∀ _x : Unit, HasSum (fun k : ℕ => (1 : ℝ) ^ k / k !) (Real.exp 1) := by
  refine ⟨inferInstance, fun _ => continuous_const, Real.summable_pow_div_factorial 1,
    fun k _ => ?_, fun _ => ?_⟩
  · rw [abs_of_nonneg (by positivity)]
  · simpa using Perspective.hasSum_exp_mul 1 1

/-- Finite spherical masses and the original transform-equality inputs
hold jointly at positive temperature for an actual mass-two Dirac law.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol`, injectivity.
The same genuine sphere point witnesses both original measures. -/
example : (0 : ℝ) < 1 ∧
    IsFiniteMeasure ((2 : ℝ≥0) • Measure.dirac (basePoint 0) : Measure (SSphere 1)) ∧
    ((2 : ℝ≥0) • Measure.dirac (basePoint 0) : Measure (SSphere 1)) ≠ 0 ∧
    ∀ x : SSphere 1,
      ∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 1) (y : EucSpace 1))
        ∂((2 : ℝ≥0) • Measure.dirac (basePoint 0)) =
      ∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 1) (y : EucSpace 1))
        ∂((2 : ℝ≥0) • Measure.dirac (basePoint 0)) :=
  ⟨one_pos, inferInstance, by
    intro heq
    have h := congrArg (fun μ : Measure (SSphere 1) => μ univ) heq
    simp at h, fun _ => rfl⟩

end Transformer.Perceptron
