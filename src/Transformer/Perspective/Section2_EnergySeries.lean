/-
# §3.2 — The interaction energy as a power series

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §3.2.

Expanding `e^{β⟨x,y⟩} = Σ_k β^k ⟨x,y⟩^k / k!` and integrating term by term,

  `𝖤_β[ν] = Σ_k (2β)⁻¹ β^k/k! Σ_I (∫ m_I dν)²`   (`hasSum_interactionEnergy`),

a series of squares of linear functionals of `ν`, each of which the continuity
equation can differentiate.  Groundwork for `eq: dissipation.softmax`.
-/

import Transformer.Perspective.Section2_EnergyMoments
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.Analysis.SpecialFunctions.Exponential

open scoped BigOperators Nat
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- A uniformly dominated series of continuous functions may be integrated term by
term. -/
theorem hasSum_integral_of_abs_le {X : Type*} [TopologicalSpace X] [CompactSpace X]
    [MeasurableSpace X] [OpensMeasurableSpace X] (ρ : Measure X) [IsProbabilityMeasure ρ]
    {F : ℕ → X → ℝ} {G : X → ℝ} {u : ℕ → ℝ} (hF : ∀ k, Continuous (F k)) (hu : Summable u)
    (hle : ∀ k x, |F k x| ≤ u k) (hG : ∀ x, HasSum (fun k => F k x) (G x)) :
    HasSum (fun k => ∫ x, F k x ∂ρ) (∫ x, G x ∂ρ) := by
  have h := hasSum_integral_of_summable_integral_norm (μ := ρ)
    (fun k => integrable_of_continuous_compact (hF k) ρ)
    (hu.of_nonneg_of_le (fun k => integral_nonneg fun _ => norm_nonneg _) fun k => ?_)
  · rwa [show (fun x => ∑' k, F k x) = G from funext fun x => (hG x).tsum_eq] at h
  · calc ∫ x, ‖F k x‖ ∂ρ ≤ ∫ _x, u k ∂ρ :=
          integral_mono_of_nonneg (ae_of_all _ fun _ => norm_nonneg _) (integrable_const _)
            (ae_of_all _ fun x => by simpa [Real.norm_eq_abs] using hle k x)
      _ = u k := by simp

/-- `e^{βu} = Σ_k β^k u^k / k!`. -/
theorem hasSum_exp_mul (β u : ℝ) : HasSum (fun k : ℕ => β ^ k / k ! * u ^ k) (exp (β * u)) := by
  have := NormedSpace.expSeries_div_hasSum_exp (β * u)
  rw [← Real.exp_eq_exp_ℝ] at this
  convert this using 1
  funext k; rw [mul_pow]; ring

/-- `β e^{βu} = Σ_k β^k k u^{k-1} / k!`, the series differentiated. -/
theorem hasSum_deriv_exp_mul (β u : ℝ) :
    HasSum (fun k : ℕ => β ^ k / k ! * (k * u ^ (k - 1))) (β * exp (β * u)) := by
  rw [← hasSum_nat_add_iff' 1]
  simp only [Finset.range_one, Finset.sum_singleton, Nat.cast_zero, zero_mul, mul_zero,
    sub_zero]
  convert (hasSum_exp_mul β u).mul_left β using 1
  funext k
  rw [Nat.factorial_succ, Nat.add_sub_cancel]
  push_cast
  field_simp
  ring

/-- On the sphere, `|⟨x, y⟩| ≤ 1`. -/
theorem abs_inner_sphere_le_one (x y : SSphere d) :
    |inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)| ≤ 1 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  calc _ ≤ ‖(x : EucSpace d)‖ * ‖(y : EucSpace d)‖ := abs_real_inner_le_norm _ _
    _ = 1 := by rw [hx, hy, one_mul]

theorem continuous_inner_sphere :
    Continuous fun p : SSphere d × SSphere d =>
      inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) :=
  (continuous_subtype_val.comp continuous_fst).inner (continuous_subtype_val.comp continuous_snd)

/-- The energy as a series: `𝖤_β[ν] = Σ_k (2β)⁻¹ β^k/k! Σ_I (∫ m_I dν)²`. -/
theorem hasSum_interactionEnergy (β : ℝ) (ν : ProbSphere d) :
    HasSum (fun k : ℕ => (2 * β)⁻¹ * (β ^ k / k ! *
        ∑ I : Fin k → Fin d, (∫ x : SSphere d, mono I (x : EucSpace d) ∂(ν : Measure _)) ^ 2))
      (interactionEnergy d β ν) := by
  set ρ : Measure (SSphere d) := (ν : Measure (SSphere d))
  have hK := continuous_inner_sphere (d := d)
  have h := hasSum_integral_of_abs_le (ρ.prod ρ)
    (F := fun k p => β ^ k / k ! * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) ^ k)
    (G := fun p => exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)))
    (u := fun k => |β| ^ k / k !) (fun k => continuous_const.mul (hK.pow k))
    (Real.summable_pow_div_factorial |β|) (fun k p => ?_) (fun p => hasSum_exp_mul β _)
  · have hint : Integrable (fun p : SSphere d × SSphere d =>
        exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d))) (ρ.prod ρ) :=
      integrable_of_continuous_compact (continuous_const.mul hK).rexp _
    rw [interactionEnergy, ← integral_prod _ hint]
    convert h.mul_left (2 * β)⁻¹ using 2 with k
    rw [integral_const_mul, integral_prod_inner_pow]
  · rw [abs_mul, abs_pow, abs_div, abs_pow, Nat.abs_cast]
    refine mul_le_of_le_one_right (by positivity) (pow_le_one₀ (abs_nonneg _) ?_)
    exact abs_inner_sphere_le_one p.1 p.2

end Perspective
end Transformer
