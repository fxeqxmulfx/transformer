module

public import Transformer.Kinetic.Section6_GoursatSeries
public import Transformer.Kinetic.Section6_HardyKernel
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-!
# Kinetic theory for Transformers — the actual spatial Volterra primitive

The Goursat series in arXiv:2605.09213v1, §6, proof of `prop:Bessel`,
has derivative `ψ_c(Y(σ;σ₀))r_λ(σ)` in the original spatial coordinate.
Its integral is the actual series value minus its source boundary value
one. The original two-step graphon factorization then evaluates the
actual Volterra term in `eq:Hardy-hom`. The causal endpoint values are
kept: they are removed only by equality on the open integration interval.
-/

@[expose] public section

open Real Set MeasureTheory

namespace Transformer
namespace Kinetic

/-- The actual Goursat series has the source's spatial derivative after
the original change of variable. Source: arXiv:2605.09213v1, §6, proof of `prop:Bessel`. -/
theorem hasDerivAt_goursat_Yfun (lam c : ℝ) {s s₀ : ℝ} (hs : 0 < s) (hs₀ : 0 < s₀) :
    HasDerivAt (fun r => goursatSeries c (Yfun lam r s₀))
      (hardyRate lam s * psi c (Yfun lam s s₀)) s := by
  have h := (hasDerivAt_goursatSeries c (Yfun lam s s₀)).comp s (hasDerivAt_Yfun lam hs hs₀)
  exact h.congr_deriv (mul_comm _ _)

example := hasDerivAt_goursat_Yfun (-1) 1 one_pos one_half_pos

/-- The actual spatial derivative integrand is continuous on the original
positive-position interval. Source: arXiv:2605.09213v1, §6, the spatial integral in `prop:Bessel`. -/
theorem continuousOn_hardyRate_psi (lam c : ℝ) {s s₀ : ℝ} (hs₀ : 0 < s₀) (horder : s₀ ≤ s) :
    ContinuousOn (fun r => hardyRate lam r * psi c (Yfun lam r s₀)) (uIcc s₀ s) := by
  rw [uIcc_of_le horder]
  intro r hr
  have hr0 := hs₀.trans_le hr.1
  exact ((continuousAt_hardyRate lam hr0).mul
    ((hasDerivAt_powerSum (isEntire_psiCoeffs c) (Yfun lam r s₀)).continuousAt.comp
      (f := fun z : ℝ => Yfun lam z s₀) (continuousAt_Yfun lam hr0 hs₀))).continuousWithinAt

example := continuousOn_hardyRate_psi (-1) 1 one_half_pos one_half_lt_one.le

/-- The original spatial primitive equals the actual Goursat series
minus its source boundary value.
Source: arXiv:2605.09213v1, §6, the integral formula for `F` in `prop:Bessel`. -/
theorem integral_hardyRate_psi (lam c : ℝ) {s s₀ : ℝ} (hs₀ : 0 < s₀) (horder : s₀ ≤ s) :
    (∫ r in s₀..s, hardyRate lam r * psi c (Yfun lam r s₀)) =
      goursatSeries c (Yfun lam s s₀) - 1 := by
  have hi := (continuousOn_hardyRate_psi lam c hs₀ horder).intervalIntegrable (μ := volume)
  have hder : ∀ r ∈ uIcc s₀ s,
      HasDerivAt (fun r => goursatSeries c (Yfun lam r s₀))
        (hardyRate lam r * psi c (Yfun lam r s₀)) r := by
    rw [uIcc_of_le horder]
    intro r hr
    exact hasDerivAt_goursat_Yfun lam c (hs₀.trans_le hr.1) hs₀
  have h := intervalIntegral.integral_eq_sub_of_hasDerivAt hder hi
  rwa [Yfun_diagonal, goursatSeries_zero] at h

example := integral_hardyRate_psi (-1) 1 one_half_pos one_half_lt_one.le

/-- The original Volterra integrand factors through the actual derivative
of its spatial variable, at every interior integration point.
Source: arXiv:2605.09213v1, §6, the reduction of `eq:Hardy-hom` in `prop:Bessel`. -/
theorem graphon_mul_hardyProfile (lam : ℝ) {a t s r s₀ : ℝ} (hs₀ : 0 < s₀)
    (hr : s₀ < r) (hs : r < s) (hat : 0 ≤ a * t) :
    graphon lam s r * hardyProfile lam a t r s₀ =
      graphon lam s s₀ * (hardyRate lam r * psi (a * t) (Yfun lam r s₀)) := by
  rw [hardyProfile_eq_psi lam hs₀ hr hat, ← mul_assoc, graphon_mul_of_order lam hr hs]
  ring

example := graphon_mul_hardyProfile (-1) (a := 1) (t := 1) (s := 1) (r := 1 / 2) (s₀ := 1 / 4)
  (by norm_num) (by norm_num) one_half_lt_one (by norm_num)

/-- The actual original Volterra term has its exact source Goursat-series
value, including the actual causal endpoints.
Source: arXiv:2605.09213v1, §6, `eq:Hardy-hom` in the proof of `prop:Bessel`. -/
theorem integral_graphon_hardyProfile (lam : ℝ) {a t s s₀ : ℝ} (hs₀ : 0 < s₀)
    (horder : s₀ < s) (hat : 0 ≤ a * t) :
    (∫ r in s₀..s, graphon lam s r * hardyProfile lam a t r s₀) =
      graphon lam s s₀ * (goursatSeries (a * t) (Yfun lam s s₀) - 1) := by
  have heq := intervalIntegral.integral_congr_Ioo_of_le (μ := volume) horder.le
    (fun r hr => graphon_mul_hardyProfile lam hs₀ hr.1 hr.2 hat)
  rw [heq, intervalIntegral.integral_const_mul, integral_hardyRate_psi lam (a * t) hs₀ horder.le]

example := integral_graphon_hardyProfile (-1) (a := 1) (t := 1) one_half_pos one_half_lt_one
  (by norm_num)

/-- The original Bessel formula satisfies the actual time equation on
the nonnegative half-line, including its one-sided derivative at zero.
The entire series is used only on the original `at ≥ 0` domain.
Source: arXiv:2605.09213v1, §6, `eq:Hardy-hom` in `prop:Bessel`. -/
theorem hasDerivWithinAt_hardyProfile (lam : ℝ) {a t s s₀ : ℝ} (ha : 0 ≤ a)
    (ht : 0 ≤ t) (hs₀ : 0 < s₀) (horder : s₀ < s) :
    HasDerivWithinAt (fun t => hardyProfile lam a t s s₀)
      (a * (graphon lam s s₀ + ∫ r in s₀..s, graphon lam s r * hardyProfile lam a t r s₀))
      (Ici 0) t := by
  have hd := (hasDerivAt_psi_time a (Yfun lam s s₀) t).const_mul (graphon lam s s₀)
  have heq : graphon lam s s₀ * (a * goursatSeries (a * t) (Yfun lam s s₀)) =
      a * (graphon lam s s₀ + ∫ r in s₀..s, graphon lam s r * hardyProfile lam a t r s₀) := by
    rw [integral_graphon_hardyProfile lam hs₀ horder (mul_nonneg ha ht)]
    ring
  have hD := (hd.hasDerivWithinAt (s := Ici 0)).congr_deriv heq
  exact hD.congr (fun u hu => hardyProfile_eq_psi lam hs₀ horder (mul_nonneg ha hu))
    (hardyProfile_eq_psi lam hs₀ horder (mul_nonneg ha ht))

/-- The entire original Bessel formula is an actual solution of the
Volterra-Hardy equation, at every real bias and positive source position.
The initial condition and the actual time derivative are proved, with
the original spatial integral and the original causal kernel.
Source: arXiv:2605.09213v1, §6, existence in `prop:Bessel`, `eq:Hardy-hom`. -/
theorem isVolterraHardySolution_hardyProfile (lam : ℝ) {a s₀ : ℝ} (ha : 0 ≤ a)
    (hs₀ : 0 < s₀) : IsVolterraHardySolution lam a s₀ (fun t s => hardyProfile lam a t s s₀) := by
  refine ⟨fun s _ => hardyProfile_zero lam a s s₀, fun s hs t ht => ?_⟩
  exact hasDerivWithinAt_hardyProfile lam ha ht hs₀ hs.1

/-- The actual source parameters `a=1`, `σ₀=1/2`, `σ=1`, `t=0` witness
all existence hypotheses, including the initial-time endpoint. Negative
bias is allowed by the manuscript, and zero bias has the same proof.
Source: arXiv:2605.09213v1, §1, `prop:Bessel`. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 ∧
    IsVolterraHardySolution (-1) 1 (1 / 2) (fun t s => hardyProfile (-1) 1 t s (1 / 2)) ∧
    IsVolterraHardySolution 0 1 (1 / 2) (fun t s => hardyProfile 0 1 t s (1 / 2)) :=
  ⟨zero_le_one, le_rfl, one_half_pos, one_half_lt_one,
    isVolterraHardySolution_hardyProfile (-1) zero_le_one one_half_pos,
    isVolterraHardySolution_hardyProfile 0 zero_le_one one_half_pos⟩

/-- The original time profile is continuous from its actual nonnegative
time domain, including zero. Source: arXiv:2605.09213v1, §6, `prop:Bessel`. -/
theorem continuousWithinAt_hardyProfile (lam : ℝ) {a t s s₀ : ℝ} (ha : 0 ≤ a)
    (ht : 0 ≤ t) (hs₀ : 0 < s₀) (horder : s₀ < s) :
    ContinuousWithinAt (fun t => hardyProfile lam a t s s₀) (Ici 0) t :=
  (hasDerivWithinAt_hardyProfile lam ha ht hs₀ horder).continuousWithinAt

/-- The entire original time profile is continuous on the nonnegative
half-line. Source: arXiv:2605.09213v1, §6, the solution in `prop:Bessel`. -/
theorem continuousOn_hardyProfile_time (lam : ℝ) {a s s₀ : ℝ} (ha : 0 ≤ a)
    (hs₀ : 0 < s₀) (horder : s₀ < s) :
    ContinuousOn (fun t => hardyProfile lam a t s s₀) (Ici 0) :=
  fun t ht => continuousWithinAt_hardyProfile lam (t := t) ha ht hs₀ horder

example := continuousWithinAt_hardyProfile (-1) (a := 1) (t := 0) zero_le_one le_rfl
  one_half_pos one_half_lt_one
example := continuousOn_hardyProfile_time 0 (a := 1) zero_le_one one_half_pos one_half_lt_one

example := hasDerivWithinAt_hardyProfile (-1) (a := 1) (t := 1) zero_le_one zero_le_one
  one_half_pos one_half_lt_one

end Kinetic
end Transformer
