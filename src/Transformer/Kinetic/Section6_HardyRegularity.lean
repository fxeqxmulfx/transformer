module

public import Transformer.Kinetic.Section6_HardyPrimitive

/-!
# Kinetic theory for Transformers — regularity of the original Bessel profile

The spatial and time integrals in arXiv:2605.09213v1, §6, proof of
`prop:Bessel`, use the classical solution. The actual Bessel formula
equals its entire-series expression at ordered positive positions.
That expression gives its continuous extension to the source boundary,
its spatial integrability and a uniform bound on every finite time
rectangle. The literal diagonal formula is kept throughout; a singleton
change is used only when identifying its actual spatial integral.
-/

@[expose] public section

open Real Set MeasureTheory

namespace Transformer
namespace Kinetic

/-- The continuous entire-series expression of the original profile,
including its spatial source-boundary extension.
Source: arXiv:2605.09213v1, §6, proof of `prop:Bessel`, the formula for `∂_y U`. -/
noncomputable def hardySeriesProfile (lam a s₀ t s : ℝ) : ℝ :=
  hardyKernel lam s s₀ * psi (a * t) (Yfun lam s s₀)

/-- The actual Bessel formula equals the original entire-series expression
on its stated domain. Source: arXiv:2605.09213v1, §6, `eq:Bessel-full`, `eq:Bessel-output`. -/
theorem hardyProfile_eq_seriesProfile (lam : ℝ) {a t s s₀ : ℝ} (hs₀ : 0 < s₀)
    (horder : s₀ < s) (hat : 0 ≤ a * t) :
    hardyProfile lam a t s s₀ = hardySeriesProfile lam a s₀ t s := by
  rw [hardyProfile_eq_psi lam hs₀ horder hat, graphon_eq_hardyKernel horder]
  rfl

example := hardyProfile_eq_seriesProfile (-1) (a := 1) (t := 1)
  one_half_pos one_half_lt_one (by norm_num)

/-- Joint continuity of the actual entire-series extension on the
positive-position domain, at every finite time.
Source: arXiv:2605.09213v1, §6, the Goursat solution in `prop:Bessel`. -/
theorem continuousAt_hardySeriesProfile (lam a : ℝ) {s₀ : ℝ} (hs₀ : 0 < s₀)
    {p : ℝ × ℝ} (hp : 0 < p.2) :
    ContinuousAt (fun p : ℝ × ℝ => hardySeriesProfile lam a s₀ p.1 p.2) p := by
  have hY : ContinuousAt (fun q : ℝ × ℝ => Yfun lam q.2 s₀) p :=
    (continuousAt_Yfun lam hp hs₀).comp continuousAt_snd
  have hA : ContinuousAt (fun q : ℝ × ℝ => a * q.1) p := by fun_prop
  have hψ : ContinuousAt (fun q : ℝ × ℝ => psi (a * q.1) (Yfun lam q.2 s₀)) p := by
    have heq : (fun q : ℝ × ℝ => psi (a * q.1) (Yfun lam q.2 s₀)) =
        fun q => (a * q.1) * psi 1 ((a * q.1) * Yfun lam q.2 s₀) :=
      funext fun q => psi_scale (a * q.1) (Yfun lam q.2 s₀)
    rw [heq]
    exact hA.mul ((hasDerivAt_powerSum (isEntire_psiCoeffs 1)
      ((a * p.1) * Yfun lam p.2 s₀)).continuousAt.comp
        (f := fun q : ℝ × ℝ => (a * q.1) * Yfun lam q.2 s₀) (hA.mul hY))
  have hK : ContinuousAt (fun q : ℝ × ℝ => hardyKernel lam q.2 s₀) p :=
    (continuousAt_hardyKernel lam (p := (p.2, s₀)) hp).comp
      (f := fun q : ℝ × ℝ => (q.2, s₀)) (continuousAt_snd.prodMk continuousAt_const)
  exact hK.mul hψ

example := continuousAt_hardySeriesProfile (-1) 1 one_half_pos (p := (0, 1)) one_pos

/-- The source's original Bessel profile is uniformly bounded on every
finite time rectangle and its actual ordered spatial domain.
Source: arXiv:2605.09213v1, §6, the classical solution in `prop:Bessel`. -/
theorem hardyProfile_bounded (lam : ℝ) {a s₀ : ℝ} (ha : 0 ≤ a) (hs₀ : 0 < s₀) (T : ℝ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ t ∈ Icc (0 : ℝ) T, ∀ s ∈ Ioc s₀ 1,
      |hardyProfile lam a t s s₀| ≤ C := by
  have hc : ContinuousOn (fun p : ℝ × ℝ => hardySeriesProfile lam a s₀ p.1 p.2)
      (Icc (0 : ℝ) T ×ˢ Icc s₀ 1) := by
    intro p hp
    exact (continuousAt_hardySeriesProfile lam a hs₀ (hs₀.trans_le hp.2.1)).continuousWithinAt
  obtain ⟨C, hC⟩ := (isCompact_Icc.prod isCompact_Icc).exists_bound_of_continuousOn hc
  refine ⟨max C 0, le_max_right _ _, fun t ht s hs => ?_⟩
  rw [hardyProfile_eq_seriesProfile lam hs₀ hs.1 (mul_nonneg ha ht.1)]
  exact (hC (t, s) ⟨ht, hs.1.le, hs.2⟩).trans (le_max_left _ _)

example := hardyProfile_bounded (-1) (a := 1) zero_le_one one_half_pos 1

/-- The original entire-series extension is spatially continuous on the
actual positive-position interval.
Source: arXiv:2605.09213v1, §6, the spatial integral in `prop:Bessel`. -/
theorem continuousOn_hardySeriesProfile (lam a t : ℝ) {s₀ s : ℝ} (hs₀ : 0 < s₀)
    (horder : s₀ ≤ s) : ContinuousOn (hardySeriesProfile lam a s₀ t) (uIcc s₀ s) := by
  rw [uIcc_of_le horder]
  intro r hr
  exact ((continuousAt_hardySeriesProfile lam a hs₀ (p := (t, r)) (hs₀.trans_le hr.1)).comp
    (continuousAt_const.prodMk continuousAt_id)).continuousWithinAt

example := continuousOn_hardySeriesProfile (-1) 1 1 one_half_pos one_half_lt_one.le

/-- The original Bessel profile is spatially integrable; its literal
source endpoint differs from the continuous extension only on a singleton.
Source: arXiv:2605.09213v1, §6, the spatial integral in `prop:Bessel`. -/
theorem intervalIntegrable_hardyProfile (lam : ℝ) {a t s₀ s : ℝ} (hs₀ : 0 < s₀)
    (horder : s₀ ≤ s) (hat : 0 ≤ a * t) :
    IntervalIntegrable (fun r => hardyProfile lam a t r s₀) volume s₀ s := by
  have hc := (continuousOn_hardySeriesProfile lam a t hs₀ horder).intervalIntegrable (μ := volume)
  apply (intervalIntegrable_congr_uIoo (μ := volume) (f := fun r => hardyProfile lam a t r s₀)
    (g := hardySeriesProfile lam a s₀ t) ?_).mpr hc
  rw [uIoo_of_le horder]
  intro r hr
  exact hardyProfile_eq_seriesProfile lam hs₀ hr.1 hat

example := intervalIntegrable_hardyProfile (-1) (a := 1) (t := 1) one_half_pos
  one_half_lt_one.le (by norm_num)

/-- The original Volterra integrand is integrable. Its smooth expression
agrees on the open interval while both causal endpoint values are retained.
Source: arXiv:2605.09213v1, §6, `eq:Hardy-hom` in `prop:Bessel`. -/
theorem intervalIntegrable_graphon_hardyProfile (lam : ℝ) {a t s s₀ : ℝ} (hs₀ : 0 < s₀)
    (horder : s₀ ≤ s) (hat : 0 ≤ a * t) :
    IntervalIntegrable (fun r => graphon lam s r * hardyProfile lam a t r s₀) volume s₀ s := by
  have hi := (intervalIntegrable_hardyProfile lam hs₀ horder hat).mul_continuousOn
    (continuous_hardyKernel_right lam s).continuousOn
  have hi' : IntervalIntegrable (fun r => hardyKernel lam s r * hardyProfile lam a t r s₀)
      volume s₀ s := by simpa only [mul_comm] using hi
  apply (intervalIntegrable_congr_uIoo (μ := volume) (f := fun r => graphon lam s r *
    hardyProfile lam a t r s₀) (g := fun r => hardyKernel lam s r * hardyProfile lam a t r s₀) ?_).mpr hi'
  rw [uIoo_of_le horder]
  intro r hr
  change graphon lam s r * hardyProfile lam a t r s₀ =
    hardyKernel lam s r * hardyProfile lam a t r s₀
  rw [graphon_eq_hardyKernel hr.2]

example := intervalIntegrable_graphon_hardyProfile (-1) (a := 1) (t := 1) one_half_pos
  one_half_lt_one.le (by norm_num)

/-- The actual continuous source-boundary extension is `r_λ(σ₀)at`,
as follows from the source's series boundary value `ψ_c(0)=c`.
Source: arXiv:2605.09213v1, §6, the spatial derivative of `U` in `prop:Bessel`. -/
theorem hardySeriesProfile_source (lam a s₀ t : ℝ) :
    hardySeriesProfile lam a s₀ t s₀ = hardyRate lam s₀ * (a * t) := by
  rw [hardySeriesProfile, hardyKernel, Yfun_diagonal]
  simp only [sub_self, mul_zero, neg_zero, Real.exp_zero, mul_one, psi, powerSum_zero]
  norm_num [psiCoeffs]

/-- The actual Bessel formula has the source's continuous right limit
at the source boundary, where its literal diagonal value is zero.
Source: arXiv:2605.09213v1, §6, the classical spatial profile in `prop:Bessel`. -/
theorem hardyProfile_source_limit (lam : ℝ) {a t s₀ : ℝ} (hs₀ : 0 < s₀) (hat : 0 ≤ a * t) :
    Filter.Tendsto (fun s => hardyProfile lam a t s s₀) (nhdsWithin s₀ (Ioi s₀))
      (nhds (hardyRate lam s₀ * (a * t))) := by
  have hc := (continuousAt_hardySeriesProfile lam a hs₀ (p := (t, s₀)) hs₀).comp
    (continuousAt_const.prodMk continuousAt_id)
  have hl := hc.tendsto.mono_left (nhdsWithin_le_nhds (s := Ioi s₀))
  change Filter.Tendsto (fun s => hardySeriesProfile lam a s₀ t s)
    (nhdsWithin s₀ (Ioi s₀)) (nhds (hardySeriesProfile lam a s₀ t s₀)) at hl
  rw [hardySeriesProfile_source] at hl
  apply hl.congr'
  filter_upwards [self_mem_nhdsWithin (a := s₀) (s := Ioi s₀)] with s hs
  exact (hardyProfile_eq_seriesProfile lam hs₀ hs hat).symm

example := hardyProfile_source_limit (-1) (a := 1) (t := 1) one_half_pos (by norm_num)

end Kinetic
end Transformer
