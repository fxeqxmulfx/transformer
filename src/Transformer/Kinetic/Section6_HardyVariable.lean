module

public import Transformer.Kinetic.Section6_ModeProfile
public import Transformer.Kinetic.Section1_RecencyEndpoint
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Kinetic theory for Transformers — the original Hardy change of variable

The proof of arXiv:2605.09213v1, §6, `prop:Bessel`, uses
`Y(σ;σ₀)=log((e^{λσ}-1)/(e^{λσ₀}-1))` and its derivative
`λ/(1-e^{-λσ})`. The zero-bias branch is `log(σ/σ₀)` with derivative
`1/σ`. Here both branches and every real bias are treated on the actual
positive-position domain. In particular `Y>0` for `0<σ₀<σ`, including
negative bias, as required by the original Bessel square roots.
-/

@[expose] public section

open Real Set

namespace Transformer
namespace Kinetic

/-- The actual derivative rate in the source's Hardy change of variable.
Source: arXiv:2605.09213v1, §6, proof of `prop:Bessel`, after `eq:Ylambda`. -/
noncomputable def hardyRate (lam s : ℝ) : ℝ :=
  if lam = 0 then s⁻¹ else lam / (1 - Real.exp (-(lam * s)))

/-- At nonzero bias no positive position makes the original exponential
denominator vanish. Source: arXiv:2605.09213v1, `eq:Ylambda`. -/
theorem exp_mul_sub_one_ne_zero {lam s : ℝ} (hlam : lam ≠ 0) (hs : 0 < s) :
    Real.exp (lam * s) - 1 ≠ 0 := by
  intro h
  have he : Real.exp (lam * s) = Real.exp 0 := by simpa using sub_eq_zero.mp h
  exact mul_ne_zero hlam hs.ne' (Real.exp_injective he)

example := exp_mul_sub_one_ne_zero (by norm_num : (1 : ℝ) ≠ 0) one_half_pos
example := exp_mul_sub_one_ne_zero (by norm_num : (-1 : ℝ) ≠ 0) one_half_pos

/-- The original rate in the equivalent exponential-denominator form.
Source: arXiv:2605.09213v1, §6, the differentiation of `eq:Ylambda`. -/
theorem hardyRate_eq {lam s : ℝ} (hlam : lam ≠ 0) (hs : 0 < s) :
    hardyRate lam s = lam * Real.exp (lam * s) / (Real.exp (lam * s) - 1) := by
  have he := exp_mul_sub_one_ne_zero hlam hs
  have hx : Real.exp (lam * s) ≠ 0 := (Real.exp_pos _).ne'
  rw [hardyRate, ite_eq_right hlam, Real.exp_neg]
  field_simp

example := hardyRate_eq (by norm_num : (-1 : ℝ) ≠ 0) one_half_pos

/-- Every real bias gives a positive actual Hardy rate at positive positions.
Source: arXiv:2605.09213v1, §6, the increasing variable `Y(σ;σ₀)`. -/
theorem hardyRate_pos (lam : ℝ) {s : ℝ} (hs : 0 < s) : 0 < hardyRate lam s := by
  rcases lt_trichotomy lam 0 with hl | rfl | hl
  · rw [hardyRate_eq hl.ne hs]
    exact div_pos_of_neg_of_neg (mul_neg_of_neg_of_pos hl (Real.exp_pos _))
      (sub_neg.mpr (Real.exp_lt_one_iff.mpr (mul_neg_of_neg_of_pos hl hs)))
  · simp only [hardyRate, ite_true]
    exact inv_pos.mpr hs
  · rw [hardyRate_eq hl.ne' hs]
    exact div_pos (mul_pos hl (Real.exp_pos _))
      (sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hl hs)))

example := hardyRate_pos (-1) one_half_pos
example := hardyRate_pos 0 one_half_pos

/-- The source's actual change of variable is positive between ordered
positive positions, for every real bias.
Source: arXiv:2605.09213v1, §6, `eq:Ylambda` and the zero-bias `eq:Bessel-output`. -/
theorem Yfun_pos (lam : ℝ) {s s₀ : ℝ} (hs₀ : 0 < s₀) (horder : s₀ < s) :
    0 < Yfun lam s s₀ := by
  rcases lt_trichotomy lam 0 with hl | rfl | hl
  · have hden : 0 < 1 - Real.exp (lam * s₀) :=
      sub_pos.mpr (Real.exp_lt_one_iff.mpr (mul_neg_of_neg_of_pos hl hs₀))
    have hlt : Real.exp (lam * s) < Real.exp (lam * s₀) :=
      Real.exp_lt_exp.mpr (mul_lt_mul_of_neg_left horder hl)
    rw [Yfun, ite_eq_right hl.ne,
      show Real.exp (lam * s) - 1 = -(1 - Real.exp (lam * s)) by ring,
      show Real.exp (lam * s₀) - 1 = -(1 - Real.exp (lam * s₀)) by ring, neg_div_neg_eq]
    exact Real.log_pos ((one_lt_div hden).mpr (by linarith))
  · rw [Yfun, ite_eq_left rfl]
    exact Real.log_pos ((one_lt_div hs₀).mpr horder)
  · have hden : 0 < Real.exp (lam * s₀) - 1 :=
      sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hl hs₀))
    have hlt : Real.exp (lam * s₀) < Real.exp (lam * s) :=
      Real.exp_lt_exp.mpr (mul_lt_mul_of_pos_left horder hl)
    rw [Yfun, ite_eq_right hl.ne']
    exact Real.log_pos ((one_lt_div hden).mpr (by linarith))

example := Yfun_pos (-1) one_half_pos one_half_lt_one
example := Yfun_pos 0 one_half_pos one_half_lt_one

/-- At nonzero bias the original variable is a difference of actual
logarithms on positive positions. Source: arXiv:2605.09213v1, `eq:Ylambda`. -/
theorem Yfun_eq_log_sub {lam s s₀ : ℝ} (hlam : lam ≠ 0) (hs : 0 < s) (hs₀ : 0 < s₀) :
    Yfun lam s s₀ = Real.log (Real.exp (lam * s) - 1) - Real.log (Real.exp (lam * s₀) - 1) := by
  rw [Yfun, ite_eq_right hlam,
    Real.log_div (exp_mul_sub_one_ne_zero hlam hs) (exp_mul_sub_one_ne_zero hlam hs₀)]

example := Yfun_eq_log_sub (by norm_num : (-1 : ℝ) ≠ 0) one_pos one_half_pos

/-- The actual spatial derivative of the original variable, for every
real bias including zero. Source: arXiv:2605.09213v1, §6, proof of `prop:Bessel`. -/
theorem hasDerivAt_Yfun (lam : ℝ) {s s₀ : ℝ} (hs : 0 < s) (hs₀ : 0 < s₀) :
    HasDerivAt (fun s => Yfun lam s s₀) (hardyRate lam s) s := by
  by_cases hl : lam = 0
  · subst lam
    have hd := ((hasDerivAt_id' (x := s)).log hs.ne').sub_const (Real.log s₀)
    rw [one_div] at hd
    have hrate : hardyRate 0 s = s⁻¹ := by simp only [hardyRate, ite_true]
    rw [hrate]
    apply hd.congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds hs] with z hz
    rw [Yfun, ite_eq_left rfl, Real.log_div hz.ne' hs₀.ne']
  · have hd := ((((hasDerivAt_id' (x := s)).const_mul lam).exp.sub_const 1).log
      (exp_mul_sub_one_ne_zero hl hs)).sub_const (Real.log (Real.exp (lam * s₀) - 1))
    have h' : HasDerivAt (fun z => Real.log (Real.exp (lam * z) - 1) -
        Real.log (Real.exp (lam * s₀) - 1)) (hardyRate lam s) s := by
      rw [hardyRate_eq hl hs]
      convert hd using 1
      ring
    apply h'.congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds hs] with z hz
    exact Yfun_eq_log_sub hl hz hs₀

example := hasDerivAt_Yfun (-1) one_pos one_half_pos
example := hasDerivAt_Yfun 0 one_pos one_half_pos

/-- The original Bessel profile equals its proved entire series at every
ordered positive pair of positions, without restricting the real bias.
Source: arXiv:2605.09213v1, §6, `eq:Bessel-full`, `eq:Bessel-output`. -/
theorem hardyProfile_eq_psi (lam : ℝ) {a t s s₀ : ℝ} (hs₀ : 0 < s₀) (horder : s₀ < s)
    (hat : 0 ≤ a * t) :
    hardyProfile lam a t s s₀ = graphon lam s s₀ * psi (a * t) (Yfun lam s s₀) := by
  rw [hardyProfile, mul_assoc, sqrt_mul_besselI hat (Yfun_pos lam hs₀ horder)]

example := hardyProfile_eq_psi (-1) (a := 1) (t := 1) one_half_pos one_half_lt_one (by norm_num)

/-- The original change of variable is nonnegative on the full closed
spatial interval, including its source endpoint.
Source: arXiv:2605.09213v1, §6, the Goursat domain in `prop:Bessel`. -/
theorem Yfun_nonneg (lam : ℝ) {s s₀ : ℝ} (hs₀ : 0 < s₀) (horder : s₀ ≤ s) :
    0 ≤ Yfun lam s s₀ := by
  rcases lt_or_eq_of_le horder with hlt | heq
  · exact (Yfun_pos lam hs₀ hlt).le
  · rw [← heq, Yfun_diagonal]

example := Yfun_nonneg (-1) one_half_pos one_half_lt_one.le

/-- The original change of variable is continuous at every positive
spatial position. Source: arXiv:2605.09213v1, §6, the Goursat change in `prop:Bessel`. -/
theorem continuousAt_Yfun (lam : ℝ) {s s₀ : ℝ} (hs : 0 < s) (hs₀ : 0 < s₀) :
    ContinuousAt (fun s => Yfun lam s s₀) s := (hasDerivAt_Yfun lam hs hs₀).continuousAt

example := continuousAt_Yfun (-1) one_pos one_half_pos

end Kinetic
end Transformer
