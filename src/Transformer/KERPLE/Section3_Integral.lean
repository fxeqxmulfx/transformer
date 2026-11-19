module

public import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
public import Mathlib.Analysis.SpecialFunctions.Integrability.Basic
public import Mathlib.MeasureTheory.Integral.IntegralEqImproper
public import Transformer.KERPLE.Section3_Schoenberg

/-!
# §3.2 Fact 3, the integral: `s^α` as a mixture of `1 − e^{−ts}`

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §3.2. Fact 3 cites Berg, Christensen and Ressel
(1984); its proof for the powers `−(−k̃)^α` writes `s^α` as an integral of the functions
`1 − e^{−ts}`, `t > 0`, which Schoenberg's theorem turns into CPD kernels.

* For `0 < α < 1` and `s ≥ 0`, `t ↦ (1 − e^{−st}) t^{−1−α}` is integrable on `(0, ∞)`
  (`integrableOn_one_sub_exp_mul_rpow`): it is at most `s t^{−α}` on `(0, 1]` and `t^{−1−α}` beyond.
* `C_α = ∫_0^∞ (1 − e^{−u}) u^{−1−α} du` is positive (`integral_one_sub_exp_rpow_pos`), and
  `s^α C_α = ∫_0^∞ (1 − e^{−st}) t^{−1−α} dt` (`rpow_mul_integral`), by the substitution `u = st`.
* An integral of CPD kernels is CPD (`IsCPDKernel.setIntegral`): the form of the integral is the
  integral of the forms.
-/

@[expose] public section

namespace Transformer.KERPLE

open Finset Filter MeasureTheory Set

variable {X : Type*}

/-- A kernel equal to a CPD kernel is CPD. -/
theorem IsCPDKernel.congr {k k' : X → X → ℝ} (hk : IsCPDKernel k) (h : ∀ x y, k x y = k' x y) :
    IsCPDKernel k' := by
  have : k = k' := funext fun x => funext (h x)
  exact this ▸ hk

/-- **An integral of CPD kernels is CPD**: for a measurable `s`, kernels `K t` CPD for `t ∈ s`,
and integrable entries, `∫_{t ∈ s} K t` is CPD. -/
theorem IsCPDKernel.setIntegral {K : ℝ → X → X → ℝ} {s : Set ℝ} (hs : MeasurableSet s)
    (hK : ∀ t ∈ s, IsCPDKernel (K t)) (hint : ∀ x y, IntegrableOn (fun t => K t x y) s) :
    IsCPDKernel fun x y => ∫ t in s, K t x y := by
  refine ⟨fun x y => setIntegral_congr_fun hs fun t ht => (hK t ht).1 x y, fun N x c hc => ?_⟩
  have hswap : ∫ t in s, ∑ i, ∑ j, c i * c j * K t (x i) (x j) =
      ∑ i, ∑ j, c i * c j * ∫ t in s, K t (x i) (x j) := by
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ =>
      (hint (x i) (x j)).const_mul _]
    refine sum_congr rfl fun i _ => ?_
    rw [integral_finsetSum _ fun j _ => (hint (x i) (x j)).const_mul _]
    exact sum_congr rfl fun j _ => integral_const_mul _ _
  rw [← hswap]
  exact setIntegral_nonneg hs fun t ht => (hK t ht).2 N x c hc

/-- The hypotheses of `IsCPDKernel.setIntegral` hold for the constant family `−‖x − y‖²` on
`[0, 1]`. -/
example : IsCPDKernel fun x y : ℝ => ∫ _ in Icc (0 : ℝ) 1, -‖x - y‖ ^ 2 :=
  IsCPDKernel.setIntegral (K := fun _ x y : ℝ => -‖x - y‖ ^ 2) measurableSet_Icc
    (fun _ _ => isCPDKernel_neg_norm_sub_sq) fun _ _ => continuous_const.integrableOn_Icc

/-- `0 ≤ 1 − e^{−st} ≤ st` for `s, t ≥ 0`. -/
theorem one_sub_exp_neg_mem {s t : ℝ} (hs : 0 ≤ s) (ht : 0 ≤ t) :
    0 ≤ 1 - Real.exp (-(s * t)) ∧ 1 - Real.exp (-(s * t)) ≤ s * t := by
  have h1 : Real.exp (-(s * t)) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
  have h2 := Real.add_one_le_exp (-(s * t))
  constructor <;> linarith

/-- The hypotheses of `one_sub_exp_neg_mem` hold for `s = t = 1`. -/
example : 0 ≤ 1 - Real.exp (-(1 * 1)) ∧ 1 - Real.exp (-(1 * 1)) ≤ 1 * 1 :=
  one_sub_exp_neg_mem zero_le_one zero_le_one

/-- `t ↦ (1 − e^{−st}) t^{−1−α}` is integrable on `(0, ∞)` for `0 < α < 1` and `s ≥ 0`. -/
theorem integrableOn_one_sub_exp_mul_rpow {α s : ℝ} (hα : 0 < α) (hα1 : α < 1) (hs : 0 ≤ s) :
    IntegrableOn (fun t : ℝ => (1 - Real.exp (-(s * t))) * t ^ (-1 - α)) (Ioi 0) := by
  have hcont : ContinuousOn (fun t : ℝ => (1 - Real.exp (-(s * t))) * t ^ (-1 - α)) (Ioi 0) :=
    (Continuous.continuousOn (by fun_prop)).mul
      (continuousOn_id.rpow_const fun t ht => Or.inl (ne_of_gt ht))
  rw [← Ioc_union_Ioi_eq_Ioi zero_le_one]
  refine IntegrableOn.union ?_ ?_
  · have hg : IntegrableOn (fun t : ℝ => s * t ^ (-α)) (Ioc 0 1) := by
      have := intervalIntegral.intervalIntegrable_rpow' (a := 0) (b := 1) (r := -α) (by linarith)
      rw [intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one] at this
      exact this.const_mul s
    refine Integrable.mono' hg ((hcont.mono Ioc_subset_Ioi_self).aestronglyMeasurable
      measurableSet_Ioc) ((ae_restrict_iff' measurableSet_Ioc).mpr (Eventually.of_forall ?_))
    intro t ht
    have h := one_sub_exp_neg_mem hs ht.1.le
    have hr := Real.rpow_nonneg ht.1.le (-1 - α)
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg h.1 hr),
      show -α = 1 + (-1 - α) by ring, Real.rpow_add ht.1, Real.rpow_one]
    calc (1 - Real.exp (-(s * t))) * t ^ (-1 - α) ≤ s * t * t ^ (-1 - α) :=
          mul_le_mul_of_nonneg_right h.2 hr
      _ = s * (t * t ^ (-1 - α)) := mul_assoc _ _ _
  · refine Integrable.mono' (integrableOn_Ioi_rpow_of_lt (a := -1 - α) (by linarith) one_pos)
      ((hcont.mono (Ioi_subset_Ioi zero_le_one)).aestronglyMeasurable measurableSet_Ioi)
      ((ae_restrict_iff' measurableSet_Ioi).mpr (Eventually.of_forall ?_))
    intro t ht
    have h := one_sub_exp_neg_mem hs (zero_le_one.trans (le_of_lt ht))
    have hr := Real.rpow_nonneg (zero_le_one.trans (le_of_lt ht)) (-1 - α)
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg h.1 hr)]
    have : 1 - Real.exp (-(s * t)) ≤ 1 := by linarith [Real.exp_pos (-(s * t))]
    simpa using mul_le_mul_of_nonneg_right this hr

/-- The hypotheses of `integrableOn_one_sub_exp_mul_rpow` hold for `α = 1/2` and `s = 2`. -/
example : IntegrableOn (fun t : ℝ => (1 - Real.exp (-(2 * t))) * t ^ (-1 - (1 / 2 : ℝ))) (Ioi 0) :=
  integrableOn_one_sub_exp_mul_rpow (by norm_num) (by norm_num) (by norm_num)

/-- `C_α = ∫_0^∞ (1 − e^{−u}) u^{−1−α} du` is positive for `0 < α < 1`. -/
theorem integral_one_sub_exp_rpow_pos {α : ℝ} (hα : 0 < α) (hα1 : α < 1) :
    0 < ∫ u in Ioi 0, (1 - Real.exp (-u)) * u ^ (-1 - α) := by
  have hint := integrableOn_one_sub_exp_mul_rpow hα hα1 zero_le_one
  simp only [one_mul] at hint
  have hpos : ∀ u ∈ Ioi (0 : ℝ), 0 < (1 - Real.exp (-u)) * u ^ (-1 - α) := fun u hu =>
    mul_pos (by linarith [Real.exp_lt_one_iff.mpr (neg_lt_zero.mpr hu)])
      (Real.rpow_pos_of_pos hu _)
  rw [setIntegral_pos_iff_support_of_nonneg_ae
    ((ae_restrict_iff' measurableSet_Ioi).mpr (Eventually.of_forall fun u hu => (hpos u hu).le))
    hint]
  have hsub : Ioi (0 : ℝ) ⊆
      Function.support (fun u : ℝ => (1 - Real.exp (-u)) * u ^ (-1 - α)) ∩ Ioi 0 :=
    fun u hu => ⟨(hpos u hu).ne', hu⟩
  calc (0 : ENNReal) < volume (Ioi (0 : ℝ)) := by rw [Real.volume_Ioi]; exact ENNReal.zero_lt_top
    _ ≤ _ := measure_mono hsub

/-- The hypotheses of `integral_one_sub_exp_rpow_pos` hold for `α = 1/2`. -/
example : 0 < ∫ u in Ioi 0, (1 - Real.exp (-u)) * u ^ (-1 - (1 / 2 : ℝ)) :=
  integral_one_sub_exp_rpow_pos (by norm_num) (by norm_num)

/-- **The integral representation of `s^α`.** For `α > 0` and `s ≥ 0`,
`s^α C_α = ∫_0^∞ (1 − e^{−st}) t^{−1−α} dt`, by the substitution `u = st`. -/
theorem rpow_mul_integral {α s : ℝ} (hα : 0 < α) (hs : 0 ≤ s) :
    s ^ α * ∫ u in Ioi 0, (1 - Real.exp (-u)) * u ^ (-1 - α) =
      ∫ t in Ioi 0, (1 - Real.exp (-(s * t))) * t ^ (-1 - α) := by
  rcases hs.eq_or_lt with rfl | hs
  · simp [Real.zero_rpow hα.ne']
  have h : EqOn (fun t : ℝ => (1 - Real.exp (-(s * t))) * t ^ (-1 - α))
      (fun t => s ^ (1 + α) * ((1 - Real.exp (-(s * t))) * (s * t) ^ (-1 - α))) (Ioi 0) := by
    intro t ht
    have hst : s ^ (1 + α) * s ^ (-1 - α) = 1 := by
      rw [← Real.rpow_add hs, show 1 + α + (-1 - α) = 0 by ring, Real.rpow_zero]
    simp only [Real.mul_rpow hs.le (le_of_lt ht)]
    calc (1 - Real.exp (-(s * t))) * t ^ (-1 - α) =
        (s ^ (1 + α) * s ^ (-1 - α)) * ((1 - Real.exp (-(s * t))) * t ^ (-1 - α)) := by
          rw [hst, one_mul]
      _ = _ := by ring
  rw [setIntegral_congr_fun measurableSet_Ioi h, integral_const_mul,
    integral_comp_mul_left_Ioi (fun u => (1 - Real.exp (-u)) * u ^ (-1 - α)) 0 hs, mul_zero,
    smul_eq_mul, ← mul_assoc, ← Real.rpow_neg_one, ← Real.rpow_add hs,
    show 1 + α + -1 = α by ring]

/-- The hypotheses of `rpow_mul_integral` hold for `α = 1/2` and `s = 4`. -/
example : (4 : ℝ) ^ (1 / 2 : ℝ) * ∫ u in Ioi 0, (1 - Real.exp (-u)) * u ^ (-1 - (1 / 2 : ℝ)) =
    ∫ t in Ioi 0, (1 - Real.exp (-(4 * t))) * t ^ (-1 - (1 / 2 : ℝ)) :=
  rpow_mul_integral (by norm_num) (by norm_num)

end Transformer.KERPLE
