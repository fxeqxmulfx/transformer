module

public import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
public import Transformer.KERPLE.Section3_Integral

/-!
# §3.2 Fact 3 and Corollary 1: powers and logarithms of CPD kernels

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §3.2.

* Fact 3 (Berg, Christensen and Ressel 1984; Proposition 4 of Schölkopf 2000): if
  `k̃ : X × X → (−∞, 0]` is CPD, so are `−(−k̃)^α` for `0 < α < 1` (`IsCPDKernel.neg_rpow`) and
  `−log(1 − k̃)` (`IsCPDKernel.neg_log_one_sub`). The first is a positive multiple of
  `∫_0^∞ (e^{t k̃} − 1) t^{−1−α} dt`, an integral of CPD kernels by Schoenberg's theorem; the second
  is the limit of the CPD kernels `(1 − (1 − k̃)^α) / α` as `α → 0⁺`.
* Corollary 1: on a real inner product space, `−a ‖x − x'‖^p` (`isCPDKernel_neg_mul_norm_rpow`)
  and `−b log(1 + a ‖x − x'‖^p)` (`isCPDKernel_neg_mul_log_one_add`) are CPD for `0 < p ≤ 2`; the
  corollary asks `a, b > 0`, and `a, b ≥ 0` suffice.
* The iterations after Corollary 1: `−a ‖x − x'‖^p − b log(1 + a ‖x − x'‖^p)`
  (`isCPDKernel_neg_power_sub_log`) and `−b log(1 + a ‖x − x'‖^p)^c` for `0 < c < 1`, read as
  `−b (log(1 + a ‖x − x'‖^p))^c` (`isCPDKernel_neg_mul_log_rpow`), are CPD.
* A CPD kernel pulled back along a map is CPD (`IsCPDKernel.comp`).
-/

@[expose] public section

namespace Transformer.KERPLE

open Finset Filter MeasureTheory Set Topology

variable {X : Type*}

/-- **Fact 3, powers.** If `k̃ : X × X → (−∞, 0]` is CPD, so is `−(−k̃)^α` for `0 < α < 1`. -/
theorem IsCPDKernel.neg_rpow {k : X → X → ℝ} (hk : IsCPDKernel k) (hle : ∀ x y, k x y ≤ 0)
    {α : ℝ} (hα : 0 < α) (hα1 : α < 1) : IsCPDKernel fun x y => -(-k x y) ^ α := by
  have hC := integral_one_sub_exp_rpow_pos hα hα1
  have hK : ∀ t ∈ Ioi (0 : ℝ), IsCPDKernel fun x y => t ^ (-1 - α) * (-1 + Real.exp (t * k x y)) :=
    fun t ht => ((hk.isPDKernel_exp_mul (le_of_lt ht)).isCPDKernel.add_const (-1)).smul
      (Real.rpow_nonneg (le_of_lt ht) _)
  have hfun : ∀ x y, (fun t : ℝ => t ^ (-1 - α) * (-1 + Real.exp (t * k x y))) =
      fun t => -((1 - Real.exp (-(-k x y * t))) * t ^ (-1 - α)) := fun x y => by
    funext t
    rw [show -(-k x y * t) = t * k x y by ring]
    ring
  have hint : ∀ x y,
      IntegrableOn (fun t : ℝ => t ^ (-1 - α) * (-1 + Real.exp (t * k x y))) (Ioi 0) :=
    fun x y => by
    rw [hfun]
    exact (integrableOn_one_sub_exp_mul_rpow hα hα1 (neg_nonneg.mpr (hle x y))).neg
  refine ((IsCPDKernel.setIntegral measurableSet_Ioi hK hint).smul (inv_nonneg.mpr hC.le)).congr
    fun x y => ?_
  rw [hfun, integral_neg, ← rpow_mul_integral hα (neg_nonneg.mpr (hle x y))]
  field_simp

/-- The hypotheses of `IsCPDKernel.neg_rpow` hold for `−‖x − y‖²` on `ℝ` and `α = 1/2`. -/
example : IsCPDKernel fun x y : ℝ => -(-(-‖x - y‖ ^ 2)) ^ (1 / 2 : ℝ) :=
  isCPDKernel_neg_norm_sub_sq.neg_rpow (fun _ _ => neg_nonpos.mpr (by positivity))
    (by norm_num) (by norm_num)

/-- **Fact 3, logarithm.** If `k̃ : X × X → (−∞, 0]` is CPD, so is `−log(1 − k̃)`. -/
theorem IsCPDKernel.neg_log_one_sub {k : X → X → ℝ} (hk : IsCPDKernel k)
    (hle : ∀ x y, k x y ≤ 0) : IsCPDKernel fun x y => -Real.log (1 - k x y) := by
  refine IsCPDKernel.of_tendsto (l := 𝓝[>] (0 : ℝ))
    (K := fun α x y => α⁻¹ * (1 + -(-(-1 + k x y)) ^ α)) ?_ fun x y => ?_
  · filter_upwards [Ioo_mem_nhdsGT (zero_lt_one : (0 : ℝ) < 1)] with α hα
    exact (((hk.add_const (-1)).neg_rpow (fun x y => show -1 + k x y ≤ 0 by linarith [hle x y])
      hα.1 hα.2).add_const 1).smul (inv_nonneg.mpr hα.1.le)
  · have hb : 0 < 1 - k x y := by linarith [hle x y]
    have h := ((Real.hasStrictDerivAt_const_rpow hb 0).hasDerivAt.tendsto_slope_zero_right).neg
    simp only [zero_add, Real.rpow_zero, one_mul, smul_eq_mul] at h
    refine h.congr fun α => ?_
    rw [show -(-1 + k x y) = 1 - k x y by ring]
    ring

/-- The hypotheses of `IsCPDKernel.neg_log_one_sub` hold for `−‖x − y‖²` on `ℝ`. -/
example : IsCPDKernel fun x y : ℝ => -Real.log (1 - -‖x - y‖ ^ 2) :=
  isCPDKernel_neg_norm_sub_sq.neg_log_one_sub fun _ _ => neg_nonpos.mpr (by positivity)

/-- `‖v‖^p = (‖v‖²)^{p/2}`. -/
theorem norm_rpow_eq_sq_rpow {F : Type*} [SeminormedAddCommGroup F] (v : F) (p : ℝ) :
    ‖v‖ ^ p = (‖v‖ ^ 2) ^ (p / 2) := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul (norm_nonneg v)]
  congr 1
  push_cast
  ring

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Corollary 1 (a).** `−a ‖x − x'‖^p` is CPD for `0 < p ≤ 2` and `a ≥ 0`. -/
theorem isCPDKernel_neg_mul_norm_rpow {a p : ℝ} (ha : 0 ≤ a) (hp : 0 < p) (hp2 : p ≤ 2) :
    IsCPDKernel fun x y : E => -a * ‖x - y‖ ^ p := by
  have hsq : IsCPDKernel fun x y : E => -‖x - y‖ ^ p := by
    rcases hp2.lt_or_eq with hp2 | rfl
    · refine (isCPDKernel_neg_norm_sub_sq.neg_rpow (fun _ _ => neg_nonpos.mpr (by positivity))
        (by linarith : 0 < p / 2) (by linarith : p / 2 < 1)).congr fun x y => ?_
      rw [neg_neg, norm_rpow_eq_sq_rpow]
    · refine isCPDKernel_neg_norm_sub_sq.congr fun x y => ?_
      rw [Real.rpow_two]
  exact (hsq.smul ha).congr fun x y => by ring

/-- The hypotheses of `isCPDKernel_neg_mul_norm_rpow` hold for `a = 2` and `p = 1`. -/
example : IsCPDKernel fun x y : ℝ => -2 * ‖x - y‖ ^ (1 : ℝ) :=
  isCPDKernel_neg_mul_norm_rpow zero_le_two one_pos one_le_two

/-- **Corollary 1 (b).** `−b log(1 + a ‖x − x'‖^p)` is CPD for `0 < p ≤ 2` and `a, b ≥ 0`. -/
theorem isCPDKernel_neg_mul_log_one_add {a b p : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hp : 0 < p)
    (hp2 : p ≤ 2) : IsCPDKernel fun x y : E => -b * Real.log (1 + a * ‖x - y‖ ^ p) := by
  have h := (isCPDKernel_neg_mul_norm_rpow (E := E) ha hp hp2).neg_log_one_sub fun x y => by
    have := Real.rpow_nonneg (norm_nonneg (x - y)) p
    nlinarith
  exact (h.smul hb).congr fun x y => by
    rw [show 1 - -a * ‖x - y‖ ^ p = 1 + a * ‖x - y‖ ^ p by ring]
    ring

/-- The hypotheses of `isCPDKernel_neg_mul_log_one_add` hold for `a = b = 1` and `p = 2`. -/
example : IsCPDKernel fun x y : ℝ => -1 * Real.log (1 + 1 * ‖x - y‖ ^ (2 : ℝ)) :=
  isCPDKernel_neg_mul_log_one_add zero_le_one zero_le_one two_pos le_rfl

/-- **§3.2, after Corollary 1.** `−a ‖x − x'‖^p − b log(1 + a ‖x − x'‖^p)` is CPD for
`0 < p ≤ 2` and `a, b ≥ 0` (Fact 2 on Corollary 1). -/
theorem isCPDKernel_neg_power_sub_log {a b p : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hp : 0 < p)
    (hp2 : p ≤ 2) :
    IsCPDKernel fun x y : E => -a * ‖x - y‖ ^ p - b * Real.log (1 + a * ‖x - y‖ ^ p) :=
  ((isCPDKernel_neg_mul_norm_rpow ha hp hp2).add
    (isCPDKernel_neg_mul_log_one_add ha hb hp hp2)).congr fun x y => by ring

/-- The hypotheses of `isCPDKernel_neg_power_sub_log` hold for `a = b = p = 1`. -/
example : IsCPDKernel fun x y : ℝ =>
    -1 * ‖x - y‖ ^ (1 : ℝ) - 1 * Real.log (1 + 1 * ‖x - y‖ ^ (1 : ℝ)) :=
  isCPDKernel_neg_power_sub_log zero_le_one zero_le_one one_pos one_le_two

/-- **§3.2, after Corollary 1.** `−b (log(1 + a ‖x − x'‖^p))^c` is CPD for `0 < p ≤ 2`, `a, b ≥ 0`
and `0 < c < 1` (Fact 3 on Corollary 1 (b)); the paper writes `−b·log(1 + a‖x − x'‖^p)^c`. -/
theorem isCPDKernel_neg_mul_log_rpow {a b c p : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 < c)
    (hc1 : c < 1) (hp : 0 < p) (hp2 : p ≤ 2) :
    IsCPDKernel fun x y : E => -b * Real.log (1 + a * ‖x - y‖ ^ p) ^ c := by
  have h := (isCPDKernel_neg_mul_log_one_add (E := E) ha zero_le_one hp hp2).neg_rpow
    (fun x y => by
      have := Real.log_nonneg (show 1 ≤ 1 + a * ‖x - y‖ ^ p by
        have := Real.rpow_nonneg (norm_nonneg (x - y)) p
        nlinarith)
      linarith) hc hc1
  exact (h.smul hb).congr fun x y => by rw [neg_mul, one_mul, neg_neg]; ring

/-- The hypotheses of `isCPDKernel_neg_mul_log_rpow` hold for `a = b = p = 1` and `c = 1/2`. -/
example : IsCPDKernel fun x y : ℝ => -1 * Real.log (1 + 1 * ‖x - y‖ ^ (1 : ℝ)) ^ (1 / 2 : ℝ) :=
  isCPDKernel_neg_mul_log_rpow zero_le_one zero_le_one (by norm_num) (by norm_num) one_pos
    one_le_two

/-- A CPD kernel pulled back along a map is CPD. -/
theorem IsCPDKernel.comp {Y : Type*} {k : X → X → ℝ} (hk : IsCPDKernel k) (f : Y → X) :
    IsCPDKernel fun y y' => k (f y) (f y') :=
  ⟨fun y y' => hk.1 (f y) (f y'), fun N y c hc => hk.2 N (f ∘ y) c hc⟩

/-- A PD kernel pulled back along a map is PD. -/
theorem IsPDKernel.comp {Y : Type*} {k : X → X → ℝ} (hk : IsPDKernel k) (f : Y → X) :
    IsPDKernel fun y y' => k (f y) (f y') :=
  ⟨fun y y' => hk.1 (f y) (f y'), fun N y c => hk.2 N (f ∘ y) c⟩

end Transformer.KERPLE
