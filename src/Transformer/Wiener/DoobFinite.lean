/-
# Doob's `L²` inequalities on a finite grid

For a non-negative submartingale `(f_k)_{k ∈ ℕ}` with running maximum
`f*_n = max_{k ≤ n} f_k` (`runningMax`):

* `ε² ℙ(f*_n ≥ ε) ≤ 𝔼[f_n²]` (`ofReal_sq_mul_measure_le`);
* `𝔼[(f*_n)²] ≤ 4 𝔼[f_n²]` (`lintegral_runningMax_sq_le`), Doob's `L²`
  inequality.

Both rest on Mathlib's `maximal_ineq`, `ε ℙ(f*_n ≥ ε) ≤ 𝔼[f_n; f*_n ≥ ε]`, whose
docstring leaves Doob's `L^p` inequality to future work.  The first adds
`2εx ≤ x² + ε²` on `{f*_n ≥ ε}`.  The second integrates `maximal_ineq` against
`2ε dε` by the layer-cake formula, once for `ℙ` and once for `f_n · ℙ`:
`𝔼[(f*_n)²] ≤ 2 𝔼[f_n f*_n] ≤ 2 𝔼[f_n²] + 𝔼[(f*_n)²] / 2`, where
`𝔼[(f*_n)²] ≤ Σ_{k ≤ n} 𝔼[f_k²] < ∞` is assumed.  Expectations are lower
Lebesgue integrals.

Reference: D. Revuz, M. Yor, *Continuous martingales and Brownian motion*,
Thm II.1.7, at a finite set of times.
-/

import Mathlib.Probability.Martingale.OptionalStopping
import Mathlib.MeasureTheory.Integral.Layercake
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

open MeasureTheory Set
open scoped ENNReal

namespace Transformer.Wiener

variable {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {𝒢 : Filtration ℕ m}
  {f : ℕ → Ω → ℝ}

/-- The running maximum `max_{k ≤ n} f_k(ω)`, as in Mathlib's `maximal_ineq`. -/
noncomputable def runningMax (f : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  (Finset.range (n + 1)).sup' Finset.nonempty_range_add_one fun k => f k ω

theorem le_runningMax {k n : ℕ} (hk : k ≤ n) (ω : Ω) : f k ω ≤ runningMax f n ω :=
  Finset.le_sup' (fun k => f k ω) (Finset.mem_range.mpr (Nat.lt_succ_of_le hk))

/-- The hypothesis of `le_runningMax` holds for `k = n`. -/
example : (3 : ℕ) ≤ 3 := le_rfl

theorem exists_runningMax_eq (f : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) :
    ∃ k ≤ n, runningMax f n ω = f k ω := by
  obtain ⟨k, hk, h⟩ := Finset.exists_mem_eq_sup' Finset.nonempty_range_add_one fun k => f k ω
  exact ⟨k, Nat.lt_succ_iff.mp (Finset.mem_range.mp hk), h⟩

theorem measurable_runningMax (hf : ∀ k, Measurable (f k)) (n : ℕ) :
    Measurable (runningMax f n) := by
  have h := Finset.measurable_sup' (Finset.nonempty_range_add_one (n := n)) fun k _ => hf k
  convert h using 1
  funext ω
  rw [Finset.sup'_apply]
  rfl

/-- The hypothesis of `measurable_runningMax` holds for constant sequences. -/
example : ∀ _ : ℕ, Measurable fun _ : ℝ => (1 : ℝ) := fun _ => measurable_const

/-- `2ab ≤ a² + b²` in `ℝ≥0∞`, for real `a, b ≥ 0`. -/
theorem two_mul_ofReal_mul_ofReal_le {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    2 * ENNReal.ofReal a * ENNReal.ofReal b ≤ ENNReal.ofReal a ^ 2 + ENNReal.ofReal b ^ 2 := by
  rw [← ENNReal.ofReal_ofNat 2, ← ENNReal.ofReal_mul (by norm_num),
    ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_pow ha, ← ENNReal.ofReal_pow hb,
    ← ENNReal.ofReal_add (by positivity) (by positivity)]
  exact ENNReal.ofReal_le_ofReal (by nlinarith [sq_nonneg (a - b)])

/-- The hypotheses of `two_mul_ofReal_mul_ofReal_le` hold for `a = b = 1`. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

/-- **Doob's maximal inequality**, `ε ℙ(f*_n ≥ ε) ≤ 𝔼[f_n; f*_n ≥ ε]`, with a
lower Lebesgue integral: Mathlib's `maximal_ineq`. -/
theorem ofReal_mul_measure_le_setLIntegral [IsFiniteMeasure P] (hsub : Submartingale f 𝒢 P)
    (hnonneg : 0 ≤ f) (ε : ℝ) (n : ℕ) :
    ENNReal.ofReal ε * P {ω | ε ≤ runningMax f n ω} ≤
      ∫⁻ ω in {ω | ε ≤ runningMax f n ω}, ENNReal.ofReal (f n ω) ∂P := by
  rcases le_or_gt ε 0 with hε | hε
  · simp [ENNReal.ofReal_of_nonpos hε]
  have h := maximal_ineq hsub hnonneg (ε := ε.toNNReal) n
  rw [Real.coe_toNNReal _ hε.le, ofReal_integral_eq_lintegral_ofReal
    (hsub.2.2 n).integrableOn (ae_of_all _ fun ω => hnonneg n ω)] at h
  exact h

/-- **Doob's weak `L²` inequality on a finite grid**:
`ε² ℙ(f*_n ≥ ε) ≤ 𝔼[f_n²]` for a non-negative submartingale.  On
`A = {f*_n ≥ ε}`, `2ε² ℙ(A) ≤ 2ε 𝔼[f_n; A] ≤ 𝔼[f_n²; A] + ε² ℙ(A)`. -/
theorem ofReal_sq_mul_measure_le [IsFiniteMeasure P] (hsub : Submartingale f 𝒢 P)
    (hnonneg : 0 ≤ f) (ε : ℝ) (n : ℕ) :
    ENNReal.ofReal ε ^ 2 * P {ω | ε ≤ runningMax f n ω} ≤
      ∫⁻ ω, ENNReal.ofReal (f n ω) ^ 2 ∂P := by
  rcases le_or_gt ε 0 with hε | hε
  · simp [ENNReal.ofReal_of_nonpos hε]
  set A := {ω | ε ≤ runningMax f n ω}
  have h := ofReal_mul_measure_le_setLIntegral hsub hnonneg ε n
  have hc : ENNReal.ofReal ε ^ 2 * P A ≠ ∞ :=
    ENNReal.mul_ne_top (ENNReal.pow_ne_top ENNReal.ofReal_ne_top) (measure_ne_top P A)
  have h2 : 2 * ENNReal.ofReal ε ≠ ∞ := ENNReal.mul_ne_top (by norm_num) ENNReal.ofReal_ne_top
  have key : ENNReal.ofReal ε ^ 2 * P A + ENNReal.ofReal ε ^ 2 * P A ≤
      ∫⁻ ω in A, ENNReal.ofReal (f n ω) ^ 2 ∂P + ENNReal.ofReal ε ^ 2 * P A := by
    calc _ = 2 * ENNReal.ofReal ε * (ENNReal.ofReal ε * P A) := by ring
      _ ≤ 2 * ENNReal.ofReal ε * ∫⁻ ω in A, ENNReal.ofReal (f n ω) ∂P := by gcongr
      _ = ∫⁻ ω in A, 2 * ENNReal.ofReal ε * ENNReal.ofReal (f n ω) ∂P :=
        (lintegral_const_mul' _ _ h2).symm
      _ ≤ ∫⁻ ω in A, (ENNReal.ofReal (f n ω) ^ 2 + ENNReal.ofReal ε ^ 2) ∂P :=
        lintegral_mono fun ω =>
          (two_mul_ofReal_mul_ofReal_le hε.le (hnonneg n ω)).trans_eq (add_comm _ _)
      _ = _ := by rw [lintegral_add_right _ measurable_const, setLIntegral_const]
  exact (ENNReal.le_of_add_le_add_right hc key).trans (setLIntegral_le_lintegral _ _)

/-- The hypotheses of `ofReal_mul_measure_le_setLIntegral` and
`ofReal_sq_mul_measure_le` hold for the constant process `1` under a Dirac mass. -/
example : Submartingale (fun _ _ => (1 : ℝ))
      (Filtration.const (ι := ℕ) (inferInstance : MeasurableSpace ℝ) le_rfl)
      (Measure.dirac (0 : ℝ)) ∧ (0 : ℕ → ℝ → ℝ) ≤ fun _ _ => 1 :=
  ⟨(martingale_const _ _ 1).submartingale, fun _ _ => zero_le_one⟩

/-- **Doob's `L²` inequality on a finite grid**: `𝔼[(f*_n)²] ≤ 4 𝔼[f_n²]` for a
non-negative submartingale with `𝔼[f_k²] < ∞`, `k ≤ n`.  By the layer-cake
formula and `maximal_ineq`, `𝔼[(f*_n)²] = ∫₀^∞ 2ε ℙ(f*_n ≥ ε) dε ≤
2 ∫₀^∞ 𝔼[f_n; f*_n ≥ ε] dε = 2 𝔼[f_n f*_n]`, and `4xy ≤ 4x² + y²`. -/
theorem lintegral_runningMax_sq_le [IsFiniteMeasure P] (hsub : Submartingale f 𝒢 P)
    (hnonneg : 0 ≤ f) (n : ℕ) (hfin : ∀ k ≤ n, ∫⁻ ω, ENNReal.ofReal (f k ω) ^ 2 ∂P ≠ ∞) :
    ∫⁻ ω, ENNReal.ofReal (runningMax f n ω) ^ 2 ∂P ≤
      4 * ∫⁻ ω, ENNReal.ofReal (f n ω) ^ 2 ∂P := by
  have hmeas : ∀ k, Measurable (f k) := fun k => ((hsub.1 k).mono (𝒢.le k)).measurable
  have hSm : Measurable (runningMax f n) := measurable_runningMax hmeas n
  have hS0 : ∀ ω, 0 ≤ runningMax f n ω :=
    fun ω => (hnonneg 0 ω).trans (le_runningMax (Nat.zero_le n) ω)
  have hXm : Measurable fun ω => ENNReal.ofReal (f n ω) := (hmeas n).ennreal_ofReal
  have hA : ∫⁻ ω, ENNReal.ofReal (runningMax f n ω) ^ 2 ∂P ≠ ∞ := by
    refine ne_top_of_le_ne_top (ENNReal.sum_ne_top.mpr fun k hk =>
      hfin k (Nat.lt_succ_iff.mp (Finset.mem_range.mp hk))) ?_
    rw [← lintegral_finsetSum' _ fun k _ => ((hmeas k).ennreal_ofReal.pow_const 2).aemeasurable]
    refine lintegral_mono fun ω => ?_
    obtain ⟨k, hk, h⟩ := exists_runningMax_eq f n ω
    rw [h]
    exact Finset.single_le_sum (f := fun k => ENNReal.ofReal (f k ω) ^ 2) (fun _ _ => zero_le)
      (Finset.mem_range.mpr (Nat.lt_succ_of_le hk))
  have hcake : ∫⁻ ω, ENNReal.ofReal (runningMax f n ω) ^ 2 ∂P =
      ∫⁻ t in Ioi 0, P {ω | t ≤ runningMax f n ω} * ENNReal.ofReal (2 * t) := by
    rw [← lintegral_comp_eq_lintegral_meas_le_mul (g := fun t => 2 * t) P (ae_of_all _ hS0)
      hSm.aemeasurable (fun t _ => (by fun_prop : Continuous fun t : ℝ => 2 * t).intervalIntegrable 0 t)
      ((ae_restrict_iff' measurableSet_Ioi).mpr (ae_of_all _ fun t (ht : 0 < t) => by positivity))]
    refine lintegral_congr fun ω => ?_
    rw [intervalIntegral.integral_const_mul, integral_id, ← ENNReal.ofReal_pow (hS0 ω)]
    congr 1
    ring
  have hstep : ∀ t ∈ Ioi (0 : ℝ), P {ω | t ≤ runningMax f n ω} * ENNReal.ofReal (2 * t) ≤
      2 * (P.withDensity fun ω => ENNReal.ofReal (f n ω)) {ω | t ≤ runningMax f n ω} := by
    intro t _
    calc P {ω | t ≤ runningMax f n ω} * ENNReal.ofReal (2 * t)
        = 2 * (ENNReal.ofReal t * P {ω | t ≤ runningMax f n ω}) := by
          rw [ENNReal.ofReal_mul zero_le_two, ENNReal.ofReal_ofNat]
          ring
      _ ≤ 2 * ∫⁻ ω in {ω | t ≤ runningMax f n ω}, ENNReal.ofReal (f n ω) ∂P := by
          gcongr
          exact ofReal_mul_measure_le_setLIntegral hsub hnonneg t n
      _ = _ := by rw [withDensity_apply _ (measurableSet_le measurable_const hSm)]
  have hcake2 : ∫⁻ t in Ioi 0,
      (P.withDensity fun ω => ENNReal.ofReal (f n ω)) {ω | t ≤ runningMax f n ω} =
      ∫⁻ ω, ENNReal.ofReal (f n ω) * ENNReal.ofReal (runningMax f n ω) ∂P := by
    rw [← lintegral_eq_lintegral_meas_le _ (ae_of_all _ hS0) hSm.aemeasurable,
      lintegral_withDensity_eq_lintegral_mul _ hXm hSm.ennreal_ofReal]
    rfl
  have hmain : ∫⁻ ω, ENNReal.ofReal (runningMax f n ω) ^ 2 ∂P ≤
      2 * ∫⁻ ω, ENNReal.ofReal (f n ω) * ENNReal.ofReal (runningMax f n ω) ∂P := by
    rw [hcake, ← hcake2, ← lintegral_const_mul' _ _ (by norm_num : (2 : ℝ≥0∞) ≠ ∞)]
    exact setLIntegral_mono' measurableSet_Ioi hstep
  have hamgm : 2 * (2 * ∫⁻ ω, ENNReal.ofReal (f n ω) * ENNReal.ofReal (runningMax f n ω) ∂P) ≤
      4 * ∫⁻ ω, ENNReal.ofReal (f n ω) ^ 2 ∂P +
        ∫⁻ ω, ENNReal.ofReal (runningMax f n ω) ^ 2 ∂P := by
    rw [← mul_assoc, show (2 : ℝ≥0∞) * 2 = 4 by norm_num,
      ← lintegral_const_mul' _ _ (by norm_num : (4 : ℝ≥0∞) ≠ ∞),
      ← lintegral_const_mul' _ _ (by norm_num : (4 : ℝ≥0∞) ≠ ∞),
      ← lintegral_add_left ((hXm.pow_const 2).const_mul 4)]
    refine lintegral_mono fun ω => ?_
    have h := two_mul_ofReal_mul_ofReal_le (mul_nonneg zero_le_two (hnonneg n ω)) (hS0 ω)
    rw [ENNReal.ofReal_mul zero_le_two, ENNReal.ofReal_ofNat] at h
    calc 4 * (ENNReal.ofReal (f n ω) * ENNReal.ofReal (runningMax f n ω))
        = 2 * (2 * ENNReal.ofReal (f n ω)) * ENNReal.ofReal (runningMax f n ω) := by ring
      _ ≤ (2 * ENNReal.ofReal (f n ω)) ^ 2 + ENNReal.ofReal (runningMax f n ω) ^ 2 := h
      _ = 4 * ENNReal.ofReal (f n ω) ^ 2 + ENNReal.ofReal (runningMax f n ω) ^ 2 := by ring
  refine ENNReal.le_of_add_le_add_right hA ?_
  calc _ = 2 * ∫⁻ ω, ENNReal.ofReal (runningMax f n ω) ^ 2 ∂P := (two_mul _).symm
    _ ≤ _ := by gcongr
    _ ≤ _ := hamgm

/-- The hypotheses of `lintegral_runningMax_sq_le` hold for the constant process
`1` under a Dirac mass. -/
example : Submartingale (fun _ _ => (1 : ℝ))
      (Filtration.const (ι := ℕ) (inferInstance : MeasurableSpace ℝ) le_rfl)
      (Measure.dirac (0 : ℝ)) ∧
    ∀ k ≤ 3, ∫⁻ _, ENNReal.ofReal (1 : ℝ) ^ 2 ∂(Measure.dirac (0 : ℝ)) ≠ ∞ :=
  ⟨(martingale_const _ _ 1).submartingale, fun _ _ => by simp⟩

end Transformer.Wiener
