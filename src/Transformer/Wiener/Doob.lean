/-
# Doob's `L²` inequalities for right-continuous martingales

For a martingale `(M_t)_{t ≥ 0}` on a finite measure space whose paths are
almost surely right-continuous (continuous paths are:
`ae_continuousWithinAt_Ici_of_continuous`), and `T ≥ 0`:

* `𝔼[(sup_{t ≤ T} |M_t|)²] ≤ 4 𝔼[M_T²]` (`lintegral_iSup_enorm_sq_le`);
* `ε² ℙ(sup_{t ≤ T} |M_t| ≥ ε) ≤ 𝔼[M_T²]` (`sq_mul_measure_le_iSup_enorm`;
  `sq_mul_measureReal_le` with a real supremum and a Bochner integral).

The supremum is measurable up to a null set (`aemeasurable_iSup_enorm`).  On the
dyadic grid of level `n` both inequalities are those of `DoobFinite` for the
submartingale `|M|`; the grid maxima increase to the supremum
(`iSup_enorm_eq_iSup_gridMax`), and monotone convergence, of integrals and of
measures, concludes.

Reference: D. Revuz, M. Yor, *Continuous martingales and Brownian motion*,
Thm II.1.7, `p = 2`.
-/

import Transformer.Wiener.DoobGrid

open MeasureTheory Filter Set
open scoped ENNReal NNReal Topology

namespace Transformer.Wiener

variable {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {ℱ : Filtration ℝ≥0 m}
  {M : ℝ≥0 → Ω → ℝ}

/-- Almost surely continuous paths are almost surely right-continuous. -/
theorem ae_continuousWithinAt_Ici_of_continuous (hc : ∀ᵐ ω ∂P, Continuous fun t => M t ω) :
    ∀ᵐ ω ∂P, ∀ t, ContinuousWithinAt (fun s => M s ω) (Ici t) t :=
  hc.mono fun _ hω _ => hω.continuousWithinAt

/-- The hypothesis of `ae_continuousWithinAt_Ici_of_continuous` holds for a
constant process. -/
example : ∀ᵐ _ : ℝ ∂(Measure.dirac 0), Continuous fun _ : ℝ≥0 => (1 : ℝ) :=
  ae_of_all _ fun _ => continuous_const

variable [IsFiniteMeasure P]

/-- **Doob's `L²` inequality**: `𝔼[(sup_{t ≤ T} |M_t|)²] ≤ 4 𝔼[M_T²]` for a
martingale with almost surely right-continuous paths.  Revuz–Yor, Thm II.1.7,
`p = 2`.  On the dyadic grid of level `n` it is `lintegral_runningMax_sq_le`,
`𝔼[M_s²] ≤ 𝔼[M_T²]` for `s ≤ T` giving its integrability hypothesis. -/
theorem lintegral_iSup_enorm_sq_le (hM : Martingale M ℱ P)
    (hc : ∀ᵐ ω ∂P, ∀ t, ContinuousWithinAt (fun s => M s ω) (Ici t) t) (T : ℝ≥0) :
    ∫⁻ ω, (⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ) ^ 2 ∂P ≤ 4 * ∫⁻ ω, ‖M T ω‖ₑ ^ 2 ∂P := by
  have hmeas : ∀ t, Measurable (M t) := fun t => ((hM.1 t).mono (ℱ.le t)).measurable
  by_cases hfin : ∫⁻ ω, ‖M T ω‖ₑ ^ 2 ∂P = ∞
  · simp [hfin]
  calc ∫⁻ ω, (⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ) ^ 2 ∂P
      = ∫⁻ ω, ⨆ n, ENNReal.ofReal (gridMax M T n ω) ^ 2 ∂P :=
        lintegral_congr_ae (hc.mono fun ω hω => by
          dsimp only
          rw [iSup_enorm_eq_iSup_gridMax hω T, ENNReal.iSup_pow])
    _ = ⨆ n, ∫⁻ ω, ENNReal.ofReal (gridMax M T n ω) ^ 2 ∂P :=
        lintegral_iSup (fun n => (measurable_gridMax hmeas T n).ennreal_ofReal.pow_const 2)
          fun i j hij ω => by
            dsimp only
            gcongr
            exact gridMax_mono M T ω hij
    _ ≤ 4 * ∫⁻ ω, ‖M T ω‖ₑ ^ 2 ∂P := iSup_le fun n => by
        have h := lintegral_runningMax_sq_le (submartingale_abs_dyadic hM T n)
          (fun _ _ => abs_nonneg _) (2 ^ n) fun k hk => by
            simp_rw [← Real.enorm_eq_ofReal_abs]
            exact ne_top_of_le_ne_top hfin (lintegral_enorm_sq_le_of_le hM (dyadicTime_le hk))
        simpa only [gridMax, ← Real.enorm_eq_ofReal_abs, dyadicTime_two_pow] using h

/-- **Doob's weak `L²` inequality**: `ε² ℙ(sup_{t ≤ T} |M_t| ≥ ε) ≤ 𝔼[M_T²]` for
a martingale with almost surely right-continuous paths.  Revuz–Yor,
Thm II.1.7, `p = 2`.  For `0 < δ < ε` the event lies, up to a null set, in the
increasing union of the grid events `{gridMax M T n ≥ δ}`, each of which
`ofReal_sq_mul_measure_le` bounds; then `δ → ε`. -/
theorem sq_mul_measure_le_iSup_enorm (hM : Martingale M ℱ P)
    (hc : ∀ᵐ ω ∂P, ∀ t, ContinuousWithinAt (fun s => M s ω) (Ici t) t) (T : ℝ≥0) (ε : ℝ) :
    ENNReal.ofReal ε ^ 2 * P {ω | ENNReal.ofReal ε ≤ ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ} ≤
      ∫⁻ ω, ‖M T ω‖ₑ ^ 2 ∂P := by
  rcases le_or_gt ε 0 with hε | hε
  · simp [ENNReal.ofReal_of_nonpos hε]
  set A := {ω | ENNReal.ofReal ε ≤ ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ}
  have hδ : ∀ δ ∈ Ioo 0 ε, ENNReal.ofReal δ ^ 2 * P A ≤ ∫⁻ ω, ‖M T ω‖ₑ ^ 2 ∂P := by
    rintro δ ⟨-, hδε⟩
    have hmono : Monotone fun n => {ω | δ ≤ gridMax M T n ω} :=
      fun i j hij ω hω => le_trans hω (gridMax_mono M T ω hij)
    have hA : P A ≤ P (⋃ n, {ω | δ ≤ gridMax M T n ω}) :=
      measure_mono_ae (hc.mono fun ω hω (hωA : ω ∈ A) => by
        have h : ENNReal.ofReal δ < ⨆ n, ENNReal.ofReal (gridMax M T n ω) :=
          ((ENNReal.ofReal_lt_ofReal_iff hε).mpr hδε).trans_le
            ((iSup_enorm_eq_iSup_gridMax hω T) ▸ hωA)
        obtain ⟨n, hn⟩ := lt_iSup_iff.mp h
        exact mem_iUnion.mpr ⟨n, (ENNReal.ofReal_lt_ofReal_iff'.mp hn).1.le⟩)
    calc ENNReal.ofReal δ ^ 2 * P A
        ≤ ENNReal.ofReal δ ^ 2 * P (⋃ n, {ω | δ ≤ gridMax M T n ω}) := by gcongr
      _ = ⨆ n, ENNReal.ofReal δ ^ 2 * P {ω | δ ≤ gridMax M T n ω} := by
        rw [hmono.measure_iUnion, ENNReal.mul_iSup]
      _ ≤ _ := iSup_le fun n => by
        simpa only [gridMax, ← Real.enorm_eq_ofReal_abs, dyadicTime_two_pow] using
          ofReal_sq_mul_measure_le (submartingale_abs_dyadic hM T n)
            (fun _ _ => abs_nonneg _) δ (2 ^ n)
  have hlim : Tendsto (fun δ => ENNReal.ofReal δ ^ 2 * P A) (𝓝[<] ε)
      (𝓝 (ENNReal.ofReal ε ^ 2 * P A)) :=
    ENNReal.Tendsto.mul_const ((((ENNReal.continuous_pow 2).comp
      ENNReal.continuous_ofReal).tendsto ε).mono_left nhdsWithin_le_nhds)
      (Or.inr (measure_ne_top P A))
  exact le_of_tendsto hlim (eventually_of_mem (Ioo_mem_nhdsLT hε) hδ)

/-- **Doob's weak `L²` inequality with real quantities**: for `M_T` square
integrable and `ε ≥ 0`, `ε² ℙ(sup_{t ≤ T} |M_t| ≥ ε) ≤ 𝔼[M_T²]`, the supremum
taken in `ℝ`.  Where `|M|` is unbounded on `[0, T]` the real supremum is `0` by
convention; its event lies in that of `sq_mul_measure_le_iSup_enorm` anyway. -/
theorem sq_mul_measureReal_le (hM : Martingale M ℱ P)
    (hc : ∀ᵐ ω ∂P, ∀ t, ContinuousWithinAt (fun s => M s ω) (Ici t) t) {T : ℝ≥0}
    (hT : MemLp (M T) 2 P) {ε : ℝ} (hε : 0 ≤ ε) :
    ε ^ 2 * P.real {ω | ε ≤ ⨆ t ∈ Icc 0 T, |M t ω|} ≤ ∫ ω, M T ω ^ 2 ∂P := by
  have hsub : {ω | ε ≤ ⨆ t ∈ Icc 0 T, |M t ω|} ⊆
      {ω | ENNReal.ofReal ε ≤ ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ} := fun ω (hω : ε ≤ _) => by
    by_contra hlt
    have hlt' : ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ < ENNReal.ofReal ε := not_le.mp hlt
    refine (ENNReal.toReal_lt_of_lt_ofReal hlt').not_ge (hω.trans ?_)
    refine Real.iSup_le (fun t => Real.iSup_le (fun ht => ?_) ENNReal.toReal_nonneg)
      ENNReal.toReal_nonneg
    rw [← Real.norm_eq_abs, ← toReal_enorm]
    exact ENNReal.toReal_mono hlt'.ne_top
      (le_iSup₂ (f := fun t (_ : t ∈ Icc 0 T) => ‖M t ω‖ₑ) t ht)
  have hB : ∫⁻ ω, ‖M T ω‖ₑ ^ 2 ∂P = ENNReal.ofReal (∫ ω, M T ω ^ 2 ∂P) := by
    rw [ofReal_integral_eq_lintegral_ofReal hT.integrable_sq (ae_of_all _ fun _ => sq_nonneg _)]
    refine lintegral_congr fun ω => ?_
    rw [Real.enorm_eq_ofReal_abs, ← ENNReal.ofReal_pow (abs_nonneg _), sq_abs]
  have h := sq_mul_measure_le_iSup_enorm hM hc T ε
  rw [hB] at h
  calc ε ^ 2 * P.real {ω | ε ≤ ⨆ t ∈ Icc 0 T, |M t ω|}
      ≤ ε ^ 2 * P.real {ω | ENNReal.ofReal ε ≤ ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ} :=
        mul_le_mul_of_nonneg_left (measureReal_mono hsub) (sq_nonneg ε)
    _ = (ENNReal.ofReal ε ^ 2 * P {ω | ENNReal.ofReal ε ≤ ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ}).toReal := by
        rw [ENNReal.toReal_mul, ENNReal.toReal_pow, ENNReal.toReal_ofReal hε]
        rfl
    _ ≤ (ENNReal.ofReal (∫ ω, M T ω ^ 2 ∂P)).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top h
    _ = ∫ ω, M T ω ^ 2 ∂P := ENNReal.toReal_ofReal (integral_nonneg fun _ => sq_nonneg _)

/-- The hypotheses of `lintegral_iSup_enorm_sq_le`, `sq_mul_measure_le_iSup_enorm`
and `sq_mul_measureReal_le` hold for a constant process under a Dirac mass. -/
example : Martingale (fun _ _ => (1 : ℝ))
      (Filtration.const (ι := ℝ≥0) (inferInstance : MeasurableSpace ℝ) le_rfl)
      (Measure.dirac (0 : ℝ)) ∧
    (∀ᵐ _ : ℝ ∂(Measure.dirac 0), ∀ t : ℝ≥0,
      ContinuousWithinAt (fun _ : ℝ≥0 => (1 : ℝ)) (Ici t) t) ∧
    MemLp (fun _ : ℝ => (1 : ℝ)) 2 (Measure.dirac 0) ∧ (0 : ℝ) ≤ 1 :=
  ⟨martingale_const _ _ 1, ae_of_all _ fun _ _ => continuousWithinAt_const, memLp_const 1,
    zero_le_one⟩

end Transformer.Wiener
