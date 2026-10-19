module

public import Transformer.Wiener.DoobFinite

/-!
# Dyadic grids for Doob's inequalities in continuous time

A process `(M_t)_{t ≥ 0}` sampled at the dyadic times `k T / 2ⁿ`, `k ≤ 2ⁿ`
(`dyadicTime`), has running maxima `gridMax M T n = max_{k ≤ 2ⁿ} |M(k T / 2ⁿ)|`
increasing in `n` (`gridMax_mono`). Along a right-continuous path their limit
is the supremum of `|M|` over `[0, T]` (`iSup_enorm_eq_iSup_gridMax`), every
`t ≤ T` being a limit from the right of dyadic times
(`exists_dyadicTime_tendsto`); so that supremum is measurable up to a null set
(`aemeasurable_iSup_enorm`). For a martingale, `|M|` on each grid is a
non-negative submartingale (`submartingale_abs_dyadic`), and `𝔼[M_s²] ≤ 𝔼[M_t²]`
for `s ≤ t` (`lintegral_enorm_sq_le_of_le`).

The grid passage follows `nonnegative_submartingale_maximal_continuous`
(`Transformer.Homogenized`, uniform grids of `[0, T] ⊂ ℝ`, continuous paths);
nested dyadic grids make the running maxima monotone in `n`.
-/

@[expose] public section

open MeasureTheory Filter Set
open scoped ENNReal NNReal Topology

namespace Transformer.Wiener

variable {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {ℱ : Filtration ℝ≥0 m}
  {M : ℝ≥0 → Ω → ℝ}

/-- The dyadic time `k T / 2ⁿ`. -/
noncomputable def dyadicTime (T : ℝ≥0) (n k : ℕ) : ℝ≥0 := T * k / 2 ^ n

theorem dyadicTime_mono (T : ℝ≥0) (n : ℕ) : Monotone (dyadicTime T n) := fun i j hij => by
  unfold dyadicTime
  gcongr

theorem dyadicTime_le {T : ℝ≥0} {n k : ℕ} (hk : k ≤ 2 ^ n) : dyadicTime T n k ≤ T := by
  unfold dyadicTime
  rw [div_le_iff₀ (by positivity)]
  gcongr
  exact_mod_cast hk

/-- The hypothesis of `dyadicTime_le` holds for `k = 2ⁿ`. -/
example : 2 ^ 3 ≤ 2 ^ 3 := le_rfl

theorem dyadicTime_two_pow (T : ℝ≥0) (n : ℕ) : dyadicTime T n (2 ^ n) = T := by
  unfold dyadicTime
  push_cast
  field_simp

theorem dyadicTime_two_mul (T : ℝ≥0) (n k : ℕ) :
    dyadicTime T (n + 1) (2 * k) = dyadicTime T n k := by
  unfold dyadicTime
  push_cast
  rw [pow_succ]
  field_simp

/-- **Every `t ≤ T` is a limit from the right of dyadic times**
`⌈2ⁿ t / T⌉ T / 2ⁿ ∈ [t, t + T / 2ⁿ]`. -/
theorem exists_dyadicTime_tendsto {T t : ℝ≥0} (htT : t ≤ T) :
    ∃ k : ℕ → ℕ, (∀ n, k n ≤ 2 ^ n) ∧
      Tendsto (fun n => dyadicTime T n (k n)) atTop (𝓝[≥] t) := by
  rcases eq_zero_or_pos T with rfl | hT
  · obtain rfl : t = 0 := nonpos_iff_eq_zero.mp htT
    refine ⟨fun _ => 0, fun _ => Nat.zero_le _, tendsto_nhdsWithin_iff.mpr ⟨?_, ?_⟩⟩
    · simp [dyadicTime]
    · exact Eventually.of_forall fun _ => mem_Ici.mpr zero_le
  have hT0 : T ≠ 0 := hT.ne'
  refine ⟨fun n => ⌈t / T * 2 ^ n⌉₊, fun n => Nat.ceil_le.mpr ?_, ?_⟩
  · calc t / T * 2 ^ n ≤ 1 * 2 ^ n := by gcongr; exact div_le_one_of_le₀ htT zero_le
      _ = ((2 ^ n : ℕ) : ℝ≥0) := by push_cast; ring
  have hlow : ∀ n : ℕ, t ≤ dyadicTime T n ⌈t / T * 2 ^ n⌉₊ := fun n => by
    unfold dyadicTime
    rw [le_div_iff₀ (by positivity)]
    calc t * 2 ^ n = T * (t / T * 2 ^ n) := by field_simp
      _ ≤ T * ⌈t / T * 2 ^ n⌉₊ := by gcongr; exact Nat.le_ceil _
  have hup : ∀ n : ℕ, dyadicTime T n ⌈t / T * 2 ^ n⌉₊ ≤ t + T * 2⁻¹ ^ n := fun n => by
    unfold dyadicTime
    rw [div_le_iff₀ (by positivity)]
    calc T * ⌈t / T * 2 ^ n⌉₊ ≤ T * (t / T * 2 ^ n + 1) := by
          gcongr; exact (Nat.ceil_lt_add_one zero_le).le
      _ = (t + T * 2⁻¹ ^ n) * 2 ^ n := by
          rw [mul_add, mul_one, add_mul, ← mul_assoc, mul_div_cancel₀ t hT0, mul_assoc T,
            ← mul_pow, inv_mul_cancel₀ two_ne_zero, one_pow, mul_one]
  refine tendsto_nhdsWithin_iff.mpr ⟨tendsto_of_tendsto_of_tendsto_of_le_of_le
    tendsto_const_nhds ?_ hlow hup, Eventually.of_forall hlow⟩
  simpa using tendsto_const_nhds.add
    ((NNReal.tendsto_pow_atTop_nhds_zero_of_lt_one (r := 2⁻¹) (by norm_num)).const_mul T)

/-- The hypothesis of `exists_dyadicTime_tendsto` holds for `t = 0`. -/
example : (0 : ℝ≥0) ≤ 1 := zero_le_one

/-- `ℱ` at the dyadic times of level `n`. -/
noncomputable def gridFiltration (ℱ : Filtration ℝ≥0 m) (T : ℝ≥0) (n : ℕ) : Filtration ℕ m :=
  ⟨fun k => ℱ (dyadicTime T n k), fun _ _ h => ℱ.mono (dyadicTime_mono T n h), fun _ => ℱ.le _⟩

/-- **`|M|` at the dyadic times of a martingale is a submartingale**, `M ⊔ -M`
being one. -/
theorem submartingale_abs_dyadic (hM : Martingale M ℱ P) (T : ℝ≥0) (n : ℕ) :
    Submartingale (fun k ω => |M (dyadicTime T n k) ω|) (gridFiltration ℱ T n) P := by
  have h := hM.submartingale.sup hM.neg.submartingale
  have habs : M ⊔ -M = fun t ω => |M t ω| := by
    funext t ω
    simp [abs_eq_max_neg]
  rw [habs] at h
  exact ⟨fun k => h.1 _, fun i j hij => h.2.1 _ _ (dyadicTime_mono T n hij), fun k => h.2.2 _⟩

/-- The hypothesis of `submartingale_abs_dyadic` holds for a constant process. -/
example : Martingale (fun _ _ => (1 : ℝ))
    (Filtration.const (ι := ℝ≥0) (inferInstance : MeasurableSpace ℝ) le_rfl)
    (Measure.dirac (0 : ℝ)) :=
  martingale_const _ _ 1

/-- `max_{k ≤ 2ⁿ} |M(k T / 2ⁿ)|`. -/
noncomputable def gridMax (M : ℝ≥0 → Ω → ℝ) (T : ℝ≥0) (n : ℕ) : Ω → ℝ :=
  runningMax (fun k ω => |M (dyadicTime T n k) ω|) (2 ^ n)

theorem gridMax_mono (M : ℝ≥0 → Ω → ℝ) (T : ℝ≥0) (ω : Ω) :
    Monotone fun n => gridMax M T n ω := by
  refine monotone_nat_of_le_succ fun n => ?_
  obtain ⟨k, hk, h⟩ := exists_runningMax_eq (fun k ω => |M (dyadicTime T n k) ω|) (2 ^ n) ω
  show runningMax _ _ ω ≤ runningMax _ _ ω
  rw [h, ← dyadicTime_two_mul T n k]
  exact le_runningMax (f := fun k ω => |M (dyadicTime T (n + 1) k) ω|)
    ((Nat.mul_le_mul_left 2 hk).trans_eq (pow_succ' 2 n).symm) ω

theorem ofReal_gridMax_le (M : ℝ≥0 → Ω → ℝ) (T : ℝ≥0) (n : ℕ) (ω : Ω) :
    ENNReal.ofReal (gridMax M T n ω) ≤ ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ := by
  obtain ⟨k, hk, h⟩ := exists_runningMax_eq (fun k ω => |M (dyadicTime T n k) ω|) (2 ^ n) ω
  rw [gridMax, h, ← Real.enorm_eq_ofReal_abs]
  exact le_iSup₂ (f := fun t (_ : t ∈ Icc 0 T) => ‖M t ω‖ₑ) (dyadicTime T n k)
    ⟨zero_le, dyadicTime_le hk⟩

/-- **Along a right-continuous path, the dyadic running maxima increase to the
supremum over `[0, T]`.** -/
theorem iSup_enorm_eq_iSup_gridMax {ω : Ω}
    (hω : ∀ t, ContinuousWithinAt (fun s => M s ω) (Ici t) t) (T : ℝ≥0) :
    ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ = ⨆ n, ENNReal.ofReal (gridMax M T n ω) := by
  refine le_antisymm (iSup₂_le fun t ht => ?_) (iSup_le fun n => ofReal_gridMax_le M T n ω)
  obtain ⟨k, hk, hlim⟩ := exists_dyadicTime_tendsto ht.2
  refine le_of_tendsto' ((hω t).enorm.tendsto.comp hlim) fun n => ?_
  calc ‖M (dyadicTime T n (k n)) ω‖ₑ = ENNReal.ofReal |M (dyadicTime T n (k n)) ω| :=
        Real.enorm_eq_ofReal_abs _
    _ ≤ ENNReal.ofReal (gridMax M T n ω) := ENNReal.ofReal_le_ofReal
        (le_runningMax (f := fun k ω => |M (dyadicTime T n k) ω|) (hk n) ω)
    _ ≤ _ := le_iSup (fun n => ENNReal.ofReal (gridMax M T n ω)) n

/-- The hypothesis of `iSup_enorm_eq_iSup_gridMax` holds for a constant path. -/
example : ∀ t : ℝ≥0, ContinuousWithinAt (fun _ : ℝ≥0 => (1 : ℝ)) (Ici t) t :=
  fun _ => continuousWithinAt_const

theorem measurable_gridMax (hM : ∀ t, Measurable (M t)) (T : ℝ≥0) (n : ℕ) :
    Measurable (gridMax M T n) :=
  measurable_runningMax (fun _ => continuous_abs.measurable.comp (hM _)) (2 ^ n)

/-- **The supremum of `|M|` over `[0, T]` is measurable up to a null set** when
almost every path is right-continuous: it is almost surely `⨆ₙ gridMax M T n`. -/
theorem aemeasurable_iSup_enorm (hM : ∀ t, Measurable (M t))
    (hc : ∀ᵐ ω ∂P, ∀ t, ContinuousWithinAt (fun s => M s ω) (Ici t) t) (T : ℝ≥0) :
    AEMeasurable (fun ω => ⨆ t ∈ Icc 0 T, ‖M t ω‖ₑ) P :=
  (Measurable.iSup fun n => (measurable_gridMax hM T n).ennreal_ofReal).aemeasurable.congr
    (hc.mono fun _ hω => (iSup_enorm_eq_iSup_gridMax hω T).symm)

/-- The hypotheses of `measurable_gridMax` and `aemeasurable_iSup_enorm` hold for
a constant process. -/
example : (∀ _ : ℝ≥0, Measurable fun _ : ℝ => (1 : ℝ)) ∧ ∀ᵐ _ : ℝ ∂(Measure.dirac 0),
    ∀ t : ℝ≥0, ContinuousWithinAt (fun _ : ℝ≥0 => (1 : ℝ)) (Ici t) t :=
  ⟨fun _ => measurable_const, ae_of_all _ fun _ _ => continuousWithinAt_const⟩

/-- **`𝔼[M_s²] ≤ 𝔼[M_t²]` for a martingale and `s ≤ t`**: `M_s = 𝔼[M_t | ℱ_s]`,
and conditional expectation contracts `L²`. -/
theorem lintegral_enorm_sq_le_of_le (hM : Martingale M ℱ P) {s t : ℝ≥0} (hst : s ≤ t) :
    ∫⁻ ω, ‖M s ω‖ₑ ^ 2 ∂P ≤ ∫⁻ ω, ‖M t ω‖ₑ ^ 2 ∂P := by
  have h : eLpNorm (M s) 2 P ≤ eLpNorm (M t) 2 P :=
    (eLpNorm_congr_ae (hM.2 s t hst)).symm.le.trans (eLpNorm_condExp_le_eLpNorm _ one_le_two)
  have hsq : ∀ u, ∫⁻ ω, ‖M u ω‖ₑ ^ 2 ∂P = eLpNorm (M u) 2 P ^ 2 := fun u => by
    rw [eLpNorm_eq_lintegral_rpow_enorm_toReal two_ne_zero ENNReal.ofNat_ne_top
      ((hM.1 u).mono (ℱ.le u)).aestronglyMeasurable, ← ENNReal.rpow_natCast,
      ← ENNReal.rpow_mul]
    norm_num
  rw [hsq, hsq]
  gcongr

/-- The hypothesis of `lintegral_enorm_sq_le_of_le` holds for a constant process. -/
example : Martingale (fun _ _ => (1 : ℝ))
    (Filtration.const (ι := ℝ≥0) (inferInstance : MeasurableSpace ℝ) le_rfl)
    (Measure.dirac (0 : ℝ)) ∧ (1 : ℝ≥0) ≤ 2 :=
  ⟨martingale_const _ _ 1, one_le_two⟩

end Transformer.Wiener
