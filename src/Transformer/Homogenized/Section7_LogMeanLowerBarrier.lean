module

public import Transformer.Homogenized.Section7_LogMeanUpperBarrier

/-!
# A deterministic logarithmic lower barrier for the original mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the comparison following the mean drift estimate.
The original regularized mean `Z=m+a` has logarithmic drift at least
`1-Z-c-K/Z`. Its compensated logarithm `F=log Z-M` is differentiable.
A strict boundary comparison with the original positive logistic
reference controls `log u-log Z` by `2r+(c+2K/u₀)t`.

The comparison only needs this bound when its endpoint is below `log 2`.
At a prospective contact point the original bounded noise forces
`u/2<Z<u`. The positive reference lower bound then controls `K/Z`
by `2K/u₀`. The complementary parameter regimes are treated by
the original bounded range and relative logistic growth later.
All drift and noise inputs are explicit deterministic hypotheses here.
-/

@[expose] public section

open Set MeasureTheory

namespace Transformer.Homogenized

/-- Between half the positive reference lower bound and the reference itself,
the genuine logarithmic lower drift strictly exceeds the barrier derivative.
Source: arXiv:2604.01978v1, §7, the mean comparison for `eq:smallbeta_main_bound`. -/
theorem regularized_logistic_lower_contact (u₀ u z c K D : ℝ) (hu₀ : 0 < u₀)
    (hK : 0 ≤ K) (hzlo : u₀ / 2 < z) (hzhi : z < u)
    (hD : 1 - z - c - K / z ≤ D) : 1 - u - (c + 2 * K / u₀) < D := by
  have hp : 0 < u₀ / 2 := by positivity
  have hratio : K / z ≤ 2 * K / u₀ := by
    calc
      _ ≤ K / (u₀ / 2) := div_le_div_of_nonneg_left hK hp hzlo.le
      _ = _ := by field_simp
  linarith

/-- The original logarithmic lower comparison retains a bounded noise error
without multiplying it by the deterministic exponential growth envelope.
Source: arXiv:2604.01978v1, §7, the comparison needed for
`eq:smallbeta_main_bound`; actual Gaussian drift bounds are supplied later. -/
theorem log_mean_lower_barrier (T r c K u₀ : ℝ) (hr : 0 ≤ r) (hc : 0 ≤ c)
    (hK : 0 ≤ K) (hu₀ : 0 < u₀)
    (z F D u : ℝ → ℝ) (hz : ∀ t ∈ Icc (0 : ℝ) T, 0 < z t)
    (hF : ContinuousOn F (Icc (0 : ℝ) T))
    (hF' : ∀ t ∈ Ico (0 : ℝ) T, HasDerivWithinAt F (D t) (Ici t) t)
    (hu : ContinuousOn (fun t => Real.log (u t)) (Icc (0 : ℝ) T))
    (hu' : ∀ t ∈ Ico (0 : ℝ) T,
      HasDerivWithinAt (fun s => Real.log (u s)) (1 - u t) (Ici t) t)
    (hbase : ∀ t ∈ Icc (0 : ℝ) T, u₀ ≤ u t)
    (hinit : Real.log (u 0) ≤ F 0)
    (hnoise : ∀ t ∈ Icc (0 : ℝ) T, |Real.log (z t) - F t| ≤ r)
    (hD : ∀ t ∈ Ico (0 : ℝ) T, 1 - z t - c - K / z t ≤ D t)
    (hsmall : 2 * r + (c + 2 * K / u₀) * T < Real.log 2) :
    ∀ t ∈ Icc (0 : ℝ) T,
      Real.log (u t) - Real.log (z t) ≤ 2 * r + (c + 2 * K / u₀) * t := by
  let C := c + 2 * K / u₀
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hbound (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : -F t ≤ -Real.log (u t) + r + C * t := by
    apply le_of_forall_pos_le_add
    intro ε hε
    let η := min ε ((Real.log 2 - (2 * r + C * T)) / 2)
    have hη : 0 < η := lt_min hε (by dsimp [C]; linarith)
    have hηle : η ≤ ε := min_le_left _ _
    have hηsmall : 2 * r + C * T + η < Real.log 2 := by
      have hm : η ≤ (Real.log 2 - (2 * r + C * T)) / 2 := min_le_right _ _
      dsimp only [C] at hm ⊢
      linarith
    have hb : -F t ≤ -Real.log (u t) + r + C * t + η := by
      apply image_le_of_deriv_right_lt_deriv_boundary' hF.neg (fun s hs => (hF' s hs).neg)
        (B := fun s => -Real.log (u s) + r + C * s + η)
        (B' := fun s => -(1 - u s) + C) (by simpa using (show -F 0 ≤ -Real.log (u 0) + r + η by linarith))
        ((((hu.neg).add_const r).add (continuousOn_const.mul continuousOn_id)).add_const η)
        (fun s hs => by
          have hd : HasDerivWithinAt (fun t : ℝ => C * t) (C * 1) (Ici s) s :=
            ((hasDerivAt_id s).const_mul C).hasDerivWithinAt
          rw [mul_one] at hd
          exact ((((hu' s hs).neg).add_const r).add hd).add_const η)
        (fun s hs heq => ?_) ht
      have hscc : s ∈ Icc (0 : ℝ) T := ⟨hs.1, hs.2.le⟩
      have hnoise' := abs_le.mp (hnoise s hscc)
      have huPos := hu₀.trans_le (hbase s hscc)
      have hCt : 0 ≤ C * s := mul_nonneg hC hs.1
      have hCT := mul_le_mul_of_nonneg_left hs.2.le hC
      change -(F s) = -Real.log (u s) + r + C * s + η at heq
      have hloghi : Real.log (z s) < Real.log (u s) := by linarith [hnoise'.2]
      have hloglo : Real.log (u s / 2) < Real.log (z s) := by
        rw [Real.log_div huPos.ne' (by norm_num : (2 : ℝ) ≠ 0)]
        linarith [hnoise'.1]
      have hzhi : z s < u s := (Real.log_lt_log_iff (hz s hscc) huPos).mp hloghi
      have hzlo : u₀ / 2 < z s := by
        have hhalf := (Real.log_lt_log_iff (by positivity : 0 < u s / 2) (hz s hscc)).mp hloglo
        linarith [hbase s hscc]
      have hcontact := regularized_logistic_lower_contact u₀ (u s) (z s) c K (D s)
        hu₀ hK hzlo hzhi (hD s hs)
      dsimp only [C]
      linarith
    linarith
  intro t ht
  have hn := (abs_le.mp (hnoise t ht)).1
  have hb := hbound t ht
  dsimp only [C] at hb
  linarith

/-- The actual continuous drift primitive has the right derivative needed
by the deterministic logarithmic comparison, including time zero.
Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean`, after the logarithmic test. -/
theorem hasDerivWithinAt_continuous_timeIntegral (D : ℝ → ℝ) (T : ℝ)
    (hD : ContinuousOn D (Icc (0 : ℝ) T)) (t : ℝ) (ht : t ∈ Ico (0 : ℝ) T) :
    HasDerivWithinAt (fun s => ∫ r in (0 : ℝ)..s, D r) (D t) (Ici t) t := by
  have hi : IntervalIntegrable D volume 0 t := by
    apply ContinuousOn.intervalIntegrable
    rw [uIcc_of_le ht.1]
    exact hD.mono (fun s hs => ⟨hs.1, hs.2.trans ht.2.le⟩)
  have hm : StronglyMeasurableAtFilter D (nhdsWithin t (Ioi t)) volume :=
    _root_.AEStronglyMeasurable.stronglyMeasurableAtFilter_of_mem
      (hD.aestronglyMeasurable measurableSet_Icc) (Icc_mem_nhdsGT_of_mem ht)
  have hc := (hD t ⟨ht.1, ht.2.le⟩).mono_of_mem_nhdsWithin (Icc_mem_nhdsGT_of_mem ht)
  exact intervalIntegral.integral_hasDerivWithinAt_right hi hm hc

/-- A nonconstant actual continuous drift and an interior time realize
every primitive-derivative premise on a positive original horizon. -/
example : ContinuousOn (fun t : ℝ => 1 + t) (Icc (0 : ℝ) 1) ∧
    (1 / 2 : ℝ) ∈ Ico (0 : ℝ) 1 := by
  exact ⟨continuousOn_const.add continuousOn_id, by norm_num⟩

/-- Positive scalar reference, a strict intermediate original mean, and
a nonzero regularization error realize every lower contact hypothesis. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 8 ∧ (1 : ℝ) / 2 < 3 / 4 ∧
    (3 / 4 : ℝ) < 1 ∧ 1 - (3 / 4 : ℝ) - 1 / 4 - (1 / 8) / (3 / 4) ≤ 0 := by
  norm_num

/-- A genuine positive noise-free equilibrium realizes all lower logarithmic
comparison premises, with positive horizon inside the comparison regime. -/
example : ∃ z F D u : ℝ → ℝ,
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, 0 < z t) ∧ ContinuousOn F (Icc (0 : ℝ) 1) ∧
    (∀ t ∈ Ico (0 : ℝ) 1, HasDerivWithinAt F (D t) (Ici t) t) ∧
    ContinuousOn (fun t => Real.log (u t)) (Icc (0 : ℝ) 1) ∧
    (∀ t ∈ Ico (0 : ℝ) 1,
      HasDerivWithinAt (fun s => Real.log (u s)) (1 - u t) (Ici t) t) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, 1 ≤ u t) ∧ Real.log (u 0) ≤ F 0 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, |Real.log (z t) - F t| ≤ 0) ∧
    (∀ t ∈ Ico (0 : ℝ) 1, 1 - z t - 0 - 0 / z t ≤ D t) ∧
    2 * (0 : ℝ) + (0 + 2 * 0 / 1) * 1 < Real.log 2 := by
  refine ⟨fun _ => 1, fun _ => 0, fun _ => 0, fun _ => 1,
    le_rfl, le_rfl, le_rfl, one_pos, fun t ht => one_pos, continuousOn_const,
    fun t ht => (hasDerivAt_const t (0 : ℝ)).hasDerivWithinAt, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa only [Real.log_one] using (continuousOn_const : ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Icc 0 1))
  · intro t ht
    simpa only [Real.log_one, sub_self] using (hasDerivAt_const t (0 : ℝ)).hasDerivWithinAt (s := Ici t)
  · exact fun t ht => le_rfl
  · simp
  · simp
  · simp
  · simpa using Real.log_pos (by norm_num : (1 : ℝ) < 2)

end Transformer.Homogenized
