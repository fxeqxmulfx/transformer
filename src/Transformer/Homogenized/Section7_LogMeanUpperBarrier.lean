module

public import Transformer.Homogenized.Section7_LogisticOdeBounds

/-!
# A deterministic logarithmic upper barrier for the original mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the comparison following the mean drift estimate.
The fixed regularized original mean is `Z=m+a`. Its compensated
logarithm is `F=log Z-M`, whose derivative is the actual logarithmic drift.
The comparison field `1-Z+K/Z` decreases strictly for positive `Z`.

A bounded original noise `|M|≤r` therefore costs only `2r` in
the logarithmic comparison. The deterministic upward logistic shift
solves `A'=A+K`, with `A(0)=a`. Its relative logarithmic derivative
is a supersolution. This keeps the stochastic error outside the
exponential deterministic envelope in the source's displayed rate.
The source's direct Gronwall bound amplifies its additive noise.
This logarithmic comparison supplies the displayed unamplified scale.
The scalar comparison inputs are explicit here; the Gaussian drift
and original solution supply them in the subsequent application.
-/

@[expose] public section

open Set

namespace Transformer.Homogenized

/-- The deterministic upward envelope for the regularized logistic reference.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`,
the deterministic comparison after the actual drift mismatch estimate. -/
noncomputable def meanLogUpperShift (a K t : ℝ) : ℝ := (a + K) * Real.exp t - K

/-- The actual deterministic shift starts at the fixed regularization.
Source: arXiv:2604.01978v1, §7, the original mean comparison. -/
theorem meanLogUpperShift_zero (a K : ℝ) : meanLogUpperShift a K 0 = a := by
  simp only [meanLogUpperShift, Real.exp_zero, mul_one, add_sub_cancel_right]

/-- The upward envelope stays above its initial regularization.
Source: arXiv:2604.01978v1, §7, the mean comparison's deterministic error envelope. -/
theorem le_meanLogUpperShift {a K t : ℝ} (ha : 0 ≤ a) (hK : 0 ≤ K) (ht : 0 ≤ t) :
    a ≤ meanLogUpperShift a K t := by
  have h := mul_le_mul_of_nonneg_left (Real.one_le_exp_iff.mpr ht) (add_nonneg ha hK)
  dsimp only [meanLogUpperShift]
  linarith

/-- The shift is nonnegative on every source horizon.
Source: arXiv:2604.01978v1, §7, the deterministic mean comparison. -/
theorem meanLogUpperShift_nonneg {a K t : ℝ} (ha : 0 ≤ a) (hK : 0 ≤ K) (ht : 0 ≤ t) :
    0 ≤ meanLogUpperShift a K t := ha.trans (le_meanLogUpperShift ha hK ht)

/-- The actual upward envelope solves the inhomogeneous error equation.
Source: arXiv:2604.01978v1, §7, the deterministic exponential error envelope. -/
theorem hasDerivAt_meanLogUpperShift (a K t : ℝ) :
    HasDerivAt (meanLogUpperShift a K) (meanLogUpperShift a K t + K) t := by
  change HasDerivAt (fun s : ℝ => (a + K) * Real.exp s - K) (((a + K) * Real.exp t - K) + K) t
  rw [sub_add_cancel]
  exact ((Real.hasDerivAt_exp t).const_mul (a + K)).sub_const K

/-- The actual shift is controlled by its positive deterministic exponential envelope.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`. -/
theorem meanLogUpperShift_le_exp {a K : ℝ} (hK : 0 ≤ K) (t : ℝ) :
    meanLogUpperShift a K t ≤ (a + K) * Real.exp t := by
  dsimp only [meanLogUpperShift]
  linarith

/-- The actual shift is continuous at every real time.
Source: arXiv:2604.01978v1, §7, the deterministic error envelope. -/
theorem continuous_meanLogUpperShift (a K : ℝ) : Continuous (meanLogUpperShift a K) :=
  (Real.continuous_exp.const_mul (a + K)).sub continuous_const

/-- The proved small-temperature relation `K≤6a` gives a universal error envelope.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`. -/
theorem meanLogUpperShift_le_seven {a K : ℝ} (hK : 0 ≤ K) (hKa : K ≤ 6 * a) (t : ℝ) :
    meanLogUpperShift a K t ≤ 7 * a * Real.exp t := by
  apply (meanLogUpperShift_le_exp hK t).trans
  exact mul_le_mul_of_nonneg_right (by linarith) (Real.exp_pos t).le

/-- The original logistic reference plus the actual upward shift has the required
logarithmic supersolution derivative. Source: arXiv:2604.01978v1, §7,
`eq:logistic_u_smallbeta` and the mean drift comparison. -/
theorem logistic_meanLog_shift_supersolution {u A K : ℝ} (hu : 0 ≤ u) (hA : 0 ≤ A)
    (hpos : 0 < u + A) :
    1 - (u + A) + K / (u + A) ≤ (u * (1 - u) + A + K) / (u + A) := by
  have h := div_le_div_of_nonneg_right (logistic_shift_supersolution_scalar u A K hu hA) hpos.le
  apply le_trans ?_ h
  apply le_of_eq
  field_simp

/-- Strict boundary comparison controls a positive original logarithmic mean
without multiplying its bounded compensated noise by an exponential in time.
Source: arXiv:2604.01978v1, §7, the comparison needed for
`eq:smallbeta_main_bound`. The actual drift and reference bounds remain
explicit scalar inputs in this deterministic lemma. -/
theorem log_mean_upper_barrier (T r K : ℝ) (hr : 0 ≤ r) (hK : 0 ≤ K)
    (z F D v v' : ℝ → ℝ) (hz : ∀ t ∈ Icc (0 : ℝ) T, 0 < z t)
    (hF : ContinuousOn F (Icc (0 : ℝ) T))
    (hF' : ∀ t ∈ Ico (0 : ℝ) T, HasDerivWithinAt F (D t) (Ici t) t)
    (hv : ContinuousOn v (Icc (0 : ℝ) T)) (hvpos : ∀ t ∈ Icc (0 : ℝ) T, 0 < v t)
    (hv' : ∀ t ∈ Ico (0 : ℝ) T, HasDerivWithinAt v (v' t) (Ici t) t)
    (hinit : F 0 ≤ Real.log (v 0))
    (hnoise : ∀ t ∈ Icc (0 : ℝ) T, |Real.log (z t) - F t| ≤ r)
    (hD : ∀ t ∈ Ico (0 : ℝ) T, D t ≤ 1 - z t + K / z t)
    (hvp : ∀ t ∈ Ico (0 : ℝ) T, 1 - v t + K / v t ≤ v' t / v t) :
    ∀ t ∈ Icc (0 : ℝ) T, Real.log (z t) ≤ Real.log (v t) + 2 * r := by
  have hlog := hv.log (fun t ht => (hvpos t ht).ne')
  have hbound (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : F t ≤ Real.log (v t) + r := by
    apply le_of_forall_pos_le_add
    intro η hη
    apply image_le_of_deriv_right_lt_deriv_boundary' hF hF'
      (B := fun s => Real.log (v s) + r + η) (B' := fun s => v' s / v s)
      (by linarith) ((hlog.add_const r).add_const η)
      (fun s hs => (((hv' s hs).log (hvpos s ⟨hs.1, hs.2.le⟩).ne').add_const r).add_const η)
      (fun s hs heq => ?_) ht
    have hscc : s ∈ Icc (0 : ℝ) T := ⟨hs.1, hs.2.le⟩
    have hn := (abs_le.mp (hnoise s hscc)).1
    have hlt : v s < z s := (Real.log_lt_log_iff (hvpos s hscc) (hz s hscc)).mp (by linarith)
    exact (hD s hs).trans_lt
      ((regularized_logistic_upper_strictAnti K (v s) (z s) hK (hvpos s hscc) hlt).trans_le (hvp s hs))
  intro t ht
  have hn := (abs_le.mp (hnoise t ht)).2
  have hb := hbound t ht
  linarith

/-- Positive shifts and a genuine deterministic logistic equilibrium realize
the envelope and supersolution hypotheses simultaneously. -/
example : (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 / 8 ∧ (0 : ℝ) ≤ 1 ∧
    (1 / 8 : ℝ) ≤ 6 * (1 / 4) ∧
    (0 : ℝ) ≤ meanLogUpperShift (1 / 4) (1 / 8) 1 ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ meanLogUpperShift (1 / 4) (1 / 8) 1 ∧
    0 < 1 + meanLogUpperShift (1 / 4) (1 / 8) 1 := by
  have hp : 0 ≤ meanLogUpperShift (1 / 4) (1 / 8) 1 :=
    meanLogUpperShift_nonneg (by norm_num) (by norm_num) (by norm_num)
  exact ⟨by norm_num, by norm_num, zero_le_one, by norm_num, hp, zero_le_one, hp, by linarith⟩

/-- A true noise-free positive equilibrium realizes every strict boundary
comparison premise, including the actual compensator derivative and initial value. -/
example : ∃ z F D v v' : ℝ → ℝ,
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, 0 < z t) ∧ ContinuousOn F (Icc (0 : ℝ) 1) ∧
    (∀ t ∈ Ico (0 : ℝ) 1, HasDerivWithinAt F (D t) (Ici t) t) ∧
    ContinuousOn v (Icc (0 : ℝ) 1) ∧ (∀ t ∈ Icc (0 : ℝ) 1, 0 < v t) ∧
    (∀ t ∈ Ico (0 : ℝ) 1, HasDerivWithinAt v (v' t) (Ici t) t) ∧
    F 0 ≤ Real.log (v 0) ∧ (∀ t ∈ Icc (0 : ℝ) 1, |Real.log (z t) - F t| ≤ 0) ∧
    (∀ t ∈ Ico (0 : ℝ) 1, D t ≤ 1 - z t + 0 / z t) ∧
    (∀ t ∈ Ico (0 : ℝ) 1, 1 - v t + 0 / v t ≤ v' t / v t) := by
  refine ⟨fun _ => 1, fun _ => 0, fun _ => 0, fun _ => 1, fun _ => 0,
    le_rfl, le_rfl, fun t ht => one_pos, continuousOn_const,
    fun t ht => (hasDerivAt_const t (0 : ℝ)).hasDerivWithinAt, continuousOn_const,
    fun t ht => one_pos, fun t ht => (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt, ?_, ?_, ?_, ?_⟩
  all_goals simp

end Transformer.Homogenized
