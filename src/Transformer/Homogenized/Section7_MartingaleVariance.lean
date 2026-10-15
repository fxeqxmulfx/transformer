import Transformer.Homogenized.Section7_BoundedMartingaleNatLimit
import Transformer.Homogenized.Section7_WeightedMartingale

/-!
# The actual variance budget of bounded martingale increments

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence.
The actual increment is orthogonal to its measurable left endpoint.
Expanding its square therefore identifies its expectation with the
increase in the original second moment. All squares and products used
in that expansion are proved integrable from the actual process bounds.

For successive integer intervals these exact increases telescope.
The uniform original bound gives a common finite budget for every
partial sum of the expected squared increments, hence also for their
nonnegative extended-real infinite sum. This is the summability needed
to control continuous-time excursions by the preceding maximal bound.
-/

open Real MeasureTheory Set Filter
open scoped Topology ENNReal BigOperators
namespace Transformer.Homogenized

/-- Actual bounded measurable squares are integrable under the original finite law.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem integrable_sq_of_bounded_stronglyMeasurable
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsFiniteMeasure P]
    (f : Ω → ℝ) (hf : StronglyMeasurable f) (C : ℝ) (hb : ∀ ω, |f ω| ≤ C) :
    Integrable (fun ω => f ω ^ 2) P := by
  apply Integrable.of_bound (hf.pow 2).aestronglyMeasurable (C ^ 2)
  apply ae_of_all
  intro ω
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
  change f ω ^ 2 ≤ C ^ 2
  simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg _) (hb ω) 2

/-- The actual increment variance is exactly the increase in the original second moment.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem integral_sq_martingale_increment
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (M : ℝ → Ω → ℝ) (hM : IsMartingaleOn T ℱ P M)
    (C : ℝ) (hb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |M t ω| ≤ C)
    (s t : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (ht : t ∈ Icc (0 : ℝ) T) (hst : s ≤ t) :
    (∫ ω, (M t ω - M s ω) ^ 2 ∂P) = (∫ ω, M t ω ^ 2 ∂P) - ∫ ω, M s ω ^ 2 ∂P := by
  have his := integrable_sq_of_bounded_stronglyMeasurable P (M s)
    ((hM.1 s hs).mono (ℱ.le s)) C (hb s hs)
  have hit := integrable_sq_of_bounded_stronglyMeasurable P (M t)
    ((hM.1 t ht).mono (ℱ.le t)) C (hb t ht)
  have hip : Integrable (fun ω => M s ω * (M t ω - M s ω)) P :=
    ((hM.2.1 t ht).sub (hM.2.1 s hs)).bdd_mul
      ((hM.1 s hs).mono (ℱ.le s)).aestronglyMeasurable
      (ae_of_all P (fun ω => by simpa only [Real.norm_eq_abs] using hb s hs ω))
  have hprod := integral_mul_martingale_increment_eq_zero P ℱ T M hM s t hs ht hst
    (M s) (hM.1 s hs) C (hb s hs)
  have hid : Integrable (fun ω => M t ω ^ 2 - M s ω ^ 2) P := hit.sub his
  have hi2p : Integrable (fun ω => 2 * (M s ω * (M t ω - M s ω))) P := hip.const_mul 2
  calc
    _ = ∫ ω, (M t ω ^ 2 - M s ω ^ 2) - 2 * (M s ω * (M t ω - M s ω)) ∂P := by
      apply integral_congr_ae
      exact ae_of_all P (fun ω => by ring)
    _ = _ := by rw [integral_sub hid hi2p, integral_sub hit his,
      integral_const_mul, hprod, mul_zero, sub_zero]

/-- The square of the actual bounded martingale increment is integrable.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem integrable_sq_martingale_increment
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (M : ℝ → Ω → ℝ) (hM : IsMartingaleOn T ℱ P M)
    (C : ℝ) (hb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |M t ω| ≤ C)
    (s t : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (ht : t ∈ Icc (0 : ℝ) T) :
    Integrable (fun ω => (M t ω - M s ω) ^ 2) P := by
  apply integrable_sq_of_bounded_stronglyMeasurable P _
    (((hM.1 t ht).mono (ℱ.le t)).sub ((hM.1 s hs).mono (ℱ.le s))) (2 * C)
  intro ω
  change |M t ω - M s ω| ≤ 2 * C
  have hts := abs_le.mp (hb t ht ω)
  have hss := abs_le.mp (hb s hs ω)
  exact abs_le.mpr ⟨by linarith [hts.1, hss.2], by linarith [hts.2, hss.1]⟩

/-- Original second moments increase because the actual increment variance is nonnegative.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem integral_sq_martingale_mono
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (M : ℝ → Ω → ℝ) (hM : IsMartingaleOn T ℱ P M)
    (C : ℝ) (hb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |M t ω| ≤ C)
    (s t : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (ht : t ∈ Icc (0 : ℝ) T) (hst : s ≤ t) :
    (∫ ω, M s ω ^ 2 ∂P) ≤ ∫ ω, M t ω ^ 2 ∂P := by
  have hv := integral_sq_martingale_increment P ℱ T M hM C hb s t hs ht hst
  have hp : 0 ≤ ∫ ω, (M t ω - M s ω) ^ 2 ∂P :=
    integral_nonneg_of_ae (ae_of_all P (fun ω => sq_nonneg _))
  linarith

/-- Actual unit-increment variances telescope to a uniform finite second-moment budget.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem sum_integral_sq_nat_martingale_increments_le
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M)
    (C : ℝ) (hb : ∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ C) (N : ℕ) :
    (∑ n ∈ Finset.range N, ∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P) ≤
      C ^ 2 * P.real univ := by
  have hdiff (n : ℕ) : (∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P) =
      (∫ ω, M ((n + 1 : ℕ) : ℝ) ω ^ 2 ∂P) - ∫ ω, M n ω ^ 2 ∂P := by
    simpa only [Nat.cast_add, Nat.cast_one] using integral_sq_martingale_increment P ℱ
      ((n : ℝ) + 1) M (hM _ (by positivity)) C (fun t ht => hb t ht.1)
      n ((n : ℝ) + 1) ⟨Nat.cast_nonneg n, by linarith⟩ ⟨by positivity, le_rfl⟩ (by linarith)
  have htel : (∑ n ∈ Finset.range N, ∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P) =
      (∫ ω, M N ω ^ 2 ∂P) - ∫ ω, M 0 ω ^ 2 ∂P := by
    simp_rw [hdiff]
    simpa only [Nat.cast_zero] using Finset.sum_range_sub (fun n : ℕ => ∫ ω, M n ω ^ 2 ∂P) N
  have hm : StronglyMeasurable (M N) :=
    ((hM N (Nat.cast_nonneg N)).1 N ⟨Nat.cast_nonneg N, le_rfl⟩).mono (ℱ.le N)
  have hi := integrable_sq_of_bounded_stronglyMeasurable P (M N) hm C (hb N (Nat.cast_nonneg N))
  have hp (ω : Ω) : M N ω ^ 2 ≤ C ^ 2 := by
    simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg _) (hb N (Nat.cast_nonneg N) ω) 2
  have hbound := integral_mono_ae hi (integrable_const (C ^ 2)) (ae_of_all P hp)
  have h0 : 0 ≤ ∫ ω, M 0 ω ^ 2 ∂P := integral_nonneg_of_ae (ae_of_all P (fun ω => sq_nonneg _))
  rw [htel]
  simp only [integral_const, smul_eq_mul] at hbound
  nlinarith

/-- The nonnegative sum of all actual unit-increment variances is finite.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem tsum_ofReal_integral_sq_nat_martingale_increments_ne_top
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M)
    (C : ℝ) (hb : ∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ C) :
    (∑' n : ℕ, ENNReal.ofReal (∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P)) ≠ ⊤ := by
  have hs : (∑' n : ℕ, ENNReal.ofReal (∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P)) ≤
      ENNReal.ofReal (C ^ 2 * P.real univ) := by
    apply ENNReal.tsum_le_of_sum_range_le
    intro N
    rw [← ENNReal.ofReal_sum_of_nonneg (fun n hn =>
      integral_nonneg_of_ae (ae_of_all P (fun ω => sq_nonneg _)))]
    exact ENNReal.ofReal_le_ofReal (sum_integral_sq_nat_martingale_increments_le P ℱ M hM C hb N)
  exact ne_top_of_le_ne_top ENNReal.ofReal_ne_top hs

/-- A nonconstant Gaussian-coordinate martingale realizes all bounds, square
integrability, and finite-horizon premises at two distinct times. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ)) (M : ℝ → ℝ → ℝ),
    (∀ T, 0 ≤ T → IsMartingaleOn T ℱ (ProbabilityTheory.gaussianReal 0 1) M) ∧
      (∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ 1) ∧ StronglyMeasurable (M 0) ∧
      (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (0 : ℝ) ≤ 1 ∧
      M 0 (Real.pi / 2) = 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  have hm : StronglyMeasurable Real.sin := Real.continuous_sin.stronglyMeasurable
  have hi := Integrable.of_bound (μ := ProbabilityTheory.gaussianReal 0 1) hm.aestronglyMeasurable 1
    (ae_of_all _ (fun ω => by simpa only [Real.norm_eq_abs] using Real.abs_sin_le_one ω))
  exact ⟨ℱ, fun _ => Real.sin, fun T hT => isMartingaleOn_timeConstant _ ℱ T Real.sin hm hi,
    fun t ht ω => Real.abs_sin_le_one ω, hm, by norm_num, by norm_num, zero_le_one, Real.sin_pi_div_two⟩

end Transformer.Homogenized
