import Transformer.Homogenized.Section7_MartingaleVariance
import Transformer.Homogenized.Section7_ContinuousSubmartingaleMaximal
import Transformer.Homogenized.Section7_MartingaleTimeShift
import Mathlib.MeasureTheory.OuterMeasure.BorelCantelli

/-!
# Excursions between consecutive integer times vanish almost surely

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence.
Translate the actual martingale increment and filtration to each unit
interval. Conditional Jensen and the proved continuous maximal estimate
control its crossing probability by its original terminal variance.
The preceding telescoping variance budget makes the sum of these
crossing probabilities finite for every positive threshold.

The first Borel-Cantelli lemma then makes large excursions occur only
finitely often. Applying it to the countable positive thresholds
1/(k+1) produces one probability-one event covering every threshold.
The crossing events are actual measurable events of the original paths.
Only the original finite-horizon martingale, continuity, and uniform
bound are premises. No independence or excursion estimate is assumed.
-/

open Real MeasureTheory Set Filter
open scoped Topology ENNReal
namespace Transformer.Homogenized

/-- The actual continuous unit-interval crossing event is measurable.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem measurableSet_martingale_unit_excursion
    {Ω : Type*} {m : MeasurableSpace Ω} (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hm : ∀ t, 0 ≤ t → StronglyMeasurable[ℱ t] (M t))
    (hc : ∀ ω, ContinuousOn (fun t => M t ω) (Ici 0)) (n : ℕ) (a : ℝ) :
    MeasurableSet {ω | ∃ t ∈ Icc (0 : ℝ) 1, a < (M ((n : ℝ) + t) ω - M n ω) ^ 2} := by
  apply measurableSet_continuous_time_crossing
    (fun t ω => (M ((n : ℝ) + t) ω - M n ω) ^ 2) 1 one_pos
  · intro t ht
    have hnt : 0 ≤ (n : ℝ) + t := add_nonneg (Nat.cast_nonneg n) ht.1
    exact ((((hm _ hnt).mono (ℱ.le _)).sub
      ((hm n (Nat.cast_nonneg n)).mono (ℱ.le n))).pow 2).measurable
  · intro ω
    exact ((continuousOn_timeShift ((n : ℝ) + 1) n 1
      ⟨Nat.cast_nonneg n, by linarith⟩ le_rfl (fun t => M t ω)
      ((hc ω).mono (fun t ht => ht.1))).sub continuousOn_const).pow 2

/-- The actual unit-interval crossing probability is bounded by the terminal variance.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem martingale_unit_excursion_bound
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M)
    (hc : ∀ ω, ContinuousOn (fun t => M t ω) (Ici 0))
    (C : ℝ) (hb : ∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ C) (n : ℕ) (a : NNReal) :
    (a : ENNReal) * P {ω | ∃ t ∈ Icc (0 : ℝ) 1, (a : ℝ) < (M ((n : ℝ) + t) ω - M n ω) ^ 2} ≤
      ENNReal.ofReal (∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P) := by
  let R : ℝ → Ω → ℝ := fun t ω => M ((n : ℝ) + t) ω - M n ω
  let G := timeShiftFiltration ℱ n
  have hN : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
  have hm : IsMartingaleOn 1 G P R := isMartingaleOn_timeShift_increment P ℱ
    ((n : ℝ) + 1) M (hM _ (by positivity)) n 1 ⟨hN, by linarith⟩ le_rfl
  have hi (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) : Integrable (fun ω => R t ω ^ 2) P := by
    exact integrable_sq_martingale_increment P ℱ ((n : ℝ) + 1) M (hM _ (by positivity)) C
      (fun s hs => hb s hs.1) n ((n : ℝ) + t) ⟨hN, by linarith⟩
      ⟨by linarith [ht.1], by linarith [ht.2]⟩
  apply nonnegative_submartingale_maximal_continuous P G (fun t ω => R t ω ^ 2) 1 one_pos
    (fun t ht => (hm.1 t ht).pow 2) hi
    (fun s hs t ht hst => condExp_sq_le_of_isMartingaleOn hm hi s t hs ht hst)
    (fun t ht ω => sq_nonneg _) _ a
  intro ω
  exact ((continuousOn_timeShift ((n : ℝ) + 1) n 1 ⟨hN, by linarith⟩ le_rfl
    (fun t => M t ω) ((hc ω).mono (fun t ht => ht.1))).sub continuousOn_const).pow 2

/-- For every positive threshold, actual unit-interval crossing probabilities have finite sum.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem tsum_martingale_unit_excursion_ne_top
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M)
    (hc : ∀ ω, ContinuousOn (fun t => M t ω) (Ici 0))
    (C : ℝ) (hb : ∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ C) (a : NNReal) (ha : 0 < a) :
    (∑' n : ℕ, P {ω | ∃ t ∈ Icc (0 : ℝ) 1,
      (a : ℝ) < (M ((n : ℝ) + t) ω - M n ω) ^ 2}) ≠ ⊤ := by
  let E : ℕ → Set Ω := fun n => {ω | ∃ t ∈ Icc (0 : ℝ) 1,
    (a : ℝ) < (M ((n : ℝ) + t) ω - M n ω) ^ 2}
  have ha0 : (a : ENNReal) ≠ 0 := ENNReal.coe_ne_zero.mpr ha.ne'
  have hineq (n : ℕ) : P (E n) ≤
      ENNReal.ofReal (∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P) / (a : ENNReal) := by
    apply (ENNReal.le_div_iff_mul_le (.inl ha0) (.inl ENNReal.coe_ne_top)).mpr
    simpa only [mul_comm] using martingale_unit_excursion_bound P ℱ M hM hc C hb n a
  have hs : (∑' n, P (E n)) ≤
      (∑' n : ℕ, ENNReal.ofReal (∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P)) / (a : ENNReal) := by
    calc
      _ ≤ ∑' n : ℕ, ENNReal.ofReal (∫ ω, (M ((n : ℝ) + 1) ω - M n ω) ^ 2 ∂P) / (a : ENNReal) :=
        ENNReal.tsum_le_tsum hineq
      _ = _ := by simp only [div_eq_mul_inv, ENNReal.tsum_mul_right]
  exact ne_top_of_le_ne_top
    (ENNReal.div_ne_top (tsum_ofReal_integral_sq_nat_martingale_increments_ne_top P ℱ M hM C hb) ha0) hs

/-- Large actual unit-interval excursions occur only finitely often almost surely.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem ae_eventually_martingale_unit_excursion_le
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M)
    (hc : ∀ ω, ContinuousOn (fun t => M t ω) (Ici 0))
    (C : ℝ) (hb : ∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ C) (r : NNReal) (hr : 0 < r) :
    ∀ᵐ ω ∂P, ∀ᶠ n : ℕ in atTop, ∀ t ∈ Icc (0 : ℝ) 1,
      |M ((n : ℝ) + t) ω - M n ω| ≤ r := by
  let a : NNReal := r ^ 2
  let E : ℕ → Set Ω := fun n => {ω | ∃ t ∈ Icc (0 : ℝ) 1,
    (a : ℝ) < (M ((n : ℝ) + t) ω - M n ω) ^ 2}
  have hfinite : (∑' n, P (E n)) ≠ ⊤ :=
    tsum_martingale_unit_excursion_ne_top P ℱ M hM hc C hb a (pow_pos hr 2)
  filter_upwards [ae_eventually_notMem hfinite] with ω hω
  filter_upwards [hω] with n hn
  intro t ht
  have hsmall : (M ((n : ℝ) + t) ω - M n ω) ^ 2 ≤ (r : ℝ) ^ 2 :=
    le_of_not_gt (fun h => hn ⟨t, ht, h⟩)
  exact (sq_le_sq₀ (abs_nonneg _) r.coe_nonneg).mp (by simpa only [sq_abs] using hsmall)

/-- One probability-one event controls all actual unit-interval excursions at every scale.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem ae_eventually_small_martingale_unit_excursions
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M)
    (hc : ∀ ω, ContinuousOn (fun t => M t ω) (Ici 0))
    (C : ℝ) (hb : ∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ C) :
    ∀ᵐ ω ∂P, ∀ k : ℕ, ∀ᶠ n : ℕ in atTop, ∀ t ∈ Icc (0 : ℝ) 1,
      |M ((n : ℝ) + t) ω - M n ω| ≤ 1 / ((k : ℝ) + 1) := by
  apply ae_all_iff.mpr
  intro k
  exact ae_eventually_martingale_unit_excursion_le P ℱ M hM hc C hb
    ⟨1 / ((k : ℝ) + 1), by positivity⟩ (by change (0 : ℝ) < 1 / ((k : ℝ) + 1); positivity)

/-- The sine of a Gaussian coordinate supplies the full original martingale,
continuity, adaptation, uniform bound, and positive threshold premises. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ)) (M : ℝ → ℝ → ℝ),
    (∀ T, 0 ≤ T → IsMartingaleOn T ℱ (ProbabilityTheory.gaussianReal 0 1) M) ∧
      (∀ ω, ContinuousOn (fun t => M t ω) (Ici 0)) ∧
      (∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ 1) ∧
      (∀ t, 0 ≤ t → StronglyMeasurable[ℱ t] (M t)) ∧
      (0 : NNReal) < 1 ∧ M 0 (Real.pi / 2) = 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  have hm : StronglyMeasurable Real.sin := Real.continuous_sin.stronglyMeasurable
  have hi := Integrable.of_bound (μ := ProbabilityTheory.gaussianReal 0 1) hm.aestronglyMeasurable 1
    (ae_of_all _ (fun ω => by simpa only [Real.norm_eq_abs] using Real.abs_sin_le_one ω))
  exact ⟨ℱ, fun _ => Real.sin, fun T hT => isMartingaleOn_timeConstant _ ℱ T Real.sin hm hi,
    fun ω => continuousOn_const, fun t ht ω => Real.abs_sin_le_one ω, fun t ht => hm,
    one_pos, Real.sin_pi_div_two⟩

end Transformer.Homogenized
