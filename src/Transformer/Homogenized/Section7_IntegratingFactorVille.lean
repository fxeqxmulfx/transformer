import Transformer.Homogenized.Section7_IntegratingFactor
import Transformer.Homogenized.Section7_WeightedSupermartingale

/-!
# The maximal inequality with the actual exponential integral

ArXiv:2604.01978v1, §7, the maximal inequality after
`eq:Gronwall_E_selfcontained`, and `def:ito.formula`.
For a nonnegative observable Y with compensated drift H, the
actual factor exp(-∫₀ᵗa) gives the continuous maximal bound
whenever H ≤ aY. Adaptedness, continuity, differentiation,
uniform domination, and the conditional inequalities for the
weighted process are all supplied by the preceding proofs.

The premise is the original compensated martingale identity,
not a martingale or tail assumption about the weighted process.
The event uses the original time integral, whose crossing event
is also proved measurable. The temporary endpoint extension
used for differentiation is removed by the proved equality
with the original integral throughout the finite horizon.
-/

open Set MeasureTheory
namespace Transformer.Homogenized

/-- Original coefficient and observable bounds control the actual product.
Source: arXiv:2604.01978v1, §7, exponential compensation. -/
theorem abs_exp_neg_timeIntegral_mul_le (T : ℝ) (a : ℝ → ℝ) (Y : ℝ → ℝ) (A BY : ℝ) (hA : 0 ≤ A)
    (ha : ∀ x ∈ Icc (0 : ℝ) T, |a x| ≤ A) (hY : ∀ x ∈ Icc (0 : ℝ) T, |Y x| ≤ BY)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    |Real.exp (-(∫ x in (0 : ℝ)..t, a x)) * Y t| ≤ Real.exp (A * T) * BY := by
  rw [← intervalIntegratingFactor_eq T a t ht, abs_mul]
  exact mul_le_mul (abs_intervalIntegratingFactor_le T a A hA ha t ht) (hY t ht)
    (abs_nonneg _) (Real.exp_nonneg _)

/-- The continuous crossing event of the original exponential product
is measurable. Source: arXiv:2604.01978v1, §7, the maximal inequality. -/
theorem measurableSet_exp_neg_timeIntegral_mul_crossing {Ω : Type*} {m : MeasurableSpace Ω} (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (a Y : ℝ → Ω → ℝ)
    (ha : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (a t))
    (hY : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (Y t))
    (hca : ∀ ω, ContinuousOn (fun t => a t ω) (Icc (0 : ℝ) T))
    (hcY : ∀ ω, ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) T)) (r : ℝ) :
    MeasurableSet {ω | ∃ t ∈ Icc (0 : ℝ) T, r < Real.exp (-(∫ x in (0 : ℝ)..t, a x ω)) * Y t ω} := by
  let Z (t : ℝ) (ω : Ω) := intervalIntegratingFactor T (fun x => a x ω) t * Y t ω
  have hm (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : Measurable (Z t) := by
    have hsm : StronglyMeasurable[ℱ t] (Z t) :=
      (stronglyMeasurable_intervalIntegratingFactor ℱ T a ha hca t ht).mul (hY t ht)
    exact (hsm.mono (ℱ.le t)).measurable
  have hc (ω : Ω) : ContinuousOn (fun t => Z t ω) (Icc (0 : ℝ) T) :=
    (continuousOn_intervalIntegratingFactor T _ (hca ω)).mul (hcY ω)
  have hmeas := measurableSet_continuous_time_crossing Z T hT hm hc r
  have heq : {ω | ∃ t ∈ Icc (0 : ℝ) T, r < Z t ω} =
      {ω | ∃ t ∈ Icc (0 : ℝ) T, r < Real.exp (-(∫ x in (0 : ℝ)..t, a x ω)) * Y t ω} := by
    ext ω
    constructor <;> rintro ⟨t, ht, hrt⟩ <;> refine ⟨t, ht, ?_⟩
    · change r < intervalIntegratingFactor T (fun x => a x ω) t * Y t ω at hrt
      rwa [intervalIntegratingFactor_eq T _ t ht] at hrt
    · change r < intervalIntegratingFactor T (fun x => a x ω) t * Y t ω
      rwa [intervalIntegratingFactor_eq T _ t ht]
  rwa [heq] at hmeas

/-- The continuous maximal bound with the actual factor exp(-∫₀ᵗa),
derived from the original compensated martingale and H ≤ aY.
Source: arXiv:2604.01978v1, §7 and `def:ito.formula`. -/
theorem exp_neg_timeIntegral_mul_ville
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (hT : 0 < T) (a Y H : ℝ → Ω → ℝ)
    (ha : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (a t))
    (hY : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (Y t))
    (hc : ∀ ω, ContinuousOn (fun t => a t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) T))
    (hM : IsMartingaleOn T ℱ P (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω))
    (A BY BH : ℝ) (hA : 0 ≤ A) (hBH : 0 ≤ BH)
    (hab : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |a t ω| ≤ A)
    (hYb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |Y t ω| ≤ BY)
    (hHb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |H t ω| ≤ BH)
    (hpos : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, 0 ≤ Y t ω)
    (hdrift : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, H t ω ≤ a t ω * Y t ω)
    (r : ℝ) (hr : 0 < r) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T, r < Real.exp (-(∫ x in (0 : ℝ)..t, a x ω)) * Y t ω} ≤
      ENNReal.ofReal (∫ ω, Y 0 ω ∂P) / ENNReal.ofReal r := by
  let C (t : ℝ) (ω : Ω) := intervalIntegratingFactor T (fun x => a x ω) t
  let V (t : ℝ) (ω : Ω) := -a t ω * C t ω
  have hC (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (C t) :=
    stronglyMeasurable_intervalIntegratingFactor ℱ T a ha (fun ω => (hc ω).1) t ht
  have hCc (ω : Ω) : ContinuousOn (fun t => C t ω) (Icc (0 : ℝ) T) :=
    continuousOn_intervalIntegratingFactor T _ (hc ω).1
  have hCb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (ω : Ω) : |C t ω| ≤ Real.exp (A * T) :=
    abs_intervalIntegratingFactor_le T (fun x => a x ω) A hA (fun x hx => hab x hx ω) t ht
  have hVb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (ω : Ω) : |V t ω| ≤ A * Real.exp (A * T) := by
    change |-a t ω * C t ω| ≤ A * Real.exp (A * T)
    rw [abs_mul, abs_neg]
    exact mul_le_mul (hab t ht ω) (hCb t ht ω) (abs_nonneg _) hA
  have hn (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (ω : Ω) : C t ω * H t ω + Y t ω * V t ω ≤ 0 := by
    have h := mul_le_mul_of_nonneg_left (hdrift t ht ω) (intervalIntegratingFactor_pos T (fun x => a x ω) t).le
    dsimp [V]
    nlinarith
  have htail := weighted_compensated_martingale_ville_continuous P ℱ T hT C Y H V hC hY
    (fun ω => ⟨hCc ω, (hc ω).2.1, (hc ω).2.2, (hc ω).1.neg.mul (hCc ω)⟩)
    (fun ω t ht => hasDerivAt_intervalIntegratingFactor T _ (hc ω).1 t ht) hM
    (Real.exp (A * T)) BY BH (A * Real.exp (A * T)) hBH (by positivity) hCb hYb hHb hVb hn
    (fun t ht ω => mul_nonneg (intervalIntegratingFactor_pos T (fun x => a x ω) t).le (hpos t ht ω)) r hr
  have hEvent : {ω | ∃ t ∈ Icc (0 : ℝ) T, r < C t ω * Y t ω} =
      {ω | ∃ t ∈ Icc (0 : ℝ) T, r < Real.exp (-(∫ x in (0 : ℝ)..t, a x ω)) * Y t ω} := by
    ext ω
    constructor <;> rintro ⟨t, ht, hrt⟩ <;> refine ⟨t, ht, ?_⟩
    · rwa [show C t ω = Real.exp (-(∫ x in (0 : ℝ)..t, a x ω)) from intervalIntegratingFactor_eq T _ t ht] at hrt
    · rwa [show C t ω = Real.exp (-(∫ x in (0 : ℝ)..t, a x ω)) from intervalIntegratingFactor_eq T _ t ht]
  rw [hEvent] at htail
  simpa only [C, intervalIntegratingFactor, intervalIntegral.integral_same, neg_zero, Real.exp_zero, one_mul] using htail

/-- A decreasing nonnegative observable and its actual negative drift
satisfy every hypothesis. The product changes from 1 to 0 and has
an actual crossing of the positive threshold 1/2 at time zero. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (a Y H : ℝ → Unit → ℝ),
    (0 : ℝ) < 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (a t)) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (Y t)) ∧
    (∀ ω, ContinuousOn (fun t => a t ω) (Icc (0 : ℝ) 1) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) 1) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) 1)) ∧
    IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |a t ω| ≤ 0) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |Y t ω| ≤ 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |H t ω| ≤ 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, 0 ≤ Y t ω) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, H t ω ≤ a t ω * Y t ω) ∧
    (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ Y 0 () = 1 ∧ Y 1 () = 0 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let a : ℝ → Unit → ℝ := fun _ _ => 0
  let Y : ℝ → Unit → ℝ := fun t _ => 1 - t
  let H : ℝ → Unit → ℝ := fun _ _ => -1
  have hM : IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) := by
    have hz : (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) =
        (fun _ : ℝ => fun _ : Unit => (0 : ℝ)) := by
      funext t ω
      simp [Y, H]
    rw [hz]
    exact isMartingaleOn_const 1 ℱ (Measure.dirac ()) 0
  refine ⟨ℱ, a, Y, H, one_pos, fun t ht => stronglyMeasurable_const,
    fun t ht => stronglyMeasurable_const,
    fun ω => ⟨continuousOn_const, (continuous_const.sub continuous_id).continuousOn, continuousOn_const⟩,
    hM, le_rfl, zero_le_one, fun t ht ω => by norm_num [a], ?_,
    fun t ht ω => by norm_num [H], ?_, fun t ht ω => by norm_num [H, a],
    by norm_num, by norm_num, by norm_num [Y], by norm_num [Y]⟩
  · intro t ht ω
    dsimp [Y]
    rw [abs_of_nonneg (by linarith [ht.2] : 0 ≤ 1 - t)]
    linarith [ht.1]
  · intro t ht ω
    dsimp [Y]
    linarith [ht.2]

end Transformer.Homogenized
