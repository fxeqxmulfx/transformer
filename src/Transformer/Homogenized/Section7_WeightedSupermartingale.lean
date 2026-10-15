import Transformer.Homogenized.Section7_WeightedConditionalOrder
import Transformer.Homogenized.Section7_MartingaleTimeShift
import Transformer.Homogenized.Section7_ContinuousVille

/-!
# The weighted supermartingale inequality on every time pair

ArXiv:2604.01978v1, §7, the exponential maximal inequality after
`eq:Gronwall_E_selfcontained`, and `def:ito.formula`.
The initial-to-final conditional product inequality is applied
to the actual shifted process and filtration on [0,t-s].
The translated compensator has already been proved to be the
original subinterval drift integral. This gives the required
conditional expectation inequality for every s ≤ t in [0,T].

The only martingale premise concerns the original compensated
observable. The weighted process's supermartingale inequality
is derived from it and the sign of its actual product drift.
The product evaluations are also integrable by the proved bounds.
-/

open Set MeasureTheory
namespace Transformer.Homogenized

/-- Bounded measurable factors give integrable product evaluations.
Source: arXiv:2604.01978v1, §7, the exponential martingale argument. -/
theorem integrable_mul_of_bounded_stronglyMeasurable {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (f g : Ω → ℝ)
    (hf : StronglyMeasurable f) (hg : StronglyMeasurable g) (B D : ℝ)
    (hB : ∀ ω, |f ω| ≤ B) (hD : ∀ ω, |g ω| ≤ D) : Integrable (fun ω => f ω * g ω) P := by
  apply (integrable_const (B * D)).mono' (hf.mul hg).aestronglyMeasurable
  apply ae_of_all
  intro ω
  change |f ω * g ω| ≤ B * D
  rw [abs_mul]
  exact mul_le_mul (hB ω) (hD ω) (abs_nonneg _) ((abs_nonneg _).trans (hB ω))

/-- The actual weighted process satisfies conditional supermartingale
order at every pair of times in the original interval. Source:
arXiv:2604.01978v1, §7 and `def:ito.formula`. -/
theorem condExp_mul_le_of_compensated_martingale
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (C Y H V : ℝ → Ω → ℝ)
    (hC : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (C t))
    (hY : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (Y t))
    (hcont : ∀ ω, ContinuousOn (fun t => C t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => V t ω) (Icc (0 : ℝ) T))
    (hd : ∀ ω, ∀ x ∈ Icc (0 : ℝ) T, HasDerivAt (fun t => C t ω) (V x ω) x)
    (hM : IsMartingaleOn T ℱ P (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω))
    (BC BY BH BV : ℝ) (hBH : 0 ≤ BH) (hBV : 0 ≤ BV)
    (hCb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |C t ω| ≤ BC)
    (hYb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |Y t ω| ≤ BY)
    (hHb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |H t ω| ≤ BH)
    (hVb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |V t ω| ≤ BV)
    (hneg : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, C t ω * H t ω + Y t ω * V t ω ≤ 0)
    (s t : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (ht : t ∈ Icc (0 : ℝ) T) (hst : s ≤ t) :
    condExp (ℱ s) P (fun ω => C t ω * Y t ω) ≤ᵐ[P] (fun ω => C s ω * Y s ω) := by
  rcases hst.eq_or_lt with heq | hlt
  · subst t
    have hi := integrable_mul_of_bounded_stronglyMeasurable P (C s) (Y s)
      ((hC s hs).mono (ℱ.le s)) ((hY s hs).mono (ℱ.le s)) BC BY (hCb s hs) (hYb s hs)
    have hsm : StronglyMeasurable[ℱ s] (fun ω => C s ω * Y s ω) := (hC s hs).mul (hY s hs)
    rw [condExp_of_stronglyMeasurable (ℱ.le s) hsm hi]
  · have hend : s + (t - s) ≤ T := by linarith [ht.2]
    have hdom (u : ℝ) (hu : u ∈ Icc (0 : ℝ) (t - s)) : s + u ∈ Icc (0 : ℝ) T :=
      timeShift_mem_Icc T s (t - s) hs hend u hu
    have hm := isMartingaleOn_compensated_timeShift P ℱ T Y H
      (fun ω => (hcont ω).2.2.1) hM s (t - s) hs hend
    have hc := condExp_mul_le_initial_of_compensated_martingale P (timeShiftFiltration ℱ s)
      (t - s) (by linarith) (fun u ω => C (s + u) ω) (fun u ω => Y (s + u) ω)
      (fun u ω => H (s + u) ω) (fun u ω => V (s + u) ω)
      (fun u hu => hC _ (hdom u hu)) (fun u hu => hY _ (hdom u hu))
      (fun ω => ⟨continuousOn_timeShift T s (t - s) hs hend _ (hcont ω).1,
        continuousOn_timeShift T s (t - s) hs hend _ (hcont ω).2.1,
        continuousOn_timeShift T s (t - s) hs hend _ (hcont ω).2.2.1,
        continuousOn_timeShift T s (t - s) hs hend _ (hcont ω).2.2.2⟩)
      (fun ω u hu => HasDerivAt.comp_const_add s u (hd ω _ (hdom u hu)))
      (by simpa only [add_zero] using hm) BC BY BH BV hBH hBV
      (fun u hu => hCb _ (hdom u hu)) (fun u hu => hYb _ (hdom u hu))
      (fun u hu => hHb _ (hdom u hu)) (fun u hu => hVb _ (hdom u hu))
      (fun u hu => hneg _ (hdom u hu))
    change condExp ((timeShiftFiltration ℱ s) 0) P
      (fun ω => C (s + (t - s)) ω * Y (s + (t - s)) ω) ≤ᵐ[P]
        (fun ω => C (s + 0) ω * Y (s + 0) ω) at hc
    have hendTime : s + (t - s) = t := by ring
    simpa only [timeShiftFiltration_time_zero, hendTime, add_zero] using hc

/-- The continuous maximal bound follows from the original compensated
martingale and product drift sign. Source: arXiv:2604.01978v1, §7,
the exponential maximal inequality after `eq:Gronwall_E_selfcontained`. -/
theorem weighted_compensated_martingale_ville_continuous
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (hT : 0 < T) (C Y H V : ℝ → Ω → ℝ)
    (hC : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (C t))
    (hY : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (Y t))
    (hcont : ∀ ω, ContinuousOn (fun t => C t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => V t ω) (Icc (0 : ℝ) T))
    (hd : ∀ ω, ∀ x ∈ Icc (0 : ℝ) T, HasDerivAt (fun t => C t ω) (V x ω) x)
    (hM : IsMartingaleOn T ℱ P (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω))
    (BC BY BH BV : ℝ) (hBH : 0 ≤ BH) (hBV : 0 ≤ BV)
    (hCb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |C t ω| ≤ BC)
    (hYb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |Y t ω| ≤ BY)
    (hHb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |H t ω| ≤ BH)
    (hVb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |V t ω| ≤ BV)
    (hneg : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, C t ω * H t ω + Y t ω * V t ω ≤ 0)
    (hpos : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, 0 ≤ C t ω * Y t ω) (a : ℝ) (ha : 0 < a) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T, a < C t ω * Y t ω} ≤
      ENNReal.ofReal (∫ ω, C 0 ω * Y 0 ω ∂P) / ENNReal.ofReal a := by
  exact nonnegative_supermartingale_ville_continuous_div P ℱ _ T hT
    (fun t ht => (hC t ht).mul (hY t ht))
    (fun t ht => integrable_mul_of_bounded_stronglyMeasurable P (C t) (Y t)
      ((hC t ht).mono (ℱ.le t)) ((hY t ht).mono (ℱ.le t)) BC BY (hCb t ht) (hYb t ht))
    (fun s hs t ht hst => condExp_mul_le_of_compensated_martingale P ℱ T C Y H V
      hC hY hcont hd hM BC BY BH BV hBH hBV hCb hYb hHb hVb hneg s t hs ht hst)
    hpos (fun ω => (hcont ω).1.mul (hcont ω).2.1) a ha

/-- Two nonconstant bounded factors under a genuine Gaussian law
realize every integrability hypothesis. -/
example : Integrable (fun ω : ℝ => Real.sin ω * Real.cos ω)
    (ProbabilityTheory.gaussianReal 0 1) := by
  exact integrable_mul_of_bounded_stronglyMeasurable (ProbabilityTheory.gaussianReal 0 1) Real.sin Real.cos
    Real.continuous_sin.stronglyMeasurable Real.continuous_cos.stronglyMeasurable 1 1
    Real.abs_sin_le_one Real.abs_cos_le_one


/-- Nonconstant factors with actual negative product drift satisfy
all hypotheses, including two different times in the horizon.
Their product strictly decreases from 1 to 0. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (C Y H V : ℝ → Unit → ℝ),
    (0 : ℝ) < 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (C t)) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (Y t)) ∧
    (∀ ω, ContinuousOn (fun t => C t ω) (Icc (0 : ℝ) 1) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) 1) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) 1) ∧
      ContinuousOn (fun t => V t ω) (Icc (0 : ℝ) 1)) ∧
    (∀ ω, ∀ x ∈ Icc (0 : ℝ) 1, HasDerivAt (fun t => C t ω) (V x ω) x) ∧
    IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) ∧
    (0 : ℝ) ≤ 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |C t ω| ≤ 2) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |Y t ω| ≤ 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |H t ω| ≤ 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |V t ω| ≤ 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, C t ω * H t ω + Y t ω * V t ω ≤ 0) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, 0 ≤ C t ω * Y t ω) ∧
    (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 2 ∧ C 1 () * Y 1 () < C 0 () * Y 0 () := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let C : ℝ → Unit → ℝ := fun t _ => 1 + t
  let Y : ℝ → Unit → ℝ := fun t _ => 1 - t
  let H : ℝ → Unit → ℝ := fun _ _ => -1
  let V : ℝ → Unit → ℝ := fun _ _ => 1
  have hM : IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) := by
    have hz : (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) =
        (fun _ : ℝ => fun _ : Unit => (0 : ℝ)) := by
      funext t ω
      simp [Y, H]
    rw [hz]
    exact isMartingaleOn_const 1 ℱ (Measure.dirac ()) 0
  refine ⟨ℱ, C, Y, H, V, one_pos, fun t ht => stronglyMeasurable_const,
    fun t ht => stronglyMeasurable_const, ?_, fun ω x hx => (hasDerivAt_id x).const_add 1,
    hM, zero_le_one, ?_, ?_, fun t ht ω => by norm_num [H], fun t ht ω => by norm_num [V],
    ?_, ?_, by norm_num, by norm_num, zero_le_one, by norm_num, by norm_num [C, Y]⟩
  · intro ω
    exact ⟨(continuous_const.add continuous_id).continuousOn,
      (continuous_const.sub continuous_id).continuousOn, continuousOn_const, continuousOn_const⟩
  · intro t ht ω
    dsimp [C]
    rw [abs_of_nonneg (by linarith [ht.1] : 0 ≤ 1 + t)]
    linarith [ht.2]
  · intro t ht ω
    dsimp [Y]
    rw [abs_of_nonneg (by linarith [ht.2] : 0 ≤ 1 - t)]
    linarith [ht.1]
  · intro t ht ω
    dsimp [C, Y, H, V]
    linarith [ht.1]
  · intro t ht ω
    dsimp [C, Y]
    exact mul_nonneg (by linarith [ht.1]) (by linarith [ht.2])

end Transformer.Homogenized
