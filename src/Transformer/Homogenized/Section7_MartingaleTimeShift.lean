module

public import Transformer.Homogenized.Section7_WeightedExpectation
public import Mathlib.Analysis.Calculus.Deriv.Shift

/-!
# Moving the compensated martingale to an actual subinterval

ArXiv:2604.01978v1, §7, the exponential maximal inequality after
`eq:Gronwall_E_selfcontained`, and `def:ito.formula`.
To apply the weighted expectation argument between s and t,
translate the clock and the filtration together. The martingale
increment M(s+u)-M(s) is adapted to the translated filtration,
and its conditional expectation identity follows from the
original one on [0,T].

For the compensated observable, the translated time integral
is exactly the original integral between s and s+u. The proof
uses this equality rather than assuming a new martingale law
on every subinterval.
-/

@[expose] public section

open Set MeasureTheory
namespace Transformer.Homogenized

/-- The original filtration at the translated time s+u.
Source: arXiv:2604.01978v1, §7, the integrating-factor argument. -/
def timeShiftFiltration {Ω : Type*} {m : MeasurableSpace Ω} (ℱ : Filtration ℝ m) (s : ℝ) :
    Filtration ℝ m :=
  ⟨fun u => ℱ (s + u),
    fun u v huv => ℱ.mono (show s + u ≤ s + v from add_le_add le_rfl huv), fun u => ℱ.le (s + u)⟩

/-- The translated initial sigma algebra is the original one at s.
Source: arXiv:2604.01978v1, §7, subinterval conditioning. -/
@[simp]
theorem timeShiftFiltration_time_zero {Ω : Type*} {m : MeasurableSpace Ω}
    (ℱ : Filtration ℝ m) (s : ℝ) : (timeShiftFiltration ℱ s) 0 = ℱ s := by
  change ℱ (s + 0) = ℱ s
  rw [add_zero]

/-- The translated terminal sigma algebra is the original one at t.
Source: arXiv:2604.01978v1, §7, subinterval conditioning. -/
@[simp]
theorem timeShiftFiltration_time_sub {Ω : Type*} {m : MeasurableSpace Ω}
    (ℱ : Filtration ℝ m) (s t : ℝ) : (timeShiftFiltration ℱ s) (t - s) = ℱ t := by
  change ℱ (s + (t - s)) = ℱ t
  congr 1
  ring

/-- Every translated time remains in the actual original interval.
Source: arXiv:2604.01978v1, §7, the finite-horizon maximal inequality. -/
theorem timeShift_mem_Icc (T s U : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (hend : s + U ≤ T)
    (u : ℝ) (hu : u ∈ Icc (0 : ℝ) U) : s + u ∈ Icc (0 : ℝ) T := by
  constructor <;> linarith [hs.1, hu.1, hu.2]

/-- Original continuous paths remain continuous on the translated interval.
Source: arXiv:2604.01978v1, §7 and `def:ito.formula`. -/
theorem continuousOn_timeShift {E : Type*} [TopologicalSpace E]
    (T s U : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (hend : s + U ≤ T) (f : ℝ → E)
    (hf : ContinuousOn f (Icc (0 : ℝ) T)) :
    ContinuousOn (fun u => f (s + u)) (Icc (0 : ℝ) U) := by
  exact hf.comp' (continuous_const.add continuous_id).continuousOn
    (fun u hu => timeShift_mem_Icc T s U hs hend u hu)

/-- Translating the clock and subtracting the actual initial value
preserves the original martingale identity. Source: arXiv:2604.01978v1,
§7, the exponential supermartingale argument. -/
theorem isMartingaleOn_timeShift_increment {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (T : ℝ)
    (M : ℝ → Ω → ℝ) (hM : IsMartingaleOn T ℱ P M) (s U : ℝ)
    (hs : s ∈ Icc (0 : ℝ) T) (hend : s + U ≤ T) :
    IsMartingaleOn U (timeShiftFiltration ℱ s) P (fun u ω => M (s + u) ω - M s ω) := by
  have hdom (u : ℝ) (hu : u ∈ Icc (0 : ℝ) U) : s + u ∈ Icc (0 : ℝ) T :=
    timeShift_mem_Icc T s U hs hend u hu
  have hMs (u : ℝ) (hu : u ∈ Icc (0 : ℝ) U) : StronglyMeasurable[ℱ (s + u)] (M s) :=
    (hM.1 s hs).mono (ℱ.mono (by linarith [hu.1]))
  refine ⟨fun u hu => (hM.1 _ (hdom u hu)).sub (hMs u hu),
    fun u hu => (hM.2.1 _ (hdom u hu)).sub (hM.2.1 s hs), ?_⟩
  intro u hu v hv huv
  have hc : condExp (ℱ (s + u)) P (M s) = M s :=
    condExp_of_stronglyMeasurable (ℱ.le _) (hMs u hu) (hM.2.1 s hs)
  filter_upwards [condExp_sub (hM.2.1 _ (hdom v hv)) (hM.2.1 s hs) (ℱ (s + u)),
    hM.2.2 _ (hdom u hu) _ (hdom v hv) (by linarith)] with ω hsub hCE
  change condExp (ℱ (s + u)) P (fun ω => M (s + v) ω - M s ω) ω = M (s + u) ω - M s ω
  change condExp (ℱ (s + u)) P (fun ω => M (s + v) ω - M s ω) ω =
    condExp (ℱ (s + u)) P (M (s + v)) ω - condExp (ℱ (s + u)) P (M s) ω at hsub
  rw [hsub, hCE, hc]

/-- The actual compensated observable satisfies the martingale identity
on the translated interval. Source: arXiv:2604.01978v1, §7 and
`def:ito.formula`; the translated compensator is proved equal to
its original subinterval integral. -/
theorem isMartingaleOn_compensated_timeShift {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (T : ℝ) (Y H : ℝ → Ω → ℝ)
    (hH : ∀ ω, ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) T))
    (hM : IsMartingaleOn T ℱ P (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω))
    (s U : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (hend : s + U ≤ T) :
    IsMartingaleOn U (timeShiftFiltration ℱ s) P
      (fun u ω => Y (s + u) ω - Y s ω - ∫ x in (0 : ℝ)..u, H (s + x) ω) := by
  let M (t : ℝ) (ω : Ω) := Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω
  have hshift := isMartingaleOn_timeShift_increment P ℱ T M hM s U hs hend
  have heq (u : ℝ) (hu : u ∈ Icc (0 : ℝ) U) :
      (fun ω => Y (s + u) ω - Y s ω - ∫ x in (0 : ℝ)..u, H (s + x) ω) =
        (fun ω => M (s + u) ω - M s ω) := by
    funext ω
    rw [intervalIntegral.integral_comp_add_left (f := fun x => H x ω) s]
    simp only [add_zero]
    symm
    exact compensated_time_increment_eq T (fun t => Y t ω) (fun t => H t ω) (hH ω)
      s (s + u) hs ⟨by linarith [hs.1, hu.1], by linarith [hu.2]⟩ (by linarith [hu.1])
  refine ⟨fun u hu => ?_, fun u hu => ?_, ?_⟩
  · change StronglyMeasurable[(timeShiftFiltration ℱ s) u]
      (fun ω => Y (s + u) ω - Y s ω - ∫ x in (0 : ℝ)..u, H (s + x) ω)
    rw [heq u hu]
    exact hshift.1 u hu
  · change Integrable (fun ω => Y (s + u) ω - Y s ω - ∫ x in (0 : ℝ)..u, H (s + x) ω) P
    rw [heq u hu]
    exact hshift.2.1 u hu
  · intro u hu v hv huv
    change condExp ((timeShiftFiltration ℱ s) u) P
      (fun ω => Y (s + v) ω - Y s ω - ∫ x in (0 : ℝ)..v, H (s + x) ω) =ᵐ[P]
        (fun ω => Y (s + u) ω - Y s ω - ∫ x in (0 : ℝ)..u, H (s + x) ω)
    rw [heq u hu, heq v hv]
    exact hshift.2.2 u hu v hv huv

/-- A quadratic observable and its actual linear drift realize every
hypothesis on [0,2] and on the nonzero translated interval [1,2].
The observable changes by 3 across that subinterval. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (Y H : ℝ → Unit → ℝ),
    IsMartingaleOn 2 ℱ (Measure.dirac ())
      (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) ∧
    (∀ ω, ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) 2)) ∧
    (∀ ω, ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) 2)) ∧
    (1 : ℝ) ∈ Icc (0 : ℝ) 2 ∧ (1 : ℝ) + 1 ≤ 2 ∧
    (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ Y 2 () - Y 1 () = 3 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let Y : ℝ → Unit → ℝ := fun t _ => t ^ 2
  let H : ℝ → Unit → ℝ := fun t _ => 2 * t
  have hM : IsMartingaleOn 2 ℱ (Measure.dirac ())
      (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) := by
    have hz : (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) =
        (fun _ : ℝ => fun _ : Unit => (0 : ℝ)) := by
      funext t ω
      dsimp [Y, H]
      rw [intervalIntegral.integral_const_mul, integral_id]
      ring
    rw [hz]
    exact isMartingaleOn_const 2 ℱ (Measure.dirac ()) 0
  exact ⟨ℱ, Y, H, hM, fun ω => (continuous_const.mul continuous_id).continuousOn,
    fun ω => (continuous_id.pow 2).continuousOn, by norm_num, by norm_num, by norm_num,
    by norm_num [Y]⟩

end Transformer.Homogenized
