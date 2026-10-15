import Transformer.Homogenized.Section7_WeightedExpectation
import Mathlib.MeasureTheory.Function.AEEqOfIntegral

/-!
# Conditional order for the random integrating factor

ArXiv:2604.01978v1, §7, the exponential maximal inequality after
`eq:Gronwall_E_selfcontained`, and `def:ito.formula`.
The actual weighted compensated martingale identity gives a
conditional expectation inequality when its product drift is
nonpositive. Bounded adapted evaluations give the required
integrability. The drift sign is used inside the original time
integrals, after the continuous weighted identity has been proved.

The first lemma transfers all sub-sigma-algebra set integral
inequalities to conditional order by trimming the measure.
The endpoint result is then available on each shifted time interval.
-/

open Set MeasureTheory
namespace Transformer.Homogenized

/-- Integral order on every event of the conditioning sigma algebra
gives actual conditional expectation order. Source: arXiv:2604.01978v1,
§7, the exponential supermartingale argument. -/
theorem condExp_le_of_setIntegral_le {Ω : Type*} {m₀ : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (m : MeasurableSpace Ω) (hm : m ≤ m₀) (f g : Ω → ℝ) (hf : Integrable f P)
    (hg : Integrable g P) (hgm : StronglyMeasurable[m] g)
    (hI : ∀ A : Set Ω, MeasurableSet[m] A → (∫ ω in A, f ω ∂P) ≤ ∫ ω in A, g ω ∂P) :
    condExp m P f ≤ᵐ[P] g := by
  apply ae_of_ae_trim hm
  apply ae_le_of_forall_setIntegral_le
    (Integrable.trim hm integrable_condExp stronglyMeasurable_condExp) (Integrable.trim hm hg hgm)
  intro A hA hAfin
  rw [← setIntegral_trim hm stronglyMeasurable_condExp hA, ← setIntegral_trim hm hgm hA,
    setIntegral_condExp hm hf hA]
  exact hI A hA

/-- A nonpositive compensated product drift gives the initial-to-final
conditional expectation inequality for the actual random factor.
Source: arXiv:2604.01978v1, §7 and `def:ito.formula`. -/
theorem condExp_mul_le_initial_of_compensated_martingale
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
    (hneg : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, C t ω * H t ω + Y t ω * V t ω ≤ 0) :
    condExp (ℱ 0) P (fun ω => C T ω * Y T ω) ≤ᵐ[P] (fun ω => C 0 ω * Y 0 ω) := by
  have hProdInt (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : Integrable (fun ω => C t ω * Y t ω) P := by
    apply (integrable_const (BC * BY)).mono'
      (((hC t ht).mul (hY t ht)).mono (ℱ.le t)).aestronglyMeasurable
    apply ae_of_all
    intro ω
    change |C t ω * Y t ω| ≤ BC * BY
    rw [abs_mul]
    exact mul_le_mul (hCb t ht ω) (hYb t ht ω) (abs_nonneg _)
      ((abs_nonneg _).trans (hCb t ht ω))
  have h0 : (0 : ℝ) ∈ Icc (0 : ℝ) T := ⟨le_rfl, hT.le⟩
  have hlast : T ∈ Icc (0 : ℝ) T := ⟨hT.le, le_rfl⟩
  apply condExp_le_of_setIntegral_le P (ℱ 0) (ℱ.le 0) _ _ (hProdInt T hlast) (hProdInt 0 h0)
    ((hC 0 h0).mul (hY 0 h0))
  intro A hA
  obtain ⟨hL, hLI⟩ := weighted_compensated_expectation_eq_zero P ℱ T hT C Y H V hC
    hcont hd hM BC BY BH BV hBH hBV hCb hYb hHb hVb A hA
  have hle (ω : Ω) : C T ω * Y T ω - C 0 ω * Y 0 ω ≤
      C T ω * Y T ω - C 0 ω * Y 0 ω - (∫ x in (0 : ℝ)..T, C x ω * H x ω) -
        ∫ x in (0 : ℝ)..T, Y x ω * V x ω := by
    have hCH : IntervalIntegrable (fun x => C x ω * H x ω) volume 0 T := by
      apply ContinuousOn.intervalIntegrable
      rw [uIcc_of_le hT.le]
      exact (hcont ω).1.mul (hcont ω).2.2.1
    have hYV : IntervalIntegrable (fun x => Y x ω * V x ω) volume 0 T := by
      apply ContinuousOn.intervalIntegrable
      rw [uIcc_of_le hT.le]
      exact (hcont ω).2.1.mul (hcont ω).2.2.2
    have hi : (∫ x in (0 : ℝ)..T, C x ω * H x ω + Y x ω * V x ω) ≤ 0 := by
      rw [intervalIntegral.integral_of_le hT.le]
      exact setIntegral_nonpos measurableSet_Ioc (fun x hx => hneg x ⟨hx.1.le, hx.2⟩ ω)
    rw [intervalIntegral.integral_add hCH hYV] at hi
    linarith
  have hExpect := integral_mono (μ := P.restrict A)
    ((hProdInt T hlast).sub (hProdInt 0 h0)).integrableOn hL.integrableOn hle
  rw [hLI] at hExpect
  change (∫ ω in A, C T ω * Y T ω - C 0 ω * Y 0 ω ∂P) ≤ 0 at hExpect
  rw [integral_sub (hProdInt T hlast).integrableOn (hProdInt 0 h0).integrableOn] at hExpect
  exact sub_nonpos.mp hExpect

/-- Constants 0 and 1 realize every conditional-order hypothesis. -/
example : condExp (⊥ : MeasurableSpace Unit) (Measure.dirac ()) (fun _ => (0 : ℝ)) ≤ᵐ[Measure.dirac ()]
    (fun _ => (1 : ℝ)) := by
  apply condExp_le_of_setIntegral_le (Measure.dirac ()) ⊥ bot_le _ _
    (integrable_const 0) (integrable_const 1) stronglyMeasurable_const
  intro A hA
  exact integral_mono (integrable_const 0) (integrable_const 1) (fun ω => zero_le_one)


/-- Both factors vary in time, their compensated product drift is
nonpositive, and their product decreases from 1 to 0. The observable
minus its actual drift integral is a genuine martingale. -/
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
    C 1 () * Y 1 () < C 0 () * Y 0 () := by
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
    ?_, by norm_num [C, Y]⟩
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

end Transformer.Homogenized
