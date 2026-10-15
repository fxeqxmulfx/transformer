import Transformer.Homogenized.Section7_LogisticModification
import Transformer.Homogenized.Section7_LogisticArcsin

/-!
# The original logistic angle and its finite expected bracket budget

ArXiv:2604.01978v1, §7, `sec: sde.logistic`.
Initial membership in [-1,1] now suffices for both of the source's
angle martingale identities. The preceding bounded modification solves
the full original equation, so its proved angle identities transfer back
to the original process. Adaptation of the actual original compensator
is proved directly from its continuous adapted integrand.

The centered-square identity implies that the expected actual quadratic
variation equals the centered angle's second moment. The latter is at
most π² P(Ω), uniformly in the time horizon, because the angle is bounded.
This budget supplies the long-time argument with its actual source
bracket, without assuming bracket integrability or an energy estimate.
It also identifies the source's printed bracket-to-zero sentence as
unnecessary: an interior limit would make the bracket diverge, whereas
this finite expectation budget will rule out that behavior almost surely.
-/

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- Actual adapted versions preserve the conditional identities under almost-everywhere equality.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem IsMartingaleOn.congr_ae {Ω : Type*} {m : MeasurableSpace Ω}
    {P : Measure Ω} {ℱ : Filtration ℝ m} {T : ℝ} {M N : ℝ → Ω → ℝ}
    (hM : IsMartingaleOn T ℱ P M)
    (hN : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (N t))
    (heq : ∀ t ∈ Icc (0 : ℝ) T, M t =ᵐ[P] N t) : IsMartingaleOn T ℱ P N := by
  refine ⟨hN, fun t ht => (hM.2.1 t ht).congr (heq t ht), ?_⟩
  intro s hs t ht hst
  exact (condExp_congr_ae (heq t ht)).symm.trans
    ((hM.2.2 s hs t ht hst).trans (heq s hs))

/-- The original conditional martingale identity preserves expectation from time zero.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem IsMartingaleOn.integral_eq_initial {Ω : Type*} {m : MeasurableSpace Ω}
    {P : Measure Ω} [IsFiniteMeasure P] {ℱ : Filtration ℝ m} {T : ℝ} {M : ℝ → Ω → ℝ}
    (hM : IsMartingaleOn T ℱ P M) (hT : 0 ≤ T) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    (∫ ω, M t ω ∂P) = ∫ ω, M 0 ω ∂P := by
  calc
    _ = ∫ ω, P[M t | ℱ 0] ω ∂P := (integral_condExp (ℱ.le 0)).symm
    _ = _ := integral_congr_ae (hM.2.2 0 ⟨le_rfl, hT⟩ t ht ht.1)

/-- Both actual angle identities follow from the equation and the initial interval alone.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem logistic_arcsin_martingale_of_initial_range
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) (T : ℝ) (hT : 0 ≤ T) :
    IsMartingaleOn T ℱ P (fun t ω => Real.arcsin (u t ω)) ∧
      IsMartingaleOn T ℱ P (fun t ω =>
        (Real.arcsin (u t ω) - Real.arcsin (u 0 ω)) ^ 2 -
          ∫ s in (0 : ℝ)..t, 2 * (1 - u s ω ^ 2)) := by
  obtain ⟨hv, heq⟩ := isLogisticSolution_clipped P ℱ u hu hinit
  obtain ⟨hθ, hq⟩ := logistic_arcsin_martingale P ℱ _ hv
    (fun t ht ω => clippedLogisticProcess_mem u t ω) T hT
  have hθm (t : ℝ) (ht : 0 ≤ t) : StronglyMeasurable[ℱ t] (fun ω => Real.arcsin (u t ω)) :=
    Real.continuous_arcsin.comp_stronglyMeasurable (hu.adapted t ht)
  refine ⟨hθ.congr_ae (fun t ht => hθm t ht.1) ?_, hq.congr_ae ?_ ?_⟩
  · intro t ht
    filter_upwards [heq] with ω hω
    rw [hω t ht.1]
  · intro t ht
    have hi := stronglyMeasurable_adapted_timeIntegral ℱ T
      (fun s ω => 2 * (1 - u s ω ^ 2))
      (fun s hs => stronglyMeasurable_const.mul (stronglyMeasurable_const.sub ((hu.adapted s hs.1).pow 2)))
      (fun ω => continuousOn_const.mul (continuousOn_const.sub
        (((hu.cont ω).mono (fun s hs => hs.1)).pow 2))) t ht
    exact (((hθm t ht.1).sub ((hθm 0 le_rfl).mono (ℱ.mono ht.1))).pow 2).sub hi
  · intro t ht
    filter_upwards [heq] with ω hω
    rw [hω t ht.1, hω 0 le_rfl]
    congr 1
    apply intervalIntegral.integral_congr
    intro s hs
    rw [uIcc_of_le ht.1] at hs
    change 2 * (1 - clippedLogisticProcess u s ω ^ 2) = 2 * (1 - u s ω ^ 2)
    rw [hω s hs.1]

/-- The expected original bracket equals its actual second moment and has a uniform finite budget.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem logistic_bracket_energy
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) (T : ℝ) (hT : 0 ≤ T) :
    Integrable (fun ω => ∫ s in (0 : ℝ)..T, 2 * (1 - u s ω ^ 2)) P ∧
      (∫ ω, (∫ s in (0 : ℝ)..T, 2 * (1 - u s ω ^ 2)) ∂P) =
        ∫ ω, (Real.arcsin (u T ω) - Real.arcsin (u 0 ω)) ^ 2 ∂P ∧
      (∫ ω, (∫ s in (0 : ℝ)..T, 2 * (1 - u s ω ^ 2)) ∂P) ≤ Real.pi ^ 2 * P.real univ := by
  have hM := (logistic_arcsin_martingale_of_initial_range P ℱ u hu hinit T hT).2
  let S : Ω → ℝ := fun ω => (Real.arcsin (u T ω) - Real.arcsin (u 0 ω)) ^ 2
  have hb (ω : Ω) : S ω ≤ Real.pi ^ 2 := by
    have hTθ := Real.arcsin_mem_Icc (u T ω)
    have h0θ := Real.arcsin_mem_Icc (u 0 ω)
    have hd : |Real.arcsin (u T ω) - Real.arcsin (u 0 ω)| ≤ Real.pi := by
      apply abs_le.mpr
      constructor <;> linarith [hTθ.1, hTθ.2, h0θ.1, h0θ.2]
    simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg _) hd 2
  have hiS : Integrable S P := Integrable.of_bound
    (((Real.continuous_arcsin.comp_stronglyMeasurable ((hu.adapted T hT).mono (ℱ.le T))).sub
      (Real.continuous_arcsin.comp_stronglyMeasurable ((hu.adapted 0 le_rfl).mono (ℱ.le 0)))).pow 2).aestronglyMeasurable
    (Real.pi ^ 2) (ae_of_all P (fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      exact hb ω))
  have hiQ : Integrable (fun ω => ∫ s in (0 : ℝ)..T, 2 * (1 - u s ω ^ 2)) P := by
    convert hiS.sub (hM.2.1 T ⟨hT, le_rfl⟩) using 1
    funext ω
    dsimp [S]
    ring
  have hE := hM.integral_eq_initial hT T ⟨hT, le_rfl⟩
  simp only [sub_self, zero_pow (by decide : 2 ≠ 0), intervalIntegral.integral_same,
    integral_zero] at hE
  have hsub : (∫ ω, S ω ∂P) - ∫ ω, (∫ s in (0 : ℝ)..T, 2 * (1 - u s ω ^ 2)) ∂P = 0 := by
    rw [← integral_sub hiS hiQ]
    exact hE
  have hbound := integral_mono_ae hiS (integrable_const (Real.pi ^ 2)) (ae_of_all P hb)
  simp only [integral_const, smul_eq_mul] at hbound
  refine ⟨hiQ, by linarith, ?_⟩
  nlinarith

/-- The source's deterministic initial angle remains the actual mean at every time.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the endpoint probability formula. -/
theorem integral_logistic_arcsin_eq_initial
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (u₀ : ℝ) (hu₀ : u₀ ∈ Icc (-1 : ℝ) 1) (hinit : ∀ ω, u 0 ω = u₀)
    (T : ℝ) (hT : 0 ≤ T) : (∫ ω, Real.arcsin (u T ω) ∂P) = Real.arcsin u₀ := by
  have hr (ω : Ω) : u 0 ω ∈ Icc (-1 : ℝ) 1 := by rw [hinit ω]; exact hu₀
  have hM := (logistic_arcsin_martingale_of_initial_range P ℱ u hu hr T hT).1
  simpa only [hinit, integral_const, probReal_univ, one_smul] using
    hM.integral_eq_initial hT T ⟨hT, le_rfl⟩

/-- The genuine negative absorbing solution realizes every source hypothesis
on a positive horizon. Its nonzero angle also supplies identical adapted
versions for the congruence lemma and the constant-mean identity. -/
example : ∃ ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit),
    IsLogisticSolution (Measure.dirac ()) ℱ (fun _ _ => -1) ∧
      (∀ _ : Unit, (-1 : ℝ) ∈ Icc (-1 : ℝ) 1) ∧ (0 : ℝ) ≤ 1 ∧
      IsMartingaleOn 1 ℱ (Measure.dirac ()) (fun _ _ => Real.arcsin (-1)) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  have hu := isLogisticSolution_neg_one (Measure.dirac ()) ℱ
  have hinit (ω : Unit) : (-1 : ℝ) ∈ Icc (-1 : ℝ) 1 := by norm_num
  exact ⟨ℱ, hu, hinit, zero_le_one,
    (logistic_arcsin_martingale_of_initial_range _ ℱ (fun _ _ => -1) hu hinit 1 zero_le_one).1⟩

end Transformer.Homogenized
