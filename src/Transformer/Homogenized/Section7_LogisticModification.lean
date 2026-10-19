module

public import Transformer.Homogenized.Section7_LogisticRange
public import Transformer.Homogenized.Section7_IntegratingFactor

/-!
# A single probability-one range event and a bounded logistic modification

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, the bounded angle argument.
Closed interval membership at each fixed time extends to all nonnegative
times on one probability-one event by continuity and countable actual grids.
Clipping the original process to [-1,1] is then indistinguishable from it.
The clipped process is adapted and has continuous paths at every sample.
Its original smooth-test martingale problem is proved by actual adapted
compensators and almost-everywhere equality of the original integrals.
No completeness of the filtration or measurability of a future event in
an earlier sigma algebra is needed. This modification lets the proved
pointwise-range angle theorem apply to the source's original solution.
The example explicitly distinguishes pathwise and almost-sure equality:
it changes an absorbing solution on a null set while keeping continuous
adapted paths, and verifies the full original martingale problem there.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- Continuous paths turn fixed-time closed interval bounds into one all-time event.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem ae_forall_mem_Icc_of_continuous {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (u : ℝ → Ω → ℝ) (a b : ℝ)
    (hc : ∀ ω, ContinuousOn (fun t => u t ω) (Ici 0))
    (hr : ∀ t, 0 ≤ t → ∀ᵐ ω ∂P, u t ω ∈ Icc a b) :
    ∀ᵐ ω ∂P, ∀ t, 0 ≤ t → u t ω ∈ Icc a b := by
  have hg : ∀ᵐ ω ∂P, ∀ N k j : ℕ, u (intervalGridTime ((N : ℝ) + 1) k j) ω ∈ Icc a b :=
    ae_all_iff.mpr (fun N => ae_all_iff.mpr (fun k => ae_all_iff.mpr (fun j =>
      hr _ (intervalGridTime_mem ((N : ℝ) + 1) (by positivity) k j).1)))
  filter_upwards [hg] with ω hω
  intro t ht
  obtain ⟨N, hN⟩ := exists_nat_gt t
  have hT : 0 < (N : ℝ) + 1 := by positivity
  have hmem : t ∈ Icc (0 : ℝ) ((N : ℝ) + 1) := ⟨ht, by linarith⟩
  have hwithin : Tendsto (fun k => intervalGridTime ((N : ℝ) + 1) k
      (intervalGridIndex ((N : ℝ) + 1) t k)) atTop (𝓝[Ici 0] t) :=
    tendsto_nhdsWithin_iff.mpr ⟨intervalGridTime_tendsto _ hT t hmem,
      Eventually.of_forall (fun k => (intervalGridTime_mem _ hT.le k _).1)⟩
  exact isClosed_Icc.mem_of_tendsto (Filter.Tendsto.comp (hc ω t ht) hwithin)
    (Eventually.of_forall (fun k => hω N k _))

/-- The entire logistic path remains in the source interval on one probability-one event.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem logistic_range_ae_all {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) :
    ∀ᵐ ω ∂P, ∀ t, 0 ≤ t → u t ω ∈ Icc (-1 : ℝ) 1 :=
  ae_forall_mem_Icc_of_continuous P u (-1) 1 hu.cont (logistic_range P ℱ u hu hinit)

/-- An adapted continuous indistinguishable modification solves the original full problem.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem IsLogisticSolution.of_indistinguishable {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) (ℱ : Filtration ℝ m) (u v : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hv : ∀ t, 0 ≤ t → StronglyMeasurable[ℱ t] (v t))
    (hc : ∀ ω, ContinuousOn (fun t => v t ω) (Ici 0))
    (heq : ∀ᵐ ω ∂P, ∀ t, 0 ≤ t → v t ω = u t ω) : IsLogisticSolution P ℱ v := by
  refine ⟨hv, hc, ?_⟩
  intro φ hφ T hT
  have hM := hu.mart φ hφ T hT
  have hnear (t : ℝ) (ht : 0 ≤ t) : logisticMart v φ t =ᵐ[P] logisticMart u φ t := by
    filter_upwards [heq] with ω hω
    unfold logisticMart
    rw [hω t ht, hω 0 le_rfl]
    congr 1
    apply intervalIntegral.integral_congr
    intro s hs
    rw [uIcc_of_le ht] at hs
    change logisticGenerator φ (v s ω) = logisticGenerator φ (u s ω)
    rw [hω s hs.1]
  have hstrong (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
      StronglyMeasurable[ℱ t] (logisticMart v φ t) := by
    have hi := stronglyMeasurable_adapted_timeIntegral ℱ T
      (fun s ω => logisticGenerator φ (v s ω))
      (fun s hs => (continuous_logisticGenerator φ hφ).comp_stronglyMeasurable (hv s hs.1))
      (fun ω => (continuous_logisticGenerator φ hφ).comp_continuousOn'
        ((hc ω).mono (fun s hs => hs.1))) t ht
    exact ((hφ.continuous.comp_stronglyMeasurable (hv t ht.1)).sub
      ((hφ.continuous.comp_stronglyMeasurable (hv 0 le_rfl)).mono (ℱ.mono ht.1))).sub hi
  refine ⟨hstrong, fun t ht => (hM.2.1 t ht).congr (hnear t ht.1).symm, ?_⟩
  intro s hs t ht hst
  exact (condExp_congr_ae (hnear t ht.1)).trans
    ((hM.2.2 s hs t ht hst).trans (hnear s hs.1).symm)

/-- The continuous pointwise clipping of the original process to its proved state interval.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
noncomputable def clippedLogisticProcess {Ω : Type*} (u : ℝ → Ω → ℝ) : ℝ → Ω → ℝ :=
  fun t ω => max (-1) (min 1 (u t ω))

/-- The actual clipped values lie in the source interval at every time and sample.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem clippedLogisticProcess_mem {Ω : Type*} (u : ℝ → Ω → ℝ) (t : ℝ) (ω : Ω) :
    clippedLogisticProcess u t ω ∈ Icc (-1 : ℝ) 1 := by
  exact ⟨le_max_left _ _, max_le (by norm_num) (min_le_left _ _)⟩

/-- Clipping fixes every actual value that already belongs to the source interval.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem clippedLogisticProcess_eq_of_mem {Ω : Type*} (u : ℝ → Ω → ℝ) (t : ℝ) (ω : Ω)
    (hu : u t ω ∈ Icc (-1 : ℝ) 1) : clippedLogisticProcess u t ω = u t ω := by
  simp only [clippedLogisticProcess, min_eq_right hu.2, max_eq_right hu.1]

/-- The bounded modification solves the full original problem and agrees for all times almost surely.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem isLogisticSolution_clipped {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) :
    IsLogisticSolution P ℱ (clippedLogisticProcess u) ∧
      (∀ᵐ ω ∂P, ∀ t, 0 ≤ t → clippedLogisticProcess u t ω = u t ω) := by
  have hclip : Continuous (fun x : ℝ => max (-1) (min 1 x)) :=
    continuous_const.max (continuous_const.min continuous_id)
  have heq : ∀ᵐ ω ∂P, ∀ t, 0 ≤ t → clippedLogisticProcess u t ω = u t ω := by
    filter_upwards [logistic_range_ae_all P ℱ u hu hinit] with ω hω
    exact fun t ht => clippedLogisticProcess_eq_of_mem u t ω (hω t ht)
  refine ⟨IsLogisticSolution.of_indistinguishable P ℱ u _ hu ?_ ?_ heq, heq⟩
  · exact fun t ht => hclip.comp_stronglyMeasurable (hu.adapted t ht)
  · exact fun ω => hclip.comp_continuousOn' (hu.cont ω)

/-- A genuine null-set modification of the absorbing solution realizes every
continuity, adaptation, interval, and indistinguishability premise. At an
exceptional sample its value leaves the interval, so clipping really changes it. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ)) (u : ℝ → ℝ → ℝ),
    IsLogisticSolution (Measure.dirac (0 : ℝ)) ℱ u ∧
      (∀ t, 0 ≤ t → StronglyMeasurable[ℱ t] (u t)) ∧
      (∀ ω, ContinuousOn (fun t => u t ω) (Ici 0)) ∧
      (∀ ω, u 0 ω = 1) ∧
      (∀ᵐ ω ∂Measure.dirac (0 : ℝ), ∀ t, 0 ≤ t → u t ω = 1) ∧
      (∀ t, 0 ≤ t → ∀ᵐ ω ∂Measure.dirac (0 : ℝ), u t ω ∈ Icc (-1 : ℝ) 1) ∧
      u 1 1 = 2 ∧ clippedLogisticProcess u 1 1 = 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  let u : ℝ → ℝ → ℝ := fun t ω => 1 + ω ^ 2 * t
  have hm (t : ℝ) (ht : 0 ≤ t) : StronglyMeasurable[ℱ t] (u t) := by
    change StronglyMeasurable (fun ω : ℝ => 1 + ω ^ 2 * t)
    fun_prop
  have hc (ω : ℝ) : ContinuousOn (fun t => u t ω) (Ici 0) := by dsimp [u]; fun_prop
  have heq : ∀ᵐ ω ∂Measure.dirac (0 : ℝ), ∀ t, 0 ≤ t → u t ω = 1 := by
    rw [ae_dirac_eq]
    simp [u]
  have hMP := IsLogisticSolution.of_indistinguishable (Measure.dirac (0 : ℝ)) ℱ
    (fun _ _ => 1) u (isLogisticSolution_one _ _) hm hc heq
  have hinit (ω : ℝ) : u 0 ω ∈ Icc (-1 : ℝ) 1 := by norm_num [u]
  refine ⟨ℱ, u, hMP, hm, hc, fun ω => by simp [u], heq,
    logistic_range _ ℱ u hMP hinit, ?_, ?_⟩
  · norm_num [u]
  · norm_num [clippedLogisticProcess, u]

end Transformer.Homogenized
