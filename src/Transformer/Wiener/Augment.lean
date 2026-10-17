import Transformer.Wiener.Defs
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.MeasureTheory.Measure.HasOuterApproxClosed
import Mathlib.Probability.Independence.Integration

/-!
# The usual augmentation keeps a Brownian motion Brownian

*Homogenized Transformers*, arXiv:2604.01978v1, §2.4 `note:probabilistic_setting` and App. A.2
`sec: cylindrical.wiener` work on a complete, right-continuous filtration and take for granted
that the usual augmentation `augment P ℱ` (`Wiener/Defs.lean`) of a filtration for which `B` is
Brownian is again one for which `B` is Brownian. The null sets do not disturb independence
(`indep_sup_nullSigma`); the right limits do not either, because the increment `B t - B s` is
the almost sure limit of the increments `B t - B u`, `u ↓ s`, each independent of `ℱ u`
(`indep_comap_of_tendsto`). The augmentation satisfies the usual conditions.
-/

open MeasureTheory ProbabilityTheory Filter Set
open scoped BoundedContinuousFunction NNReal Topology

namespace Transformer.Wiener

variable {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {ℱ : Filtration ℝ≥0 m}

/-- Every set of `m₂ ⊔ nullSigma P` agrees `P`-almost everywhere with a set of `m₂`. -/
theorem exists_measurableSet_ae_eq {m₂ : MeasurableSpace Ω} {s : Set Ω}
    (hs : MeasurableSet[m₂ ⊔ nullSigma P] s) : ∃ t, MeasurableSet[m₂] t ∧ s =ᵐ[P] t := by
  have h : m₂ ⊔ nullSigma P =
      MeasurableSpace.generateFrom ({t | MeasurableSet[m₂] t} ∪ {t | P t = 0}) := by
    rw [← MeasurableSpace.generateFrom_sup_generateFrom, @MeasurableSpace.generateFrom_measurableSet
      Ω m₂, nullSigma]
  rw [h] at hs
  induction s, hs using MeasurableSpace.generateFrom_induction with
  | hC t ht _ =>
    rcases ht with ht | ht
    · exact ⟨t, ht, EventuallyEqSet.rfl⟩
    · exact ⟨∅, MeasurableSet.empty, ae_eq_empty.2 ht⟩
  | empty => exact ⟨∅, MeasurableSet.empty, EventuallyEqSet.rfl⟩
  | compl t _ ih =>
    obtain ⟨t', ht', h⟩ := ih
    exact ⟨t'ᶜ, ht'.compl, h.compl⟩
  | iUnion s _ ih =>
    choose t ht h using ih
    exact ⟨⋃ i, t i, MeasurableSet.iUnion ht, EventuallyEqSet.countable_iUnion h⟩

/-- Adjoining the `P`-null sets to one side keeps two σ-algebras independent. -/
theorem indep_sup_nullSigma {m₁ m₂ : MeasurableSpace Ω} (h : Indep m₁ m₂ P) :
    Indep m₁ (m₂ ⊔ nullSigma P) P := by
  rw [Indep_iff] at h ⊢
  intro a s ha hs
  obtain ⟨t, ht, hst⟩ := exists_measurableSet_ae_eq hs
  rw [measure_congr hst, measure_congr (EventuallyEqSet.rfl.inter hst), h a t ha ht]

/-- `∫_A f ∘ Z = P(A) 𝔼[f ∘ Z]` for `σ(Z)` independent of `𝒢 ∋ A` and `f` bounded continuous. -/
theorem setIntegral_comp_eq_mul {𝒢 : MeasurableSpace Ω} (h𝒢 : 𝒢 ≤ m) {Z : Ω → ℝ}
    (hZ : Measurable[m] Z) (hind : Indep (MeasurableSpace.comap Z inferInstance) 𝒢 P)
    (f : ℝ →ᵇ ℝ) {A : Set Ω} (hA : MeasurableSet[𝒢] A) :
    ∫ ω in A, f (Z ω) ∂P = P.real A * ∫ ω, f (Z ω) ∂P := by
  have hf := f.continuous.measurable
  have hi : IndepFun (fun ω => f (Z ω)) (A.indicator (1 : Ω → ℝ)) P :=
    (IndepFun_iff_Indep _ _ _).2 (indep_of_indep_of_le_left
      (indep_of_indep_of_le_right hind (measurable_one.indicator hA).comap_le)
      (hf.comp (comap_measurable Z)).comap_le)
  calc ∫ ω in A, f (Z ω) ∂P = ∫ ω, f (Z ω) * A.indicator 1 ω ∂P := by
        rw [← integral_indicator (h𝒢 A hA)]
        congr 1
        ext ω
        by_cases hω : ω ∈ A <;> simp [hω]
    _ = (∫ ω, f (Z ω) ∂P) * ∫ ω, A.indicator 1 ω ∂P :=
        hi.integral_mul_eq_mul_integral (hf.comp hZ).aestronglyMeasurable
          ((measurable_one.indicator hA).mono h𝒢 le_rfl).aestronglyMeasurable
    _ = P.real A * ∫ ω, f (Z ω) ∂P := by rw [integral_indicator_one (h𝒢 A hA), mul_comm]

/-- Independence of a σ-algebra `𝒢` passes to almost sure limits of real random variables. -/
theorem indep_comap_of_tendsto [IsProbabilityMeasure P] {𝒢 : MeasurableSpace Ω} (h𝒢 : 𝒢 ≤ m)
    {X : ℕ → Ω → ℝ} {Y : Ω → ℝ} (hX : ∀ n, Measurable[m] (X n)) (hY : Measurable[m] Y)
    (hind : ∀ n, Indep (MeasurableSpace.comap (X n) inferInstance) 𝒢 P)
    (hlim : ∀ᵐ ω ∂P, Tendsto (fun n => X n ω) atTop (𝓝 (Y ω))) :
    Indep (MeasurableSpace.comap Y inferInstance) 𝒢 P := by
  rw [Indep_iff]
  rintro _ A ⟨S, hS, rfl⟩ hA
  have hmul n (f : ℝ →ᵇ ℝ) := setIntegral_comp_eq_mul h𝒢 (hX n) (hind n) f hA
  clear hind hA h𝒢 𝒢
  have : IsFiniteMeasure (P A • P.map Y) := Measure.smul_finite _ (measure_ne_top P A)
  have key : (P.restrict A).map Y = P A • P.map Y := by
    refine ext_of_forall_integral_eq_of_IsFiniteMeasure fun f => ?_
    rw [integral_map hY.aemeasurable f.continuous.aestronglyMeasurable, integral_smul_measure,
      integral_map hY.aemeasurable f.continuous.aestronglyMeasurable, smul_eq_mul]
    have hF n : AEStronglyMeasurable (fun ω => f (X n ω)) P :=
      (f.continuous.measurable.comp (hX n)).aestronglyMeasurable
    have hb n : ∀ᵐ ω ∂P, ‖f (X n ω)‖ ≤ ‖f‖ := ae_of_all _ fun ω => f.norm_coe_le_norm _
    have hl : ∀ᵐ ω ∂P, Tendsto (fun n => f (X n ω)) atTop (𝓝 (f (Y ω))) :=
      hlim.mono fun ω hω => (f.continuous.tendsto _).comp hω
    have h1 := tendsto_integral_of_dominated_convergence (fun _ => ‖f‖)
      (fun n => (hF n).restrict) (integrable_const (μ := P.restrict A) _)
      (fun n => ae_restrict_of_ae (hb n))
      (ae_restrict_of_ae hl)
    simp_rw [hmul _ f] at h1
    exact tendsto_nhds_unique h1
      ((tendsto_integral_of_dominated_convergence _ hF (integrable_const _) hb hl).const_mul _)
  have h := congrArg (fun μ => μ S) key
  simp only [Measure.map_apply hY hS, Measure.restrict_apply (hY hS), Measure.smul_apply,
    smul_eq_mul] at h
  rw [h, mul_comm]

/-- The usual augmentation of a filtration contains it. -/
theorem le_augment [P.IsComplete] (𝒢 : Filtration ℝ≥0 m) : 𝒢 ≤ augment P 𝒢 :=
  le_sup_left.trans (Filtration.le_rightCont _)

/-- `augment P 𝒢 s ≤ 𝒢 u ⊔ nullSigma P` for every `u > s`. -/
theorem augment_le_of_lt [P.IsComplete] (𝒢 : Filtration ℝ≥0 m) {s u : ℝ≥0} (hsu : s < u) :
    augment P 𝒢 s ≤ 𝒢 u ⊔ nullSigma P := by
  rw [augment, Filtration.rightCont_eq]
  exact iInf₂_le u hsu

/-- App. A.2, §2.4: an `(ℱ t)`-Brownian motion is Brownian for the usual augmentation of `ℱ`. -/
theorem isBrownian_augment [P.IsComplete] {v : ℝ≥0} {B : ℝ≥0 → Ω → ℝ}
    (hB : IsBrownian ℱ P v B) : IsBrownian (augment P ℱ) P v B where
  zero := hB.zero
  continuous := hB.continuous
  adapted t := (hB.adapted t).mono (le_augment ℱ t)
  hasLaw := hB.hasLaw
  indep s t hst := by
    have := hB.isProbabilityMeasure
    have hm r : Measurable (B r) := ((hB.adapted r).mono (ℱ.le r)).measurable
    rcases hst.eq_or_lt with rfl | hst
    · rw [sub_self, Pi.zero_def, MeasurableSpace.comap_const]
      exact indep_bot_left _
    obtain ⟨u, -, hu, hlim⟩ := exists_seq_strictAnti_tendsto' hst
    refine indep_comap_of_tendsto ((augment P ℱ).le s) (fun n => (hm t).sub (hm (u n)))
      ((hm t).sub (hm s)) (fun n => indep_of_indep_of_le_right
        (indep_sup_nullSigma (hB.indep (u n) t (hu n).2.le)) (augment_le_of_lt ℱ (hu n).1)) ?_
    filter_upwards [hB.continuous] with ω hω
    exact tendsto_const_nhds.sub ((hω.tendsto s).comp hlim)

/-- §2.4: the usual augmentation of a filtration satisfies the usual conditions. -/
theorem usualConditions_augment [hP : P.IsComplete] (𝒢 : Filtration ℝ≥0 m) :
    UsualConditions (augment P 𝒢) P where
  isComplete := hP
  measurableSet_of_null s hs := Filtration.le_rightCont _ 0 s
    (le_sup_right (a := 𝒢 0) s (MeasurableSpace.measurableSet_generateFrom hs))
  isRightContinuous := ⟨(Filtration.rightCont_self _).le⟩

/-- The hypotheses hold for the zero process under the Dirac mass on `Unit`, a complete
probability space, with the constant filtration. -/
example (f : ℝ →ᵇ ℝ) :
    haveI : (Measure.dirac ()).IsComplete := ⟨fun _ _ => Subsingleton.measurableSet⟩
    let 𝒢 := Filtration.const ℝ≥0 (inferInstance : MeasurableSpace Unit) le_rfl
    (∃ t, MeasurableSet[⊥] t ∧ (∅ : Set Unit) =ᵐ[Measure.dirac ()] t) ∧
      Indep (⊥ : MeasurableSpace Unit) (⊥ ⊔ nullSigma (Measure.dirac ())) (Measure.dirac ()) ∧
      ∫ _ in univ, f 0 ∂Measure.dirac () =
        (Measure.dirac ()).real (univ : Set Unit) * ∫ _, f 0 ∂Measure.dirac () ∧
      Indep (MeasurableSpace.comap (fun _ : Unit => (0 : ℝ)) inferInstance) ⊥ (Measure.dirac ()) ∧
      𝒢 ≤ augment (Measure.dirac ()) 𝒢 ∧
      augment (Measure.dirac ()) 𝒢 0 ≤ 𝒢 1 ⊔ nullSigma (Measure.dirac ()) ∧
      IsBrownian (augment (Measure.dirac ()) 𝒢) (Measure.dirac ()) 0 (fun _ _ => 0) ∧
      UsualConditions (augment (Measure.dirac ()) 𝒢) (Measure.dirac ()) := by
  have : (Measure.dirac ()).IsComplete := ⟨fun _ _ => Subsingleton.measurableSet⟩
  exact ⟨exists_measurableSet_ae_eq (MeasurableSpace.measurableSet_empty _),
    indep_sup_nullSigma (indep_bot_left _),
    setIntegral_comp_eq_mul (𝒢 := ⊥) bot_le (Z := fun _ => 0) measurable_const
      (indep_bot_right _) f (@MeasurableSet.univ _ ⊥),
    indep_comap_of_tendsto (X := fun _ _ => 0) bot_le (fun _ => measurable_const)
      measurable_const (fun _ => indep_bot_right _) (ae_of_all _ fun _ => tendsto_const_nhds),
    le_augment _, augment_le_of_lt _ zero_lt_one, isBrownian_augment (isBrownian_zero _),
    usualConditions_augment _⟩

end Transformer.Wiener
