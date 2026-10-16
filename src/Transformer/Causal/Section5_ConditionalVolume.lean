/-
# Lebesgue conditioning on the original frozen-system prefix

ArXiv:2411.04990v2, §5 and Appendix A, the initialization
argument for `thm:fixed_centers` on lifted real angles.
Splitting off the last original angle is a measurable
equivalence from the full angular tuple to prefix times last
angle. It preserves exactly the original product Lebesgue
measure. Its inverse is the original tuple concatenation.

For a Borel exceptional set, Fubini therefore converts null
last-angle fibers over almost every original prefix into a
null exceptional set of full original initializations. A
countable fiber is null in the original real Lebesgue measure.
The result then transfers to every original initialization
measure absolutely continuous with respect to that volume.
Measurability is an explicit premise of the conditioning
statements; no null-set property is built into a definition.
-/

import Transformer.Causal.FixedCenters
import Mathlib.MeasureTheory.Measure.Prod

open Real Set Filter MeasureTheory

namespace Transformer.Causal

/-- The measurable original angular coordinate split into its actual prefix and last angle.
Source: arXiv:2411.04990v2, §5 and Appendix A, conditioning on the causal prefix in `thm:fixed_centers`. -/
noncomputable def frozenAngleSplit (n : ℕ) : (Idx (n + 1) → ℝ) ≃ᵐ ((Idx n → ℝ) × ℝ) :=
  (MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) (Fin.last n)).trans
    (MeasurableEquiv.prodComm : ℝ × (Idx n → ℝ) ≃ᵐ (Idx n → ℝ) × ℝ)

/-- The original measurable split has precisely the original prefix and last-coordinate values.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual coordinate conditioning for `thm:fixed_centers`. -/
theorem frozenAngleSplit_apply (n : ℕ) (φ : Idx (n + 1) → ℝ) :
    frozenAngleSplit n φ = ((fun j => φ j.castSucc), φ (Fin.last n)) := by
  change ((fun j => φ ((Fin.last n).succAbove j)), φ (Fin.last n)) = _
  simp only [Fin.succAbove_last_apply]

/-- The inverse original coordinate split is exactly original last-angle concatenation.
Source: arXiv:2411.04990v2, §5 and Appendix A, reconstructing the original angular tuple in `thm:fixed_centers`. -/
theorem frozenAngleSplit_symm_apply (n : ℕ) (q : (Idx n → ℝ) × ℝ) :
    (frozenAngleSplit n).symm q = Fin.snoc q.1 q.2 := by
  apply (frozenAngleSplit n).injective
  rw [(frozenAngleSplit n).apply_symm_apply, frozenAngleSplit_apply]
  simp only [Fin.snoc_castSucc, Fin.snoc_last]

/-- The original prefix-and-last split preserves the original full Lebesgue measure as the actual prefix-times-last product volume.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original Lebesgue initialization used for the conditional argument of `thm:fixed_centers`. -/
theorem measurePreserving_frozenAngleSplit (n : ℕ) :
    MeasurePreserving (frozenAngleSplit n) volume
      ((volume : Measure (Idx n → ℝ)).prod (volume : Measure ℝ)) := by
  exact (Measure.measurePreserving_swap (μ := (volume : Measure ℝ)) (ν := (volume : Measure (Idx n → ℝ)))).comp
    (measurePreserving_piFinSuccAbove (fun _ : Idx (n + 1) => (volume : Measure ℝ)) (Fin.last n))

/-- The actual original prefix projection pulls back every original prefix-volume null exceptional set to an original full-volume null set.
Source: arXiv:2411.04990v2, §5 and Appendix A, lifting the original almost-everywhere prefix induction in `thm:fixed_centers`. -/
theorem quasiMeasurePreserving_frozenPrefix (n : ℕ) :
    Measure.QuasiMeasurePreserving (fun φ : Idx (n + 1) → ℝ => fun j : Idx n => φ j.castSucc)
      volume volume := by
  have h := (Measure.quasiMeasurePreserving_fst (μ := (volume : Measure (Idx n → ℝ)))
    (ν := (volume : Measure ℝ))).comp (measurePreserving_frozenAngleSplit n).quasiMeasurePreserving
  simpa only [Function.comp_def, frozenAngleSplit_apply] using h

/-- Actual prefix-and-last concatenation preserves the original product Lebesgue measure as full angular volume.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original product initialization in `thm:fixed_centers`. -/
theorem measurePreserving_frozenAngleSnoc (n : ℕ) :
    MeasurePreserving (fun q : (Idx n → ℝ) × ℝ => (Fin.snoc q.1 q.2 : Idx (n + 1) → ℝ))
      ((volume : Measure (Idx n → ℝ)).prod (volume : Measure ℝ)) volume := by
  have h := MeasurePreserving.symm (frozenAngleSplit n) (measurePreserving_frozenAngleSplit n)
  have hfun : ((frozenAngleSplit n).symm : ((Idx n → ℝ) × ℝ) → Idx (n + 1) → ℝ) =
      (fun q => Fin.snoc q.1 q.2) := funext (frozenAngleSplit_symm_apply n)
  rw [hfun] at h
  exact h

/-- The actual original concatenation map is measurable in the original real angular coordinates.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original conditional initialization in `thm:fixed_centers`. -/
theorem measurable_frozenAngleSnoc (n : ℕ) :
    Measurable (fun q : (Idx n → ℝ) × ℝ => (Fin.snoc q.1 q.2 : Idx (n + 1) → ℝ)) :=
  (measurePreserving_frozenAngleSnoc n).measurable

/-- Every Borel exceptional set of original tuples has a Borel last-angle fiber over every actual original prefix.
Source: arXiv:2411.04990v2, §5 and Appendix A, measurability of conditional original exceptional sets in `thm:fixed_centers`. -/
theorem measurableSet_frozenAngle_fiber (n : ℕ) (B : Set (Idx (n + 1) → ℝ)) (hB : MeasurableSet B)
    (p : Idx n → ℝ) : MeasurableSet {u : ℝ | Fin.snoc p u ∈ B} :=
  hB.preimage ((measurable_frozenAngleSnoc n).comp (measurable_const.prodMk measurable_id))

/-- The original full Lebesgue measure of a Borel set is its exact prefix-times-last product measure under the original concatenation map.
Source: arXiv:2411.04990v2, §5 and Appendix A, product conditioning for the original initialization of `thm:fixed_centers`. -/
theorem volume_frozenAngle_preimage (n : ℕ) (B : Set (Idx (n + 1) → ℝ)) (hB : MeasurableSet B) :
    volume B = ((volume : Measure (Idx n → ℝ)).prod (volume : Measure ℝ))
      ((fun q : (Idx n → ℝ) × ℝ => Fin.snoc q.1 q.2) ⁻¹' B) := by
  rw [← (measurePreserving_frozenAngleSnoc n).map_eq,
    Measure.map_apply (measurable_frozenAngleSnoc n) hB]

/-- A Borel original exceptional set with almost-everywhere null last-angle fibers is avoided by almost every original full angular initialization.
Source: arXiv:2411.04990v2, §5 and Appendix A, the measurable Fubini step for `thm:fixed_centers`. -/
theorem frozen_ae_of_ae_prefix_ae_fibers (n : ℕ) (B : Set (Idx (n + 1) → ℝ)) (hB : MeasurableSet B)
    (hfiber : ∀ᵐ p : Idx n → ℝ, ∀ᵐ u : ℝ, Fin.snoc p u ∉ B) : ∀ᵐ φ : Idx (n + 1) → ℝ, φ ∉ B := by
  have hpre : MeasurableSet {q : (Idx n → ℝ) × ℝ | Fin.snoc q.1 q.2 ∉ B} :=
    hB.compl.preimage (measurable_frozenAngleSnoc n)
  have hprod := (Measure.ae_prod_iff_ae_ae hpre).mpr hfiber
  have hmp := measurePreserving_frozenAngleSnoc n
  have hmap := (ae_map_iff hmp.measurable.aemeasurable hB.compl).mpr hprod
  simpa only [hmp.map_eq] using hmap

/-- Countable original exceptional last-angle fibers over almost every original prefix imply avoidance for almost every original full initialization.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original countable-fiber consequence used for `thm:fixed_centers`. -/
theorem frozen_ae_of_countable_bad_fibers (n : ℕ) (B : Set (Idx (n + 1) → ℝ)) (hB : MeasurableSet B)
    (hfiber : ∀ᵐ p : Idx n → ℝ, {u : ℝ | Fin.snoc p u ∈ B}.Countable) :
    ∀ᵐ φ : Idx (n + 1) → ℝ, φ ∉ B := by
  apply frozen_ae_of_ae_prefix_ae_fibers n B hB
  filter_upwards [hfiber] with p hp
  rw [ae_iff]
  simpa only [not_not] using hp.measure_zero (volume : Measure ℝ)

/-- The actual Borel exceptional set has zero original full Lebesgue measure when almost every original prefix has a countable exceptional last-angle fiber.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original null-set conclusion needed for `thm:fixed_centers`. -/
theorem frozen_volume_null_of_countable_bad_fibers (n : ℕ) (B : Set (Idx (n + 1) → ℝ)) (hB : MeasurableSet B)
    (hfiber : ∀ᵐ p : Idx n → ℝ, {u : ℝ | Fin.snoc p u ∈ B}.Countable) : volume B = 0 := by
  have h := ae_iff.mp (frozen_ae_of_countable_bad_fibers n B hB hfiber)
  simpa only [not_not, ofPred_mem_eq] using h

/-- The original exceptional set is avoided under every original initialization measure absolutely continuous with respect to full angular volume.
Source: arXiv:2411.04990v2, §5 and Appendix A, the final change from reference Lebesgue measure to the original initialization in `thm:fixed_centers`. -/
theorem frozen_mu_ae_of_countable_bad_fibers (n : ℕ) (B : Set (Idx (n + 1) → ℝ)) (hB : MeasurableSet B)
    (hfiber : ∀ᵐ p : Idx n → ℝ, {u : ℝ | Fin.snoc p u ∈ B}.Countable)
    (μ : Measure (Idx (n + 1) → ℝ)) (hμ : μ ≪ volume) : ∀ᵐ φ ∂μ, φ ∉ B := by
  have h := frozen_ae_of_countable_bad_fibers n B hB hfiber
  rw [ae_iff] at h ⊢
  exact hμ h

/-- The nonempty original angular hyperplane with last coordinate zero has singleton fibers over every original prefix and is avoided almost everywhere.
Source: arXiv:2411.04990v2, §5 and Appendix A, a genuine original real-coordinate null-fiber configuration. -/
example (n : ℕ) : ∀ᵐ φ : Idx (n + 1) → ℝ, φ (Fin.last n) ≠ 0 := by
  apply frozen_ae_of_countable_bad_fibers n {φ | φ (Fin.last n) = 0}
    (isClosed_eq (continuous_apply (Fin.last n)) continuous_const).measurableSet
  apply Eventually.of_forall
  intro p
  simpa only [mem_ofPred_eq, Fin.snoc_last, ← mem_singleton_iff, ofPred_mem_eq] using (countable_singleton (0 : ℝ))

/-- The same nonempty actual null-fiber configuration is avoided for an actual absolutely continuous restricted-volume initialization.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original absolutely continuous initialization convention. -/
example (n : ℕ) : ∀ᵐ φ ∂(volume : Measure (Idx (n + 1) → ℝ)).restrict (Metric.ball 0 1), φ (Fin.last n) ≠ 0 := by
  apply frozen_mu_ae_of_countable_bad_fibers n {φ | φ (Fin.last n) = 0}
    (isClosed_eq (continuous_apply (Fin.last n)) continuous_const).measurableSet _ _
    (Measure.absolutelyContinuous_of_le Measure.restrict_le_self)
  apply Eventually.of_forall
  intro p
  simpa only [mem_ofPred_eq, Fin.snoc_last, ← mem_singleton_iff, ofPred_mem_eq] using (countable_singleton (0 : ℝ))

/-- The original exceptional hyperplane used above contains an explicit original full angular initialization.
Source: arXiv:2411.04990v2, §5 and Appendix A, a nonempty null exceptional set in the original initialization space. -/
example (n : ℕ) : Set.Nonempty {φ : Idx (n + 1) → ℝ | φ (Fin.last n) = 0} := by
  refine ⟨fun _ => 0, ?_⟩
  rfl

end Transformer.Causal
