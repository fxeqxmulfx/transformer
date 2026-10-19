module

public import Transformer.Causal.Section4_TwoTokenConvergence
public import Mathlib.MeasureTheory.Constructions.Pi

/-!
# Conditioning the original uniform tuple law on its prefix

ArXiv:2411.04990v2, §4, proof of `thm1`, law of total probability.
The original finite product law splits exactly into its last-token
marginal and its preceding-token product. The measurable equivalence
is the original finite-tuple decomposition; its inverse appends the
actual last token with `Fin.snoc`. Fubini therefore transfers an
almost-everywhere conditional statement back to the original law.

In particular, a Borel exceptional set is null if its conditional
last-token fibers have at most one point for almost every original
prefix and the actual original marginal has zero singleton masses.
The marginal's atomlessness and the dynamics' singleton-fiber result
remain explicit inputs here. The original uniform invariant marginal
and the actual antipodal two-token fibers give concrete witnesses.
-/

@[expose] public section

open MeasureTheory Filter

namespace Transformer.Causal

variable {d n : ℕ}

/-- A conditional set with at most one token is null under an actual marginal with zero singleton masses.
Source: arXiv:2411.04990v2, §4, the conditional null exceptional sets in `thm1`. -/
theorem csaMeasure_subsingleton_eq_zero (σ : Measure (SSphere d)) (hσ : ∀ x, σ {x} = 0)
    (S : Set (SSphere d)) (hS : S.Subsingleton) : σ S = 0 := by
  rcases S.eq_empty_or_nonempty with h | ⟨x, hx⟩
  · rw [h, measure_empty]
  · apply measure_mono_null (t := {x}) _ (hσ x)
    intro y hy
    exact Set.mem_singleton_iff.mpr (hS hy hx)

/-- Appending an actual last token to its original prefix is measurable in both original factors.
Source: arXiv:2411.04990v2, §4, the conditional original tuple of `thm1`. -/
theorem measurable_csaSnoc : Measurable
    (fun z : SphereTuple d n × SSphere d => (Fin.snoc z.1 z.2 : SphereTuple d (n + 1))) := by
  apply measurable_pi_iff.mpr
  intro i
  refine Fin.lastCases ?_ (fun j => ?_) i
  · simpa only [Fin.snoc_last] using
      (measurable_snd : Measurable (fun z : SphereTuple d n × SSphere d => z.2))
  · simpa only [Function.comp_def, Fin.snoc_castSucc] using (measurable_pi_apply j).comp
      (measurable_fst : Measurable (fun z : SphereTuple d n × SSphere d => z.1))

/-- The inverse original last-coordinate measurable split is exactly the actual finite-tuple append operation.
Source: arXiv:2411.04990v2, §4, conditioning on the original prefix in `thm1`. -/
theorem csaLastSplit_symm (z : SSphere d × SphereTuple d n) :
    (MeasurableEquiv.piFinSuccAbove (fun _ : Idx (n + 1) => SSphere d) (Fin.last n)).symm z =
      Fin.snoc z.2 z.1 := by
  rw [MeasurableEquiv.piFinSuccAbove_symm_apply, Fin.insertNthEquiv_last]
  rfl

/-- The original last-coordinate split preserves exactly the original finite product probability law.
Source: arXiv:2411.04990v2, §4, the original independent initialization in `thm1`. -/
theorem measurePreserving_csaLastSplit (σ : Measure (SSphere d)) [IsProbabilityMeasure σ] :
    MeasurePreserving (MeasurableEquiv.piFinSuccAbove (fun _ : Idx (n + 1) => SSphere d) (Fin.last n))
      (Measure.pi (fun _ : Idx (n + 1) => σ)) (σ.prod (Measure.pi (fun _ : Idx n => σ))) :=
  measurePreserving_piFinSuccAbove (fun _ : Idx (n + 1) => σ) (Fin.last n)

/-- Taking the actual original prefix preserves its original finite product marginal law.
Source: arXiv:2411.04990v2, §4, independence of original preceding tokens in `thm1`. -/
theorem measurePreserving_csaPrefix (σ : Measure (SSphere d)) [IsProbabilityMeasure σ] :
    MeasurePreserving (fun X : SphereTuple d (n + 1) => (fun j : Idx n => X j.castSucc))
      (Measure.pi (fun _ : Idx (n + 1) => σ)) (Measure.pi (fun _ : Idx n => σ)) := by
  let e := MeasurableEquiv.piFinSuccAbove (fun _ : Idx (n + 1) => SSphere d) (Fin.last n)
  have heq : (fun X : SphereTuple d (n + 1) => (fun j : Idx n => X j.castSucc)) = Prod.snd ∘ e := by
    funext X j
    change X j.castSucc = X ((Fin.last n).succAbove j)
    rw [Fin.succAbove_last_apply]
  rw [heq]
  exact (measurePreserving_snd (μ := σ) (ν := Measure.pi (fun _ : Idx n => σ))).comp
    (measurePreserving_csaLastSplit (n := n) σ)

/-- Every almost-sure original prefix assertion pulls back to the original full tuple, without a Borel predicate assumption.
Source: arXiv:2411.04990v2, §4, the induction over original prefix dynamics in `thm1`. -/
theorem csa_ae_prefix_of_ae (σ : Measure (SSphere d)) [IsProbabilityMeasure σ]
    (p : SphereTuple d n → Prop) (h : ∀ᵐ Y ∂Measure.pi (fun _ : Idx n => σ), p Y) :
    ∀ᵐ X ∂Measure.pi (fun _ : Idx (n + 1) => σ), p (fun j => X j.castSucc) := by
  have hm := measurePreserving_csaPrefix (n := n) σ
  apply ae_of_ae_map hm.measurable.aemeasurable
  rwa [hm.map_eq]

/-- An actual almost-sure conditional assertion for original prefixes and last tokens holds for the original tuple law.
Source: arXiv:2411.04990v2, §4, the law-of-total-probability step for `thm1`. -/
theorem csa_ae_snoc_of_ae_ae (σ : Measure (SSphere d)) [IsProbabilityMeasure σ]
    (p : SphereTuple d (n + 1) → Prop) (hp : MeasurableSet {X | p X})
    (h : ∀ᵐ Y ∂Measure.pi (fun _ : Idx n => σ), ∀ᵐ z ∂σ, p (Fin.snoc Y z)) :
    ∀ᵐ X ∂Measure.pi (fun _ : Idx (n + 1) => σ), p X := by
  let e := MeasurableEquiv.piFinSuccAbove (fun _ : Idx (n + 1) => SSphere d) (Fin.last n)
  have he := measurePreserving_csaLastSplit (n := n) σ
  have hme : MeasurableSet {z : SSphere d × SphereTuple d n | p (e.symm z)} :=
    hp.preimage e.symm.measurable
  have hs : ∀ᵐ Y ∂Measure.pi (fun _ : Idx n => σ), ∀ᵐ z ∂σ, p (e.symm (z, Y)) := by
    simpa only [e, csaLastSplit_symm] using h
  have hprod : ∀ᵐ z ∂σ.prod (Measure.pi (fun _ : Idx n => σ)), p (e.symm z) :=
    (Measure.ae_prod_iff_ae_ae hme).mpr ((Measure.ae_ae_comm hme).mpr hs)
  have hinv := MeasurePreserving.symm e he
  rw [← hinv.map_eq]
  exact (ae_map_iff e.symm.measurable.aemeasurable hp).mpr hprod

/-- A Borel original exceptional set is null when its almost-everywhere original conditional fibers have at most one token.
Source: arXiv:2411.04990v2, §4, eliminating the conditional antipodal exception in `thm1`. -/
theorem csa_ae_not_mem_of_subsingleton_snoc_fibers (σ : Measure (SSphere d)) [IsProbabilityMeasure σ]
    (hσ : ∀ x, σ {x} = 0) (B : Set (SphereTuple d (n + 1))) (hB : MeasurableSet B)
    (hf : ∀ᵐ Y ∂Measure.pi (fun _ : Idx n => σ), {z : SSphere d | Fin.snoc Y z ∈ B}.Subsingleton) :
    ∀ᵐ X ∂Measure.pi (fun _ : Idx (n + 1) => σ), X ∉ B := by
  apply csa_ae_snoc_of_ae_ae σ (fun X => X ∉ B) hB.compl
  filter_upwards [hf] with Y hY
  rw [ae_iff]
  simpa only [not_not] using csaMeasure_subsingleton_eq_zero σ hσ _ hY

/-- The actual two-token antipodal relation is Borel in the original product sphere.
Source: arXiv:2411.04990v2, §4, the exceptional two-token set of `thm1`. -/
theorem measurableSet_csaAntipodalPair_two :
    MeasurableSet {X : SphereTuple d 2 | X 1 = Perspective.antipode d (X 0)} :=
  measurableSet_eq_fun (measurable_pi_apply 1)
    (continuous_csaAntipode.measurable.comp (measurable_pi_apply 0))

/-- Each actual two-token antipodal conditional fiber has exactly its one original antipodal token at most.
Source: arXiv:2411.04990v2, §4, conditioning the two-token exception in `thm1`. -/
theorem subsingleton_csaSnoc_antipode_fiber (Y : SphereTuple d 1) :
    {z : SSphere d | (Fin.snoc Y z : SphereTuple d 2) 1 =
      Perspective.antipode d ((Fin.snoc Y z : SphereTuple d 2) 0)}.Subsingleton := by
  change {z : SSphere d | (Fin.snoc Y z : SphereTuple d 2) (Fin.last 1) =
    Perspective.antipode d ((Fin.snoc Y z : SphereTuple d 2) (0 : Idx 1).castSucc)}.Subsingleton
  simp only [Fin.snoc_castSucc, Fin.snoc_last]
  intro x hx y hy
  exact hx.trans hy.symm

/-- The actual invariant uniform circle marginal jointly realizes probability and zero singleton masses.
Source: arXiv:2411.04990v2, §4, the original independent uniform initialization of `thm1`. -/
example : IsProbabilityMeasure (MeanField.uniformLaw 2) ∧
    (∀ x : SSphere 2, MeanField.uniformLaw 2 {x} = 0) := by
  have h := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  have := h.1
  exact ⟨h.1, fun x => Perspective.measure_singleton_eq_zero_of_invariant 2 le_rfl _ h.2 x⟩

/-- The same actual law and nonempty antipodal fibers realize the full conditional-null argument.
Source: arXiv:2411.04990v2, §4, the original two-token exception in `thm1`. -/
example : ∀ᵐ X ∂Measure.pi (fun _ : Idx 2 => MeanField.uniformLaw 2),
    X 1 ≠ Perspective.antipode 2 (X 0) := by
  have h := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  have := h.1
  exact csa_ae_not_mem_of_subsingleton_snoc_fibers _
    (fun x => Perspective.measure_singleton_eq_zero_of_invariant 2 le_rfl _ h.2 x) _
    measurableSet_csaAntipodalPair_two (ae_of_all _ subsingleton_csaSnoc_antipode_fiber)

end Transformer.Causal
