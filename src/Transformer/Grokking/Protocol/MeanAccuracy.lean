module

public import Transformer.Grokking.Protocol.DivisionMeasurable
public import Transformer.Grokking.Protocol.LabelProbability
public import Transformer.Grokking.Protocol.ReadoutWitness
public import Transformer.Grokking.Protocol.RefutationModes

/-!
# The mean held-out accuracy of division stays below `1`

Source: arXiv:2201.02177v1 (Power et al.), Figure 1; `experiments/grok_division/PROTOCOL.md`,
*The split*; `review/grok_division_convergence.md`, section 2 and bridge 2. The modes of
convergence are those of `Transformer.Grokking.Protocol.Target`; the bound used is
`uniformPerm_le_of_subset_compl_labelled` of `LabelProbability`.

The held-out accuracy is a measurable function of the sample (`DivisionMeasurable`), so its
mean is the integral of a bounded measurable function. A wrong held-out answer keeps the
accuracy at most `1 - 1/9312`, and on the splits that are not labelled one occurs with a
conditional probability of at least one half at every update. So over the runs whose split is
in a set `S`, the accuracy integrates to at most the probability of `S` less `1/18624` of the
probability of the splits of `S` that are not labelled (`setIntegral_divisionAccuracy_le`),
at every update, under Adam and under AdamW at every weight decay; the same holds for every
set of splits on which a held-out answer is wrong half the time
(`setIntegral_divisionAccuracy_le_of_half`). With `S` all the splits:
the mean held-out accuracy is at most `1` less a positive constant at every update
(`integral_divisionAccuracy_le`), so it does not tend to `1`
(`not_division_tendsToOneInMean`): the third mode fails as the other two do. And given a
condition on the split that admits a split that is not labelled, the mean accuracy given the
condition does not tend to `1` (`not_tendsto_setIntegral_divisionAccuracy`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- A wrong held-out answer keeps the held-out accuracy at most `1 - 1/9312`. -/
lemma divisionAccuracy_le_of_mem_wrongAt (decay : ℝ) (t : ℕ)
    {ω : Sample (97 * 96) 4656 (Param paperSizes)} (hω : ω ∈ wrongAt decay t) :
    divisionAccuracy decay t ω ≤ 1 - 1 / 9312 := by
  have h := wrongAt_subset decay t hω
  have h1 := (heldoutAccuracy_mem answerLogits (fun x => residueToken x.1) (Hyper.source decay)
    divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 t ω).2
  rw [Set.mem_ofPred_eq, edist_dist, ENNReal.ofReal_le_ofReal_iff dist_nonneg, Real.dist_eq,
    abs_sub_comm, abs_of_nonneg (sub_nonneg.2 h1)] at h
  linarith

/-- **A set of splits on which a held-out answer is wrong half the time lowers the mean
accuracy of every condition that contains it**: over the runs whose split is in `S` the
held-out accuracy integrates to at most the probability of `S` less `1/18624` of that of the
set. -/
theorem setIntegral_divisionAccuracy_le_of_half (decay : ℝ) (t : ℕ)
    {A S : Set (Equiv.Perm (Fin (97 * 96)))} (hAS : A ⊆ S)
    (hA : uniformPerm (97 * 96) A ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn A ∩ wrongAt decay t)) :
    ∫ ω in splitIn S, divisionAccuracy decay t ω ∂law (97 * 96) 4656 (initMeasure paperSizes) ≤
      (uniformPerm (97 * 96)).real S - (uniformPerm (97 * 96)).real A / 18624 := by
  set μ := law (97 * 96) 4656 (initMeasure paperSizes) with hμ
  have hW := measurableSet_wrongAt decay t
  have hg : Integrable (fun ω => 1 - (wrongAt decay t).indicator (fun _ => (1 / 9312 : ℝ)) ω)
      (μ.restrict (splitIn S)) := (integrable_const 1).sub ((integrable_const _).indicator hW)
  have hle : divisionAccuracy decay t ≤
      fun ω => 1 - (wrongAt decay t).indicator (fun _ => (1 / 9312 : ℝ)) ω := fun ω => by
    by_cases hω : ω ∈ wrongAt decay t
    · simp only [Set.indicator_of_mem hω]
      exact divisionAccuracy_le_of_mem_wrongAt decay t hω
    · simp only [Set.indicator_of_notMem hω, sub_zero]
      exact (heldoutAccuracy_mem answerLogits (fun x => residueToken x.1) (Hyper.source decay)
        divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 t ω).2
  have h1 := integral_mono (integrable_divisionAccuracy decay t _) hg hle
  rw [integral_sub (integrable_const 1) ((integrable_const _).indicator hW), integral_const,
    integral_indicator_const _ hW, smul_eq_mul, mul_one, smul_eq_mul,
    measureReal_restrict_apply_univ, measureReal_restrict_apply hW] at h1
  have hu : μ.real (splitIn S) = (uniformPerm (97 * 96)).real S := by
    rw [measureReal_def, measureReal_def, hμ, law, Measure.prod_prod, measure_univ, mul_one]
  have h3 : (uniformPerm (97 * 96)).real A ≤ 2 * μ.real (wrongAt decay t ∩ splitIn S) := by
    have hsub : splitIn A ∩ wrongAt decay t ⊆ wrongAt decay t ∩ splitIn S :=
      fun ω hω => ⟨hω.2, hAS hω.1.1, hω.1.2⟩
    have h4 : uniformPerm (97 * 96) A ≤ 2 * μ (wrongAt decay t ∩ splitIn S) :=
      hA.trans (mul_le_mul_right (measure_mono hsub) 2)
    have h5 := ENNReal.toReal_mono (by finiteness) h4
    rwa [ENNReal.toReal_mul, ENNReal.toReal_ofNat] at h5
  rw [hu] at h1
  linarith

/-- **Over the runs whose split is in `S` the held-out accuracy integrates to at most the
probability of `S` less `1/18624` of that of the splits of `S` that are not labelled**, at
every update. -/
theorem setIntegral_divisionAccuracy_le (decay : ℝ) (t : ℕ)
    (S : Set (Equiv.Perm (Fin (97 * 96)))) :
    ∫ ω in splitIn S, divisionAccuracy decay t ω ∂law (97 * 96) 4656 (initMeasure paperSizes) ≤
      (uniformPerm (97 * 96)).real S - (uniformPerm (97 * 96)).real (S \ labelled) / 18624 :=
  setIntegral_divisionAccuracy_le_of_half decay t Set.sdiff_subset
    (uniformPerm_le_of_subset_compl_labelled decay t fun _ hπ => hπ.2)

/-- **The mean held-out accuracy is at most `1` less a positive constant, at every update**:
`1/18624` of the probability that the split is not labelled. -/
theorem integral_divisionAccuracy_le (decay : ℝ) (t : ℕ) :
    ∫ ω, divisionAccuracy decay t ω ∂law (97 * 96) 4656 (initMeasure paperSizes) ≤
      1 - (uniformPerm (97 * 96)).real labelledᶜ / 18624 := by
  have h := setIntegral_divisionAccuracy_le decay t Set.univ
  rwa [splitIn, Set.univ_prod_univ, setIntegral_univ, probReal_univ,
    ← Set.compl_eq_univ_sdiff] at h

/-- A set of splits that contains one has a positive probability, as a real number. -/
lemma uniformPerm_real_pos_of_mem {S : Set (Equiv.Perm (Fin (97 * 96)))}
    {π : Equiv.Perm (Fin (97 * 96))} (hπ : π ∈ S) : 0 < (uniformPerm (97 * 96)).real S :=
  ENNReal.toReal_pos (uniformPerm_pos_of_mem hπ).ne' (measure_ne_top _ _)

/-- **The mean held-out accuracy does not tend to `1`**, for Adam and for AdamW at every
weight decay. -/
theorem not_division_tendsToOneInMean (decay : ℝ) :
    ¬ TendsToOneInMean (law (97 * 96) 4656 (initMeasure paperSizes))
      (divisionAccuracy decay) := fun h => by
  have hle := le_of_tendsto' h fun t => integral_divisionAccuracy_le decay t
  have hpos := uniformPerm_real_pos_of_mem (S := labelledᶜ) one_notMem_labelled
  linarith [div_pos hpos (by norm_num : (0 : ℝ) < 18624)]

/-- **Under a condition on the split that contains a set, not empty, on which a held-out
answer is wrong half the time at every update, the mean accuracy given the condition does not
tend to `1`**: the integral of the accuracy over the runs that meet the condition does not
tend to the probability of the condition. -/
theorem not_tendsto_setIntegral_of_half (decay : ℝ) {A S : Set (Equiv.Perm (Fin (97 * 96)))}
    (hAS : A ⊆ S) {π : Equiv.Perm (Fin (97 * 96))} (hπ : π ∈ A)
    (hA : ∀ t : ℕ, uniformPerm (97 * 96) A ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn A ∩ wrongAt decay t)) :
    ¬ Tendsto (fun t => ∫ ω in splitIn S, divisionAccuracy decay t ω
      ∂law (97 * 96) 4656 (initMeasure paperSizes)) atTop
        (𝓝 ((uniformPerm (97 * 96)).real S)) := fun h => by
  have hle := le_of_tendsto' h fun t =>
    setIntegral_divisionAccuracy_le_of_half decay t hAS (hA t)
  have hpos := uniformPerm_real_pos_of_mem hπ
  linarith [div_pos hpos (by norm_num : (0 : ℝ) < 18624)]

/-- **Under a condition on the split that admits a split that is not labelled the mean
accuracy given the condition does not tend to `1`.** -/
theorem not_tendsto_setIntegral_divisionAccuracy (decay : ℝ)
    {S : Set (Equiv.Perm (Fin (97 * 96)))} {π : Equiv.Perm (Fin (97 * 96))} (hπS : π ∈ S)
    (hπ : π ∉ labelled) :
    ¬ Tendsto (fun t => ∫ ω in splitIn S, divisionAccuracy decay t ω
      ∂law (97 * 96) 4656 (initMeasure paperSizes)) atTop
        (𝓝 ((uniformPerm (97 * 96)).real S)) :=
  not_tendsto_setIntegral_of_half decay (Set.singleton_subset_iff.2 hπS) (Set.mem_singleton π)
    fun t => uniformPerm_singleton_le decay t hπ

/-! The witnesses: the sample `zeroSample`, whose run stays at zero, where every logit is `0`
and the first token is predicted, which is no residue; and the condition `anchored`, which the
split `1` meets without being labelled. -/

/-- The run from zero has a wrong held-out answer at every update. -/
lemma zeroSample_mem_wrongAt (decay : ℝ) (t : ℕ) : zeroSample ∈ wrongAt decay t := by
  obtain ⟨x, hx⟩ := heldout_nonempty equations (by norm_num : 4656 < 97 * 96) zeroSample.1
  refine ⟨x, hx, fun h => ?_⟩
  have h0 : trajectory (Hyper.source decay) divisionGrad equations
      (by norm_num : 4656 ≤ 97 * 96) 512 zeroSample t = 0 := divisionRun_zeroSample decay t
  have hz : answerLogits 0 x = fun _ => 0 := by
    funext c
    simp [answerLogits, logits]
  rw [h0, hz, predict_eq_iff] at h
  exact lt_irrefl _ (h.2 0 (Fin.lt_def.2 (by simp [residueToken])))

example (decay : ℝ) (t : ℕ) : divisionAccuracy decay t zeroSample ≤ 1 - 1 / 9312 :=
  divisionAccuracy_le_of_mem_wrongAt decay t (zeroSample_mem_wrongAt decay t)

example : 0 < (uniformPerm (97 * 96)).real labelledᶜ :=
  uniformPerm_real_pos_of_mem one_notMem_labelled

example (decay : ℝ) : ¬ Tendsto (fun t => ∫ ω in splitIn anchored, divisionAccuracy decay t ω
    ∂law (97 * 96) 4656 (initMeasure paperSizes)) atTop
      (𝓝 ((uniformPerm (97 * 96)).real anchored)) :=
  not_tendsto_setIntegral_divisionAccuracy decay one_mem_anchored one_notMem_labelled

end Transformer.Grokking.Protocol
