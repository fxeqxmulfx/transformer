module

public import Transformer.Grokking.Protocol.Causal
public import Transformer.Grokking.Protocol.Exchange

/-!
# The target for division is false without a condition on the split

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 and the appendix *Additional experimental
details*, "we chose a fraction of all available equations at random"; openai/grok at 3d64b1d8;
`experiments/grok_division/PROTOCOL.md`. The target is that of `division_generalizes_iff`.
The paper claims nothing about every split; this file shows that no theorem can.

On an uncovered split the tokens of the residues `1` and `2` are read by no training equation
and are the target of none, and the equations `1 / 3 = 65` and `2 / 3 = 33` (`hiddenOne`,
`hiddenTwo`) are held out (`mem_heldout_of_not_avoids`). Exchanging the two tokens turns the
run into the exchanged run (`trajectory_exchange`), and the logits after `=` of `1 / 3` at
exchanged parameters are those of `2 / 3` with the two tokens renamed
(`answerLogits_exchange`): the decoder is causal and has read the same row. If `2 / 3` is
answered `33` at some parameters, the logit of `33` is the largest there, so `1 / 3` is not
answered `65` at the exchanged ones (`not_both_predicted`). Hence, at every update, a sample
with an uncovered split or its exchange has a wrong held-out answer (`mem_wrongAt_or`).

The exchange preserves the law of a run. Therefore:

* at every update the samples with a wrong held-out answer have at least half the probability
  of the uncovered splits (`uniformPerm_uncovered_le`), which is positive;
* a condition on the split under which, almost surely, every held-out equation is eventually
  answered excludes every uncovered split (`not_mem_of_mem_uncovered`);
* without a condition the target fails: `not_division_generalizes`, for Adam and for AdamW at
  every weight decay.

The same holds for any two residues in place of `1` and `2`. A true statement has to exclude
such splits or to hold with a probability below `1`; which splits suffice is open.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter MeasureTheory

/-- `1 / 3 = 65` modulo 97: an equation with the hidden numerator `1`. -/
def hiddenOne : Equation := (65, 2)

/-- `2 / 3 = 33` modulo 97: an equation with the hidden numerator `2`. -/
def hiddenTwo : Equation := (33, 2)

/-- **On an uncovered split an equation that shows a hidden residue is held out.** -/
lemma mem_heldout_of_not_avoids {π : Equiv.Perm (Fin (97 * 96))} (hπ : π ∈ uncovered)
    {x : Equation} (hx : ¬ Avoids x) : x ∈ heldout equations 4656 π := by
  obtain ⟨j₀, rfl⟩ := equations_bijective.surjective x
  refine Finset.mem_image.2 ⟨π.symm j₀, Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩,
    by rw [Equiv.apply_symm_apply]⟩
  by_contra hlt
  have h := hπ ⟨(π.symm j₀ : ℕ), not_le.1 hlt⟩
  have hcast : Fin.castLE (by norm_num : 4656 ≤ 97 * 96) ⟨(π.symm j₀ : ℕ), not_le.1 hlt⟩ =
      π.symm j₀ := Fin.ext rfl
  rw [trainAt, hcast, Equiv.apply_symm_apply] at h
  exact hx h

/-- **The logits after `=` of `1 / 3` at exchanged parameters are those of `2 / 3`**, with the
two hidden tokens renamed. -/
lemma answerLogits_exchange (θ : Param paperSizes → ℝ) (c : Fin 239) :
    answerLogits (θ ∘ exchange) hiddenOne c = answerLogits θ hiddenTwo (hiddenSwap c) := by
  have hrow : ∀ s : Fin 6, s ≤ 4 → (hiddenSwap ∘ read hiddenOne) s = read hiddenTwo s := by
    decide
  have h := logits_comp_paramPerm (S := paperSizes) (τ := hiddenSwap) (θ := θ)
    (act := relu paperSizes) (read hiddenOne) (4 : Fin 6) c
  exact h.trans (congrFun (logits_congr (S := paperSizes) (t := (4 : Fin 6)) hrow) (hiddenSwap c))

/-- **`2 / 3` at some parameters and `1 / 3` at the exchanged ones are not both answered.** -/
theorem not_both_predicted (θ : Param paperSizes → ℝ) :
    ¬ (predict (answerLogits θ hiddenTwo) = residueToken 33 ∧
      predict (answerLogits (θ ∘ exchange) hiddenOne) = residueToken 65) := fun ⟨h2, h1⟩ => by
  have hle := (predict_eq_iff.1 h2).1 (residueToken 65)
  have hlt := (predict_eq_iff.1 h1).2 (residueToken 33) (by decide)
  have e33 : hiddenSwap (residueToken 33) = residueToken 33 := by decide
  have e65 : hiddenSwap (residueToken 65) = residueToken 65 := by decide
  rw [answerLogits_exchange, answerLogits_exchange, e33, e65] at hlt
  exact absurd hle (not_le.2 hlt)

/-- The samples with a held-out equation answered wrongly after `t` updates. -/
def wrongAt (decay : ℝ) (t : ℕ) : Set (Sample (97 * 96) 4656 (Param paperSizes)) :=
  {ω | ∃ x ∈ heldout equations 4656 ω.1, predict (answerLogits (trajectory (Hyper.source decay)
    divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 ω t) x) ≠ residueToken x.1}

/-- **A sample with an uncovered split or its exchange has a wrong held-out answer**, at every
update. -/
theorem mem_wrongAt_or (decay : ℝ) (t : ℕ) {ω : Sample (97 * 96) 4656 (Param paperSizes)}
    (hω : ω.1 ∈ uncovered) : ω ∈ wrongAt decay t ∨ exchangeSample ω ∈ wrongAt decay t := by
  by_contra hnot
  rw [not_or] at hnot
  simp only [wrongAt, Set.mem_ofPred_eq, not_exists, not_and, not_not] at hnot
  have hx2 := hnot.1 hiddenTwo (mem_heldout_of_not_avoids hω (by decide))
  have hx1 := hnot.2 hiddenOne (mem_heldout_of_not_avoids hω (by decide))
  rw [show exchangeSample ω = (ω.1, ω.2.1 ∘ exchange, ω.2.2) from rfl,
    trajectory_exchange decay hω] at hx1
  exact not_both_predicted _ ⟨hx2, hx1⟩

/-- **At every update the samples with a wrong held-out answer have at least half the
probability of the uncovered splits.** -/
theorem uniformPerm_uncovered_le (decay : ℝ) (t : ℕ) :
    uniformPerm (97 * 96) uncovered ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (wrongAt decay t) := by
  set μ := law (97 * 96) 4656 (initMeasure paperSizes) with hμ
  have hF : μ (uncovered ×ˢ Set.univ) = uniformPerm (97 * 96) uncovered := by
    rw [hμ, law, Measure.prod_prod, measure_univ, mul_one]
  have hpre : μ (exchangeSample ⁻¹' wrongAt decay t) ≤ μ (wrongAt decay t) :=
    calc μ (exchangeSample ⁻¹' wrongAt decay t)
        ≤ μ (exchangeSample ⁻¹' toMeasurable μ (wrongAt decay t)) :=
          measure_mono (Set.preimage_mono (subset_toMeasurable _ _))
      _ = μ (toMeasurable μ (wrongAt decay t)) :=
          measurePreserving_exchangeSample.measure_preimage
            (measurableSet_toMeasurable _ _).nullMeasurableSet
      _ = μ (wrongAt decay t) := measure_toMeasurable _
  calc uniformPerm (97 * 96) uncovered = μ (uncovered ×ˢ Set.univ) := hF.symm
    _ ≤ μ (wrongAt decay t ∪ exchangeSample ⁻¹' wrongAt decay t) :=
        measure_mono fun ω hω => mem_wrongAt_or decay t hω.1
    _ ≤ μ (wrongAt decay t) + μ (exchangeSample ⁻¹' wrongAt decay t) := measure_union_le _ _
    _ ≤ μ (wrongAt decay t) + μ (wrongAt decay t) := add_le_add le_rfl hpre
    _ = 2 * μ (wrongAt decay t) := (two_mul _).symm

/-- **A condition on the split that makes the target true excludes every uncovered split.** If
for almost every sample whose split is in `S` every held-out equation is eventually answered,
then no uncovered split is in `S`. -/
theorem not_mem_of_mem_uncovered (decay : ℝ) {S : Set (Equiv.Perm (Fin (97 * 96)))}
    (hS : ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω.1 ∈ S →
      ∀ᶠ t in atTop, ω ∉ wrongAt decay t)
    {π : Equiv.Perm (Fin (97 * 96))} (hπ : π ∈ uncovered) : π ∉ S := fun hπS => by
  have h2 := measurePreserving_exchangeSample.quasiMeasurePreserving.ae hS
  have hnull : ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes),
      ω ∉ ({π} : Set (Equiv.Perm (Fin (97 * 96)))) ×ˢ (Set.univ :
        Set ((Param paperSizes → ℝ) × (ℕ → Equiv.Perm (Fin 4656)))) := by
    filter_upwards [hS, h2] with ω hω1 hω2 hmem
    have hπω : ω.1 = π := hmem.1
    obtain ⟨t, ht1, ht2⟩ := ((hω1 (hπω ▸ hπS)).and (hω2 (hπω ▸ hπS))).exists
    rcases mem_wrongAt_or decay t (hπω ▸ hπ) with h | h
    exacts [ht1 h, ht2 h]
  have hzero := measure_eq_zero_iff_ae_notMem.2 hnull
  rw [law, Measure.prod_prod, measure_univ, mul_one] at hzero
  exact (uniformPerm_pos_of_mem (Set.mem_singleton π)).ne' hzero

/-- **The target of Figure 1 is false as it stands**, for Adam and for AdamW at every weight
decay: with positive probability the split is uncovered, and then the held-out accuracy does
not tend to `1` for the sample or for its exchange. -/
theorem not_division_generalizes (decay : ℝ) :
    ¬ Generalizes (initMeasure paperSizes) answerLogits (fun x => residueToken x.1)
      (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 :=
  fun hgen => by
    obtain ⟨π, hπ⟩ := exists_uncovered
    refine not_mem_of_mem_uncovered decay (S := Set.univ) ?_ hπ (Set.mem_univ π)
    filter_upwards [(division_generalizes_iff decay).1 hgen] with ω hω _
    filter_upwards [hω] with t ht
    simpa only [wrongAt, Set.mem_ofPred_eq, not_exists, not_and, not_not] using ht

/-! The witnesses: an uncovered split exists, so its hypotheses are met; on it `1 / 3` is held
out, and at every update some sample has a wrong held-out answer. -/

example : ∃ π, hiddenOne ∈ heldout equations 4656 π := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact ⟨π, mem_heldout_of_not_avoids hπ (by decide)⟩

example (decay : ℝ) (t : ℕ) : ∃ ω, ω ∈ wrongAt decay t := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  rcases mem_wrongAt_or decay t (ω := (π, 0, fun _ => 1)) hπ with h | h
  exacts [⟨_, h⟩, ⟨_, h⟩]

/-- The empty condition on the split meets the hypothesis of `not_mem_of_mem_uncovered`. -/
example (decay : ℝ) : ∀ π ∈ uncovered, π ∉ (∅ : Set (Equiv.Perm (Fin (97 * 96)))) := fun _ hπ =>
  not_mem_of_mem_uncovered decay (Eventually.of_forall fun _ h => absurd h (Set.notMem_empty _)) hπ

end Transformer.Grokking.Protocol
