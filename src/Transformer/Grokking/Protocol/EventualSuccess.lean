module

public import Transformer.Grokking.Protocol.LabelNotSufficient

/-!
# The probability that a run eventually answers every held-out equation

Source: arXiv:2201.02177v1 (Power et al.), Figure 1; `experiments/grok_division/PROTOCOL.md`,
*The split* and *Randomness*; `review/grok_division_convergence.md`, section 2. The target is
`Generalizes` of `Transformer.Grokking.Protocol.Target`; the paper states no probability.

The runs that from some update on answer every held-out equation form an event
(`eventuallyRight`, `measurableSet_eventuallyRight`), and the target for division says that
this event has probability `1` (`division_generalizes_iff_law_eq_one`). The refutations of the
target bound that probability. If at every update a sample or its image has a wrong held-out
answer, one of the two has a wrong held-out answer at infinitely many updates
(`not_eventuallyRight_or`). So, under Adam and under AdamW at every weight decay:

* given that the split is in a set of splits that are not labelled, with a conditional
  probability of at least one half the run does not eventually answer every held-out equation
  (`uniformPerm_le_not_eventuallyRight`); the same given a set of splits with a symmetry that
  moves a held-out answer (`uniformPerm_le_not_eventuallyRight_of_twisted`);
* the probability that a run eventually answers every held-out equation is at most `1` less
  half the probability that the split is not labelled (`law_eventuallyRight_le`), so less
  half the probability that it leaves a given residue unlabelled
  (`law_eventuallyRight_le_unlabelled`), and at most `1` less half the probability of a set
  of splits with such a symmetry (`law_eventuallyRight_le_of_twisted`);
* the target with a tolerance, "with probability at least `1 - δ` every held-out equation is
  eventually answered", is false for every `δ` below half the probability that the split is
  not labelled (`not_le_law_eventuallyRight`).

Nothing here bounds the probability of the event from below.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- **The event of the target**: the runs that from some update on answer every held-out
equation. -/
def eventuallyRight (decay : ℝ) : Set (Sample (97 * 96) 4656 (Param paperSizes)) :=
  {ω | ∀ᶠ t in atTop, ω ∉ wrongAt decay t}

/-- **The runs that eventually answer every held-out equation form an event.** -/
theorem measurableSet_eventuallyRight (decay : ℝ) : MeasurableSet (eventuallyRight decay) := by
  have hset : eventuallyRight decay =
      ⋃ n : ℕ, ⋂ t : ℕ, ⋂ (_ : n ≤ t), (wrongAt decay t)ᶜ := by
    ext ω
    simp [eventuallyRight, eventually_atTop]
  rw [hset]
  exact MeasurableSet.iUnion fun n => MeasurableSet.iInter fun t =>
    MeasurableSet.iInter fun _ => (measurableSet_wrongAt decay t).compl

/-- **The target for division says that this event has probability `1`.** -/
theorem division_generalizes_iff_law_eq_one (decay : ℝ) :
    Generalizes (initMeasure paperSizes) answerLogits (fun x => residueToken x.1)
        (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 ↔
      law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) = 1 := by
  rw [division_generalizes_iff, ← prob_compl_eq_zero_iff (measurableSet_eventuallyRight decay),
    ← mem_ae_iff]
  refine eventually_congr (Eventually.of_forall fun ω => ?_)
  simp only [wrongAt, Set.mem_ofPred_eq, not_exists, not_and, not_not]

variable {decay : ℝ} {A : Set (Equiv.Perm (Fin (97 * 96)))}

/-- If at every update one of two samples has a wrong held-out answer, one of them has a wrong
held-out answer at infinitely many updates. -/
lemma not_eventuallyRight_or {ω ω' : Sample (97 * 96) 4656 (Param paperSizes)}
    (h : ∀ t, ω ∈ wrongAt decay t ∨ ω' ∈ wrongAt decay t) :
    ω ∈ (eventuallyRight decay)ᶜ ∨ ω' ∈ (eventuallyRight decay)ᶜ := by
  by_contra hnot
  rw [not_or, Set.notMem_compl_iff, Set.notMem_compl_iff] at hnot
  obtain ⟨t, h1, h2⟩ := (hnot.1.and hnot.2).exists
  exact (h t).elim h1 h2

/-- A set of splits with at most twice the probability of its runs that are not eventually
right bounds the probability of the event of the target. -/
lemma law_eventuallyRight_le_of_half
    (hA : uniformPerm (97 * 96) A ≤ 2 * law (97 * 96) 4656 (initMeasure paperSizes)
      (splitIn A ∩ (eventuallyRight decay)ᶜ)) :
    law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) ≤
      1 - uniformPerm (97 * 96) A / 2 := by
  set μ := law (97 * 96) 4656 (initMeasure paperSizes) with hμ
  have hhalf : uniformPerm (97 * 96) A / 2 ≤ μ (eventuallyRight decay)ᶜ :=
    ENNReal.div_le_of_le_mul ((hA.trans (mul_le_mul_right
      (measure_mono Set.inter_subset_right) 2)).trans_eq (mul_comm _ _))
  refine ENNReal.le_sub_of_add_le_right
    (ne_top_of_le_ne_top (measure_ne_top μ _) hhalf) ?_
  calc μ (eventuallyRight decay) + uniformPerm (97 * 96) A / 2
      ≤ μ (eventuallyRight decay) + μ (eventuallyRight decay)ᶜ := add_le_add le_rfl hhalf
    _ = 1 := by rw [measure_add_measure_compl (measurableSet_eventuallyRight decay), measure_univ]

variable (decay)

/-- **Given that the split is in a set of splits that are not labelled, with a conditional
probability of at least one half the run does not eventually answer every held-out
equation.** -/
theorem uniformPerm_le_not_eventuallyRight (hA : A ⊆ labelledᶜ) :
    uniformPerm (97 * 96) A ≤ 2 * law (97 * 96) 4656 (initMeasure paperSizes)
      (splitIn A ∩ (eventuallyRight decay)ᶜ) :=
  uniformPerm_le_two_mul_of_singleton fun π hπ => by
    obtain ⟨r, hr⟩ := not_mem_labelled_iff.1 (hA hπ)
    exact uniformPerm_le_two_mul_of_or (measurePreserving_relabelSample r) (fun _ hω => hω)
      fun ω hω => not_eventuallyRight_or fun t =>
        mem_wrongAt_or_relabel decay t (by rwa [Set.mem_singleton_iff.1 hω])

/-- **The same given a set of splits with a symmetry that moves a held-out answer.** -/
theorem uniformPerm_le_not_eventuallyRight_of_twisted {τ : Equiv.Perm (Fin 239)}
    {ρ : Equiv.Perm (Fin (97 * 96))} {x y : Equation} (h : Twisted τ ρ A x y) :
    uniformPerm (97 * 96) A ≤ 2 * law (97 * 96) 4656 (initMeasure paperSizes)
      (splitIn A ∩ (eventuallyRight decay)ᶜ) :=
  uniformPerm_le_two_mul_of_or (measurePreserving_carrySample τ ρ) (fun _ hω => h.closed _ hω)
    fun _ hω => not_eventuallyRight_or fun t => mem_wrongAt_or_carry decay t h hω

/-- **The probability that a run eventually answers every held-out equation is at most `1`
less half the probability that the split is not labelled.** -/
theorem law_eventuallyRight_le :
    law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) ≤
      1 - uniformPerm (97 * 96) labelledᶜ / 2 :=
  law_eventuallyRight_le_of_half (uniformPerm_le_not_eventuallyRight decay subset_rfl)

/-- **It is at most `1` less half the probability that the split leaves a given residue
unlabelled.** -/
theorem law_eventuallyRight_le_unlabelled (r : Fin 97) :
    law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) ≤
      1 - uniformPerm (97 * 96) (unlabelled r) / 2 :=
  law_eventuallyRight_le_of_half (uniformPerm_le_not_eventuallyRight decay
    fun _ hπ => not_mem_labelled_iff.2 ⟨r, hπ⟩)

/-- **It is at most `1` less half the probability of a set of splits with a symmetry that
moves a held-out answer.** -/
theorem law_eventuallyRight_le_of_twisted {τ : Equiv.Perm (Fin 239)}
    {ρ : Equiv.Perm (Fin (97 * 96))} {x y : Equation} (h : Twisted τ ρ A x y) :
    law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) ≤
      1 - uniformPerm (97 * 96) A / 2 :=
  law_eventuallyRight_le_of_half (uniformPerm_le_not_eventuallyRight_of_twisted decay h)

/-- **The target with a tolerance below half the probability that the split is not labelled
is false**: the runs that eventually answer every held-out equation do not have probability
`1 - δ` or more. -/
theorem not_le_law_eventuallyRight {δ : ENNReal}
    (hδ : δ < uniformPerm (97 * 96) labelledᶜ / 2) :
    ¬ 1 - δ ≤ law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) := fun h => by
  have hle := h.trans (law_eventuallyRight_le decay)
  have hone : uniformPerm (97 * 96) labelledᶜ / 2 ≤ 1 :=
    ENNReal.div_le_of_le_mul (prob_le_one.trans (by norm_num))
  exact hδ.not_ge ((ENNReal.sub_le_sub_iff_left hone ENNReal.one_ne_top).1 hle)

/-! The witnesses: the labelled twin splits, a set with a symmetry; the tolerance `0`, which
is the target itself; the split `1`, which is not labelled; and two samples of which one has a
wrong held-out answer at every update, on an uncovered split. -/

example : uniformPerm (97 * 96) (twin ∩ labelled) ≤ 2 * law (97 * 96) 4656
    (initMeasure paperSizes) (splitIn (twin ∩ labelled) ∩ (eventuallyRight decay)ᶜ) :=
  uniformPerm_le_not_eventuallyRight_of_twisted decay twisted_twin

example : law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) ≤
    1 - uniformPerm (97 * 96) (twin ∩ labelled) / 2 :=
  law_eventuallyRight_le_of_twisted decay twisted_twin

example : ¬ 1 - 0 ≤ law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) :=
  not_le_law_eventuallyRight decay (ENNReal.div_pos
    (uniformPerm_pos_of_mem (Set.mem_compl one_notMem_labelled)).ne' ENNReal.ofNat_ne_top)

example : law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) ≤
    1 - uniformPerm (97 * 96) {1} / 2 :=
  law_eventuallyRight_le_of_half
    (uniformPerm_le_not_eventuallyRight decay (Set.singleton_subset_iff.2 one_notMem_labelled))

example : ∃ ω, ω ∈ (eventuallyRight decay)ᶜ ∨ exchangeSample ω ∈ (eventuallyRight decay)ᶜ := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact ⟨(π, 0, fun _ => 1), not_eventuallyRight_or fun t => mem_wrongAt_or decay t hπ⟩

end Transformer.Grokking.Protocol
