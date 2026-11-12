module

public import Transformer.Grokking.Protocol.Refutation
public import Transformer.Grokking.Protocol.Unlabelled

/-!
# Every residue must label a training equation

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 and the appendix *Additional experimental
details*: a random half of the equations is the training set; openai/grok at 3d64b1d8;
`experiments/grok_division/PROTOCOL.md`, *The split*; `review/grok_division_convergence.md`,
section 2. The paper states no condition on the split. This file proves one that every true
form of "held-out accuracy tends to 1" needs.

On a split that leaves the residue `r` unlabelled (`Unlabelled`) the 96 equations with the
answer `r` are held out (`mem_heldout_of_unlabelled`), and exchanging the readout rows of the
token of `r` and of a token that division never shows is a symmetry of the run and of its law.
The run reads an equation the same way from both starts, and its logits of the two tokens are
exchanged (`answerLogits_relabel`), so the token of `r` is not predicted at both
(`not_both_predicted_relabel`: at equal logits the tie goes to the earlier token, which is
the spare one). Hence a sample or its relabelling has a wrong held-out answer at every update
(`mem_wrongAt_or_relabel`), and

* at every update the samples with a wrong held-out answer have at least half the
  probability of the splits that leave `r` unlabelled (`uniformPerm_unlabelled_le`), which is
  positive (`uniformPerm_unlabelled_pos`);
* a condition on the split that admits one split that is not `labelled`, one that leaves some
  residue without a training equation with that answer, does not make the target true: it is
  false that almost surely, on the splits of the condition, every held-out equation is
  eventually answered (`not_ae_eventually_of_not_mem_labelled`), for Adam and for AdamW at
  every weight decay. So a condition under which the target holds admits labelled splits only.

A uniformly random split is not labelled with positive probability
(`uniformPerm_compl_labelled_pos`). `LabelWitness` compares `labelled` with the conditions of
`Uncovered` and of `OrbitAnchors` and has the splits. Whether the target holds on the
labelled splits is open.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter MeasureTheory

/-- **On a split that leaves `r` unlabelled every equation with the quotient `r` is held
out.** -/
lemma mem_heldout_of_unlabelled {r : Fin 97} {π : Equiv.Perm (Fin (97 * 96))}
    (hπ : π ∈ unlabelled r) (b : Fin 96) : (r, b) ∈ heldout equations 4656 π := by
  rcases mem_heldout_or_exists_trainEq π (r, b) with h | ⟨i, hi⟩
  · exact h
  · exact absurd (congrArg Prod.fst hi) (hπ i)

/-- **The logits at the relabelled parameters are the logits with the two tokens exchanged**,
for every equation: the readout rows moved, and what is read did not. -/
lemma answerLogits_relabel (r : Fin 97) (θ : Param paperSizes → ℝ) (x : Equation)
    (c : Fin 239) : answerLogits (θ ∘ relabel r) x c = answerLogits θ x (labelSwap r c) :=
  logits_comp_outPerm (S := paperSizes) (τ := labelSwap r) (θ := θ) (act := relu paperSizes)
    (read x) (4 : Fin 6) c

/-- **The token of `r` is not predicted both at some parameters and at the relabelled ones**,
for any equation. -/
theorem not_both_predicted_relabel (r : Fin 97) (θ : Param paperSizes → ℝ) (x : Equation) :
    ¬ (predict (answerLogits θ x) = residueToken r ∧
      predict (answerLogits (θ ∘ relabel r) x) = residueToken r) := fun ⟨h, h'⟩ => by
  have hlt : spareToken < residueToken r := by
    rw [Fin.lt_def]
    simp only [spareToken, residueToken]
    omega
  have h1 := (predict_eq_iff.1 h).2 spareToken hlt
  have h2 := (predict_eq_iff.1 h').1 spareToken
  rw [answerLogits_relabel, answerLogits_relabel,
    show labelSwap r spareToken = residueToken r from Equiv.swap_apply_right _ _,
    show labelSwap r (residueToken r) = spareToken from Equiv.swap_apply_left _ _] at h2
  exact absurd h2 (not_le.2 h1)

/-- **A sample whose split leaves `r` unlabelled, or its relabelling, has a wrong held-out
answer**, at every update. -/
theorem mem_wrongAt_or_relabel (decay : ℝ) (t : ℕ) {r : Fin 97}
    {ω : Sample (97 * 96) 4656 (Param paperSizes)} (hω : ω.1 ∈ unlabelled r) :
    ω ∈ wrongAt decay t ∨ relabelSample r ω ∈ wrongAt decay t := by
  by_contra hnot
  rw [not_or] at hnot
  simp only [wrongAt, Set.mem_ofPred_eq, not_exists, not_and, not_not] at hnot
  have hx := hnot.1 (r, 0) (mem_heldout_of_unlabelled hω 0)
  have hx' := hnot.2 (r, 0) (mem_heldout_of_unlabelled hω 0)
  rw [show relabelSample r ω = (ω.1, ω.2.1 ∘ relabel r, ω.2.2) from rfl,
    trajectory_relabel decay hω] at hx'
  exact not_both_predicted_relabel r _ (r, 0) ⟨hx, hx'⟩

/-- **At every update the samples with a wrong held-out answer have at least half the
probability of the splits that leave `r` unlabelled.** -/
theorem uniformPerm_unlabelled_le (decay : ℝ) (t : ℕ) (r : Fin 97) :
    uniformPerm (97 * 96) (unlabelled r) ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (wrongAt decay t) :=
  uniformPerm_le_two_mul_law_of_or (measurePreserving_relabelSample r) (fun _ hω => hω)
    fun _ hω => mem_wrongAt_or_relabel decay t hω

/-- **The labelled splits**: every residue is the quotient of a training equation. -/
def labelled : Set (Equiv.Perm (Fin (97 * 96))) := {π | ∀ r : Fin 97, ∃ i, (trainEq π i).1 = r}

lemma not_mem_labelled_iff {π : Equiv.Perm (Fin (97 * 96))} :
    π ∉ labelled ↔ ∃ r, π ∈ unlabelled r := by
  simp only [labelled, unlabelled, Set.mem_ofPred_eq, not_forall, not_exists]

/-- **A condition on the split that admits a split that is not labelled does not make the
target true.** If `S` has a split that leaves some residue without a training equation with
that answer, it is false that for almost every sample whose split is in `S` every held-out
equation is eventually answered. -/
theorem not_ae_eventually_of_not_mem_labelled (decay : ℝ)
    {S : Set (Equiv.Perm (Fin (97 * 96)))} {π : Equiv.Perm (Fin (97 * 96))} (hπS : π ∈ S)
    (hπ : π ∉ labelled) :
    ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω.1 ∈ S →
      ∀ᶠ t in atTop, ω ∉ wrongAt decay t := fun hS => by
  obtain ⟨r, hr⟩ := not_mem_labelled_iff.1 hπ
  have h2 := (measurePreserving_relabelSample r).quasiMeasurePreserving.ae hS
  have hnull : ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes),
      ω ∉ ({π} : Set (Equiv.Perm (Fin (97 * 96)))) ×ˢ (Set.univ :
        Set ((Param paperSizes → ℝ) × (ℕ → Equiv.Perm (Fin 4656)))) := by
    filter_upwards [hS, h2] with ω hω1 hω2 hmem
    have hπω : ω.1 = π := hmem.1
    obtain ⟨t, ht1, ht2⟩ := ((hω1 (hπω ▸ hπS)).and (hω2 (hπω ▸ hπS))).exists
    rcases mem_wrongAt_or_relabel decay t (hπω ▸ hr) with h | h
    exacts [ht1 h, ht2 h]
  have hzero := measure_eq_zero_iff_ae_notMem.2 hnull
  rw [law, Measure.prod_prod, measure_univ, mul_one] at hzero
  exact (uniformPerm_pos_of_mem (Set.mem_singleton π)).ne' hzero

/-- **A uniformly random split leaves some residue unlabelled with positive probability.** -/
theorem uniformPerm_compl_labelled_pos : 0 < uniformPerm (97 * 96) labelledᶜ := by
  obtain ⟨π, hπ⟩ := exists_unlabelled 0
  exact uniformPerm_pos_of_mem (Set.mem_compl (not_mem_labelled_iff.2 ⟨0, hπ⟩))

/-! The witnesses: a split that leaves `7` unlabelled exists; on it `7 / 1 = 7` is held out, a
sample has a wrong held-out answer at every update, and the condition that admits every split
does not make the target true. -/

example : ∃ π, ((7 : Fin 97), (0 : Fin 96)) ∈ heldout equations 4656 π := by
  obtain ⟨π, hπ⟩ := exists_unlabelled 7
  exact ⟨π, mem_heldout_of_unlabelled hπ 0⟩

example (decay : ℝ) (t : ℕ) : ∃ ω, ω ∈ wrongAt decay t := by
  obtain ⟨π, hπ⟩ := exists_unlabelled 7
  rcases mem_wrongAt_or_relabel decay t (ω := (π, 0, fun _ => 1)) hπ with h | h
  exacts [⟨_, h⟩, ⟨_, h⟩]

example (decay : ℝ) : ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes),
    ω.1 ∈ (Set.univ : Set (Equiv.Perm (Fin (97 * 96)))) →
      ∀ᶠ t in atTop, ω ∉ wrongAt decay t := by
  obtain ⟨π, hπ⟩ := exists_unlabelled 7
  exact not_ae_eventually_of_not_mem_labelled decay (Set.mem_univ π)
    (not_mem_labelled_iff.2 ⟨7, hπ⟩)

end Transformer.Grokking.Protocol
