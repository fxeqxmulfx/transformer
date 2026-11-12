module

public import Transformer.Grokking.Protocol.TwinSplit

/-!
# A twin split that is labelled exists

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "we
chose a fraction of all available equations at random and declared them to be the training
set"; `experiments/grok_division/PROTOCOL.md`, *The split*. The twin splits are those of
`Transformer.Grokking.Protocol.TwinSplit`, the count is `card_filter_avoid_ge` of `Uncovered`.

A shuffle can put any `k` marked items in the first `k` slots and, among them, every item of
a given set of at most `k` marked items (`exists_perm_castLE_of_subset`). All but `2 · 289`
equations show no twin residue, so at least `4656` equations may be trained on by a twin
split (`card_twinAllowed_ge`). Among them are the 97 equations `r / 1 = r`, one for each
residue `r`: for `r = 2` and `r = 95` they are the two equations that a twin split may show,
and the others show no twin residue (`twinAllowed_unit`). A shuffle that trains on these 97
and on allowed equations only is a twin split and labels every residue
(`exists_twin_labelled`), so the labelled twin splits have positive probability
(`uniformPerm_twin_labelled_pos`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- A shuffle can put `k` marked items in the first `k` slots, among them every item of a
given set of at most `k` marked items. -/
lemma exists_perm_castLE_of_subset {N k : ℕ} (hk : k ≤ N) (g : Fin N → Prop) [DecidablePred g]
    (F : Finset (Fin N)) (hF : ∀ j ∈ F, g j) (hFk : F.card ≤ k)
    (hg : k ≤ (Finset.univ.filter g).card) :
    ∃ π : Equiv.Perm (Fin N), (∀ i : Fin k, g (π (Fin.castLE hk i))) ∧
      ∀ j ∈ F, ∃ i : Fin k, π (Fin.castLE hk i) = j := by
  have hsub : F ⊆ Finset.univ.filter g := fun j hj =>
    Finset.mem_filter.2 ⟨Finset.mem_univ j, hF j hj⟩
  obtain ⟨Q, hFQ, hQ, hcard⟩ := Finset.exists_subsuperset_card_eq hsub hFk hg
  let e : {j : Fin N // (j : ℕ) < k} ≃ {j : Fin N // j ∈ Q} :=
    (Fin.castLEquiv hk).symm.trans (Q.equivFinOfCardEq hcard).symm
  refine ⟨e.extendSubtype, fun i => ?_, fun j hj => ?_⟩
  · have hi : ((Fin.castLE hk i : Fin N) : ℕ) < k := i.isLt
    rw [Equiv.extendSubtype_apply_of_mem e _ hi]
    exact (Finset.mem_filter.1 (hQ (e ⟨_, hi⟩).2)).2
  · obtain ⟨s, hs⟩ := e.surjective ⟨j, hFQ hj⟩
    refine ⟨⟨((s : Fin N) : ℕ), s.2⟩, ?_⟩
    have hcast : Fin.castLE hk ⟨((s : Fin N) : ℕ), s.2⟩ = (s : Fin N) := Fin.ext rfl
    rw [hcast, Equiv.extendSubtype_apply_of_mem e _ s.2, Subtype.coe_eta, hs]

/-- **At least `4656` equations may be trained on by a twin split.** -/
theorem card_twinAllowed_ge :
    4656 ≤ (Finset.univ.filter fun j : Fin (97 * 96) => TwinAllowed (equations j)).card := by
  have h := card_filter_avoid_ge twins
  have hsub : (Finset.univ.filter fun j : Fin (97 * 96) => numerator (equations j) ∉ twins ∧
      denominator (equations j) ∉ twins ∧ (equations j).1 ∉ twins) ⊆
      Finset.univ.filter fun j => TwinAllowed (equations j) := fun j hj =>
    Finset.mem_filter.2 ⟨Finset.mem_univ j, Or.inl (Finset.mem_filter.1 hj).2⟩
  have hle := Finset.card_le_card hsub
  have hc : twins.card = 2 := rfl
  omega

/-- **A twin split may train on every equation `r / 1 = r`.** -/
theorem twinAllowed_unit : ∀ r : Fin 97, TwinAllowed (r, 0) := by decide

/-- **Some twin split is labelled**: it trains on `r / 1 = r` for every residue `r`. -/
theorem exists_twin_labelled : ∃ π, π ∈ twin ∩ labelled := by
  obtain ⟨π, hπ, hF⟩ := exists_perm_castLE_of_subset (by norm_num : 4656 ≤ 97 * 96)
    (fun j => TwinAllowed (equations j))
    (Finset.univ.image fun r : Fin 97 => finProdFinEquiv (r, (0 : Fin 96)))
    (fun j hj => by
      obtain ⟨r, -, rfl⟩ := Finset.mem_image.1 hj
      have he : equations (finProdFinEquiv (r, (0 : Fin 96))) = (r, 0) :=
        Equiv.symm_apply_apply _ _
      rw [he]
      exact twinAllowed_unit r)
    (Finset.card_image_le.trans (by simp))
    card_twinAllowed_ge
  refine ⟨π, fun i => hπ i, fun r => ?_⟩
  obtain ⟨i, hi⟩ := hF _ (Finset.mem_image_of_mem _ (Finset.mem_univ r))
  refine ⟨i, ?_⟩
  simp only [trainEq, trainAt, hi, equations, Equiv.symm_apply_apply]

/-- **A uniformly random split is a labelled twin split with positive probability.** -/
theorem uniformPerm_twin_labelled_pos : 0 < uniformPerm (97 * 96) (twin ∩ labelled) := by
  obtain ⟨π, hπ⟩ := exists_twin_labelled
  exact uniformPerm_pos_of_mem hπ

/-! The witnesses: three slots of four, of which the items `0`, `2` and `3` are marked and `2`
is asked for; and the hypotheses of `TwinSplit` on a split, met by a labelled twin split. -/

example : ∃ π : Equiv.Perm (Fin 4), (∀ i : Fin 3, π (Fin.castLE (by norm_num) i) ≠ 1) ∧
    ∀ j ∈ ({2} : Finset (Fin 4)), ∃ i : Fin 3, π (Fin.castLE (by norm_num) i) = j :=
  exists_perm_castLE_of_subset (by norm_num) (fun j => j ≠ 1) {2} (by decide) (by decide)
    (by decide)

example : ∃ π, Carries twinSwap twinOrder π ∧ twinOrder * π ∈ twin ∧ twinOrder * π ∈ labelled := by
  obtain ⟨π, hπ, hl⟩ := exists_twin_labelled
  exact ⟨π, carries_twin hπ, twinOrder_mul_mem_twin hπ, twinOrder_mul_mem_labelled hπ hl⟩

end Transformer.Grokking.Protocol
