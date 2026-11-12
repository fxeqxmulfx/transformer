module

public import Transformer.Grokking.Protocol.OrbitAnchors
public import Transformer.Grokking.Protocol.ReadoutStationary
public import Transformer.Grokking.Protocol.Uncovered

/-!
# Almost surely every residue is missing from infinitely many batches

Source: openai/grok at 3d64b1d8, `grok/data.py`: `ArithmeticIterator` draws a permutation of
the training set for each epoch and cuts it into batches in order;
`experiments/grok_division/PROTOCOL.md`, *The split* and *The optimizer*. arXiv:2201.02177v1
(Power et al.), appendix *Additional experimental details*, gives the batch size; it states
no probability.

An equation shows a residue as its numerator, its denominator or its quotient (`Shows`). At
most `289` equations show a given residue, so whatever the split at least `4367` of its `4656`
training equations do not (`card_trainEq_not_shows_ge`), more than the `512` of a batch. Hence
some shuffle of the training set puts `512` of them in the first batch of its epoch
(`firstBatchHides`, `firstBatchHides_nonempty`).

The epochs' shuffles are independent and uniform. In a sequence of independent draws an event
of positive probability occurs infinitely often, almost surely
(`ae_frequently_mem_infinitePi`): the sequences in which it never occurs from the draw `n` on
are, for every `M`, in a box of probability `(1 - p) ^ M`. The event may depend on the split,
of which there are finitely many (`ae_law_frequently_mem`). So for almost every sample
infinitely many batches have no equation that shows the residue `r`
(`ae_frequently_batch_not_shows`): the first batch of infinitely many epochs is one.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- **In independent draws an event of positive probability occurs infinitely often**, for
almost every sequence. -/
theorem ae_frequently_mem_infinitePi {α : Type*} [MeasurableSpace α] (μ : Measure α)
    [IsProbabilityMeasure μ] {G : Set α} (hG : MeasurableSet G) (hpos : μ G ≠ 0) :
    ∀ᵐ σ ∂Measure.infinitePi fun _ : ℕ => μ, ∃ᶠ e in atTop, σ e ∈ G := by
  have hlt : μ Gᶜ < 1 := by
    rw [prob_compl_eq_one_sub hG]
    exact ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero hpos
  have hnull : ∀ n : ℕ,
      Measure.infinitePi (fun _ : ℕ => μ) {σ | ∀ e, n ≤ e → σ e ∉ G} = 0 := fun n => by
    refine le_antisymm (ge_of_tendsto' (ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one hlt)
      fun M => ?_) zero_le
    calc Measure.infinitePi (fun _ : ℕ => μ) {σ | ∀ e, n ≤ e → σ e ∉ G}
        ≤ Measure.infinitePi (fun _ : ℕ => μ)
            (Set.pi (Finset.Ico n (n + M)) fun _ => Gᶜ) :=
          measure_mono fun σ hσ e he => hσ e (Finset.mem_Ico.1 he).1
      _ = ∏ _e ∈ Finset.Ico n (n + M), μ Gᶜ := Measure.infinitePi_pi _ fun _ _ => hG.compl
      _ = μ Gᶜ ^ M := by simp
  rw [ae_iff]
  refine measure_mono_null (t := ⋃ n : ℕ, {σ | ∀ e, n ≤ e → σ e ∉ G}) (fun σ hσ => ?_)
    (measure_iUnion_null hnull)
  simp only [Set.mem_ofPred_eq, not_frequently, eventually_atTop] at hσ
  exact Set.mem_iUnion.2 hσ

/-- **An event of the epoch's shuffle that the split leaves possible occurs in infinitely many
epochs**, for almost every sample. -/
theorem ae_law_frequently_mem {ι : Type*} {N k : ℕ} {ν : Measure (ι → ℝ)} [SFinite ν]
    (G : Equiv.Perm (Fin N) → Set (Equiv.Perm (Fin k))) (hG : ∀ π, (G π).Nonempty) :
    ∀ᵐ ω ∂law N k ν, ∃ᶠ e in atTop, ω.2.2 e ∈ G ω.1 := by
  have h : ∀ᵐ σ ∂Measure.infinitePi fun _ : ℕ => uniformPerm k,
      ∀ π, ∃ᶠ e in atTop, σ e ∈ G π := ae_all_iff.2 fun π => by
    obtain ⟨τ, hτ⟩ := hG π
    exact ae_frequently_mem_infinitePi (uniformPerm k) MeasurableSpace.measurableSet_top
      (uniformPerm_pos_of_mem hτ).ne'
  filter_upwards [ae_law_shuffles h] with ω hω
  exact hω ω.1

/-- The first batch of an epoch: the first `b` slots of the epoch's shuffle. -/
lemma batchSlots_mul_add_one (k b : ℕ) (σ : ℕ → Equiv.Perm (Fin k)) (e : ℕ)
    (h : 0 < batchesPerEpoch k b) :
    batchSlots k b σ (batchesPerEpoch k b * e + 1) = ((List.finRange k).map (σ e)).take b := by
  simp only [batchSlots, Nat.add_sub_cancel, Nat.mul_div_cancel_left _ h, Nat.mul_mod_right,
    Nat.zero_mul, List.drop_zero]

/-- A slot of the first batch of an epoch is the image of one of the first `b` places. -/
lemma exists_of_mem_batchSlots_first {k b : ℕ} {σ : ℕ → Equiv.Perm (Fin k)} {e : ℕ}
    (h : 0 < batchesPerEpoch k b) {s : Fin k}
    (hs : s ∈ batchSlots k b σ (batchesPerEpoch k b * e + 1)) :
    ∃ j : Fin k, (j : ℕ) < b ∧ σ e j = s := by
  rw [batchSlots_mul_add_one k b σ e h, List.mem_take_iff_getElem] at hs
  obtain ⟨j, hj, rfl⟩ := hs
  have hjk : j < k := by simpa using (lt_min_iff.1 hj).2
  exact ⟨⟨j, hjk⟩, (lt_min_iff.1 hj).1, by simp⟩

/-- **An equation shows a residue**: as its numerator, its denominator or its quotient. -/
def Shows (r : Fin 97) (x : Equation) : Prop :=
  numerator x = r ∨ denominator x = r ∨ x.1 = r

instance (r : Fin 97) : DecidablePred (Shows r) := fun x => by
  unfold Shows
  infer_instance

/-- **Whatever the split, at least `4367` training equations do not show a given residue.** -/
theorem card_trainEq_not_shows_ge (r : Fin 97) (π : Equiv.Perm (Fin (97 * 96))) :
    4367 ≤ (Finset.univ.filter fun i : Fin 4656 => ¬Shows r (trainEq π i)).card := by
  have hall := card_filter_avoid_ge {r}
  have hsub : (Finset.univ.filter fun j : Fin (97 * 96) => numerator (equations j) ∉ ({r} :
      Finset (Fin 97)) ∧ denominator (equations j) ∉ ({r} : Finset (Fin 97)) ∧
      (equations j).1 ∉ ({r} : Finset (Fin 97))) ⊆
      Finset.univ.filter fun j => ¬Shows r (equations j) := fun j hj => by
    have h := (Finset.mem_filter.1 hj).2
    simp only [Finset.mem_singleton] at h
    exact Finset.mem_filter.2 ⟨Finset.mem_univ j, fun hs => hs.elim h.1 fun hs =>
      hs.elim h.2.1 h.2.2⟩
  have hshow := Finset.card_filter_add_card_filter_not
    (s := (Finset.univ : Finset (Fin (97 * 96)))) fun j => Shows r (equations j)
  have htrain := Finset.card_filter_add_card_filter_not
    (s := (Finset.univ : Finset (Fin 4656))) fun i => Shows r (trainEq π i)
  have hinj : (Finset.univ.filter fun i : Fin 4656 => Shows r (trainEq π i)).card ≤
      (Finset.univ.filter fun j : Fin (97 * 96) => Shows r (equations j)).card :=
    Finset.card_le_card_of_injOn (fun i => π (Fin.castLE (by norm_num : 4656 ≤ 97 * 96) i))
      (fun i hi => Finset.mem_coe.2 (Finset.mem_filter.2
        ⟨Finset.mem_univ _, (Finset.mem_filter.1 hi).2⟩))
      fun a _ b _ hab => Fin.castLE_injective _ (π.injective hab)
  have hle := Finset.card_le_card hsub
  rw [Finset.card_univ, Fintype.card_fin] at hshow htrain
  rw [Finset.card_singleton] at hall
  omega

/-- The shuffles of the training set whose first batch has no equation that shows `r`. -/
def firstBatchHides (r : Fin 97) (π : Equiv.Perm (Fin (97 * 96))) :
    Set (Equiv.Perm (Fin 4656)) :=
  {τ | ∀ j : Fin 4656, (j : ℕ) < 512 → ¬Shows r (trainEq π (τ j))}

/-- **Some shuffle of the training set has a first batch that hides the residue.** -/
theorem firstBatchHides_nonempty (r : Fin 97) (π : Equiv.Perm (Fin (97 * 96))) :
    (firstBatchHides r π).Nonempty := by
  obtain ⟨τ, hτ⟩ := exists_perm_forall_castLE (by norm_num : 512 ≤ 4656)
    (fun i : Fin 4656 => ¬Shows r (trainEq π i))
    (le_trans (by norm_num) (card_trainEq_not_shows_ge r π))
  exact ⟨τ, fun j hj => hτ ⟨j, hj⟩⟩

/-- **For almost every sample infinitely many batches have no equation that shows a given
residue**: the first batch of infinitely many epochs is one. -/
theorem ae_frequently_batch_not_shows {ν : Measure (Param paperSizes → ℝ)} [SFinite ν]
    (r : Fin 97) : ∀ᵐ ω ∂law (97 * 96) 4656 ν,
      ∃ᶠ t in atTop, ∀ x ∈ divisionBatch ω (t + 1), ¬Shows r x := by
  have hb : batchesPerEpoch 4656 512 = 10 := by decide
  have hmap : Tendsto (fun e : ℕ => 10 * e) atTop atTop :=
    tendsto_atTop_mono (fun e => Nat.le_mul_of_pos_left e (by norm_num)) tendsto_id
  filter_upwards [ae_law_frequently_mem (firstBatchHides r) (firstBatchHides_nonempty r)]
    with ω hω
  refine hmap.frequently (hω.mono fun e he x hx => ?_)
  obtain ⟨s, hs, rfl⟩ := List.mem_map.1 hx
  rw [← hb] at hs
  obtain ⟨j, hj, rfl⟩ := exists_of_mem_batchSlots_first (by decide) hs
  exact he j hj

/-! The witnesses: two shuffles of two items, of which the identity is drawn infinitely often;
an epoch of four slots in batches of three; `3 / 5 = 20`, which shows `3`, `5` and `20`. -/

example : ∀ᵐ σ ∂Measure.infinitePi fun _ : ℕ => uniformPerm 2, ∃ᶠ e in atTop, σ e ∈ ({1} :
    Set (Equiv.Perm (Fin 2))) :=
  ae_frequently_mem_infinitePi _ MeasurableSpace.measurableSet_top
    (uniformPerm_pos_of_mem (Set.mem_singleton _)).ne'

example : ∀ᵐ ω ∂law 2 2 (Measure.dirac (0 : Fin 1 → ℝ)), ∃ᶠ e in atTop, ω.2.2 e = ω.1 :=
  ae_law_frequently_mem (fun π => {π}) fun π => ⟨π, rfl⟩

example : ∀ᵐ ω ∂law 2 2 (Measure.dirac (0 : Fin 1 → ℝ)), ω.2.2 = ω.2.2 :=
  ae_law_shuffles (P := fun σ => σ = σ) (ae_of_all _ fun _ => rfl)

example : batchSlots 4 3 (fun _ => 1) (batchesPerEpoch 4 3 * 5 + 1) =
    ((List.finRange 4).map (1 : Equiv.Perm (Fin 4))).take 3 :=
  batchSlots_mul_add_one 4 3 _ 5 (by decide)

example : ∃ j : Fin 4, (j : ℕ) < 3 ∧ (1 : Equiv.Perm (Fin 4)) j = 2 :=
  exists_of_mem_batchSlots_first (σ := fun _ => 1) (e := 0) (by decide) (by decide)

example : Shows 3 (20, 4) ∧ Shows 5 (20, 4) ∧ Shows 20 (20, 4) ∧ ¬Shows 7 (20, 4) := by decide

example := ae_frequently_batch_not_shows (ν := initMeasure paperSizes) 0

end Transformer.Grokking.Protocol
