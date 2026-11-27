module

public import Transformer.Grokking.FiniteRecovery.ArgandBinaryRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandHomogeneous
public import Transformer.Grokking.FiniteRecovery.ArgandSamples

/-! Recovery on the two omitted constant words in a fixed-length domain.
The extra first-position labels repeat graph observations on mixed words.
Every layer and pass takes its ordinary full-gradient update. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter MeasureTheory
open scoped Topology

/-- Two words attain the information threshold for the active first
query, or for any chosen query with zero passes. Both observed labels
are final-token graph labels because the two training words are constant. -/
theorem twoObservations_momentum_recovers {layers T : ℕ}
    (mu : Measure (ComplexParameter Bool 1 1 layers → ℂ)) [mu.IsOpenPosMeasure]
    (repeats : ℕ) (i : Fin (T + 1)) (hi : i = 0 ∨ repeats = 0) (mode : Option Bool) :
    ∃ lam₀ > 0, ∀ lam, 0 < lam → lam ≤ lam₀ →
      ∃ U : Set (ComplexParameter Bool 1 1 layers → ℂ), IsOpen U ∧ 0 < mu U ∧
        ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ U,
          ∃ star : Weights Bool 1 1 layers,
            Tendsto (fun t => (fullGradientMomentumRun true repeats (binaryPositionBatch i mode)
              lam (fun _ => η) draw t).weights) atTop (𝓝 star) ∧ ∀ word : Word Bool (T + 1),
                (Model.fromWeights star).logits repeats word i (!(mode.getD (word i))) <
                (Model.fromWeights star).logits repeats word i (mode.getD (word i)) := by
  let batch := binaryPositionBatch i mode
  apply binaryBatch_momentum_posMeasure mu repeats batch
    (nativeBinaryMargins_of_token_labels _ _ _ mode ?_) i (fun word => mode.getD (word i))
  · intro word
    refine ⟨⟨T, fun _ => word i, i, mode.getD (word i)⟩, ?_, rfl, ?_⟩
    · cases word i <;> simp [batch, binaryPositionBatch]
    · intro w a
      rcases hi with rfl | rfl
      · exact (Model.fromWeights w).logits_first_eq_of_token_eq repeats (u := word) (v := fun _ => word 0) rfl a
      · exact (Model.fromWeights w).logits_no_passes_eq_of_token_eq (u := word) (v := fun _ => word i) rfl a
  · intro p hp
    rcases List.mem_cons.mp hp with rfl | hp
    · rfl
    · rcases List.mem_singleton.mp hp with rfl
      rfl

example (mu : Measure (ComplexParameter Bool 1 1 2 → ℂ)) [mu.IsOpenPosMeasure] :=
  twoObservations_momentum_recovers mu 2 (0 : Fin 32) (Or.inl rfl) none

/-- All binary words of length at least three except the two homogeneous words. -/
noncomputable def mixedInputs (n : ℕ) : Finset (Word Bool (n + 3)) :=
  Finset.univ \ {fun _ => false, fun _ => true}

theorem card_mixedInputs (n : ℕ) : (mixedInputs n).card = 2 ^ (n + 3) - 2 := by
  classical
  have hne : (fun _ : Fin (n + 3) => false) ≠ fun _ => true := by
    intro h
    exact Bool.noConfusion (congrFun h 0)
  rw [mixedInputs, Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ,
    card_word, Fintype.card_bool]
  simp [hne]

/-- The first and last tokens agree and the middle token at index one differs. -/
def markedWord (n : ℕ) (b : Bool) : Word Bool (n + 3) :=
  fun j => if j = 1 then !b else b

private theorem marked_first (n : ℕ) (b : Bool) : markedWord n b 0 = b := by
  simp [markedWord]

private theorem marked_last (n : ℕ) (b : Bool) : markedWord n b (Fin.last (n + 2)) = b := by
  have hn : Fin.last (n + 2) ≠ (1 : Fin (n + 3)) := by
    intro h
    have hv := congrArg Fin.val h
    simp at hv
  simp [markedWord, hn]

/-- Both extra labels are already present on the observed graph. -/
theorem markedWord_observed (n : ℕ) (b : Bool) (mode : Option Bool) :
    markedWord n b ∈ mixedInputs n ∧
      lastTokenTask mode (markedWord n b) = mode.getD (markedWord n b 0) := by
  have mixed (c : Bool) : markedWord n b ≠ fun _ => c := by
    intro h
    have h0 := congrFun h 0
    have h1 := congrFun h 1
    rw [marked_first] at h0
    simp [markedWord, ← h0] at h1
  exact ⟨by simp [mixedInputs, mixed], by simp [lastTokenTask, marked_first, marked_last]⟩

noncomputable def mixedPositionBatch (n : ℕ) (mode : Option Bool) : List (LabeledPosition Bool) :=
  observedLastPositions (lastTokenTask mode) (mixedInputs n) ++
    [⟨n + 2, markedWord n false, 0, mode.getD false⟩,
     ⟨n + 2, markedWord n true, 0, mode.getD true⟩]

/-- The ordinary last-position logits recover all words of length n+3,
including both entirely unseen input words, on a positive-measure set of
native draws. The two repeated labels supervise position zero on the
marked mixed words. The architecture has one complex coordinate and
arbitrary depth and shared repeat count. -/
theorem mixedWords_momentum_recovers {layers : ℕ}
    (mu : Measure (ComplexParameter Bool 1 1 layers → ℂ)) [mu.IsOpenPosMeasure]
    (n repeats : ℕ) (mode : Option Bool) :
    ∃ lam₀ > 0, ∀ lam, 0 < lam → lam ≤ lam₀ →
      ∃ U : Set (ComplexParameter Bool 1 1 layers → ℂ), IsOpen U ∧ 0 < mu U ∧
        ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ U,
          ∃ star : Weights Bool 1 1 layers,
            Tendsto (fun t => (fullGradientMomentumRun true repeats (mixedPositionBatch n mode) lam
              (fun _ => η) draw t).weights) atTop (𝓝 star) ∧
            (Model.fromWeights star).realizes repeats (lastTokenTask (T := n + 2) mode) := by
  let batch := mixedPositionBatch n mode
  have hlabels (p : LabeledPosition Bool) (hp : p ∈ batch) :
      p.2.2.2 = mode.getD (p.2.1 p.2.2.1) := by
    rcases List.mem_append.mp hp with hp | hp
    · obtain ⟨w, _, rfl⟩ := List.mem_map.mp hp
      rfl
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl <;> simp [marked_first]
  have hcover (word : Word Bool (n + 3)) :
      ∃ p ∈ batch, p.2.2.2 = lastTokenTask mode word ∧ ∀ (w : Weights Bool 1 1 layers) a,
        (Model.fromWeights w).logits repeats word (Fin.last (n + 2)) a =
        (Model.fromWeights w).logits repeats p.2.1 p.2.2.1 a := by
    by_cases hm : word ∈ mixedInputs n
    · refine ⟨⟨n + 2, word, Fin.last (n + 2), lastTokenTask mode word⟩,
        List.mem_append_left _ (List.mem_map.mpr ⟨word, ?_, rfl⟩), rfl, fun _ _ => rfl⟩
      simpa using hm
    have hex : ∃ b : Bool, word = fun _ => b := by
      have h : word = (fun _ => false) ∨ word = (fun _ => true) := by
        by_cases hz : word = (fun _ => false)
        · exact Or.inl hz
        · exact Or.inr (by simpa [mixedInputs, hz] using hm)
      rcases h with h | h
      · exact ⟨false, h⟩
      · exact ⟨true, h⟩
    obtain ⟨b, rfl⟩ := hex
    refine ⟨⟨n + 2, markedWord n b, 0, mode.getD b⟩, List.mem_append_right _ ?_, rfl, ?_⟩
    · cases b <;> simp
    · intro w a
      exact ((Model.fromWeights w).logits_const_eq_first repeats b _ a).trans
        ((Model.fromWeights w).logits_first_eq_of_token_eq repeats
          (u := fun _ => b) (v := markedWord n b) (marked_first n b).symm a)
  obtain ⟨lam₀, hlam₀, hfit⟩ := binaryBatch_momentum_posMeasure mu repeats batch
    (nativeBinaryMargins_of_token_labels layers repeats batch mode hlabels)
    (Fin.last (n + 2)) (lastTokenTask mode) hcover
  refine ⟨lam₀, hlam₀, fun lam hlam hle => ?_⟩
  obtain ⟨U, hU, hp, η₀, hη₀, hr⟩ := hfit lam hlam hle
  refine ⟨U, hU, hp, η₀, hη₀, fun η hη hηle draw hd => ?_⟩
  obtain ⟨star, hs, hstrict⟩ := hr η hη hηle draw hd
  refine ⟨star, hs, fun word a ha => ?_⟩
  have heq : a = !(lastTokenTask mode word) := by
    cases h : lastTokenTask mode word <;> cases a <;> simp_all
  exact heq ▸ hstrict word

example (mu : Measure (ComplexParameter Bool 1 1 2 → ℂ)) [mu.IsOpenPosMeasure] (mode : Option Bool) :=
  mixedWords_momentum_recovers mu 5 2 mode
example (mu : Measure (ComplexParameter Bool 1 1 2 → ℂ)) [mu.IsOpenPosMeasure] (mode : Option Bool) :=
  mixedWords_momentum_recovers mu 13 2 mode
example (mu : Measure (ComplexParameter Bool 1 1 2 → ℂ)) [mu.IsOpenPosMeasure] (mode : Option Bool) :=
  mixedWords_momentum_recovers mu 29 2 mode

end Transformer.Grokking.FiniteRecovery.Argand
