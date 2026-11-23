module

public import Transformer.Phasor.TemporalRecurrenceScore

/-!
# Answer corruption in the GPT series anchors

Source: DiffGPT `5a37c515`, `series_data.corrupt_answers` and `anchor`.
Only selected training answers are replaced. A uniform replacement may
equal the clean residue. The source fits the held-out support filter to
clean training tokens before evaluating either noise level.
-/

@[expose] public section
namespace Transformer.Phasor.NoisyAnchors

def answer (selected : Finset ℕ) (clean replacement : ℕ → Fin 97) (i : ℕ) : Fin 97 :=
  if i ∈ selected then replacement i else clean i

/-- `corrupt_answers`. Sampling indices strictly in the training segment
preserves the answer at every held-out index for arbitrary replacements. -/
theorem held_out (selected : Finset ℕ) (clean replacement : ℕ → Fin 97) (train i : ℕ)
    (inside : ∀ j ∈ selected, j < train) (later : train ≤ i) :
    answer selected clean replacement i = clean i := by
  have absent : i ∉ selected := fun h => by have := inside i h; omega
  simp [answer, absent]

/-- `anchor` encodes the full noisy Fibonacci window to obtain its target.
Dropping that token yields exactly the clean embedding input for either
the first- or second-difference codec. -/
theorem unchanged_prefix (o : DifferencedDivisionScore.Order) (p : List ℤ) (a b y z : ℤ) :
    (TemporalRecurrenceScore.tokens o (p ++ [a,b,y])).dropLast =
      (TemporalRecurrenceScore.tokens o (p ++ [a,b,z])).dropLast := by
  rw [TemporalRecurrenceScore.append_future, TemporalRecurrenceScore.append_future]
  simp

/-- `corrupt_answers` draws a replacement among all 97 residues. Summing
the changed-row counts over the 97 possible replacement values gives 96
times the number selected, regardless of the original answers. By linearity
the mean is also the expectation when each selected row is drawn independently. -/
theorem uniform_changes (selected : Finset ℕ) (clean : ℕ → Fin 97) :
    (∑ r : Fin 97, ∑ i ∈ selected, if r ≠ clean i then 1 else 0 : ℕ) = selected.card * 96 := by
  rw [Finset.sum_comm]
  simp only [Finset.sum_boole, Finset.filter_ne', Finset.card_erase_of_mem (Finset.mem_univ _)]
  simp

example : answer {2,7} (fun i => ⟨i % 97, by omega⟩) (fun _ => 0) 11 = 11 := by
  apply held_out (train := 8)
  · intro i hi
    simp only [Finset.mem_insert, Finset.mem_singleton] at hi
    omega
  · decide

end Transformer.Phasor.NoisyAnchors
