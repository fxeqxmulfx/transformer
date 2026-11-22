module

public import Transformer.Grokking.FlatSpectrum.BlockCut.Basic

/-!
# Prefix labels under positive repetitions of letters

Source: the long repeated-letter transfer test in `experiments/phasor_value_transport/README.md`.
Yang, Cadilhac and Chiang, §2.4, Equation `eq:altplus`, define the
alternating block languages. Their label is invariant under independent
positive repetitions of every input letter, including an incomplete final
run. This gives an independent way to check every target on a long prompt.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer.CRASP Transformer.Grokking.FlatSpectrum.BlockCut

/-- Each pair gives a letter and its number of additional copies. -/
def repeatLetters (runs : List (Bool × ℕ)) : List Bool :=
  runs.flatMap fun (letter, extra) => List.replicate (extra + 1) letter

/-- Repetitions preserve the first letter, including the empty input. -/
theorem repeatLetters_head (runs : List (Bool × ℕ)) :
    (repeatLetters runs).head? = (runs.map Prod.fst).head? := by
  cases runs with
  | nil => rfl
  | cons run rest => simp [repeatLetters, List.replicate_succ]

/-- Positive repetitions of arbitrary and unequal sizes preserve the exact
number of blocks, even when successive source letters are equal. -/
theorem repeatLetters_blocks (runs : List (Bool × ℕ)) :
    blocks (repeatLetters runs) = blocks (runs.map Prod.fst) := by
  induction runs with
  | nil => rfl
  | cons run rest ih =>
    change blocks (List.replicate (run.2 + 1) run.1 ++ repeatLetters rest) =
      blocks (run.1 :: rest.map Prod.fst)
    rw [blocks_replicate_append, blocks_cons, repeatLetters_head, ih]

/-- A repeated word has the original membership label for each positive
block target, for either starting letter. -/
theorem repeatLetters_mem (runs : List (Bool × ℕ)) (start : Bool) (k : ℕ) :
    repeatLetters runs ∈ altPlus start (k+1) ↔ runs.map Prod.fst ∈ altPlus start (k+1) := by
  rw [mem_altPlus_succ_iff, mem_altPlus_succ_iff, repeatLetters_head, repeatLetters_blocks]

/-- At every copy of the current letter, including the first, the correct
prefix label equals the label after a single copy in the original word. -/
theorem repeatLetters_prefix_label (prior : List (Bool × ℕ)) (letter start : Bool) (extra k : ℕ) :
    repeatLetters prior ++ List.replicate (extra+1) letter ∈ altPlus start (k+1) ↔
      prior.map Prod.fst ++ [letter] ∈ altPlus start (k+1) := by
  simpa [repeatLetters] using repeatLetters_mem (prior ++ [(letter, extra)]) start k

/-- Uniform repetition gives the stated payload length independent of the
binary word, including a payload whose first and last letters coincide. -/
theorem repeatLetters_length (w : List Bool) (extra : ℕ) :
    (repeatLetters (w.map (fun b => (b, extra)))).length = w.length * (extra+1) := by
  induction w with
  | nil => simp [repeatLetters]
  | cons b rest ih =>
    change (List.replicate (extra+1) b ++ repeatLetters (rest.map (fun t => (t, extra)))).length =
      (rest.length+1) * (extra+1)
    rw [List.length_append, List.length_replicate, ih, Nat.succ_mul, Nat.add_comm]

example : repeatLetters [(false, 0), (false, 2), (true, 1), (false, 2)] ∈ altPlus false 3 :=
  (repeatLetters_mem _ _ 2).2 ((mem_altPlus_succ_iff).2 ⟨rfl, by decide⟩)

end Transformer.Phasor
