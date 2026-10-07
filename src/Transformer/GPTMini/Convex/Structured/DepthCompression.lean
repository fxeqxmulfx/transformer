import Transformer.Basis.Depth
import Mathlib.Data.List.Destutter

/-!
# Compact depth data semantics versus the independent Basis language

Source: arXiv:2506.16055v3, §2.4 and Appendix F's E_k, the independent
Basis.depthNext alternating-subsequence test at cbafbe9, and Mathlib's
proved maximal unequal-neighbor destuttering. This proof identifies the
actual run-compressed word with the required alternating word; it does
not define task correctness using a supplied encoder or automaton state.

The result is a semantic foundation for six learned finite state rows.
Neutral deletion, genuine raw-token serialization, actual state scans
and finite learned decoder confidence remain separate coupling steps.
All positive k are handled, including the source's necessary k>0
restriction and both Basis values 2 and 4.
As in CRASP.altPlus_eq, the paper's printed even-a exponent k in
eq:altsingle is corrected to k/2; the equality also requires positive k.
The independent k-block language itself is unchanged.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.CRASP
open scoped Classical

/-- Every independent alternating word is an actual unequal-neighbor chain.
Source: the true altList recurrence, including both initial phases and empty words. -/
theorem alternatingList_chain (phase : Bool) (n : ℕ) :
    (altList phase n).IsChain (· ≠ ·) := by
  induction n generalizing phase with
  | zero => exact .nil
  | succ n ih =>
      cases n with
      | zero => exact .singleton phase
      | succ n =>
          apply List.IsChain.cons_cons
          · cases phase <;> decide
          · exact ih (!phase)

/-- Over the genuine two-letter alphabet, every unequal-neighbor chain is its uniquely phased alternating word.
Source: Bool has exactly two values, so each next unequal letter is the previous letter's negation. -/
theorem alternatingChain_cons (phase : Bool) (tail : List Bool)
    (hchain : (phase :: tail).IsChain (· ≠ ·)) : phase :: tail = altList phase (tail.length + 1) := by
  induction tail generalizing phase with
  | nil => rfl
  | cons next tail ih =>
      cases hchain with
      | cons_cons hne htail =>
          have hn : next = !phase := by cases phase <;> cases next <;> simp_all
          rw [List.length_cons, altList_succ, ← hn, ← ih next htail]

example : ([false, true, false] : List Bool).IsChain (· ≠ ·) := by decide

/-- Every genuine two-letter run-compressed chain has a complete alternating-word representation.
Source: the proved consecutive-letter invariant; the empty word may use either phase. -/
theorem alternatingChain_canonical (word : List Bool) (hchain : word.IsChain (· ≠ ·)) :
    ∃ phase, word = altList phase word.length := by
  cases word with
  | nil => exact ⟨false, rfl⟩
  | cons phase tail => exact ⟨phase, alternatingChain_cons phase tail hchain⟩

example : ([true, false] : List Bool).IsChain (· ≠ ·) := by decide

/-- A shorter alternating word remains an actual subsequence of the same longer phase.
Source: CRASP.altList_sublist_succ iterated along real lengths, without a run-count premise. -/
theorem alternatingList_mono (phase : Bool) (k n : ℕ) (hlen : k ≤ n) :
    (altList phase k).Sublist (altList phase n) := by
  induction n with
  | zero =>
      have hk : k = 0 := by omega
      subst k
      exact List.Sublist.refl _
  | succ n ih =>
      by_cases he : k = n + 1
      · rw [he]
      · have hk : k ≤ n := by omega
        exact (ih hk).trans (altList_sublist_succ phase n)

example : (2 : ℕ) ≤ 4 := by omega

/-- One additional opposite-phase block already contains the shorter forbidden starting-b subsequence.
Source: altList's physical head/tail and the same-phase subsequence monotonicity. -/
theorem alternatingList_opposite (k n : ℕ) (hlen : k + 1 ≤ n) :
    (altList true k).Sublist (altList false n) := by
  have htail : (altList true k).Sublist (altList false (k + 1)) := by
    rw [altList_succ]
    exact List.sublist_cons_self false _
  exact htail.trans (alternatingList_mono false (k + 1) n hlen)

example : (2 : ℕ) + 1 ≤ 4 := by omega

/-- The independent Basis test is exactly equality of the actual run-compressed word to its required phase and block count.
Source: arXiv:2506.16055v3 §2.4, eq:altsingle, with the existing even-a exponent correction k to k/2 and positive-k hypothesis, genuine destuttering maximality and the real two-letter alphabet. -/
theorem depthCompression_criterion (k : ℕ) (hk : 0 < k) (word : List Bool) :
    ((altList false k).Sublist word ∧ ¬(altList true k).Sublist word) ↔
      word.destutter (· ≠ ·) = altList false k := by
  constructor
  · rintro ⟨ha, hb⟩
    let compact := word.destutter (· ≠ ·)
    have hchain : compact.IsChain (· ≠ ·) := List.isChain_destutter (· ≠ ·) word
    obtain ⟨phase, hc⟩ := alternatingChain_canonical compact hchain
    have hmax : k ≤ compact.length := by
      have h := List.IsChain.length_le_length_destutter_ne ha (alternatingList_chain false k)
      rw [length_altList] at h
      exact h
    have hsub : compact.Sublist word := List.destutter_sublist (· ≠ ·) word
    cases phase with
    | false =>
        have hlen : compact.length = k := by
          by_contra hne
          have hlong : k + 1 ≤ compact.length := by omega
          have hf : (altList true k).Sublist compact := by
            rw [hc]
            exact alternatingList_opposite k compact.length hlong
          exact hb (hf.trans hsub)
        change compact = altList false k
        rw [hc, hlen]
    | true =>
        have hf : (altList true k).Sublist compact := by
          rw [hc]
          exact alternatingList_mono true k compact.length hmax
        exact False.elim (hb (hf.trans hsub))
  · intro hc
    have ha : (altList false k).Sublist word := by
      rw [← hc]
      exact List.destutter_sublist (· ≠ ·) word
    refine ⟨ha, ?_⟩
    intro hb
    have hlong := altList_succ_of_both false k hk word ha hb
    have hupper : ∀ phase, ¬(altList phase (k + 1)).Sublist word := by
      intro phase hs
      have h := List.IsChain.length_le_length_destutter_ne hs (alternatingList_chain phase (k + 1))
      rw [length_altList, hc, length_altList] at h
      omega
    rcases hlong with hf | ht
    · exact hupper false hf
    · exact hupper true ht

example : (0 : ℕ) < 2 := by omega

/-- Deleting actual neutral letters and then compressing active runs characterizes the original Basis accept label.
Source: Basis.depthNext_body's independent raw subsequence test and the proved semantic compression identity, without a learned encoder premise. -/
theorem depthCompression_accept (k : ℕ) (hk : 0 < k) (word : List (Option Bool)) :
    Transformer.Basis.depthNext k (Transformer.Basis.bos :: Transformer.Basis.depthBody word) = Transformer.Basis.accept ↔
      word.reduceOption.destutter (· ≠ ·) = altList false k := by
  rw [Transformer.Basis.depthNext_body]
  have he := depthCompression_criterion k hk word.reduceOption
  rw [← he]
  split_ifs with h
  · exact iff_of_true rfl h
  · exact iff_of_false (by norm_num [Transformer.Basis.reject, Transformer.Basis.accept]) h

example : (0 : ℕ) < 4 := by omega

end Transformer.GPTMini.Convex.Structured
