import Transformer.GPTMini.Semantics.DepthMatrices
import Transformer.GPTMini.Semantics.Order

/-!
# Ordered depth recurrence at actual positions including neutral tokens

Source: Basis's raw E_2/E_4 semantics at cbafbe9 and the independent
alternating-subsequence criterion of arXiv:2506.16055v3, §2.4/Appendix F.
Uses CRASP.altPlus_eq's checked correction: the source's printed even-a
exponent k in eq:altsingle is k/2, and its characterization requires
k>0. Here the original E_2/E_4 block languages are unchanged.
Subsequence occurrence is transported to the original Option Bool word
without compressing its indices or silently dropping neutral positions.

An alternating pattern ending in a letter extends exactly an earlier
opposite-ending occurrence. Its current token must have that actual
letter, so the opposite branch's current feature is absent. This is
the semantic obligation that will derive zero self-value for XSA.
The two final full-prefix pattern presences give the actual E_2/E_4
accept/reject criterion, including rejection of a wrong active start.

These are data-side predicates and proven recurrence laws. They are
not substituted for the model's computed hidden states; the original
raw embedding, head/FFN matrices and layer induction must realize them.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis
open scoped Classical

/-- A neutral-preserving original word contains the lifted pattern exactly when its neutral-free word contains that pattern.
Source: exact Option.some serialization and actual subsequence order, proved on every original word position. -/
theorem depthSome_sublist_iff {α : Type*} (pattern : List α) (word : List (Option α)) :
    (pattern.map some).Sublist word ↔ pattern.Sublist word.reduceOption := by
  induction word generalizing pattern with
  | nil =>
      cases pattern with
      | nil => exact ⟨fun _ => .slnil, fun _ => .slnil⟩
      | cons letter pattern =>
          simp only [List.map_cons, List.reduceOption_nil]
          constructor <;> intro h <;> cases h
  | cons current word ih =>
      cases pattern with
      | nil => simp only [List.map_nil, List.nil_sublist]
      | cons letter pattern =>
          have houter := ih (letter :: pattern)
          have hinner := ih pattern
          simp only [List.map_cons] at houter
          cases current with
          | none =>
              have hn : (some letter : Option α) ≠ none := by intro h; cases h
              simp only [List.map_cons, List.reduceOption_cons_of_none, List.cons_sublist_cons',
                hn, false_and, or_false, houter]
          | some current =>
              simp only [List.map_cons, List.reduceOption_cons_of_some, List.cons_sublist_cons',
                Option.some.injEq, houter, hinner]

/-- The last letter of a neutral-free pattern has a genuine original word index and a strictly earlier original prefix.
Source: indexed subsequence inversion, transported without changing the locations of neutral tokens. -/
theorem depthSublist_snoc_iff {α : Type*} (pattern : List α) (word : List (Option α)) (letter : α) :
    (pattern ++ [letter]).Sublist word.reduceOption ↔
      ∃ j : Fin word.length, pattern.Sublist (word.take j.val).reduceOption ∧ word.get j = some letter := by
  rw [← depthSome_sublist_iff, List.map_append]
  change ((pattern.map some) ++ [some letter]).Sublist word ↔ _
  rw [sublist_snoc_iff]
  simp only [depthSome_sublist_iff]

/-- A recursively constructed genuine alternating pattern with the specified final active letter.
Source: new ending-letter presentation of the same E_k alternating patterns, with one ordinary append per level. -/
def depthEndPattern : Bool → ℕ → List Bool
  | _, 0 => []
  | letter, n + 1 => depthEndPattern (!letter) n ++ [letter]

/-- A data-side ordered occurrence ending at an actual original position, retaining the neutral prefix indices.
Source: the preceding indexed pattern criterion; this is a predicate of the word/position, not a model encoder definition. -/
def DepthOccurrence (n : ℕ) (letter : Bool) (word : List (Option Bool)) (i : Fin word.length) : Prop :=
  (depthEndPattern (!letter) n).Sublist (word.take i.val).reduceOption ∧ word.get i = some letter

/-- The initial recurrence feature is precisely the actual current raw letter, with no earlier-pattern test required.
Source: the genuine one-letter starting detector and the empty-prefix subsequence base case. -/
theorem depthOccurrence_zero (letter : Bool) (word : List (Option Bool)) (i : Fin word.length) :
    DepthOccurrence 0 letter word i ↔ word.get i = some letter := by
  constructor
  · intro h
    exact h.2
  · intro h
    exact ⟨List.nil_sublist _, h⟩

/-- Existence of an actual ending occurrence is precisely the full-prefix alternating subsequence property.
Source: exact original-position snoc semantics and the recursive alternating pattern, with all neutrals retained. -/
theorem depthOccurrence_exists (n : ℕ) (letter : Bool) (word : List (Option Bool)) :
    (∃ i : Fin word.length, DepthOccurrence n letter word i) ↔
      (depthEndPattern letter (n + 1)).Sublist word.reduceOption := by
  rw [depthEndPattern, depthSublist_snoc_iff]
  rfl

/-- Each next alternating occurrence is exactly a current matching letter preceded by an earlier opposite-ending occurrence.
Source: the ordered prefix recurrence, with strictly earlier genuine positions rather than bag membership or compressed indices. -/
theorem depthOccurrence_succ (n : ℕ) (letter : Bool) (word : List (Option Bool)) (i : Fin word.length) :
    DepthOccurrence (n + 1) letter word i ↔ word.get i = some letter ∧
      ∃ j : Fin word.length, j.val < i.val ∧ DepthOccurrence n (!letter) word j := by
  have hlen : (word.take i.val).length = i.val := List.length_take_of_le i.isLt.le
  change (depthEndPattern (!letter) (n + 1)).Sublist (word.take i.val).reduceOption ∧ _ ↔ _
  rw [depthEndPattern, depthSublist_snoc_iff, Bool.not_not]
  constructor
  · rintro ⟨⟨j, hp, hj⟩, hi⟩
    have hji : j.val < i.val := by simpa only [hlen] using j.isLt
    let selected : Fin word.length := ⟨j.val, hji.trans i.isLt⟩
    refine ⟨hi, selected, hji, ?_, ?_⟩
    · simpa only [DepthOccurrence, Bool.not_not, List.take_take, min_eq_left hji.le] using hp
    · simpa only [List.get_eq_getElem, List.getElem_take, selected] using hj
  · rintro ⟨hi, j, hji, hp, hj⟩
    let selected : Fin (word.take i.val).length := ⟨j.val, by rw [hlen]; exact hji⟩
    refine ⟨⟨selected, ?_, ?_⟩, hi⟩
    · simpa only [Bool.not_not, List.take_take, selected, min_eq_left hji.le] using hp
    · simpa only [List.get_eq_getElem, List.getElem_take, selected] using hj

/-- A real current letter cannot simultaneously carry the opposite-ending branch's occurrence feature.
Source: disjoint raw a/b types; this semantic fact derives the original XSA zero-self condition in ordered detection. -/
theorem depthOccurrence_opposite (n : ℕ) (letter : Bool) (word : List (Option Bool)) (i : Fin word.length)
    (hcurrent : word.get i = some letter) : ¬DepthOccurrence n (!letter) word i := by
  intro h
  have he : some letter = some (!letter) := hcurrent.symm.trans h.2
  cases letter <;> cases he

example : ([some false, none, some true] : List (Option Bool)).get (2 : Fin 3) = some true := by rfl

/-- Both actual easy ending-patterns are exactly the original ab/ba alternating targets.
Source: Basis E_2's independent alternating-subsequence criterion, with neither active start omitted. -/
theorem depthEndPattern_easy : depthEndPattern true 2 = Transformer.CRASP.altList false 2 ∧
    depthEndPattern false 2 = Transformer.CRASP.altList true 2 := by decide

/-- Both actual hard ending-patterns are exactly the original abab/baba alternating targets.
Source: Basis E_4's independent alternating-subsequence criterion, not a longer-pattern relaxation. -/
theorem depthEndPattern_hard : depthEndPattern true 4 = Transformer.CRASP.altList false 4 ∧
    depthEndPattern false 4 = Transformer.CRASP.altList true 4 := by decide

/-- The true raw easy depth answer is determined by the two ordered ending-occurrence presences.
Source: the independent actual integer oracle and proved original-position E_2 pattern identities. -/
theorem depthNext_easy_occurrences (word : List (Option Bool)) :
    depthNext 2 (bos :: depthBody word) =
      if (∃ i : Fin word.length, DepthOccurrence 1 true word i) ∧
          ¬(∃ i : Fin word.length, DepthOccurrence 1 false word i) then accept else reject := by
  rw [depthNext_body]
  simp only [depthOccurrence_exists, depthEndPattern_easy.1, depthEndPattern_easy.2]

/-- The true raw hard depth answer is determined by actual four-letter ordered ending occurrences.
Source: the independent integer oracle and exact original-position E_4 identities, including wrong-start rejection. -/
theorem depthNext_hard_occurrences (word : List (Option Bool)) :
    depthNext 4 (bos :: depthBody word) =
      if (∃ i : Fin word.length, DepthOccurrence 3 true word i) ∧
          ¬(∃ i : Fin word.length, DepthOccurrence 3 false word i) then accept else reject := by
  rw [depthNext_body]
  simp only [depthOccurrence_exists, depthEndPattern_hard.1, depthEndPattern_hard.2]

end Transformer.GPTMini.Semantics
