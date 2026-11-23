module

public import Transformer.Phasor.ModularChainScore
public import Mathlib.Algebra.Field.ZMod
public import Transformer.Basis.RecallSemantics

/-!
# Latest register values and exact scoring for single-compute GPT equations

Source: DiffGPT `mqar_div97.queries`, `labeled_rows`, `Task.tensors` and
`current_grokking.adapters.associative` at 5a37c515. Arbitrary overwrites
and distinct queries supply two operands. The external label table determines
the only supervised entry after the final query key.
-/

@[expose] public section

namespace Transformer.Phasor.AssociativeEquationScore

open Transformer.Basis
variable {A Y : Type*}

/-- DiffGPT 5a37c515, mqar_div97.queries: writes are chronological triples.
The two keys of the compute row conclude the observable prefix. -/
def prompt (encode : ℤ → A) (writes : List (ℤ × ℤ)) (u v : ℤ) : List A :=
  writes.flatMap (fun (k,x) => [encode k, encode (-96), encode x]) ++ [encode u, encode v]

theorem prompt_length (encode : ℤ → A) (writes : List (ℤ × ℤ)) (u v : ℤ) :
    (prompt encode writes u v).length = 3 * writes.length + 2 := by
  induction writes with
  | nil => simp [prompt]
  | cons row rest ih =>
    obtain ⟨k,x⟩ := row
    simp only [prompt, List.flatMap_cons, List.cons_append, List.nil_append,
      List.length_cons] at ih ⊢
    omega

/-- A latest write can be followed by arbitrary writes to other keys.
This also describes the source when a key was overwritten. -/
def Last (ws : List (ℤ × ℤ)) (k x : ℤ) : Prop :=
  ∃ before after, ws = before ++ (k,x) :: after ∧ ∀ row ∈ after, row.1 ≠ k

/-- DiffGPT 5a37c515, labeled_rows and queries. The operation is external
and may also be the frozen random-label table on distinct equations. -/
def label (op : ℤ → ℤ → Y) (ws : List (ℤ × ℤ)) (u v : ℤ) : Option Y :=
  match latestValue ws u, latestValue ws v with
  | some a, some b => some (op a b)
  | _, _ => none

theorem label_iff (op : ℤ → ℤ → Y) (ws : List (ℤ × ℤ)) (u v : ℤ) (y : Y) :
    label op ws u v = some y ↔ ∃ a b, Last ws u a ∧ Last ws v b ∧ op a b = y := by
  change _ ↔ ∃ a b, (∃ before after, ws = before ++ (u,a) :: after ∧
    ∀ row ∈ after, row.1 ≠ u) ∧ (∃ before after, ws = before ++ (v,b) :: after ∧
    ∀ row ∈ after, row.1 ≠ v) ∧ _
  simp only [← latestValue_some_iff]
  cases hu : latestValue ws u <;> cases hv : latestValue ws v <;> simp [label,hu,hv]

/-- Renaming memory keys commutes with both latest-write lookups, even when
keys recur. Values are carried through without renaming. -/
theorem latest_rekey (e : ℤ ≃ ℤ) (ws : List (ℤ × ℤ)) (k : ℤ) :
    latestValue (ws.map (fun row => (e row.1,row.2))) (e k) = latestValue ws k := by
  induction ws with
  | nil => rfl
  | cons row rest ih =>
    obtain ⟨j,x⟩ := row
    simp [latestValue,ih]

theorem label_rekey (e : ℤ ≃ ℤ) (op : ℤ → ℤ → Y) (ws : List (ℤ × ℤ)) (u v : ℤ) :
    label op (ws.map (fun row => (e row.1,row.2))) (e u) (e v) = label op ws u v := by
  simp [label,latest_rekey]

/-- At any write count the source masks every position through the first
queried key. Exact-sequence accuracy compares only the full-prefix query
with the encoded operation on both most recent registers. -/
theorem score_iff (encode : ℤ → A) (out : Y → A) (op : ℤ → ℤ → Y)
    (ws : List (ℤ × ℤ)) (u v a b : ℤ) (ha : Last ws u a) (hb : Last ws v b)
    (predict : List A → A) (y : Y) (truth : label op ws u v = some y) :
    (∀ i z, (ModularChainScore.targets (3 * ws.length + 1) (out y))[i]? = some (some z) →
      predict ((prompt encode ws u v).take (i+1)) = z) ↔
      predict (prompt encode ws u v) = out (op a b) := by
  have hu := (latestValue_some_iff ws u a).2 ha
  have hv := (latestValue_some_iff ws v b).2 hb
  have hy : op a b = y := by simpa [label,hu,hv] using truth
  subst y
  have full : (prompt encode ws u v).take (3 * ws.length + 1 + 1) = prompt encode ws u v :=
    List.take_of_length_le (by rw [prompt_length])
  constructor
  · intro h
    simpa only [full] using h _ _ ((ModularChainScore.supervised_iff _ _ _ _).2 ⟨rfl,rfl⟩)
  · intro h i z ok
    obtain ⟨rfl,rfl⟩ := (ModularChainScore.supervised_iff _ _ _ _).1 ok
    simpa only [full] using h

private def overwritten : List (ℤ × ℤ) := [(-2,11),(-1,4),(-2,8),(-3,0)]
local instance : Fact (Nat.Prime 97) := ⟨by decide⟩
private def quotient (a b : ℤ) : ZMod 97 := (a : ZMod 97) / (b : ZMod 97)

example : label quotient overwritten (-2) (-1) = some 2 ∧
    (prompt id overwritten (-2) (-1)).length = 14 := by
  constructor
  · change some ((8 : ZMod 97) / 4) = some 2
    congr 1
    apply (div_eq_iff (by decide : (4 : ZMod 97) ≠ 0)).2
    decide
  · decide

example (predict : List ℤ → ℤ) :
    (∀ i z, (ModularChainScore.targets 13 (2 : ℤ))[i]? = some (some z) →
      predict ((prompt id overwritten (-2) (-1)).take (i+1)) = z) ↔
      predict (prompt id overwritten (-2) (-1)) = 2 := by
  exact score_iff id id (· / ·) overwritten (-2) (-1) 8 4
    (⟨[(-2,11),(-1,4)],[(-3,0)],rfl,by simp⟩)
    (⟨[(-2,11)],[(-2,8),(-3,0)],rfl,by simp⟩) predict 2 (by decide)

end Transformer.Phasor.AssociativeEquationScore
