module

public import Transformer.Phasor.AssociativeEquationScore

/-! # Exact scores for GPT register programs with multiple compute rows

DiffGPT 5a37c515, `mqar_div97._program_attempt`, `Task.tensors`, `rollout`,
`current_grokking.adapters.associative` and `DenseTask.accumulate`.
The model sees a chronological load prefix and the supplied keys of each
compute row. A source frame stores the correct earlier results. Exact
sequence scoring agrees with inserting independently predicted earlier
results, even when an incorrect earlier answer changes later predictions.
-/

@[expose] public section

namespace Transformer.Phasor.RegisterProgramScore
open Transformer.Basis
variable {A : Type*}

/-- Append both externally supplied keys, then append the model's own answer
before the next row. No future compute result enters the predictor. -/
def generate (next : List A → A) : List (A × A) → List A → List A
  | [], _ => []
  | (u,v)::rows, xs =>
    let p := xs ++ [u,v]
    let y := next p
    y :: generate next rows (p ++ [y])

/-- The dense source instead stores the annotated answer in subsequent inputs.
It still predicts each supervised entry after seeing only its two query keys. -/
def onFrame (next : List A → A) : List ((A × A) × A) → List A → List A
  | [], _ => []
  | ((u,v),y)::rows, xs =>
    let p := xs ++ [u,v]
    next p :: onFrame next rows (p ++ [y])

/-- The source's single exact bit for all supervised positions equals
the score from greedy, causal generation with the same supplied query keys. -/
theorem exact_iff (next : List A → A) (rows : List ((A × A) × A)) (xs : List A) :
    generate next (rows.map Prod.fst) xs = rows.map Prod.snd ↔
      onFrame next rows xs = rows.map Prod.snd := by
  induction rows generalizing xs with
  | nil => simp [generate,onFrame]
  | cons row rows ih =>
    obtain ⟨⟨u,v⟩,y⟩ := row
    simp only [List.map_cons,generate,onFrame,List.cons.injEq]
    by_cases hit : next (xs ++ [u,v]) = y
    · simp only [hit,true_and]
      exact ih (xs ++ [u,v] ++ [y])
    · simp [hit]

/-- The source masks all load entries. At each compute row the second query
key predicts the following value, including the answer omitted from the input. -/
def target (loads : ℕ) (answers : List A) (i : ℕ) : Option A :=
  if 3*loads ≤ i ∧ (i-3*loads)%3=1 then answers[(i-3*loads)/3]? else none

theorem target_iff (loads : ℕ) (answers : List A) (i : ℕ) (y : A) :
    target loads answers i = some y ↔
      ∃ j, i = 3*(loads+j)+1 ∧ answers[j]? = some y := by
  constructor
  · intro h
    unfold target at h
    split at h
    · rename_i ok
      exact ⟨(i-3*loads)/3, by omega, h⟩
    · contradiction
  · rintro ⟨j,rfl,h⟩
    have lo : 3*loads ≤ 3*(loads+j)+1 := by omega
    have rem : (3*(loads+j)+1-3*loads)%3=1 := by omega
    have quo : (3*(loads+j)+1-3*loads)/3=j := by omega
    simpa only [target,lo,rem,and_self,ite_true,quo] using h

/-- The source machine divides the latest register values and writes the
answer back to the first queried register. The subsequent rows run on that state. -/
def execute (op : ℤ → ℤ → ℤ) (ws : List (ℤ × ℤ)) :
    List (ℤ × ℤ) → Option (List ℤ)
  | [] => some []
  | (u,v)::rest => do
    let y ← AssociativeEquationScore.label op ws u v
    let tail ← execute op (ws ++ [(u,y)]) rest
    return y::tail

/-- Renaming all register keys leaves every repeated in-place computation's
operand values and complete list of answers unchanged. -/
theorem execute_rekey (e : ℤ ≃ ℤ) (op : ℤ → ℤ → ℤ) (ws queries : List (ℤ × ℤ)) :
    execute op (ws.map (fun r => (e r.1,r.2))) (queries.map (fun q => (e q.1,e q.2)))
      = execute op ws queries := by
  induction queries generalizing ws with
  | nil => rfl
  | cons row qs ih =>
    obtain ⟨u,v⟩ := row
    simp only [List.map_cons, execute, AssociativeEquationScore.label_rekey]
    cases r : AssociativeEquationScore.label op ws u v with
    | none => rfl
    | some y =>
      have eq := ih (ws ++ [(u,y)])
      simpa [List.map_append] using congrArg (fun (result : Option (List ℤ)) => result.bind (fun tail => some (y::tail))) eq

private def writes : List (ℤ × ℤ) := [(-1,12),(-2,4),(-3,2)]
example : execute (·/·) writes [(-1,-2),(-2,-3),(-1,-2)] = some [3,2,1] := by decide
example : execute (·/·) writes [(-1,-2),(-9,-3)] = none := by decide
example : target 8 [3,2,1] 25 = some 3 ∧ target 8 [3,2,1] 28 = some 2 ∧
    target 8 [3,2,1] 31 = some 1 ∧ target 8 [3,2,1] 29 = none := by decide

/-- A mistaken first answer changes the second prediction. Both binary
exact-sequence scorers count the whole example as incorrect. -/
example :
    let rows : List ((ℕ × ℕ) × ℕ) := [((1,2),9),((3,4),19)]
    let predict := fun xs : List ℕ => xs.sum
    generate predict (rows.map Prod.fst) [] = [3,13] ∧
    onFrame predict rows [] = [3,19] ∧
    generate predict (rows.map Prod.fst) [] ≠ rows.map Prod.snd ∧
    onFrame predict rows [] ≠ rows.map Prod.snd := by decide

end Transformer.Phasor.RegisterProgramScore
