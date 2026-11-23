module

public import Transformer.Phasor.TripleTransport.AdjointStorage

/-! DiffGPT 5a37c515, complex_geometry_suite/native/src/metrics.rs, evaluate.
Each input row contributes one Boolean after checking all supervised
positions. Long recall probes can be scored in consecutive subranges.
Coverage, counts and correctness use the original ordered rows. -/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.ProbeRows

variable {A : Type*}

def groups (rows : List A) : List ℕ → List (List A)
  | [] => []
  | n :: ns => rows.take n :: groups (rows.drop n) ns

/-- An ordered list of chunk sizes covering the complete split recovers
every original row in order, even for a short last chunk. -/
theorem concatenate (rows : List A) (sizes : List ℕ) (cover : rows.length ≤ sizes.sum) :
    (groups rows sizes).flatten = rows := by
  induction sizes generalizing rows with
  | nil =>
    have h : rows = [] := List.length_eq_zero_iff.mp (by simpa using cover)
    simp [groups,h]
  | cons n ns ih =>
    have remaining : (rows.drop n).length ≤ ns.sum := by
      simp only [List.length_drop]
      simp only [List.sum_cons] at cover
      omega
    simpa only [groups,List.flatten_cons,ih _ remaining] using List.take_append_drop n rows

/-- The aggregate exact count equals scoring the original split once.
The pass predicate can check an entire generation including its termination. -/
theorem correct (pass : A → Bool) (rows : List A) (sizes : List ℕ)
    (cover : rows.length ≤ sizes.sum) :
    ((groups rows sizes).map (List.countP pass)).sum = rows.countP pass := by
  have flat (parts : List (List A)) :
      (parts.map (List.countP pass)).sum = parts.flatten.countP pass := by
    induction parts with
    | nil => simp
    | cons p ps ih =>
      change p.countP pass + (ps.map (List.countP pass)).sum = (p ++ ps.flatten).countP pass
      rw [List.countP_append, ih]
  rw [flat,concatenate rows sizes cover]

example : (groups [1,3,2,0] [3,5]).flatten = [1,3,2,0] :=
  concatenate _ _ (by decide)

example : ((groups [1,3,2,0] [3,5]).map (List.countP (fun n => decide (n ≤ 1)))).sum =
    [1,3,2,0].countP (fun n => decide (n ≤ 1)) :=
  correct _ _ _ (by decide)

end Transformer.Phasor.TripleTransport.ProbeRows
