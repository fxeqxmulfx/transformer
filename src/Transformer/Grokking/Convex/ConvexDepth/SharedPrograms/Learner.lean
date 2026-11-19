module

public import Transformer.Grokking.Convex.ConvexDepth.SharedPrograms.Basic

/-!
# Executable elimination and consensus for shared input programs

Source: `review/convex/moments.md`, CM52. The learner receives the fixed program
bank, a candidate order and observed input/label pairs. It never receives the
unknown target program. Query consensus can determine a raw class absent from
the observed labels, conditional on target realizability within this bank.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.SharedPrograms

variable {J X C : Type*} [DecidableEq C]

/-- Keep exactly the fixed input programs agreeing with all supplied labels,
CM52. Repeated observations and candidate order are retained. -/
def retained (program : J → X → C) (order : List J) (samples : List (X × C)) : List J :=
  order.filter fun j => samples.all fun s => decide (program j s.1 = s.2)

/-- The executable filter represents precisely the training-compatible
programs, CM52. No statement about missing labels is hidden in the predicate. -/
theorem mem_retained (program : J → X → C) (order : List J) (samples : List (X × C))
    (j : J) : j ∈ retained program order samples ↔
      j ∈ order ∧ ∀ s ∈ samples, program j s.1 = s.2 := by
  simp only [retained, List.mem_filter, List.all_eq_true, decide_eq_true_eq]

/-- An empty version space or disagreement returns none; otherwise return
the shared predicted symbol, CM52. All comparisons are executable. -/
def consensus (program : J → X → C) (candidates : List J) (x : X) : Option C :=
  match candidates with
  | [] => none
  | j :: rest => if rest.all (fun k => decide (program k x = program j x)) then
      some (program j x) else none

/-- Acceptance is equivalent to nonempty unanimous program predictions,
CM52. The checked unanimity concerns visible program outputs, not an oracle. -/
theorem consensus_some_iff (program : J → X → C) (candidates : List J) (x : X) (y : C) :
    consensus program candidates x = some y ↔
      candidates ≠ [] ∧ ∀ j ∈ candidates, program j x = y := by
  cases candidates with
  | nil => simp [consensus]
  | cons j rest =>
    simp only [consensus]
    split
    · rename_i h
      have hall : ∀ k ∈ rest, program k x = program j x := by
        simpa only [List.all_eq_true, decide_eq_true_eq] using h
      simp only [Option.some.injEq]
      constructor
      · intro hy
        refine ⟨by simp, ?_⟩
        intro k hk
        rcases List.mem_cons.mp hk with hkj | hkr
        · subst k
          exact hy
        · exact (hall k hkr).trans hy
      · intro hgood
        exact hgood.2 j (by simp)
    · rename_i h
      constructor
      · intro impossible
        cases impossible
      · intro hgood
        have hall : ∀ k ∈ rest, program k x = program j x := by
          intro k hk
          exact (hgood.2 k (by simp [hk])).trans (hgood.2 j (by simp)).symm
        have accepted : rest.all (fun k => decide (program k x = program j x)) = true := by
          simpa only [List.all_eq_true, decide_eq_true_eq] using hall
        exact False.elim (h accepted)

/-- A training-only learner followed by input-only consensus, CM52. -/
def predict (program : J → X → C) (order : List J) (samples : List (X × C))
    (x : X) : Option C := consensus program (retained program order samples) x

/-- Realizability and correct observations guarantee that the executable
filter keeps at least one program, CM52. Its unknown index is used only in proof. -/
theorem retained_nonempty (program : J → X → C) (order : List J) (samples : List (X × C))
    (target : X → C) (hreal : ∃ j ∈ order, ∀ x, program j x = target x)
    (hobs : ∀ s ∈ samples, s.2 = target s.1) : retained program order samples ≠ [] := by
  obtain ⟨j, hj, hp⟩ := hreal
  have kept : j ∈ retained program order samples := by
    apply (mem_retained program order samples j).mpr
    exact ⟨hj, fun s hs => (hp s.1).trans (hobs s hs).symm⟩
  intro empty
  rw [empty] at kept
  exact List.not_mem_nil kept

/-- If the unknown target is one enumerated program and observations are
correct, each accepted answer is correct, CM52. Neither its index nor any
held-out label is an argument of predict; flatness supplies no realization. -/
theorem predict_sound (program : J → X → C) (order : List J) (samples : List (X × C))
    (target : X → C) (hreal : ∃ j ∈ order, ∀ x, program j x = target x)
    (hobs : ∀ s ∈ samples, s.2 = target s.1) (x : X) (y : C)
    (hp : predict program order samples x = some y) : y = target x := by
  obtain ⟨j, hj, hprogram⟩ := hreal
  have kept : j ∈ retained program order samples := by
    apply (mem_retained program order samples j).mpr
    refine ⟨hj, ?_⟩
    intro s hs
    exact (hprogram s.1).trans (hobs s hs).symm
  have agreed := (consensus_some_iff program (retained program order samples) x y).mp hp
  exact (agreed.2 j kept).symm.trans (hprogram x)

/-- Fixed generic coordinate programs used to exhibit unseen-symbol transfer.
The Boolean index is unknown to the learner until observations separate it. -/
def witnessProgram (j : Bool) (x : Fin 4 × Fin 4) : Fin 4 := if j then x.2 else x.1

/-- One observed pair distinguishes first-coordinate from last-coordinate copy. -/
def witnessSamples : List ((Fin 4 × Fin 4) × Fin 4) := [((0, 1), 1)]

/-- The executable filter identifies the last-coordinate program from labels,
CM52, rather than receiving the target index as an argument. -/
theorem witness_retained : retained witnessProgram [false, true] witnessSamples = [true] := by
  decide

/-- Class2 never occurs as an observed label, but the actual learner returns
it at a new input under the explicit copy prior, CM52. -/
theorem witness_unseen_symbol :
    predict witnessProgram [false, true] witnessSamples (0, 2) = some 2 ∧
      ∀ s ∈ witnessSamples, s.2 ≠ 2 := by
  decide

/-- With a nonseparating observation both programs survive and disagree at
the query, so the same learner refuses to invent the unseen answer, CM52. -/
theorem witness_ambiguous :
    retained witnessProgram [false, true] [((0, 0), 0)] = [false, true] ∧
      predict witnessProgram [false, true] [((0, 0), 0)] (0, 2) = none ∧
      witnessProgram false (0, 2) ≠ witnessProgram true (0, 2) := by
  decide

/-- The two ambiguous programs also give opposite actual complex margins
with the same flat source code and canonical head, CM52. -/
theorem witness_ambiguous_complex :
    headLogits (ContextMixture.features (bank fourier4 witnessProgram)
        (fun j => if j then 0 else 1)) (copyHead fourier4 24) (0, 3) 0 = 24 ∧
    headLogits (ContextMixture.features (bank fourier4 witnessProgram)
        (fun j => if j then 0 else 1)) (copyHead fourier4 24) (0, 3) 3 = 0 ∧
    headLogits (ContextMixture.features (bank fourier4 witnessProgram)
        (fun j => if j then 1 else 0)) (copyHead fourier4 24) (0, 3) 0 = 0 ∧
    headLogits (ContextMixture.features (bank fourier4 witnessProgram)
        (fun j => if j then 1 else 0)) (copyHead fourier4 24) (0, 3) 3 = 24 := by
  norm_num [mixed_logits fourier4 fourier4_unitary, Fintype.sum_bool, witnessProgram]

example := mem_retained witnessProgram [false, true] witnessSamples true

example := consensus_some_iff witnessProgram [true] (0, 2) 2

example : retained witnessProgram [false, true] witnessSamples ≠ [] := by
  apply retained_nonempty witnessProgram [false, true] witnessSamples Prod.snd
  · exact ⟨true, by simp, by simp [witnessProgram]⟩
  · simp [witnessSamples]

example : (2 : Fin 4) = (Prod.snd : Fin 4 × Fin 4 → Fin 4) (0, 2) := by
  apply predict_sound witnessProgram [false, true] witnessSamples Prod.snd
  · exact ⟨true, by simp, by simp [witnessProgram]⟩
  · simp [witnessSamples]
  · exact witness_unseen_symbol.1

end Transformer.Grokking.Convex.ConvexDepth.SharedPrograms
