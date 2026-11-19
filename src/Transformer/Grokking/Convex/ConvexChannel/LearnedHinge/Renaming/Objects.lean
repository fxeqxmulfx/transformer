module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming.Prediction

/-!
# Immutable object witnesses with a fixed control object

Source: `review/convex/moments.md`, CM78. `none` is an ordinary control object;
the two other objects exchange names. The same set supplies inputs and next
objects. Two observed prefixes have different labels. A third prefix is
unseen and obtained by the declared renaming. The answer rule is only an
inhabited data assumption; it is never called by the model forward.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming.Witness

open PatternReduction

/-- Three immutable objects, including one fixed control identity. -/
abbrev Object := Option Bool

/-- A fixed object renaming that leaves the control object alone, CM78. -/
def flip : Equiv.Perm Object :=
  Function.Involutive.toPerm (Option.map Bool.not) (by
    intro o
    cases o with
    | none => rfl
    | some b => cases b <;> rfl)

/-- This nontrivial renaming really is involutive, CM78. -/
theorem flip_involutive : Function.Involutive flip := by
  intro o
  cases o with
  | none => rfl
  | some b => cases b <;> rfl

/-- First observed immutable sequence. -/
def seen : Prefix Object 1 := ⟨[some false], by decide⟩

/-- Second observed sequence has a different next-object label. -/
def control : Prefix Object 1 := ⟨[none], by decide⟩

/-- This third sequence is absent from both original observed records. -/
def unseen : Prefix Object 1 := ⟨[some true], by decide⟩

/-- The actual renaming connects the observed and unseen sequences, CM78. -/
theorem renamed_seen : renamed flip seen = unseen := rfl

/-- Control-object inputs remain fixed, CM78. -/
theorem renamed_control : renamed flip control = control := rfl

/-- The evaluation prefix differs from both observed inputs, CM78. -/
theorem unseen_distinct : unseen ≠ seen ∧ unseen ≠ control := by decide

/-- A task answer rule used only to inhabit the transfer assumption, CM78.
It is neither an encoder nor a model parameter or inference program. -/
def answer (tokens : Prefix Object 1) : Object := tokens.val.headD none

/-- The example task respects the declared renaming at every admitted
prefix, empty inputs included, CM78. -/
theorem answer_symmetry (tokens : Prefix Object 1) :
    answer (renamed flip tokens) = flip (answer tokens) := by
  change (tokens.val.map flip).headD (flip none) = flip (tokens.val.headD none)
  exact List.headD_map

/-- The observed answers really differ and the missing answer is moved,
CM78. Their identities belong to the same original object set. -/
theorem answer_values : answer seen = some false ∧ answer control = none ∧
    answer unseen = some true ∧ answer seen ≠ answer control := by decide

/-- Two permitted learned directions yield an actual feasible base point,
CM78. Every admitted prefix and every original output object is specified.
This is an existence witness, not a claimed training trajectory. -/
theorem base_exists :
    ∃ p : ReducedSequenceParameters Object 1,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ tokens : Prefix Object 1, ∀ c : Object, sequenceLogits p tokens c =
        (if tokens.val = [some false] ∧ c = some false then 4 else 0) +
        (if tokens.val = [none] ∧ c = none then 2 else 0) := by
  obtain ⟨p, hp, hpred⟩ := Unseen.shifted_logits_exists (O := Object) (T := 1)
    0 (zero_mem _ _ _) (fun _ => some false) (some false) 4
  obtain ⟨q, hq, hqpred⟩ := Unseen.shifted_logits_exists p hp (fun _ => none) none 2
  refine ⟨q, hq, ?_⟩
  intro tokens c
  rw [hqpred, hpred, Unseen.Witness.sequenceLogits_zero]
  simp [List.ofFn_succ]

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming.Witness
