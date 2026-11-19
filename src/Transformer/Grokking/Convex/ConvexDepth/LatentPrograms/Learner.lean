module

public import Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.Basic

/-!
# Discovery and consensus with an unknown shared raw decoder

Source: `review/convex/moments.md`, CM53. The target program and its possibly
many-to-one decoder occur only in soundness hypotheses. The executable learner
receives a fixed input-program bank, its order and observed raw labels.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.LatentPrograms

variable {J X S C : Type*} [DecidableEq S] [DecidableEq C]

/-- Apply the training-learned partial decoder to the query's input state,
CM53. A state unseen under this program returns none. -/
def output (program : J → X → S) (samples : List (X × C)) (j : J) (x : X) : Option C :=
  lookup (program j) samples (program j x)

/-- Unanimous known labels from every retained program, CM53. Flattening
the option also rejects the unanimous but wholly unknown answer `some none`. -/
def predict (program : J → X → S) (order : List J) (samples : List (X × C))
    (x : X) : Option C :=
  (SharedPrograms.consensus (output program samples) (retained program order samples) x).join

/-- Acceptance checks nonempty support, observable state coverage and raw
agreement of all surviving programs, CM53. No truth label enters the test. -/
theorem predict_some_iff (program : J → X → S) (order : List J)
    (samples : List (X × C)) (x : X) (y : C) :
    predict program order samples x = some y ↔
      retained program order samples ≠ [] ∧
        ∀ j ∈ retained program order samples, output program samples j x = some y := by
  unfold predict
  rw [Option.join_eq_some_iff, SharedPrograms.consensus_some_iff]

/-- Every candidate admitting a raw decoder for the recorded labels
survives the executable filter, CM53. The decoder is used only in proof. -/
theorem realized_mem_retained (program : J → X → S) (order : List J)
    (samples : List (X × C)) (j : J) (hj : j ∈ order) (decoder : S → C)
    (hfit : ∀ t ∈ samples, decoder (program j t.1) = t.2) :
    j ∈ retained program order samples := by
  apply (mem_retained program order samples j).mpr
  refine ⟨hj, ?_⟩
  intro t ht u hu he
  rw [← hfit t ht, ← hfit u hu, he]

/-- The filter is complete for every deterministic decoder of the observed
latent states, CM53. A supplied default completes unobserved states only. -/
theorem mem_retained_iff_decoder (program : J → X → S) (order : List J)
    (samples : List (X × C)) (default : C) (j : J) :
    j ∈ retained program order samples ↔ j ∈ order ∧
      ∃ decoder : S → C, ∀ t ∈ samples, decoder (program j t.1) = t.2 := by
  constructor
  · intro hj
    refine ⟨((mem_retained program order samples j).mp hj).1,
      completion (program j) samples default, ?_⟩
    intro t ht
    exact completion_of_lookup (program j) samples default (program j t.1) t.2
      (retained_observed program order samples j hj t ht)
  · rintro ⟨hj, decoder, hf⟩
    exact realized_mem_retained program order samples j hj decoder hf

/-- Realizability by an unknown program and unknown decoder guarantees a
nonempty trained version space, CM53. Neither is a learner argument. -/
theorem retained_nonempty (program : J → X → S) (order : List J)
    (samples : List (X × C)) (target : X → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) : retained program order samples ≠ [] := by
  obtain ⟨j, hj, decoder, hd⟩ := hreal
  have kept := realized_mem_retained program order samples j hj decoder
    (fun t ht => (hd t.1).trans (hobs t ht).symm)
  intro empty
  rw [empty] at kept
  exact List.not_mem_nil kept

/-- Each accepted raw label is correct under the explicit latent-program
realizability premise, CM53. Coverage is checked by the executable predictor. -/
theorem predict_sound (program : J → X → S) (order : List J)
    (samples : List (X × C)) (target : X → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) (x : X) (y : C)
    (hp : predict program order samples x = some y) : y = target x := by
  obtain ⟨j, hj, decoder, hd⟩ := hreal
  have kept := realized_mem_retained program order samples j hj decoder
    (fun t ht => (hd t.1).trans (hobs t ht).symm)
  have found := ((predict_some_iff program order samples x y).mp hp).2 j kept
  obtain ⟨t, ht, hs, hy⟩ := lookup_some_mem (program j) samples (program j x) y found
  calc
    y = t.2 := hy.symm
    _ = target t.1 := hobs t ht
    _ = decoder (program j t.1) := (hd t.1).symm
    _ = decoder (program j x) := congrArg decoder hs
    _ = target x := hd x

/-- A surviving candidate with an unknown query state prevents consensus,
CM53. A numerical default cannot turn this refusal into a certificate. -/
theorem predict_none_of_unknown (program : J → X → S) (order : List J)
    (samples : List (X × C)) (x : X) (j : J)
    (hj : j ∈ retained program order samples) (hu : output program samples j x = none) :
    predict program order samples x = none := by
  cases hp : predict program order samples x with
  | none => rfl
  | some y =>
    have h := ((predict_some_iff program order samples x y).mp hp).2 j hj
    rw [hu] at h
    cases h

/-- Without CM52's copy prior or another identifying assumption, this
learned dictionary cannot invent an unobserved raw class, CM53. It can identify
new inputs mapping to already covered latent states. -/
theorem accepted_label_observed (program : J → X → S) (order : List J)
    (samples : List (X × C)) (x : X) (y : C)
    (hp : predict program order samples x = some y) : ∃ t ∈ samples, t.2 = y := by
  obtain ⟨hne, hall⟩ := (predict_some_iff program order samples x y).mp hp
  cases he : retained program order samples with
  | nil => exact False.elim (hne he)
  | cons j rest =>
    have hj : j ∈ retained program order samples := by rw [he]; simp
    obtain ⟨t, ht, _, hy⟩ := lookup_some_mem (program j) samples (program j x) y (hall j hj)
    exact ⟨t, ht, hy⟩

/-- Two fixed coordinate programs for inhabited examples of the learner. -/
def exampleProgram (j : Bool) (x : Bool × Bool) : Bool := if j then x.2 else x.1

/-- These observed labels eliminate the first-coordinate latent program. -/
def exampleSamples : List ((Bool × Bool) × Bool) :=
  [((false, false), false), ((false, true), true)]

example := predict_some_iff exampleProgram [false, true] exampleSamples (true, true) true

example := realized_mem_retained exampleProgram [false, true] exampleSamples true (by simp)
  id (by simp [exampleProgram, exampleSamples])

example := mem_retained_iff_decoder exampleProgram [false, true] exampleSamples false true

example := retained_nonempty exampleProgram [false, true] exampleSamples Prod.snd
  ⟨true, by simp, id, by simp [exampleProgram]⟩ (by simp [exampleSamples])

example := predict_sound exampleProgram [false, true] exampleSamples Prod.snd
  ⟨true, by simp, id, by simp [exampleProgram]⟩ (by simp [exampleSamples])
  (true, true) true (by decide)

example := predict_none_of_unknown exampleProgram [true] [((false, false), false)]
  (true, true) true (by decide) (by decide)

example := accepted_label_observed exampleProgram [false, true] exampleSamples
  (true, true) true (by decide)

end Transformer.Grokking.Convex.ConvexDepth.LatentPrograms
