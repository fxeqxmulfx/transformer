module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Training

/-!
# Separating data and a failure without it

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR4.
Injective codes give ordinary training dictionaries that separate keys:
each code coordinate is one dictionary, queried at every key. The model
does not receive the codes. Conversely, constant training dictionaries
leave a wrong matching matrix globally optimal, despite convex training.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexRecall

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {κ ν ι : Type*}

/-- An injective family of value codes supplies separating training data.
Each `(query, coordinate)` is an ordinary labeled recall example (CR4). -/
theorem separates_of_injective_code (code : κ → ι → ν) (hc : Function.Injective code) :
    Separates (fun s : κ × ι => s.1) (fun s k => code k s.2) := by
  intro q k hk
  obtain ⟨r, hr⟩ : ∃ r, code k r ≠ code q r := by
    by_contra h
    push Not at h
    exact hk (hc (funext h))
  exact ⟨(q, r), rfl, hr⟩

/-- Four keys represented by two Boolean coordinates, used only to
construct two ordinary training dictionaries (CR4). -/
def twoBitCode (k : Bool × Bool) (r : Bool) : Bool := if r then k.2 else k.1

theorem twoBitCode_injective : Function.Injective twoBitCode := by
  intro x y h
  have h0 := congrFun h false
  have h1 := congrFun h true
  exact Prod.ext (by simpa [twoBitCode] using h0) (by simpa [twoBitCode] using h1)

/-- Eight samples (four queries on two dictionaries) identify matching
over all sixteen dictionaries of four Boolean bindings (CR4). -/
example : Separates (fun s : (Bool × Bool) × Bool => s.1)
    (fun s k => twoBitCode k s.2) :=
  separates_of_injective_code twoBitCode twoBitCode_injective

variable [Fintype κ] [DecidableEq κ]

/-- A deterministic but not necessarily correct key-matching matrix (CR4). -/
def selector (f : κ → κ) : Matrix κ κ ℝ := fun q k => if k = f q then 1 else 0

/-- Every deterministic routing function belongs to the same convex
parameter domain; incorrect matching is not excluded by definition (CR4). -/
theorem stochastic_selector (f : κ → κ) : IsStochastic (selector f) := by
  constructor
  · intro q k
    simp only [selector]
    split_ifs <;> norm_num
  · intro q
    simp [selector]

variable [DecidableEq ν]

/-- A selector copies the value of its selected key, even if wrong (CR4). -/
theorem logits_selector (f : κ → κ) (q : κ) (D : κ → ν) (c : ν) :
    logits (selector f) q D c = if D (f q) = c then 1 else 0 := by
  unfold logits selector
  rw [Finset.sum_eq_single (f q)]
  · simp
  · intro k _ hk
    simp [hk]
  · simp

variable {S : Type*} [Fintype S] [Fintype ν]

/-- If the training values never distinguish a selected key from its
query, that selection has the same global-minimum answer CE (CR4). -/
theorem selector_isMinOn (query : S → κ) (data : S → κ → ν) (f : κ → κ)
    (h : ∀ s, data s (f (query s)) = data s (query s)) :
    IsMinOn (answerLoss query data) {A | IsStochastic A} (selector f) := by
  have he : answerLoss query data (selector f) = answerLoss query data 1 := by
    apply Finset.sum_congr rfl
    intro s _
    congr 1
    funext c
    rw [logits_selector, logits_one, h s]
  intro A hA
  rw [he]
  exact answerLoss_one_le query data hA

/-- Both labels occur in this nonseparating dataset: sample `s` assigns
value `s` to every key and asks query `s`. -/
theorem constant_data_not_separating :
    ¬ Separates (id : Bool → Bool) (fun s : Bool => fun _ : Bool => s) := by
  intro h
  obtain ⟨s, -, hs⟩ := h false true (by decide)
  exact hs rfl

/-- **Convex answer training can have an incorrect global optimizer.**
With constant training dictionaries, swapping two keys is a feasible
global minimum, yet it gives the strictly wrong answer on `D = id` (CR4).
This prevents interpreting convexity alone as a generalization theorem. -/
theorem wrong_global_minimum_without_separation :
    IsStochastic (selector Bool.not) ∧
    IsMinOn (answerLoss (id : Bool → Bool) (fun s : Bool => fun _ : Bool => s))
      {A | IsStochastic A} (selector Bool.not) ∧
    logits (selector Bool.not) false (id : Bool → Bool) false <
      logits (selector Bool.not) false (id : Bool → Bool) true := by
  refine ⟨stochastic_selector _, selector_isMinOn _ _ _ (fun _ => rfl), ?_⟩
  norm_num [logits_selector]

example := stochastic_selector (id : Bool → Bool)

example := logits_selector Bool.not false (id : Bool → Bool) true

example := selector_isMinOn (id : Bool → Bool) (fun s : Bool => fun _ : Bool => s)
  Bool.not (fun _ => rfl)

end Transformer.Grokking.FlatSpectrum.ConvexRecall
