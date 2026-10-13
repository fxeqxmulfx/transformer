import Mathlib.Data.List.Range
import Mathlib.Tactic

/-!
# Threshold events on a finite observed training prefix

Empirical source: Power et al., arXiv:2201.02177v1, sections 1 and 3.1,
and Figure 1, which separates training fit from first 99% validation
accuracy. Diagnostic source: lab.domain.grokking and lab.domain.phases at
43d4d66; patience and sampling cadence are explicit study choices.

These are exact-real predicates on observation-indexed accuracy traces.
They formalize first crossings and finite-budget censoring, not an ODE,
an optimizer or an asymptotic generalization theorem. A first threshold
crossing and confirmation after several observations are different events.
The complete Python progress heuristic, including medians and structure
alarms, is not identified with these primitive predicates.

An isolated crossing can be followed by a decline. Neither its uniqueness
nor its stability under extension establishes sustained generalization.
The observation clock also omits changes between recorded checkpoints;
the first observed crossing need not be the first optimizer-step crossing.
No missing checkpoint or unrecorded intermediate accuracy is reconstructed.
-/

namespace Transformer.Grokking.Operational

/-- First observed attainment of a target. Source: Power et al.,
arXiv:2201.02177v1, section 3.1 learning-time curves; observation indices
replace optimizer steps until an explicit sampling clock is supplied.
The non-strict boundary follows our study; the paper also uses `> 99%`. -/
def FirstCrossing (score : ℕ → ℝ) (target : ℝ) (time : ℕ) : Prop :=
  target ≤ score time ∧ ∀ earlier, earlier < time → score earlier < target

/-- No target attainment in the available prefix, including its final
observation. Source: finite optimization budgets in arXiv:2201.02177v1,
section 3.1; this predicate makes no statement about later observations. -/
def NoCrossingThrough (score : ℕ → ℝ) (target : ℝ) (budget : ℕ) : Prop :=
  ∀ time, time ≤ budget → score time < target

/-- The actual values available through a budget. Source: causal history
slicing in lab.domain.grokking at 43d4d66. This records the prefix rather
than assuming any property of an observer or its future prediction. -/
def scorePrefix (score : ℕ → ℝ) (budget : ℕ) : List ℝ :=
  (List.range (budget + 1)).map score

/-- A first crossing is unique. Source: the first-attainment statistic of
arXiv:2201.02177v1, section 3.1, with the threshold explicit. -/
theorem firstCrossing_unique (score : ℕ → ℝ) (target : ℝ) (s t : ℕ)
    (hs : FirstCrossing score target s) (ht : FirstCrossing score target t) : s = t := by
  rcases lt_trichotomy s t with hst | heq | hts
  · have hlow := ht.2 s hst
    linarith [hs.1]
  · exact heq
  · have hlow := hs.2 t hts
    linarith [ht.1]

example : FirstCrossing (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 3 ∧
    FirstCrossing (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 3 := by
  constructor <;> constructor
  · norm_num
  · intro n hn
    have h : ¬3 ≤ n := by omega
    simp [h]
  · norm_num
  · intro n hn
    have h : ¬3 ≤ n := by omega
    simp [h]

/-- A crossing after a censored prefix must occur beyond its budget.
Source: finite-budget reporting in arXiv:2201.02177v1, section 3.1.
Its existence remains a premise; censoring alone does not imply it. -/
theorem firstCrossing_after_censored_prefix (score : ℕ → ℝ) (target : ℝ) (budget time : ℕ)
    (hc : NoCrossingThrough score target budget) (ht : FirstCrossing score target time) :
    budget < time := by
  by_contra h
  have hle : time ≤ budget := by omega
  have hlow := hc time hle
  linarith [ht.1]

example : NoCrossingThrough (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 2 ∧
    FirstCrossing (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 3 := by
  constructor
  · intro n hn
    have h : ¬3 ≤ n := by omega
    simp [h]
  · constructor
    · norm_num
    · intro n hn
      have h : ¬3 ≤ n := by omega
      simp [h]

/-- Replacing only future observations cannot rewrite a recorded first
crossing. Source: the causal-prefix requirement of lab.domain.grokking at
43d4d66; no claim that its full Python implementation is proved follows. -/
theorem firstCrossing_prefix_invariant (f g : ℕ → ℝ) (target : ℝ) (budget time : ℕ)
    (ht : time ≤ budget) (heq : ∀ n, n ≤ budget → f n = g n) :
    FirstCrossing f target time ↔ FirstCrossing g target time := by
  constructor
  · intro h
    refine ⟨?_, ?_⟩
    · rw [← heq time ht]
      exact h.1
    · intro n hn
      rw [← heq n (by omega)]
      exact h.2 n hn
  · intro h
    refine ⟨?_, ?_⟩
    · rw [heq time ht]
      exact h.1
    · intro n hn
      rw [heq n (by omega)]
      exact h.2 n hn

example : (3 : ℕ) ≤ 10 ∧ ∀ n : ℕ, n ≤ 10 →
    (if n ≤ 10 then (1 : ℝ) else 0) = 1 := by
  refine ⟨by norm_num, ?_⟩
  intro n hn
  simp [hn]

/-- Prefix equality is proved from exactly the available observations.
Source: causal history slicing in lab.domain.grokking at 43d4d66. -/
theorem scorePrefix_eq (f g : ℕ → ℝ) (budget : ℕ)
    (heq : ∀ n, n ≤ budget → f n = g n) : scorePrefix f budget = scorePrefix g budget := by
  unfold scorePrefix
  apply List.map_congr_left
  intro n hn
  have hr := List.mem_range.mp hn
  exact heq n (by omega)

example : ∀ n : ℕ, n ≤ 10 → (if n ≤ 10 then (1 : ℝ) else 0) = 1 := by
  intro n hn
  simp [hn]

/-- Confirmation of `patience` consecutive observations adds exactly
`cadence * (patience - 1)` clock steps to the window's onset. Source:
lab.domain.phases.sustained at 43d4d66; this is measurement latency,
not additional optimization required for the first threshold crossing. -/
theorem confirmation_clock_delay (cadence onset patience : ℕ) (hp : 0 < patience) :
    cadence * (onset + patience - 1) - cadence * onset = cadence * (patience - 1) := by
  have he : onset + patience - 1 = onset + (patience - 1) := by omega
  rw [he, Nat.mul_add, Nat.add_sub_cancel_left]

example : 0 < (5 : ℕ) := by norm_num

/-- The actual study's sampling clock contributes 1,000 confirmation
steps for five observations spaced by 250. Source: ProgressCriterion at
43d4d66 and the measured onset 35,500 versus confirmation 36,500. -/
theorem study_confirmation_delay : (250 : ℕ) * (5 - 1) = 1000 := by
  norm_num

end Transformer.Grokking.Operational
