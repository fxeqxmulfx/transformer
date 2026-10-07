import Transformer.Grokking.Operational.Basic

/-!+# Sustained fit and memorization windows

Empirical source: Power et al., arXiv:2201.02177v1, sections 1 and 3.1,
which distinguish fitted training data from later validation success.
Operational source: lab.domain.phases.sustained and phases, and
lab.domain.grokking.ProgressCriterion at 43d4d66.

These predicates impose bounds on actual observation-indexed scores.
They do not include a desired order of learning events or a prediction
about the future. Window width is positive, so an empty interval cannot
supply fit, memorization or sustained-success evidence. Above windows
use the study's non-strict threshold; below windows use its explicit
near-chance ceiling, which is not a constant from the paper.

Separated thresholds derive disjointness of low and successful windows.
Either order is possible: a low window may follow an earlier success.
The prefix theorems require the whole confirmation window to be observed.
They justify these primitives, not Python's median, longest-stretch,
structure-alarm or complete phase-selection algorithms.
-/

namespace Transformer.Grokking.Operational

/-- A positive number of consecutive observations at or above target.
Source: phases.sustained at 43d4d66; the input is accuracy, not loss. -/
def AboveWindow (score : ℕ → ℝ) (target : ℝ) (start width : ℕ) : Prop :=
  0 < width ∧ ∀ offset, offset < width → target ≤ score (start + offset)

/-- A positive window at or below an explicit ceiling. Source:
phases.phases at 43d4d66, with the diagnostic ceiling made explicit. -/
def BelowWindow (score : ℕ → ℝ) (ceiling : ℝ) (start width : ℕ) : Prop :=
  0 < width ∧ ∀ offset, offset < width → score (start + offset) ≤ ceiling

/-- Train fit and low held-out accuracy at every point of one window.
Source: phases.phases at 43d4d66, adapting the gap in arXiv:2201.02177v1,
section 3.1. First attainment and future success are separate properties. -/
def MemorizationWindow (train held : ℕ → ℝ) (fit ceiling : ℝ)
    (start width : ℕ) : Prop :=
  AboveWindow train fit start width ∧ BelowWindow held ceiling start width

/-- Each point of a memorization window has the measured train/test gap.
Source: the separated fit/ceiling diagnostic at 43d4d66, adapting
arXiv:2201.02177v1, section 3.1. The gap follows from the two score bounds. -/
theorem memorization_window_gap (train held : ℕ → ℝ) (fit ceiling : ℝ)
    (start width offset : ℕ) (hm : MemorizationWindow train held fit ceiling start width)
    (ho : offset < width) :
    fit - ceiling ≤ train (start + offset) - held (start + offset) := by
  have hfit := hm.1.2 offset ho
  have hlow := hm.2.2 offset ho
  linarith

example : MemorizationWindow (fun _ => (1 : ℝ)) (fun _ => (0 : ℝ))
    1 (1 / 10) 2 3 ∧ (1 : ℕ) < 3 := by
  refine ⟨⟨⟨by norm_num, ?_⟩, ⟨by norm_num, ?_⟩⟩, by norm_num⟩ <;>
    intro n hn <;> norm_num

/-- Separated low and successful windows cannot overlap, in either
temporal order. Source: phases.phases at 43d4d66, adapting the gap in
arXiv:2201.02177v1, section 3.1; later recovery is not excluded. -/
theorem below_above_windows_disjoint (score : ℕ → ℝ) (ceiling target : ℝ)
    (lowStart lowWidth highStart highWidth : ℕ)
    (hl : BelowWindow score ceiling lowStart lowWidth)
    (hh : AboveWindow score target highStart highWidth) (hs : ceiling < target) :
    lowStart + lowWidth ≤ highStart ∨ highStart + highWidth ≤ lowStart := by
  by_cases horder : lowStart ≤ highStart
  · left
    by_contra h
    have ho : highStart - lowStart < lowWidth := by omega
    have hlow := hl.2 (highStart - lowStart) ho
    have he : lowStart + (highStart - lowStart) = highStart := by omega
    rw [he] at hlow
    have hhigh := hh.2 0 hh.1
    simp only [Nat.add_zero] at hhigh
    linarith
  · right
    by_contra h
    have ho : lowStart - highStart < highWidth := by omega
    have hhigh := hh.2 (lowStart - highStart) ho
    have he : highStart + (lowStart - highStart) = lowStart := by omega
    rw [he] at hhigh
    have hlow := hl.2 0 hl.1
    simp only [Nat.add_zero] at hlow
    linarith

example : BelowWindow (fun n => if 3 ≤ n then (1 : ℝ) else 0) (1 / 10) 0 3 ∧
    AboveWindow (fun n => if 3 ≤ n then (1 : ℝ) else 0) (99 / 100) 3 2 ∧
    (1 / 10 : ℝ) < 99 / 100 := by
  refine ⟨⟨by norm_num, ?_⟩, ⟨by norm_num, ?_⟩, by norm_num⟩
  · intro n hn
    have h : ¬3 ≤ n := by omega
    norm_num [h]
  · intro n hn
    have h : 3 ≤ 3 + n := by omega
    norm_num [h]

/-- An above-threshold window uses only its fully observed prefix.
Source: causal slicing in lab.domain.grokking at 43d4d66; its last
observation, rather than merely its onset, must be within the budget. -/
theorem above_window_prefix_invariant (f g : ℕ → ℝ) (target : ℝ)
    (start width budget : ℕ) (hw : start + width ≤ budget + 1)
    (he : ∀ n, n ≤ budget → f n = g n) :
    AboveWindow f target start width ↔ AboveWindow g target start width := by
  constructor <;> intro h
  · refine ⟨h.1, ?_⟩
    intro n hn
    rw [← he (start + n) (by omega)]
    exact h.2 n hn
  · refine ⟨h.1, ?_⟩
    intro n hn
    rw [he (start + n) (by omega)]
    exact h.2 n hn

example : (2 : ℕ) + 3 ≤ 9 + 1 ∧ ∀ n : ℕ, n ≤ 9 →
    (if n ≤ 9 then (1 : ℝ) else 0) = 1 := by
  refine ⟨by norm_num, ?_⟩
  intro n hn
  simp [hn]

/-- A below-ceiling window also uses only its fully observed prefix.
Source: the memorization-window scan in lab.domain.grokking at 43d4d66;
no statement about future accuracy is present in the predicate. -/
theorem below_window_prefix_invariant (f g : ℕ → ℝ) (ceiling : ℝ)
    (start width budget : ℕ) (hw : start + width ≤ budget + 1)
    (he : ∀ n, n ≤ budget → f n = g n) :
    BelowWindow f ceiling start width ↔ BelowWindow g ceiling start width := by
  constructor <;> intro h
  · refine ⟨h.1, ?_⟩
    intro n hn
    rw [← he (start + n) (by omega)]
    exact h.2 n hn
  · refine ⟨h.1, ?_⟩
    intro n hn
    rw [he (start + n) (by omega)]
    exact h.2 n hn

example : (2 : ℕ) + 3 ≤ 9 + 1 ∧ ∀ n : ℕ, n ≤ 9 →
    (if n ≤ 9 then (0 : ℝ) else 1) = 0 := by
  refine ⟨by norm_num, ?_⟩
  intro n hn
  simp [hn]

/-- Changing either score only after the observation budget cannot
rewrite a completed memorization window. Source: causal diagnostic
histories in lab.domain.grokking at 43d4d66. -/
theorem memorization_window_prefix_invariant (train₁ train₂ held₁ held₂ : ℕ → ℝ)
    (fit ceiling : ℝ) (start width budget : ℕ) (hw : start + width ≤ budget + 1)
    (ht : ∀ n, n ≤ budget → train₁ n = train₂ n)
    (hh : ∀ n, n ≤ budget → held₁ n = held₂ n) :
    MemorizationWindow train₁ held₁ fit ceiling start width ↔
      MemorizationWindow train₂ held₂ fit ceiling start width := by
  unfold MemorizationWindow
  rw [above_window_prefix_invariant train₁ train₂ fit start width budget hw ht,
    below_window_prefix_invariant held₁ held₂ ceiling start width budget hw hh]

example : (2 : ℕ) + 3 ≤ 9 + 1 ∧
    (∀ n : ℕ, n ≤ 9 → (if n ≤ 9 then (1 : ℝ) else 0) = 1) ∧
    (∀ n : ℕ, n ≤ 9 → (if n ≤ 9 then (0 : ℝ) else 1) = 0) := by
  refine ⟨by norm_num, ?_, ?_⟩ <;> intro n hn <;> simp [hn]

end Transformer.Grokking.Operational
