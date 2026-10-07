import Transformer.Grokking.Operational.Windows
import Transformer.Grokking.Operational.Censoring

/-!
# First sustained success and confirmation

Empirical source: Power et al., arXiv:2201.02177v1, section 3.1, reports
first validation-threshold attainment. Operational source:
lab.domain.phases.sustained at 43d4d66 instead asks for consecutive
observations. This module formalizes that distinct study choice.

The predicate computes no future expectation. To confirm an onset, every
point in its positive-width window must already be observed. The budget
must cover onset plus patience, rather than only the onset.
Earlier windows also fit inside that confirmation prefix.

A valid bounded trace shows why a first threshold crossing and a first
sustained window can have different onsets: one isolated early success
fails the patience test, whereas the later successful run passes it.
The trace is a model-free counterexample; it is not an asserted AdamW
trajectory. The Python finite-list scan and its timestamps remain an
additional implementation bridge to these observation-indexed predicates.
-/

namespace Transformer.Grokking.Operational

/-- First positive-width window of consecutive successful observations.
Source: phases.sustained at 43d4d66; patience is an observation count,
not an optimizer-step duration or an asymptotic stability guarantee. -/
def FirstSustained (score : ℕ → ℝ) (target : ℝ) (patience onset : ℕ) : Prop :=
  AboveWindow score target onset patience ∧
    ∀ earlier, earlier < onset → ¬AboveWindow score target earlier patience

/-- A step-shaped score's first sustained onset is its actual rise.
Source: a bounded observation-model specialization of the first-attainment
notion in arXiv:2201.02177v1, section 3.1; no optimizer law is claimed. -/
theorem firstSustained_rise (target : ℝ) (patience onset : ℕ)
    (ht : 0 < target) (hu : target ≤ 1) (hp : 0 < patience) :
    FirstSustained (fun n => if onset ≤ n then (1 : ℝ) else 0)
      target patience onset := by
  refine ⟨⟨hp, ?_⟩, ?_⟩
  · intro n hn
    have h : onset ≤ onset + n := by omega
    simpa only [ite_eq_left h] using hu
  · intro n hn hwindow
    have hhigh := hwindow.2 0 hp
    have h : ¬onset ≤ n := by omega
    simp only [Nat.add_zero, ite_eq_right h] at hhigh
    linarith

example : 0 < (99 / 100 : ℝ) ∧ (99 / 100 : ℝ) ≤ 1 ∧ 0 < (5 : ℕ) := by
  norm_num

/-- The first sustained onset is unique for a fixed patience and target.
Source: the first-window scan in phases.sustained at 43d4d66. This
does not imply all later windows remain successful. -/
theorem firstSustained_unique (score : ℕ → ℝ) (target : ℝ) (patience s t : ℕ)
    (hs : FirstSustained score target patience s)
    (ht : FirstSustained score target patience t) : s = t := by
  rcases Nat.lt_trichotomy s t with hst | heq | hts
  · exact False.elim (ht.2 s hst hs.1)
  · exact heq
  · exact False.elim (hs.2 t hts ht.1)

example : FirstSustained (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 2 3 ∧
    FirstSustained (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 2 3 := by
  have h := firstSustained_rise 1 2 3 (by norm_num) (by norm_num) (by norm_num)
  exact ⟨h, h⟩

/-- The confirmation prefix fixes first sustained success. Source:
causal observation histories in lab.domain.grokking and the first-window
scan in phases.sustained at 43d4d66. Future extensions are arbitrary. -/
theorem firstSustained_prefix_invariant (f g : ℕ → ℝ) (target : ℝ)
    (patience onset budget : ℕ) (hw : onset + patience ≤ budget + 1)
    (he : ∀ n, n ≤ budget → f n = g n) :
    FirstSustained f target patience onset ↔ FirstSustained g target patience onset := by
  have hon := above_window_prefix_invariant f g target onset patience budget hw he
  constructor
  · intro h
    refine ⟨hon.mp h.1, ?_⟩
    intro n hn hg
    have hp := above_window_prefix_invariant f g target n patience budget (by omega) he
    exact h.2 n hn (hp.mpr hg)
  · intro h
    refine ⟨hon.mpr h.1, ?_⟩
    intro n hn hf
    have hp := above_window_prefix_invariant f g target n patience budget (by omega) he
    exact h.2 n hn (hp.mp hf)

example : (3 : ℕ) + 2 ≤ 9 + 1 ∧ ∀ n : ℕ, n ≤ 9 →
    (if n ≤ 9 then (1 : ℝ) else 0) = 1 := by
  refine ⟨by norm_num, ?_⟩
  intro n hn
  simp [hn]

/-- A first isolated crossing is no later than the first sustained onset.
Source: the distinction between arXiv:2201.02177v1, section 3.1, and
phases.sustained at 43d4d66. Equality needs additional score information. -/
theorem firstCrossing_le_firstSustained (score : ℕ → ℝ) (target : ℝ)
    (crossing patience onset : ℕ) (hc : FirstCrossing score target crossing)
    (hs : FirstSustained score target patience onset) : crossing ≤ onset := by
  have hhigh := hs.1.2 0 hs.1.1
  simp only [Nat.add_zero] at hhigh
  by_contra h
  have hlow := hc.2 onset (by omega)
  linarith

example : FirstCrossing (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 3 ∧
    FirstSustained (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 2 3 := by
  refine ⟨⟨by norm_num, ?_⟩,
    firstSustained_rise 1 2 3 (by norm_num) (by norm_num) (by norm_num)⟩
  intro n hn
  have h : ¬3 ≤ n := by omega
  simp [h]

/-- An explicitly constructed bounded trace with early success and a
later recovery. Source: model-free counterexamples to interpreting the
learning-time statistic of arXiv:2201.02177v1, section 3.1, as monotonicity.
The initial successful count and the recovery time are explicit inputs. -/
def intermittentScore (initialCount resumedAt time : ℕ) : ℝ :=
  if time < initialCount ∨ resumedAt ≤ time then 1 else 0

/-- One early successful point and a later sustained run have different
first onsets. Source: the patience distinction in phases.sustained at
43d4d66, compared with arXiv:2201.02177v1, section 3.1. The constructed
scores stay in the accuracy range; no neural-network dynamics are assumed. -/
theorem isolated_crossing_precedes_sustained :
    ValidAccuracy (intermittentScore 1 4) ∧
      FirstCrossing (intermittentScore 1 4) 1 0 ∧
        FirstSustained (intermittentScore 1 4) 1 2 4 := by
  refine ⟨?_, ⟨by norm_num [intermittentScore], ?_⟩, ⟨⟨by norm_num, ?_⟩, ?_⟩⟩
  · intro n
    unfold intermittentScore
    split_ifs <;> norm_num
  · intro n hn
    omega
  · intro n hn
    have h : 4 ≤ 4 + n := by omega
    norm_num [intermittentScore, h]
  · intro n hn hwindow
    by_cases hz : n = 0
    · subst n
      have hhigh := hwindow.2 1 (by norm_num)
      norm_num [intermittentScore] at hhigh
    · have hhigh := hwindow.2 0 hwindow.1
      have hi : ¬n < 1 := by omega
      have hr : ¬4 ≤ n := by omega
      norm_num [intermittentScore, hi, hr] at hhigh

end Transformer.Grokking.Operational
