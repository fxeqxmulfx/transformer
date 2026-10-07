import Transformer.Grokking.Operational.Sustained

/-!
# When a memorization window proves an observed delay

Empirical source: Power et al., arXiv:2201.02177v1, sections 1 and 3.1,
compares first training fit and first validation-threshold attainment.
Operational source: lab.domain.phases and lab.domain.grokking at 43d4d66.

A memorization window alone excludes a crossing inside that window.
It does not place the first crossing after the window: the model may
already have succeeded earlier and then lost accuracy. An explicit
earlier below-threshold prefix eliminates that alternative. Combined
with the first train fit, the score bounds imply a lower bound on the
observed train-to-generalization delay without defining that order into
the memorization predicate.

The bounded recovery construction has an early first sustained success,
a later positive-width memorization window and successful recovery.
It formalizes why a recovery is not necessarily first grokking, relevant
to the seed 2 observations in geometry_all_results.json at df4debc.
These are trace laws, not an optimizer mechanism or a proof that a
transformer trajectory must exhibit any of the supplied windows.
-/

namespace Transformer.Grokking.Operational

/-- A first successful point cannot lie in a separated memorization
window. Source: fit/ceiling bounds at 43d4d66, adapting first-attainment
times of arXiv:2201.02177v1, section 3.1. Earlier success remains possible. -/
theorem firstCrossing_outside_memory_window (train held : ℕ → ℝ)
    (fit ceiling target : ℝ) (start width time : ℕ)
    (hm : MemorizationWindow train held fit ceiling start width)
    (hc : FirstCrossing held target time) (hs : ceiling < target) :
    time < start ∨ start + width ≤ time := by
  by_cases hearly : time < start
  · exact Or.inl hearly
  · right
    by_contra h
    have ho : time - start < width := by omega
    have hlow := hm.2.2 (time - start) ho
    have he : start + (time - start) = time := by omega
    rw [he] at hlow
    linarith [hc.1]

example : MemorizationWindow (fun _ => (1 : ℝ))
    (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 (1 / 10) 1 2 ∧
    FirstCrossing (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 3 ∧
    (1 / 10 : ℝ) < 1 := by
  refine ⟨⟨⟨by norm_num, ?_⟩, ⟨by norm_num, ?_⟩⟩,
    ⟨by norm_num, ?_⟩, by norm_num⟩
  · intro n hn
    norm_num
  · intro n hn
    have h : ¬3 ≤ 1 + n := by omega
    norm_num [h]
  · intro n hn
    have h : ¬3 ≤ n := by omega
    simp [h]

/-- An observed earlier failure prefix forces the crossing after the
entire memorization window. Source: arXiv:2201.02177v1, section 3.1,
first-attainment accounting; the prefix is additional measured input. -/
theorem firstCrossing_after_memory_window (train held : ℕ → ℝ)
    (fit ceiling target : ℝ) (start width time : ℕ)
    (hm : MemorizationWindow train held fit ceiling start width)
    (hc : FirstCrossing held target time)
    (hp : NoCrossingThrough held target start) (hs : ceiling < target) :
    start + width ≤ time := by
  have hlater := firstCrossing_after_censored_prefix held target start time hp hc
  obtain hearly | hlate := firstCrossing_outside_memory_window
    train held fit ceiling target start width time hm hc hs
  · omega
  · exact hlate

example : MemorizationWindow (fun _ => (1 : ℝ))
    (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 (1 / 10) 1 2 ∧
    FirstCrossing (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 3 ∧
    NoCrossingThrough (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 1 ∧
    (1 / 10 : ℝ) < 1 := by
  refine ⟨⟨⟨by norm_num, ?_⟩, ⟨by norm_num, ?_⟩⟩,
    ⟨by norm_num, ?_⟩, ?_, by norm_num⟩
  · intro n hn
    norm_num
  · intro n hn
    have h : ¬3 ≤ 1 + n := by omega
    norm_num [h]
  · intro n hn
    have h : ¬3 ≤ n := by omega
    simp [h]
  · intro n hn
    have h : ¬3 ≤ n := by omega
    simp [h]

/-- The actual score windows bound the delay from first train fit to
first held-out success. Source: the learning-time comparison of
arXiv:2201.02177v1, section 3.1, and the memorization diagnostic at
43d4d66. The bound counts observation intervals, not optimizer steps. -/
theorem train_to_generalization_delay_ge_memory_width (train held : ℕ → ℝ)
    (fit ceiling target : ℝ) (start width fitTime time : ℕ)
    (hm : MemorizationWindow train held fit ceiling start width)
    (hf : FirstCrossing train fit fitTime) (hc : FirstCrossing held target time)
    (hp : NoCrossingThrough held target start) (hs : ceiling < target) :
    width ≤ time - fitTime := by
  have hfit := hm.1.2 0 hm.1.1
  simp only [Nat.add_zero] at hfit
  have hfirst : fitTime ≤ start := by
    by_contra h
    have hlow := hf.2 start (by omega)
    linarith
  have hlate := firstCrossing_after_memory_window train held
    fit ceiling target start width time hm hc hp hs
  omega

example : MemorizationWindow (fun _ => (1 : ℝ))
    (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 (1 / 10) 1 2 ∧
    FirstCrossing (fun _ : ℕ => (1 : ℝ)) 1 0 ∧
    FirstCrossing (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 3 ∧
    NoCrossingThrough (fun n => if 3 ≤ n then (1 : ℝ) else 0) 1 1 ∧
    (1 / 10 : ℝ) < 1 := by
  refine ⟨⟨⟨by norm_num, ?_⟩, ⟨by norm_num, ?_⟩⟩,
    ⟨by norm_num, ?_⟩, ⟨by norm_num, ?_⟩, ?_, by norm_num⟩
  · intro n hn
    norm_num
  · intro n hn
    have h : ¬3 ≤ 1 + n := by omega
    norm_num [h]
  · intro n hn
    omega
  · intro n hn
    have h : ¬3 ≤ n := by omega
    simp [h]
  · intro n hn
    have h : ¬3 ≤ n := by omega
    simp [h]

/-- Identical train and held-out scores cannot supply a separated
memorization window. Source: the train/test-gap distinction in
arXiv:2201.02177v1, section 3.1; immediate fitting is not delayed learning. -/
theorem no_memory_window_for_identical_scores (score : ℕ → ℝ) (fit ceiling : ℝ)
    (start width : ℕ) (hs : ceiling < fit) :
    ¬MemorizationWindow score score fit ceiling start width := by
  intro hm
  have hfit := hm.1.2 0 hm.1.1
  have hlow := hm.2.2 0 hm.2.1
  linarith

example : (1 / 10 : ℝ) < 99 / 100 := by norm_num

/-- A later memorization window and recovery do not establish first
delayed generalization. Source: recovery observations for seed 2 at
df4debc and the first-attainment distinction of arXiv:2201.02177v1,
section 3.1. This valid binary-valued trace is a logical counterexample,
not an equation for the empirical seed 2 trajectory. -/
theorem recovery_window_is_not_first_success :
    ValidAccuracy (intermittentScore 2 6) ∧
      MemorizationWindow (fun _ => (1 : ℝ)) (intermittentScore 2 6) 1 (1 / 10) 2 4 ∧
        FirstSustained (intermittentScore 2 6) 1 2 0 ∧
          AboveWindow (intermittentScore 2 6) 1 6 2 := by
  refine ⟨?_, ⟨⟨by norm_num, ?_⟩, ⟨by norm_num, ?_⟩⟩,
    ⟨⟨by norm_num, ?_⟩, ?_⟩, ⟨by norm_num, ?_⟩⟩
  · intro n
    unfold intermittentScore
    split_ifs <;> norm_num
  · intro n hn
    norm_num
  · intro n hn
    have hi : ¬2 + n < 2 := by omega
    have hr : ¬6 ≤ 2 + n := by omega
    norm_num [intermittentScore, hi, hr]
  · intro n hn
    norm_num [intermittentScore, hn]
  · intro n hn
    omega
  · intro n hn
    have h : 6 ≤ 6 + n := by omega
    norm_num [intermittentScore, h]

end Transformer.Grokking.Operational
