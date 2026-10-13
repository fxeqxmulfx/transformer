import Transformer.Grokking.Operational.Basic

/-!
# What a finite accuracy history does not determine

Empirical source: Power et al., arXiv:2201.02177v1, section 3.1, including
finite optimization budgets and first target-attainment times. Operational
source: lab.domain.grokking at 43d4d66, which reports current evidence and
explicitly supplies no guarantee of future success.

For any bounded accuracy prefix below a positive target at most one,
construct two bounded continuations: one first reaches the target at the
next observation, and the other stays below it forever. Every function
of the same recorded prefix has the same output on both continuations.
Thus no finite accuracy-only observer can be sound and complete for
eventual crossing over the unrestricted class of bounded accuracy traces.

No optimizer equation or fixed deterministic training state is imposed on
these traces. A dynamical prior or the full AdamW state can restrict the
class; this theorem does not establish an impossibility for such inputs.

The success continuation may jump immediately after the available prefix.
This is permitted by the unrestricted observation model, and is not an
assertion about continuity or the update sizes of a neural network.
Consequently the conclusion limits exact model-free prediction; it does
not rule out probabilistic forecasts with stated dynamical assumptions.
-/

namespace Transformer.Grokking.Operational

/-- Valid range of an accuracy trace. Source: accuracy checks in
lab.domain.grokking.progress at 43d4d66; no learning law is included. -/
def ValidAccuracy (score : ℕ → ℝ) : Prop := ∀ n, 0 ≤ score n ∧ score n ≤ 1

/-- Keep the available prefix and choose a constant later continuation.
Source: finite-budget accounting of arXiv:2201.02177v1, section 3.1;
this is a counterexample construction, not a purported AdamW trajectory. -/
def extendAt (past : ℕ → ℝ) (budget : ℕ) (later : ℝ) (time : ℕ) : ℝ :=
  if time ≤ budget then past time else later

/-- A bounded prefix and bounded tail give a valid complete trace.
Source: the finite-history counterexample above, respecting the actual
accuracy range rather than using an invalid negative failure tail. -/
theorem extendAt_valid (past : ℕ → ℝ) (budget : ℕ) (later : ℝ)
    (hp : ∀ n, n ≤ budget → 0 ≤ past n ∧ past n ≤ 1) (hl : 0 ≤ later ∧ later ≤ 1) :
    ValidAccuracy (extendAt past budget later) := by
  intro n
  by_cases hn : n ≤ budget
  · simpa only [extendAt, ite_eq_left hn] using hp n hn
  · simpa only [extendAt, ite_eq_right hn] using hl

example : (∀ n : ℕ, n ≤ 10 → 0 ≤ (1 / 100 : ℝ) ∧ (1 / 100 : ℝ) ≤ 1) ∧
    0 ≤ (1 : ℝ) ∧ (1 : ℝ) ≤ 1 := by norm_num

/-- The observable finite prefix is independent of the chosen tail.
Source: the causal-prefix requirement of lab.domain.grokking at 43d4d66. -/
theorem extension_preserves_scorePrefix (past : ℕ → ℝ) (budget : ℕ) (later : ℝ) :
    scorePrefix (extendAt past budget later) budget = scorePrefix past budget := by
  apply scorePrefix_eq
  intro n hn
  simp only [extendAt, ite_eq_left hn]

/-- One admissible continuation first succeeds at the next observation.
Source: finite-budget censoring in arXiv:2201.02177v1, section 3.1;
earlier failure and the upper bound on the target are explicit premises. -/
theorem extension_first_crossing (past : ℕ → ℝ) (target : ℝ) (budget : ℕ)
    (hc : NoCrossingThrough past target budget) (ht : target ≤ 1) :
    FirstCrossing (extendAt past budget 1) target (budget + 1) := by
  constructor
  · have hn : ¬budget + 1 ≤ budget := by omega
    simpa only [extendAt, ite_eq_right hn] using ht
  · intro n hn
    have hb : n ≤ budget := by omega
    simpa only [extendAt, ite_eq_left hb] using hc n hb

example : NoCrossingThrough (fun _ : ℕ => (1 / 100 : ℝ)) (99 / 100) 10 ∧
    (99 / 100 : ℝ) ≤ 1 := by
  constructor
  · intro n hn
    norm_num
  · norm_num

/-- Another admissible continuation never attains the target. Source:
finite-budget censoring in arXiv:2201.02177v1, section 3.1; positivity
of the target makes a zero-accuracy tail a genuine failure. -/
theorem extension_never_crosses (past : ℕ → ℝ) (target : ℝ) (budget : ℕ)
    (hc : NoCrossingThrough past target budget) (ht : 0 < target) :
    ∀ n, extendAt past budget 0 n < target := by
  intro n
  by_cases hn : n ≤ budget
  · simpa only [extendAt, ite_eq_left hn] using hc n hn
  · simpa only [extendAt, ite_eq_right hn] using ht

example : NoCrossingThrough (fun _ : ℕ => (1 / 100 : ℝ)) (99 / 100) 10 ∧
    0 < (99 / 100 : ℝ) := by
  constructor
  · intro n hn
    norm_num
  · norm_num

/-- Every valid censored prefix has bounded success and failure
continuations with exactly the same recorded values. Source: the
model-free finite-budget interpretation of arXiv:2201.02177v1, section 3.1. -/
theorem finite_prefix_future_ambiguity (past : ℕ → ℝ) (target : ℝ) (budget : ℕ)
    (hp : ∀ n, n ≤ budget → 0 ≤ past n ∧ past n ≤ 1)
    (hc : NoCrossingThrough past target budget) (ht : 0 < target ∧ target ≤ 1) :
    ∃ good bad : ℕ → ℝ, ValidAccuracy good ∧ ValidAccuracy bad ∧
      scorePrefix good budget = scorePrefix past budget ∧
      scorePrefix bad budget = scorePrefix past budget ∧
      FirstCrossing good target (budget + 1) ∧ ∀ n, bad n < target := by
  refine ⟨extendAt past budget 1, extendAt past budget 0,
    extendAt_valid past budget 1 hp (by norm_num),
    extendAt_valid past budget 0 hp (by norm_num),
    extension_preserves_scorePrefix past budget 1,
    extension_preserves_scorePrefix past budget 0,
    extension_first_crossing past target budget hc ht.2,
    extension_never_crosses past target budget hc ht.1⟩

example : (∀ n : ℕ, n ≤ 10 → 0 ≤ (1 / 100 : ℝ) ∧ (1 / 100 : ℝ) ≤ 1) ∧
    NoCrossingThrough (fun _ : ℕ => (1 / 100 : ℝ)) (99 / 100) 10 ∧
    0 < (99 / 100 : ℝ) ∧ (99 / 100 : ℝ) ≤ 1 := by
  refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
  intro n hn
  norm_num

/-- A finite accuracy-only observer cannot exactly decide eventual
crossing on every valid trace. Source: the no-future-guarantee scope of
lab.domain.grokking at 43d4d66; a dynamical restriction would be additional
input. The observer is arbitrary, not defined to satisfy the conclusion. -/
theorem no_universal_finite_crossing_detector (observer : List ℝ → Bool)
    (target : ℝ) (budget : ℕ) (ht : 0 < target ∧ target ≤ 1) :
    ¬(∀ score : ℕ → ℝ, ValidAccuracy score →
      (observer (scorePrefix score budget) = true ↔ ∃ time, FirstCrossing score target time)) := by
  intro hall
  have hp : ∀ n : ℕ, n ≤ budget → 0 ≤ (0 : ℝ) ∧ (0 : ℝ) ≤ 1 := by norm_num
  have hc : NoCrossingThrough (fun _ : ℕ => (0 : ℝ)) target budget := by
    intro n hn
    exact ht.1
  obtain ⟨good, bad, hgood, hbad, hpg, hpb, hfirst, hnever⟩ :=
    finite_prefix_future_ambiguity (fun _ => 0) target budget hp hc ht
  have hyes := (hall good hgood).2 ⟨budget + 1, hfirst⟩
  have heq : scorePrefix good budget = scorePrefix bad budget := by rw [hpg, hpb]
  rw [heq] at hyes
  obtain ⟨time, htime⟩ := (hall bad hbad).1 hyes
  have hlow := hnever time
  linarith [htime.1]

example : 0 < (99 / 100 : ℝ) ∧ (99 / 100 : ℝ) ≤ 1 := by norm_num

end Transformer.Grokking.Operational
