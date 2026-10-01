import Transformer.DoubleDescent.Section2_Risk
import Mathlib.Data.ENat.Lattice

/-!
# Effective model complexity

arXiv:1912.02292v1, Section 2, Definition 1. The source uses a maximum
without assuming it exists. The extended-natural supremum below agrees with
every existing maximum, gives zero for an empty feasible set, and gives
infinity for unbounded feasible sample sizes. No monotonicity in sample size
is assumed: the definition does not imply that the feasible sizes form an interval.
-/

namespace Transformer.DoubleDescent

open scoped ENNReal

/-- Section 2, Definition 1, corrected to a supremum in `ℕ∞`: largest
approximately interpolatable sample size, or infinity if unbounded. -/
noncomputable def effectiveComplexity (risk : ℕ → ℝ≥0∞) (ε : ℝ≥0∞) : ℕ∞ :=
  ⨆ n : ℕ, ⨆ (_ : risk n ≤ ε), (n : ℕ∞)

/-- Section 2, Definition 1: EMC of an actual procedure and data distribution. -/
noncomputable def EMC {Z M : Type*} [MeasurableSpace Z]
    (D : MeasureTheory.Measure Z) (loss : M → Z → ℝ≥0∞)
    (train : TrainingProcedure Z M) (ε : ℝ≥0∞) : ℕ∞ :=
  effectiveComplexity (expectedTrainingRisk D loss train) ε

/-- Section 2, Definition 1: every fitted sample size is at most the EMC. -/
theorem le_effectiveComplexity (risk : ℕ → ℝ≥0∞) (ε : ℝ≥0∞)
    {n : ℕ} (h : risk n ≤ ε) : (n : ℕ∞) ≤ effectiveComplexity risk ε := by
  exact le_iSup_of_le n (le_iSup_of_le h le_rfl)

/-- Section 2: the feasibility premise is realized by zero training risk. -/
example (ε : ℝ≥0∞) (n : ℕ) : (fun _ : ℕ => (0 : ℝ≥0∞)) n ≤ ε := bot_le

/-- Section 2, Definition 1: an upper bound on all fitted sizes bounds EMC. -/
theorem effectiveComplexity_le (risk : ℕ → ℝ≥0∞) (ε : ℝ≥0∞) (N : ℕ)
    (h : ∀ n, risk n ≤ ε → n ≤ N) : effectiveComplexity risk ε ≤ (N : ℕ∞) := by
  refine iSup_le fun n => iSup_le fun hn => ?_
  exact ENat.natCast_le_natCast.mpr (h n hn)

/-- Section 2: bounded feasibility is satisfiable by a threshold risk profile. -/
example (N : ℕ) : ∀ n, (if n ≤ N then (0 : ℝ≥0∞) else 1) ≤ 0 → n ≤ N := by
  intro n h
  by_contra hn
  simp [hn] at h

/-- Section 2, Definition 1: the corrected supremum equals the paper's maximum
whenever a greatest feasible size exists. -/
theorem effectiveComplexity_eq_max (risk : ℕ → ℝ≥0∞) (ε : ℝ≥0∞) (N : ℕ)
    (hfit : risk N ≤ ε) (hmax : ∀ n, risk n ≤ ε → n ≤ N) :
    effectiveComplexity risk ε = (N : ℕ∞) := by
  exact le_antisymm (effectiveComplexity_le risk ε N hmax)
    (le_effectiveComplexity risk ε hfit)

/-- Section 2: both maximum hypotheses hold for a finite threshold profile. -/
example (N : ℕ) :
    (if N ≤ N then (0 : ℝ≥0∞) else 1) ≤ 0 ∧
      ∀ n, (if n ≤ N then (0 : ℝ≥0∞) else 1) ≤ 0 → n ≤ N := by
  constructor
  · simp
  · intro n h
    by_contra hn
    simp [hn] at h

/-- Section 2, Definition 1: relaxing the error tolerance cannot reduce EMC. -/
theorem effectiveComplexity_mono_tolerance (risk : ℕ → ℝ≥0∞)
    {ε δ : ℝ≥0∞} (hεδ : ε ≤ δ) :
    effectiveComplexity risk ε ≤ effectiveComplexity risk δ := by
  refine iSup_le fun n => iSup_le fun hn => ?_
  exact le_effectiveComplexity risk δ (hn.trans hεδ)

/-- Section 2: ordered positive tolerances exist. -/
example : (0 : ℝ≥0∞) < 1 ∧ (1 : ℝ≥0∞) ≤ 2 := by norm_num

/-- Sections 1, 2, and 6: reducing training risk at every sample size increases
EMC. The source's assertion about longer training needs this extra premise;
arbitrary optimization trajectories need not satisfy it. -/
theorem effectiveComplexity_antitone_risk (risk₁ risk₂ : ℕ → ℝ≥0∞)
    (ε : ℝ≥0∞) (h : ∀ n, risk₁ n ≤ risk₂ n) :
    effectiveComplexity risk₂ ε ≤ effectiveComplexity risk₁ ε := by
  refine iSup_le fun n => iSup_le fun hn => ?_
  exact le_effectiveComplexity risk₁ ε ((h n).trans hn)

/-- Sections 1 and 6: the risk improvement premise is satisfiable by equal risks. -/
example (risk : ℕ → ℝ≥0∞) : ∀ n, risk n ≤ risk n := fun _ => le_rfl

/-- Sections 1, 2, and 6: actual samplewise training-loss improvement implies
EMC improvement, after averaging over the iid sample. -/
theorem EMC_mono_of_training_improvement {Z M : Type*} [MeasurableSpace Z]
    (D : MeasureTheory.Measure Z) (loss : M → Z → ℝ≥0∞)
    (train₁ train₂ : TrainingProcedure Z M) (ε : ℝ≥0∞)
    (h : ∀ n sample, empiricalRisk loss (train₁ n sample) sample ≤
      empiricalRisk loss (train₂ n sample) sample) :
    EMC D loss train₂ ε ≤ EMC D loss train₁ ε := by
  exact effectiveComplexity_antitone_risk _ _ ε
    (expectedTrainingRisk_le D loss train₁ train₂ h)

/-- Section 2: samplewise training improvement is satisfiable by equal procedures. -/
example {Z M : Type*} (loss : M → Z → ℝ≥0∞) (train : TrainingProcedure Z M) :
    ∀ n sample, empiricalRisk loss (train n sample) sample ≤
      empiricalRisk loss (train n sample) sample := fun _ _ => le_rfl

/-- Section 2, Definition 1: zero risk at every size gives infinite EMC.
This is precisely the case omitted by the source's use of `max`. -/
theorem effectiveComplexity_zero (ε : ℝ≥0∞) :
    effectiveComplexity (fun _ => 0) ε = ⊤ := by
  apply top_unique
  rw [← ENat.iSup_natCast]
  exact iSup_le fun n => le_effectiveComplexity (fun _ => 0) ε (bot_le : 0 ≤ ε)

/-- Section 2, Definition 1: all sample sizes are feasible for zero risk,
but none is greatest. This refutes the unqualified existence of the maximum. -/
theorem zero_risk_has_no_maximum (ε : ℝ≥0∞) :
    ¬ ∃ N : ℕ, ∀ n : ℕ, (0 : ℝ≥0∞) ≤ ε → n ≤ N := by
  rintro ⟨N, hN⟩
  exact Nat.not_succ_le_self N (hN (N + 1) bot_le)

/-- Section 2, Definition 1: a genuinely perfectly fitted procedure realizes
the infinite case, rather than just an abstract numerical profile. -/
theorem EMC_eq_top_of_perfect_training {Z M : Type*} [MeasurableSpace Z]
    (D : MeasureTheory.Measure Z) (loss : M → Z → ℝ≥0∞)
    (train : TrainingProcedure Z M) (ε : ℝ≥0∞)
    (h : ∀ n sample i, loss (train n sample) (sample i) = 0) :
    EMC D loss train ε = ⊤ := by
  have hr : expectedTrainingRisk D loss train = fun _ => 0 := by
    funext n
    exact expectedTrainingRisk_zero D loss train h n
  rw [EMC, hr]
  exact effectiveComplexity_zero ε

/-- Section 2: singleton-label classification realizes the infinite-EMC premise. -/
example {Z : Type*} (train : TrainingProcedure Z (Z → Unit)) :
    ∀ n sample i, (if train n sample (sample i) = () then (0 : ℝ≥0∞) else 1) = 0 := by
  intro n sample i
  simp

end Transformer.DoubleDescent
