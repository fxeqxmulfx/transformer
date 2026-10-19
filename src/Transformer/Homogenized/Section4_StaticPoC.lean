module

public import Transformer.Homogenized.CoupledSystem
public import Transformer.Homogenized.RandomChain
public import Transformer.Wasserstein.Basic

/-!
# Actual stationary coupled systems and a single original token discrepancy

ArXiv:2604.01978v1, §4, `prop: poc`, its displayed maximum
of token errors compared with the average initial squared error.
At the admitted zero head law every original projected noise field
vanishes. Consequently any unit first configuration and a constant
second configuration give an actual joint smooth-test martingale
solution driven by the same noise. The copies have the common
conditional Dirac law and are conditionally independent.

A single first-block token can start at the antipode of the common
copy location while all other tokens agree. Its squared error is four,
and the sum of all initial squared errors is also four, independently
of the number of tokens. These are actual original stationary systems,
including their complete conditional-law and joint-test conditions.
The time-supremum variables are their genuine constant values.
The displayed source proposition imposes conditional independence on
the copies, and no exchangeability assumption on the first block.
-/

@[expose] public section

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory

namespace Transformer.Homogenized

/-- Every original joint generator vanishes at the admitted zero head law.
Source: arXiv:2604.01978v1, §4, `prop: poc`, with the actual `eq:G_def` fields. -/
theorem pocGenerator_dirac_zero {d n : ℕ} (β : ℝ) (ν : Measure (EucSpace d))
    (φ : (Idx n → EucSpace d) × (Idx n → EucSpace d) → ℝ)
    (p : (Idx n → EucSpace d) × (Idx n → EucSpace d)) :
    pocGenerator β (Measure.dirac (0 : HeadParam d)) ν φ p = 0 := by
  simp only [pocGenerator, integral_dirac, GfieldOf_dirac_zero]
  have h0 : (fun _ : Idx n => (0 : EucSpace d)) = 0 := rfl
  rw [h0, pocHess_zero, mul_zero]

/-- Arbitrary stationary unit particles and constant conditional copies solve the actual original coupled problem.
Source: arXiv:2604.01978v1, §4, `prop: poc`, at the permitted zero head law. -/
theorem isPoCSystem_static_dirac_zero {d n : ℕ} (β T : ℝ)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P]
    (ℱ : Filtration ℝ m) (A : Idx n → EucSpace d) (hA : ∀ i, ‖A i‖ = 1)
    (e : EucSpace d) (he : ‖e‖ = 1) :
    IsPoCSystem β T (Measure.dirac (0 : HeadParam d)) P ℱ ℱ (Measure.dirac e)
      (fun _ _ => A) (fun _ _ _ => e) (fun _ _ => Measure.dirac e) := by
  have hcopies := isPoCSystem_dirac_zero (n := n) β T P ℱ e he
  refine ⟨fun _ _ => hA, fun _ _ => stronglyMeasurable_const,
    hcopies.copies, hcopies.condIid, ?_⟩
  intro φ hφ
  have hzero : pocMart β (Measure.dirac (0 : HeadParam d))
      (fun (_ : ℝ) (_ : Ω) => A) (fun (_ : ℝ) (_ : Ω) (_ : Idx n) => e)
      (fun (_ : ℝ) (_ : Ω) => Measure.dirac e) φ = fun _ _ => (0 : ℝ) := by
    funext t ω
    simp only [pocMart, pocGenerator_dirac_zero, intervalIntegral.integral_zero, sub_self]
  rw [hzero]
  exact isMartingaleOn_const T ℱ P 0

/-- The original unit configuration with one specified antipodal token.
Source: arXiv:2604.01978v1, §4, the unrestricted initial particle data of `prop: poc`. -/
noncomputable def singleTokenOutlier {d n : ℕ} (i : Idx (n + 1)) (e : EucSpace d) :
    Idx (n + 1) → EucSpace d := fun j => if j = i then -e else e

/-- The single-discrepancy configuration retains every original sphere constraint.
Source: arXiv:2604.01978v1, §4, `prop: poc`, its particle initial data. -/
theorem singleTokenOutlier_unit {d n : ℕ} (i : Idx (n + 1)) (e : EucSpace d)
    (he : ‖e‖ = 1) : ∀ j, ‖singleTokenOutlier i e j‖ = 1 := by
  intro j
  by_cases hji : j = i <;> simp [singleTokenOutlier, hji, he]

/-- One original token has squared error four and all others have squared error zero.
Source: arXiv:2604.01978v1, §4, the two sides of the displayed `prop: poc` estimate. -/
theorem singleTokenOutlier_error {d n : ℕ} (i : Idx (n + 1)) (e : EucSpace d)
    (he : ‖e‖ = 1) (j : Idx (n + 1)) :
    ‖singleTokenOutlier i e j - e‖ ^ 2 = if j = i then 4 else 0 := by
  have hdiff : -e - e = (-2 : ℝ) • e := by module
  by_cases hji : j = i
  · simp only [singleTokenOutlier, ite_eq_left hji, hdiff, norm_smul, he]
    norm_num
  · simp [singleTokenOutlier, hji]

/-- The original sum of initial errors is four for every positive particle count.
Source: arXiv:2604.01978v1, §4, the initial average in `prop: poc`. -/
theorem sum_singleTokenOutlier_error {d n : ℕ} (i : Idx (n + 1)) (e : EucSpace d)
    (he : ‖e‖ = 1) : ∑ j, ‖singleTokenOutlier i e j - e‖ ^ 2 = 4 := by
  simp only [singleTokenOutlier_error i e he]
  simp

/-- A genuinely constant error has its constant value as the exact time supremum on every nonnegative horizon.
Source: arXiv:2604.01978v1, §4, the time supremum in `prop: poc`. -/
theorem isLUB_constant_time_image (T c : ℝ) (hT : 0 ≤ T) :
    IsLUB ((fun _ : ℝ => c) '' Set.Icc (0 : ℝ) T) c := by
  constructor
  · rintro a ⟨t, ht, rfl⟩
    exact le_rfl
  · intro b hb
    exact hb ⟨0, ⟨le_rfl, hT⟩, rfl⟩

/-- The maximal original error of a single stationary discrepancy is four on every nonnegative horizon.
Source: arXiv:2604.01978v1, §4, the token error supremum of `prop: poc`. -/
theorem singleTokenOutlier_error_sup {d n : ℕ} (T : ℝ) (hT : 0 ≤ T)
    (i : Idx (n + 1)) (e : EucSpace d) (he : ‖e‖ = 1) :
    IsLUB ((fun _ : ℝ => ‖singleTokenOutlier i e i - e‖ ^ 2) '' Set.Icc (0 : ℝ) T) 4 := by
  simpa only [singleTokenOutlier_error i e he, ite_true] using isLUB_constant_time_image T 4 hT

/-- The actual Wasserstein error of two original stationary empirical laws has a genuine time supremum.
Source: arXiv:2604.01978v1, §4, the empirical-measure term of `prop: poc`. -/
theorem static_empirical_W2_error_sup {d n : ℕ} (T : ℝ) (hT : 0 ≤ T)
    (A B : Idx n → EucSpace d) :
    IsLUB ((fun _ : ℝ => Wasserstein.W2 (empMeasure A) (empMeasure B) ^ 2) '' Set.Icc (0 : ℝ) T)
      (Wasserstein.W2 (empMeasure A) (empMeasure B) ^ 2) :=
  isLUB_constant_time_image T _ hT

/-- For every proposed count-independent prefactor, one original discrepancy
exceeds the averaged initial-error bound, even after any nonnegative error
is added to the left side. The dimension is the genuine circle dimension.
Source: arXiv:2604.01978v1, §4, the displayed maximum and initial average in `prop: poc`. -/
theorem exists_singleTokenOutlier_max_average_gap (C : ℝ) :
    ∃ n : ℕ, ∀ e : EucSpace 2, ‖e‖ = 1 → ∀ R : ℝ, 0 ≤ R →
      C / ((n : ℝ) + 1) * ∑ j : Idx (n + 1), ‖singleTokenOutlier 0 e j - e‖ ^ 2 <
        ‖singleTokenOutlier (0 : Idx (n + 1)) e 0 - e‖ ^ 2 + R := by
  obtain ⟨n, hn⟩ := exists_nat_gt C
  have hd : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hr : C / ((n : ℝ) + 1) < 1 := by
    apply (div_lt_iff₀ hd).mpr
    linarith
  refine ⟨n, ?_⟩
  intro e he R hR
  rw [sum_singleTokenOutlier_error 0 e he, singleTokenOutlier_error 0 e he, ite_eq_left rfl]
  nlinarith

/-- The complete static source premises occur with a nonzero discrepancy on a genuine circle.
Source: arXiv:2604.01978v1, §4, `prop: poc`, including its full joint martingale problem. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (e : EucSpace 2) (A : Idx 3 → EucSpace 2),
    HasHighOrderLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) ∧
    ‖e‖ = 1 ∧ (∀ j, ‖A j‖ = 1) ∧ ‖A 0 - e‖ ^ 2 = 4 ∧
    (∑ j, ‖A j - e‖ ^ 2) = 4 ∧
    IsPoCSystem 1 1 (Measure.dirac (0 : HeadParam 2)) (Measure.dirac ()) ℱ ℱ
      (Measure.dirac e) (fun _ _ => A) (fun _ _ _ => e) (fun _ _ => Measure.dirac e) := by
  let ℱ := Filtration.const ℝ (inferInstance : MeasurableSpace Unit) le_rfl
  let e : EucSpace 2 := (basePoint 1 : SSphere 2)
  have he : ‖e‖ = 1 := by simp [e, basePoint]
  refine ⟨ℱ, e, singleTokenOutlier (0 : Idx 3) e, hasHighOrderLaw_dirac_zero 2, he,
    singleTokenOutlier_unit 0 e he, ?_, sum_singleTokenOutlier_error 0 e he,
    isPoCSystem_static_dirac_zero 1 1 (Measure.dirac ()) ℱ _ (singleTokenOutlier_unit 0 e he) e he⟩
  simpa only [ite_true] using singleTokenOutlier_error (0 : Idx 3) e he 0

end Transformer.Homogenized
