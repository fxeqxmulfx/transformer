/-
# The printed propagation-of-chaos maximum estimate is false

ArXiv:2604.01978v1, §4, `prop: poc`, displays

  max_i E sup_[0,T] ‖x_i^n-x̄_i‖² + E sup_[0,T] W₂²(μ^n,μ̄^n)
    ≤ (C_T/n) Σ_i E ‖x_i^n(0)-x̄_i(0)‖².

Its count-independent prefactor cannot control the maximum by that
initial average for the source's unrestricted particle data. The source
imposes conditional independence and a common law on the copies;
it imposes no exchangeability assumption on the first particle block
or on its coupling with the copies. The source's `C_T`, as in the
claimed mean-field estimate, is fixed before the particle count.

At the admitted zero head law the original fields vanish. In dimension
two take every conditional copy at a fixed unit point, one original
particle at its antipode, and all remaining particles at the same point.
The copies are conditionally iid, and both systems solve their full
original joint martingale problem under the same noise. The largest
squared token error is four at every time, whereas the sum of all
initial squared errors is four. For n>C_T the right side is less
than four. The genuine squared Wasserstein error is nonnegative,
so adding it on the left preserves the contradiction.

Both time suprema are actual least upper bounds, and their constant
random variables are integrable. The counterexample uses original
sphere-valued trajectories and a head law satisfying
`ass:high_order_short`; no approximation to the dynamics is used.
The literal uniform maximum claim is refuted. No alternative averaged
or exchangeable estimate is asserted under its name.
-/

import Transformer.Homogenized.Section4_StaticPoC

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory

namespace Transformer.Homogenized

/-- For the actual one-discrepancy initial data, the maximum-versus-average
inequality holds exactly when its prefactor is at least the particle count.
This identifies the source obstruction before any evolution occurs.
Source: arXiv:2604.01978v1, §4, `prop: poc`, at time zero. -/
theorem singleTokenOutlier_initial_bound_iff (n : ℕ) (C : ℝ) (e : EucSpace 2) (he : ‖e‖ = 1) :
    (∀ i : Idx (n + 1), ‖singleTokenOutlier 0 e i - e‖ ^ 2 ≤
      C / ((n : ℝ) + 1) * ∑ j : Idx (n + 1), ‖singleTokenOutlier 0 e j - e‖ ^ 2) ↔
      (n : ℝ) + 1 ≤ C := by
  have hd : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  rw [sum_singleTokenOutlier_error 0 e he]
  constructor
  · intro h
    have hi := h 0
    rw [singleTokenOutlier_error 0 e he, ite_eq_left rfl] at hi
    have hr : 1 ≤ C / ((n : ℝ) + 1) := by linarith
    have hm := (le_div_iff₀ hd).mp hr
    simpa only [one_mul] using hm
  · intro h i
    have hr : 1 ≤ C / ((n : ℝ) + 1) := (le_div_iff₀ hd).mpr (by simpa only [one_mul] using h)
    rw [singleTokenOutlier_error 0 e he]
    split_ifs <;> linarith

/-- No count-independent prefactor bounds every initial token error by the initial average,
even for original unit configurations on the circle.
Source: arXiv:2604.01978v1, §4, the maximum and average in `prop: poc`. -/
theorem not_uniform_initial_max_average_bound :
    ¬ ∃ C : ℝ, ∀ n : ℕ, ∀ e : EucSpace 2, ‖e‖ = 1 → ∀ i : Idx (n + 1),
      ‖singleTokenOutlier 0 e i - e‖ ^ 2 ≤
        C / ((n : ℝ) + 1) * ∑ j : Idx (n + 1), ‖singleTokenOutlier 0 e j - e‖ ^ 2 := by
  rintro ⟨C, hC⟩
  obtain ⟨n, hn⟩ := exists_nat_gt C
  let e : EucSpace 2 := (basePoint 1 : SSphere 2)
  have he : ‖e‖ = 1 := by simp [e, basePoint]
  have hc := (singleTokenOutlier_initial_bound_iff n C e he).mp (hC n e he)
  linarith

/-- Every proposed count-independent prefactor fails for a genuine stationary original coupled system.
Source: arXiv:2604.01978v1, §4, `prop: poc`, its displayed maximum and initial average.
All conditional laws, common-noise joint tests, exact suprema, and integrability premises are retained. -/
theorem propagation_of_chaos_counterexample (C : ℝ) (e : EucSpace 2) (he : ‖e‖ = 1) :
    ∃ (n : ℕ) (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
      (X Xbar : ℝ → Unit → Idx (n + 1) → EucSpace 2)
      (μ : ℝ → Unit → Measure (EucSpace 2)) (S R : Unit → ℝ),
      IsPoCSystem 1 1 (Measure.dirac (0 : HeadParam 2)) (Measure.dirac ()) ℱ ℱ (Measure.dirac e) X Xbar μ ∧
      (∀ ω, IsLUB ((fun t => ‖X t ω 0 - Xbar t ω 0‖ ^ 2) '' Set.Icc 0 1) (S ω)) ∧
      (∀ ω, IsLUB ((fun t => Wasserstein.W2 (empMeasure (X t ω)) (empMeasure (Xbar t ω)) ^ 2) ''
        Set.Icc 0 1) (R ω)) ∧
      Integrable S (Measure.dirac ()) ∧ Integrable R (Measure.dirac ()) ∧
      C / ((n : ℝ) + 1) * ∑ j : Idx (n + 1), ∫ ω, ‖X 0 ω j - Xbar 0 ω j‖ ^ 2 ∂Measure.dirac () <
        (∫ ω, S ω ∂Measure.dirac ()) + ∫ ω, R ω ∂Measure.dirac () := by
  obtain ⟨n, hn⟩ := exists_singleTokenOutlier_max_average_gap C
  let A := singleTokenOutlier (0 : Idx (n + 1)) e
  let B : Idx (n + 1) → EucSpace 2 := fun _ => e
  let r := Wasserstein.W2 (empMeasure A) (empMeasure B) ^ 2
  let ℱ := Filtration.const ℝ (inferInstance : MeasurableSpace Unit) le_rfl
  refine ⟨n, ℱ, fun _ _ => A, fun _ _ => B, fun _ _ => Measure.dirac e,
    fun _ => 4, fun _ => r, ?_, ?_, ?_, integrable_const 4, integrable_const r, ?_⟩
  · exact isPoCSystem_static_dirac_zero 1 1 (Measure.dirac ()) ℱ A (singleTokenOutlier_unit 0 e he) e he
  · exact fun _ => singleTokenOutlier_error_sup 1 zero_le_one 0 e he
  · exact fun _ => static_empirical_W2_error_sup 1 zero_le_one A B
  · simpa only [integral_dirac, A, B, singleTokenOutlier_error 0 e he, ite_true] using
      hn e he r (sq_nonneg _)

/-- **The uniform maximum estimate of `prop: poc` is false.**
This is the negation of the carried original assertion, with its prefactor before the token count.
The source leaves the first particle block and the initial coupling unrestricted;
the actual stationary circle counterexample satisfies the admitted high-order law.
Source: arXiv:2604.01978v1, §4, `prop: poc`. -/
theorem not_propagation_of_chaos_printed :
    ¬ ∀ (d : ℕ) (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)),
      HasHighOrderLaw d σV σA ρ → ∀ T : ℝ, 0 < T →
      ∃ CT : ℝ, 0 < CT ∧
        ∀ (n : ℕ) (Ω : Type) (mΩ : MeasurableSpace Ω) (P : Measure Ω),
          IsProbabilityMeasure P →
        ∀ (ℱ 𝒢 : Filtration ℝ mΩ) (μ₀ : Measure (EucSpace d))
          (X Xbar : ℝ → Ω → Idx (n + 1) → EucSpace d)
          (μ : ℝ → Ω → Measure (EucSpace d)),
          IsPoCSystem β T ρ P ℱ 𝒢 μ₀ X Xbar μ →
        ∀ (i : Idx (n + 1)) (S R : Ω → ℝ),
          (∀ ω, IsLUB ((fun t => ‖X t ω i - Xbar t ω i‖ ^ 2) '' Set.Icc 0 T) (S ω)) →
          (∀ ω, IsLUB ((fun t => Wasserstein.W2 (empMeasure (X t ω)) (empMeasure (Xbar t ω)) ^ 2) ''
            Set.Icc 0 T) (R ω)) →
          Integrable S P → Integrable R P →
          (∫ ω, S ω ∂P) + ∫ ω, R ω ∂P ≤
            CT / ((n : ℝ) + 1) * ∑ j : Idx (n + 1), ∫ ω, ‖X 0 ω j - Xbar 0 ω j‖ ^ 2 ∂P := by
  intro h
  obtain ⟨C, _, hC⟩ := h 2 1 0 0 (Measure.dirac (0 : HeadParam 2)) (hasHighOrderLaw_dirac_zero 2) 1 one_pos
  let e : EucSpace 2 := (basePoint 1 : SSphere 2)
  have he : ‖e‖ = 1 := by simp [e, basePoint]
  obtain ⟨n, ℱ, X, Xbar, μ, S, R, hsys, hS, hR, hiS, hiR, hbad⟩ :=
    propagation_of_chaos_counterexample C e he
  have hbound := hC n Unit inferInstance (Measure.dirac ()) inferInstance ℱ ℱ (Measure.dirac e)
    X Xbar μ hsys 0 S R hS hR hiS hiR
  exact not_lt_of_ge hbound hbad

/-- The counterexample's nonzero unit point and positive horizon are genuine source data.
The complete original coupled-system witness with a nonzero discrepancy is in `Section4_StaticPoC`. -/
example : ∃ e : EucSpace 2, ‖e‖ = 1 ∧ (0 : ℝ) < 1 ∧
    HasHighOrderLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) :=
  ⟨(basePoint 1 : SSphere 2), by simp [basePoint], one_pos, hasHighOrderLaw_dirac_zero 2⟩

/-- The sharp count-dependent initial inequality is attained by the same nonzero source data.
Source: arXiv:2604.01978v1, §4, the initial comparison in `prop: poc`. -/
example : ∃ e : EucSpace 2, ‖e‖ = 1 ∧ ∀ i : Idx 3,
    ‖singleTokenOutlier 0 e i - e‖ ^ 2 ≤
      (3 : ℝ) / 3 * ∑ j : Idx 3, ‖singleTokenOutlier 0 e j - e‖ ^ 2 := by
  let e : EucSpace 2 := (basePoint 1 : SSphere 2)
  have he : ‖e‖ = 1 := by simp [e, basePoint]
  refine ⟨e, he, ?_⟩
  have h := (singleTokenOutlier_initial_bound_iff 2 3 e he).mpr (by norm_num)
  norm_num at h ⊢
  exact h

end Transformer.Homogenized
