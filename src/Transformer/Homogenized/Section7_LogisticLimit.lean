import Transformer.Homogenized.Section7_LogisticBracketLimit

/-!
# The original logistic limit and its exact endpoint probabilities

ArXiv:2604.01978v1, §7, `sec: sde.logistic`.
The original martingale problem gives the full real-time angle limit,
with the deterministic initial angle as its mean. The actual bracket
budget excludes interior state limits. The limiting angle therefore
takes precisely the two values ±π/2 almost surely. Integrating its
actual two-value representation determines the endpoint probabilities.

The theorem retains the original probability law, filtration, equation,
deterministic initial value, full real-time convergence, and both exact
probability formulas. No convergence, endpoint support, bracket bound,
or probability formula is an input. The source's printed sin substitution
is corrected to arcsin, and its bracket-to-zero sentence is corrected to
divergence in Section7_LogisticBracketLimit. At the source's initial
value u₀=0, both formulas give probability 1/2 as claimed there.
-/

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- The actual mean of a two-value random variable determines its mass at either value.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the endpoint probability computation.
Measurability of the endpoint event and integrability of the indicator representation are proved. -/
theorem probability_eq_of_two_values
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (Y : Ω → ℝ) (hY : StronglyMeasurable Y) (a b : ℝ) (hab : a ≠ b)
    (hv : ∀ᵐ ω ∂P, Y ω = a ∨ Y ω = b) :
    (P {ω | Y ω = a}).toReal = ((∫ ω, Y ω ∂P) - b) / (a - b) := by
  classical
  let A : Set Ω := {ω | Y ω = a}
  have hA : MeasurableSet A := measurableSet_eq_fun hY.measurable measurable_const
  have hi : Integrable (A.indicator (fun _ : Ω => (1 : ℝ))) P :=
    (integrable_const (1 : ℝ)).indicator hA
  have heq : Y =ᵐ[P] (fun ω => b + (a - b) * A.indicator (fun _ => (1 : ℝ)) ω) := by
    filter_upwards [hv] with ω hω
    rcases hω with h | h
    · simp only [A, Set.indicator_of_mem (show ω ∈ {ω | Y ω = a} from h)]
      linarith
    · have hnot : ω ∉ A := by change Y ω ≠ a; rw [h]; exact hab.symm
      rw [Set.indicator_of_notMem hnot, mul_zero, add_zero]
      exact h
  have hmean : (∫ ω, Y ω ∂P) = b + (a - b) * P.real A := by
    rw [integral_congr_ae heq,
      integral_add (integrable_const b) (hi.const_mul (a - b)), integral_const,
      integral_const_mul, integral_indicator_const 1 hA, probReal_univ, one_smul, smul_eq_mul, mul_one]
  change P.real A = _
  apply (eq_div_iff (sub_ne_zero.mpr hab)).mpr
  nlinarith

/-- **The limit of the original logistic equation.** From a deterministic
u₀ in [-1,1], the process converges almost surely at all real times to ±1,
with P(u∞=1)=arcsin(u₀)/π+1/2 and P(u∞=-1)=1/2-arcsin(u₀)/π.
This is the source's angle formula after the corrected substitution θ=arcsin(u).
All original hypotheses and both probability constants are retained.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, following `eq: sde.logistic`. -/
theorem logistic_limit {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    (hP : IsProbabilityMeasure P) (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ)
    (hu : IsLogisticSolution P ℱ u) (u₀ : ℝ) (hu₀ : u₀ ∈ Icc (-1 : ℝ) 1)
    (hinit : ∀ ω, u 0 ω = u₀) :
    ∃ uinf : Ω → ℝ,
      (∀ᵐ ω ∂P, Tendsto (fun t => u t ω) atTop (𝓝 (uinf ω))) ∧
        (∀ᵐ ω ∂P, uinf ω = 1 ∨ uinf ω = -1) ∧
          (P {ω | uinf ω = 1}).toReal = Real.arcsin u₀ / Real.pi + 1 / 2 ∧
            (P {ω | uinf ω = -1}).toReal = 1 / 2 - Real.arcsin u₀ / Real.pi := by
  have : IsProbabilityMeasure P := hP
  have hr (ω : Ω) : u 0 ω ∈ Icc (-1 : ℝ) 1 := by rw [hinit ω]; exact hu₀
  obtain ⟨L, hm, hi, hl, hb, hmean⟩ := logistic_angle_limit P ℱ u hu u₀ hu₀ hinit
  have hlim := logistic_state_limit_from_angle P ℱ u hu hr L hl
  have hends := logistic_limit_is_endpoint P ℱ u hu hr (fun ω => Real.sin (L ω)) hlim
  have hangle : ∀ᵐ ω ∂P, L ω = Real.pi / 2 ∨ L ω = -(Real.pi / 2) := by
    filter_upwards [hb, hends] with ω hω hend
    have hid := Real.arcsin_sin (abs_le.mp hω).1 (abs_le.mp hω).2
    rcases hend with h | h
    · rw [h, Real.arcsin_one] at hid
      exact Or.inl hid.symm
    · rw [h, Real.arcsin_neg_one] at hid
      exact Or.inr hid.symm
  have hne : Real.pi / 2 ≠ -(Real.pi / 2) := by linarith [Real.pi_pos]
  have hplus := probability_eq_of_two_values P L hm (Real.pi / 2) (-(Real.pi / 2)) hne hangle
  have hminus := probability_eq_of_two_values P L hm (-(Real.pi / 2)) (Real.pi / 2) hne.symm
    (hangle.mono (fun ω hω => hω.symm))
  rw [hmean] at hplus hminus
  have hposEvent : {ω | Real.sin (L ω) = 1} =ᵐ[P] {ω | L ω = Real.pi / 2} := by
    filter_upwards [hangle] with ω hω
    apply propext
    rcases hω with h | h
    · simp [h]
    · rw [h, Real.sin_neg, Real.sin_pi_div_two]
      constructor
      · norm_num
      · intro hbad
        exact (hne hbad.symm).elim
  have hnegEvent : {ω | Real.sin (L ω) = -1} =ᵐ[P] {ω | L ω = -(Real.pi / 2)} := by
    filter_upwards [hangle] with ω hω
    apply propext
    rcases hω with h | h
    · rw [h, Real.sin_pi_div_two]
      constructor
      · norm_num
      · intro hbad
        exact (hne hbad).elim
    · simp [h, Real.sin_neg]
  refine ⟨fun ω => Real.sin (L ω), hlim, hends, ?_, ?_⟩
  · rw [measure_congr hposEvent, hplus]
    field_simp [Real.pi_ne_zero]
    ring
  · rw [measure_congr hnegEvent, hminus]
    field_simp [Real.pi_ne_zero]
    ring

/-- A genuine Gaussian-coordinate binary observable takes both distinct values
and satisfies every measurable two-value probability premise. -/
example : ∃ Y : ℝ → ℝ, StronglyMeasurable Y ∧ (1 : ℝ) ≠ -1 ∧
    (∀ᵐ ω ∂ProbabilityTheory.gaussianReal 0 1, Y ω = 1 ∨ Y ω = -1) ∧
      Y (-1) = 1 ∧ Y 1 = -1 := by
  classical
  let Y : ℝ → ℝ := (Iic 0).piecewise (fun _ => 1) (fun _ => -1)
  refine ⟨Y, StronglyMeasurable.piecewise measurableSet_Iic stronglyMeasurable_const
    stronglyMeasurable_const, by norm_num, ?_, by norm_num [Y, Set.piecewise],
    by norm_num [Y, Set.piecewise]⟩
  apply ae_of_all
  intro ω
  by_cases hω : ω ≤ 0
  · exact Or.inl (by simp [Y, hω])
  · exact Or.inr (by simp [Y, hω])

/-- The genuine positive absorbing solution realizes every original equation
and deterministic initial-value hypothesis, with exact endpoint masses 1 and 0. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (u : ℝ → Unit → ℝ),
    IsLogisticSolution (Measure.dirac ()) ℱ u ∧
      (1 : ℝ) ∈ Icc (-1 : ℝ) 1 ∧ (∀ ω, u 0 ω = 1) ∧
      ∃ uinf : Unit → ℝ,
        (∀ᵐ ω ∂Measure.dirac (), Tendsto (fun t => u t ω) atTop (𝓝 (uinf ω))) ∧
        ((Measure.dirac ()) {ω | uinf ω = 1}).toReal = 1 ∧
        ((Measure.dirac ()) {ω | uinf ω = -1}).toReal = 0 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let u : ℝ → Unit → ℝ := fun _ _ => 1
  have hu := isLogisticSolution_one (Measure.dirac ()) ℱ
  have hinit (ω : Unit) : u 0 ω = 1 := rfl
  obtain ⟨uinf, hl, he, hp, hn⟩ := logistic_limit (Measure.dirac ()) inferInstance ℱ u hu
    1 (by norm_num) hinit
  have hratio : (Real.pi / 2) / Real.pi = 1 / 2 := by field_simp
  rw [Real.arcsin_one, hratio] at hp hn
  norm_num only at hp hn
  exact ⟨ℱ, u, hu, by norm_num, hinit, uinf, hl, hp, hn⟩

end Transformer.Homogenized
