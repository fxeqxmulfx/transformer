import Transformer.Homogenized.Section7_MartingaleUnitExcursions

/-!
# Full real-time convergence of the bounded logistic angle

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence.
Integer-time convergence and the actual vanishing unit-interval excursions
imply convergence along all real times. The floor of each sufficiently
large time identifies its original unit interval, so no uniform path
modulus or convergence between sampling times is an extra premise.

For a bounded continuous martingale this gives an integrable measurable
limit, with its bound and initial expectation preserved. The original
logistic angle satisfies every hypothesis from its actual martingale
problem and deterministic initial condition. The joint almost-sure state
range then transfers the angle limit through sine back to the original
logistic process. Identification of the absorbing endpoint is separate.
The deterministic example varies in time and verifies the path argument;
the probabilistic examples are genuine martingale and equation witnesses.
-/

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- Actual integer limits and vanishing within-unit excursions give the full real-time limit.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem tendsto_real_of_nat_and_small_unit_excursions (f : ℝ → ℝ) (L : ℝ)
    (hl : Tendsto (fun n : ℕ => f n) atTop (𝓝 L))
    (he : ∀ k : ℕ, ∀ᶠ n : ℕ in atTop, ∀ t ∈ Icc (0 : ℝ) 1,
      |f ((n : ℝ) + t) - f n| ≤ 1 / ((k : ℝ) + 1)) : Tendsto f atTop (𝓝 L) := by
  apply Metric.tendsto_atTop.mpr
  intro ε hε
  obtain ⟨k, hk⟩ := exists_nat_one_div_lt (show (0 : ℝ) < ε / 2 by positivity)
  obtain ⟨N, hN⟩ := Metric.tendsto_atTop.mp hl (ε / 2) (by positivity)
  obtain ⟨U, hU⟩ := eventually_atTop.mp (he k)
  refine ⟨((max N U : ℕ) : ℝ), ?_⟩
  intro t ht
  have ht0 : 0 ≤ t := (Nat.cast_nonneg (max N U)).trans ht
  let n : ℕ := ⌊t⌋₊
  have hn : max N U ≤ n := Nat.le_floor ht
  have hpart : t - (n : ℝ) ∈ Icc (0 : ℝ) 1 :=
    ⟨by dsimp [n]; linarith [Nat.floor_le ht0], by dsimp [n]; linarith [Nat.lt_floor_add_one t]⟩
  have hunit := hU n ((le_max_right N U).trans hn) (t - n) hpart
  have hident : (n : ℝ) + (t - n) = t := by ring
  rw [hident] at hunit
  have hnat := hN n ((le_max_left N U).trans hn)
  rw [Real.dist_eq] at hnat ⊢
  have htri := abs_sub_le (f t) (f n) L
  linarith

/-- A bounded continuous original martingale converges at all real times with preserved mean.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem exists_limit_of_bounded_continuous_martingale
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P]
    (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M)
    (hc : ∀ ω, ContinuousOn (fun t => M t ω) (Ici 0))
    (C : NNReal) (hb : ∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ C) :
    ∃ L : Ω → ℝ, StronglyMeasurable L ∧ Integrable L P ∧
      (∀ᵐ ω ∂P, Tendsto (fun t => M t ω) atTop (𝓝 (L ω))) ∧
      (∀ᵐ ω ∂P, |L ω| ≤ C) ∧ (∫ ω, L ω ∂P) = ∫ ω, M 0 ω ∂P := by
  obtain ⟨L, hm, hi, hlim, hbound, hmean⟩ := exists_nat_limit_of_bounded_martingale P ℱ M hM C
    (fun n => hb n (Nat.cast_nonneg n))
  have he := ae_eventually_small_martingale_unit_excursions P ℱ M hM hc C hb
  refine ⟨L, hm, hi, ?_, hbound, hmean⟩
  filter_upwards [hlim, he] with ω hω heω
  exact tendsto_real_of_nat_and_small_unit_excursions (fun t => M t ω) (L ω) hω heω

/-- The original logistic angle has a measurable integrable real-time limit and the initial mean.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem logistic_angle_limit
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (u₀ : ℝ) (hu₀ : u₀ ∈ Icc (-1 : ℝ) 1) (hinit : ∀ ω, u 0 ω = u₀) :
    ∃ L : Ω → ℝ, StronglyMeasurable L ∧ Integrable L P ∧
      (∀ᵐ ω ∂P, Tendsto (fun t => Real.arcsin (u t ω)) atTop (𝓝 (L ω))) ∧
      (∀ᵐ ω ∂P, |L ω| ≤ Real.pi / 2) ∧ (∫ ω, L ω ∂P) = Real.arcsin u₀ := by
  have hr (ω : Ω) : u 0 ω ∈ Icc (-1 : ℝ) 1 := by rw [hinit ω]; exact hu₀
  obtain ⟨L, hm, hi, hl, hb, hmean⟩ := exists_limit_of_bounded_continuous_martingale P ℱ
    (fun t ω => Real.arcsin (u t ω))
    (fun T hT => (logistic_arcsin_martingale_of_initial_range P ℱ u hu hr T hT).1)
    (fun ω => Real.continuous_arcsin.comp_continuousOn' (hu.cont ω))
    ⟨Real.pi / 2, by positivity⟩ (fun t ht ω => abs_le.mpr (Real.arcsin_mem_Icc _))
  refine ⟨L, hm, hi, hl, hb, ?_⟩
  simpa only [hinit, integral_const, probReal_univ, one_smul] using hmean

/-- The original logistic state converges to the sine of its actual angle limit.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem logistic_state_limit_from_angle
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) (L : Ω → ℝ)
    (hl : ∀ᵐ ω ∂P, Tendsto (fun t => Real.arcsin (u t ω)) atTop (𝓝 (L ω))) :
    ∀ᵐ ω ∂P, Tendsto (fun t => u t ω) atTop (𝓝 (Real.sin (L ω))) := by
  filter_upwards [hl, logistic_range_ae_all P ℱ u hu hinit] with ω hω hr
  apply ((Real.continuous_sin.tendsto (L ω)).comp hω).congr'
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
  exact Real.sin_arcsin (hr t ht).1 (hr t ht).2

/-- The genuinely varying path 1+exp(-t) realizes the integer limit and
actual small-excursion hypotheses; its initial value differs from its limit. -/
example : ∃ f : ℝ → ℝ, Tendsto (fun n : ℕ => f n) atTop (𝓝 1) ∧
    (∀ k : ℕ, ∀ᶠ n : ℕ in atTop, ∀ t ∈ Icc (0 : ℝ) 1,
      |f ((n : ℝ) + t) - f n| ≤ 1 / ((k : ℝ) + 1)) ∧
      f 0 = 2 ∧ Tendsto f atTop (𝓝 1) := by
  let f : ℝ → ℝ := fun t => 1 + Real.exp (-t)
  have hExp : Tendsto (fun n : ℕ => Real.exp (-(n : ℝ))) atTop (𝓝 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero.comp tendsto_natCast_atTop_atTop
  have hn : Tendsto (fun n : ℕ => f n) atTop (𝓝 1) := by
    simpa only [f, add_zero] using hExp.const_add 1
  have he (k : ℕ) : ∀ᶠ n : ℕ in atTop, ∀ t ∈ Icc (0 : ℝ) 1,
      |f ((n : ℝ) + t) - f n| ≤ 1 / ((k : ℝ) + 1) := by
    filter_upwards [hExp.eventually (eventually_lt_nhds
      (show (0 : ℝ) < 1 / ((k : ℝ) + 1) by positivity))] with n hn
    intro t ht
    have hle : Real.exp (-((n : ℝ) + t)) ≤ Real.exp (-(n : ℝ)) :=
      Real.exp_le_exp.mpr (by linarith [ht.1])
    dsimp [f]
    rw [abs_of_nonpos (by linarith : 1 + Real.exp (-((n : ℝ) + t)) - (1 + Real.exp (-(n : ℝ))) ≤ 0)]
    linarith [Real.exp_nonneg (-((n : ℝ) + t))]
  exact ⟨f, hn, he, by norm_num [f], tendsto_real_of_nat_and_small_unit_excursions f 1 hn he⟩

/-- A nonconstant Gaussian-coordinate martingale realizes every probabilistic
bound and path-continuity premise for the full martingale convergence theorem. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ)) (M : ℝ → ℝ → ℝ),
    (∀ T, 0 ≤ T → IsMartingaleOn T ℱ (ProbabilityTheory.gaussianReal 0 1) M) ∧
      (∀ ω, ContinuousOn (fun t => M t ω) (Ici 0)) ∧
      (∀ t, 0 ≤ t → ∀ ω, |M t ω| ≤ (1 : NNReal)) ∧ M 0 (Real.pi / 2) = 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  have hm : StronglyMeasurable Real.sin := Real.continuous_sin.stronglyMeasurable
  have hi := Integrable.of_bound (μ := ProbabilityTheory.gaussianReal 0 1) hm.aestronglyMeasurable 1
    (ae_of_all _ (fun ω => by simpa only [Real.norm_eq_abs] using Real.abs_sin_le_one ω))
  exact ⟨ℱ, fun _ => Real.sin, fun T hT => isMartingaleOn_timeConstant _ ℱ T Real.sin hm hi,
    fun ω => continuousOn_const, fun t ht ω => Real.abs_sin_le_one ω, Real.sin_pi_div_two⟩

/-- The genuine negative absorbing solution realizes all source hypotheses
and both full real-time convergence conclusions. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (u : ℝ → Unit → ℝ),
    IsLogisticSolution (Measure.dirac ()) ℱ u ∧
      (-1 : ℝ) ∈ Icc (-1 : ℝ) 1 ∧ (∀ ω, u 0 ω = -1) ∧
      ∃ L : Unit → ℝ, (∀ᵐ ω ∂Measure.dirac (),
        Tendsto (fun t => Real.arcsin (u t ω)) atTop (𝓝 (L ω))) ∧
        (∀ᵐ ω ∂Measure.dirac (), Tendsto (fun t => u t ω) atTop (𝓝 (Real.sin (L ω)))) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let u : ℝ → Unit → ℝ := fun _ _ => -1
  have hu := isLogisticSolution_neg_one (Measure.dirac ()) ℱ
  have hinit (ω : Unit) : u 0 ω = -1 := rfl
  have hr (ω : Unit) : u 0 ω ∈ Icc (-1 : ℝ) 1 := by norm_num [u]
  obtain ⟨L, hm, hi, hl, hb, hmean⟩ := logistic_angle_limit (Measure.dirac ()) ℱ u hu (-1) (by norm_num) hinit
  exact ⟨ℱ, u, hu, by norm_num, hinit, L, hl,
    logistic_state_limit_from_angle (Measure.dirac ()) ℱ u hu hr L hl⟩

end Transformer.Homogenized
