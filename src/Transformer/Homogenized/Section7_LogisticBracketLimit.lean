import Transformer.Homogenized.Section7_BoundedContinuousMartingaleLimit
import Mathlib.MeasureTheory.Integral.Lebesgue.Markov

/-!
# The logistic bracket budget forces absorbing limits

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, identification of the limit.
Monotone convergence turns a uniform expected budget for nonnegative
increasing observables into an almost-sure bound on their actual values.
The original accumulated logistic bracket has these properties on the
proved joint probability-one state-range event. Its expectation budget
was derived from the actual centered-square martingale identity.

If the original state had an interior real-time limit, the actual bracket
density 2(1-u²) would eventually be bounded below by a positive constant.
Its integrals at integer times would then exceed every finite bound,
contradicting the proved almost-sure bracket bound. Thus any actual limit
is one of the original absorbing endpoints.
The source says that a positive limiting density would force the bracket
to tend to zero; the corrected direction is divergence to infinity.
The proof here establishes that divergence contradiction by ordinary
interval integrals and requires no assumed Brownian time-change result.
-/

open Real MeasureTheory Set Filter
open scoped Topology ENNReal
namespace Transformer.Homogenized

/-- The actual nonnegative monotone sequence is bounded almost surely by its expected budget.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the endpoint argument. -/
theorem ae_bounded_of_monotone_nonnegative_integrals_bounded
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) (F : ℕ → Ω → ℝ)
    (hi : ∀ n, Integrable (F n) P) (hp : ∀ᵐ ω ∂P, ∀ n, 0 ≤ F n ω)
    (hm : ∀ᵐ ω ∂P, Monotone (fun n => F n ω)) (K : ℝ)
    (hK : ∀ n, (∫ ω, F n ω ∂P) ≤ K) :
    ∀ᵐ ω ∂P, ∃ B : ℝ, ∀ n, F n ω ≤ B := by
  let G : ℕ → Ω → ENNReal := fun n ω => ENNReal.ofReal (F n ω)
  have hgm (n : ℕ) : AEMeasurable (G n) P :=
    ENNReal.measurable_ofReal.comp_aemeasurable (hi n).aestronglyMeasurable.aemeasurable
  have hmono : ∀ᵐ ω ∂P, Monotone (fun n => G n ω) := by
    filter_upwards [hm] with ω hω
    exact fun i j hij => ENNReal.ofReal_le_ofReal (hω hij)
  have hbudget : (∫⁻ ω, ⨆ n, G n ω ∂P) ≤ ENNReal.ofReal K := by
    rw [lintegral_iSup' hgm hmono]
    apply iSup_le
    intro n
    rw [show (∫⁻ ω, G n ω ∂P) = ENNReal.ofReal (∫ ω, F n ω ∂P) from
      (ofReal_integral_eq_lintegral_ofReal (hi n) (hp.mono (fun ω hω => hω n))).symm]
    exact ENNReal.ofReal_le_ofReal (hK n)
  have hfinite := ae_lt_top' (AEMeasurable.iSup hgm) (ne_top_of_le_ne_top ENNReal.ofReal_ne_top hbudget)
  filter_upwards [hfinite, hp] with ω hω hpos
  refine ⟨(⨆ n, G n ω).toReal, ?_⟩
  intro n
  have h := ENNReal.toReal_mono hω.ne (le_iSup (fun n => G n ω) n)
  simpa only [G, ENNReal.toReal_ofReal (hpos n)] using h

/-- The actual accumulated nonnegative continuous density increases along integer clocks.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the endpoint argument. -/
theorem nonnegative_integral_clock_monotone (q : ℝ → ℝ)
    (hc : ContinuousOn q (Ici 0)) (hp : ∀ t, 0 ≤ t → 0 ≤ q t) :
    (∀ n : ℕ, 0 ≤ ∫ s in (0 : ℝ)..n, q s) ∧
      Monotone (fun n : ℕ => ∫ s in (0 : ℝ)..n, q s) := by
  have hi (n : ℕ) : IntervalIntegrable q volume 0 n := by
    apply ContinuousOn.intervalIntegrable
    rw [uIcc_of_le (Nat.cast_nonneg n)]
    exact hc.mono (fun s hs => hs.1)
  refine ⟨fun n => intervalIntegral.integral_nonneg (Nat.cast_nonneg n) (fun s hs => hp s hs.1), ?_⟩
  intro i j hij
  apply intervalIntegral.integral_mono_interval le_rfl (Nat.cast_nonneg i) (by exact_mod_cast hij) ?_ (hi j)
  change ∀ᵐ s ∂volume.restrict (Ioc (0 : ℝ) (j : ℝ)), 0 ≤ q s
  rw [ae_restrict_iff' measurableSet_Ioc]
  exact ae_of_all _ (fun s hs => hp s hs.1.le)

/-- The actual original logistic bracket is bounded along all integer times almost surely.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the endpoint argument. -/
theorem logistic_bracket_bounded
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) :
    ∀ᵐ ω ∂P, ∃ B : ℝ, ∀ n : ℕ, (∫ s in (0 : ℝ)..n, 2 * (1 - u s ω ^ 2)) ≤ B := by
  let F : ℕ → Ω → ℝ := fun n ω => ∫ s in (0 : ℝ)..n, 2 * (1 - u s ω ^ 2)
  have hclock : ∀ᵐ ω ∂P, (∀ n, 0 ≤ F n ω) ∧ Monotone (fun n => F n ω) := by
    filter_upwards [logistic_range_ae_all P ℱ u hu hinit] with ω hω
    apply nonnegative_integral_clock_monotone (fun s => 2 * (1 - u s ω ^ 2))
      (continuousOn_const.mul (continuousOn_const.sub ((hu.cont ω).pow 2)))
    intro t ht
    have hs : u t ω ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one _).mpr (abs_le.mpr (hω t ht))
    nlinarith
  exact ae_bounded_of_monotone_nonnegative_integrals_bounded P F
    (fun n => (logistic_bracket_energy P ℱ u hu hinit n (Nat.cast_nonneg n)).1)
    (hclock.mono (fun ω hω => hω.1)) (hclock.mono (fun ω hω => hω.2))
    (Real.pi ^ 2 * P.real univ)
    (fun n => (logistic_bracket_energy P ℱ u hu hinit n (Nat.cast_nonneg n)).2.2)

/-- A bounded actual bracket excludes an interior limit of a continuous logistic-range path.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the endpoint argument. -/
theorem limit_eq_endpoint_of_bounded_logistic_clock (u : ℝ → ℝ)
    (hc : ContinuousOn u (Ici 0)) (hr : ∀ t, 0 ≤ t → u t ∈ Icc (-1 : ℝ) 1)
    (L : ℝ) (hl : Tendsto u atTop (𝓝 L)) (B : ℝ)
    (hB : ∀ n : ℕ, (∫ s in (0 : ℝ)..n, 2 * (1 - u s ^ 2)) ≤ B) : L = 1 ∨ L = -1 := by
  have hL : L ∈ Icc (-1 : ℝ) 1 := isClosed_Icc.mem_of_tendsto hl
    ((eventually_ge_atTop (0 : ℝ)).mono (fun t ht => hr t ht))
  have hL2 : L ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one L).mpr (abs_le.mpr hL)
  by_cases hsq : L ^ 2 = 1
  · have hf : (L - 1) * (L + 1) = 0 := by nlinarith
    rcases mul_eq_zero.mp hf with h | h
    · exact Or.inl (by linarith)
    · exact Or.inr (by linarith)
  have hcpos : 0 < 1 - L ^ 2 := by rcases lt_or_eq_of_le hL2 with h | h; linarith; exact (hsq h).elim
  let q : ℝ → ℝ := fun s => 2 * (1 - u s ^ 2)
  have hqc : ContinuousOn q (Ici 0) := continuousOn_const.mul (continuousOn_const.sub (hc.pow 2))
  have hqp (t : ℝ) (ht : 0 ≤ t) : 0 ≤ q t := by
    have hx : u t ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one _).mpr (abs_le.mpr (hr t ht))
    dsimp [q]
    nlinarith
  have hqi (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : IntervalIntegrable q volume a b :=
    (hqc.mono (fun t ht => (le_min ha hb).trans ht.1)).intervalIntegrable
  have hqlim : Tendsto q atTop (𝓝 (2 * (1 - L ^ 2))) :=
    (tendsto_const_nhds.sub (hl.pow 2)).const_mul 2
  obtain ⟨S, hS⟩ := eventually_atTop.mp (hqlim.eventually (eventually_gt_nhds
    (show 1 - L ^ 2 < 2 * (1 - L ^ 2) by linarith)))
  obtain ⟨N, hN⟩ := exists_nat_gt S
  have hB0 : 0 ≤ B := by simpa only [Nat.cast_zero, intervalIntegral.integral_same] using hB 0
  obtain ⟨n, hn⟩ := exists_nat_gt (B / (1 - L ^ 2) + (N : ℝ))
  have hNn : (N : ℝ) ≤ n := by linarith [div_nonneg hB0 hcpos.le]
  have hlower := intervalIntegral.integral_mono_on hNn (intervalIntegrable_const (c := 1 - L ^ 2))
    (hqi N n (Nat.cast_nonneg N) (Nat.cast_nonneg n))
    (fun t ht => (hS t (hN.le.trans ht.1)).le)
  simp only [intervalIntegral.integral_const, smul_eq_mul] at hlower
  have h0 := (nonnegative_integral_clock_monotone q hqc hqp).1 N
  have hadd := intervalIntegral.integral_add_adjacent_intervals
    (hqi 0 N le_rfl (Nat.cast_nonneg N)) (hqi N n (Nat.cast_nonneg N) (Nat.cast_nonneg n))
  have hbig : B < ((n : ℝ) - N) * (1 - L ^ 2) :=
    (div_lt_iff₀ hcpos).mp (by linarith : B / (1 - L ^ 2) < (n : ℝ) - N)
  have hb := hB n
  change (∫ s in (0 : ℝ)..n, q s) ≤ B at hb
  linarith

/-- Every actual almost-sure real-time logistic limit lies in the two absorbing endpoints.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the endpoint argument. -/
theorem logistic_limit_is_endpoint
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) (L : Ω → ℝ)
    (hl : ∀ᵐ ω ∂P, Tendsto (fun t => u t ω) atTop (𝓝 (L ω))) :
    ∀ᵐ ω ∂P, L ω = 1 ∨ L ω = -1 := by
  filter_upwards [hl, logistic_range_ae_all P ℱ u hu hinit, logistic_bracket_bounded P ℱ u hu hinit]
    with ω hω hr ⟨B, hB⟩
  exact limit_eq_endpoint_of_bounded_logistic_clock (fun t => u t ω) (hu.cont ω) hr (L ω) hω B hB

/-- The nonconstant Gaussian observable sin² realizes the monotone budget
premises with a nonzero integrable bounded family. -/
example : ∃ F : ℕ → ℝ → ℝ,
    (∀ n, Integrable (F n) (ProbabilityTheory.gaussianReal 0 1)) ∧
      (∀ᵐ ω ∂ProbabilityTheory.gaussianReal 0 1, ∀ n, 0 ≤ F n ω) ∧
      (∀ᵐ ω ∂ProbabilityTheory.gaussianReal 0 1, Monotone (fun n => F n ω)) ∧
      (∀ n, (∫ ω, F n ω ∂ProbabilityTheory.gaussianReal 0 1) ≤ 1) := by
  let F : ℕ → ℝ → ℝ := fun _ ω => Real.sin ω ^ 2
  have hi := integrable_sq_of_bounded_stronglyMeasurable (ProbabilityTheory.gaussianReal 0 1)
    Real.sin Real.continuous_sin.stronglyMeasurable 1 Real.abs_sin_le_one
  refine ⟨F, fun n => hi, ae_of_all _ (fun ω n => sq_nonneg _),
    ae_of_all _ (fun ω => monotone_const), ?_⟩
  intro n
  have hh := integral_mono_ae hi (integrable_const (1 : ℝ)) (ae_of_all _ (fun ω => Real.sin_sq_le_one ω))
  simpa only [integral_const, probReal_univ, one_smul] using hh

/-- A genuinely varying positive density realizes the clock hypotheses. -/
example : ContinuousOn (fun t : ℝ => 1 + Real.sin t ^ 2) (Ici 0) ∧
    (∀ t : ℝ, 0 ≤ t → 0 ≤ 1 + Real.sin t ^ 2) := by
  exact ⟨by fun_prop, fun t ht => by positivity⟩

/-- The genuine absorbing solution realizes the full equation, state-range,
real-time limit, and actual bounded-bracket premises simultaneously. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (u : ℝ → Unit → ℝ),
    IsLogisticSolution (Measure.dirac ()) ℱ u ∧
      (∀ t, 0 ≤ t → ∀ ω, u t ω ∈ Icc (-1 : ℝ) 1) ∧
      (∀ ω, Tendsto (fun t => u t ω) atTop (𝓝 1)) ∧
      (∀ (ω : Unit) (n : ℕ), (∫ s in (0 : ℝ)..n, 2 * (1 - u s ω ^ 2)) ≤ 0) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let u : ℝ → Unit → ℝ := fun _ _ => 1
  refine ⟨ℱ, u, isLogisticSolution_one (Measure.dirac ()) ℱ,
    fun t ht ω => by norm_num [u], fun ω => tendsto_const_nhds, ?_⟩
  intro ω n
  simp [u]

end Transformer.Homogenized
