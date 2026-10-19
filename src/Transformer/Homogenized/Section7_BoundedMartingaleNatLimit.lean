module

public import Transformer.Homogenized.Section7_LogisticAngle
public import Mathlib.Probability.Martingale.Convergence

/-!
# Integer-time limits of the original bounded logistic angle

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence.
Sample the actual real-time martingale at the nonnegative integer clock.
Its original finite-horizon identities give a genuine natural-indexed
martingale with the sampled original filtration. The explicit uniform
bound supplies the L¹ bound required by Mathlib's convergence theorem.
The limit is measurable and integrable, retains the bound almost surely,
and has the original expectation by bounded dominated convergence.

For the logistic equation, the actual arcsin process supplies all these
premises from the source's deterministic initial condition. Its integer
limit therefore retains the initial angle's mean. This result concerns
integer times; extension to every real time requires a separate maximal
estimate and is not assumed by the integer convergence theorem here.
Bounded convergence of the second moments is also recorded for that step.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology ENNReal
namespace Transformer.Homogenized

/-- The original filtration evaluated at the actual nonnegative integer clock.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
def natTimeFiltration {Ω : Type*} {m : MeasurableSpace Ω} (ℱ : Filtration ℝ m) : Filtration ℕ m :=
  ⟨fun n => ℱ (n : ℝ), fun i j hij => ℱ.mono (by exact_mod_cast hij), fun n => ℱ.le _⟩

/-- All original finite-horizon identities give the actual sampled martingale.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem martingale_natTimeFiltration {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M) :
    Martingale (fun n : ℕ => M (n : ℝ)) (natTimeFiltration ℱ) P := by
  refine ⟨fun n => (hM n (Nat.cast_nonneg n)).1 n ⟨Nat.cast_nonneg n, le_rfl⟩, ?_⟩
  intro i j hij
  exact (hM j (Nat.cast_nonneg j)).2.2 i ⟨Nat.cast_nonneg i, by exact_mod_cast hij⟩
    j ⟨Nat.cast_nonneg j, le_rfl⟩ (by exact_mod_cast hij)

/-- The bounded original martingale has an integrable integer limit with the same mean.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem exists_nat_limit_of_bounded_martingale
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P]
    (ℱ : Filtration ℝ m) (M : ℝ → Ω → ℝ)
    (hM : ∀ T, 0 ≤ T → IsMartingaleOn T ℱ P M)
    (C : NNReal) (hbound : ∀ (n : ℕ) ω, |M n ω| ≤ C) :
    ∃ L : Ω → ℝ, StronglyMeasurable L ∧ Integrable L P ∧
      (∀ᵐ ω ∂P, Tendsto (fun n : ℕ => M n ω) atTop (𝓝 (L ω))) ∧
      (∀ᵐ ω ∂P, |L ω| ≤ C) ∧ (∫ ω, L ω ∂P) = ∫ ω, M 0 ω ∂P := by
  let G := natTimeFiltration ℱ
  let f : ℕ → Ω → ℝ := fun n => M n
  have hm : Martingale f G P := martingale_natTimeFiltration P ℱ M hM
  have hb (n : ℕ) : eLpNorm (f n) 1 P ≤ (C : ENNReal) := by
    have hh := eLpNorm_le_of_ae_bound (p := (1 : ENNReal))
      ((hm.stronglyAdapted n).mono (G.le n)).aestronglyMeasurable
      (ae_of_all P (fun ω => by simpa only [Real.norm_eq_abs] using hbound n ω))
    simpa only [measure_univ, ENNReal.one_rpow, one_mul, ENNReal.ofReal_coe_nnreal] using hh
  let L := Filtration.limitProcess f G P
  have hlim : ∀ᵐ ω ∂P, Tendsto (fun n : ℕ => M n ω) atTop (𝓝 (L ω)) :=
    hm.submartingale.ae_tendsto_limitProcess hb
  have hLm : StronglyMeasurable L :=
    Filtration.stronglyMeasurable_limitProcess.mono (iSup_le (fun n => G.le n))
  have hLb : ∀ᵐ ω ∂P, |L ω| ≤ C := by
    filter_upwards [hlim] with ω hω
    exact le_of_tendsto hω.abs (Eventually.of_forall (fun n => hbound n ω))
  have hLi : Integrable L P := Integrable.of_bound hLm.aestronglyMeasurable C
    (by simpa only [Real.norm_eq_abs] using hLb)
  have hD := tendsto_integral_of_dominated_convergence (fun _ => (C : ℝ))
    (fun n => ((hm.stronglyAdapted n).mono (G.le n)).aestronglyMeasurable)
    (integrable_const (C : ℝ)) (fun n => ae_of_all P (fun ω => by
      simpa only [Real.norm_eq_abs] using hbound n ω)) hlim
  have hsame (n : ℕ) : (∫ ω, M n ω ∂P) = ∫ ω, M 0 ω ∂P :=
    (hM n (Nat.cast_nonneg n)).integral_eq_initial (Nat.cast_nonneg n) n
      ⟨Nat.cast_nonneg n, le_rfl⟩
  have hconst : Tendsto (fun n : ℕ => ∫ ω, M n ω ∂P) atTop (𝓝 (∫ ω, M 0 ω ∂P)) := by
    simpa only [hsame] using (tendsto_const_nhds :
      Tendsto (fun _ : ℕ => ∫ ω, M 0 ω ∂P) atTop (𝓝 (∫ ω, M 0 ω ∂P)))
  exact ⟨L, hLm, hLi, hlim, hLb, tendsto_nhds_unique hD hconst⟩

/-- The integer-time limit of the actual logistic angle retains its initial mean.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem logistic_angle_nat_limit
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (u₀ : ℝ) (hu₀ : u₀ ∈ Icc (-1 : ℝ) 1) (hinit : ∀ ω, u 0 ω = u₀) :
    ∃ L : Ω → ℝ, StronglyMeasurable L ∧ Integrable L P ∧
      (∀ᵐ ω ∂P, Tendsto (fun n : ℕ => Real.arcsin (u n ω)) atTop (𝓝 (L ω))) ∧
      (∀ᵐ ω ∂P, |L ω| ≤ Real.pi / 2) ∧ (∫ ω, L ω ∂P) = Real.arcsin u₀ := by
  have hr (ω : Ω) : u 0 ω ∈ Icc (-1 : ℝ) 1 := by rw [hinit ω]; exact hu₀
  obtain ⟨L, hm, hi, hlim, hb, hmean⟩ := exists_nat_limit_of_bounded_martingale P ℱ
    (fun t ω => Real.arcsin (u t ω))
    (fun T hT => (logistic_arcsin_martingale_of_initial_range P ℱ u hu hr T hT).1)
    ⟨Real.pi / 2, by positivity⟩ (fun n ω => abs_le.mpr (Real.arcsin_mem_Icc _))
  refine ⟨L, hm, hi, hlim, hb, ?_⟩
  simpa only [hinit, integral_const, probReal_univ, one_smul] using hmean

/-- Actual bounded almost-sure limits preserve their second moments.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem tendsto_integral_sq_of_bounded_ae_limit
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsFiniteMeasure P]
    (f : ℕ → Ω → ℝ) (g : Ω → ℝ) (C : ℝ)
    (hf : ∀ n, StronglyMeasurable (f n)) (hb : ∀ n ω, |f n ω| ≤ C)
    (hl : ∀ᵐ ω ∂P, Tendsto (fun n => f n ω) atTop (𝓝 (g ω))) :
    Tendsto (fun n => ∫ ω, f n ω ^ 2 ∂P) atTop (𝓝 (∫ ω, g ω ^ 2 ∂P)) := by
  apply tendsto_integral_of_dominated_convergence (fun _ => C ^ 2)
    (fun n => ((hf n).pow 2).aestronglyMeasurable) (integrable_const (C ^ 2))
  · intro n
    apply ae_of_all
    intro ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    change f n ω ^ 2 ≤ C ^ 2
    simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg _) (hb n ω) 2
  · filter_upwards [hl] with ω hω
    exact hω.pow 2

/-- The sine of a Gaussian coordinate is nonconstant in the sample point.
Its time-constant martingale realizes every sampling, bound, and limit premise. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ)) (M : ℝ → ℝ → ℝ),
    (∀ T, 0 ≤ T → IsMartingaleOn T ℱ (ProbabilityTheory.gaussianReal 0 1) M) ∧
      (∀ (n : ℕ) ω, |M n ω| ≤ (1 : NNReal)) ∧
      (∀ n : ℕ, StronglyMeasurable (M n)) ∧
      (∀ᵐ ω ∂ProbabilityTheory.gaussianReal 0 1,
        Tendsto (fun n : ℕ => M n ω) atTop (𝓝 (Real.sin ω))) ∧
      M 0 0 = 0 ∧ M 0 (Real.pi / 2) = 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  have hm : StronglyMeasurable Real.sin := Real.continuous_sin.stronglyMeasurable
  have hi : Integrable Real.sin (ProbabilityTheory.gaussianReal 0 1) :=
    Integrable.of_bound hm.aestronglyMeasurable 1
      (ae_of_all _ (fun ω => by simpa only [Real.norm_eq_abs] using Real.abs_sin_le_one ω))
  exact ⟨ℱ, fun _ => Real.sin, fun T hT => isMartingaleOn_timeConstant _ ℱ T Real.sin hm hi,
    fun n ω => Real.abs_sin_le_one ω, fun n => hm,
    ae_of_all _ (fun ω => tendsto_const_nhds), Real.sin_zero, Real.sin_pi_div_two⟩

/-- A genuine absorbing logistic solution realizes every equation and initial
condition premise, and its actual integer limit has mean π/2. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (u : ℝ → Unit → ℝ),
    IsLogisticSolution (Measure.dirac ()) ℱ u ∧
      (1 : ℝ) ∈ Icc (-1 : ℝ) 1 ∧ (∀ ω, u 0 ω = 1) ∧
      ∃ L : Unit → ℝ, StronglyMeasurable L ∧
        (∀ᵐ ω ∂Measure.dirac (), Tendsto (fun n : ℕ => Real.arcsin (u n ω)) atTop (𝓝 (L ω))) ∧
        (∫ ω, L ω ∂Measure.dirac ()) = Real.pi / 2 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let u : ℝ → Unit → ℝ := fun _ _ => 1
  have hu := isLogisticSolution_one (Measure.dirac ()) ℱ
  have hinit (ω : Unit) : u 0 ω = 1 := rfl
  obtain ⟨L, hm, hi, hlim, hb, hmean⟩ := logistic_angle_nat_limit (Measure.dirac ()) ℱ u hu
    1 (by norm_num) hinit
  exact ⟨ℱ, u, hu, by norm_num, hinit, L, hm, hlim, by simpa only [Real.arcsin_one] using hmean⟩

end Transformer.Homogenized
