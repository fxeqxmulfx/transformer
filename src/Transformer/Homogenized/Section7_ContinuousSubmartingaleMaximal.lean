import Transformer.Homogenized.Section7_ContinuousVille
import Transformer.Homogenized.Section7_LogisticAngle
import Mathlib.MeasureTheory.Function.ConditionalExpectation.CondJensen
import Mathlib.Analysis.Convex.Mul
import Mathlib.Probability.Martingale.OptionalStopping

/-!
# A continuous maximal bound for the actual squared martingale increments

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence.
Conditional Jensen makes the square of the actual martingale a
submartingale whenever its second moments are integrable.
Mathlib's finite Doob inequality then controls a crossing on each
actual interval grid by the original terminal expectation.

Continuity detects every strict crossing on all sufficiently fine grids.
The increasing events of eventual grid detection transfer the same
bound to the whole original interval. The final clock time is exactly
the original horizon. No pathwise uniform modulus, stopping identity,
or continuous-time probability bound is an extra hypothesis.
This bound will control excursions between the logistic angle's
integer sampling times in the long-time convergence argument.
-/

open Real MeasureTheory Set Filter
open scoped Topology ENNReal
namespace Transformer.Homogenized

/-- Conditional Jensen gives the actual square-process conditional order.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem condExp_sq_le_of_isMartingaleOn {Ω : Type*} {m : MeasurableSpace Ω}
    {P : Measure Ω} [IsFiniteMeasure P] {ℱ : Filtration ℝ m} {T : ℝ} {M : ℝ → Ω → ℝ}
    (hM : IsMartingaleOn T ℱ P M)
    (hi : ∀ t ∈ Icc (0 : ℝ) T, Integrable (fun ω => M t ω ^ 2) P)
    (s t : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (ht : t ∈ Icc (0 : ℝ) T) (hst : s ≤ t) :
    (fun ω => M s ω ^ 2) ≤ᵐ[P] P[fun ω => M t ω ^ 2 | ℱ s] := by
  have hconv : ConvexOn ℝ univ (fun x : ℝ => x ^ 2) :=
    Even.convexOn_pow (show Even (2 : ℕ) from ⟨1, rfl⟩)
  have hj := hconv.map_condExp_le_of_finiteDimensional (ℱ.le s) (hM.2.1 t ht) (hi t ht)
  filter_upwards [hj, hM.2.2 s hs t ht hst] with ω hω hm
  change P[M t | ℱ s] ω ^ 2 ≤ P[fun ω => M t ω ^ 2 | ℱ s] ω at hω
  rwa [hm] at hω

/-- The finite crossing probability is bounded by the actual terminal expectation.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem nonnegative_submartingale_maximal_finite
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (G : Filtration ℕ m) (Z : ℕ → Ω → ℝ) (hZ : Submartingale Z G P)
    (hpos : ∀ k ω, 0 ≤ Z k ω) (a : NNReal) (N : ℕ) :
    (a : ENNReal) * P {ω | ∃ k ≤ N, (a : ℝ) ≤ Z k ω} ≤ ENNReal.ofReal (∫ ω, Z N ω ∂P) := by
  let B : Set Ω := {ω | (a : ℝ) ≤ (Finset.range (N + 1)).sup'
    Finset.nonempty_range_add_one (fun k => Z k ω)}
  have hsub : {ω | ∃ k ≤ N, (a : ℝ) ≤ Z k ω} ⊆ B := by
    rintro ω ⟨k, hk, ha⟩
    exact ha.trans (Finset.le_sup' (fun k => Z k ω) (Finset.mem_range.mpr (Nat.lt_succ_of_le hk)))
  calc
    _ ≤ (a : ENNReal) * P B := by gcongr
    _ ≤ ENNReal.ofReal (∫ ω in B, Z N ω ∂P) := maximal_ineq hZ hpos N
    _ ≤ _ := ENNReal.ofReal_le_ofReal (setIntegral_le_integral (hZ.integrable N) (ae_of_all P (hpos N)))

/-- Continuous paths transfer the finite maximal bound to the entire original interval.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, bounded martingale convergence. -/
theorem nonnegative_submartingale_maximal_continuous
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (Z : ℝ → Ω → ℝ) (T : ℝ) (hT : 0 < T)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (Z t))
    (hint : ∀ t ∈ Icc (0 : ℝ) T, Integrable (Z t) P)
    (hsub : ∀ s ∈ Icc (0 : ℝ) T, ∀ t ∈ Icc (0 : ℝ) T, s ≤ t →
      Z s ≤ᵐ[P] P[Z t | ℱ s])
    (hpos : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, 0 ≤ Z t ω)
    (hc : ∀ ω, ContinuousOn (fun t => Z t ω) (Icc (0 : ℝ) T)) (a : NNReal) :
    (a : ENNReal) * P {ω | ∃ t ∈ Icc (0 : ℝ) T, (a : ℝ) < Z t ω} ≤
      ENNReal.ofReal (∫ ω, Z T ω ∂P) := by
  let G (n : ℕ) : Filtration ℕ m :=
    ⟨fun k => ℱ (intervalGridTime T n k),
      fun i j hij => ℱ.mono (intervalGridTime_monotone T hT.le n hij), fun k => ℱ.le _⟩
  have hsamp (n : ℕ) : Submartingale (fun k => Z (intervalGridTime T n k)) (G n) P :=
    ⟨fun k => hadapt _ (intervalGridTime_mem T hT.le n k),
      fun i j hij => hsub _ (intervalGridTime_mem T hT.le n i)
        _ (intervalGridTime_mem T hT.le n j) (intervalGridTime_monotone T hT.le n hij),
      fun k => hint _ (intervalGridTime_mem T hT.le n k)⟩
  let A : ℕ → Set Ω := fun n => {ω | ∃ k ≤ n + 1, (a : ℝ) ≤ Z (intervalGridTime T n k) ω}
  have hgrid (n : ℕ) : (a : ENNReal) * P (A n) ≤ ENNReal.ofReal (∫ ω, Z T ω ∂P) := by
    simpa only [intervalGridTime_last] using nonnegative_submartingale_maximal_finite
      P (G n) (fun k => Z (intervalGridTime T n k)) (hsamp n)
      (fun k ω => hpos _ (intervalGridTime_mem T hT.le n k) ω) a (n + 1)
  let B : ℕ → Set Ω := fun N => {ω | ∀ n ≥ N, ω ∈ A n}
  have hBmono : Monotone B := fun N M hNM ω hω n hn => hω n (hNM.trans hn)
  have hB (N : ℕ) : (a : ENNReal) * P (B N) ≤ ENNReal.ofReal (∫ ω, Z T ω ∂P) := by
    have hsub : B N ⊆ A N := fun ω hω => hω N le_rfl
    exact (mul_le_mul_right (measure_mono hsub) (a : ENNReal)).trans (hgrid N)
  have hcover : {ω | ∃ t ∈ Icc (0 : ℝ) T, (a : ℝ) < Z t ω} ⊆ ⋃ N, B N := by
    rintro ω ⟨t, ht, hat⟩
    obtain ⟨N, hN⟩ := eventually_atTop.mp (intervalGridTime_eventual_crossing T hT _ (hc ω) t ht a hat)
    apply mem_iUnion.mpr
    refine ⟨N, fun n hn => ?_⟩
    exact ⟨intervalGridIndex T t n, intervalGridIndex_le T hT t ht n, (hN n hn).le⟩
  calc
    _ ≤ (a : ENNReal) * P (⋃ N, B N) := mul_le_mul_right (measure_mono hcover) _
    _ = ⨆ N, (a : ENNReal) * P (B N) := by rw [hBmono.measure_iUnion, ENNReal.mul_iSup]
    _ ≤ _ := iSup_le hB

/-- A nonconstant Gaussian-coordinate martingale has actual integrable squares
and realizes two distinct times in the square-order hypothesis. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ)) (M : ℝ → ℝ → ℝ),
    IsMartingaleOn 1 ℱ (ProbabilityTheory.gaussianReal 0 1) M ∧
      (∀ t ∈ Icc (0 : ℝ) 1, Integrable (fun ω => M t ω ^ 2) (ProbabilityTheory.gaussianReal 0 1)) ∧
      (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (0 : ℝ) ≤ 1 ∧
      M 0 0 = 0 ∧ M 0 (Real.pi / 2) = 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  have hm : StronglyMeasurable Real.sin := Real.continuous_sin.stronglyMeasurable
  have hi := Integrable.of_bound (μ := ProbabilityTheory.gaussianReal 0 1) hm.aestronglyMeasurable 1
    (ae_of_all _ (fun ω => by simpa only [Real.norm_eq_abs] using Real.abs_sin_le_one ω))
  have hs : Integrable (fun ω => Real.sin ω ^ 2) (ProbabilityTheory.gaussianReal 0 1) :=
    Integrable.of_bound (hm.pow 2).aestronglyMeasurable 1 (ae_of_all _ (fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      exact Real.sin_sq_le_one ω))
  exact ⟨ℱ, fun _ => Real.sin, isMartingaleOn_timeConstant _ ℱ 1 Real.sin hm hi,
    fun t ht => hs, by norm_num, by norm_num, zero_le_one, Real.sin_zero, Real.sin_pi_div_two⟩

/-- The increasing deterministic process 1+n is a genuine nonnegative
submartingale on every finite discrete horizon. -/
example : ∃ G : Filtration ℕ (inferInstance : MeasurableSpace Unit),
    Submartingale (fun n : ℕ => fun _ : Unit => 1 + (n : ℝ)) G (Measure.dirac ()) ∧
      (∀ (n : ℕ) (_ : Unit), (0 : ℝ) ≤ 1 + (n : ℝ)) := by
  let G := Filtration.const (ι := ℕ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  refine ⟨G, ⟨fun n => stronglyMeasurable_const, ?_, fun n => integrable_const _⟩,
    fun n ω => by positivity⟩
  intro i j hij
  rw [condExp_const (G.le i)]
  exact ae_of_all _ (fun ω => by
    change (1 : ℝ) + i ≤ 1 + j
    exact_mod_cast Nat.add_le_add_left hij 1)

/-- The increasing continuous process 1+t realizes every continuous-time
maximal-inequality premise on a positive horizon. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (Z : ℝ → Unit → ℝ),
    (0 : ℝ) < 1 ∧
      (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (Z t)) ∧
      (∀ t ∈ Icc (0 : ℝ) 1, Integrable (Z t) (Measure.dirac ())) ∧
      (∀ s ∈ Icc (0 : ℝ) 1, ∀ t ∈ Icc (0 : ℝ) 1, s ≤ t →
        Z s ≤ᵐ[Measure.dirac ()] (Measure.dirac ())[Z t | ℱ s]) ∧
      (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, 0 ≤ Z t ω) ∧
      (∀ ω, ContinuousOn (fun t => Z t ω) (Icc (0 : ℝ) 1)) ∧ Z 0 () < Z 1 () := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let Z : ℝ → Unit → ℝ := fun t _ => 1 + t
  refine ⟨ℱ, Z, one_pos, fun t ht => stronglyMeasurable_const, fun t ht => integrable_const _,
    ?_, fun t ht ω => by dsimp [Z]; linarith [ht.1],
    fun ω => (continuous_const.add continuous_id).continuousOn, by norm_num [Z]⟩
  intro s hs t ht hst
  rw [show (Measure.dirac ())[Z t | ℱ s] = fun _ => 1 + t from condExp_const (ℱ.le s) (1 + t)]
  exact ae_of_all _ (fun ω => by dsimp [Z]; linarith)

end Transformer.Homogenized
