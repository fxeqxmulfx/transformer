module

public import Transformer.Homogenized.Section7_CompensatedPartition
public import Transformer.Homogenized.Section7_WeightedMartingale
public import Mathlib.MeasureTheory.Integral.DominatedConvergence

/-!
# The weighted martingale identity in expectation

ArXiv:2604.01978v1, §7, the exponential maximal inequality after
`eq:Gronwall_E_selfcontained`, using the compensated observable from
`def:ito.formula`. The actual adapted finite partition sums have
zero expectation on every event measurable at time zero. The
proved deterministic partition limit and its uniform bound allow
dominated convergence to pass this identity to continuous time.

The compensated observable itself is required to be a martingale.
The weighted process is not assumed to have a stochastic integral
or a martingale property. The limit is also proved integrable.
-/

@[expose] public section

open scoped BigOperators
open Set MeasureTheory Filter Topology
namespace Transformer.Homogenized

/-- The increment of the actual compensated observable is the observable
increment minus its original drift integral. Source: arXiv:2604.01978v1,
§7, the decomposition preceding `eq:Gronwall_E_selfcontained`. -/
theorem compensated_time_increment_eq (T : ℝ) (Y H : ℝ → ℝ)
    (hH : ContinuousOn H (Icc (0 : ℝ) T)) (a b : ℝ)
    (ha : a ∈ Icc (0 : ℝ) T) (hb : b ∈ Icc (0 : ℝ) T) (hab : a ≤ b) :
    (Y b - Y 0 - ∫ x in (0 : ℝ)..b, H x) -
      (Y a - Y 0 - ∫ x in (0 : ℝ)..a, H x) = Y b - Y a - ∫ x in a..b, H x := by
  have h0a : IntervalIntegrable H volume 0 a := by
    apply ContinuousOn.intervalIntegrable
    rw [uIcc_of_le ha.1]
    exact hH.mono (fun x hx => ⟨hx.1, hx.2.trans ha.2⟩)
  have habInt : IntervalIntegrable H volume a b := by
    apply ContinuousOn.intervalIntegrable
    rw [uIcc_of_le hab]
    exact hH.mono (fun x hx => ⟨ha.1.trans hx.1, hx.2.trans hb.2⟩)
  have h := intervalIntegral.integral_add_adjacent_intervals h0a habInt
  linarith

/-- The weighted continuous-time compensation identity follows from
the original martingale and bounded continuous paths. Source:
arXiv:2604.01978v1, §7 and `def:ito.formula`. The random factor is
differentiated through actual finite partitions and dominated convergence. -/
theorem weighted_compensated_expectation_eq_zero
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (hT : 0 < T) (C Y H V : ℝ → Ω → ℝ)
    (hC : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (C t))
    (hcont : ∀ ω, ContinuousOn (fun t => C t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => V t ω) (Icc (0 : ℝ) T))
    (hd : ∀ ω, ∀ x ∈ Icc (0 : ℝ) T, HasDerivAt (fun t => C t ω) (V x ω) x)
    (hM : IsMartingaleOn T ℱ P (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω))
    (BC BY BH BV : ℝ) (hBH : 0 ≤ BH) (hBV : 0 ≤ BV)
    (hCb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |C t ω| ≤ BC)
    (hYb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |Y t ω| ≤ BY)
    (hHb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |H t ω| ≤ BH)
    (hVb : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, |V t ω| ≤ BV)
    (A : Set Ω) (hA : MeasurableSet[ℱ 0] A) :
    Integrable (fun ω => C T ω * Y T ω - C 0 ω * Y 0 ω -
      (∫ x in (0 : ℝ)..T, C x ω * H x ω) - ∫ x in (0 : ℝ)..T, Y x ω * V x ω) P ∧
    (∫ ω in A, C T ω * Y T ω - C 0 ω * Y 0 ω -
      (∫ x in (0 : ℝ)..T, C x ω * H x ω) - (∫ x in (0 : ℝ)..T, Y x ω * V x ω) ∂P) = 0 := by
  let M (t : ℝ) (ω : Ω) := Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω
  let F (n : ℕ) (ω : Ω) := ∑ k ∈ Finset.range (n + 1), C (intervalGridTime T n k) ω *
    (M (intervalGridTime T n (k + 1)) ω - M (intervalGridTime T n k) ω)
  let L (ω : Ω) := C T ω * Y T ω - C 0 ω * Y 0 ω -
    (∫ x in (0 : ℝ)..T, C x ω * H x ω) - ∫ x in (0 : ℝ)..T, Y x ω * V x ω
  let K := 2 * BC * BY + BC * BH * T + BY * BV * T
  have hFn (n : ℕ) (ω : Ω) : F n ω =
      ∑ k ∈ Finset.range (n + 1), C (intervalGridTime T n k) ω *
        (Y (intervalGridTime T n (k + 1)) ω - Y (intervalGridTime T n k) ω -
        ∫ x in intervalGridTime T n k..intervalGridTime T n (k + 1), H x ω) := by
    apply Finset.sum_congr rfl
    intro k hk
    congr 1
    exact compensated_time_increment_eq T (fun t => Y t ω) (fun t => H t ω) (hcont ω).2.2.1
      _ _ (intervalGridTime_mem T hT.le n k) (intervalGridTime_mem T hT.le n (k + 1))
      (intervalGridTime_monotone T hT.le n (Nat.le_succ k))
  have hInt (n : ℕ) : Integrable (F n) P := by
    apply integrable_finsetSum
    intro k hk
    have hInc := (hM.2.1 _ (intervalGridTime_mem T hT.le n (k + 1))).sub
      (hM.2.1 _ (intervalGridTime_mem T hT.le n k))
    exact hInc.bdd_mul ((hC _ (intervalGridTime_mem T hT.le n k)).mono (ℱ.le _)).aestronglyMeasurable
      (ae_of_all _ fun ω => by simpa only [Real.norm_eq_abs] using hCb _ (intervalGridTime_mem T hT.le n k) ω)
  have hbound (n : ℕ) (ω : Ω) : ‖F n ω‖ ≤ K := by
    rw [Real.norm_eq_abs, hFn]
    exact abs_weighted_compensated_intervalGrid_le T hT.le
      (fun t => C t ω) (fun t => Y t ω) (fun t => H t ω) (fun t => V t ω)
      (hcont ω).2.2.1 (hcont ω).2.2.2 (hd ω) BC BY BH BV hBH hBV
      (fun t ht => hCb t ht ω) (fun t ht => hYb t ht ω)
      (fun t ht => hHb t ht ω) (fun t ht => hVb t ht ω) n
  have hlim (ω : Ω) : Tendsto (fun n => F n ω) atTop (𝓝 (L ω)) := by
    simp_rw [hFn]
    exact weighted_compensated_intervalGrid_tendsto T hT
      (fun t => C t ω) (fun t => Y t ω) (fun t => H t ω) (fun t => V t ω)
      (hcont ω).1 (hcont ω).2.1 (hcont ω).2.2.1 (hcont ω).2.2.2 (hd ω)
      BH BV hBH hBV (fun t ht => hHb t ht ω) (fun t ht => hVb t ht ω)
  have hL : Integrable L P := (integrable_const K).mono'
    (aestronglyMeasurable_of_tendsto_ae atTop (fun n => (hInt n).aestronglyMeasurable)
      (ae_of_all _ hlim))
    (ae_of_all _ fun ω => le_of_tendsto (hlim ω).norm (Eventually.of_forall fun n => hbound n ω))
  refine ⟨hL, ?_⟩
  have hz (n : ℕ) : (∫ ω in A, F n ω ∂P) = 0 := by
    apply setIntegral_weighted_martingale_partition_eq_zero P ℱ T M hM (intervalGridTime T n)
      (intervalGridTime_monotone T hT.le n) (n + 1)
      (fun k hk => intervalGridTime_mem T hT.le n k) (fun k => C (intervalGridTime T n k))
      (fun k hk => hC _ (intervalGridTime_mem T hT.le n k)) BC
      (fun k hk ω => hCb _ (intervalGridTime_mem T hT.le n k) ω) A
    rw [intervalGridTime_zero]
    exact hA
  have hconv := tendsto_integral_of_dominated_convergence (μ := P.restrict A) (fun _ : Ω => K)
    (fun n => (hInt n).integrableOn.aestronglyMeasurable)
    (integrable_const K) (fun n => ae_of_all _ (hbound n)) (ae_of_all _ hlim)
  simp_rw [hz] at hconv
  exact tendsto_nhds_unique hconv tendsto_const_nhds

/-- A nonconstant quadratic observable, its linear drift, and a
nonconstant differentiable factor satisfy every hypothesis. Their
compensated process is the actual zero martingale and their final
product is 2. The initial event is the full probability space. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (C Y H V : ℝ → Unit → ℝ),
    (0 : ℝ) < 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (C t)) ∧
    (∀ ω, ContinuousOn (fun t => C t ω) (Icc (0 : ℝ) 1) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) 1) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) 1) ∧
      ContinuousOn (fun t => V t ω) (Icc (0 : ℝ) 1)) ∧
    (∀ ω, ∀ x ∈ Icc (0 : ℝ) 1, HasDerivAt (fun t => C t ω) (V x ω) x) ∧
    IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) ∧
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |C t ω| ≤ 2) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |Y t ω| ≤ 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |H t ω| ≤ 2) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |V t ω| ≤ 1) ∧
    MeasurableSet[ℱ 0] (univ : Set Unit) ∧
    (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ C 1 () * Y 1 () = 2 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let C : ℝ → Unit → ℝ := fun t _ => 1 + t
  let Y : ℝ → Unit → ℝ := fun t _ => t ^ 2
  let H : ℝ → Unit → ℝ := fun t _ => 2 * t
  let V : ℝ → Unit → ℝ := fun _ _ => 1
  have hM : IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) := by
    have hz : (fun t ω => Y t ω - Y 0 ω - ∫ x in (0 : ℝ)..t, H x ω) =
        (fun _ : ℝ => fun _ : Unit => (0 : ℝ)) := by
      funext t ω
      dsimp [Y, H]
      rw [intervalIntegral.integral_const_mul, integral_id]
      ring
    rw [hz]
    exact isMartingaleOn_const 1 ℱ (Measure.dirac ()) 0
  refine ⟨ℱ, C, Y, H, V, one_pos, fun t ht => stronglyMeasurable_const, ?_,
    fun ω x hx => (hasDerivAt_id x).const_add 1, hM, by norm_num, zero_le_one,
    ?_, ?_, ?_, fun t ht ω => by norm_num [V], MeasurableSet.univ, by norm_num, by norm_num, by norm_num [C, Y]⟩
  · intro ω
    exact ⟨(continuous_const.add continuous_id).continuousOn, (continuous_id.pow 2).continuousOn,
      (continuous_const.mul continuous_id).continuousOn, continuousOn_const⟩
  · intro t ht ω
    dsimp [C]
    rw [abs_of_nonneg (by linarith [ht.1] : 0 ≤ 1 + t)]
    linarith [ht.2]
  · intro t ht ω
    dsimp [Y]
    rw [abs_of_nonneg (sq_nonneg t)]
    nlinarith [ht.1, ht.2]
  · intro t ht ω
    dsimp [H]
    rw [abs_of_nonneg (by linarith [ht.1] : 0 ≤ 2 * t)]
    linarith [ht.2]

end Transformer.Homogenized
