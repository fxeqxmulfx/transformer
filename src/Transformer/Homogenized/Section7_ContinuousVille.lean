/-
# Continuous-time Ville inequality on the original finite horizon

ArXiv:2604.01978v1, §7, the maximal inequality following
eq:Gronwall_E_selfcontained.
Sample actual continuous paths on the proved interval grids.
The filtration and supermartingale properties at those clock
times give the discrete Ville bound on every finite grid.

Every strict crossing is detected on all sufficiently fine grids.
The increasing events that all later grids detect a crossing
therefore transfer the same bound to continuous time. No uniform
continuity in the sample point, tail estimate or optional stopping
theorem at a continuous stopping time is assumed.
The continuous crossing event is also proved measurable.
-/

import Transformer.Homogenized.Section7_DiscreteVille
import Transformer.Homogenized.Section7_IntervalGrid

open MeasureTheory Set Filter
open scoped ENNReal

namespace Transformer.Homogenized

/-- A strict crossing by a process with continuous paths and measurable
time evaluations is an actual measurable event.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem measurableSet_continuous_time_crossing {Ω : Type*} [MeasurableSpace Ω]
    (Z : ℝ → Ω → ℝ) (T : ℝ) (hT : 0 < T)
    (hm : ∀ t ∈ Icc (0 : ℝ) T, Measurable (Z t))
    (hc : ∀ ω, ContinuousOn (fun t => Z t ω) (Icc (0 : ℝ) T)) (a : ℝ) :
    MeasurableSet {ω | ∃ t ∈ Icc (0 : ℝ) T, a < Z t ω} := by
  have heq : {ω | ∃ t ∈ Icc (0 : ℝ) T, a < Z t ω} =
      ⋃ n : ℕ, ⋃ k : ℕ, ⋃ hk : k ≤ n + 1, {ω | a < Z (intervalGridTime T n k) ω} := by
    ext ω
    simp only [mem_iUnion, mem_ofPred_eq]
    constructor
    · rintro ⟨t, ht, hat⟩
      obtain ⟨N, hN⟩ := eventually_atTop.mp
        (intervalGridTime_eventual_crossing T hT _ (hc ω) t ht a hat)
      exact ⟨N, intervalGridIndex T t N, intervalGridIndex_le T hT t ht N, hN N le_rfl⟩
    · rintro ⟨n, k, hk, hak⟩
      exact ⟨intervalGridTime T n k, intervalGridTime_mem T hT.le n k, hak⟩
  rw [heq]
  exact MeasurableSet.iUnion fun n => MeasurableSet.iUnion fun k =>
    MeasurableSet.iUnion fun hk => measurableSet_lt measurable_const (hm _ (intervalGridTime_mem T hT.le n k))

/-- The event that the whole original continuous path stays below a
threshold is measurable as the complement of the proved crossing event.
Source: arXiv:2604.01978v1, §7, the uniform noise-bound event. -/
theorem measurableSet_continuous_time_le {Ω : Type*} [MeasurableSpace Ω]
    (Z : ℝ → Ω → ℝ) (T : ℝ) (hT : 0 < T)
    (hm : ∀ t ∈ Icc (0 : ℝ) T, Measurable (Z t))
    (hc : ∀ ω, ContinuousOn (fun t => Z t ω) (Icc (0 : ℝ) T)) (a : ℝ) :
    MeasurableSet {ω | ∀ t ∈ Icc (0 : ℝ) T, Z t ω ≤ a} := by
  have heq : {ω | ∀ t ∈ Icc (0 : ℝ) T, Z t ω ≤ a} =
      {ω | ∃ t ∈ Icc (0 : ℝ) T, a < Z t ω}ᶜ := by
    ext ω
    simp only [mem_ofPred_eq, mem_compl_iff, not_exists, not_and, not_lt]
  rw [heq]
  exact (measurableSet_continuous_time_crossing Z T hT hm hc a).compl

/-- Continuous-time Ville inequality from the actual supermartingale
conditions on [0,T]. Source: arXiv:2604.01978v1, §7,
the martingale maximal inequality. The finite-horizon adaptedness,
integrability and conditional expectation inequalities are explicit;
the crossing probability bound is proved from them. -/
theorem nonnegative_supermartingale_ville_continuous {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (Z : ℝ → Ω → ℝ)
    (T : ℝ) (hT : 0 < T)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (Z t))
    (hint : ∀ t ∈ Icc (0 : ℝ) T, Integrable (Z t) P)
    (hsuper : ∀ s ∈ Icc (0 : ℝ) T, ∀ t ∈ Icc (0 : ℝ) T, s ≤ t →
      condExp (ℱ s) P (Z t) ≤ᵐ[P] Z s)
    (hpos : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, 0 ≤ Z t ω)
    (hcont : ∀ ω, ContinuousOn (fun t => Z t ω) (Icc (0 : ℝ) T)) (a : ℝ) (ha : 0 ≤ a) :
    ENNReal.ofReal a * P {ω | ∃ t ∈ Icc (0 : ℝ) T, a < Z t ω} ≤
      ENNReal.ofReal (∫ ω, Z 0 ω ∂P) := by
  let G (n : ℕ) : Filtration ℕ m :=
    ⟨fun k => ℱ (intervalGridTime T n k),
      fun i j hij => ℱ.mono (intervalGridTime_monotone T hT.le n hij),
      fun k => ℱ.le _⟩
  have hsamp (n : ℕ) : Supermartingale (fun k => Z (intervalGridTime T n k)) (G n) P := by
    refine ⟨fun k => hadapt _ (intervalGridTime_mem T hT.le n k), ?_,
      fun k => hint _ (intervalGridTime_mem T hT.le n k)⟩
    intro i j hij
    exact hsuper _ (intervalGridTime_mem T hT.le n i) _ (intervalGridTime_mem T hT.le n j)
      (intervalGridTime_monotone T hT.le n hij)
  let A : ℕ → Set Ω := fun n => {ω | ∃ k ≤ n + 1, a ≤ Z (intervalGridTime T n k) ω}
  have hgrid (n : ℕ) : ENNReal.ofReal a * P (A n) ≤ ENNReal.ofReal (∫ ω, Z 0 ω ∂P) := by
    simpa only [intervalGridTime_zero] using
      nonnegative_supermartingale_ville_finite P (G n) (fun k => Z (intervalGridTime T n k))
        (hsamp n) (fun k ω => hpos _ (intervalGridTime_mem T hT.le n k) ω) a ha (n + 1)
  let B : ℕ → Set Ω := fun N => {ω | ∀ n ≥ N, ω ∈ A n}
  have hBmono : Monotone B := fun N M hNM ω hω n hn => hω n (hNM.trans hn)
  have hB (N : ℕ) : ENNReal.ofReal a * P (B N) ≤ ENNReal.ofReal (∫ ω, Z 0 ω ∂P) := by
    have hsub : B N ⊆ A N := fun ω hω => hω N le_rfl
    exact (mul_le_mul_right (measure_mono hsub) (ENNReal.ofReal a)).trans (hgrid N)
  have hcover : {ω | ∃ t ∈ Icc (0 : ℝ) T, a < Z t ω} ⊆ ⋃ N, B N := by
    rintro ω ⟨t, ht, hat⟩
    obtain ⟨N, hN⟩ := eventually_atTop.mp
      (intervalGridTime_eventual_crossing T hT _ (hcont ω) t ht a hat)
    apply mem_iUnion.mpr
    refine ⟨N, fun n hn => ?_⟩
    exact ⟨intervalGridIndex T t n, intervalGridIndex_le T hT t ht n, (hN n hn).le⟩
  calc
    _ ≤ ENNReal.ofReal a * P (⋃ N, B N) := mul_le_mul_right (measure_mono hcover) _
    _ = ⨆ N, ENNReal.ofReal a * P (B N) := by rw [hBmono.measure_iUnion, ENNReal.mul_iSup]
    _ ≤ _ := iSup_le hB

/-- The continuous maximal estimate at a positive threshold, in its
usual probability form. Source: arXiv:2604.01978v1, §7,
the martingale maximal-inequality step. -/
theorem nonnegative_supermartingale_ville_continuous_div {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (Z : ℝ → Ω → ℝ)
    (T : ℝ) (hT : 0 < T)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (Z t))
    (hint : ∀ t ∈ Icc (0 : ℝ) T, Integrable (Z t) P)
    (hsuper : ∀ s ∈ Icc (0 : ℝ) T, ∀ t ∈ Icc (0 : ℝ) T, s ≤ t →
      condExp (ℱ s) P (Z t) ≤ᵐ[P] Z s)
    (hpos : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, 0 ≤ Z t ω)
    (hcont : ∀ ω, ContinuousOn (fun t => Z t ω) (Icc (0 : ℝ) T)) (a : ℝ) (ha : 0 < a) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T, a < Z t ω} ≤
      ENNReal.ofReal (∫ ω, Z 0 ω ∂P) / ENNReal.ofReal a := by
  apply (ENNReal.le_div_iff_mul_le (.inl (by positivity)) (.inl ENNReal.ofReal_ne_top)).2
  simpa only [mul_comm] using
    nonnegative_supermartingale_ville_continuous P ℱ Z T hT hadapt hint hsuper hpos hcont a ha.le

/-- A strictly decreasing continuous positive supermartingale has a
nonempty crossing event at threshold 1/2 on [0,1]. This realizes
every finite-horizon and measurability premise without a constant path. -/
example : ENNReal.ofReal (1 / 2 : ℝ) * (Measure.dirac ())
    (Set.ofPred fun _ : Unit => ∃ t ∈ Icc (0 : ℝ) 1, (1 / 2 : ℝ) < Real.exp (-t)) ≤ 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let Z : ℝ → Unit → ℝ := fun t _ => Real.exp (-t)
  have hadapt (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) : StronglyMeasurable[ℱ t] (Z t) :=
    stronglyMeasurable_const
  have hint (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) : Integrable (Z t) (Measure.dirac ()) := integrable_const _
  have hsuper : ∀ s ∈ Icc (0 : ℝ) 1, ∀ t ∈ Icc (0 : ℝ) 1, s ≤ t →
      condExp (ℱ s) (Measure.dirac ()) (Z t) ≤ᵐ[Measure.dirac ()] Z s := by
    intro s hs t ht hst
    rw [condExp_const (ℱ.le s)]
    apply ae_of_all
    intro ω
    exact Real.exp_le_exp.mpr (by linarith)
  have hpos : ∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, 0 ≤ Z t ω := fun t ht ω => Real.exp_nonneg _
  have hcont : ∀ ω, ContinuousOn (fun t => Z t ω) (Icc (0 : ℝ) 1) := by
    intro ω
    exact (Real.continuous_exp.comp continuous_neg).continuousOn
  have h := nonnegative_supermartingale_ville_continuous (Measure.dirac ()) ℱ Z 1 one_pos
    hadapt hint hsuper hpos hcont (1 / 2) (by norm_num)
  simpa [Z] using h

/-- The continuous example has an actual crossing and changes value
over its horizon. Source: arXiv:2604.01978v1, §7, the maximal inequality. -/
example : (∃ t ∈ Icc (0 : ℝ) 1, (1 / 2 : ℝ) < Real.exp (-t)) ∧
    Real.exp (-(1 : ℝ)) < Real.exp (-(0 : ℝ)) := by
  refine ⟨⟨0, by norm_num, by norm_num⟩, ?_⟩
  norm_num

end Transformer.Homogenized
