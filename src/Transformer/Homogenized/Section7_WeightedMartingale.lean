module

public import Transformer.Homogenized.MvGenerator
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut
public import Mathlib.Probability.Distributions.Gaussian.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Bounded weights for the martingale identity

For the integrating factor in arXiv:2604.01978v1, §7, we use the martingale
identity supplied by the paper's definition `def:ito.formula`. A bounded weight
measurable at the left endpoint preserves the zero conditional expectation of
an increment. The finite partition statement also holds after restriction to
an event measurable at the initial endpoint. No stochastic integral is assumed
for the weighted process.
-/

@[expose] public section

open MeasureTheory Set
open scoped BigOperators

namespace Transformer.Homogenized

/-- arXiv:2604.01978v1, §7 and `def:ito.formula`: a bounded adapted left weight
preserves the zero conditional expectation of the compensated increment. -/
theorem condExp_mul_martingale_increment_eq_zero {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (T : ℝ)
    (M : ℝ → Ω → ℝ) (hM : IsMartingaleOn T ℱ P M)
    (s t : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (ht : t ∈ Icc (0 : ℝ) T) (hst : s ≤ t)
    (C : Ω → ℝ) (hC : StronglyMeasurable[ℱ s] C) (B : ℝ) (hbound : ∀ ω, |C ω| ≤ B) :
    condExp (ℱ s) P (fun ω => C ω * (M t ω - M s ω)) =ᵐ[P] 0 := by
  have hInc : Integrable (fun ω => M t ω - M s ω) P := (hM.2.1 t ht).sub (hM.2.1 s hs)
  have hProd : Integrable (fun ω => C ω * (M t ω - M s ω)) P :=
    hInc.bdd_mul (hC.mono (ℱ.le s)).aestronglyMeasurable (ae_of_all _ fun ω => by
      simpa only [Real.norm_eq_abs] using hbound ω)
  have hZero : condExp (ℱ s) P (fun ω => M t ω - M s ω) =ᵐ[P] 0 := by
    filter_upwards [condExp_sub (hM.2.1 t ht) (hM.2.1 s hs) (ℱ s),
      hM.2.2 s hs t ht hst, hM.2.2 s hs s hs le_rfl] with ω hsub hts hss
    change condExp (ℱ s) P (fun ω => M t ω - M s ω) ω = 0
    change condExp (ℱ s) P (fun ω => M t ω - M s ω) ω =
      condExp (ℱ s) P (M t) ω - condExp (ℱ s) P (M s) ω at hsub
    rw [hsub, hts, hss, sub_self]
  have hPull := condExp_mul_of_stronglyMeasurable_left hC hProd hInc
  filter_upwards [hPull, hZero] with ω hp hz
  change condExp (ℱ s) P (fun ω => C ω * (M t ω - M s ω)) ω = 0
  change condExp (ℱ s) P (fun ω => C ω * (M t ω - M s ω)) ω =
    C ω * condExp (ℱ s) P (fun ω => M t ω - M s ω) ω at hp
  change condExp (ℱ s) P (fun ω => M t ω - M s ω) ω = 0 at hz
  rw [hp, hz, mul_zero]

/-- arXiv:2604.01978v1, §7: the corresponding weighted increment has mean zero. -/
theorem integral_mul_martingale_increment_eq_zero {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (T : ℝ)
    (M : ℝ → Ω → ℝ) (hM : IsMartingaleOn T ℱ P M)
    (s t : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (ht : t ∈ Icc (0 : ℝ) T) (hst : s ≤ t)
    (C : Ω → ℝ) (hC : StronglyMeasurable[ℱ s] C) (B : ℝ) (hbound : ∀ ω, |C ω| ≤ B) :
    (∫ ω, C ω * (M t ω - M s ω) ∂P) = 0 := by
  calc
    _ = ∫ ω, condExp (ℱ s) P (fun ω => C ω * (M t ω - M s ω)) ω ∂P :=
      (integral_condExp (ℱ.le s)).symm
    _ = ∫ ω : Ω, (0 : ℝ) ∂P := integral_congr_ae
      (condExp_mul_martingale_increment_eq_zero P ℱ T M hM s t hs ht hst C hC B hbound)
    _ = 0 := integral_zero Ω ℝ

/-- arXiv:2604.01978v1, §7: the weighted increment also has mean zero on every
event measurable at its left endpoint. -/
theorem setIntegral_mul_martingale_increment_eq_zero {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (T : ℝ)
    (M : ℝ → Ω → ℝ) (hM : IsMartingaleOn T ℱ P M)
    (s t : ℝ) (hs : s ∈ Icc (0 : ℝ) T) (ht : t ∈ Icc (0 : ℝ) T) (hst : s ≤ t)
    (C : Ω → ℝ) (hC : StronglyMeasurable[ℱ s] C) (B : ℝ) (hbound : ∀ ω, |C ω| ≤ B)
    (A : Set Ω) (hA : MeasurableSet[ℱ s] A) :
    (∫ ω in A, C ω * (M t ω - M s ω) ∂P) = 0 := by
  have hInc : Integrable (fun ω => M t ω - M s ω) P := (hM.2.1 t ht).sub (hM.2.1 s hs)
  have hProd : Integrable (fun ω => C ω * (M t ω - M s ω)) P :=
    hInc.bdd_mul (hC.mono (ℱ.le s)).aestronglyMeasurable (ae_of_all _ fun ω => by
      simpa only [Real.norm_eq_abs] using hbound ω)
  calc
    _ = ∫ ω in A, condExp (ℱ s) P (fun ω => C ω * (M t ω - M s ω)) ω ∂P :=
      (setIntegral_condExp (ℱ.le s) hProd hA).symm
    _ = ∫ ω in A, (0 : ℝ) ∂P := setIntegral_congr_ae (ℱ.le s A hA) (by
      filter_upwards [condExp_mul_martingale_increment_eq_zero
        P ℱ T M hM s t hs ht hst C hC B hbound] with ω hω
      exact fun _ => hω)
    _ = 0 := integral_zero Ω ℝ

/-- arXiv:2604.01978v1, §7: the finite adapted partition sum used for the random
integrating factor has mean zero on each event measurable at the initial time. -/
theorem setIntegral_weighted_martingale_partition_eq_zero
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (M : ℝ → Ω → ℝ) (hM : IsMartingaleOn T ℱ P M)
    (θ : ℕ → ℝ) (hθ : Monotone θ) (N : ℕ) (hdom : ∀ k ≤ N, θ k ∈ Icc (0 : ℝ) T)
    (W : ℕ → Ω → ℝ) (hW : ∀ k < N, StronglyMeasurable[ℱ (θ k)] (W k))
    (B : ℝ) (hbound : ∀ k < N, ∀ ω, |W k ω| ≤ B)
    (A : Set Ω) (hA : MeasurableSet[ℱ (θ 0)] A) :
    (∫ ω in A, ∑ k ∈ Finset.range N, W k ω * (M (θ (k + 1)) ω - M (θ k) ω) ∂P) = 0 := by
  have hInt (k : ℕ) (hk : k < N) : Integrable
      (fun ω => W k ω * (M (θ (k + 1)) ω - M (θ k) ω)) P := by
    have hi : Integrable (fun ω => M (θ (k + 1)) ω - M (θ k) ω) P :=
      (hM.2.1 _ (hdom _ (Nat.succ_le_of_lt hk))).sub (hM.2.1 _ (hdom _ hk.le))
    exact hi.bdd_mul ((hW k hk).mono (ℱ.le _)).aestronglyMeasurable
      (ae_of_all _ fun ω => by simpa only [Real.norm_eq_abs] using hbound k hk ω)
  rw [integral_finsetSum _ (fun k hk => (hInt k (Finset.mem_range.mp hk)).integrableOn)]
  apply Finset.sum_eq_zero
  intro k hk
  have hkN := Finset.mem_range.mp hk
  exact setIntegral_mul_martingale_increment_eq_zero P ℱ T M hM (θ k) (θ (k + 1))
    (hdom _ hkN.le) (hdom _ (Nat.succ_le_of_lt hkN)) (hθ (Nat.le_succ k)) (W k) (hW k hkN) B
    (hbound k hkN) A (ℱ.mono (hθ (Nat.zero_le k)) A hA)

/-- A time-constant Gaussian martingale, a nonconstant sine weight, and a
nontrivial initial event satisfy the hypotheses, including the partition data. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ)) (M : ℝ → ℝ → ℝ),
    IsMartingaleOn 1 ℱ (ProbabilityTheory.gaussianReal 0 1) M ∧
    StronglyMeasurable[ℱ 0] Real.sin ∧ (∀ ω : ℝ, |Real.sin ω| ≤ 1) ∧
    (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (0 : ℝ) ≤ 1 ∧
    MeasurableSet[ℱ 0] (Icc (-1 : ℝ) 1) ∧ M 0 0 = 0 ∧ M 0 1 = 1 ∧
    Monotone (fun k : ℕ => (k : ℝ)) ∧
    (∀ k : ℕ, k ≤ 1 → (k : ℝ) ∈ Icc (0 : ℝ) 1) ∧
    ∀ k : ℕ, k < 1 → StronglyMeasurable[ℱ (k : ℝ)] Real.sin := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  let M : ℝ → ℝ → ℝ := fun _ ω => ω
  have hId : Integrable (fun ω : ℝ => ω) (ProbabilityTheory.gaussianReal 0 1) := by
    exact memLp_one_iff_integrable.mp (ProbabilityTheory.memLp_id_gaussianReal (μ := 0) (v := 1) 1)
  have hIdMeas : StronglyMeasurable (fun ω : ℝ => ω) := stronglyMeasurable_id
  have hSinMeas : StronglyMeasurable Real.sin := Real.continuous_sin.stronglyMeasurable
  have hSet : MeasurableSet (Icc (-1 : ℝ) 1) := measurableSet_Icc
  have hM : IsMartingaleOn 1 ℱ (ProbabilityTheory.gaussianReal 0 1) M := by
    refine ⟨fun t ht => hIdMeas, fun t ht => hId, ?_⟩
    intro s hs t ht hst
    exact Filter.EventuallyEq.of_eq
      (condExp_of_stronglyMeasurable (ℱ.le s) hIdMeas hId)
  refine ⟨ℱ, M, hM, hSinMeas, Real.abs_sin_le_one,
    by norm_num, by norm_num, zero_le_one, hSet, rfl, rfl, ?_, ?_, fun k hk => hSinMeas⟩
  · intro i j hij
    change (i : ℝ) ≤ (j : ℝ)
    exact_mod_cast hij
  · intro k hk
    exact ⟨Nat.cast_nonneg _, by exact_mod_cast hk⟩

end Transformer.Homogenized
