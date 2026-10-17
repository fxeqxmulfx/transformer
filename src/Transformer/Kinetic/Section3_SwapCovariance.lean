/-
# The covariance inequality

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 2, `eq:est-cov-phi`: for
a product law,

  `|Cov(U, H)| ≤ ½ Σ_l s_l(U) s_l(H)`,

where `s_l(U)` bounds the change of `U` when coordinate `l` alone moves
(`abs_covariance_le_sum`). This is the source's
`|Cov(U, H)| ≤ Σ_l ‖D_l U‖ ‖D_l H‖` with the Glauber derivative `D_l` bounded by
its supremum. It follows from the swap identity `covariance_eq_sum_swap`.
-/

import Transformer.Kinetic.Section3_Swap

open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {N : ℕ} (μs : Idx N → Measure ℝ) [∀ i, IsProbabilityMeasure (μs i)]

/-- A bounded measurable function is integrable for a finite measure. -/
theorem integrable_of_abs_le {α : Type*} [MeasurableSpace α] {μ : Measure α} [IsFiniteMeasure μ]
    {f : α → ℝ} (hf : Measurable f) {C : ℝ} (hC : ∀ x, |f x| ≤ C) : Integrable f μ :=
  Integrable.of_bound hf.aestronglyMeasurable C (ae_of_all _ fun x => by simpa using hC x)

/-- **The swap identity.** -/
theorem covariance_eq_sum_swap {U H : (Idx N → ℝ) → ℝ} (hU : Measurable U) (hH : Measurable H)
    {CU CH : ℝ} (hUb : ∀ x, |U x| ≤ CU) (hHb : ∀ x, |H x| ≤ CH) :
    cov[U, H; Measure.pi μs] = ∑ l : Idx N, 1 / 2 *
      ∫ p, (U p.1 - U (Function.update p.1 l (p.2 l))) *
        (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2))
        ∂((Measure.pi μs).prod (Measure.pi μs)) := by
  set μ := Measure.pi μs
  set ν := μ.prod μ
  have hCU : 0 ≤ CU := (abs_nonneg _).trans (hUb 0)
  have hCH : 0 ≤ CH := (abs_nonneg _).trans (hHb 0)
  have hA : cov[U, H; μ] = ∫ p, U p.1 * (H p.1 - H p.2) ∂ν := by
    rw [covariance_eq_sub (MemLp.of_bound hU.aestronglyMeasurable CU
      (ae_of_all _ fun x => by simpa using hUb x))
      (MemLp.of_bound hH.aestronglyMeasurable CH (ae_of_all _ fun x => by simpa using hHb x))]
    have h1 : ∫ p, U p.1 * H p.1 ∂ν = ∫ x, (U * H) x ∂μ := by
      simpa using integral_prod_mul (μ := μ) (ν := μ) (fun x => U x * H x) (fun _ => (1 : ℝ))
    have h2 : ∫ p, U p.1 * H p.2 ∂ν = (∫ x, U x ∂μ) * ∫ x, H x ∂μ := integral_prod_mul U H
    have hb : ∀ p : (Idx N → ℝ) × (Idx N → ℝ), |U p.1 * H p.1| ≤ CU * CH := fun p => by
      rw [abs_mul]
      exact mul_le_mul (hUb _) (hHb _) (abs_nonneg _) hCU
    have hb' : ∀ p : (Idx N → ℝ) × (Idx N → ℝ), |U p.1 * H p.2| ≤ CU * CH := fun p => by
      rw [abs_mul]
      exact mul_le_mul (hUb _) (hHb _) (abs_nonneg _) hCU
    simp_rw [mul_sub]
    rw [integral_sub (integrable_of_abs_le (f := fun p : (Idx N → ℝ) × (Idx N → ℝ) => U p.1 * H p.1)
      ((hU.comp measurable_fst).mul (hH.comp measurable_fst)) hb)
      (integrable_of_abs_le (f := fun p : (Idx N → ℝ) × (Idx N → ℝ) => U p.1 * H p.2)
        ((hU.comp measurable_fst).mul (hH.comp measurable_snd)) hb'), h1, h2]
  have hB : ∀ p : (Idx N → ℝ) × (Idx N → ℝ), H p.1 - H p.2 =
      ∑ l : Idx N, (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)) := fun p => by
    rw [Fin.sum_univ_eq_sum_range (fun l => H (hybrid l p.1 p.2) - H (hybrid (l + 1) p.1 p.2)) N,
      Finset.sum_range_sub', hybrid_zero, hybrid_last]
  have hD : ∀ l : ℕ, Measurable fun p : (Idx N → ℝ) × (Idx N → ℝ) =>
      H (hybrid l p.1 p.2) - H (hybrid (l + 1) p.1 p.2) := fun l =>
    (hH.comp (measurable_hybrid _)).sub (hH.comp (measurable_hybrid _))
  have hDb : ∀ l : ℕ, ∀ p : (Idx N → ℝ) × (Idx N → ℝ),
      |H (hybrid l p.1 p.2) - H (hybrid (l + 1) p.1 p.2)| ≤ 2 * CH := fun l p =>
    (abs_sub _ _).trans (by linarith [hHb (hybrid l p.1 p.2), hHb (hybrid (l + 1) p.1 p.2)])
  have hUD : ∀ l : ℕ, ∀ p : (Idx N → ℝ) × (Idx N → ℝ),
      |U p.1 * (H (hybrid l p.1 p.2) - H (hybrid (l + 1) p.1 p.2))| ≤ CU * (2 * CH) := fun l p => by
    rw [abs_mul]
    exact mul_le_mul (hUb _) (hDb l p) (abs_nonneg _) hCU
  rw [hA]
  simp_rw [hB, Finset.mul_sum]
  rw [integral_finsetSum _ fun (l : Idx N) _ =>
    integrable_of_abs_le (f := fun p : (Idx N → ℝ) × (Idx N → ℝ) =>
      U p.1 * (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)))
      ((hU.comp measurable_fst).mul (hD l)) (hUD l)]
  refine Finset.sum_congr rfl fun l _ => ?_
  have hσ : MeasurePreserving (swapAt l) ν ν := measurePreserving_swapAt μs l
  have hUs : Measurable fun p : (Idx N → ℝ) × (Idx N → ℝ) => U (Function.update p.1 l (p.2 l)) :=
    hU.comp (measurable_fst.comp hσ.measurable)
  have hUsD : ∀ p : (Idx N → ℝ) × (Idx N → ℝ), |U (Function.update p.1 l (p.2 l)) *
      (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2))| ≤ CU * (2 * CH) := fun p => by
    rw [abs_mul]
    exact mul_le_mul (hUb _) (hDb l p) (abs_nonneg _) hCU
  have hswap : ∫ p, U p.1 * (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)) ∂ν =
      -∫ p, U (Function.update p.1 l (p.2 l)) *
        (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)) ∂ν := by
    conv_lhs => rw [← hσ.map_eq]
    rw [integral_map hσ.measurable.aemeasurable
      (by rw [hσ.map_eq]; exact ((hU.comp measurable_fst).mul (hD l)).aestronglyMeasurable),
      ← integral_neg]
    refine integral_congr_ae (ae_of_all _ fun p => ?_)
    simp only [hybrid_swapAt, hybrid_succ_swapAt]
    simp only [swapAt]
    ring
  simp_rw [sub_mul]
  rw [integral_sub (integrable_of_abs_le (f := fun p : (Idx N → ℝ) × (Idx N → ℝ) =>
      U p.1 * (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)))
      ((hU.comp measurable_fst).mul (hD l)) (hUD l))
    (integrable_of_abs_le (f := fun p : (Idx N → ℝ) × (Idx N → ℝ) =>
      U (Function.update p.1 l (p.2 l)) *
        (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2))) (hUs.mul (hD l)) hUsD)]
  linarith

/-- **The covariance inequality (`eq:est-cov-phi`).** -/
theorem abs_covariance_le_sum {U H : (Idx N → ℝ) → ℝ} (hU : Measurable U) (hH : Measurable H)
    {CU CH : ℝ} (hUb : ∀ x, |U x| ≤ CU) (hHb : ∀ x, |H x| ≤ CH) {sU sH : Idx N → ℝ}
    (hsU : ∀ x l z, |U x - U (Function.update x l z)| ≤ sU l)
    (hsH : ∀ x l z, |H x - H (Function.update x l z)| ≤ sH l) :
    |cov[U, H; Measure.pi μs]| ≤ 1 / 2 * ∑ l, sU l * sH l := by
  rw [covariance_eq_sum_swap μs hU hH hUb hHb, Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun l _ => ?_)
  rw [abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 1 / 2)]
  refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
  have h0 : 0 ≤ sU l := by
    have := hsU 0 l ((0 : Idx N → ℝ) l)
    rwa [Function.update_eq_self, sub_self, abs_zero] at this
  have hb : ∀ p : (Idx N → ℝ) × (Idx N → ℝ), ‖(U p.1 - U (Function.update p.1 l (p.2 l))) *
      (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2))‖ ≤ sU l * sH l := fun p => by
    rw [Real.norm_eq_abs, abs_mul, hybrid_succ]
    exact mul_le_mul (hsU _ _ _) (hsH _ _ _) (abs_nonneg _) h0
  simpa [Real.norm_eq_abs] using norm_integral_le_of_norm_le_const
    (μ := (Measure.pi μs).prod (Measure.pi μs)) (ae_of_all _ hb)

/-- The hypotheses of `integrable_of_abs_le` hold for a constant. -/
example := integrable_of_abs_le (μ := Measure.dirac (0 : ℝ)) (f := fun _ => 0)
  measurable_const (C := 0) (fun _ => by simp)

/-- The hypotheses of `covariance_eq_sum_swap` and `abs_covariance_le_sum` hold for
constant observables under Dirac masses. -/
example := And.intro (covariance_eq_sum_swap (fun _ : Idx 2 => Measure.dirac (0 : ℝ))
    (U := fun _ => 0) (H := fun _ => 0) measurable_const measurable_const (CU := 0) (CH := 0)
    (fun _ => by simp) (fun _ => by simp))
  (abs_covariance_le_sum (fun _ : Idx 2 => Measure.dirac (0 : ℝ)) (U := fun _ => 0)
    (H := fun _ => 0) measurable_const measurable_const (CU := 0) (CH := 0) (fun _ => by simp)
    (fun _ => by simp) (sU := fun _ => 0) (sH := fun _ => 0) (fun _ _ _ => by simp)
    (fun _ _ _ => by simp))

end Kinetic
end Transformer
