module

public import Transformer.Modes.Section2_CriticalCharts
public import Transformer.Modes.Section2_GaussianCriticalImages

/-!
# Almost-sure nondegeneracy of the original Gaussian KDE critical points

ArXiv:2412.09080v3, §2.1, the mode/upcrossing identity after
`eq:Fn`. A sample outside the explicit pair exceptions has a
regular sample coordinate at every zero. Its displacement lies
in one of the three actual score branches, and the original
sample vector is recovered by that coordinate's zero-set chart.

At a query-critical zero the chart parameter is in the genuine
critical domain. The chart derivative determinant is zero there,
so its image is Gaussian-null by equal-dimension Sard and absolute
continuity of the actual Gaussian sample law. There are only
three charts per sample coordinate. The excluded pair exceptions
also have zero Gaussian probability. Thus the original field
has almost surely no simultaneous zero of itself and its query
derivative, with the original positive bandwidth and sample size.
-/

@[expose] public section

open Real Set MeasureTheory

namespace Transformer.Modes

/-- Every original regular query-critical zero is covered by an actual critical chart image.
Source: arXiv:2412.09080v3, §2.1, the critical projection of the original `eq:Fn` zero set. -/
theorem fieldF_regular_critical_subset_images {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    {X : Idx n → ℝ | (∀ i j, i ≠ j → X i ≠ X j ∧ β * (X i - X j) ^ 2 ≠ 4) ∧
      ∃ t : ℝ, fieldF β X t = 0 ∧ deriv (fieldF β X) t = 0} ⊆
      ⋃ a : Idx n × Fin 3, kdeZeroChart β a.1 a.2 '' kdeCriticalChartDomain β a.1 a.2 := by
  rintro X ⟨hgap, t, ht, hd⟩
  obtain ⟨i, hi⟩ := fieldF_sample_transverse hn β X hgap t ht
  have hs : deriv (kdeScore β) (t - X i) ≠ 0 := by
    intro he
    apply hi
    rw [(hasDerivAt_fieldF_sample β X t i (X i)).deriv,
      ← (hasDerivAt_kdeScore β (t - X i)).deriv, he, mul_zero]
  obtain ⟨b, hb⟩ := exists_kdeScoreBranch_of_deriv_ne_zero hβ (t - X i) hs
  let u := Function.update X i t
  have hu : kdeScoreRemainder β i u ∈ kdeScore β '' kdeScoreBranch β b := by
    rw [kdeScoreRemainder_eq_score_of_fieldF_zero hn β X t ht i]
    exact ⟨t - X i, hb, rfl⟩
  have he : kdeZeroChart β i b u = X :=
    kdeZeroChart_update_of_fieldF_zero hβ hn X t ht i b hb
  have hui : u i = t := Function.update_self _ _ _
  apply Set.mem_iUnion.mpr
  refine ⟨(i, b), u, ⟨hu, ?_⟩, he⟩
  rw [he, hui]
  exact hd

/-- Each actual KDE critical-chart image is null under the original product Gaussian sample law.
Source: arXiv:2412.09080v3, §2.1, the equal-dimension critical-value step for `eq:Fn`. -/
theorem gaussianSample_kdeCriticalChart_image_null {β : ℝ} (hβ : 0 < β) {n : ℕ}
    (i : Idx n) (b : Fin 3) :
    gaussianSample n (kdeZeroChart β i b '' kdeCriticalChartDomain β i b) = 0 := by
  exact gaussianSample_critical_image_null n (kdeZeroChart β i b) (kdeZeroChartDeriv β i b)
    (kdeCriticalChartDomain β i b) (fun u hu => hasFDerivWithinAt_kdeCriticalChart hβ i b u hu)
    (fun u hu => kdeCriticalChart_det_zero hβ i b u hu)

/-- The finite union of the original critical-chart images has zero actual Gaussian sample probability.
Source: arXiv:2412.09080v3, §2.1, the critical zero-set projection for `eq:Fn`. -/
theorem gaussianSample_kdeCriticalChart_images_null {β : ℝ} (hβ : 0 < β) (n : ℕ) :
    gaussianSample n (⋃ a : Idx n × Fin 3,
      kdeZeroChart β a.1 a.2 '' kdeCriticalChartDomain β a.1 a.2) = 0 := by
  apply measure_iUnion_null_iff.mpr
  exact fun a => gaussianSample_kdeCriticalChart_image_null hβ a.1 a.2

/-- Original regular data with a query-critical zero form a Gaussian-null set.
Source: arXiv:2412.09080v3, §2.1, projection of the original critical zero set of `eq:Fn`. -/
theorem gaussianSample_regular_critical_null {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    gaussianSample n {X | (∀ i j, i ≠ j → X i ≠ X j ∧ β * (X i - X j) ^ 2 ≠ 4) ∧
      ∃ t : ℝ, fieldF β X t = 0 ∧ deriv (fieldF β X) t = 0} = 0 := by
  exact measure_mono_null (fieldF_regular_critical_subset_images hβ hn)
    (gaussianSample_kdeCriticalChart_images_null hβ n)

/-- Almost surely every zero of the original Gaussian KDE field has nonzero original query derivative.
Source: arXiv:2412.09080v3, §2.1, the missing nondegeneracy behind the mode/upcrossing identity.
Both the critical charts and the exceptional-pair null sets are proved from the actual source model. -/
theorem ae_gaussianSample_fieldF_query_nondegenerate {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    ∀ᵐ X ∂gaussianSample n, ∀ t : ℝ, fieldF β X t = 0 → deriv (fieldF β X) t ≠ 0 := by
  exact ae_fieldF_query_nondegenerate_of_critical_cover n β hβ
    (fun a : Idx n × Fin 3 => kdeZeroChart β a.1 a.2)
    (fun a => kdeZeroChartDeriv β a.1 a.2)
    (fun a => kdeCriticalChartDomain β a.1 a.2)
    (fun a u hu => hasFDerivWithinAt_kdeCriticalChart hβ a.1 a.2 u hu)
    (fun a u hu => kdeCriticalChart_det_zero hβ a.1 a.2 u hu)
    (fieldF_regular_critical_subset_images hβ hn)

/-- The set of all original degenerate queries is null for the actual Gaussian sample distribution.
Source: arXiv:2412.09080v3, §2.1, the almost-sure version of the identity following `eq:Fn`. -/
theorem gaussianSample_fieldF_degenerate_null {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    gaussianSample n {X | ∃ t : ℝ, fieldF β X t = 0 ∧ deriv (fieldF β X) t = 0} = 0 := by
  apply measure_mono_null (μ := gaussianSample n) (t :=
    {X | ¬ ∀ t : ℝ, fieldF β X t = 0 → deriv (fieldF β X) t ≠ 0})
  · rintro X ⟨t, hf, hd⟩ hX
    exact hX t hf hd
  · exact ae_iff.mp (ae_gaussianSample_fieldF_query_nondegenerate hβ hn)

/-- The original field/KDE derivative proportionality has a strictly positive normalization.
Source: arXiv:2412.09080v3, §2.1, the square root in `eq:Fn`. -/
theorem fieldF_normalization_pos {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    0 < Real.sqrt (2 * π * n / β ^ 3) := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  exact Real.sqrt_pos.mpr (by positivity)

/-- The original query derivative is the negative source normalization times the original KDE second derivative.
Source: arXiv:2412.09080v3, §2.1, differentiating the second equality in `eq:Fn`. -/
theorem deriv_fieldF_eq_neg_mul_deriv_deriv {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (X : Idx n → ℝ) (t : ℝ) : deriv (fieldF β X) t =
      -Real.sqrt (2 * π * n / β ^ 3) * deriv (deriv (kde β X)) t := by
  have hfun : fieldF β X = fun s => -Real.sqrt (2 * π * n / β ^ 3) * deriv (kde β X) s :=
    funext fun s => fieldF_eq hβ hn X s
  rw [hfun, deriv_const_mul_field]

/-- Almost surely every critical point of the actual Gaussian KDE has nonzero second derivative.
Source: arXiv:2412.09080v3, §2.1, the almost-sure qualification needed after `eq:Fn`.
This is for the original estimator and original Gaussian data, with no extra genericity hypothesis. -/
theorem ae_gaussianSample_kde_nondegenerate {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    ∀ᵐ X ∂gaussianSample n, ∀ t : ℝ,
      deriv (kde β X) t = 0 → deriv (deriv (kde β X)) t ≠ 0 := by
  filter_upwards [ae_gaussianSample_fieldF_query_nondegenerate hβ hn] with X hX
  intro t hd hdd
  apply hX t
  · rw [fieldF_eq hβ hn, hd, mul_zero]
  · rw [deriv_fieldF_eq_neg_mul_deriv_deriv hβ hn, hdd, mul_zero]

/-- The actual Gaussian KDE samples admitting any degenerate critical point form a null set.
Source: arXiv:2412.09080v3, §2.1, the probability-zero qualification of the mode/upcrossing identity. -/
theorem gaussianSample_kde_degenerate_null {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    gaussianSample n {X | ∃ t : ℝ, deriv (kde β X) t = 0 ∧ deriv (deriv (kde β X)) t = 0} = 0 := by
  apply measure_mono_null (μ := gaussianSample n) (t :=
    {X | ¬ ∀ t : ℝ, deriv (kde β X) t = 0 → deriv (deriv (kde β X)) t ≠ 0})
  · rintro X ⟨t, hf, hd⟩ hX
    exact hX t hf hd
  · exact ae_iff.mp (ae_gaussianSample_kde_nondegenerate hβ hn)

/-- Positive original bandwidth and a nonempty sample size jointly realize the probability statements' premises. -/
example : (0 : ℝ) < 1 ∧ (0 : ℕ) < 5 := ⟨by norm_num, by norm_num⟩

/-- The critical cover really covers nonempty original regular query-critical data.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`, at the five-sample symmetric construction. -/
example : ∃ X : Idx 5 → ℝ,
    (∀ i j, i ≠ j → X i ≠ X j ∧ (1 : ℝ) * (X i - X j) ^ 2 ≠ 4) ∧
    (∃ t : ℝ, fieldF 1 X t = 0 ∧ deriv (fieldF 1 X) t = 0) ∧
    X ∈ ⋃ a : Idx 5 × Fin 3, kdeZeroChart 1 a.1 a.2 '' kdeCriticalChartDomain 1 a.1 a.2 := by
  obtain ⟨X, hgap, hf, hd, hs⟩ := exists_regular_degenerate_fieldF
  have hmem : X ∈ {X : Idx 5 → ℝ | (∀ i j, i ≠ j → X i ≠ X j ∧
      (1 : ℝ) * (X i - X j) ^ 2 ≠ 4) ∧ ∃ t : ℝ, fieldF 1 X t = 0 ∧ deriv (fieldF 1 X) t = 0} :=
    ⟨hgap, 0, hf, hd⟩
  exact ⟨X, hgap, ⟨0, hf, hd⟩, fieldF_regular_critical_subset_images (by norm_num) (by norm_num) hmem⟩

end Transformer.Modes
