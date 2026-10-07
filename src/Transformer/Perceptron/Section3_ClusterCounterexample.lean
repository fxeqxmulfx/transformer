/-
# A counterexample to the exclusion of a single cluster at every temperature

The second part of `thm: bound`, §3.2 of arXiv:2601.21366v2, says that
perceptron weights below `0.16547` exclude a single cluster satisfying
`eq: pairwise.distance` for any `β > 0`. This universal conclusion is false.
With zero perceptron weights, the equal-mass antipodal pair is SOPD for every
positive `β`, with every second variation well-defined. Its circular diameter
is `π`, so it satisfies the stated cluster condition whenever `0 < β ≤ 1/(4π²)`.

The source first establishes its concavity estimate for sufficiently large
`β`, then uses the same estimate to assert the second part for every `β > 0`.
The counterexample addresses that quantifier. It does not contradict the
large-temperature mass bound in the first part of the theorem.
-/

import Transformer.Perceptron.Section3_Antipodal
import Mathlib.Analysis.Real.Sqrt

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- The antipodal pair satisfies the source's cluster condition at small
temperature; arXiv:2601.21366v2, §3.2, `eq: pairwise.distance`.
The integer shift `k = 0` already gives the required bound. -/
theorem antipodalCircleAngles_clustered (β : ℝ) (hβ : 0 < β)
    (hsmall : β ≤ 1 / (4 * π ^ 2)) :
    ∀ i j : Idx 2, ∃ k : ℤ,
      |antipodalCircleAngles i - antipodalCircleAngles j + 2 * π * (k : ℝ)| ≤
        1 / (2 * Real.sqrt β) := by
  have hs : Real.sqrt β ≤ 1 / (2 * π) := by
    apply Real.sqrt_le_iff.mpr
    refine ⟨by positivity, hsmall.trans_eq ?_⟩
    field_simp
    norm_num
  have hπ : π * (1 / (2 * π)) = 1 / 2 := by field_simp
  have hR : π ≤ 1 / (2 * Real.sqrt β) := by
    apply (le_div_iff₀ (by positivity : 0 < 2 * Real.sqrt β)).mpr
    have h := mul_le_mul_of_nonneg_left hs Real.pi_pos.le
    rw [hπ] at h
    nlinarith
  intro i j
  refine ⟨0, ?_⟩
  fin_cases i <;> fin_cases j <;>
    simp [antipodalCircleAngles, abs_of_pos Real.pi_pos] <;>
    simpa using hR

/-- Both small-temperature hypotheses hold at `β = 1/(4π²)`;
arXiv:2601.21366v2, §3.2, `eq: pairwise.distance`. -/
example : 0 < 1 / (4 * π ^ 2) ∧
    (1 / (4 * π ^ 2) : ℝ) ≤ 1 / (4 * π ^ 2) := ⟨by positivity, le_rfl⟩

/-- Every integer shift leaves the distance of the antipodal pair at
least `π`; arXiv:2601.21366v2, §3.2, `eq: pairwise.distance`.
The odd integer `2k-1` cannot vanish. This verifies that the small-temperature
condition comes from the source's circular distance, rather than from a
choice of representatives making the two angles artificially close. -/
theorem antipodalCircleAngles_distance_ge (k : ℤ) :
    π ≤ |antipodalCircleAngles 0 - antipodalCircleAngles 1 + 2 * π * (k : ℝ)| := by
  have hk : (2 * k - 1 : ℤ) ≠ 0 := by omega
  have hi : (1 : ℤ) ≤ |2 * k - 1| := Int.one_le_abs hk
  have hr : (1 : ℝ) ≤ |2 * (k : ℝ) - 1| := by exact_mod_cast hi
  change π ≤ |0 - π + 2 * π * (k : ℝ)|
  rw [show 0 - π + 2 * π * (k : ℝ) = π * (2 * (k : ℝ) - 1) by ring,
    abs_mul, abs_of_pos Real.pi_pos]
  simpa using mul_le_mul_of_nonneg_left hr Real.pi_pos.le

/-- The exact range of temperatures for which this pair satisfies the
source's cluster condition; arXiv:2601.21366v2, §3.2,
`eq: pairwise.distance`. In particular, the pair ceases to be such a cluster
in the large-temperature regime of the theorem's first part. -/
theorem antipodalCircleAngles_clustered_iff (β : ℝ) (hβ : 0 < β) :
    (∀ i j : Idx 2, ∃ k : ℤ,
      |antipodalCircleAngles i - antipodalCircleAngles j + 2 * π * (k : ℝ)| ≤
        1 / (2 * Real.sqrt β)) ↔ β ≤ 1 / (4 * π ^ 2) := by
  constructor
  · intro hcluster
    obtain ⟨k, hk⟩ := hcluster 0 1
    have hR := (antipodalCircleAngles_distance_ge k).trans hk
    rw [le_div_iff₀ (by positivity : 0 < 2 * Real.sqrt β)] at hR
    have hp : π * Real.sqrt β ≤ 1 / 2 := by nlinarith
    have hs : (π * Real.sqrt β) ^ 2 ≤ (1 / 2 : ℝ) ^ 2 :=
      (sq_le_sq₀ (by positivity) (by norm_num)).mpr hp
    rw [mul_pow, Real.sq_sqrt hβ.le] at hs
    rw [le_div_iff₀ (by positivity : 0 < 4 * π ^ 2)]
    nlinarith
  · exact antipodalCircleAngles_clustered β hβ

/-- The temperature hypothesis for the exact characterization is met at
`β = 1`; arXiv:2601.21366v2, §3.2, `eq: pairwise.distance`. -/
example : (0 : ℝ) < 1 := one_pos

/-- A counterexample satisfying every hypothesis of the second part of
`thm: bound`, including SOPD and the atomic presentation;
arXiv:2601.21366v2, §3.2, `eq: theta.bound` and `eq: pairwise.distance`.
Here `σ(s) = s`, `φ(s) = s²`, both neuron weights and vectors vanish,
and the masses are `1/2` at the distinct angles `0,π`. The existence of
every second variation is also certified explicitly. -/
theorem cluster_univ_of_weights_small_counterexample :
    ∃ β : ℝ, 0 < β ∧
      (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id s) s) ∧
      LipschitzWith 1 (id : ℝ → ℝ) ∧ id (0 : ℝ) = 0 ∧
      (|(0 : Idx 2 → ℝ) 0| * ‖(0 : Idx 2 → EucSpace 2) 0‖ ^ 2 +
        |(0 : Idx 2 → ℝ) 1| * ‖(0 : Idx 2 → EucSpace 2) 1‖ ^ 2 < 0.16547) ∧
      IsAtomicOnCircle 2 (fun _ => 1 / 2) antipodalCircleAngles antipodalCircleProb ∧
      IsSOPD β (fun t : ℝ => t ^ 2) id 0 0 antipodalCircleProb ∧
      (∀ (ξ : SSphere 2 → EucSpace 2) (ν : ℝ → Perspective.ProbSphere 2),
        IsGeodesicFrom ξ antipodalCircleProb ν →
        ∃ H : ℝ, HasDerivAt (deriv fun t => energy β (fun s : ℝ => s ^ 2) 0 0 (ν t)) H 0) ∧
      ∀ i j : Idx 2, ∃ k : ℤ,
        |antipodalCircleAngles i - antipodalCircleAngles j + 2 * π * (k : ℝ)| ≤
          1 / (2 * Real.sqrt β) := by
  have hβ : 0 < 1 / (4 * π ^ 2) := by positivity
  refine ⟨1 / (4 * π ^ 2), hβ, ?_, LipschitzWith.id, rfl,
    by norm_num, isAtomicOnCircle_antipodalCircleProb,
    isSOPD_twoAtomProb_antipodal_zero_weights _ hβ _ _ (basePoint 1),
    fun ξ ν hν => exists_secondDeriv_energy_twoAtom_geodesic _ _ _ _ ξ ν hν,
    antipodalCircleAngles_clustered _ hβ le_rfl⟩
  intro s
  simpa using hasDerivAt_pow 2 s

/-- The universal exclusion of a single cluster in the second part of
`thm: bound` is false; arXiv:2601.21366v2, §3.2.
This is the full former statement of `not_cluster_univ_of_weights_small`,
negated without changing its hypotheses or its numerical constant. -/
theorem not_not_cluster_univ_of_weights_small :
    ¬ (∀ (β : ℝ), 0 < β → ∀ φ σ : ℝ → ℝ,
      (∀ s : ℝ, HasDerivAt φ (2 * σ s) s) → LipschitzWith 1 σ → σ 0 = 0 →
      ∀ (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2),
        |ω 0| * ‖a 0‖ ^ 2 + |ω 1| * ‖a 1‖ ^ 2 < 0.16547 →
      ∀ (N : ℕ) (m θ : Idx N → ℝ) (μ : Perspective.ProbSphere 2),
        IsAtomicOnCircle N m θ μ → IsSOPD β φ σ ω a μ →
        ¬ ∀ i j : Idx N, ∃ k : ℤ,
          |θ i - θ j + 2 * π * (k : ℝ)| ≤ 1 / (2 * Real.sqrt β)) := by
  intro h
  obtain ⟨β, hβ, hφ, hlip, hσ0, hω, hatom, hSOPD, -, hcluster⟩ :=
    cluster_univ_of_weights_small_counterexample
  exact (h β hβ _ _ hφ hlip hσ0 0 0 hω 2 _ _ _ hatom hSOPD) hcluster

/-- Counterexamples exist below every positive temperature threshold;
arXiv:2601.21366v2, §3.2, the assertion "for any `β > 0`" in `thm: bound`.
Thus excluding one exceptional temperature would not repair the claim. -/
theorem arbitrarily_small_beta_cluster_univ (ε : ℝ) (hε : 0 < ε) :
    ∃ β : ℝ, 0 < β ∧ β < ε ∧
      IsSOPD β (fun t : ℝ => t ^ 2) id 0 0 antipodalCircleProb ∧
      ∀ i j : Idx 2, ∃ k : ℤ,
        |antipodalCircleAngles i - antipodalCircleAngles j + 2 * π * (k : ℝ)| ≤
          1 / (2 * Real.sqrt β) := by
  let β := min (ε / 2) (1 / (4 * π ^ 2))
  have hβ : 0 < β := lt_min (by positivity) (by positivity)
  have hβε : β < ε := (min_le_left _ _).trans_lt (by linarith)
  have hsmall : β ≤ 1 / (4 * π ^ 2) := min_le_right _ _
  exact ⟨β, hβ, hβε,
    isSOPD_twoAtomProb_antipodal_zero_weights β hβ _ _ (basePoint 1),
    antipodalCircleAngles_clustered β hβ hsmall⟩

/-- The threshold hypothesis can be `ε = 1`;
arXiv:2601.21366v2, §3.2, `thm: bound`. -/
example : (0 : ℝ) < 1 := one_pos

end Transformer.Perceptron
