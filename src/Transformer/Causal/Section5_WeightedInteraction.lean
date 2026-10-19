module

public import Transformer.Causal.Section5_CircularInteraction

/-!
# Weighted original force and curvature comparisons for frozen centers

ArXiv:2411.04990v2, §5, proof of `thm:fixed_centers`.
The original proof bounds sums of the actual `h` and `g` interactions
by the actual total mass. We prove these finite-sum estimates for
arbitrary original nonnegative masses and actual circular gaps.

Two separated centers cannot both lie within the original peak
radius of any one token. Consequently one of their curvatures is
strictly negative. If their positively weighted total curvature is
nonnegative, some other term is strictly positive, locating the
token inside an actual original peak neighborhood. This includes
the zero-total-curvature case, where nonnegativity alone would not
justify claiming a positive summand without the separation argument.

Finally we retain one actual positive central contribution while
bounding every remaining actual far contribution by its mass.
All sums keep their original signs, original angular distances,
and original frozen masses.
-/

@[expose] public section

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {ι : Type*}

/-- The original weighted far force is bounded by the original total mass times the original force at the minimum circular distance.
Source: arXiv:2411.04990v2, §5, the far-force sum estimate in `thm:fixed_centers`. -/
theorem weighted_h_pot_far_le (s : Finset ι) (a z : ι → ℝ) (ha : ∀ i ∈ s, 0 ≤ a i)
    (β : ℝ) (hβ : 0 < β) (r u : ℝ) (hr : interactionPeak β ≤ r)
    (hfar : ∀ i ∈ s, r ≤ circDist (u - z i)) :
    |∑ i ∈ s, a i * h_pot β (u - z i)| ≤ (∑ i ∈ s, a i) * h_pot β r := by
  calc
    _ ≤ ∑ i ∈ s, |a i * h_pot β (u - z i)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ s, a i * h_pot β r := by
      apply Finset.sum_le_sum
      intro i hi
      rw [abs_mul, abs_of_nonneg (ha i hi)]
      exact mul_le_mul_of_nonneg_left (abs_h_pot_far_le β hβ r (u - z i) hr (hfar i hi)) (ha i hi)
    _ = _ := (Finset.sum_mul s a (h_pot β r)).symm

/-- The original weighted far curvature is bounded below by the original total mass times the original curvature at the minimum circular distance.
Source: arXiv:2411.04990v2, §5, the signed far-curvature sum estimate in `thm:fixed_centers`. -/
theorem weighted_g_pot_far_lower (s : Finset ι) (a z : ι → ℝ) (ha : ∀ i ∈ s, 0 ≤ a i)
    (β : ℝ) (hβ : 0 < β) (r u : ℝ) (hr : 2 * β ^ (-(1 / 2 : ℝ)) ≤ r)
    (hfar : ∀ i ∈ s, r ≤ circDist (u - z i)) :
    (∑ i ∈ s, a i) * g_pot β r ≤ ∑ i ∈ s, a i * g_pot β (u - z i) := by
  rw [Finset.sum_mul]
  exact Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left
    (g_pot_circular_far_bounds β hβ r (u - z i) hr (hfar i hi)).1 (ha i hi)

/-- A nonnegative original weighted curvature sum with an actual strictly negative term must have an actual positive-curvature interaction.
Source: arXiv:2411.04990v2, §5, the positive-summand step in `thm:fixed_centers`, with its strictness justified. -/
theorem weighted_g_pot_exists_pos (s : Finset ι) (a z : ι → ℝ) (ha : ∀ i ∈ s, 0 < a i)
    (β u : ℝ) (hsum : 0 ≤ ∑ i ∈ s, a i * g_pot β (u - z i))
    (hneg : ∃ i ∈ s, g_pot β (u - z i) < 0) : ∃ i ∈ s, 0 < g_pot β (u - z i) := by
  by_contra h
  push Not at h
  have hall : ∀ i ∈ s, a i * g_pot β (u - z i) ≤ 0 :=
    fun i hi => mul_nonpos_of_nonneg_of_nonpos (ha i hi).le (h i hi)
  obtain ⟨i, hi, hgi⟩ := hneg
  have hn := Finset.sum_neg' hall ⟨i, hi, mul_neg_of_pos_of_neg (ha i hi) hgi⟩
  linarith

/-- Two actual original centers separated by more than twice the actual peak radius force an actual negative-curvature center interaction at every token.
Source: arXiv:2411.04990v2, §5, the separated-center geometry used in `thm:fixed_centers`. -/
theorem separated_centers_exists_neg_g (s : Finset ι) (z : ι → ℝ) (β : ℝ) (hβ : 0 < β)
    (u : ℝ) (hsep : ∃ i ∈ s, ∃ j ∈ s, 2 * interactionPeak β < circDist (z i - z j)) :
    ∃ i ∈ s, g_pot β (u - z i) < 0 := by
  by_contra h
  push Not at h
  obtain ⟨i, hi, j, hj, hij⟩ := hsep
  have hdi := (g_pot_nonneg_iff_circDist_le_peak β hβ (u - z i)).mp (h i hi)
  have hdj := (g_pot_nonneg_iff_circDist_le_peak β hβ (u - z j)).mp (h j hj)
  have ht := circDist_triangle (z i) u (z j)
  rw [show z i - u = -(u - z i) by ring, circDist_neg] at ht
  linarith

/-- Nonnegative actual weighted center curvature places the original token strictly inside an original peak neighborhood, including when the total is zero.
Source: arXiv:2411.04990v2, §5, the localization step of `thm:fixed_centers` for separated original centers. -/
theorem weighted_center_curvature_localizes (s : Finset ι) (a z : ι → ℝ) (ha : ∀ i ∈ s, 0 < a i)
    (β : ℝ) (hβ : 0 < β) (u : ℝ) (hsum : 0 ≤ ∑ i ∈ s, a i * g_pot β (u - z i))
    (hsep : ∃ i ∈ s, ∃ j ∈ s, 2 * interactionPeak β < circDist (z i - z j)) :
    ∃ i ∈ s, circDist (u - z i) < interactionPeak β := by
  obtain ⟨i, hi, hgi⟩ := weighted_g_pot_exists_pos s a z ha β u hsum
    (separated_centers_exists_neg_g s z β hβ u hsep)
  exact ⟨i, hi, (g_pot_pos_iff_circDist_lt_peak β hβ _).mp hgi⟩

/-- Retaining one actual original central mass at least one and bounding every other actual far mass gives the manuscript's signed curvature lower bound.
Source: arXiv:2411.04990v2, §5, the central-versus-far stability estimate in `thm:fixed_centers`. -/
theorem weighted_g_pot_core_tail_lower (s : Finset ι) (a z : ι → ℝ) (ha : ∀ j ∈ s, 0 ≤ a j)
    (β : ℝ) (hβ : 0 < β) (u ε r : ℝ) (i : ι) (hi : i ∈ s) (hai : 1 ≤ a i)
    (hcore : circDist (u - z i) ≤ ε) (he : ε < interactionPeak β)
    (hr : 2 * β ^ (-(1 / 2 : ℝ)) ≤ r) (hfar : ∀ j ∈ s, j ≠ i → r ≤ circDist (u - z j)) :
    g_pot β ε + ((∑ j ∈ s, a j) - a i) * g_pot β r ≤ ∑ j ∈ s, a j * g_pot β (u - z j) := by
  classical
  have hc := g_pot_core_lower_pos β hβ (circDist (u - z i)) ε (circDist_nonneg _) hcore he
  rw [g_pot_circDist] at hc
  have hgi : 0 ≤ g_pot β (u - z i) := hc.1.le.trans hc.2
  have hcentral : g_pot β ε ≤ a i * g_pot β (u - z i) := by
    have hm := mul_le_mul_of_nonneg_right hai hgi
    simp only [one_mul] at hm
    exact hc.2.trans hm
  have ht := weighted_g_pot_far_lower (s.erase i) a z
    (fun j hj => ha j (Finset.mem_erase.mp hj).2) β hβ r u hr
    (fun j hj => hfar j (Finset.mem_erase.mp hj).2 (Finset.mem_erase.mp hj).1)
  rw [Finset.sum_erase_eq_sub hi] at ht
  have hsplit := Finset.add_sum_erase s (fun j => a j * g_pot β (u - z j)) hi
  linarith

/-- Actual unit masses at the original antipodal centers jointly realize positive total curvature and the separated-center localization premises.
Source: arXiv:2411.04990v2, §5, the original two-center frozen configuration in `thm:fixed_centers`. -/
example : (∀ i ∈ (Finset.univ : Finset (Idx 2)), (0 : ℝ) < (fun _ => 1 : Idx 2 → ℝ) i) ∧
    0 ≤ ∑ i : Idx 2, (1 : ℝ) * g_pot 1 (0 - (![0, Real.pi] : Idx 2 → ℝ) i) ∧
    (∃ i ∈ (Finset.univ : Finset (Idx 2)), ∃ j ∈ (Finset.univ : Finset (Idx 2)),
      2 * interactionPeak 1 < circDist ((![0, Real.pi] : Idx 2 → ℝ) i - (![0, Real.pi] : Idx 2 → ℝ) j)) := by
  have ht := interactionPeak_lt_scale 1 one_pos
  norm_num at ht
  have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
  refine ⟨fun _ _ => one_pos, ?_, 0, Finset.mem_univ _, 1, Finset.mem_univ _, ?_⟩
  · simp only [Fin.sum_univ_two]
    norm_num [g_pot, Real.cos_pi, Real.sin_pi, Real.cos_neg, Real.sin_neg]
  · simp only [Matrix.cons_val_zero, Matrix.cons_val_one, zero_sub, circDist_neg, hpi]
    linarith [Real.pi_gt_three]

/-- Original center zero, an actual antipodal far center, and a genuine positive subpeak radius satisfy the central-and-tail hypotheses jointly.
Source: arXiv:2411.04990v2, §5, the original signed central-and-far comparison in `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ circDist (0 - 0) ≤ interactionPeak 1 / 2 ∧
    interactionPeak 1 / 2 < interactionPeak 1 ∧ 2 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ 2 ∧
    (2 : ℝ) ≤ circDist (0 - Real.pi) := by
  have ht := (interactionPeak_properties 1 one_pos).1
  have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
  simp only [sub_self, zero_sub, circDist_neg, circDist_zero, hpi, Real.one_rpow, mul_one]
  exact ⟨one_pos, le_rfl, by linarith, by linarith, le_rfl, by linarith [Real.pi_gt_three]⟩

/-- A genuine original antipodal center supplies the strictly negative term required by the positive-summand argument.
Source: arXiv:2411.04990v2, §5, the actual separated-center localization in `thm:fixed_centers`. -/
example : ∃ i ∈ (Finset.univ : Finset (Idx 2)), g_pot 1 (0 - (![0, Real.pi] : Idx 2 → ℝ) i) < 0 := by
  refine ⟨1, Finset.mem_univ _, ?_⟩
  norm_num [g_pot]
  exact Real.exp_pos _

/-- An actual original antipodal singleton with unit mass satisfies both weighted far-sum hypotheses on the same nonempty finite set.
Source: arXiv:2411.04990v2, §5, the original weighted far-force and far-curvature estimates. -/
example : (∀ i ∈ (Finset.univ : Finset (Idx 1)), (0 : ℝ) ≤ (fun _ => 1 : Idx 1 → ℝ) i) ∧
    interactionPeak 1 ≤ 2 ∧ 2 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ 2 ∧
    ∀ i ∈ (Finset.univ : Finset (Idx 1)), (2 : ℝ) ≤ circDist (0 - (fun _ : Idx 1 => Real.pi) i) := by
  have ht := interactionPeak_lt_scale 1 one_pos
  norm_num at ht
  have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
  refine ⟨fun _ _ => zero_le_one, by linarith, by norm_num, fun _ _ => ?_⟩
  simp only [zero_sub, circDist_neg, hpi]
  linarith [Real.pi_gt_three]

end Transformer.Causal
