module

public import Transformer.Causal.InteractionPeak

/-!
# The original interaction peak at every positive temperature

ArXiv:2411.04990v2, Appendix B, `lemma:interaction` (3), used in
§5's proof of `thm:fixed_centers`. The original peak is the arccosine
of the positive quadratic root printed in the manuscript. Its
root identity factors the actual original derivative `g`, giving
the original increasing and decreasing interaction intervals.

The manuscript bounds the peak above by `beta^(-1/2)` for beta at
least one. We prove that upper bound for every positive beta,
as required by the unrestricted positive-temperature hypotheses
of the original frozen-center theorem. If that scale is at least
pi/2, the peak's location makes the bound immediate. Below pi/2,
the manuscript's quartic cosine comparison still applies: the
fourth power of the scale is less than sixteen. The original
interaction functions and the original peak formula are unchanged.
-/

@[expose] public section

open Real Set

namespace Transformer.Causal

/-- The actual positive cosine root printed for the original interaction peak.
Source: arXiv:2411.04990v2, Appendix B, `lemma:interaction` (3). -/
noncomputable def interactionPeakCos (β : ℝ) : ℝ := (-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)

/-- The original peak angle, with exactly the original quadratic-root cosine.
Source: arXiv:2411.04990v2, Appendix B, `lemma:interaction` (3). -/
noncomputable def interactionPeak (β : ℝ) : ℝ := Real.arccos (interactionPeakCos β)

/-- The actual original quadratic root lies in `(0,1)`, solves its original equation, and has a positive complementary factor.
Source: arXiv:2411.04990v2, Appendix B, the factorization in the proof of `lemma:interaction` (3). -/
theorem interactionPeakCos_properties (β : ℝ) (hβ : 0 < β) :
    0 < interactionPeakCos β ∧ interactionPeakCos β < 1 ∧
      β * interactionPeakCos β ^ 2 + interactionPeakCos β - β = 0 ∧
      ∀ u : ℝ, -1 ≤ u → 0 < β * (u + interactionPeakCos β) + 1 := by
  let S := Real.sqrt (4 * β ^ 2 + 1)
  have hβne : β ≠ 0 := hβ.ne'
  have hS2 : S ^ 2 = 4 * β ^ 2 + 1 := Real.sq_sqrt (by positivity)
  have hSgt : 2 * β < S := (Real.lt_sqrt (by positivity)).mpr (by nlinarith)
  have hSlt : S < 2 * β + 1 := (Real.sqrt_lt (by positivity) (by positivity)).mpr (by nlinarith)
  have hSone : 1 < S := (Real.lt_sqrt (by norm_num)).mpr (by nlinarith [sq_pos_of_pos hβ])
  have hc0 : 0 < interactionPeakCos β := by
    unfold interactionPeakCos
    apply div_pos _ (by positivity)
    change 0 < -1 + S
    linarith
  have hc1 : interactionPeakCos β < 1 := by
    rw [interactionPeakCos, div_lt_one (by positivity)]
    change -1 + S < 2 * β
    linarith
  have hroot : β * interactionPeakCos β ^ 2 + interactionPeakCos β - β = 0 := by
    have he : β * interactionPeakCos β ^ 2 + interactionPeakCos β - β =
        (S ^ 2 - 1 - 4 * β ^ 2) / (4 * β) := by
      unfold interactionPeakCos
      dsimp [S]
      field_simp
      ring
    rw [he, hS2]
    ring
  have hbc : β * interactionPeakCos β = (S - 1) / 2 := by unfold interactionPeakCos; dsimp [S]; field_simp; ring
  refine ⟨hc0, hc1, hroot, ?_⟩
  intro u hu
  nlinarith [mul_nonneg hβ.le (by linarith : 0 ≤ u + 1)]

/-- The actual original peak lies strictly between zero and pi/2 and has its original prescribed cosine.
Source: arXiv:2411.04990v2, Appendix B, `lemma:interaction` (3). -/
theorem interactionPeak_properties (β : ℝ) (hβ : 0 < β) :
    0 < interactionPeak β ∧ interactionPeak β < Real.pi / 2 ∧
      Real.cos (interactionPeak β) = interactionPeakCos β := by
  have hc := interactionPeakCos_properties β hβ
  exact ⟨Real.arccos_pos.mpr hc.2.1, Real.arccos_lt_pi_div_two.mpr hc.1,
    Real.cos_arccos (by linarith) hc.2.1.le⟩

/-- The actual original derivative factors at exactly the original interaction peak.
Source: arXiv:2411.04990v2, Appendix B, the sign computation for `lemma:interaction` (3). -/
theorem g_pot_peak_factor (β : ℝ) (hβ : 0 < β) (x : ℝ) :
    g_pot β x = Real.exp (β * (Real.cos x - 1)) *
      ((Real.cos x - interactionPeakCos β) * (β * (Real.cos x + interactionPeakCos β) + 1)) := by
  rw [g_pot]
  congr 1
  linear_combination (-β) * Real.sin_sq_add_cos_sq x + (interactionPeakCos_properties β hβ).2.2.1

/-- The actual original interaction derivative is positive exactly before the original peak on the first semicircle.
Source: arXiv:2411.04990v2, Appendix B, the sign change in `lemma:interaction` (3). -/
theorem g_pot_pos_iff_lt_peak (β : ℝ) (hβ : 0 < β) (x : ℝ) (hx : x ∈ Icc 0 Real.pi) :
    0 < g_pot β x ↔ x < interactionPeak β := by
  have hc := interactionPeakCos_properties β hβ
  have ht := interactionPeak_properties β hβ
  have hmem : interactionPeak β ∈ Icc 0 Real.pi := ⟨ht.1.le, by linarith [Real.pi_pos]⟩
  rw [g_pot_peak_factor β hβ, mul_pos_iff_of_pos_left (Real.exp_pos _),
    mul_pos_iff_of_pos_right (hc.2.2.2 _ (Real.neg_one_le_cos x)), sub_pos, ← ht.2.2]
  exact Real.strictAntiOn_cos.lt_iff_gt hmem hx

/-- The actual original interaction derivative cannot be positive at or beyond the original peak.
Source: arXiv:2411.04990v2, Appendix B, the decreasing branch of `lemma:interaction` (3). -/
theorem g_pot_nonpos_of_peak_le (β : ℝ) (hβ : 0 < β) (x : ℝ) (hx : x ∈ Icc 0 Real.pi)
    (hp : interactionPeak β ≤ x) : g_pot β x ≤ 0 := by
  exact le_of_not_gt (fun hg => (not_lt_of_ge hp) ((g_pot_pos_iff_lt_peak β hβ x hx).mp hg))

/-- The original peak is below the original interaction scale at every positive temperature.
Source: arXiv:2411.04990v2, Appendix B, `lemma:interaction` (3).
The source states this upper bound for beta at least one; the proof here extends it to every positive beta. -/
theorem interactionPeak_lt_scale (β : ℝ) (hβ : 0 < β) : interactionPeak β < β ^ (-(1 / 2 : ℝ)) := by
  have ht := interactionPeak_properties β hβ
  let y := β ^ (-(1 / 2 : ℝ))
  have hy : 0 < y := Real.rpow_pos_of_pos hβ _
  by_cases hwide : Real.pi / 2 ≤ y
  · exact ht.2.1.trans_le hwide
  have hyn : y < Real.pi / 2 := lt_of_not_ge hwide
  have hy2 : β * y ^ 2 = 1 := by
    have he : y ^ 2 = β⁻¹ := by
      dsimp [y]
      rw [← Real.rpow_natCast (β ^ (-(1 / 2 : ℝ))) 2, ← Real.rpow_mul hβ.le]
      norm_num
      rw [Real.rpow_neg_one]
    rw [he, mul_inv_cancel₀ hβ.ne']
  have hyle : y ≤ 2 := by linarith [Real.pi_lt_four]
  have hy2le : y ^ 2 ≤ 4 := by nlinarith
  have hy4 : y ^ 2 * y ^ 2 ≤ 16 := by nlinarith [mul_self_le_mul_self (sq_nonneg y) hy2le]
  let S := Real.sqrt (4 * β ^ 2 + 1)
  have hSgt : 2 * β + y ^ 2 / 12 < S := by
    apply (Real.lt_sqrt (by positivity)).mpr
    nlinarith
  have hby4 : β * y ^ 4 = y ^ 2 := by
    calc
      _ = (β * y ^ 2) * y ^ 2 := by ring
      _ = _ := by rw [hy2, one_mul]
  have hclow : 1 - y ^ 2 / 2 + y ^ 4 / 24 < interactionPeakCos β := by
    rw [interactionPeakCos, lt_div_iff₀ (by positivity)]
    nlinarith
  have hcosy := cos_le_quartic y
  have hymem : y ∈ Icc 0 Real.pi := ⟨hy.le, by linarith [Real.pi_pos]⟩
  have htmem : interactionPeak β ∈ Icc 0 Real.pi := ⟨ht.1.le, by linarith [Real.pi_pos]⟩
  exact (Real.strictAntiOn_cos.lt_iff_gt hymem htmem).mp (by rw [ht.2.2]; linarith)

/-- The actual original interaction rises to its original peak and falls afterwards for every positive temperature.
Source: arXiv:2411.04990v2, Appendix B, the monotonicity intervals of `lemma:interaction` (3). -/
theorem h_pot_monotone_peak (β : ℝ) (hβ : 0 < β) :
    StrictMonoOn (h_pot β) (Icc 0 (interactionPeak β)) ∧
      StrictAntiOn (h_pot β) (Icc (interactionPeak β) Real.pi) := by
  have ht := interactionPeak_properties β hβ
  have hc := interactionPeakCos_properties β hβ
  have hcont : Continuous (h_pot β) := by unfold h_pot; fun_prop
  constructor
  · apply strictMonoOn_of_deriv_pos (convex_Icc _ _) hcont.continuousOn
    intro x hx
    rw [interior_Icc] at hx
    rw [(hasDerivAt_h_pot β x).deriv]
    exact (g_pot_pos_iff_lt_peak β hβ x ⟨hx.1.le, by linarith [hx.2, ht.2.1, Real.pi_pos]⟩).mpr hx.2
  · apply strictAntiOn_of_deriv_neg (convex_Icc _ _) hcont.continuousOn
    intro x hx
    rw [interior_Icc] at hx
    rw [(hasDerivAt_h_pot β x).deriv, g_pot_peak_factor β hβ]
    have hcx : Real.cos x < interactionPeakCos β := by
      rw [← ht.2.2]
      exact Real.cos_lt_cos_of_nonneg_of_le_pi ht.1.le hx.2.le hx.1
    exact mul_neg_of_pos_of_neg (Real.exp_pos _) (mul_neg_of_neg_of_pos (sub_neg.mpr hcx)
      (hc.2.2.2 _ (Real.neg_one_le_cos x)))

/-- An actual positive temperature and an original point before the original peak satisfy the sign and scale premises jointly.
Source: arXiv:2411.04990v2, Appendix B, `lemma:interaction` (3). -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ∈ Icc 0 Real.pi ∧ (0 : ℝ) < interactionPeak 1 :=
  ⟨one_pos, ⟨le_rfl, Real.pi_pos.le⟩, (interactionPeak_properties 1 one_pos).1⟩

/-- The same original positive temperature has an actual semicircle point beyond its original peak.
Source: arXiv:2411.04990v2, Appendix B, the decreasing branch of `lemma:interaction` (3). -/
example : (0 : ℝ) < 1 ∧ Real.pi ∈ Icc 0 Real.pi ∧ interactionPeak 1 ≤ Real.pi :=
  ⟨one_pos, ⟨Real.pi_pos.le, le_rfl⟩, by
    have h := (interactionPeak_properties 1 one_pos).2.1
    linarith [Real.pi_pos]⟩

end Transformer.Causal
