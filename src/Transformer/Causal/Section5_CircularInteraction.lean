module

public import Transformer.Causal.Section5_FrozenDefs
public import Transformer.Causal.SectionB_FarCurvature

/-!
# Original interaction bounds in the manuscript's circular distance

ArXiv:2411.04990v2, §5, `thm:fixed_centers` and its proof, with
the original interaction functions from Appendix B.
The formal statement uses lifted real angles and the actual
shortest circular distance `circDist`, rather than the absolute
difference of arbitrary lifts. We identify that original distance
with the norm in the original angle quotient and prove its
triangle and separation inequalities.

The original curvature is unchanged by replacing an angular
difference by this shortest distance; the absolute original force
becomes the nonnegative semicircle force. Thus the proved central
and far interaction bounds apply to the actual lifted dynamics,
including angles crossing the choice of representative at pi.
-/

@[expose] public section

open Real Set

namespace Transformer.Causal

/-- The original shortest circular distance is nonnegative.
Source: arXiv:2411.04990v2, §5, the circle-distance convention in `thm:fixed_centers`. -/
theorem circDist_nonneg (x : ℝ) : 0 ≤ circDist x := abs_nonneg _

/-- The original shortest circular distance never exceeds the antipodal distance pi.
Source: arXiv:2411.04990v2, §5, the circle geometry of `thm:fixed_centers`. -/
theorem circDist_le_pi (x : ℝ) : circDist x ≤ Real.pi := Real.Angle.abs_toReal_le_pi _

/-- Identical original lifted angles have zero original circular distance.
Source: arXiv:2411.04990v2, §5, the actual circle geometry in `thm:fixed_centers`. -/
theorem circDist_zero : circDist 0 = 0 := by simp [circDist]

/-- The original circular distance equals the actual angle-quotient norm, retaining the original lift convention.
Source: arXiv:2411.04990v2, §5, the geodesic interpretation of distances in `thm:fixed_centers`. -/
theorem circDist_eq_norm (x : ℝ) : circDist x = ‖(x : Real.Angle)‖ := by
  have hp : (2 * Real.pi : ℝ) ≠ 0 := by positivity
  have hb : |(x : Real.Angle).toReal| ≤ |2 * Real.pi| / 2 := by
    simpa only [abs_of_pos (show (0 : ℝ) < 2 * Real.pi by positivity), mul_div_cancel_left₀
      Real.pi (show (2 : ℝ) ≠ 0 by norm_num)] using Real.Angle.abs_toReal_le_pi (x : Real.Angle)
  have hn := (AddCircle.norm_coe_eq_abs_iff (2 * Real.pi) hp).mpr hb
  change ‖(((x : Real.Angle).toReal : ℝ) : Real.Angle)‖ = circDist x at hn
  rw [Real.Angle.coe_toReal] at hn
  exact hn.symm

/-- Reversing an original angular difference preserves its original circular distance.
Source: arXiv:2411.04990v2, §5, the symmetric circle distance in `thm:fixed_centers`. -/
theorem circDist_neg (x : ℝ) : circDist (-x) = circDist x := by
  rw [circDist_eq_norm, Real.Angle.coe_neg, norm_neg, ← circDist_eq_norm]

/-- The original circular distance obeys the actual angle-quotient triangle inequality.
Source: arXiv:2411.04990v2, §5, the intercluster distance estimates in the proof of `thm:fixed_centers`. -/
theorem circDist_add_le (x y : ℝ) : circDist (x + y) ≤ circDist x + circDist y := by
  simp only [circDist_eq_norm, Real.Angle.coe_add]
  exact norm_add_le _ _

/-- Three original lifted angles satisfy the manuscript's circular triangle inequality.
Source: arXiv:2411.04990v2, §5, the distance comparison used in `thm:fixed_centers`. -/
theorem circDist_triangle (x y z : ℝ) : circDist (x - z) ≤ circDist (x - y) + circDist (y - z) := by
  rw [show x - z = (x - y) + (y - z) by ring]
  exact circDist_add_le _ _

/-- The original circular separation decreases by at most the actual displacement of one endpoint.
Source: arXiv:2411.04990v2, §5, the distance subtraction used in `thm:fixed_centers`. -/
theorem circDist_reverse_triangle (x y z : ℝ) : circDist (x - z) - circDist (x - y) ≤ circDist (y - z) := by
  linarith [circDist_triangle x y z]

/-- On the original first semicircle the actual shortest circular distance is the given nonnegative angle.
Source: arXiv:2411.04990v2, Appendix B and §5, the semicircle interaction convention in `thm:fixed_centers`. -/
theorem circDist_eq_self (x : ℝ) (hx : x ∈ Icc 0 Real.pi) : circDist x = x :=
  Real.Angle.abs_toReal_coe_eq_self_iff.mpr hx

/-- The original cosine is unchanged by taking the actual shortest original circular distance.
Source: arXiv:2411.04990v2, Appendix B and §5, the original circular interaction in `thm:fixed_centers`. -/
theorem cos_circDist (x : ℝ) : Real.cos (circDist x) = Real.cos x := by
  simp only [circDist, Real.cos_abs, Real.Angle.cos_toReal, Real.Angle.cos_coe]

/-- The original sine magnitude equals the sine of the actual original shortest circular distance.
Source: arXiv:2411.04990v2, Appendix B and §5, the force-magnitude comparisons in `thm:fixed_centers`. -/
theorem abs_sin_eq_sin_circDist (x : ℝ) : |Real.sin x| = Real.sin (circDist x) := by
  have h := Real.abs_sin_eq_sin_abs_of_abs_le_pi (Real.Angle.abs_toReal_le_pi (x : Real.Angle))
  simpa only [Real.Angle.sin_toReal, Real.Angle.sin_coe, circDist] using h

/-- The actual original force magnitude is the original positive-semicircle force at the original circular distance.
Source: arXiv:2411.04990v2, §5, the absolute interaction-force estimates in `thm:fixed_centers`. -/
theorem abs_h_pot_eq_h_pot_circDist (β x : ℝ) : |h_pot β x| = h_pot β (circDist x) := by
  rw [h_pot, abs_mul, abs_of_pos (Real.exp_pos _), abs_sin_eq_sin_circDist]
  rw [h_pot, cos_circDist]

/-- The actual original interaction curvature is unchanged by the actual original shortest circular distance.
Source: arXiv:2411.04990v2, §5, the signed curvature comparisons in `thm:fixed_centers`. -/
theorem g_pot_circDist (β x : ℝ) : g_pot β (circDist x) = g_pot β x := by
  have hs : Real.sin (circDist x) ^ 2 = Real.sin x ^ 2 := by
    rw [← abs_sin_eq_sin_circDist, sq_abs]
  rw [g_pot, cos_circDist, hs]
  rfl

/-- An actual original circular interaction has positive curvature exactly inside the actual original peak radius.
Source: arXiv:2411.04990v2, §5, the positive-term localization used in `thm:fixed_centers`. -/
theorem g_pot_pos_iff_circDist_lt_peak (β : ℝ) (hβ : 0 < β) (x : ℝ) :
    0 < g_pot β x ↔ circDist x < interactionPeak β := by
  rw [← g_pot_circDist β x]
  exact g_pot_pos_iff_lt_peak β hβ _ ⟨circDist_nonneg x, circDist_le_pi x⟩

/-- An actual original circular interaction has negative curvature exactly beyond the actual original peak radius.
Source: arXiv:2411.04990v2, §5, the signed interaction localization in `thm:fixed_centers`. -/
theorem g_pot_neg_iff_peak_lt_circDist (β : ℝ) (hβ : 0 < β) (x : ℝ) :
    g_pot β x < 0 ↔ interactionPeak β < circDist x := by
  constructor
  · intro hg
    by_contra h
    have hp := g_pot_nonneg_of_le_peak β hβ (circDist x) (circDist_nonneg x) (le_of_not_gt h)
    rw [g_pot_circDist] at hp
    linarith
  · intro hx
    have ht := interactionPeak_properties β hβ
    have hc := Real.cos_lt_cos_of_nonneg_of_le_pi ht.1.le (circDist_le_pi x) hx
    rw [ht.2.2, cos_circDist] at hc
    rw [g_pot_peak_factor β hβ]
    exact mul_neg_of_pos_of_neg (Real.exp_pos _) (mul_neg_of_neg_of_pos (sub_neg.mpr hc)
      ((interactionPeakCos_properties β hβ).2.2.2 _ (Real.neg_one_le_cos x)))

/-- Nonnegative actual original circular curvature forces the actual original interaction inside the closed peak neighborhood.
Source: arXiv:2411.04990v2, §5, the nonpositive-Jacobian localization in `thm:fixed_centers`. -/
theorem g_pot_nonneg_iff_circDist_le_peak (β : ℝ) (hβ : 0 < β) (x : ℝ) :
    0 ≤ g_pot β x ↔ circDist x ≤ interactionPeak β := by
  simpa only [not_lt] using not_congr (g_pot_neg_iff_peak_lt_circDist β hβ x)

/-- A minimum original circular distance beyond the original peak bounds the actual original force magnitude from above.
Source: arXiv:2411.04990v2, §5, the far-force bound in the proof of `thm:fixed_centers`. -/
theorem abs_h_pot_far_le (β : ℝ) (hβ : 0 < β) (r x : ℝ)
    (hr : interactionPeak β ≤ r) (hrx : r ≤ circDist x) : |h_pot β x| ≤ h_pot β r := by
  rw [abs_h_pot_eq_h_pot_circDist]
  exact (h_pot_monotone_peak β hβ).2.antitoneOn ⟨hr, hrx.trans (circDist_le_pi x)⟩
    ⟨hr.trans hrx, circDist_le_pi x⟩ hrx

/-- The original minimum-distance far-curvature bound applies directly to arbitrary actual original lifted angle differences.
Source: arXiv:2411.04990v2, §5, the far-curvature bound in `thm:fixed_centers`. -/
theorem g_pot_circular_far_bounds (β : ℝ) (hβ : 0 < β) (r x : ℝ)
    (hr : 2 * β ^ (-(1 / 2 : ℝ)) ≤ r) (hrx : r ≤ circDist x) :
    g_pot β r ≤ g_pot β x ∧ g_pot β x < 0 := by
  rw [← g_pot_circDist β x]
  exact g_pot_far_bounds β hβ r (circDist x) hr hrx (circDist_le_pi x)

/-- Original separated centers and actual nearby original tokens retain the manuscript's exact separation minus both neighborhood radii.
Source: arXiv:2411.04990v2, §5, the intercluster triangle estimate in `thm:fixed_centers`. -/
theorem circDist_between_neighborhoods (θ ψ x y d r s : ℝ)
    (hsep : d < circDist (θ - ψ)) (hx : circDist (x - θ) ≤ r) (hy : circDist (y - ψ) ≤ s) :
    d - r - s < circDist (x - y) := by
  have h1 := circDist_triangle θ x ψ
  have h2 := circDist_triangle x y ψ
  have hsym : circDist (θ - x) = circDist (x - θ) := by
    rw [show θ - x = -(x - θ) by ring, circDist_neg]
  rw [hsym] at h1
  linarith

/-- Original unit temperature and actual positive semicircle distances jointly satisfy the circular interaction comparison hypotheses.
Source: arXiv:2411.04990v2, Appendix B and §5, the original force and curvature comparisons. -/
example : (0 : ℝ) < 1 ∧ (2 : ℝ) ∈ Icc 0 Real.pi ∧ interactionPeak 1 ≤ 2 ∧
    2 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ 2 ∧ (2 : ℝ) ≤ circDist 3 := by
  have hpi : (3 : ℝ) ≤ Real.pi := Real.pi_gt_three.le
  rw [circDist_eq_self 3 ⟨by norm_num, hpi⟩]
  have ht := interactionPeak_lt_scale 1 one_pos
  norm_num at ht ⊢
  exact ⟨by linarith, by linarith⟩

/-- Actual distinct original centers and their actual coincident tokens realize the original neighborhood-separation hypotheses jointly.
Source: arXiv:2411.04990v2, §5, the circular intercluster distance estimate in `thm:fixed_centers`. -/
example : (1 : ℝ) < circDist (0 - 2) ∧ circDist (0 - 0) ≤ (0 : ℝ) ∧
    circDist (2 - 2) ≤ (0 : ℝ) := by
  have htwo : circDist 2 = 2 := circDist_eq_self 2 ⟨by norm_num, by linarith [Real.pi_gt_three]⟩
  simp only [zero_sub, sub_self, circDist_neg, htwo, circDist_zero]
  norm_num

end Transformer.Causal
