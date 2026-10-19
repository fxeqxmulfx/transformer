module

public import Transformer.Causal.Section5_CircleGeometry
public import Transformer.Causal.InteractionPeak

/-!
# Actual vector interaction decays beyond the circle peak

ArXiv:2411.04990v2, §5, `lemma:meta`, `eq:c`, and `eq:bddotphi`;
Appendix A, `lemma:interaction` (3).

The metastability lemma assumes only `β > 0`, so its decreasing-potential
argument needs that full range, including `β < 1`. The previously proved
interaction-scale bounds require `β ≥ 1`; the decreasing branch itself does
not. We prove it directly from the exact peak in `eq:c`.

The actual projected vector interaction has norm `sin(arccos ⟪x,y⟫)`.
Chord distance is no larger than angular distance, so a lower chord gap
beyond the peak bounds the weighted vector interaction by the source's
`h(r)`. This transfers the angular bound to the actual spherical equation.
-/

@[expose] public section

open Real

namespace Transformer.Causal

/-- **The interaction decreases after its exact peak for every `β > 0`.**
The scale bounds of `lemma:interaction` (3) require `β ≥ 1`; this branch
does not. Source: arXiv:2411.04990v2, Appendix A, `lemma:interaction` (3),
and §5, `eq:c` in `lemma:meta`. -/
theorem h_pot_decreasing_after_peak (β : ℝ) (hβ : 0 < β) :
    StrictAntiOn (h_pot β)
      (Set.Icc (Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β))) π) := by
  let S := Real.sqrt (4 * β ^ 2 + 1)
  have hS2 : S ^ 2 = 4 * β ^ 2 + 1 := Real.sq_sqrt (by positivity)
  have hSβ : 2 * β < S := (Real.lt_sqrt (by positivity)).mpr (by nlinarith)
  have hS1 : 1 < S := (Real.lt_sqrt (by norm_num)).mpr (by nlinarith)
  have hSlt : S < 2 * β + 1 :=
    (Real.sqrt_lt (by positivity) (by positivity)).mpr (by nlinarith)
  let c := (-1 + S) / (2 * β)
  have hc0 : 0 < c := div_pos (by linarith) (by positivity)
  have hc1 : c < 1 := (div_lt_one (by positivity)).mpr (by linarith)
  have hroot : β * c ^ 2 + c - β = 0 := by
    have heq : β * c ^ 2 + c - β = (S ^ 2 - 1 - 4 * β ^ 2) / (4 * β) := by
      dsimp [c]
      field_simp
      ring
    rw [heq, hS2]
    ring
  have hbc : β * c = (S - 1) / 2 := by dsimp [c]; field_simp; ring
  have hs (u : ℝ) (hu : -1 ≤ u) : 0 < β * (u + c) + 1 := by
    nlinarith [mul_nonneg hβ.le (by linarith : 0 ≤ u + 1)]
  have hg (x : ℝ) : g_pot β x = Real.exp (β * (Real.cos x - 1)) *
      ((Real.cos x - c) * (β * (Real.cos x + c) + 1)) := by
    rw [g_pot]
    congr 1
    linear_combination (-β) * Real.sin_sq_add_cos_sq x + hroot
  have hτ : 0 < Real.arccos c := Real.arccos_pos.mpr hc1
  have hcos : Real.cos (Real.arccos c) = c := Real.cos_arccos (by linarith) hc1.le
  refine strictAntiOn_of_deriv_neg (convex_Icc _ _) (by
    exact (show Continuous (h_pot β) from by unfold h_pot; fun_prop).continuousOn) ?_
  intro x hx
  rw [interior_Icc] at hx
  rw [(hasDerivAt_h_pot β x).deriv, hg]
  have hcx : Real.cos x < c := by
    rw [← hcos]
    exact Real.cos_lt_cos_of_nonneg_of_le_pi hτ.le hx.2.le hx.1
  exact mul_neg_of_pos_of_neg (Real.exp_pos _)
    (mul_neg_of_neg_of_pos (by linarith) (hs _ (Real.neg_one_le_cos x)))

example : (0 : ℝ) < 1 / 2 := by norm_num

/-- The projected interaction norm is the sine of the actual spherical
angle. Source: arXiv:2411.04990v2, §5, `eq:bddotphi`. -/
theorem circle_norm_proj_eq_sin_arccos (x y : SSphere 2) :
    ‖proj 2 (x : EucSpace 2) (y : EucSpace 2)‖ =
      Real.sin (Real.arccos (inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2))) := by
  have hb := abs_le.mp (abs_circle_inner_le_one x y)
  have hc := Real.cos_arccos hb.1 hb.2
  have hs := Real.sin_sq_add_cos_sq
    (Real.arccos (inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)))
  rw [hc] at hs
  have hn : 0 ≤ Real.sin
      (Real.arccos (inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2))) :=
    Real.sin_nonneg_of_mem_Icc
    ⟨Real.arccos_nonneg _, Real.arccos_le_pi _⟩
  nlinarith [circle_norm_proj_sq x y, norm_nonneg (proj 2 (x : EucSpace 2) (y : EucSpace 2))]

/-- A circle chord is no longer than its actual spherical angle.
Source: arXiv:2411.04990v2, §5, the chord separation `eq:renyi`
and the angular comparison in the proof of `lemma:meta`. -/
theorem circle_chord_le_arccos (x y : SSphere 2) :
    ‖(x : EucSpace 2) - (y : EucSpace 2)‖ ≤
      Real.arccos (inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) := by
  have hb := abs_le.mp (abs_circle_inner_le_one x y)
  have hc := Real.cos_arccos hb.1 hb.2
  have hs := Real.one_sub_sq_div_two_le_cos
    (x := Real.arccos (inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)))
  rw [hc] at hs
  have hch := norm_sub_sq_real (x : EucSpace 2) (y : EucSpace 2)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2] at hch
  nlinarith [Real.arccos_nonneg
    (inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)), norm_nonneg ((x : EucSpace 2) - y)]

/-- Ordering circle inner products reverses the ordering of chord gaps.
Source: arXiv:2411.04990v2, §5, the closest-token comparison
following `eq:bddotphi`. -/
theorem circle_chord_le_of_inner_le (x y z : SSphere 2)
    (h : inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2) ≤
      inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) :
    ‖(x : EucSpace 2) - (y : EucSpace 2)‖ ≤ ‖(y : EucSpace 2) - (z : EucSpace 2)‖ := by
  have hxy := norm_sub_sq_real (x : EucSpace 2) (y : EucSpace 2)
  have hyz := norm_sub_sq_real (y : EucSpace 2) (z : EucSpace 2)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2] at hxy
  rw [mem_sphere_zero_iff_norm.mp y.2, mem_sphere_zero_iff_norm.mp z.2] at hyz
  nlinarith [norm_nonneg ((x : EucSpace 2) - y), norm_nonneg ((y : EucSpace 2) - z)]

example (x : SSphere 2) : inner (𝕜 := ℝ) (x : EucSpace 2) (x : EucSpace 2) ≤
    inner (𝕜 := ℝ) (x : EucSpace 2) (x : EucSpace 2) := le_rfl

/-- **The actual weighted vector interaction is bounded by `h(r)` beyond
the peak whenever the chord gap is at least `r`.**
Source: arXiv:2411.04990v2, §5, `eq:c` and `eq:bddotphi`.
The temperature range is the source's full `β > 0`. -/
theorem circle_weighted_projection_le (β : ℝ) (hβ : 0 < β) (r : ℝ)
    (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (x y : SSphere 2) (hgap : r ≤ ‖(x : EucSpace 2) - (y : EucSpace 2)‖) :
    Real.exp (β * (inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) - 1)) *
      ‖proj 2 (x : EucSpace 2) (y : EucSpace 2)‖ ≤ h_pot β r := by
  have hb := abs_le.mp (abs_circle_inner_le_one x y)
  have hcos := Real.cos_arccos hb.1 hb.2
  have hrθ := hgap.trans (circle_chord_le_arccos x y)
  have hmono := (h_pot_decreasing_after_peak β hβ).antitoneOn
    ⟨hrpeak, hrπ⟩ ⟨hrpeak.trans hrθ, Real.arccos_le_pi _⟩ hrθ
  rw [h_pot, hcos] at hmono
  rwa [circle_norm_proj_eq_sin_arccos]

/-- Every bound hypothesis is attained at `β = 1`, `r = 2` by the
antipodal unit tokens `(1,0)` and `(-1,0)`. -/
example : ∃ x y : SSphere 2, (0 : ℝ) < 1 ∧ (2 : ℝ) ≤ π ∧
    Real.arccos ((-1 + Real.sqrt (4 * (1 : ℝ) ^ 2 + 1)) / (2 * 1)) ≤ 2 ∧
    (2 : ℝ) ≤ ‖(x : EucSpace 2) - (y : EucSpace 2)‖ := by
  let x : SSphere 2 := basePoint 1
  let y : SSphere 2 := ⟨-(x : EucSpace 2), by
    simp only [mem_sphere_zero_iff_norm, norm_neg, mem_sphere_zero_iff_norm.mp x.2]⟩
  obtain ⟨τ, hτ0, hτπ, hcos, _, hτ1, _⟩ := h_pot_unimodal 1 le_rfl
  refine ⟨x, y, one_pos, by linarith [Real.pi_gt_three], ?_, ?_⟩
  · rw [← hcos, Real.arccos_cos hτ0.le hτπ.le]
    rw [Real.one_rpow] at hτ1
    linarith
  · change 2 ≤ ‖(x : EucSpace 2) - (-(x : EucSpace 2))‖
    have heq : (x : EucSpace 2) + (x : EucSpace 2) = (2 : ℝ) • (x : EucSpace 2) :=
      (two_smul ℝ (x : EucSpace 2)).symm
    rw [sub_neg_eq_add, heq, norm_smul, mem_sphere_zero_iff_norm.mp x.2]
    norm_num

end Transformer.Causal
