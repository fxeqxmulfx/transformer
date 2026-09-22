/-
# Measure-to-measure interpolation — the separation flow settles exponentially

Step 2 of the proof of the interpolation theorem of arXiv:2411.04551v3 (§4)
freezes the perceptron at `𝐔_1 = 𝟏 γᵀ`, `b_1 = -(ε/2) 𝟏`, `𝐖_1 𝟏 = ω_+`, so
that `eq: neural.ode.sphere` becomes `eq: neural.ode.separation`,

  `ẋ(t) = (⟨γ, x(t)⟩ - ε/2)_+ Proj_{x(t)} ω_+`,

and `eq: Hartman.Grobman` asserts that every trajectory started in
`𝒮_+ = {x ∈ 𝕊^{d-1} : ⟨γ, x⟩ ≥ ε}` settles onto `ω_+ ∈ 𝒮_+`,

  `d_g(x(t), ω_+) ≤ K e^{-λ t}`,   `t ≥ 0`,   `λ > 0`, `K ≥ 1`.

This file proves it from the equation alone.  The argument is not the paper's
linearization but the three invariants the paper's geodesic picture rests on,
each obtained by a fencing argument (`nonpos_of_deriv_le_mul`):

* the sphere is invariant, `‖x(t)‖ = 1`;
* `u = ⟨x, ω_+⟩` does not decrease, `u̇ = (⟨γ, x⟩ - ε/2)_+ (1 - u²) ≥ 0`;
* `𝒮_+` is invariant, and more: `⟨γ, x(t)⟩ ≥ min(⟨γ, x_0⟩, ⟨γ, ω_+⟩) ≥ ε`.

Then the gate is open at least `ε/2`, `1 + u ≥ 1 + u_0 > 0` because `x_0` and
`ω_+` cannot be antipodal inside `𝒮_+`, and the field contracts towards `ω_+`
at the rate `λ = ε (1 + u_0)/4` — the hypothesis of
`norm_sub_le_of_contraction`, here proved.  The geodesic distance
`arccos ⟨x, ω_+⟩` is at most `π/2` times the chordal one (Jordan's
inequality).

Source: arXiv:2411.04551v3, §4, proof of the main theorem, Step 2,
`eq: neural.ode.separation`, `eq: estimate.neural`, `eq: Hartman.Grobman`.
-/

import Transformer.Interpolation.SeparationField

open Real

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- **Equation (eq: Hartman.Grobman).** *Exponential settling of the separation
flow.*

Let `‖γ‖ = 1`, `ε > 0`, and `ω_+` a unit vector of
`𝒮_+ = {⟨γ, ·⟩ ≥ ε}`.  Every solution of `eq: neural.ode.separation` started
at a unit vector `x_0 ∈ 𝒮_+` satisfies

  `d_g(x(t), ω_+) = arccos ⟨x(t), ω_+⟩ ≤ K e^{-λ t}`   for all `t ≥ 0`,

for some `λ > 0` and `K ≥ 1`.  The trajectory is not assumed to stay on the
sphere or in `𝒮_+`: both are proved (see the module header).

**What the source says and what is changed here.**  The paper sets
`ε := min{ε, π/4}`; the bound `π/4` is not used and is dropped, so the
statement covers every `ε > 0`.  The paper obtains the rate from the
Hartman-Grobman theorem; here it is explicit, `λ = ε (1 + ⟨x_0, ω_+⟩)/4`, and
`K = max(1, (π/2) ‖x_0 - ω_+‖)`.

Source: arXiv:2411.04551v3, §4, Step 2, `eq: Hartman.Grobman`. -/
theorem Hartman_Grobman (γ ω_plus : EucSpace d) (ε : ℝ) (hε : 0 < ε)
    (hγ : ‖γ‖ = 1) (hω : ‖ω_plus‖ = 1) (hωS : ε ≤ inner (𝕜 := ℝ) γ ω_plus)
    (x : ℝ → EucSpace d) (hx0 : ‖x 0‖ = 1) (hx0S : ε ≤ inner (𝕜 := ℝ) γ (x 0))
    (hflow : ∀ t : ℝ, 0 ≤ t → HasDerivAt x (separationField d γ ω_plus ε (x t)) t) :
    ∃ K lam : ℝ, 1 ≤ K ∧ 0 < lam ∧ ∀ t : ℝ, 0 ≤ t →
      Real.arccos (inner (𝕜 := ℝ) (x t) ω_plus) ≤ K * Real.exp (-(lam * t)) := by
  set F : ℝ → ℝ := fun s => max (inner (𝕜 := ℝ) γ (x s) - ε / 2) 0 with hF
  set u : ℝ → ℝ := fun s => inner (𝕜 := ℝ) (x s) ω_plus with hu
  set g : ℝ → ℝ := fun s => inner (𝕜 := ℝ) γ (x s) with hg
  have hωω : inner (𝕜 := ℝ) ω_plus ω_plus = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, hω]; norm_num
  have hF0 : ∀ s, 0 ≤ F s := fun s => le_max_right _ _
  -- the sphere is invariant
  have hsph : ∀ T : ℝ, 0 ≤ T → ‖x T‖ = 1 := by
    intro T hT
    have hcont : ContinuousOn x (Set.Icc 0 T) :=
      fun s hs => (hflow s hs.1).continuousAt.continuousWithinAt
    obtain ⟨M, hM⟩ := isCompact_Icc.exists_bound_of_continuousOn (f := fun s => F s * u s)
      (ContinuousOn.mul (ContinuousOn.sup' ((continuousOn_const.inner hcont).sub
        continuousOn_const) continuousOn_const) (hcont.inner continuousOn_const))
    have hd : ∀ s ∈ Set.Icc (0 : ℝ) T,
        HasDerivAt (fun r => (1 - inner (𝕜 := ℝ) (x r) (x r)) ^ 2)
          (-4 * (F s * u s) * (1 - inner (𝕜 := ℝ) (x s) (x s)) ^ 2) s := by
      intro s hs
      have h := (((hflow s hs.1).inner ℝ (hflow s hs.1)).const_sub 1).pow 2
      refine h.congr_deriv ?_
      rw [real_inner_comm (separationField d γ ω_plus ε (x s)) (x s),
        inner_separationField, real_inner_comm (x s) ω_plus]
      simp only [hF, hu]
      push_cast
      ring
    have hφ := nonpos_of_deriv_le_mul _ _ (4 * M) T hd
      (by rw [real_inner_self_eq_norm_sq, hx0]; norm_num)
      (fun s hs _ => by
        have hb := hM s (Set.Ico_subset_Icc_self hs)
        rw [Real.norm_eq_abs] at hb
        nlinarith [neg_abs_le (F s * u s),
          sq_nonneg (1 - inner (𝕜 := ℝ) (x s) (x s))])
      T ⟨hT, le_rfl⟩
    have h1 : inner (𝕜 := ℝ) (x T) (x T) = (1 : ℝ) := by
      nlinarith [sq_nonneg (1 - inner (𝕜 := ℝ) (x T) (x T))]
    rw [real_inner_self_eq_norm_sq] at h1
    nlinarith [norm_nonneg (x T)]
  -- the elementary bounds on the sphere
  have hu1 : ∀ s, 0 ≤ s → |u s| ≤ 1 := fun s hs => by
    have := abs_real_inner_le_norm (x s) ω_plus
    rwa [hsph s hs, hω, one_mul] at this
  have hg1 : ∀ s, 0 ≤ s → |g s| ≤ 1 := fun s hs => by
    have := abs_real_inner_le_norm γ (x s)
    rwa [hsph s hs, hγ, one_mul] at this
  have hF1 : ∀ s, 0 ≤ s → F s ≤ 1 := fun s hs =>
    max_le (by linarith [(abs_le.mp (hg1 s hs)).2]) zero_le_one
  have hxx : ∀ s, 0 ≤ s → inner (𝕜 := ℝ) (x s) (x s) = (1 : ℝ) := fun s hs => by
    rw [real_inner_self_eq_norm_sq, hsph s hs]; norm_num
  -- `eq: estimate.neural`: `u̇ = F (1 - u²)`, and `ġ = F (⟨γ, ω_+⟩ - u g)`
  have hdu : ∀ s, 0 ≤ s → HasDerivAt u (F s * (1 - u s ^ 2)) s := fun s hs => by
    refine ((hflow s hs).inner ℝ (hasDerivAt_const s ω_plus)).congr_deriv ?_
    rw [inner_zero_right, zero_add, inner_separationField, hωω]
    simp only [hF, hu]
    ring
  have hdg : ∀ s, 0 ≤ s →
      HasDerivAt g (F s * (inner (𝕜 := ℝ) γ ω_plus - u s * g s)) s := fun s hs => by
    refine ((hasDerivAt_const s γ).inner ℝ (hflow s hs)).congr_deriv ?_
    rw [inner_zero_left, add_zero, real_inner_comm _ γ, inner_separationField,
      real_inner_comm γ ω_plus, real_inner_comm γ (x s)]
  -- `u` does not decrease
  have hmono : ∀ t, 0 ≤ t → u 0 ≤ u t := fun t ht => by
    have h := nonpos_of_deriv_le_mul (fun s => u 0 - u s) (fun s => -(F s * (1 - u s ^ 2)))
      0 t (fun s hs => (hdu s hs.1).const_sub (u 0)) (by simp)
      (fun s hs _ => by
        have h1 : u s ^ 2 ≤ 1 := by
          have := abs_le.mp (hu1 s hs.1); nlinarith
        nlinarith [mul_nonneg (hF0 s) (sub_nonneg.mpr h1)])
      t ⟨ht, le_rfl⟩
    linarith
  -- `𝒮_+` is invariant: `⟨γ, x⟩` stays above `m = min(⟨γ, x_0⟩, ⟨γ, ω_+⟩) ≥ ε`
  set m := min (g 0) (inner (𝕜 := ℝ) γ ω_plus) with hm
  have hmε : ε ≤ m := le_min hx0S hωS
  have hgm : ∀ t, 0 ≤ t → m ≤ g t := fun t ht => by
    have h := nonpos_of_deriv_le_mul (fun s => m - g s)
      (fun s => -(F s * (inner (𝕜 := ℝ) γ ω_plus - u s * g s)))
      1 t (fun s hs => (hdg s hs.1).const_sub m) (by simp [hm])
      (fun s hs hpos => by
        have hmω : m ≤ inner (𝕜 := ℝ) γ ω_plus := min_le_right _ _
        have hus := abs_le.mp (hu1 s hs.1)
        have ha : 0 ≤ inner (𝕜 := ℝ) γ ω_plus - u s * m := by nlinarith
        have hFu : |F s * u s| ≤ 1 := by
          rw [abs_mul, abs_of_nonneg (hF0 s)]
          nlinarith [hF1 s hs.1, hF0 s, abs_nonneg (u s), hu1 s hs.1]
        have hb : F s * u s * (g s - m) ≤ 1 * (m - g s) := by
          nlinarith [neg_abs_le (F s * u s), le_abs_self (F s * u s)]
        nlinarith [mul_nonneg (hF0 s) ha])
      t ⟨ht, le_rfl⟩
    linarith
  -- `x_0` and `ω_+` are not antipodal: `(⟨γ, x_0⟩ + ⟨γ, ω_+⟩)² ≤ ‖x_0 + ω_+‖² = 2 + 2 u_0`
  have hu0 : 0 < 1 + u 0 := by
    have hcs := abs_real_inner_le_norm γ (x 0 + ω_plus)
    rw [hγ, one_mul, inner_add_right] at hcs
    have hsq : ‖x 0 + ω_plus‖ ^ 2 = 2 + 2 * u 0 := by
      rw [@norm_add_sq_real, hx0, hω]
      simp only [hu]
      ring
    have hpos : 0 < inner (𝕜 := ℝ) γ (x 0) + inner (𝕜 := ℝ) γ ω_plus := by linarith
    rw [abs_of_pos hpos] at hcs
    nlinarith [norm_nonneg (x 0 + ω_plus)]
  -- the field contracts towards `ω_+` at the rate `λ = ε (1 + u_0)/4`
  set lam := ε * (1 + u 0) / 4 with hlam
  have hlam0 : 0 < lam := by positivity
  have hcontr : ∀ t, 0 ≤ t →
      inner (𝕜 := ℝ) (separationField d γ ω_plus ε (x t)) (x t - ω_plus)
        ≤ -(lam * ‖x t - ω_plus‖ ^ 2) := fun t ht => by
    have hsq : ‖x t - ω_plus‖ ^ 2 = 2 - 2 * u t := by
      rw [@norm_sub_sq_real, hsph t ht, hω]
      simp only [hu]
      ring
    have hin : inner (𝕜 := ℝ) (separationField d γ ω_plus ε (x t)) (x t - ω_plus)
        = -(F t * (1 - u t) * (1 + u t)) := by
      rw [inner_separationField, inner_sub_right, inner_sub_right, hxx t ht, hωω,
        real_inner_comm (x t) ω_plus]
      simp only [hF, hu]
      ring
    have hFε : ε / 2 ≤ F t := le_max_of_le_left (by linarith [hgm t ht])
    have hut := abs_le.mp (hu1 t ht)
    have hmt := hmono t ht
    have h2 : ε / 2 * (1 + u 0) ≤ F t * (1 + u t) :=
      mul_le_mul hFε (by linarith) hu0.le (hF0 t)
    rw [hin, hsq, hlam]
    nlinarith [mul_le_mul_of_nonneg_right h2 (by linarith : (0 : ℝ) ≤ 1 - u t)]
  have hN := norm_sub_le_of_contraction d x (fun s => separationField d γ ω_plus ε (x s))
    ω_plus lam hflow hcontr
  refine ⟨max 1 (π / 2 * ‖x 0 - ω_plus‖), lam, le_max_left _ _, hlam0, fun t ht => ?_⟩
  calc Real.arccos (inner (𝕜 := ℝ) (x t) ω_plus) ≤ π / 2 * ‖x t - ω_plus‖ :=
        arccos_inner_le_norm_sub d (hsph t ht) hω
    _ ≤ π / 2 * (‖x 0 - ω_plus‖ * Real.exp (-(lam * t))) := by
        gcongr
        exact hN t ht
    _ ≤ max 1 (π / 2 * ‖x 0 - ω_plus‖) * Real.exp (-(lam * t)) := by
        rw [← mul_assoc]
        gcongr
        exact le_max_right _ _

/-- The hypotheses of `Hartman_Grobman` are satisfiable: in `ℝ^1`, `γ = ω_+ =
x_0 = e_1` and `ε = 1`, where the field vanishes and the trajectory stands at
`ω_+`. -/
example :
    let e : EucSpace 1 := ((basePoint 0 : SSphere 1) : EucSpace 1)
    (0 : ℝ) < 1 ∧ ‖e‖ = 1 ∧ (1 : ℝ) ≤ inner (𝕜 := ℝ) e e ∧
      (∀ t : ℝ, 0 ≤ t → HasDerivAt (fun _ : ℝ => e) (separationField 1 e e 1 e) t) := by
  intro e
  have he : ‖e‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hee : inner (𝕜 := ℝ) e e = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, he]; norm_num
  refine ⟨one_pos, he, hee.ge, fun t _ => ?_⟩
  have h0 : separationField 1 e e 1 e = 0 := by
    rw [separationField, proj, hee, one_smul, sub_self, smul_zero]
  rw [h0]
  exact hasDerivAt_const t e

end Interpolation
end Transformer
