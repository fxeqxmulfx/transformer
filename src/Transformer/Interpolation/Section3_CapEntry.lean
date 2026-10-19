module

public import Transformer.Interpolation.Section3_CapCenter
public import Transformer.Interpolation.Section3_CapBoundary

/-!
# Entry into the contracting region of the actual two-neuron flow

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
with the actual autonomous particle equation from §1, `eq: ips`.

An active particle must eventually activate the second neuron. Otherwise
its trajectory agrees on every nonnegative interval with the center drift,
which converges to a point strictly inside the second activation cap.
This agreement follows from the original Lipschitz equation and Grönwall.

Once that activation is positive, the inward boundary drift keeps it
positive and forces it to reach the inner core in finite time. No bound
uniform in the starting particle is asserted or needed.
-/

@[expose] public section

open Real Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every active outer-cap particle eventually activates the second neuron.
The inner cap contains the center strictly. Auxiliary for arXiv:2411.04551v3,
Appendix, the entry step toward `lem: two.balls`. -/
theorem exists_time_twoRidge_inner_active (z w x₀ : SSphere d) (q : EucSpace d) (c h : ℝ)
    (hc : -1 ≤ c) (hactive : 0 < inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c)
    (hz : h < inner ℝ (z : EucSpace d) (w : EucSpace d)) :
    ∃ t : ℝ, 0 ≤ t ∧ h < inner ℝ
      (twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀ t : EucSpace d)
        (w : EucSpace d) := by
  by_contra hnot
  have hinactive (s : ℝ) (hs : 0 ≤ s) : inner ℝ
      (twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀ s : EucSpace d)
        (w : EucSpace d) ≤ h := by
    by_contra hh
    exact hnot ⟨s, hs, lt_of_not_ge hh⟩
  let X := twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀
  let Y := ridgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) x₀
  let F (x : EucSpace d) := max (-c - inner ℝ x (-(z : EucSpace d))) 0 • proj d x z
  have hX (s : ℝ) (hs : 0 ≤ s) : HasDerivAt (fun r => (X r : EucSpace d)) (F (X s)) s := by
    have hd := hasDerivAt_twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀ s
    change HasDerivAt (fun r => (X r : EucSpace d))
      (twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) (X s)) s at hd
    have hact : -h - inner ℝ (X s : EucSpace d) (-(w : EucSpace d)) ≤ 0 := by
      rw [inner_neg_right]
      have hb := hinactive s hs
      change inner ℝ (X s : EucSpace d) (w : EucSpace d) ≤ h at hb
      linarith
    simpa only [twoRidgeField, max_eq_right hact, zero_smul, add_zero, F] using hd
  have hY (s : ℝ) : HasDerivAt (fun r => (Y r : EucSpace d)) (F (Y s)) s :=
    hasDerivAt_ridgeFlow _ _ _ _ s
  obtain ⟨K, hK⟩ := exists_lipschitzOnWith_ridge_field (z : EucSpace d) (-(z : EucSpace d)) (-c)
  have heq (t : ℝ) (ht : 0 ≤ t) : X t = Y t := by
    have hd (s : ℝ) (hs : 0 ≤ s) := (hX s hs).sub (hY s)
    have hG := norm_le_gronwallBound_of_norm_deriv_right_le
      (a := 0) (b := t) (δ := 0) (K := (K : ℝ)) (ε := 0)
      (fun s hs => (hd s hs.1).continuousAt.continuousWithinAt)
      (fun s hs => (hd s hs.1).hasDerivWithinAt)
      (by simp [X, Y, twoRidgeFlow_zero, ridgeFlow_zero])
      (fun s _ => by
        simpa only [add_zero, F, Pi.sub_apply] using lipschitzOnWith_iff_norm_sub_le.mp hK (X s).2 (Y s).2)
      t ⟨ht, le_rfl⟩
    simp only [gronwallBound_ε0, zero_mul] at hG
    exact Subtype.ext (sub_eq_zero.mp (norm_le_zero_iff.mp hG))
  obtain ⟨t, ht, henter⟩ := exists_time_ridge_cap_center_enters z x₀ w c h hc hactive hz
  have hh := hinactive t ht.le
  change inner ℝ (X t : EucSpace d) (w : EucSpace d) ≤ h at hh
  rw [heq t ht.le] at hh
  exact (not_lt_of_ge hh) henter

/-- Positive drift throughout a bounded strip forces finite-time exit
through its upper boundary. Auxiliary for arXiv:2411.04551v3,
Appendix, the cap-entry step toward `lem: two.balls`. -/
theorem exists_time_scalar_reaches_strip (f f' : ℝ → ℝ) (σ γ : ℝ)
    (hσ : 0 < σ) (hγ : 0 < γ) (hf0 : 0 < f 0)
    (hd : ∀ s : ℝ, 0 ≤ s → HasDerivAt f (f' s) s)
    (hband : ∀ s : ℝ, 0 ≤ s → 0 ≤ f s → f s ≤ σ → γ ≤ f' s) :
    ∃ t : ℝ, 0 ≤ t ∧ σ ≤ f t := by
  by_contra hnot
  have hsmall (s : ℝ) (hs : 0 ≤ s) : f s < σ :=
    lt_of_not_ge (fun hh => hnot ⟨s, hs, hh⟩)
  let δ := min (f 0) σ
  have hδ : 0 < δ := lt_min hf0 hσ
  have hδσ : δ ≤ σ := min_le_right _ _
  have hpos (s : ℝ) (hs : 0 ≤ s) : 0 ≤ f s := by
    have hh := scalar_lower_barrier f f' δ hd (min_le_left _ _)
      (fun r hr he => hγ.trans_le (hband r hr (by linarith) (by linarith))) s hs
    linarith
  have hderiv (s : ℝ) (hs : 0 < s) : γ ≤ deriv f s := by
    rw [(hd s hs.le).deriv]
    exact hband s hs.le (hpos s hs.le) (hsmall s hs.le).le
  have hlin : ∀ x ∈ Ici (0 : ℝ), ∀ y ∈ Ici (0 : ℝ),
      x ≤ y → γ * (y - x) ≤ f y - f x := by
    intro x hx y hy hxy
    exact (convex_Ici (0 : ℝ)).mul_sub_le_image_sub_of_le_deriv
      (fun s hs => (hd s hs).continuousAt.continuousWithinAt)
      (fun s hs => (hd s (show 0 < s by simpa only [interior_Ici, mem_Ioi] using hs).le).differentiableAt.differentiableWithinAt)
      (fun s hs => hderiv s (by simpa only [interior_Ici, mem_Ioi] using hs)) x hx y hy hxy
  let T := (σ - f 0 + 1) / γ
  have hT : 0 ≤ T := div_nonneg (by have hh := hsmall 0 le_rfl; linarith) hγ.le
  have hg := hlin 0 (mem_Ici.mpr le_rfl) T hT hT
  have he : γ * T = σ - f 0 + 1 := by dsimp only [T]; field_simp
  rw [sub_zero, he] at hg
  have hh := hsmall T hT
  linarith

/-- An actually activated particle reaches the inner core in finite time.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`;
the actual field supplies the derivative and the uniform strip drift. -/
theorem exists_time_twoRidge_cap_core (z w x₀ : SSphere d) (q : EucSpace d) (c h m σ : ℝ)
    (hh : 0 ≤ h) (hm : 0 < m) (hσ : 0 < σ)
    (hσz : σ < inner ℝ (z : EucSpace d) (w : EucSpace d) - h)
    (hmargin : ∀ x : SSphere d, h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) →
      m ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hq : (h + σ) * ‖q‖ ≤ inner ℝ q (w : EucSpace d))
    (h0 : 0 < inner ℝ (x₀ : EucSpace d) (w : EucSpace d) - h) :
    ∃ t : ℝ, 0 ≤ t ∧ σ ≤ inner ℝ
      (twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀ t : EucSpace d)
      (w : EucSpace d) - h := by
  let X := twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀
  apply exists_time_scalar_reaches_strip _
    (fun s => inner ℝ (twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c) q
      (-(w : EucSpace d)) (-h) (X s)) (w : EucSpace d)) σ
    (m * (inner ℝ (z : EucSpace d) (w : EucSpace d) - h - σ)) hσ
    (mul_pos hm (by linarith)) ?_
    (fun s _ => hasDerivAt_twoRidge_cap_activation _ _ _ _ _ _ _ _ _ _) ?_
  · simpa only [X, twoRidgeFlow_zero] using h0
  · intro s _ hs hsσ
    exact twoRidge_cap_boundary_drift z w (X s) q c h m σ hh hm hσz
      (hmargin (X s) (by linarith)) hs hsσ hq

/-- The scalar strip hypotheses hold for a genuine increasing trajectory. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 0 < (fun t : ℝ => t + 1 / 2) 0 ∧
    (∀ s : ℝ, 0 ≤ s → HasDerivAt (fun t : ℝ => t + 1 / 2) 1 s) ∧
    (∀ s : ℝ, 0 ≤ s → 0 ≤ s + 1 / 2 → s + 1 / 2 ≤ 1 → (1 : ℝ) ≤ 1) := by
  exact ⟨zero_lt_one, zero_lt_one, by norm_num,
    fun s _ => (hasDerivAt_id s).add_const (1 / 2), fun _ _ _ _ => le_rfl⟩

/-- Actual unit axes and a nonzero output satisfy the entry and core
hypotheses jointly, including the positive initial activation. -/
example : ∃ (z w x₀ : SSphere 2) (q : EucSpace 2),
    (-1 : ℝ) ≤ 0 ∧ 0 < inner ℝ (x₀ : EucSpace 2) (z : EucSpace 2) - 0 ∧
    (1 / 2 : ℝ) < inner ℝ (z : EucSpace 2) (w : EucSpace 2) ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 4 ∧
    (1 / 4 : ℝ) < inner ℝ (z : EucSpace 2) (w : EucSpace 2) - 1 / 2 ∧
    (∀ x : SSphere 2, (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (w : EucSpace 2) →
      (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (z : EucSpace 2) - 0) ∧
    (1 / 2 + 1 / 4 : ℝ) * ‖q‖ ≤ inner ℝ q (w : EucSpace 2) ∧
    0 < inner ℝ (x₀ : EucSpace 2) (w : EucSpace 2) - 1 / 2 := by
  let z : SSphere 2 := basePoint 1
  refine ⟨z, z, z, (2 : ℝ) • (z : EucSpace 2), by norm_num, ?_, ?_,
    by norm_num, by norm_num, by norm_num, ?_, ?_, ?_, ?_⟩
  · norm_num [z, basePoint]
  · norm_num [z, basePoint]
  · norm_num [z, basePoint]
  · intro x hx
    simpa only [sub_zero] using hx
  · norm_num [norm_smul, real_inner_smul_left, z, basePoint, EuclideanSpace.inner_single_left]
  · norm_num [z, basePoint]

end Transformer.Interpolation
