/-
# An actual constant two-neuron attractor for every proper spherical cap

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
using the original projected perceptron in §1, `eq: vf`.

The actual trajectory first activates the inner neuron, then reaches the
forward invariant inner core, and subsequently contracts to the target.
The autonomous flow identity carries this tail convergence back to the
original initial point. All initially active particles converge, while
every point outside the outer cap is fixed for all real times.

The construction retains constant controls and uses two actual neurons.
It corrects the source's activation sign and replaces its insufficient
single-neuron argument for arbitrary targets in large caps. No small-cap
restriction is imposed: the chordal ball may have any radius up to `2`.
-/

import Transformer.Interpolation.Section3_CapContraction
import Transformer.Interpolation.Section3_CapEntry

open Real Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every active original trajectory converges to the target after its
finite-time entry into the inner core. Auxiliary for arXiv:2411.04551v3,
Appendix, the concentration step toward `lem: two.balls`. -/
theorem tendsto_twoRidge_cap_active (z w ω x₀ : SSphere d) (c h m σ ℓ : ℝ)
    (hc : -1 ≤ c) (hh : 0 ≤ h) (hm : 0 < m) (hσ : 0 < σ) (hℓ : 0 ≤ ℓ)
    (hb : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)
    (hσz : σ < inner ℝ (z : EucSpace d) (w : EucSpace d) - h)
    (hmargin : ∀ x : SSphere d, h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) →
      m ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hq : (h + σ) * ‖capTargetOutput z w ω c h ℓ‖ ≤
      inner ℝ (capTargetOutput z w ω c h ℓ) (w : EucSpace d))
    (hlarge : 2 * ‖(z : EucSpace d) -
        ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) • (w : EucSpace d)‖ + 1 ≤
        ℓ * σ * ((h + inner ℝ (ω : EucSpace d) (w : EucSpace d)) ^ 2 / 2) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h))
    (hactive : 0 < inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c) :
    Tendsto (twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) (capTargetOutput z w ω c h ℓ)
      (-(w : EucSpace d)) (-h) x₀) atTop (nhds ω) := by
  let q := capTargetOutput z w ω c h ℓ
  let Φ := twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c)
    q (-(w : EucSpace d)) (-h)
  obtain ⟨t₀, _, henter⟩ :=
    exists_time_twoRidge_inner_active z w x₀ q c h hc hactive (by linarith)
  obtain ⟨t₁, _, hcore⟩ := exists_time_twoRidge_cap_core z w (Φ x₀ t₀) q c h m σ
    hh hm hσ hσz hmargin hq (by linarith)
  have hlim := tendsto_twoRidge_cap_core z w ω (Φ (Φ x₀ t₀) t₁) c h m σ ℓ
    hh hm hσ hℓ hb hσz hmargin hq hlarge hcore
  have hs : Tendsto (fun t : ℝ => t + (-(t₀ + t₁))) atTop atTop :=
    tendsto_atTop_add_const_right atTop (-(t₀ + t₁)) tendsto_id
  have hl := hlim.comp hs
  convert hl using 1
  funext t
  change Φ x₀ t = Φ (Φ (Φ x₀ t₀) t₁) (t + (-(t₀ + t₁)))
  dsimp only [Φ]
  rw [← twoRidgeFlow_add, ← twoRidgeFlow_add]
  congr 1
  ring

/-- An explicit constant two-neuron field attracts every active point of
any proper cap to its prescribed target and fixes its complement pointwise.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem exists_twoRidge_cap_attractor (z ω : SSphere d) (c : ℝ) (hc : -1 ≤ c)
    (hω : c < inner ℝ (ω : EucSpace d) (z : EucSpace d)) :
    ∃ (w : SSphere d) (h ℓ : ℝ),
      (∀ x : SSphere d, c < inner ℝ (x : EucSpace d) (z : EucSpace d) →
        Tendsto (twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) (capTargetOutput z w ω c h ℓ)
          (-(w : EucSpace d)) (-h) x) atTop (nhds ω)) ∧
      (∀ x : SSphere d, ¬c < inner ℝ (x : EucSpace d) (z : EucSpace d) → ∀ t : ℝ,
        twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) (capTargetOutput z w ω c h ℓ)
          (-(w : EucSpace d)) (-h) x t = x) := by
  obtain ⟨w, h, m, hh, hz, hωw, hm, hmargin⟩ :=
    exists_nested_hemispherical_cap z ω c hc hω
  obtain ⟨σ, ℓ, hσ, hℓ, hσz, _, hq, hlarge⟩ := exists_capTarget_coefficients z w ω c h
    (sub_pos.mpr hω) (sub_pos.mpr hωw) hz hh
  refine ⟨w, h, ℓ, ?_, ?_⟩
  · intro x hx
    exact tendsto_twoRidge_cap_active z w ω x c h m σ ℓ hc hh hm hσ hℓ.le (sub_pos.mpr hωw)
      hσz hmargin hq hlarge (sub_pos.mpr hx)
  · intro x hx t
    have hi := inward_caps_inactive_outside z w c h (fun y hy => by
      have hmarg := hmargin y hy
      change c < inner ℝ (y : EucSpace d) (z : EucSpace d)
      linarith) x hx
    exact twoRidgeFlow_inactive _ _ _ _ _ _ x hi.1 hi.2 t

/-- Every proper chordal ball has an actual constant two-neuron attractor
at any of its points. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: two.balls`; radius `2` includes the sphere minus its antipode. -/
theorem exists_twoRidge_ball_attractor (z ω : SSphere d) (R : ℝ)
    (hR : 0 < R) (hR2 : R ≤ 2) (hω : ω ∈ Metric.ball z R) :
    ∃ (w : SSphere d) (h ℓ : ℝ),
      (∀ x ∈ Metric.ball z R,
        Tendsto (twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-(1 - R ^ 2 / 2))
          (capTargetOutput z w ω (1 - R ^ 2 / 2) h ℓ) (-(w : EucSpace d)) (-h) x) atTop (nhds ω)) ∧
      (∀ x : SSphere d, x ∉ Metric.ball z R → ∀ t : ℝ,
        twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-(1 - R ^ 2 / 2))
          (capTargetOutput z w ω (1 - R ^ 2 / 2) h ℓ) (-(w : EucSpace d)) (-h) x t = x) := by
  have hc : (-1 : ℝ) ≤ 1 - R ^ 2 / 2 := by nlinarith
  have hωc := (mem_spherical_ball_iff_inner z ω R hR).mp hω
  obtain ⟨w, h, ℓ, hactive, hinactive⟩ :=
    exists_twoRidge_cap_attractor z ω (1 - R ^ 2 / 2) hc hωc
  refine ⟨w, h, ℓ,
    fun x hx => hactive x ((mem_spherical_ball_iff_inner z x R hR).mp hx), ?_⟩
  intro x hx t
  exact hinactive x (fun hh => hx ((mem_spherical_ball_iff_inner z x R hR).mpr hh)) t

/-- The nested-cap and coefficient constructions jointly realize all
trajectory-convergence hypotheses with actual unit vectors and positive
margins; the outer cap and chosen active initial point are nonempty. -/
example : ∃ (z w ω x₀ : SSphere 2) (c h m σ ℓ : ℝ),
    -1 ≤ c ∧ 0 ≤ h ∧ 0 < m ∧ 0 < σ ∧ 0 ≤ ℓ ∧
    0 < inner ℝ (ω : EucSpace 2) (w : EucSpace 2) - h ∧
    σ < inner ℝ (z : EucSpace 2) (w : EucSpace 2) - h ∧
    (∀ x : SSphere 2, h ≤ inner ℝ (x : EucSpace 2) (w : EucSpace 2) →
      m ≤ inner ℝ (x : EucSpace 2) (z : EucSpace 2) - c) ∧
    (h + σ) * ‖capTargetOutput z w ω c h ℓ‖ ≤
      inner ℝ (capTargetOutput z w ω c h ℓ) (w : EucSpace 2) ∧
    (2 * ‖(z : EucSpace 2) -
      ((inner ℝ (ω : EucSpace 2) (z : EucSpace 2) - c) /
        (inner ℝ (ω : EucSpace 2) (w : EucSpace 2) - h)) • (w : EucSpace 2)‖ + 1 ≤
      ℓ * σ * ((h + inner ℝ (ω : EucSpace 2) (w : EucSpace 2)) ^ 2 / 2) /
        (inner ℝ (ω : EucSpace 2) (w : EucSpace 2) - h)) ∧
    0 < inner ℝ (x₀ : EucSpace 2) (z : EucSpace 2) - c := by
  let z : SSphere 2 := basePoint 1
  have hz : (0 : ℝ) < inner ℝ (z : EucSpace 2) (z : EucSpace 2) := by
    norm_num [z, basePoint]
  obtain ⟨w, h, m, hh, hzw, hωw, hm, hmargin⟩ :=
    exists_nested_hemispherical_cap z z 0 (by norm_num) hz
  obtain ⟨σ, ℓ, hσ, hℓ, hσz, _, hq, hlarge⟩ :=
    exists_capTarget_coefficients z w z 0 h (by simpa only [sub_zero] using hz)
      (sub_pos.mpr hωw) hzw hh
  exact ⟨z, w, z, z, 0, h, m, σ, ℓ, by norm_num, hh, hm, hσ, hℓ.le,
    sub_pos.mpr hωw, hσz, hmargin, hq, hlarge, by simpa only [sub_zero] using hz⟩

/-- A genuine ball and target satisfy the positive-radius, radius bound
and target-membership hypotheses jointly. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 ∧
    (basePoint 1 : SSphere 2) ∈ Metric.ball (basePoint 1) 1 := by
  norm_num

end Transformer.Interpolation
