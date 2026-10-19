module

public import Transformer.Normalization.Section2_SphereInvariant
public import Transformer.Normalization.Velocities
public import Mathlib.Analysis.Calculus.Deriv.MeanValue
public import Mathlib.Order.LiminfLimsup

/-!
# Normalization — radial bounds and their lower limit

For Pre-LN and Peri-LN, the corollary to `thm:convergence` in
arXiv:2510.22026v2, §3, excludes stalling when `n ≤ exp β`. The existing
Pre-LN bound extends to Peri-LN because `‖A_j‖ ≤ 1`. Both radial velocities
are also at most one, by Cauchy–Schwarz.

The upper bound makes the real-valued lower limit well-defined in the
sense needed by Mathlib's conditional completeness lemmas. The positive
lower bound then gives a strictly positive `liminf` along each actual
trajectory, using the sphere invariant rather than a unit path assumption.
Integration also gives `r_j(t) ≥ r_j(0) + t/(n exp β)`.
-/

@[expose] public section

open scoped BigOperators
open Real Set Filter

namespace Transformer.Normalization

/-- Both actual radial velocities lie between `(n exp β)⁻¹` and one.
For Peri-LN the lower bound also proves that its attention denominator is
nonzero. Source: arXiv:2510.22026v2, §3, proof of the corollary to
`thm:convergence`, and Table 2. -/
theorem radialDerivative_pre_peri_bounds {d n : ℕ}
    (β : ℝ) (hβ : 0 < β) (hn : (n : ℝ) ≤ Real.exp β)
    (θ : ℝ → Idx n → EucSpace d) (t τ : ℝ)
    (hθ : ∀ l, ‖θ t l‖ = 1) (scheme : Scheme)
    (hscheme : scheme = .pre ∨ scheme = .peri) (j : Idx n) :
    ((n : ℝ) * Real.exp β)⁻¹ ≤
        radialDerivative d n β (idParams d) (idParams d) (idParams d) θ τ scheme t j ∧
      radialDerivative d n β (idParams d) (idParams d) (idParams d) θ τ scheme t j ≤ 1 := by
  let A := attentionVec d n β (idParams d t) (idParams d t) (idParams d t) (θ t) j
  have hpre : ((n : ℝ) * Real.exp β)⁻¹ ≤ inner (𝕜 := ℝ) (θ t j) A :=
    radialDerivative_pre_lower_bound d n β hβ θ t τ hθ hn j
  have hA : ‖A‖ ≤ 1 := norm_attentionVec_le_one d n β
    (idParams d t) (idParams d t) (idParams d t) ContinuousLinearMap.norm_id_le (θ t) hθ j
  have hdot : inner (𝕜 := ℝ) (θ t j) A ≤ ‖A‖ := by
    simpa only [hθ j, one_mul] using real_inner_le_norm (θ t j) A
  have hnpos : (0 : ℝ) < n := by exact_mod_cast Fin.pos j
  have hc : (0 : ℝ) < ((n : ℝ) * Real.exp β)⁻¹ := by positivity
  rcases hscheme with rfl | rfl
  · exact ⟨hpre, hdot.trans hA⟩
  · change ((n : ℝ) * Real.exp β)⁻¹ ≤ inner (𝕜 := ℝ) (θ t j) A / ‖A‖ ∧
        inner (𝕜 := ℝ) (θ t j) A / ‖A‖ ≤ 1
    have hApos : 0 < ‖A‖ := hc.trans_le (hpre.trans hdot)
    refine ⟨(le_div_iff₀ hApos).mpr ?_, (div_le_one hApos).mpr hdot⟩
    calc ((n : ℝ) * Real.exp β)⁻¹ * ‖A‖
        ≤ ((n : ℝ) * Real.exp β)⁻¹ * 1 := mul_le_mul_of_nonneg_left hA hc.le
      _ = ((n : ℝ) * Real.exp β)⁻¹ := mul_one _
      _ ≤ inner (𝕜 := ℝ) (θ t j) A := hpre

/-- The pointwise hypotheses hold for a unit singleton at `β = 1`. -/
example (d : ℕ) : (0 : ℝ) < 1 ∧ ((1 : ℕ) : ℝ) ≤ Real.exp 1 ∧
    (∀ j : Idx 1, ‖(fun _ _ => (basePoint d : EucSpace (d + 1))) (0 : ℝ) j‖ = 1) ∧
    (Scheme.peri = .pre ∨ Scheme.peri = .peri) :=
  ⟨one_pos, by simp, fun _ => by simp [basePoint, PiLp.norm_single], Or.inr rfl⟩

/-- The radial lower limit is strictly positive on each actual trajectory.
The eventual upper bound one supplies coboundedness, so the conclusion
does not use a default value of a conditionally complete lower limit.
Source: arXiv:2510.22026v2, §3, corollary to `thm:convergence`. -/
theorem SchemeDynamics.radial_liminf_pos {d n : ℕ} {β τ : ℝ}
    {α : ℝ → ℝ} {scheme : Scheme} {θ : ℝ → Idx n → EucSpace d} {r : ℝ → Idx n → ℝ}
    (hD : SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ scheme θ r)
    (hβ : 0 < β) (hn : (n : ℝ) ≤ Real.exp β)
    (hscheme : scheme = .pre ∨ scheme = .peri) (h0 : ∀ j, ‖θ 0 j‖ = 1) (j : Idx n) :
    0 < Filter.liminf
      (fun t => radialDerivative d n β (idParams d) (idParams d) (idParams d)
        θ τ scheme t j) atTop := by
  let f := fun t => radialDerivative d n β (idParams d) (idParams d) (idParams d)
    θ τ scheme t j
  have hb (t : ℝ) (ht : 0 ≤ t) : ((n : ℝ) * Real.exp β)⁻¹ ≤ f t ∧ f t ≤ 1 :=
    radialDerivative_pre_peri_bounds β hβ hn θ t τ
      (hD.norm_eq_one hscheme h0 t ht) scheme hscheme j
  have hup : ∀ᶠ t in atTop, f t ≤ 1 :=
    (eventually_ge_atTop (0 : ℝ)).mono fun t ht => (hb t ht).2
  have hlow : ∀ᶠ t in atTop, ((n : ℝ) * Real.exp β)⁻¹ ≤ f t :=
    (eventually_ge_atTop (0 : ℝ)).mono fun t ht => (hb t ht).1
  have hlim := le_liminf_of_le (isCoboundedUnder_ge_of_eventually_le atTop hup) hlow
  have hnpos : (0 : ℝ) < n := by exact_mod_cast Fin.pos j
  exact (show (0 : ℝ) < ((n : ℝ) * Real.exp β)⁻¹ by positivity).trans_le hlim

/-- The actual Pre-LN singleton realizes every lower-limit hypothesis. -/
example (d : ℕ) :
    SchemeDynamics (d + 1) 1 1 (idParams (d + 1)) (idParams (d + 1)) (idParams (d + 1))
      0 0 .pre (fun _ _ => (basePoint d : EucSpace (d + 1))) (fun t _ => 1 + t) ∧
    (0 : ℝ) < 1 ∧ ((1 : ℕ) : ℝ) ≤ Real.exp 1 ∧
    (Scheme.pre = .pre ∨ Scheme.pre = .peri) ∧
    (∀ j : Idx 1, ‖(fun _ _ => (basePoint d : EucSpace (d + 1))) (0 : ℝ) j‖ = 1) := by
  have hx : ‖(basePoint d : EucSpace (d + 1))‖ = 1 := by
    simp [basePoint, PiLp.norm_single]
  exact ⟨schemeDynamics_singleton 1 1 0 0 _ hx .pre (Or.inl rfl), one_pos,
    by simp, Or.inl rfl, fun _ => hx⟩

/-- Integrating the radial lower bound yields linear growth, even when the
direction variables are supplied as ambient curves.
Source: arXiv:2510.22026v2, §3, the bound in the proof of the corollary to
`thm:convergence`. -/
theorem SchemeDynamics.radius_ge_initial {d n : ℕ} {β τ : ℝ}
    {α : ℝ → ℝ} {scheme : Scheme} {θ : ℝ → Idx n → EucSpace d} {r : ℝ → Idx n → ℝ}
    (hD : SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ scheme θ r)
    (hβ : 0 < β) (hn : (n : ℝ) ≤ Real.exp β)
    (hscheme : scheme = .pre ∨ scheme = .peri) (h0 : ∀ j, ‖θ 0 j‖ = 1)
    (t : ℝ) (ht : 0 ≤ t) (j : Idx n) :
    r 0 j + ((n : ℝ) * Real.exp β)⁻¹ * t ≤ r t j := by
  let c := ((n : ℝ) * Real.exp β)⁻¹
  let f := fun u => r u j - c * u
  let v := fun u => radialDerivative d n β (idParams d) (idParams d) (idParams d)
    θ τ scheme u j - c
  have hf (u : ℝ) (hu : 0 ≤ u) : HasDerivWithinAt f (v u) (Ici 0) u := by
    simpa only [f, v, mul_one, id_eq] using
      (hD u hu j).2.fun_sub ((hasDerivAt_id u).const_mul c).hasDerivWithinAt
  have hv (u : ℝ) (hu : 0 ≤ u) : 0 ≤ v u := by
    have hb := radialDerivative_pre_peri_bounds β hβ hn θ u τ
      (hD.norm_eq_one hscheme h0 u hu) scheme hscheme j
    exact sub_nonneg.mpr hb.1
  have hm : MonotoneOn f (Ici 0) := monotoneOn_of_hasDerivWithinAt_nonneg
    (convex_Ici (0 : ℝ)) (fun u hu => (hf u hu).continuousWithinAt)
    (fun u hu => (hf u (interior_subset hu)).mono interior_subset)
    (fun u hu => hv u (interior_subset hu))
  have hle : f 0 ≤ f t := hm (by simp) ht ht
  dsimp [f] at hle
  linarith

/-- The Peri-LN singleton satisfies all linear-growth hypotheses. -/
example (d : ℕ) :
    SchemeDynamics (d + 1) 1 1 (idParams (d + 1)) (idParams (d + 1)) (idParams (d + 1))
      0 0 .peri (fun _ _ => (basePoint d : EucSpace (d + 1))) (fun t _ => 1 + t) ∧
    (0 : ℝ) < 1 ∧ ((1 : ℕ) : ℝ) ≤ Real.exp 1 ∧
    (Scheme.peri = .pre ∨ Scheme.peri = .peri) ∧
    (∀ j : Idx 1, ‖(fun _ _ => (basePoint d : EucSpace (d + 1))) (0 : ℝ) j‖ = 1) ∧
    (0 : ℝ) ≤ 1 := by
  have hx : ‖(basePoint d : EucSpace (d + 1))‖ = 1 := by
    simp [basePoint, PiLp.norm_single]
  exact ⟨schemeDynamics_singleton 1 1 0 0 _ hx .peri (Or.inr rfl), one_pos,
    by simp, Or.inr rfl, fun _ => hx, zero_le_one⟩

/-- In either scheme, every singleton direction is constant in forward
time. This also gives a non-vacuous witness for a clustering hypothesis.
Source: arXiv:2510.22026v2, §3, `thm:convergence`, the one-token case. -/
theorem SchemeDynamics.direction_constant_singleton {d : ℕ} {β τ : ℝ}
    {α : ℝ → ℝ} {scheme : Scheme} {θ : ℝ → Idx 1 → EucSpace d} {r : ℝ → Idx 1 → ℝ}
    (hD : SchemeDynamics d 1 β (idParams d) (idParams d) (idParams d) α τ scheme θ r)
    (hscheme : scheme = .pre ∨ scheme = .peri) (h0 : ∀ j, ‖θ 0 j‖ = 1)
    (t : ℝ) (ht : 0 ≤ t) (j : Idx 1) : θ t j = θ 0 j := by
  apply eq_initial_of_hasDerivWithinAt_zero (fun u => θ u j) ?_ t ht
  intro u hu
  have hA : attentionVec d 1 β (idParams d u) (idParams d u) (idParams d u) (θ u) j =
      θ u j := by simpa [idParams] using
        attentionVec_singleton β (idParams d u) (idParams d u) (idParams d u) (θ u) j
  have hp : proj d (θ u j) (θ u j) = 0 := by
    simpa only [one_smul] using
      Transformer.proj_smul_self (hD.norm_eq_one hscheme h0 u hu j) (1 : ℝ)
  simpa only [hA, hp, smul_zero] using (hD u hu j).1

/-- The unit singleton gives all constant-direction hypotheses. -/
example (d : ℕ) :
    SchemeDynamics (d + 1) 1 1 (idParams (d + 1)) (idParams (d + 1)) (idParams (d + 1))
      0 0 .pre (fun _ _ => (basePoint d : EucSpace (d + 1))) (fun t _ => 1 + t) ∧
    (Scheme.pre = .pre ∨ Scheme.pre = .peri) ∧
    (∀ j : Idx 1, ‖(fun _ _ => (basePoint d : EucSpace (d + 1))) (0 : ℝ) j‖ = 1) ∧
    (0 : ℝ) ≤ 1 := by
  have hx : ‖(basePoint d : EucSpace (d + 1))‖ = 1 := by
    simp [basePoint, PiLp.norm_single]
  exact ⟨schemeDynamics_singleton 1 1 0 0 _ hx .pre (Or.inl rfl),
    Or.inl rfl, fun _ => hx, zero_le_one⟩

end Transformer.Normalization
