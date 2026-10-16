/-
# Local asymptotic stability of an attracting frozen extension

ArXiv:2411.04990v2, §5 and Appendix A, the stability part
of `thm:fixed_centers` for the actual smooth frozen system.
An asymptotically stable original prefix and a strictly
negative actual last-angle derivative at a critical extension
give asymptotic stability of that full original extension.

For Lyapunov stability, choose a small inward last interval
and keep the actual original prefix in the parameter ball
preserving its two boundary signs. Scalar forward barriers
trap the last angle. The original maximum norm then controls
the full tuple. Prefix attraction gives an actual prefix limit;
the trapped scalar attraction theorem gives the last limit.
Both parts quantify over every actual original global solution,
as required by the original asymptotic-stability predicate.
No spectral or stable-manifold assertion is taken as an input.
-/

import Transformer.Causal.Section5_RowDerivative
import Transformer.Causal.Section5_RowConvergence
import Transformer.Causal.SectionA_ScalarAttraction

open Real Set Filter

namespace Transformer.Causal

variable {n m : ℕ}

/-- The actual original maximum angular error bounds its original prefix error.
Source: arXiv:2411.04990v2, §5 and Appendix A, the causal prefix comparison for original local stability in `thm:fixed_centers`. -/
theorem norm_frozenPrefix_sub_le (φ : Idx (n + 1) → ℝ) (p : Idx n → ℝ) (l : ℝ) :
    ‖(fun j => φ j.castSucc) - p‖ ≤ ‖φ - Fin.snoc p l‖ := by
  have h := pi_norm_comp_le (φ - Fin.snoc p l) (fun j : Idx n => j.castSucc)
  change ‖fun j : Idx n => φ j.castSucc - (Fin.snoc p l : Idx (n + 1) → ℝ) j.castSucc‖ ≤ _ at h
  change ‖fun j : Idx n => φ j.castSucc - p j‖ ≤ _
  simpa only [Fin.snoc_castSucc] using h

/-- The actual original maximum angular error bounds its actual last-angle error.
Source: arXiv:2411.04990v2, §5 and Appendix A, the last-coordinate comparison for original local stability in `thm:fixed_centers`. -/
theorem abs_frozenLast_sub_le (φ : Idx (n + 1) → ℝ) (p : Idx n → ℝ) (l : ℝ) :
    |φ (Fin.last n) - l| ≤ ‖φ - Fin.snoc p l‖ := by
  have h := norm_le_pi_norm (φ - Fin.snoc p l) (Fin.last n)
  change ‖φ (Fin.last n) - (Fin.snoc p l : Idx (n + 1) → ℝ) (Fin.last n)‖ ≤ _ at h
  simpa only [Fin.snoc_last, Real.norm_eq_abs] using h

/-- An actual asymptotically stable original prefix extends to an actual asymptotically stable original full point when its last scalar field vanishes and its actual own-angle derivative is strictly negative.
Source: arXiv:2411.04990v2, §5 and Appendix A, the local stability induction for the smooth original system in `thm:fixed_centers`. -/
theorem isAsymptoticallyStable_frozen_extension (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (pstar : Idx n → ℝ) (l : ℝ) (hpstable : IsAsymptoticallyStable n (frozenVelocity n m β θ a) pstar)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc pstar l) (Fin.last n) = 0)
    (hneg : frozenLastVelocityDeriv β θ a pstar l < 0) :
    IsAsymptoticallyStable (n + 1) (frozenVelocity (n + 1) m β θ a) (Fin.snoc pstar l) := by
  let f (q : (Idx n → ℝ) × ℝ) : ℝ := frozenVelocity (n + 1) m β θ a (Fin.snoc q.1 q.2) (Fin.last n)
  have hF : Continuous f := continuous_frozenLastVelocity_joint β θ a ha
  obtain ⟨R, hR, hintervals⟩ := frozenScalar_inward_intervals f
    (fun q : (Idx n → ℝ) × ℝ => frozenLastVelocityDeriv β θ a q.1 q.2) hF
    (continuous_frozenLastVelocityDeriv_joint β θ a ha)
    (fun p u => hasDerivAt_frozenLastVelocity β θ a ha p u) pstar l hcrit hneg
  have htrap (ρ : ℝ) (hρ : 0 < ρ) (hρR : ρ ≤ R) (η : ℝ) (hη : 0 < η) :
      ∃ δ : ℝ, 0 < δ ∧ ∀ φ : ℝ → Idx (n + 1) → ℝ,
        (∀ t, HasDerivAt φ (frozenVelocity (n + 1) m β θ a (φ t)) t) → ‖φ 0 - Fin.snoc pstar l‖ < δ →
        ∀ t, 0 ≤ t → ‖(fun j => φ t j.castSucc) - pstar‖ < η ∧ φ t (Fin.last n) ∈ Icc (l - ρ) (l + ρ) := by
    obtain ⟨δp, hδp, hboundary, hanti⟩ := hintervals ρ hρ hρR
    obtain ⟨δ0, hδ0, hprefix⟩ := hpstable.1 (min δp η) (lt_min hδp hη)
    refine ⟨min δ0 ρ, lt_min hδ0 hρ, ?_⟩
    intro φ hφ h0
    have hp0 : ‖(fun j => φ 0 j.castSucc) - pstar‖ < δ0 :=
      (norm_frozenPrefix_sub_le (φ 0) pstar l).trans_lt (h0.trans_le (min_le_left _ _))
    have hpt := hprefix (fun t j => φ t j.castSucc) (frozenODE_castSucc β θ a φ hφ) hp0
    have hd := frozenODE_last_scalar β θ a φ hφ
    have hinit : φ 0 (Fin.last n) ∈ Icc (l - ρ) (l + ρ) := by
      have hb := abs_lt.mp ((abs_frozenLast_sub_le (φ 0) pstar l).trans_lt (h0.trans_le (min_le_right _ _)))
      constructor <;> linarith [hb.1, hb.2]
    have hrow : ∀ t, 0 ≤ t → φ t (Fin.last n) ∈ Icc (l - ρ) (l + ρ) := by
      apply frozenScalar_trapped_by_inward_levels _ _ hd _ _ hinit
      · intro t ht he
        rw [he]
        exact (hboundary _ ((hpt t ht).trans_le (min_le_left _ _))).1
      · intro t ht he
        rw [he]
        exact (hboundary _ ((hpt t ht).trans_le (min_le_left _ _))).2
    intro t ht
    exact ⟨(hpt t ht).trans_le (min_le_right _ _), hrow t ht⟩
  constructor
  · intro ε hε
    let ρ := min (R / 2) (ε / 2)
    have hρ : 0 < ρ := lt_min (by positivity) (by positivity)
    have hρR : ρ ≤ R := (min_le_left _ _).trans (by linarith)
    have hρε : ρ < ε := (min_le_right _ _).trans_lt (by linarith)
    obtain ⟨δ, hδ, hgood⟩ := htrap ρ hρ hρR (ε / 2) (by positivity)
    refine ⟨δ, hδ, ?_⟩
    intro φ hφ h0 t ht
    obtain ⟨hp, hu⟩ := hgood φ hφ h0 t ht
    apply (pi_norm_lt_iff hε).mpr
    intro k
    refine Fin.lastCases ?_ (fun j => ?_) k
    · have hb : |φ t (Fin.last n) - l| ≤ ρ := abs_le.mpr ⟨by linarith [hu.1], by linarith [hu.2]⟩
      simpa only [Pi.sub_apply, Fin.snoc_last, Real.norm_eq_abs] using hb.trans_lt hρε
    · have hb := (norm_le_pi_norm ((fun j => φ t j.castSucc) - pstar) j).trans_lt hp
      simpa only [Pi.sub_apply, Fin.snoc_castSucc] using hb.trans (show ε / 2 < ε by linarith)
  · have hρ : 0 < R / 2 := by positivity
    have hρR : R / 2 ≤ R := by linarith
    obtain ⟨δ, hδ, hgood⟩ := htrap (R / 2) hρ hρR 1 zero_lt_one
    obtain ⟨η, hη, hprefixAttr⟩ := hpstable.2
    refine ⟨min δ η, lt_min hδ hη, ?_⟩
    intro φ hφ h0
    have hbound := hgood φ hφ (h0.trans_le (min_le_left _ _))
    have hp0 : ‖(fun j => φ 0 j.castSucc) - pstar‖ < η :=
      (norm_frozenPrefix_sub_le (φ 0) pstar l).trans_lt (h0.trans_le (min_le_right _ _))
    have hp := hprefixAttr (fun t j => φ t j.castSucc) (frozenODE_castSucc β θ a φ hφ) hp0
    obtain ⟨δp, hδp, hboundary, hanti⟩ := hintervals (R / 2) hρ hρR
    have hr : Tendsto (fun t => φ t (Fin.last n)) atTop (nhds l) :=
      frozenScalar_tendsto_of_trapped_antitone f hF (fun t j => φ t j.castSucc) pstar hp _
        (frozenODE_last_scalar β θ a φ hφ) (l - R / 2) (l + R / 2) l
        ((eventually_ge_atTop (0 : ℝ)).mono (fun t ht => (hbound t ht).2))
        ⟨by linarith, by linarith⟩ hcrit hanti
    apply tendsto_pi_nhds.mpr
    intro k
    refine Fin.lastCases ?_ (fun j => ?_) k
    · simpa only [Fin.snoc_last] using hr
    · simpa only [Fin.snoc_castSucc] using (tendsto_pi_nhds.mp hp) j

/-- In the empty original moving-token space, every actual point satisfies the original local asymptotic-stability predicate for every original field on that space.
Source: arXiv:2411.04990v2, §5, the zero-moving-token boundary case of `thm:fixed_centers`. -/
theorem isAsymptoticallyStable_empty (F : (Idx 0 → ℝ) → Idx 0 → ℝ) (p : Idx 0 → ℝ) :
    IsAsymptoticallyStable 0 F p := by
  constructor
  · intro ε hε
    refine ⟨1, zero_lt_one, ?_⟩
    intro φ hφ h0 t ht
    have he : φ t - p = 0 := Subsingleton.elim _ _
    simpa only [he, norm_zero] using hε
  · refine ⟨1, zero_lt_one, ?_⟩
    intro φ hφ h0
    have he : φ = fun _ => p := funext (fun _ => Subsingleton.elim _ _)
    rw [he]
    exact tendsto_const_nhds

/-- A genuine original one-center critical point has strictly negative actual last derivative and an actual stable empty prefix, giving original asymptotic stability in a nonempty state space.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual one-center attracting equilibrium in `thm:fixed_centers`. -/
example : IsAsymptoticallyStable 1 (frozenVelocity 1 1 1 (fun _ => 0) (fun _ => 1))
    (Fin.snoc (fun _ : Idx 0 => (0 : ℝ)) 0) := by
  have hv : frozenVelocity 1 1 1 (fun _ => 0) (fun _ => 1)
      (Fin.snoc (fun _ : Idx 0 => (0 : ℝ)) 0) (Fin.last 0) = 0 := by
    rw [frozenVelocity_last_eq_force]
    simp [frozenLastForce, h_pot]
  apply isAsymptoticallyStable_frozen_extension 1 (fun _ => 0) (fun _ => 1) (fun _ => zero_le_one)
    (fun _ : Idx 0 => (0 : ℝ)) 0 (isAsymptoticallyStable_empty _ _) hv
  apply (frozenLastVelocityDeriv_neg_iff 1 _ _ (fun _ => zero_le_one) _ _ hv).mpr
  norm_num [frozenLastCurvature, Fin.sum_univ_zero, Fin.sum_univ_one, g_pot]

end Transformer.Causal
