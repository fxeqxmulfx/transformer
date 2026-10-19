module

public import Transformer.Causal.Section5_LocalStability
public import Transformer.Causal.Section5_AEConvergence

/-!
# Asymptotic stability of the actual clustered frozen limit

ArXiv:2411.04990v2, §5 and Appendix A, the stability and
clustering conclusions of `thm:fixed_centers` with at least
two separated original frozen centers.
The original signed curvature inequality makes the actual
last derivative negative throughout each narrow cluster when
all actual predecessors are clustered. A critical clustered
full tuple has a critical clustered original prefix. Induction
and the proved local extension theorem therefore give original
asymptotic stability for every such actual critical tuple.

The actual almost-everywhere convergence theorem reaches
these tuples in the derived narrow neighborhoods. Combining
the results gives convergence to an actual asymptotically
stable critical point, for every actual global solution and
every original absolutely continuous initialization measure.
The parameter reduction preserves the original force and
signed curvature conditions and yields the original requested
clustering radius. The single-center case is separate.
-/

@[expose] public section

open Real Set Filter MeasureTheory

namespace Transformer.Causal

variable {m : ℕ}

/-- Every actual original critical tuple clustered in the proved narrow frozen-center neighborhoods is asymptotically stable for the original full angular field.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual local stability portion of `thm:fixed_centers`, proved by the causal prefix and scalar extension arguments. -/
theorem isAsymptoticallyStable_frozen_clustered_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j)
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (n : ℕ) (hN : (n : ℝ) + (∑ j, a j) ≤ N) (φstar : Idx n → ℝ)
    (hcrit : frozenVelocity n m β θ a φstar = 0)
    (hnear : ∀ k, ∃ i, circDist (φstar k - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    IsAsymptoticallyStable n (frozenVelocity n m β θ a) φstar := by
  induction n with
  | zero => exact isAsymptoticallyStable_empty _ _
  | succ n ih =>
      have hNp : (n : ℝ) + (∑ j, a j) ≤ N := by
        rw [Nat.cast_add, Nat.cast_one] at hN
        linarith
      let pstar : Idx n → ℝ := fun j => φstar j.castSucc
      let l : ℝ := φstar (Fin.last n)
      have hsnoc : Fin.snoc pstar l = φstar := Fin.snoc_init_self φstar
      have hpcrit : frozenVelocity n m β θ a pstar = 0 := by
        funext j
        have h := congrFun hcrit j.castSucc
        simpa only [frozenVelocity_castSucc, Pi.zero_apply, pstar] using h
      have hpnear : ∀ k, ∃ i, circDist (pstar k - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)) := fun k => hnear k.castSucc
      have hpstable := ih hNp pstar hpcrit hpnear
      have hlcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc pstar l) (Fin.last n) = 0 := by
        rw [hsnoc]
        exact congrFun hcrit (Fin.last n)
      obtain ⟨i, hi⟩ := hnear (Fin.last n)
      have hG := frozen_last_curvature_pos_near_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hNp hg
        pstar hpnear l i hi
      have hD := (frozenLastVelocityDeriv_neg_iff β θ a (fun j => le_trans zero_le_one (ha j)) pstar l hlcrit).mpr hG
      rw [← hsnoc]
      exact isAsymptoticallyStable_frozen_extension β θ a
        (fun j => le_trans zero_le_one (ha j)) pstar l hpstable hlcrit hD

/-- A genuine original two-center critical tuple jointly realizes narrow clustering, criticality, and the hypotheses of the actual asymptotic-stability induction.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original frozen-center equilibrium used to witness the local stability assertion in `thm:fixed_centers`. -/
example : ∃ ε : ℝ, 0 < ε ∧
    frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) (fun _ => 0) = 0 ∧
    IsAsymptoticallyStable 1 (frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1)) (fun _ => 0) ∧
    ∀ k : Idx 1, ∃ i : Idx 2, circDist ((fun _ : Idx 1 => (0 : ℝ)) k - (![0, Real.pi]) i) ≤
      ε * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ :=
    frozen_sample_narrow_prefix_data
  have hv : frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity, Fin.sum_univ_two]
  refine ⟨ε, hε, hv, ?_, hpre⟩
  exact isAsymptoticallyStable_frozen_clustered_narrow 3 11 ε (142 / 25) hβ hε hc hcap
    (![0, Real.pi]) (fun _ => 1) ha hsep hpair hg 1
    (by simpa only [Nat.cast_one] using hN) (fun _ => 0) hv hpre

/-- In the proved narrow parameter window, almost every original initialization converges under every actual global solution to an actual asymptotically stable critical tuple clustered around the original centers.
Source: arXiv:2411.04990v2, §5 and Appendix A, all conclusions of `thm:fixed_centers` for the narrow separated case. -/
theorem frozenODE_mu_ae_tendsto_stable_clustered_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (n : ℕ) (hN : (n : ℝ) + (∑ j, a j) ≤ N) (μ : Measure (Idx n → ℝ)) (hμ : μ ≪ volume) :
    ∀ᵐ φ₀ ∂μ, ∀ φ : ℝ → Idx n → ℝ, φ 0 = φ₀ →
      (∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t) → ∃ φstar : Idx n → ℝ,
        Tendsto φ atTop (nhds φstar) ∧ frozenVelocity n m β θ a φstar = 0 ∧
        IsAsymptoticallyStable n (frozenVelocity n m β θ a) φstar ∧
        ∀ k, ∃ i, circDist (φstar k - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  filter_upwards [frozenODE_mu_ae_tendsto_clustered_critical_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hh hg n hN μ hμ]
    with φ₀ hφ₀
  intro φ h0 hφ
  obtain ⟨φstar, hlim, hcrit, hnear⟩ := hφ₀ φ h0 hφ
  exact ⟨φstar, hlim, hcrit, isAsymptoticallyStable_frozen_clustered_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hg
    n hN φstar hcrit (fun k => by obtain ⟨i, hi⟩ := hnear k; exact ⟨i, hi.le⟩), hnear⟩

/-- The original parameter inequalities give all original convergence, stability, and clustering conclusions for any separated frozen-center pair and total mass at least three.
Source: arXiv:2411.04990v2, §5 and Appendix A, `thm:fixed_centers` for the nontrivial separated case, with the original documented restored minus sign in its `g` condition. -/
theorem frozenODE_mu_ae_tendsto_stable_clustered (N β ε c : ℝ) (hN3 : 3 ≤ N) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (n : ℕ) (hN : (n : ℝ) + (∑ j, a j) ≤ N) (μ : Measure (Idx n → ℝ)) (hμ : μ ≪ volume) :
    ∀ᵐ φ₀ ∂μ, ∀ φ : ℝ → Idx n → ℝ, φ 0 = φ₀ →
      (∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t) → ∃ φstar : Idx n → ℝ,
        Tendsto φ atTop (nhds φstar) ∧ frozenVelocity n m β θ a φstar = 0 ∧
        IsAsymptoticallyStable n (frozenVelocity n m β θ a) φstar ∧
        ∀ k, ∃ i, circDist (φstar k - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  have hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi := by
    obtain ⟨i, j, hij⟩ := hpair
    exact frozen_separation_scale_lt_pi β c (θ i) (θ j) (hsep i j hij)
  obtain ⟨hη, hηε, hcap, hcη, hhη, hgη⟩ := frozen_parameters_narrow N β ε c hN3 hβ hε hc hcpi hh hg
  filter_upwards [frozenODE_mu_ae_tendsto_stable_clustered_narrow N β (frozenNarrowRadius β ε) c
    hβ hη hcη hcap θ a ha hsep hpair hhη hgη n hN μ hμ]
    with φ₀ hφ₀
  intro φ h0 hφ
  obtain ⟨φstar, hlim, hcrit, hstable, hnear⟩ := hφ₀ φ h0 hφ
  refine ⟨φstar, hlim, hcrit, hstable, ?_⟩
  intro k
  obtain ⟨i, hi⟩ := hnear k
  exact ⟨i, hi.trans_le (mul_le_mul_of_nonneg_right hηε (Real.rpow_pos_of_pos hβ _).le)⟩

/-- Genuine original two-center parameters realize the convergence, stability, mass, separation, and absolute-continuity premises in a nonempty original moving-token space.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual two-center initialization in `thm:fixed_centers`. -/
example : ∀ᵐ φ₀ : Idx 1 → ℝ, ∀ φ : ℝ → Idx 1 → ℝ, φ 0 = φ₀ →
    (∀ t, HasDerivAt φ (frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) (φ t)) t) →
    ∃ φstar : Idx 1 → ℝ, Tendsto φ atTop (nhds φstar) ∧ frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) φstar = 0 ∧
      IsAsymptoticallyStable 1 (frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1)) φstar ∧
      ∀ k, ∃ i, circDist (φstar k - (![0, Real.pi] : Idx 2 → ℝ) i) < (9 / 100 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  obtain ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩ := frozen_sample_parameters
  obtain ⟨η, hβη, hη, hcη, hcap, ha, hsep, hpair, hNη, hhη, hgη, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  exact frozenODE_mu_ae_tendsto_stable_clustered 3 11 (9 / 100) (142 / 25) le_rfl hβ hε hc
    (![0, Real.pi]) (fun _ => 1) ha hsep hpair hh hg 1 (by norm_num) volume (Measure.AbsolutelyContinuous.refl volume)

end Transformer.Causal
