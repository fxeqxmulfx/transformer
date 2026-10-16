/-
# Almost-everywhere convergence to one original frozen center

ArXiv:2411.04990v2, §5 and Appendix A, the single-center
case of `thm:fixed_centers`.
The original center has positive mass and every limiting
predecessor is circularly at that center. The actual limiting
last force then has nonempty simple zeros. The actual scalar
convergence and countable repeller fibers apply directly.
The Borel detector and original-product Fubini exclude the
repelling alternative for almost every original initialization.
Every remaining last limit is exactly at the original center
on the circle. Real lifts and all original predecessors are
retained throughout this induction.

Every actual consensus critical tuple is asymptotically stable
by the original prefix stability and the negative actual last
derivative. Absolute continuity and global original uniqueness
give all conclusions for every actual original global solution
from almost every original initialization. No separation
condition or two-center parameter reduction is needed here.
-/

import Transformer.Causal.Section5_SimpleRows
import Transformer.Causal.Section5_LocalStability
import Transformer.Causal.Section5_ConditionalVolume

open Real Set Filter MeasureTheory

namespace Transformer.Causal

/-- Almost every original initialization converges under the actual original one-center frozen flow to an actual critical tuple lying exactly at the original center on the circle.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual single-center convergence and clustering portions of `thm:fixed_centers`; positive mass suffices at every real temperature. -/
theorem frozenOneCenterFlow_ae_tendsto_consensus (β θ a : ℝ) (ha : 0 < a) (n : ℕ) :
    ∀ᵐ φ₀ : Idx n → ℝ, ∃ φstar : Idx n → ℝ,
      Tendsto (frozenFlow β (fun _ : Idx 1 => θ) (fun _ => a) (fun _ => ha.le) φ₀) atTop (nhds φstar) ∧
      frozenVelocity n 1 β (fun _ => θ) (fun _ => a) φstar = 0 ∧ ∀ k, circDist (φstar k - θ) = 0 := by
  induction n with
  | zero =>
      apply Eventually.of_forall
      intro φ₀
      refine ⟨φ₀, ?_, Subsingleton.elim _ _, fun k => Fin.elim0 k⟩
      have he : frozenFlow β (fun _ : Idx 1 => θ) (fun _ => a) (fun _ => ha.le) φ₀ = fun _ => φ₀ :=
        funext (fun _ => Subsingleton.elim _ _)
      rw [he]
      exact tendsto_const_nhds
  | succ n ih =>
      have hnotAE : ∀ᵐ φ₀ : Idx (n + 1) → ℝ, φ₀ ∉ frozenRepellerDetector β (fun _ : Idx 1 => θ)
          (fun _ => a) (fun _ => ha.le) := by
        apply frozen_ae_of_countable_bad_fibers n _ (measurableSet_frozenRepellerDetector β _ _ (fun _ => ha.le))
        filter_upwards [ih] with p₀ hp₀
        obtain ⟨pstar, hp, hpcrit, hpre⟩ := hp₀
        exact frozenRepellerDetector_fiber_countable_of_simple_limit β _ _ (fun _ => ha.le) p₀ pstar hp
          ⟨θ, (frozenOneCenter_center_critical β θ a ha pstar hpre).1⟩
          (frozenOneCenter_critical_curvature_ne β θ a ha pstar hpre)
      have hpull := (quasiMeasurePreserving_frozenPrefix n).ae ih
      filter_upwards [hpull, hnotAE] with φ₀ hp₀ hnot
      obtain ⟨pstar, hp, hpcrit, hpre⟩ := hp₀
      have hpfull : Tendsto (fun (t : ℝ) (j : Idx n) => frozenFlow β (fun _ : Idx 1 => θ) (fun _ => a)
          (fun _ => ha.le) φ₀ t j.castSucc) atTop (nhds pstar) := by
        simpa! only [frozenFlow_castSucc] using hp
      obtain ⟨l, hl, hF, hG⟩ := frozenFlow_last_attracting_of_simple_limit β _ _ (fun _ => ha.le) φ₀ pstar hpfull
        ⟨θ, (frozenOneCenter_center_critical β θ a ha pstar hpre).1⟩
        (frozenOneCenter_critical_curvature_ne β θ a ha pstar hpre) hnot
      have hnear := (frozenOneCenter_nonrepelling_critical β θ a ha pstar hpre l hF hG.le).2
      have hv := (frozenVelocity_last_eq_zero_iff β (fun _ : Idx 1 => θ) (fun _ => a) (fun _ => ha.le) pstar l).mpr hF
      refine ⟨Fin.snoc pstar l, ?_, ?_, ?_⟩
      · apply tendsto_pi_nhds.mpr
        intro k
        refine Fin.lastCases ?_ (fun j => ?_) k
        · simpa only [Fin.snoc_last] using hl
        · simpa only [Fin.snoc_castSucc] using (tendsto_pi_nhds.mp hpfull) j
      · funext k
        refine Fin.lastCases hv (fun j => ?_) k
        rw [frozenVelocity_castSucc]
        simpa only [Fin.snoc_castSucc, Pi.zero_apply] using congrFun hpcrit j
      · intro k
        refine Fin.lastCases ?_ (fun j => ?_) k
        · simpa only [Fin.snoc_last] using hnear
        · simpa only [Fin.snoc_castSucc] using hpre j

/-- Every actual original one-center consensus critical tuple is asymptotically stable for the original full angular field.
Source: arXiv:2411.04990v2, §5 and Appendix A, the single-center local stability conclusion of `thm:fixed_centers`, with the original maximum-norm stability predicate. -/
theorem isAsymptoticallyStable_frozenOneCenter_consensus (β θ a : ℝ) (ha : 0 < a) (n : ℕ) (φstar : Idx n → ℝ)
    (hcrit : frozenVelocity n 1 β (fun _ => θ) (fun _ => a) φstar = 0)
    (hnear : ∀ k, circDist (φstar k - θ) = 0) :
    IsAsymptoticallyStable n (frozenVelocity n 1 β (fun _ => θ) (fun _ => a)) φstar := by
  induction n with
  | zero => exact isAsymptoticallyStable_empty _ _
  | succ n ih =>
      let pstar : Idx n → ℝ := fun j => φstar j.castSucc
      let l : ℝ := φstar (Fin.last n)
      have hsnoc : Fin.snoc pstar l = φstar := Fin.snoc_init_self φstar
      have hpcrit : frozenVelocity n 1 β (fun _ => θ) (fun _ => a) pstar = 0 := by
        funext j
        have h := congrFun hcrit j.castSucc
        simpa only [frozenVelocity_castSucc, Pi.zero_apply, pstar] using h
      have hpnear : ∀ k, circDist (pstar k - θ) = 0 := fun k => hnear k.castSucc
      have hpstable := ih pstar hpcrit hpnear
      have hlcrit : frozenVelocity (n + 1) 1 β (fun _ => θ) (fun _ => a) (Fin.snoc pstar l) (Fin.last n) = 0 := by
        rw [hsnoc]
        exact congrFun hcrit (Fin.last n)
      have hG : 0 < frozenLastCurvature β (fun _ : Idx 1 => θ) (fun _ => a) pstar l := by
        rw [frozenOneCenter_curvature β θ a pstar hpnear, ← g_pot_circDist β (l - θ)]
        have hlnear : circDist (l - θ) = 0 := hnear (Fin.last n)
        rw [hlnear]
        norm_num [g_pot]
        linarith [Nat.cast_nonneg (α := ℝ) n]
      have hD := (frozenLastVelocityDeriv_neg_iff β _ _ (fun _ => ha.le) pstar l hlcrit).mpr hG
      rw [← hsnoc]
      exact isAsymptoticallyStable_frozen_extension β (fun _ : Idx 1 => θ) (fun _ => a)
        (fun _ => ha.le) pstar l hpstable hlcrit hD

/-- Under every original absolutely continuous initialization, every actual original one-center global solution converges almost surely to an actual asymptotically stable critical tuple exactly at the original center on the circle.
Source: arXiv:2411.04990v2, §5 and Appendix A, all original conclusions of the single-center case of `thm:fixed_centers`; the helper requires only positive mass and absolute continuity. -/
theorem frozenOneCenterODE_mu_ae_tendsto_stable_consensus (β θ a : ℝ) (ha : 0 < a) (n : ℕ)
    (μ : Measure (Idx n → ℝ)) (hμ : μ ≪ volume) :
    ∀ᵐ φ₀ ∂μ, ∀ φ : ℝ → Idx n → ℝ, φ 0 = φ₀ →
      (∀ t, HasDerivAt φ (frozenVelocity n 1 β (fun _ => θ) (fun _ => a) (φ t)) t) →
      ∃ φstar : Idx n → ℝ, Tendsto φ atTop (nhds φstar) ∧
        frozenVelocity n 1 β (fun _ => θ) (fun _ => a) φstar = 0 ∧
        IsAsymptoticallyStable n (frozenVelocity n 1 β (fun _ => θ) (fun _ => a)) φstar ∧
        ∀ k, circDist (φstar k - θ) = 0 := by
  have hAE : ∀ᵐ φ₀ ∂μ, ∃ φstar : Idx n → ℝ,
      Tendsto (frozenFlow β (fun _ : Idx 1 => θ) (fun _ => a) (fun _ => ha.le) φ₀) atTop (nhds φstar) ∧
      frozenVelocity n 1 β (fun _ => θ) (fun _ => a) φstar = 0 ∧ ∀ k, circDist (φstar k - θ) = 0 := by
    have h := frozenOneCenterFlow_ae_tendsto_consensus β θ a ha n
    rw [ae_iff] at h ⊢
    exact hμ h
  filter_upwards [hAE] with φ₀ hφ₀
  intro φ h0 hφ
  obtain ⟨φstar, hlim, hcrit, hnear⟩ := hφ₀
  have heq := frozenFlow_eq_of_ODE β (fun _ : Idx 1 => θ) (fun _ => a) (fun _ => ha.le) φ hφ
  rw [h0] at heq
  exact ⟨φstar, heq ▸ hlim, hcrit, isAsymptoticallyStable_frozenOneCenter_consensus β θ a ha n φstar hcrit hnear, hnear⟩

/-- An actual positive-mass original center realizes the full almost-everywhere assertion on a nonempty original moving-token space and an actual absolutely continuous initialization.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual one-center initialization for `thm:fixed_centers`. -/
example : ∀ᵐ φ₀ : Idx 2 → ℝ, ∀ φ : ℝ → Idx 2 → ℝ, φ 0 = φ₀ →
    (∀ t, HasDerivAt φ (frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1) (φ t)) t) →
    ∃ φstar : Idx 2 → ℝ, Tendsto φ atTop (nhds φstar) ∧ frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1) φstar = 0 ∧
      IsAsymptoticallyStable 2 (frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1)) φstar ∧ ∀ k, circDist (φstar k - 0) = 0 :=
  frozenOneCenterODE_mu_ae_tendsto_stable_consensus 1 0 1 zero_lt_one 2 volume (Measure.AbsolutelyContinuous.refl volume)

/-- An actual original stationary consensus trajectory jointly witnesses the global-solution, criticality, and consensus premises with positive original center mass.
Source: arXiv:2411.04990v2, §5, the actual original one-center equilibrium in `thm:fixed_centers`. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (fun _ : Idx 2 => (0 : ℝ)))
    (frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1) (fun _ => 0)) t) ∧
    IsAsymptoticallyStable 2 (frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1)) (fun _ => 0) := by
  have hv : frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity]
  refine ⟨fun t => ?_, isAsymptoticallyStable_frozenOneCenter_consensus 1 0 1 zero_lt_one 2 (fun _ => 0)
    hv (fun _ => by rw [sub_self, circDist_zero])⟩
  simpa only [hv] using hasDerivAt_const t (fun _ : Idx 2 => (0 : ℝ))

end Transformer.Causal
