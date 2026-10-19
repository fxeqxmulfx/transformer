module

public import Transformer.Causal.Section5_RowAvoidance
public import Transformer.Causal.Section5_ConditionalVolume

/-!
# Almost-everywhere convergence for the original frozen system

ArXiv:2411.04990v2, §5 and Appendix A, convergence and
clustering in `thm:fixed_centers` for at least two centers.
The proved narrow parameter window ensures that every
last-token critical angle over a clustered critical prefix
is either repelling or attracting near a frozen center.
Actual deterministic convergence and the countable repelling
fibers apply to every fixed converging clustered prefix.

The repeller detector is Borel in the full original initial
angular tuple. Original-product Fubini excludes it almost
everywhere. The original prefix projection preserves null
sets, so the almost-everywhere induction input also holds
for almost every original full tuple. Induction therefore
gives a genuinely critical, clustered real limit of the
actual original flow for almost every original initialization.
Absolute continuity transfers the conclusion to the original
initialization measures. Actual global-solution uniqueness
also gives the assertion for every original global solution.
Asymptotic stability remains a separate local assertion.
-/

@[expose] public section

open Real Set Filter MeasureTheory

namespace Transformer.Causal

variable {m : ℕ}

/-- Almost every original initialization has an actual critical and strictly clustered real limit under the original frozen flow in the proved narrow parameter window.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual convergence and clustering portions of `thm:fixed_centers` with at least two original centers; asymptotic stability is proved separately. -/
theorem frozenFlow_ae_tendsto_clustered_critical_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (n : ℕ) (hN : (n : ℝ) + (∑ j, a j) ≤ N) :
    ∀ᵐ φ₀ : Idx n → ℝ, ∃ φstar : Idx n → ℝ,
      Tendsto (frozenFlow β θ a (fun j => le_trans zero_le_one (ha j)) φ₀) atTop (nhds φstar) ∧
      frozenVelocity n m β θ a φstar = 0 ∧ ∀ k, ∃ i, circDist (φstar k - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  let ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  induction n with
  | zero =>
      apply Eventually.of_forall
      intro φ₀
      refine ⟨φ₀, ?_, Subsingleton.elim _ _, fun k => Fin.elim0 k⟩
      have hflow : frozenFlow β θ a ha0 φ₀ = fun _ => φ₀ := funext (fun _ => Subsingleton.elim _ _)
      rw [hflow]
      exact tendsto_const_nhds
  | succ n ih =>
      have hNp : (n : ℝ) + (∑ j, a j) ≤ N := by
        rw [Nat.cast_add, Nat.cast_one] at hN
        linarith
      have hpAE := ih hNp
      have hnotAE : ∀ᵐ φ₀ : Idx (n + 1) → ℝ, φ₀ ∉ frozenRepellerDetector β θ a ha0 := by
        apply frozen_ae_of_countable_bad_fibers n _ (measurableSet_frozenRepellerDetector β θ a ha0)
        filter_upwards [hpAE] with p₀ hp₀
        obtain ⟨pstar, hp, hpcrit, hpre⟩ := hp₀
        exact frozenRepellerDetector_fiber_countable N β ε c hβ hε hc hnarrow θ a ha hsep hpair hNp hh hg
          p₀ pstar hp (fun j => by obtain ⟨i, hi⟩ := hpre j; exact ⟨i, hi.le⟩)
      have hpull := (quasiMeasurePreserving_frozenPrefix n).ae hpAE
      filter_upwards [hpull, hnotAE] with φ₀ hp₀ hnot
      obtain ⟨pstar, hp, hpcrit, hpre⟩ := hp₀
      have hpfull : Tendsto (fun (t : ℝ) (j : Idx n) => frozenFlow β θ a ha0 φ₀ t j.castSucc) atTop (nhds pstar) := by
        simpa! only [frozenFlow_castSucc] using hp
      obtain ⟨l, hl, hcrit, hD, hnear⟩ := frozenFlow_tendsto_attracting_extension N β ε c hβ hε hc hnarrow θ a ha
        hsep hpair hNp hh hg φ₀ pstar hpfull (fun j => by obtain ⟨i, hi⟩ := hpre j; exact ⟨i, hi.le⟩) hpcrit hnot
      refine ⟨Fin.snoc pstar l, hl, hcrit, ?_⟩
      intro k
      refine Fin.lastCases ?_ (fun j => ?_) k
      · simpa only [Fin.snoc_last] using hnear
      · simpa only [Fin.snoc_castSucc] using hpre j

/-- Every original initialization measure absolutely continuous with respect to full original angular volume inherits actual critical clustered convergence in the proved narrow window.
Source: arXiv:2411.04990v2, §5 and Appendix A, transfer from original reference Lebesgue measure to the original initialization measure in `thm:fixed_centers`; the helper does not need normalization of the measure. -/
theorem frozenFlow_mu_ae_tendsto_clustered_critical_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (n : ℕ) (hN : (n : ℝ) + (∑ j, a j) ≤ N) (μ : Measure (Idx n → ℝ)) (hμ : μ ≪ volume) :
    ∀ᵐ φ₀ ∂μ, ∃ φstar : Idx n → ℝ,
      Tendsto (frozenFlow β θ a (fun j => le_trans zero_le_one (ha j)) φ₀) atTop (nhds φstar) ∧
      frozenVelocity n m β θ a φstar = 0 ∧ ∀ k, ∃ i, circDist (φstar k - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  have h := frozenFlow_ae_tendsto_clustered_critical_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hh hg n hN
  rw [ae_iff] at h ⊢
  exact hμ h

/-- Almost every original initialization yields the same critical clustered convergence for every actual original global solution through it in the proved narrow parameter window.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual global-solution quantifier in the convergence and clustering portions of `thm:fixed_centers`. -/
theorem frozenODE_mu_ae_tendsto_clustered_critical_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
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
        ∀ k, ∃ i, circDist (φstar k - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  filter_upwards [frozenFlow_mu_ae_tendsto_clustered_critical_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hh hg n hN μ hμ]
    with φ₀ hφ₀
  intro φ h0 hφ
  have heq := frozenFlow_eq_of_ODE β θ a (fun j => le_trans zero_le_one (ha j)) φ hφ
  rw [h0] at heq
  obtain ⟨φstar, hlim, hcrit, hnear⟩ := hφ₀
  exact ⟨φstar, heq ▸ hlim, hcrit, hnear⟩

/-- Under the original parameter inequalities, the derived narrow radius gives actual almost-everywhere critical clustered convergence within the original requested radius.
Source: arXiv:2411.04990v2, §5 and Appendix A, the convergence and clustering portions of `thm:fixed_centers` for the nontrivial separated case; the original `g` inequality has the documented restored minus sign. -/
theorem frozenFlow_mu_ae_tendsto_clustered_critical (N β ε c : ℝ) (hN3 : 3 ≤ N) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (n : ℕ) (hN : (n : ℝ) + (∑ j, a j) ≤ N) (μ : Measure (Idx n → ℝ)) (hμ : μ ≪ volume) :
    ∀ᵐ φ₀ ∂μ, ∃ φstar : Idx n → ℝ,
      Tendsto (frozenFlow β θ a (fun j => le_trans zero_le_one (ha j)) φ₀) atTop (nhds φstar) ∧
      frozenVelocity n m β θ a φstar = 0 ∧ ∀ k, ∃ i, circDist (φstar k - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  have hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi := by
    obtain ⟨i, j, hij⟩ := hpair
    exact frozen_separation_scale_lt_pi β c (θ i) (θ j) (hsep i j hij)
  obtain ⟨hη, hηε, hcap, hcη, hhη, hgη⟩ := frozen_parameters_narrow N β ε c hN3 hβ hε hc hcpi hh hg
  filter_upwards [frozenFlow_mu_ae_tendsto_clustered_critical_narrow N β (frozenNarrowRadius β ε) c hβ hη hcη hcap
    θ a ha hsep hpair hhη hgη n hN μ hμ] with φ₀ hφ₀
  obtain ⟨φstar, hlim, hcrit, hnear⟩ := hφ₀
  refine ⟨φstar, hlim, hcrit, ?_⟩
  intro k
  obtain ⟨i, hi⟩ := hnear k
  exact ⟨i, hi.trans_le (mul_le_mul_of_nonneg_right hηε (Real.rpow_pos_of_pos hβ _).le)⟩

/-- Actual original two-center parameters jointly realize all convergence, mass, separation, and absolute-continuity premises with a nonempty moving-token space.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual two-center specialization of the convergence and clustering assertion in `thm:fixed_centers`. -/
example : ∃ ε : ℝ, ∀ᵐ φ₀ : Idx 1 → ℝ, ∃ φstar : Idx 1 → ℝ,
    Tendsto (frozenFlow 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ => zero_le_one) φ₀) atTop (nhds φstar) ∧
    frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) φstar = 0 ∧
    ∀ k, ∃ i, circDist (φstar k - (![0, Real.pi] : Idx 2 → ℝ) i) < ε * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  exact ⟨ε, frozenFlow_mu_ae_tendsto_clustered_critical_narrow 3 11 ε (142 / 25) hβ hε hc hcap
    (![0, Real.pi]) (fun _ => 1) ha hsep hpair hh hg 1 (by simpa only [Nat.cast_one] using hN)
    volume (Measure.AbsolutelyContinuous.refl volume)⟩

/-- The original unmodified requested radius and actual original parameters realize the same almost-everywhere convergence and clustering conclusion.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original two-center parameter configuration in `thm:fixed_centers`. -/
example : ∀ᵐ φ₀ : Idx 1 → ℝ, ∃ φstar : Idx 1 → ℝ,
    Tendsto (frozenFlow 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ => zero_le_one) φ₀) atTop (nhds φstar) ∧
    frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) φstar = 0 ∧
    ∀ k, ∃ i, circDist (φstar k - (![0, Real.pi] : Idx 2 → ℝ) i) < (9 / 100 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  obtain ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩ := frozen_sample_parameters
  obtain ⟨η, hβη, hη, hcη, hcap, ha, hsep, hpair, hNη, hhη, hgη, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  exact frozenFlow_mu_ae_tendsto_clustered_critical 3 11 (9 / 100) (142 / 25) le_rfl hβ hε hc
    (![0, Real.pi]) (fun _ => 1) ha hsep hpair hh hg 1 (by norm_num)
    volume (Measure.AbsolutelyContinuous.refl volume)

/-- An actual original global stationary trajectory also exists for the same genuine two-center parameters.
Source: arXiv:2411.04990v2, §5, the original center equilibrium witnessing the solution premises of `thm:fixed_centers`. -/
example : ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (fun _ : Idx 1 => (0 : ℝ)))
    (frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) (fun _ => 0)) t := by
  have hv : frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity, Fin.sum_univ_two]
  intro t
  simpa only [hv] using hasDerivAt_const t (fun _ : Idx 1 => (0 : ℝ))

end Transformer.Causal
