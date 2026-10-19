module

public import Transformer.Causal.AppendixA_CuspCollapse

/-!
# Counterexample to the printed C¹ sequential-flow convergence lemma

ArXiv:2411.04990v2, Appendix A, `lemma:convergence`, claims that
C¹ energies and positive bounded C¹ normalizations give convergence
to strongly stable critical points for almost every initialization,
assuming isolated own-angle critical points and the stated spectral
alternative at every critical truncation. The claim is false under
these printed hypotheses.

The actual two-particle periodic energies are `E₀ = cos ψ₀` and
`E₁ = cos ψ₁ + sin² ψ₀ · cuspPrimitive ψ₁`, with normalization one.
`cuspEnergy_sequential_hypotheses` proves every printed hypothesis,
including genuine derivatives and hyperbolicity at critical points.
The explicit global trajectories from `AppendixA_CuspCollapse`
reach second angle zero by time one and converge to `(π,0)`, whose
actual Jacobian has the positive eigenvalue one.

The set below is an actual open, nonempty set of initial angles.
The inverse half-angle formulas recover every such initialization,
and product Lebesgue measure is strictly positive on this set.
Every point in it admits an actual solution with no strongly stable
limit. The final theorem negates the full universal statement,
already for two particles, with its original C¹ hypotheses intact.
The source's unstable-manifold step requires regularity of the
velocity that does not follow from its C¹ energy assumption.
-/

@[expose] public section

open Real Filter Topology MeasureTheory Set
namespace Transformer.Causal

/-- The actual open range of original angles that collapse by time one.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, counterexample. -/
def cuspCollapseStarts : Set (Idx 2 → ℝ) :=
  {ψ | ψ 0 ∈ Ioo 0 π ∧ ψ 1 ∈ Ioo 0 π ∧
    Real.sqrt (Real.tan (ψ 1 / 2)) < cuspCollapseClock (Real.tan (ψ 0 / 2)) 1}

/-- The collapse condition is open in the original angular coordinates.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, counterexample. -/
theorem isOpen_cuspCollapseStarts : IsOpen cuspCollapseStarts := by
  apply isOpen_iff_mem_nhds.mpr
  rintro ψ ⟨h₀, h₁, hgap⟩
  have htan (i : Idx 2) (hi : ψ i ∈ Ioo 0 π) :
      ContinuousAt (fun η : Idx 2 → ℝ => Real.tan (η i / 2)) ψ := by
    have hcos : Real.cos (ψ i / 2) ≠ 0 :=
      ne_of_gt (Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.pi_pos, hi.1], by linarith [hi.2]⟩)
    have hhalf : ContinuousAt (fun η : Idx 2 → ℝ => η i / (2 : ℝ)) ψ := by
      have hcoord : ContinuousAt (fun η : Idx 2 → ℝ => η i) ψ :=
        (continuous_apply i).continuousAt
      exact hcoord.div_const (2 : ℝ)
    exact (Real.continuousAt_tan.mpr hcos).comp (f := fun η : Idx 2 → ℝ => η i / (2 : ℝ)) hhalf
  have hleft := (htan 1 h₁).sqrt
  have hright : ContinuousAt (fun η : Idx 2 → ℝ => cuspCollapseClock (Real.tan (η 0 / 2)) 1) ψ :=
    continuous_cuspCollapseClock_one.continuousAt.comp (htan 0 h₀)
  have hnear (i : Idx 2) (hi : ψ i ∈ Ioo 0 π) :
      ∀ᶠ η : Idx 2 → ℝ in nhds ψ, η i ∈ Ioo 0 π :=
    (continuous_apply i).continuousAt.eventually_mem (isOpen_Ioo.mem_nhds hi)
  filter_upwards [hnear 0 h₀, hnear 1 h₁, hleft.eventually_lt hright hgap] with η hη₀ hη₁ hηgap
  exact ⟨hη₀, hη₁, hηgap⟩

/-- A strictly positive actual clock supplies a genuine collapsing initialization.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, counterexample. -/
theorem cuspCollapseStarts_nonempty : cuspCollapseStarts.Nonempty := by
  let c := cuspCollapseClock 1 1 / 2
  have hc : 0 < c := div_pos (cuspCollapseClock_one_pos zero_lt_one) (by norm_num)
  have hcgap : c < cuspCollapseClock 1 1 := by
    dsimp [c]
    linarith [cuspCollapseClock_one_pos zero_lt_one]
  refine ⟨![2 * Real.arctan 1, 2 * Real.arctan (c ^ 2)], ?_⟩
  change (2 * Real.arctan 1 ∈ Ioo 0 π) ∧ (2 * Real.arctan (c ^ 2) ∈ Ioo 0 π) ∧
    Real.sqrt (Real.tan (2 * Real.arctan (c ^ 2) / 2)) <
      cuspCollapseClock (Real.tan (2 * Real.arctan 1 / 2)) 1
  refine ⟨⟨by positivity, by linarith [Real.arctan_lt_pi_div_two (1 : ℝ)]⟩,
    ⟨by exact mul_pos (by norm_num) (Real.arctan_pos.mpr (sq_pos_of_pos hc)),
      by linarith [Real.arctan_lt_pi_div_two (c ^ 2)]⟩, ?_⟩
  simpa only [mul_div_cancel_left₀ _ (by norm_num : (2 : ℝ) ≠ 0), Real.tan_arctan,
    Real.sqrt_sq hc.le] using hcgap

/-- The actual collapsing initial set has positive product Lebesgue measure.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, counterexample. -/
theorem cuspCollapseStarts_measure_pos :
    0 < (Measure.pi fun _ : Idx 2 => (volume : Measure ℝ)) cuspCollapseStarts :=
  isOpen_cuspCollapseStarts.measure_pos _ cuspCollapseStarts_nonempty

/-- The inverse half-angle parameters recover each original initialization.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, counterexample. -/
theorem cuspCollapsedTrajectory_initial_of_mem {ψ : Idx 2 → ℝ} (hψ : ψ ∈ cuspCollapseStarts) :
    cuspCollapsedTrajectory (Real.tan (ψ 0 / 2)) (Real.sqrt (Real.tan (ψ 1 / 2))) 0 = ψ := by
  have htan : 0 ≤ Real.tan (ψ 1 / 2) :=
    (Real.tan_pos_of_pos_of_lt_pi_div_two (by linarith [hψ.2.1.1])
      (by linarith [hψ.2.1.2])).le
  rw [cuspCollapsedTrajectory_zero _ _ (Real.sqrt_nonneg _), Real.sq_sqrt htan]
  have hhalf (i : Idx 2) (hi : ψ i ∈ Ioo 0 π) :
      Real.arctan (Real.tan (ψ i / 2)) = ψ i / 2 :=
    Real.arctan_tan (by linarith [Real.pi_pos, hi.1]) (by linarith [hi.2])
  rw [hhalf 0 hψ.1, hhalf 1 hψ.2.1]
  ext i
  fin_cases i <;> simp <;> ring

/-- Every initialization in the actual open set admits a globally defined bad flow.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, counterexample. -/
theorem cuspCollapseStarts_bad_flow {ψ : Idx 2 → ℝ} (hψ : ψ ∈ cuspCollapseStarts) :
    ∃ φ : ℝ → Idx 2 → ℝ, φ 0 = ψ ∧ SequentialFlow 2 cuspEnergy (fun _ _ => 1) φ ∧
      ¬ ∃ ψstar : Idx 2 → ℝ, Tendsto φ atTop (nhds ψstar) ∧
        (∀ k : Idx 2, IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψstar ∧
          StronglyStablePrefix 2 cuspEnergy (fun _ _ => 1) k ψstar) := by
  refine ⟨cuspCollapsedTrajectory (Real.tan (ψ 0 / 2)) (Real.sqrt (Real.tan (ψ 1 / 2))),
    cuspCollapsedTrajectory_initial_of_mem hψ, sequentialFlow_cuspCollapsedTrajectory _ _, ?_⟩
  exact cuspCollapsedTrajectory_no_stable_limit
    (Real.tan_pos_of_pos_of_lt_pi_div_two (by linarith [hψ.1.1]) (by linarith [hψ.1.2])) hψ.2.2

/-- The membership hypotheses of the inverse-parameter and bad-flow theorems
are jointly realized by an actual initialization in the open set. -/
example : ∃ ψ : Idx 2 → ℝ, ψ ∈ cuspCollapseStarts := cuspCollapseStarts_nonempty

/-- The almost-everywhere conclusion of the printed lemma fails for these energies.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, counterexample. -/
theorem cuspEnergy_not_ae_stable :
    ¬ (∀ᵐ ψ₀ : Idx 2 → ℝ ∂(Measure.pi fun _ : Idx 2 => (volume : Measure ℝ)),
      ∀ φ : ℝ → Idx 2 → ℝ, φ 0 = ψ₀ → SequentialFlow 2 cuspEnergy (fun _ _ => 1) φ →
        ∃ ψstar : Idx 2 → ℝ, Tendsto φ atTop (nhds ψstar) ∧
          (∀ k : Idx 2, IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψstar ∧
            StronglyStablePrefix 2 cuspEnergy (fun _ _ => 1) k ψstar)) := by
  intro h
  have hzero : (Measure.pi fun _ : Idx 2 => (volume : Measure ℝ)) cuspCollapseStarts = 0 := by
    apply measure_mono_null ?_ (ae_iff.mp h)
    intro ψ hψ hgood
    obtain ⟨φ, hstart, hflow, hbad⟩ := cuspCollapseStarts_bad_flow hψ
    exact hbad (hgood φ hstart hflow)
  exact (ne_of_gt cuspCollapseStarts_measure_pos) hzero

/-- Counterexample to `lemma:convergence` as printed: its full universal C¹
statement is false, already for two particles. The actual periodic energies
satisfy every original hypothesis with `c = 1/2`, `C = 2` and `Z = 1`,
but every initialization in a positive-measure open set admits a genuine
global trajectory converging to the strongly unstable tuple `(π,0)`.
No regularity assumption is added.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`. -/
theorem not_sequentialFlow_converges :
    ¬ (∀ (E Z : Idx 2 → (Idx 2 → ℝ) → ℝ) (c C : ℝ), 0 < c →
      (∀ k : Idx 2, ContDiff ℝ 1 (E k) ∧ ContDiff ℝ 1 (Z k)) →
      (∀ k : Idx 2, ∀ ψ : Idx 2 → ℝ, c < Z k ψ ∧ Z k ψ < C) →
      PeriodicAngles 2 E → PeriodicAngles 2 Z → DependsOnPrefix 2 E → DependsOnPrefix 2 Z →
      IsolatedCritical 2 E →
      (∀ k : Idx 2, ∀ ψ : Idx 2 → ℝ, IsCriticalPrefix 2 E Z k ψ →
        StronglyStablePrefix 2 E Z k ψ ∨ StronglyUnstablePrefix 2 E Z k ψ) →
      ∀ᵐ ψ₀ : Idx 2 → ℝ ∂(Measure.pi fun _ : Idx 2 => (volume : Measure ℝ)),
        ∀ φ : ℝ → Idx 2 → ℝ, φ 0 = ψ₀ → SequentialFlow 2 E Z φ →
          ∃ ψstar : Idx 2 → ℝ, Tendsto φ atTop (nhds ψstar) ∧
            (∀ k : Idx 2, IsCriticalPrefix 2 E Z k ψstar ∧
              StronglyStablePrefix 2 E Z k ψstar)) := by
  intro h
  obtain ⟨hc, hsmooth, hZ, hEper, hZper, hEpre, hZpre, hiso, hhyp⟩ :=
    cuspEnergy_sequential_hypotheses
  exact cuspEnergy_not_ae_stable
    (h cuspEnergy (fun _ _ => 1) (1 / 2) 2 hc hsmooth hZ hEper hZper hEpre hZpre hiso hhyp)

end Transformer.Causal
