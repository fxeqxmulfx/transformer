import Transformer.Causal.AppendixA_CuspEnergy
import Mathlib.Analysis.Analytic.IsolatedZeros

/-!
# Isolated critical points of the actual periodic cusp energies

ArXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 1,
requires isolated critical points in each particle's own angle,
for every fixed value of the earlier angles. The actual C¹ energies
constructed in `AppendixA_CuspEnergy` satisfy that requirement.

The actual second velocity is `sin y - a cuspForce(y)`, where
`a = sin² ψ₀` lies between zero and one. Every zero is a zero of
the explicit analytic trigonometric function
`sin⁴ y · ((1-cos y) - a⁴(1+cos y))`. Its value at `2π/3` is
strictly positive throughout the original coefficient range.
Analytic uniqueness therefore isolates its zeros on the real line,
and hence isolates the original actual velocity's zeros.

The own-angle derivatives are proved from the original energy
expressions and the actual primitive derivative, including the
source's minus sign in the velocity. Both sine coordinates zero
also give an actual critical point of every truncation. This
provides the critical-point inputs for the separate Jacobian check.
No convergence or stability conclusion of the printed lemma is used.
-/

open Real Filter Set Topology
namespace Transformer.Causal

/-- Nontrivial real analyticity gives a punctured neighborhood without zeros.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 1;
the proof uses Mathlib's actual analytic uniqueness theorem on the real line. -/
theorem exists_radius_no_other_zero_of_real_analytic
    (f : ℝ → ℝ) (hf : AnalyticOnNhd ℝ f univ) (hne : ∃ z : ℝ, f z ≠ 0) (u : ℝ) :
    ∃ ε : ℝ, 0 < ε ∧ ∀ v : ℝ, |v - u| < ε → v ≠ u → f v ≠ 0 := by
  have hz : ∀ᶠ v in 𝓝[≠] u, f v ≠ 0 := by
    by_contra h
    have hfreq : ∃ᶠ v in 𝓝[≠] u, f v = 0 := by simpa only [not_not] using (not_eventually.mp h)
    have hall := hf.eqOn_zero_of_preconnected_of_frequently_eq_zero isPreconnected_univ (mem_univ u) hfreq
    obtain ⟨z, hz⟩ := hne
    exact hz (hall (mem_univ z))
  obtain ⟨ε, hε, hball⟩ := Metric.eventually_nhds_iff.mp (eventually_nhdsWithin_iff.mp hz)
  refine ⟨ε, hε, fun v hv hne => hball ?_ ?_⟩
  · simpa only [Real.dist_eq] using hv
  · simpa only [mem_compl_iff, mem_singleton_iff] using hne

/-- The explicit trigonometric expression containing all actual velocity zeros.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, critical-point audit. -/
noncomputable def cuspCriticalPolynomial (a y : ℝ) : ℝ :=
  Real.sin y ^ 4 * ((1 - Real.cos y) - a ^ 4 * (1 + Real.cos y))

/-- This actual finite trigonometric expression is analytic everywhere.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 1. -/
theorem analyticOnNhd_cuspCriticalPolynomial (a : ℝ) :
    AnalyticOnNhd ℝ (cuspCriticalPolynomial a) univ := by
  intro y _
  unfold cuspCriticalPolynomial
  fun_prop

/-- Every true zero of the actual velocity satisfies the explicit analytic equation.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, critical-point audit. -/
theorem cuspCriticalPolynomial_zero_of_velocity_zero (a y : ℝ)
    (h : Real.sin y - a * cuspForce y = 0) : cuspCriticalPolynomial a y = 0 :=
  cuspForce_critical_relation a y (sub_eq_zero.mp h)

/-- A genuine fixed angular value rules out the identically zero analytic case.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 1. -/
theorem cuspCriticalPolynomial_pos (a : ℝ) (ha₀ : 0 ≤ a) (ha₁ : a ≤ 1) :
    0 < cuspCriticalPolynomial a (2 * π / 3) := by
  have hc : Real.cos (2 * π / 3) = -(1 / 2 : ℝ) := by
    rw [show 2 * π / 3 = π - π / 3 by ring, Real.cos_pi_sub, Real.cos_pi_div_three]
  have hs : 0 < Real.sin (2 * π / 3) :=
    Real.sin_pos_of_pos_of_lt_pi (by linarith [Real.pi_pos]) (by linarith [Real.pi_pos])
  have ha₄ : a ^ 4 ≤ 1 := pow_le_one₀ ha₀ ha₁
  unfold cuspCriticalPolynomial
  rw [hc]
  exact mul_pos (pow_pos hs _) (by linarith)

/-- The actual velocity zeros are isolated for every original coefficient.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 1. -/
theorem exists_radius_no_other_cusp_velocity_zero (a : ℝ) (ha₀ : 0 ≤ a) (ha₁ : a ≤ 1)
    (u : ℝ) :
    ∃ ε : ℝ, 0 < ε ∧ ∀ v : ℝ, |v - u| < ε → v ≠ u →
      Real.sin v - a * cuspForce v ≠ 0 := by
  obtain ⟨ε, hε, hz⟩ := exists_radius_no_other_zero_of_real_analytic (cuspCriticalPolynomial a)
    (analyticOnNhd_cuspCriticalPolynomial a) ⟨2 * π / 3, (cuspCriticalPolynomial_pos a ha₀ ha₁).ne'⟩ u
  exact ⟨ε, hε, fun v hv hne hzero =>
    hz v hv hne (cuspCriticalPolynomial_zero_of_velocity_zero a v hzero)⟩

/-- The actual first own-angle derivative retains the original energy sign.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, original ODE. -/
theorem deriv_cuspEnergy_update_zero (ψ : Idx 2 → ℝ) (u : ℝ) :
    deriv (fun v => cuspEnergy 0 (Function.update ψ 0 v)) u = -Real.sin u := by
  have heq : (fun v : ℝ => cuspEnergy 0 (Function.update ψ 0 v)) = Real.cos := by
    funext v
    simp
  rw [heq]
  exact (Real.hasDerivAt_cos u).deriv

/-- The actual second own-angle derivative retains the original cusp primitive.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, original ODE. -/
theorem deriv_cuspEnergy_update_one (ψ : Idx 2 → ℝ) (u : ℝ) :
    deriv (fun v => cuspEnergy 1 (Function.update ψ 1 v)) u =
      -Real.sin u + Real.sin (ψ 0) ^ 2 * cuspForce u := by
  have heq : (fun v : ℝ => cuspEnergy 1 (Function.update ψ 1 v)) =
      fun v => Real.cos v + Real.sin (ψ 0) ^ 2 * cuspPrimitive v := by
    funext v
    simp
  rw [heq]
  exact ((Real.hasDerivAt_cos u).add ((hasDerivAt_cuspPrimitive u).const_mul (Real.sin (ψ 0) ^ 2))).deriv

/-- The actual two C¹ energies satisfy the printed isolated-critical-point condition.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 1. -/
theorem isolatedCritical_cuspEnergy : IsolatedCritical 2 cuspEnergy := by
  intro k ψ u
  fin_cases k
  · intro _
    obtain ⟨ε, hε, hz⟩ := exists_radius_no_other_cusp_velocity_zero 0 le_rfl zero_le_one u
    refine ⟨ε, hε, fun v hv hne => ?_⟩
    change deriv (fun w => cuspEnergy 0 (Function.update ψ 0 w)) v ≠ 0
    rw [deriv_cuspEnergy_update_zero]
    simpa only [zero_mul, sub_zero, neg_ne_zero] using hz v hv hne
  · intro _
    have ha₀ : 0 ≤ Real.sin (ψ 0) ^ 2 := sq_nonneg _
    have ha₁ : Real.sin (ψ 0) ^ 2 ≤ 1 := Real.sin_sq_le_one _
    obtain ⟨ε, hε, hz⟩ := exists_radius_no_other_cusp_velocity_zero _ ha₀ ha₁ u
    refine ⟨ε, hε, fun v hv hne hzero => ?_⟩
    change deriv (fun w => cuspEnergy 1 (Function.update ψ 1 w)) v = 0 at hzero
    rw [deriv_cuspEnergy_update_one] at hzero
    apply hz v hv hne
    linarith

/-- Zero sine coordinates give an actual critical point of every truncation.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 2 inputs. -/
theorem isCriticalPrefix_cuspEnergy_of_sine_zero (k : Idx 2) (ψ : Idx 2 → ℝ)
    (h₀ : Real.sin (ψ 0) = 0) (h₁ : Real.sin (ψ 1) = 0) :
    IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψ := by
  intro i
  by_cases hi : i.1 = 0
  · rw [hi, seqVelocity_cuspEnergy_zero, h₀]
  · have hi₁ : i.1 = 1 := by
      apply Fin.ext
      have hn : (i.1 : ℕ) ≠ 0 := fun h => hi (Fin.ext h)
      omega
    rw [hi₁, seqVelocity_cuspEnergy_one, h₀, h₁]
    ring

/-- The actual zero coefficient, origin, nontrivial analytic function and
critical tuple `(π,0)` jointly realize all coefficient, zero and analytic
hypotheses, and every original truncated critical-point predicate. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ Real.sin 0 - 0 * cuspForce 0 = 0 ∧
    AnalyticOnNhd ℝ (cuspCriticalPolynomial 0) univ ∧
    (∃ z : ℝ, cuspCriticalPolynomial 0 z ≠ 0) ∧ IsolatedCritical 2 cuspEnergy ∧
    (∀ k : Idx 2, IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k (![π, 0] : Idx 2 → ℝ)) :=
  ⟨le_rfl, zero_le_one, by simp, analyticOnNhd_cuspCriticalPolynomial 0,
    ⟨2 * π / 3, (cuspCriticalPolynomial_pos 0 le_rfl zero_le_one).ne'⟩,
    isolatedCritical_cuspEnergy,
    fun k => isCriticalPrefix_cuspEnergy_of_sine_zero k _ (by simp) (by simp)⟩

end Transformer.Causal
