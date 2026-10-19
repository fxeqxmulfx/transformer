module

public import Transformer.Causal.AppendixA_CuspEnergy
public import Mathlib.Analysis.Asymptotics.Defs
public import Mathlib.Analysis.Calculus.FDeriv.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

/-!
# Actual Jacobians of the periodic C¹ cusp system

ArXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 2,
uses the actual Jacobian at each truncated critical point.
The constructed energies are only C¹, but their actual velocity
still has a genuine Fréchet derivative at every such point.

The actual cusp force is uniformly bounded by two. If the first
sine is zero, its square therefore makes the entire cusp coupling
quadratically small in the actual full configuration displacement.
The derivative of that actual coupling is consequently zero there.
This uses the genuine remainder estimate, with no assigned value
for a derivative that does not exist.

Differentiating the remaining actual sine coordinates proves that
the original truncated Jacobian is diagonal with entries `cos ψ_i`.
The actual characteristic polynomial over the complex numbers is
then the product of the corresponding linear factors. These are
the Jacobians and eigenvalues used by the original strong-stability
and strong-instability predicates in the subsequent hypothesis check.
The actual target tuple `(π,0)` realizes the zero-sine input.
-/

@[expose] public section

open Real Asymptotics Filter Topology
namespace Transformer.Causal

/-- The true continuous cusp force has a global finite bound.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, Jacobian audit. -/
theorem abs_cuspForce_le (y : ℝ) : |cuspForce y| ≤ 2 := by
  have h₄ : cuspForce y ^ 2 ≤ 4 := by
    calc
      _ = 4 * |Real.sin (y / 2)| * Real.cos (y / 2) ^ 2 * |Real.cos (y / 2)| := cuspForce_sq y
      _ ≤ 4 * 1 * 1 * 1 := by
        gcongr
        · exact Real.abs_sin_le_one _
        · exact Real.cos_sq_le_one _
        · exact Real.abs_cos_le_one _
      _ = 4 := by norm_num
  nlinarith [sq_abs (cuspForce y), abs_nonneg (cuspForce y)]

/-- The original coupling is genuinely quadratic at each first critical angle.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 2. -/
theorem abs_cuspCoupling_le_sq_norm (ψ : Idx 2 → ℝ) (hψ : Real.sin (ψ 0) = 0)
    (η : Idx 2 → ℝ) :
    |Real.sin (η 0) ^ 2 * cuspForce (η 1)| ≤ 2 * ‖η - ψ‖ ^ 2 := by
  have hs : |Real.sin (η 0)| ≤ ‖η - ψ‖ := by
    have ht := Real.abs_sin_sub_sin_le (η 0) (ψ 0)
    rw [hψ, sub_zero] at ht
    apply ht.trans
    simpa only [Pi.sub_apply, Real.norm_eq_abs] using norm_le_pi_norm (η - ψ) 0
  have hs₂ : Real.sin (η 0) ^ 2 ≤ ‖η - ψ‖ ^ 2 := by
    nlinarith [sq_abs (Real.sin (η 0)), abs_nonneg (Real.sin (η 0)), norm_nonneg (η - ψ)]
  calc
    _ = Real.sin (η 0) ^ 2 * |cuspForce (η 1)| := by rw [abs_mul, abs_of_nonneg (sq_nonneg _)]
    _ ≤ Real.sin (η 0) ^ 2 * 2 := mul_le_mul_of_nonneg_left (abs_cuspForce_le _) (sq_nonneg _)
    _ ≤ _ := by linarith

/-- The actual coupling derivative exists and is zero by its quadratic remainder.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 2. -/
theorem hasFDerivAt_cuspCoupling_zero (ψ : Idx 2 → ℝ) (hψ : Real.sin (ψ 0) = 0) :
    HasFDerivAt (fun η : Idx 2 → ℝ => Real.sin (η 0) ^ 2 * cuspForce (η 1))
      (0 : (Idx 2 → ℝ) →L[ℝ] ℝ) ψ := by
  have hO : (fun η : Idx 2 → ℝ => Real.sin (η 0) ^ 2 * cuspForce (η 1)) =O[𝓝 ψ]
      fun η => ‖η - ψ‖ ^ 2 := by
    apply IsBigO.of_bound 2
    filter_upwards with η
    rw [Real.norm_eq_abs, Real.norm_of_nonneg (sq_nonneg _)]
    exact abs_cuspCoupling_le_sq_norm ψ hψ η
  exact hO.hasFDerivAt (by decide : 1 < 2)

/-- The original actual velocity has this genuine Fréchet derivative.
The cusp term has already been differentiated by its actual estimate.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 2. -/
theorem hasFDerivAt_seqVelocity_cuspEnergy (ψ : Idx 2 → ℝ) (hψ : Real.sin (ψ 0) = 0)
    (i : Idx 2) :
    HasFDerivAt (seqVelocity 2 cuspEnergy (fun _ _ => 1) i)
      (Real.cos (ψ i) • (ContinuousLinearMap.proj i : (Idx 2 → ℝ) →L[ℝ] ℝ)) ψ := by
  fin_cases i
  · change HasFDerivAt (seqVelocity 2 cuspEnergy (fun _ _ => 1) 0)
      (Real.cos (ψ 0) • (ContinuousLinearMap.proj 0 : (Idx 2 → ℝ) →L[ℝ] ℝ)) ψ
    have heq : seqVelocity 2 cuspEnergy (fun _ _ => 1) 0 =
        fun η : Idx 2 → ℝ => Real.sin (η 0) := funext seqVelocity_cuspEnergy_zero
    rw [heq]
    exact (hasFDerivAt_apply 0 ψ).sin
  · change HasFDerivAt (seqVelocity 2 cuspEnergy (fun _ _ => 1) 1)
      (Real.cos (ψ 1) • (ContinuousLinearMap.proj 1 : (Idx 2 → ℝ) →L[ℝ] ℝ)) ψ
    have heq : seqVelocity 2 cuspEnergy (fun _ _ => 1) 1 =
        fun η : Idx 2 → ℝ => Real.sin (η 1) - Real.sin (η 0) ^ 2 * cuspForce (η 1) :=
      funext seqVelocity_cuspEnergy_one
    rw [heq]
    simpa only [sub_zero] using ((hasFDerivAt_apply 1 ψ).sin).fun_sub (hasFDerivAt_cuspCoupling_zero ψ hψ)

/-- The actual truncated Jacobian is the computed cosine diagonal matrix.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 2. -/
theorem seqJacobian_cuspEnergy_eq_diagonal (k : Idx 2) (ψ : Idx 2 → ℝ)
    (hψ : Real.sin (ψ 0) = 0) :
    seqJacobian 2 cuspEnergy (fun _ _ => 1) k ψ =
      Matrix.diagonal (fun i : Prefix 2 k => Real.cos (ψ i.1)) := by
  ext i j
  unfold seqJacobian
  rw [(hasFDerivAt_seqVelocity_cuspEnergy ψ hψ i.1).fderiv]
  simp only [smul_apply, ContinuousLinearMap.proj_apply, smul_eq_mul,
    Pi.single_apply, Matrix.diagonal_apply]
  by_cases hij : i = j
  · subst j
    simp
  · have hval : i.1 ≠ j.1 := fun h => hij (Subtype.ext h)
    simp [hij, hval]

/-- The original complex characteristic polynomial has these actual linear factors.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, strong stability. -/
theorem seqJacobian_cuspEnergy_charpoly_eval (k : Idx 2) (ψ : Idx 2 → ℝ)
    (hψ : Real.sin (ψ 0) = 0) (z : ℂ) :
    (((seqJacobian 2 cuspEnergy (fun _ _ => 1) k ψ).map (fun r : ℝ => (r : ℂ))).charpoly).eval z =
      ∏ i : Prefix 2 k, (z - (Real.cos (ψ i.1) : ℂ)) := by
  rw [seqJacobian_cuspEnergy_eq_diagonal k ψ hψ,
    Matrix.diagonal_map (by norm_num : ((0 : ℝ) : ℂ) = 0), Matrix.charpoly_diagonal,
    Polynomial.eval_prod]
  simp only [Polynomial.eval_sub, Polynomial.eval_X, Polynomial.eval_C]

/-- The original target tuple has the actual positive eigenvalue one.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, strong instability. -/
theorem stronglyUnstablePrefix_cuspEnergy_target :
    StronglyUnstablePrefix 2 cuspEnergy (fun _ _ => 1) 1 (![π, 0] : Idx 2 → ℝ) := by
  refine ⟨1, ?_, by norm_num⟩
  rw [seqJacobian_cuspEnergy_charpoly_eval 1 _ (by simp)]
  apply Finset.prod_eq_zero_iff.mpr
  refine ⟨⟨1, le_rfl⟩, Finset.mem_univ _, ?_⟩
  simp

/-- The actual target cannot satisfy the original strong-stability predicate.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, claimed limit. -/
theorem not_stronglyStablePrefix_cuspEnergy_target :
    ¬ StronglyStablePrefix 2 cuspEnergy (fun _ _ => 1) 1 (![π, 0] : Idx 2 → ℝ) := by
  intro hs
  obtain ⟨z, hz, hpos⟩ := stronglyUnstablePrefix_cuspEnergy_target
  exact (not_lt_of_ge hpos.le) (hs z hz)

/-- The actual critical tuple `(π,0)` jointly realizes the first sine
condition and both genuine velocity derivatives. No local smoothness
of the cusp away from the critical slice is assumed. -/
example : Real.sin ((![π, 0] : Idx 2 → ℝ) 0) = 0 ∧
    ∀ i : Idx 2, HasFDerivAt (seqVelocity 2 cuspEnergy (fun _ _ => 1) i)
      (Real.cos ((![π, 0] : Idx 2 → ℝ) i) •
        (ContinuousLinearMap.proj i : (Idx 2 → ℝ) →L[ℝ] ℝ)) (![π, 0] : Idx 2 → ℝ) := by
  have hψ : Real.sin ((![π, 0] : Idx 2 → ℝ) 0) = 0 := by simp
  exact ⟨hψ, hasFDerivAt_seqVelocity_cuspEnergy _ hψ⟩

end Transformer.Causal
