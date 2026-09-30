/-
# DASH — the missing sample-count restriction

arXiv:2602.02016v2, Appendix A, coefficient-fitting algorithm.
The mode at the sample count vanishes at every fitting node, so the
source's orthogonality explanation requires a degree below that count.
-/

import Transformer.DASH.SectionA_FitReproduction

noncomputable section

namespace Transformer.DASH

/-- The mode of degree `N` vanishes at every one of the source's `N`
fitting angles. Source: arXiv:2602.02016v2, Appendix A, midpoint cosine nodes. -/
theorem chebAngle_cos_sample_count (N : ℕ) (i : Fin N) :
    Real.cos ((N : ℝ) * chebAngle N i) = 0 := by
  have hN : 0 < N := Nat.zero_lt_of_lt i.isLt
  have hn : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  have harg : (N : ℝ) * chebAngle N i = Real.pi / 2 + (i.val : ℝ) * Real.pi := by
    unfold chebAngle
    field_simp
    ring
  rw [harg, Real.cos_add_nat_mul_pi, Real.cos_pi_div_two, mul_zero]

/-- The fitting algorithm always sets its degree-`N` coefficient to zero,
regardless of the sampled function. Thus no halving of the constant
coefficient makes all degrees orthogonal without a sampling restriction.
Source: arXiv:2602.02016v2, Appendix A, the orthogonality comment in coefficient fitting. -/
theorem chebFitCoefficient_sample_count_zero (f : ℝ → ℝ) (a b : ℝ) (N : ℕ) :
    chebFitCoefficient f a b N N = 0 := by
  simp [chebFitCoefficient, chebAngle_cos_sample_count]

/-- The polynomial `T_N` is completely invisible at the `N` fitting nodes
after mapping to any nondegenerate interval.
Source: arXiv:2602.02016v2, Appendix A, the scalar fitting map. -/
theorem cheb_sample_count_nodes_zero (a b : ℝ) (N : ℕ) (hab : a < b) (i : Fin N) :
    chebT (chebToCoordinate a b (chebNode a b N i)) N = 0 := by
  rw [chebNode, chebCoordinate_reverse a b _ hab, chebT_cos,
    chebAngle_cos_sample_count]

/-- Nondegenerate sampling intervals exist,
arXiv:2602.02016v2, Appendix A. -/
example : (-1 : ℝ) < 1 := by norm_num

/-- With `d=N`, fitting `T_N` on `[-1,1]` returns zero at the endpoint
where `T_N(1)=1`. This refutes unrestricted exact polynomial recovery and
identifies the missing condition `d<N` in the source's orthogonality claim.
Source: arXiv:2602.02016v2, Appendix A, coefficient fitting and scalar Clenshaw. -/
theorem chebFit_sample_count_counterexample (N : ℕ) :
    clenshaw (1 : ℝ) (chebFit (fun x => chebT x N) (-1) 1 N N) = 0 ∧
      chebT (1 : ℝ) N = 1 := by
  have hsamples (i : Fin N) : chebT (chebNode (-1) 1 N i) N = 0 := by
    have h := cheb_sample_count_nodes_zero (-1) 1 N (by norm_num) i
    have hcoord : chebToCoordinate (-1) 1 (chebNode (-1) 1 N i) =
        chebNode (-1) 1 N i := by unfold chebToCoordinate; ring
    rwa [hcoord] at h
  constructor
  · rw [chebFit, clenshaw_ofFn]
    simp [chebFitCoefficient, hsamples]
  · rw [chebT_eq_eval]
    simp

end Transformer.DASH
