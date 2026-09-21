/-
# The emergence of clusters in self-attention dynamics — `l:fj`

§9 of arXiv:2305.05465v6: along `e:Rres`, for an eigenfunctional `φ*_k` with
`λ_k ≥ 0`, the largest coordinate `max_j φ*_k(z_j(t))` does not increase and
the smallest does not decrease.

The source argues at a time and an index realizing the extremum.  What makes
that an argument about the extremum is `antitone_sup'_of_hasDerivAt`, in
`Clusters.Extremum`.

Source: arXiv:2305.05465v6, `l:fj`.
-/

import Transformer.Clusters.Section9_Eigen
import Transformer.Clusters.Extremum

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

/-- A row of the attention matrix averages: `‖Σ_k P_jk v_k‖ ≤ max_k ‖v_k‖`. -/
theorem norm_sum_attentionMatrix_smul_le {d n : ℕ} {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (Q K : ParamMatrix d) (Y : Idx n → EucSpace d) (j : Idx n) (v : Idx n → E) {B : ℝ}
    (hv : ∀ k, ‖v k‖ ≤ B) : ‖∑ k : Idx n, attentionMatrix Q K Y j k • v k‖ ≤ B := by
  refine (norm_sum_le _ _).trans ?_
  calc ∑ k : Idx n, ‖attentionMatrix Q K Y j k • v k‖
      ≤ ∑ k : Idx n, attentionMatrix Q K Y j k * B := by
        refine Finset.sum_le_sum fun k _ => ?_
        rw [norm_smul, Real.norm_of_nonneg (attentionMatrix_pos Q K Y j k).le]
        exact mul_le_mul_of_nonneg_left (hv k) (attentionMatrix_pos Q K Y j k).le
    _ = B := by rw [← Finset.sum_mul, sum_attentionMatrix (Fin.pos_iff_nonempty.mpr ⟨j⟩), one_mul]

variable {d m : ℕ}

/-- `min_j φ(x_j) = -max_j (-φ(x_j))`. -/
theorem minCoord_eq_neg (f : EucSpace d →L[ℝ] ℝ) (X : Idx (m + 1) → EucSpace d) :
    minCoord f X = -Finset.univ.sup' Finset.univ_nonempty fun j => -f (X j) := by
  refine le_antisymm (le_neg.2 (Finset.sup'_le _ _ fun j _ => neg_le_neg (minCoord_le f X j)))
    (Finset.le_inf' _ _ fun j _ =>
      neg_le.1 (Finset.le_sup' (fun k => -f (X k)) (Finset.mem_univ j)))

/-- **Lemma (l:fj).**  If `λ_k ≥ 0` then `t ↦ max_j φ*_k(z_j(t))` is
non-increasing on `[0,+∞)` and `t ↦ min_j φ*_k(z_j(t))` is non-decreasing
there.

The source's argument: at a time `t` and an index `i` realizing the minimum,
`eq:phistarvar` gives `d/dt φ*_k(z_i(t)) = λ_k Σ_j P_ij (φ*_k(z_j) - φ*_k(z_i)) ≥ 0`.
That the extremum is then monotone is `antitone_sup'_of_hasDerivAt`, where the
right slope of the maximum is controlled by the indices attaining it.

Source: arXiv:2305.05465v6, `l:fj`. -/
theorem maxCoord_antitoneOn_minCoord_monotoneOn (Q K V : ParamMatrix d)
    (f : EucSpace d →L[ℝ] ℝ) (lam : ℝ) (hf : IsEigenFunctional V f lam) (hlam : 0 ≤ lam)
    (Z : ℝ → Idx (m + 1) → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    AntitoneOn (fun t => maxCoord f (Z t)) (Set.Ici 0) ∧
      MonotoneOn (fun t => minCoord f (Z t)) (Set.Ici 0) := by
  have hd := hasDerivAt_eigenFunctional Q K V f lam hf Z hZ
  refine ⟨(antitone_sup'_of_hasDerivAt (fun i t => f (Z t i)) _ (fun i t => hd t i)
    fun i t hi => ?_).antitoneOn _, fun s _ t _ hst => ?_⟩
  · refine mul_nonpos_of_nonneg_of_nonpos hlam (Finset.sum_nonpos fun j _ => ?_)
    exact mul_nonpos_of_nonneg_of_nonpos (attentionMatrix_pos _ _ _ _ _).le (sub_nonpos.2 (hi j))
  · have hN := antitone_sup'_of_hasDerivAt (fun i t => -f (Z t i)) _ (fun i t => (hd t i).neg)
      (fun i t hi => ?_) hst
    · simp only [minCoord_eq_neg]
      exact neg_le_neg hN
    · refine neg_nonpos.2 (mul_nonneg hlam (Finset.sum_nonneg fun j _ => ?_))
      exact mul_nonneg (attentionMatrix_pos _ _ _ _ _).le (sub_nonneg.2 (neg_le_neg_iff.1 (hi j)))

/-- The hypotheses of `maxCoord_antitoneOn_minCoord_monotoneOn` are
satisfiable. -/
example (f : EucSpace d →L[ℝ] ℝ) (z : EucSpace d) :
    IsEigenFunctional (1 : ParamMatrix d) f 1 ∧ (0 : ℝ) ≤ 1 ∧
      RescaledDynamics (n := m + 1) (1 : ParamMatrix d) 1 1 (fun _ _ => z) :=
  ⟨isEigenFunctional_one f, zero_le_one, rescaledDynamics_one_const _ _ z⟩

end Clusters
end Transformer
