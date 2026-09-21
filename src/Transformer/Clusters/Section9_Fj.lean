/-
# The emergence of clusters in self-attention dynamics — `l:fj`

§9 of arXiv:2305.05465v6: along `e:Rres`, for an eigenfunctional `φ*_k` with
`λ_k ≥ 0`, the largest coordinate `max_j φ*_k(z_j(t))` does not increase and
the smallest does not decrease.

The source argues at a time and an index realizing the extremum.  What makes
that an argument about the extremum is `antitone_sup'_of_hasDerivAt`: the
maximum of finitely many differentiable functions does not increase when each
has a nonpositive derivative wherever it attains the maximum.  It is proved by
Mathlib's fencing theorem for right slopes, since the maximum itself need not
be differentiable.

Source: arXiv:2305.05465v6, `l:fj`.
-/

import Transformer.Clusters.Section9_Eigen
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Topology.Order.Lattice

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

/-- The maximum of finitely many differentiable functions does not increase if
each of them has a nonpositive derivative whenever it attains the maximum. -/
theorem antitone_sup'_of_hasDerivAt {ι : Type*} [Fintype ι] [Nonempty ι]
    (g g' : ι → ℝ → ℝ) (hg : ∀ i t, HasDerivAt (g i) (g' i t) t)
    (hact : ∀ i t, (∀ j, g j t ≤ g i t) → g' i t ≤ 0) :
    Antitone fun t => Finset.univ.sup' Finset.univ_nonempty fun i => g i t := by
  set M := fun t => Finset.univ.sup' Finset.univ_nonempty fun i => g i t
  have hcont : Continuous M :=
    Continuous.finset_sup'_apply _ fun i _ => continuous_iff_continuousAt.2 fun t =>
      (hg i t).continuousAt
  intro a b hab
  refine image_le_of_liminf_slope_right_le_deriv_boundary (f := M) (B := fun _ => M a)
    (B' := fun _ => 0) hcont.continuousOn le_rfl continuousOn_const
    (fun x _ => hasDerivWithinAt_const _ _ _) (fun x _ r hr => ?_) ⟨hab, le_rfl⟩
  refine Eventually.frequently ?_
  have hi : ∀ i, ∀ᶠ z in 𝓝[>] x, g i z < M x + r * (z - x) := by
    intro i
    by_cases hmax : g i x = M x
    · have hle : g' i x ≤ 0 := hact i x fun j =>
        hmax ▸ Finset.le_sup' (fun k => g k x) (Finset.mem_univ j)
      have ht := (hasDerivAt_iff_tendsto_slope.1 (hg i x)).mono_left
        (nhdsWithin_mono x (fun z (hz : x < z) => hz.ne'))
      filter_upwards [ht.eventually (gt_mem_nhds (lt_of_le_of_lt hle hr)),
        self_mem_nhdsWithin] with z hz (hxz : x < z)
      rw [slope_def_field, div_lt_iff₀ (sub_pos.2 hxz)] at hz
      linarith
    · have hlt : g i x < M x :=
        lt_of_le_of_ne (Finset.le_sup' (fun k => g k x) (Finset.mem_univ i)) hmax
      have hc := ((hg i x).continuousAt.eventually (gt_mem_nhds hlt))
      filter_upwards [nhdsWithin_le_nhds hc, self_mem_nhdsWithin] with z hz (hxz : x < z)
      nlinarith
  filter_upwards [eventually_all.2 hi, self_mem_nhdsWithin] with z hz (hxz : x < z)
  rw [slope_def_field, div_lt_iff₀ (sub_pos.2 hxz)]
  have : M z < M x + r * (z - x) := (Finset.sup'_lt_iff _).2 fun i _ => hz i
  linarith

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
