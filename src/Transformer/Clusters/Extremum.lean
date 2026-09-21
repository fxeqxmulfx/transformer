/-
# The maximum of finitely many differentiable functions

The tool behind every "at an index realizing the maximum" argument of
arXiv:2305.05465v6 (`p:noninc`, `l:fj`, `l:nottoofassst`):
`antitone_sup'_of_hasDerivAt`, the maximum of finitely many differentiable
functions does not increase when each has a nonpositive derivative wherever it
attains the maximum.  It is proved by Mathlib's fencing theorem for right
slopes, since the maximum itself need not be differentiable.  Grönwall for the
maximum, `le_sup'_mul_exp_of_hasDerivAt`, follows.

Source: arXiv:2305.05465v6, the proofs of `p:noninc`, `l:fj`,
`l:nottoofassst`.
-/

import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Topology.Order.Lattice
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Filter Topology

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

/-- Grönwall for the maximum: if each `q_i` grows at rate at most `c q_i`
wherever it attains the maximum, then `q_i(t) ≤ max_j q_j(0) e^{ct}` for `t ≥ 0`. -/
theorem le_sup'_mul_exp_of_hasDerivAt {ι : Type*} [Fintype ι] [Nonempty ι]
    (q q' : ι → ℝ → ℝ) (c : ℝ) (hq : ∀ i t, HasDerivAt (q i) (q' i t) t)
    (hact : ∀ i t, (∀ j, q j t ≤ q i t) → q' i t ≤ c * q i t) (i : ι) {t : ℝ} (ht : 0 ≤ t) :
    q i t ≤ (Finset.univ.sup' Finset.univ_nonempty fun j => q j 0) * Real.exp (c * t) := by
  have hg : ∀ j s, HasDerivAt (fun s => Real.exp (-c * s) * q j s)
      (Real.exp (-c * s) * (q' j s - c * q j s)) s := by
    intro j s
    have := (((hasDerivAt_id s).const_mul (-c)).exp).mul (hq j s)
    convert this using 1
    · rfl
    · simp only [id, mul_one]; ring
  have hA := antitone_sup'_of_hasDerivAt _ _ hg (fun j s hj => by
    have hk : ∀ k, q k s ≤ q j s := fun k => le_of_mul_le_mul_left (hj k) (Real.exp_pos _)
    exact mul_nonpos_of_nonneg_of_nonpos (Real.exp_pos _).le
      (sub_nonpos.2 (hact j s hk))) ht
  simp only [mul_zero, Real.exp_zero, one_mul] at hA
  have h1 := (Finset.le_sup' (fun j => Real.exp (-c * t) * q j t) (Finset.mem_univ i)).trans hA
  have he : Real.exp (c * t) * Real.exp (-c * t) = 1 := by
    rw [← Real.exp_add]; simp
  calc q i t = Real.exp (c * t) * (Real.exp (-c * t) * q i t) := by rw [← mul_assoc, he, one_mul]
    _ ≤ _ := by rw [mul_comm _ (Real.exp _)]; exact mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le

/-- Taking square roots in a bound `x² ≤ S e^{2a}`, with room to make the
constant positive. -/
theorem le_sqrt_add_one_mul_exp {x S a : ℝ} (hx : 0 ≤ x) (hS : 0 ≤ S)
    (h : x ^ 2 ≤ S * Real.exp (2 * a)) : x ≤ (Real.sqrt S + 1) * Real.exp a := by
  have hp : 0 ≤ (Real.sqrt S + 1) * Real.exp a := by positivity
  refine (pow_le_pow_iff_left₀ hx hp two_ne_zero).1 (h.trans ?_)
  have e2 : Real.exp (2 * a) = Real.exp a ^ 2 := by rw [sq, ← Real.exp_add]; ring_nf
  have hs := Real.sq_sqrt hS
  have := Real.sqrt_nonneg S
  rw [e2, mul_pow]
  exact mul_le_mul_of_nonneg_right (by nlinarith) (by positivity)

/-- `d/dt |w(t)|² = 2 Re(conj(w) w')` for a curve in `ℝ` or `ℂ`. -/
theorem hasDerivAt_norm_sq_rclike {𝕜 : Type*} [RCLike 𝕜] {w : ℝ → 𝕜} {w' : 𝕜} {t : ℝ}
    (hw : HasDerivAt w w' t) :
    HasDerivAt (fun s => ‖w s‖ ^ 2)
      (2 * RCLike.re ((starRingEnd 𝕜) (w t) * w')) t := by
  have hre := (RCLike.reCLM (K := 𝕜)).hasFDerivAt.comp_hasDerivAt t hw
  have him := (RCLike.imCLM (K := 𝕜)).hasFDerivAt.comp_hasDerivAt t hw
  simp only [Function.comp_def, RCLike.reCLM_apply, RCLike.imCLM_apply] at hre him
  have := (hre.mul hre).add (him.mul him)
  convert this using 1
  · funext s; exact RCLike.norm_sq_eq_def
  · simp only [RCLike.mul_re, RCLike.conj_re, RCLike.conj_im]; ring

end Clusters
end Transformer
