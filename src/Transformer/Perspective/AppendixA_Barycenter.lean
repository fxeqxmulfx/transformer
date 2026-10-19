module

public import Transformer.Perspective.AppendixA_CriticalShape

/-!
# Appendix A — the barycentre along gradient ascent

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A.

`𝖤_0` is `‖s‖² / n` for the barycentre `s = Σ_j x_j`, and it increases along
`e:gradfl`, so `‖s‖` never shrinks. This file draws from that the two cases of
`l:nosaddleconv` that need no stable-manifold argument:

* a trajectory converging to a configuration with `s = 0` had `s = 0` from the
  start (`sum_eq_zero_of_tendsto`);
* a trajectory converging to a configuration in which two particles sit at one
  point *opposite* to the barycentre started with those two particles equal
  (`coord_eq_of_tendsto`): `D = 1 - ⟨x_i, x_j⟩` solves
  `D' = -(2/n)(⟨x_i, s⟩ + ⟨x_j, s⟩) D`, whose coefficient is eventually
  positive, so `D` can tend to `0` only if it vanishes.
-/

@[expose] public section

open scoped BigOperators Topology
open Real Filter Set

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- `𝖤_0` is the squared length of the barycentre, divided by `n`. -/
theorem E0_eq_norm_sum_sq (X : SphereTuple d n) :
    E0 d n X = (n : ℝ)⁻¹ * ‖∑ j : Idx n, (X j : EucSpace d)‖ ^ 2 := by
  rw [E0, ← real_inner_self_eq_norm_sq, sum_inner]
  simp_rw [inner_sum]

/-- The length of the barycentre is nondecreasing along gradient ascent. -/
theorem monotone_norm_sum_sq (X : ℝ → SphereTuple d n) (hX : E0GradientAscent d n X) :
    Monotone fun t => ‖∑ j : Idx n, (X t j : EucSpace d)‖ ^ 2 := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · intro s t _
    simp
  have hmono : Monotone fun t => E0 d n (X t) :=
    monotone_of_hasDerivAt_nonneg (hasDerivAt_E0_ascent d n X hX) fun t => by
      simp only [Pi.zero_apply]
      positivity
  intro s t hst
  have h := hmono hst
  simp only [E0_eq_norm_sum_sq] at h
  exact le_of_mul_le_mul_left h (by positivity)

/-- **Case `s = 0`.**  A gradient-ascent trajectory converging to a
configuration with zero barycentre has zero barycentre at time `0`: the
nondecreasing `‖s(t)‖²` tends to `0`. -/
theorem sum_eq_zero_of_tendsto (X : ℝ → SphereTuple d n) (hX : E0GradientAscent d n X)
    (Z : SphereTuple d n) (hZ : ∑ j : Idx n, (Z j : EucSpace d) = 0)
    (hlim : ∀ i : Idx n,
      Tendsto (fun t => (X t i : EucSpace d) - (Z i : EucSpace d)) atTop (𝓝 0)) :
    ∑ j : Idx n, (X 0 j : EucSpace d) = 0 := by
  have hsum : Tendsto (fun t => ‖∑ j : Idx n, (X t j : EucSpace d)‖ ^ 2) atTop (𝓝 0) := by
    have h := tendsto_finsetSum Finset.univ fun j _ => tendsto_sub_nhds_zero_iff.mp (hlim j)
    rw [hZ] at h
    simpa using h.norm.pow 2
  have hle : ‖∑ j : Idx n, (X 0 j : EucSpace d)‖ ^ 2 ≤ 0 :=
    ge_of_tendsto hsum (eventually_atTop.mpr ⟨0, fun t ht => monotone_norm_sum_sq X hX ht⟩)
  exact norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp
    (le_antisymm hle (by positivity)))

/-- `d/dt ⟨x_i, x_j⟩` when both particles follow the projection of one drive `v`. -/
theorem inner_proj_add_inner_proj (xi xj v : EucSpace d) :
    inner (𝕜 := ℝ) xi (proj d xj v) + inner (𝕜 := ℝ) (proj d xi v) xj
      = (inner (𝕜 := ℝ) xi v + inner (𝕜 := ℝ) xj v) * (1 - inner (𝕜 := ℝ) xi xj) := by
  simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_smul_right, real_inner_comm xj v]
  ring

/-- **Case of two particles opposite to the barycentre.**  If a
gradient-ascent trajectory converges to `Z` and `z_i = z_j` with
`⟨z_i, Σ_k z_k⟩ < 0`, then `x_i(0) = x_j(0)`. -/
theorem coord_eq_of_tendsto (X : ℝ → SphereTuple d n) (hX : E0GradientAscent d n X)
    (Z : SphereTuple d n)
    (hlim : ∀ i : Idx n,
      Tendsto (fun t => (X t i : EucSpace d) - (Z i : EucSpace d)) atTop (𝓝 0))
    {i j : Idx n} (hij : (Z i : EucSpace d) = Z j)
    (hneg : inner (𝕜 := ℝ) (Z i : EucSpace d) (∑ k : Idx n, (Z k : EucSpace d)) < 0) :
    X 0 i = X 0 j := by
  have hn : (0 : ℝ) < n := by exact_mod_cast i.pos
  set c : ℝ := 2 * (n : ℝ)⁻¹ with hc
  have hc0 : 0 < c := by positivity
  have hunit : ∀ t k, ‖(X t k : EucSpace d)‖ = 1 := fun t k =>
    mem_sphere_zero_iff_norm.mp (X t k).2
  set s : ℝ → EucSpace d := fun t => ∑ k : Idx n, (X t k : EucSpace d) with hs
  set a : ℝ → ℝ := fun t => inner (𝕜 := ℝ) (X t i : EucSpace d) (s t)
    + inner (𝕜 := ℝ) (X t j : EucSpace d) (s t) with ha
  set D : ℝ → ℝ := fun t => 1 - inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d)
    with hDdef
  have hD : ∀ t, HasDerivAt D (-(c * a t) * D t) t := by
    intro t
    refine ((hX t i).inner ℝ (hX t j)).const_sub 1 |>.congr_deriv ?_
    rw [inner_proj_add_inner_proj, real_inner_smul_right, real_inner_smul_right]
    simp only [ha, hDdef, hs]
    ring
  have hD0 : ∀ t, 0 ≤ D t := fun t => by
    have := real_inner_le_norm (X t i : EucSpace d) (X t j : EucSpace d)
    rw [hunit, hunit, one_mul] at this
    simp only [hDdef]
    linarith
  have hnorm_s : ∀ t, ‖s t‖ ≤ n := fun t => by
    refine (norm_sum_le _ _).trans ?_
    simp [hunit]
  have hcoef : ∀ t, |c * a t| ≤ 4 := fun t => by
    have h1 := abs_real_inner_le_norm (X t i : EucSpace d) (s t)
    have h2 := abs_real_inner_le_norm (X t j : EucSpace d) (s t)
    rw [hunit, one_mul] at h1 h2
    rw [abs_mul, abs_of_pos hc0]
    calc c * |a t| ≤ c * (2 * n) := by
          gcongr
          exact (abs_add_le _ _).trans (by linarith [hnorm_s t])
      _ = 4 := by rw [hc]; field_simp; ring
  -- the limits of the coefficient and of `D`
  have hx : ∀ k, Tendsto (fun t => (X t k : EucSpace d)) atTop (𝓝 (Z k : EucSpace d)) :=
    fun k => tendsto_sub_nhds_zero_iff.mp (hlim k)
  have hslim : Tendsto s atTop (𝓝 (∑ k : Idx n, (Z k : EucSpace d))) :=
    tendsto_finsetSum _ fun k _ => hx k
  have halim : Tendsto a atTop (𝓝 (2 * inner (𝕜 := ℝ) (Z i : EucSpace d)
      (∑ k : Idx n, (Z k : EucSpace d)))) := by
    have h := ((hx i).inner (𝕜 := ℝ) hslim).add ((hx j).inner (𝕜 := ℝ) hslim)
    rwa [← hij, ← two_mul] at h
  have hDlim : Tendsto D atTop (𝓝 0) := by
    have h := ((hx i).inner (𝕜 := ℝ) (hx j)).const_sub 1
    rwa [← hij, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (Z i).2, one_pow,
      sub_self] at h
  -- phase 1: `D` is eventually nondecreasing and tends to `0`, so it vanishes
  obtain ⟨T₀, hT₀⟩ := eventually_atTop.mp (halim.eventually (gt_mem_nhds
    (show 2 * inner (𝕜 := ℝ) (Z i : EucSpace d) (∑ k : Idx n, (Z k : EucSpace d)) < 0 by
      linarith)))
  set T := max T₀ 0 with hT
  have hmono : MonotoneOn D (Ici T) := by
    refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici T)
      (fun t _ => (hD t).continuousAt.continuousWithinAt)
      (fun t _ => (hD t).hasDerivWithinAt) fun t ht => ?_
    rw [interior_Ici] at ht
    have hat := hT₀ t (le_trans (le_max_left _ _) (le_of_lt ht))
    have : c * a t < 0 := mul_neg_of_pos_of_neg hc0 hat
    nlinarith [hD0 t]
  have hDT : D T = 0 := le_antisymm
    (ge_of_tendsto hDlim (eventually_atTop.mpr ⟨T, fun t ht => hmono (mem_Ici.mpr le_rfl) ht ht⟩)) (hD0 T)
  -- phase 2: `D e^{4t}` is nondecreasing, so `D 0 ≤ D T e^{4T} = 0`
  have hH : ∀ t, HasDerivAt (fun r => D r * Real.exp (4 * r))
      (-(c * a t) * D t * Real.exp (4 * t) + D t * (Real.exp (4 * t) * 4)) t := fun t =>
    (hD t).mul (by simpa using ((hasDerivAt_id t).const_mul 4).exp)
  have hHmono := monotone_of_hasDerivAt_nonneg hH fun t => by
    have h4 := (abs_le.mp (hcoef t)).2
    have hpos := mul_nonneg (hD0 t) (Real.exp_pos (4 * t)).le
    simp only [Pi.zero_apply]
    nlinarith
  have hD00 : D 0 = 0 := by
    have h := hHmono (show (0 : ℝ) ≤ T from le_max_right _ _)
    simp only [hDT, zero_mul, mul_zero, Real.exp_zero, mul_one] at h
    exact le_antisymm h (hD0 0)
  apply Subtype.ext
  have hsq := norm_sub_sq_real (X 0 i : EucSpace d) (X 0 j : EucSpace d)
  simp only [hDdef, sub_eq_zero] at hD00
  rw [hunit, hunit, ← hD00] at hsq
  exact sub_eq_zero.mp (norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp (by linarith)))

/-- The hypotheses of `sum_eq_zero_of_tendsto` hold for the antipodal pair at
rest: it is critical, with barycentre `0`. -/
example (x : SSphere 2) :
    E0GradientAscent 2 2 (fun _ => signedTuple x ({1} : Finset (Idx 2))) ∧
      ∑ j : Idx 2, (signedTuple x ({1} : Finset (Idx 2)) j : EucSpace 2) = 0 ∧
      ∀ i : Idx 2, Tendsto (fun _ : ℝ => (signedTuple x ({1} : Finset (Idx 2)) i : EucSpace 2)
        - (signedTuple x ({1} : Finset (Idx 2)) i : EucSpace 2)) atTop (𝓝 0) := by
  refine ⟨E0GradientAscent_const _ (isCriticalE0_signedTuple x _), ?_, fun i => ?_⟩
  · rw [sum_signedTuple]
    norm_num
  · simp

/-- The hypotheses of `coord_eq_of_tendsto` hold for `(-x, -x, x, x, x)` at
rest: the barycentre is `x` and the first two particles sit at `-x`. -/
example (x : SSphere 2) :
    E0GradientAscent 2 5 (fun _ => signedTuple x ({0, 1} : Finset (Idx 5))) ∧
      (signedTuple x ({0, 1} : Finset (Idx 5)) 0 : EucSpace 2)
        = signedTuple x ({0, 1} : Finset (Idx 5)) 1 ∧
      inner (𝕜 := ℝ) (signedTuple x ({0, 1} : Finset (Idx 5)) 0 : EucSpace 2)
        (∑ k : Idx 5, (signedTuple x ({0, 1} : Finset (Idx 5)) k : EucSpace 2)) < 0 := by
  refine ⟨E0GradientAscent_const _ (isCriticalE0_signedTuple x _), ?_, ?_⟩
  · simp [coe_signedTuple]
  · have hx : inner (𝕜 := ℝ) (x : EucSpace 2) (x : EucSpace 2) = 1 := by
      rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp x.2, one_pow]
    rw [sum_signedTuple, coe_signedTuple]
    norm_num [real_inner_smul_right, hx]

end Perspective
end Transformer
