module

public import Transformer.Normalization.Section3_LogarithmicClock
public import Mathlib.Analysis.Calculus.Deriv.MeanValue
public import Mathlib.Order.LiminfLimsup

/-!
# §3 — Positive radial lower limits give compact rescaled radii

Karagodin, Polyanskiy, Rigollet, arXiv:2510.22026v2, supplement,
Appendix D, proof of Theorem 3.1, Step 1.

The source passes from `liminf r_j' > 0` to positive upper and lower bounds
on `q_j=r_j/t`. The mean value theorem proves this from the original radial
equations and bounded velocities. Finite token sets give one common time
and one positive lower bound. The lower-limit argument explicitly verifies
eventual boundedness, so it does not rely on a default value of `liminf`.

No initial-radius sign is assumed: the positive velocity lower limit itself
makes the radius positive at all sufficiently late times.
-/

@[expose] public section

open Real Set Filter
open scoped BigOperators

namespace Transformer.Normalization

/-- A radial velocity bounded above by one gives `r(t) ≤ r(0)+t`.
The radius need not be positive, and the one-sided equation at zero suffices.
Source: arXiv:2510.22026v2, Appendix D, Step 1, upper bounded rescaled radii.
This isolates the upper estimate from the separate positive-lower-limit
assumption used below. -/
theorem radius_le_initial_add_time (r f : ℝ → ℝ)
    (hr : ∀ t, 0 ≤ t → HasDerivWithinAt r (f t) (Ici 0) t)
    (hf : ∀ t, 0 ≤ t → f t ≤ 1) (t : ℝ) (ht : 0 ≤ t) : r t ≤ r 0 + t := by
  have h := (convex_Ici (0 : ℝ)).image_sub_le_mul_sub_of_deriv_le
    (fun u hu => (hr u hu).continuousWithinAt)
    (fun u hu => (hr u (interior_subset hu)).differentiableWithinAt.mono interior_subset)
    (C := 1) (fun u hu => ?_) 0 le_rfl t ht ht
  · linarith
  · have hu' : 0 < u := by simpa only [interior_Ici, mem_Ioi] using hu
    rw [((hr u hu'.le).hasDerivAt (Ici_mem_nhds hu')).deriv]
    exact hf u hu'.le

/-- The actual singleton radius `1+t` attains the upper estimate. -/
example : (∀ t : ℝ, 0 ≤ t → HasDerivWithinAt (fun u : ℝ => 1 + u) 1 (Ici 0) t) ∧
    (∀ t : ℝ, 0 ≤ t → (fun _ : ℝ => (1 : ℝ)) t ≤ 1) ∧ (0 : ℝ) ≤ 1 :=
  ⟨fun t _ => ((hasDerivAt_id t).const_add 1).hasDerivWithinAt,
    fun _ _ => le_rfl, zero_le_one⟩

/-- A forward scalar radius with `|r'| ≤ 1` and positive derivative lower
limit satisfies `a ≤ r(t)/t ≤ b` eventually, for some `a > 0`.
Source: arXiv:2510.22026v2, Appendix D, Step 1, linear radial growth. -/
theorem eventually_ratio_bounds_of_pos_liminf (r f : ℝ → ℝ)
    (hr : ∀ t, 0 ≤ t → HasDerivWithinAt r (f t) (Ici 0) t)
    (hf : ∀ t, 0 ≤ t → |f t| ≤ 1) (hlim : 0 < liminf f atTop) :
    ∃ a b : ℝ, 0 < a ∧ ∀ᶠ t in atTop, a ≤ r t / t ∧ r t / t ≤ b := by
  let δ := liminf f atTop / 2
  have hδ : 0 < δ := half_pos hlim
  have hbdd : IsBoundedUnder (· ≥ ·) atTop f :=
    isBoundedUnder_of_eventually_ge ((eventually_ge_atTop (0 : ℝ)).mono fun t ht =>
      (abs_le.mp (hf t ht)).1)
  have hδlim : δ < liminf f atTop := by dsimp [δ]; linarith
  obtain ⟨s, hs⟩ := eventually_atTop.mp (eventually_lt_of_lt_liminf hδlim hbdd)
  let S := max s 1
  have hS : 0 < S := lt_of_lt_of_le zero_lt_one (le_max_right _ _)
  have hlow (t : ℝ) (ht : S ≤ t) : δ * (t - S) ≤ r t - r S := by
    have hder (u : ℝ) (hu : S ≤ u) : HasDerivAt r (f u) u :=
      (hr u (hS.le.trans hu)).hasDerivAt (Ici_mem_nhds (hS.trans_le hu))
    apply (convex_Ici S).mul_sub_le_image_sub_of_le_deriv
      (fun u hu => (hder u hu).continuousAt.continuousWithinAt)
      (fun u hu => (hder u (interior_subset hu)).differentiableAt.differentiableWithinAt)
      (fun u hu => ?_) S le_rfl t ht ht
    rw [(hder u (interior_subset hu)).deriv]
    exact (hs u ((le_max_left _ _).trans (interior_subset hu))).le
  have hupp (t : ℝ) (ht : 0 ≤ t) : r t - r 0 ≤ t := by
    linarith [radius_le_initial_add_time r f hr (fun u hu => (abs_le.mp (hf u hu)).2) t ht]
  refine ⟨δ / 2, |r 0| + 1, half_pos hδ, ?_⟩
  filter_upwards [eventually_ge_atTop S, eventually_ge_atTop (1 : ℝ),
    eventually_ge_atTop (2 * |r S - δ * S| / δ)] with t htS ht1 htδ
  have ht : 0 < t := zero_lt_one.trans_le ht1
  have hδt := (div_le_iff₀ hδ).mp htδ
  constructor
  · rw [le_div_iff₀ ht]
    nlinarith [hlow t htS, neg_abs_le (r S - δ * S)]
  · rw [div_le_iff₀ ht]
    nlinarith [hupp t ht.le, le_abs_self (r 0),
      mul_nonneg (abs_nonneg (r 0)) (sub_nonneg.mpr ht1)]

/-- The scalar hypotheses hold for the genuine radial law `r(t)=1+t`,
with constant derivative one. -/
example : (∀ t : ℝ, 0 ≤ t → HasDerivWithinAt (fun u : ℝ => 1 + u) 1 (Ici 0) t) ∧
    (∀ t : ℝ, 0 ≤ t → |(fun _ : ℝ => (1 : ℝ)) t| ≤ 1) ∧
    0 < liminf (fun _ : ℝ => (1 : ℝ)) atTop := by
  exact ⟨fun t _ => ((hasDerivAt_id t).const_add 1).hasDerivWithinAt,
    fun _ _ => by norm_num, by simp⟩

/-- Finitely many positive radial lower limits supply common bounds
`a ≤ q_j(u) ≤ b` in one logarithmic clock, for every `u ≥ 0` and every token.
Source: arXiv:2510.22026v2, Appendix D, Step 1 and the event in Step 2. -/
theorem exists_rescaled_bounds_of_pos_liminf {n : ℕ} (hn : 0 < n)
    (r f : ℝ → Idx n → ℝ)
    (hr : ∀ t, 0 ≤ t → ∀ j, HasDerivWithinAt (fun u => r u j) (f t j) (Ici 0) t)
    (hf : ∀ t, 0 ≤ t → ∀ j, |f t j| ≤ 1)
    (hlim : ∀ j, 0 < liminf (fun t => f t j) atTop) :
    ∃ a b T : ℝ, 0 < a ∧ a ≤ b ∧ 0 < T ∧
      ∀ u, 0 ≤ u → ∀ j, a ≤ rescaledRadius T r u j ∧ rescaledRadius T r u j ≤ b := by
  choose a b ha hb using fun j => eventually_ratio_bounds_of_pos_liminf
    (fun t => r t j) (fun t => f t j) (fun t ht => hr t ht j) (fun t ht => hf t ht j) (hlim j)
  let A := (∑ j, (a j)⁻¹)⁻¹
  let B := (∑ j, |b j|) + 1
  have hA : 0 < A := inv_pos.mpr (Finset.sum_pos (fun j _ => inv_pos.mpr (ha j))
    ⟨⟨0, hn⟩, Finset.mem_univ _⟩)
  have hAa (j : Idx n) : A ≤ a j := by
    have hsum : (a j)⁻¹ ≤ ∑ i, (a i)⁻¹ := Finset.single_le_sum
      (fun i _ => (inv_pos.mpr (ha i)).le) (Finset.mem_univ j)
    simpa only [A, inv_inv] using inv_anti₀ (inv_pos.mpr (ha j)) hsum
  have hbB (j : Idx n) : b j ≤ B := by
    have hsum : |b j| ≤ ∑ i, |b i| := Finset.single_le_sum (f := fun i => |b i|)
      (fun i _ => abs_nonneg (b i)) (Finset.mem_univ j)
    dsimp [B]
    linarith [le_abs_self (b j)]
  have hev : ∀ᶠ t in atTop, ∀ j, A ≤ r t j / t ∧ r t j / t ≤ B := by
    filter_upwards [eventually_all.mpr hb] with t ht j
    exact ⟨(hAa j).trans (ht j).1, (ht j).2.trans (hbB j)⟩
  obtain ⟨s, hs⟩ := eventually_atTop.mp hev
  let T := max s 1
  have hT : 0 < T := lt_of_lt_of_le zero_lt_one (le_max_right _ _)
  have hsT : s ≤ T := le_max_left _ _
  have hAB : A ≤ B := (hs T hsT ⟨0, hn⟩).1.trans (hs T hsT ⟨0, hn⟩).2
  refine ⟨A, B, T, hA, hAB, hT, fun u hu j => ?_⟩
  have htime : T ≤ logTime T u := by
    have he := Real.one_le_exp_iff.mpr hu
    dsimp [logTime]
    nlinarith
  exact hs _ (hsT.trans htime) j

/-- The common-bound hypotheses hold for one actual radius `1+t`.
The sample size, derivative bound and positive lower limit are all realized. -/
example : 0 < 1 ∧
    (∀ t : ℝ, 0 ≤ t → ∀ j : Idx 1,
      HasDerivWithinAt (fun u : ℝ => (fun s _ => 1 + s) u j) 1 (Ici 0) t) ∧
    (∀ t : ℝ, 0 ≤ t → ∀ j : Idx 1, |(fun _ _ => (1 : ℝ)) t j| ≤ 1) ∧
    ∀ j : Idx 1, 0 < liminf (fun t : ℝ => (fun _ (_ : Idx 1) => (1 : ℝ)) t j) atTop := by
  exact ⟨one_pos, fun t _ _ => ((hasDerivAt_id t).const_add 1).hasDerivWithinAt,
    fun _ _ _ => by norm_num, fun _ => by simp⟩

end Transformer.Normalization
