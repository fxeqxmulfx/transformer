/-
# Scalar barriers and expansion for the causal convergence proof

ArXiv:2411.04990v2, §4, proof of `thm1`.
The scalar overlap argument uses two precise one-dimensional facts.
A positive derivative at a level prevents a downward crossing of
that level. A uniformly positive derivative throughout a closed
strip forces a bounded curve to leave that strip through its top.
These statements spell out the barrier and escape steps in the
manuscript's induction, with every derivative assumption explicit.

A third fact treats the antipodal instability: a bounded nonnegative
distance whose derivative is at least a positive constant times
that distance must vanish. It will be applied to squared distances
between two actual followers with the same original prefix, after
establishing the required expansion from the original causal field.
-/

import Transformer.Causal.Section4_SecondOverlap

open Real Set Filter

namespace Transformer.Causal

/-- A uniform scalar derivative lower bound integrates to its actual linear increment.
Source: arXiv:2411.04990v2, §4, the positive-overlap drift argument in `thm1`. -/
theorem csaScalar_linear_growth (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (c s : ℝ) (hc : ∀ t, s ≤ t → c ≤ r' t) (t : ℝ) (hst : s ≤ t) :
    c * (t - s) ≤ r t - r s := by
  apply (convex_Ici s).mul_sub_le_image_sub_of_le_deriv
    (fun x hx => (hd x).continuousAt.continuousWithinAt)
    (fun x hx => (hd x).differentiableAt.differentiableWithinAt) ?_
    s (Set.mem_Ici.mpr le_rfl) t hst hst
  intro x hx
  rw [interior_Ici] at hx
  rw [(hd x).deriv]
  exact hc x hx.le

/-- An actual positive derivative at the boundary prevents any subsequent downward crossing.
Source: arXiv:2411.04990v2, §4, the invariant-overlap levels used in the proof of `thm1`. -/
theorem csaScalar_forward_barrier (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (a T s : ℝ) (hTs : T ≤ s) (hrs : a ≤ r s)
    (hb : ∀ t, T ≤ t → r t = a → 0 < r' t) :
    ∀ t, s ≤ t → a ≤ r t := by
  intro t hst
  have h := image_le_of_deriv_right_lt_deriv_boundary
    (f := fun u => -r u) (f' := fun u => -r' u)
    (B := fun _ => -a) (B' := fun _ => 0)
    (fun x hx => (hd x).neg.continuousAt.continuousWithinAt)
    (fun x hx => (hd x).neg.hasDerivWithinAt)
    (neg_le_neg hrs) (fun x => hasDerivAt_const x (-a))
    (fun x hx he => by
      have hp := hb x (hTs.trans hx.1) (by linarith)
      linarith) (show t ∈ Icc s t from ⟨hst, le_rfl⟩)
  exact neg_le_neg_iff.mp h

/-- A genuine positive drift on a scalar strip forces a bounded overlap to its upper level permanently.
Source: arXiv:2411.04990v2, §4, the final barrier-and-escape step in `thm1`. -/
theorem csaScalar_escape_strip (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (hr : ∀ t, -1 ≤ r t ∧ r t ≤ 1) (a b T c s : ℝ) (hab : a ≤ b) (hc : 0 < c)
    (hTs : T ≤ s) (hrs : a ≤ r s)
    (hb : ∀ t, T ≤ t → a ≤ r t → r t ≤ b → c ≤ r' t) :
    ∀ᶠ t in atTop, b ≤ r t := by
  have hlo := csaScalar_forward_barrier r r' hd a T s hTs hrs
    (fun t ht he => hc.trans_le (hb t ht (by rw [he]) (by rw [he]; exact hab)))
  have hex : ∃ u, s ≤ u ∧ b ≤ r u := by
    by_contra hn
    push Not at hn
    have hlin : ∀ u, s ≤ u → c * (u - s) ≤ r u - r s :=
      csaScalar_linear_growth r r' hd c s
        (fun t ht => hb t (hTs.trans ht) (hlo t ht) (hn t ht).le)
    have htime : s ≤ s + 3 / c := by
      have hdiv : 0 ≤ 3 / c := by positivity
      linarith
    have h := hlin (s + 3 / c) htime
    have he : c * (s + 3 / c - s) = 3 := by field_simp; ring
    rw [he] at h
    linarith [(hr s).1, (hr (s + 3 / c)).2]
  obtain ⟨u, hsu, hbu⟩ := hex
  apply eventually_atTop.mpr
  refine ⟨u, csaScalar_forward_barrier r r' hd b T u (hTs.trans hsu) hbu ?_⟩
  intro t ht he
  exact hc.trans_le (hb t ht (by rw [he]; exact hab) (by rw [he]))

/-- A bounded actual nonnegative scalar distance cannot have a persistent positive relative expansion.
Source: arXiv:2411.04990v2, §4, the antipodal instability in `thm1`.
The expansion is an explicit hypothesis to be verified for the original follower field. -/
theorem csaScalar_eq_zero_of_expansion (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (M c T : ℝ) (hr : ∀ t, 0 ≤ r t ∧ r t ≤ M) (hc : 0 < c)
    (he : ∀ t, T ≤ t → c * r t ≤ r' t) : ∀ s, T ≤ s → r s = 0 := by
  have hm : MonotoneOn r (Ici T) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici T)
      (fun x hx => (hd x).continuousAt.continuousWithinAt)
      (fun x hx => (hd x).differentiableAt.differentiableWithinAt)
    intro x hx
    rw [interior_Ici] at hx
    rw [(hd x).deriv]
    exact (mul_nonneg hc.le (hr x).1).trans (he x hx.le)
  intro s hTs
  by_contra hn
  have hs : 0 < r s := lt_of_le_of_ne (hr s).1 (fun heq => hn heq.symm)
  have hp : 0 < c * r s := mul_pos hc hs
  have hM : 0 ≤ M := (hr s).1.trans (hr s).2
  let u := s + (M + 1) / (c * r s)
  have hsu : s ≤ u := by
    have hdiv : 0 ≤ (M + 1) / (c * r s) := by positivity
    dsimp [u]
    linarith
  have hlin := csaScalar_linear_growth r r' hd (c * r s) s
    (fun t ht => (mul_le_mul_of_nonneg_left (hm hTs (hTs.trans ht) ht) hc.le).trans
      (he t (hTs.trans ht))) u hsu
  have hval : c * r s * (u - s) = M + 1 := by dsimp [u]; field_simp; ring
  rw [hval] at hlin
  linarith [(hr u).2]

/-- An actual bounded overlap with eventual positive drift on every interior strip has an endpoint limit.
Source: arXiv:2411.04990v2, §4, the scalar convergence argument in the induction for `thm1`.
The drift is quantitatively bounded away from zero on each fixed strip, rather than merely positive. -/
theorem csaScalar_tendsto_endpoints (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (hr : ∀ t, -1 ≤ r t ∧ r t ≤ 1)
    (hstrip : ∀ a b : ℝ, -1 < a → a ≤ b → b < 1 →
      ∃ T c : ℝ, 0 < c ∧ ∀ t, T ≤ t → a ≤ r t → r t ≤ b → c ≤ r' t) :
    Tendsto r atTop (nhds (-1)) ∨ Tendsto r atTop (nhds 1) := by
  by_cases h0 : ∀ᶠ t in atTop, 0 ≤ r t
  · right
    apply tendsto_order.mpr
    constructor
    · intro a ha
      obtain ⟨b, hb, hb1⟩ := exists_between (max_lt (by norm_num : (0 : ℝ) < 1) ha)
      have hb0 : 0 ≤ b := (le_max_left 0 a).trans hb.le
      obtain ⟨T, c, hc, hdr⟩ := hstrip 0 b (by norm_num) hb0 hb1
      obtain ⟨s, hs⟩ := eventually_atTop.mp h0
      have htop := csaScalar_escape_strip r r' hd hr 0 b T c (max T s) hb0 hc
        (le_max_left _ _) (hs _ (le_max_right _ _)) hdr
      filter_upwards [htop] with t ht
      exact ((le_max_right 0 a).trans_lt hb).trans_le ht
    · intro a ha
      exact Eventually.of_forall (fun t => (hr t).2.trans_lt ha)
  · left
    apply tendsto_order.mpr
    constructor
    · intro a ha
      exact Eventually.of_forall (fun t => ha.trans_le (hr t).1)
    · intro b hb
      obtain ⟨a, h1, ha⟩ := exists_between (lt_min hb (by norm_num : (-1 : ℝ) < 0))
      have ha0 : a ≤ 0 := ha.le.trans (min_le_right b 0)
      obtain ⟨T, c, hc, hdr⟩ := hstrip a 0 h1 ha0 (by norm_num)
      have hlo : ∀ t, T ≤ t → r t < a := by
        intro t ht
        by_contra hn
        exact h0 (csaScalar_escape_strip r r' hd hr a 0 T c t ha0 hc ht (le_of_not_gt hn) hdr)
      filter_upwards [eventually_ge_atTop T] with t ht
      exact (hlo t ht).trans (ha.trans_le (min_le_left b 0))

/-- The linear-growth and actual boundary-crossing premises hold jointly for a curve traversing the boundary.
Source: arXiv:2411.04990v2, §4, the scalar comparison steps in `thm1`. -/
example : (∀ t : ℝ, HasDerivAt (fun u : ℝ => u) 1 t) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (∀ t : ℝ, 0 ≤ t → t = 0 → 0 < (1 : ℝ)) :=
  ⟨fun t => hasDerivAt_id t, le_rfl, zero_le_one, fun _ _ _ => one_pos⟩

/-- The bounded-strip hypotheses hold for the actual original consensus overlap, which already exceeds the strip.
Source: arXiv:2411.04990v2, §4, the original consensus equilibria of `thm1`. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    (∀ t : ℝ, -(1 : ℝ) ≤ (fun _ : ℝ => (1 : ℝ)) t ∧ (fun _ : ℝ => (1 : ℝ)) t ≤ 1) ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ t : ℝ, (0 : ℝ) ≤ (fun _ : ℝ => (1 : ℝ)) t →
      (fun _ : ℝ => (1 : ℝ)) t ≤ 1 / 2 → (1 : ℝ) ≤ 0) := by
  refine ⟨fun t => hasDerivAt_const t 1, fun _ => ⟨by norm_num, le_rfl⟩,
    by norm_num, one_pos, le_rfl, zero_le_one, ?_⟩
  intro t ht hb
  norm_num at hb

/-- A genuine nonnegative bounded distance realizes every expansion premise at the actual diagonal.
Source: arXiv:2411.04990v2, §4, identical follower trajectories in `thm1`. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 t) ∧
    (∀ t : ℝ, (0 : ℝ) ≤ (fun _ : ℝ => (0 : ℝ)) t ∧ (fun _ : ℝ => (0 : ℝ)) t ≤ 4) ∧
    (0 : ℝ) < 1 ∧ (∀ t : ℝ, (0 : ℝ) ≤ t → 1 * (fun _ : ℝ => (0 : ℝ)) t ≤ 0) :=
  ⟨fun t => hasDerivAt_const t 0, fun _ => ⟨le_rfl, by norm_num⟩,
    one_pos, fun _ _ => by norm_num⟩

/-- The endpoint-limit premises are jointly realized by the original actual consensus overlap.
Source: arXiv:2411.04990v2, §4, the consensus equilibrium in `thm1`. -/
example : Tendsto (fun _ : ℝ => (1 : ℝ)) atTop (nhds (-1)) ∨
    Tendsto (fun _ : ℝ => (1 : ℝ)) atTop (nhds 1) := by
  apply csaScalar_tendsto_endpoints _ (fun _ => 0) (fun t => hasDerivAt_const t 1)
    (fun _ => ⟨by norm_num, le_rfl⟩)
  intro a b ha hab hb
  exact ⟨0, 1, one_pos, fun t ht hlo hhi => False.elim ((not_le.mpr hb) hhi)⟩

end Transformer.Causal
