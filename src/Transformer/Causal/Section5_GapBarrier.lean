/-
# A barrier for the nearest of finitely many tokens

ArXiv:2411.04990v2, §5, proof of `lemma:meta`, the comparison following
`eq:bddotphi`. A nearest-token distance may fail to be differentiable
when its minimizing index changes. The finite minimum nevertheless obeys
a strict barrier whenever every curve attaining it points inward at
contact. The right-slope argument also appears in
`Transformer.Perspective.MinCurve` for the hemisphere comparison.
-/

import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Real.Sqrt

open scoped Topology

namespace Transformer.Causal

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- **Finite-minimum fencing.** Every active curve stays above a smooth
barrier when its derivative is strictly greater at contact.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, the comparison of
the closest preceding token. This formulation allows its index to change. -/
theorem min_curve_ge_barrier (f f' : ι → ℝ → ℝ) (B B' : ℝ → ℝ) (a b : ℝ)
    (hd : ∀ i t, HasDerivAt (f i) (f' i t) t)
    (hB : ∀ t, HasDerivAt B (B' t) t)
    (ha : ∀ i, B a ≤ f i a)
    (hcontact : ∀ t ∈ Set.Ico a b, ∀ i,
      (∀ j, f i t ≤ f j t) → f i t = B t → B' t < f' i t) :
    ∀ t ∈ Set.Icc a b, ∀ i, B t ≤ f i t := by
  classical
  have hne : (Finset.univ : Finset ι).Nonempty := Finset.univ_nonempty
  let r : ℝ → ℝ := Finset.univ.inf' hne f
  have hmin (t : ℝ) (i : ι) : r t ≤ f i t := by
    dsimp only [r]
    rw [Finset.inf'_apply]
    exact Finset.inf'_le _ (Finset.mem_univ i)
  have hatt (t : ℝ) : ∃ i, r t = f i t := by
    obtain ⟨i, _, hi⟩ := Finset.exists_mem_eq_inf' hne (fun i => f i t)
    exact ⟨i, by simpa [r, Finset.inf'_apply] using hi⟩
  have hcont : Continuous r := Continuous.finset_inf' hne fun i _ =>
    continuous_iff_continuousAt.mpr fun t => (hd i t).continuousAt
  let A (t : ℝ) := Finset.univ.filter (fun i : ι => f i t = r t)
  have hA (t : ℝ) : (A t).Nonempty := by
    obtain ⟨i, hi⟩ := hatt t
    exact ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ i, hi.symm⟩⟩
  let D (t : ℝ) := (A t).sup' (hA t) (fun i => -f' i t)
  have hDini : ∀ x ρ : ℝ, D x < ρ →
      ∀ᶠ z in 𝓝[>] x, slope (fun t => -r t) x z < ρ := by
    intro x ρ hρ
    have hstep : ∀ i : ι, ∀ᶠ z in 𝓝[>] x, r x < f i z + ρ * (z - x) := by
      intro i
      by_cases hact : f i x = r x
      · have hle : -f' i x ≤ D x :=
          Finset.le_sup' (fun i => -f' i x)
            (Finset.mem_filter.mpr ⟨Finset.mem_univ i, hact⟩)
        have hg : HasDerivAt (fun z : ℝ => f i z + ρ * (z - x)) (f' i x + ρ) x := by
          have h1 : HasDerivAt (fun z : ℝ => ρ * (z - x)) ρ x := by
            simpa using ((hasDerivAt_id x).sub_const x).const_mul ρ
          exact (hd i x).add h1
        have hslope := (hasDerivAt_iff_tendsto_slope.mp hg).mono_left
          (nhdsWithin_mono x fun z hz => ne_of_gt hz)
        filter_upwards [hslope.eventually_const_lt (by linarith : 0 < f' i x + ρ),
          self_mem_nhdsWithin] with z hz hzx
        have hzx' : 0 < z - x := sub_pos.mpr hzx
        rw [slope_def_field] at hz
        have hnum := mul_pos hz hzx'
        rw [div_mul_cancel₀ _ hzx'.ne', hact] at hnum
        linarith
      · have hlt : r x < f i x := lt_of_le_of_ne (hmin x i) (Ne.symm hact)
        have hc : Filter.Tendsto (fun z : ℝ => f i z + ρ * (z - x)) (𝓝[>] x)
            (𝓝 (f i x + ρ * (x - x))) :=
          ((hd i x).continuousAt.add (by fun_prop)).continuousWithinAt
        exact hc.eventually_const_lt (by simpa using hlt)
    filter_upwards [Filter.eventually_all.2 hstep, self_mem_nhdsWithin] with z hz hzx
    obtain ⟨i, hi⟩ := hatt z
    have h1 : r x < r z + ρ * (z - x) := by rw [hi]; exact hz i
    rw [slope_def_field, div_lt_iff₀ (sub_pos.mpr hzx)]
    linarith
  have hfence : ∀ t ∈ Set.Icc a b, -r t ≤ -B t := by
    apply image_le_of_liminf_slope_right_lt_deriv_boundary (f' := D)
      hcont.neg.continuousOn
    · intro x _ ρ hρ
      exact (hDini x ρ hρ).frequently
    · have hr : B a ≤ r a := by
        dsimp only [r]
        rw [Finset.inf'_apply]
        exact Finset.le_inf' hne _ (fun i _ => ha i)
      change -r a ≤ -B a
      linarith
    · exact fun x => (hB x).neg
    · intro x hx heq
      change -r x = -B x at heq
      apply (Finset.sup'_lt_iff (hA x) (f := fun i => -f' i x)).mpr
      intro i hi
      have hact : f i x = r x := (Finset.mem_filter.mp hi).2
      have heq' : f i x = B x := by linarith
      have hbound := hcontact x hx i (fun j => by rw [hact]; exact hmin x j) heq'
      linarith
  intro t ht i
  have h := hfence t ht
  exact (by linarith : B t ≤ r t).trans (hmin t i)

/-- All hypotheses are simultaneously witnessed by two distinct parallel
curves above a decreasing barrier. Source: arXiv:2411.04990v2, §5,
the scalar comparison in the proof of `lemma:meta`. -/
example : ∀ t ∈ Set.Icc (0 : ℝ) 1, ∀ i : Fin 2, -t ≤ t + (i : ℝ) := by
  apply min_curve_ge_barrier (f' := fun _ _ => 1) (B' := fun _ => -1)
  · intro i t
    simpa using (hasDerivAt_id t).add_const (i : ℝ)
  · intro t
    exact (hasDerivAt_id t).fun_neg
  · intro i
    simp only [neg_zero, zero_add]
    exact_mod_cast i.val.zero_le
  · intro _ _ _ _ _
    norm_num

/-- **The squared-distance barrier used by `lemma:meta`.** If the initial
gaps are at least `R > r` and a nearest squared gap has derivative at least
`-4 K √f` while at least `r²`, then `2KT < R-r` keeps every gap at least `r`.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, the comparison
following `eq:bddotphi`. Squaring avoids differentiating a norm at zero. -/
theorem min_sq_curve_ge (f f' : ι → ℝ → ℝ) (r R K T : ℝ)
    (hr : 0 < r) (hR : r < R) (hT : 0 < T) (hbudget : 2 * K * T < R - r)
    (hd : ∀ i t, HasDerivAt (f i) (f' i t) t) (hinit : ∀ i, R ^ 2 ≤ f i 0)
    (hbound : ∀ t ∈ Set.Ico (0 : ℝ) T, ∀ i,
      (∀ j, f i t ≤ f j t) → r ^ 2 ≤ f i t →
        -4 * K * Real.sqrt (f i t) ≤ f' i t) :
    ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ i, r ^ 2 ≤ f i t := by
  let L := (R - r) / T
  have hL : 0 < L := div_pos (sub_pos.mpr hR) hT
  have hLT : L * T = R - r := div_mul_cancel₀ _ hT.ne'
  have hLK : 2 * K < L := (lt_div_iff₀ hT).mpr hbudget
  have hlinear (t : ℝ) (ht : t ≤ T) : r ≤ R - L * t := by
    have hm := mul_le_mul_of_nonneg_left ht hL.le
    linarith
  have hbarrier := min_curve_ge_barrier f f'
    (fun t => (R - L * t) ^ 2) (fun t => -2 * L * (R - L * t)) 0 T hd
    (fun t => by
      have h1 := ((hasDerivAt_id t).const_mul L).const_sub R
      have h2 : HasDerivAt (fun z : ℝ => R - L * z) (-L) t := by simpa using h1
      convert h2.fun_pow 2 using 1
      ring) (by simpa using hinit)
    (by
      intro t ht i hmin heq
      have hb : r ≤ R - L * t := hlinear t ht.2.le
      have hbpos : 0 < R - L * t := hr.trans_le hb
      have hf : r ^ 2 ≤ f i t := by nlinarith
      have hderiv := hbound t ht i hmin hf
      rw [heq, Real.sqrt_sq hbpos.le] at hderiv
      have hstrict := mul_pos (sub_pos.mpr hLK) hbpos
      nlinarith)
  intro t ht i
  have hb := hlinear t ht.2
  have hf := hbarrier t ht i
  nlinarith

/-- The squared-distance comparison has actual simultaneous witnesses:
two constant positive gaps, `r = 1`, `R = 2`, `T = 1`, `K = 0`.
Source: arXiv:2411.04990v2, §5, the gap comparison in `lemma:meta`. -/
example : ∀ t ∈ Set.Icc (0 : ℝ) 1, ∀ i : Fin 2, (1 : ℝ) ^ 2 ≤ (2 + (i : ℝ)) ^ 2 := by
  apply min_sq_curve_ge (f := fun (i : Fin 2) _ => (2 + (i : ℝ)) ^ 2)
    (f' := fun _ _ => 0) (r := 1) (R := 2) (K := 0) (T := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · intro i t
    exact hasDerivAt_const t _
  · intro i
    have hi : (0 : ℝ) ≤ i := by exact_mod_cast i.val.zero_le
    nlinarith
  · intro _ _ _ _ _
    simp

end Transformer.Causal
