/-
# The minimum of finitely many curves does not decrease while it is positive

Auxiliary calculus for arXiv:2312.10794v5, *A mathematical perspective on
Transformers*, §6.1, step 1 of `lem: hemisphere.clustering`, where
`r(t) = min_i ⟨x_i(t), w⟩` is shown to be non-decreasing on `ℝ_{≥0}`.

The step is stated in the survey as "`ṙ ≥ 0` at a minimising index", which is
not literally a derivative: a minimum of several smooth curves need not be
differentiable at the times where the minimising index changes.  What is true
is the one-sided statement proved here.  `r` is a minimum of finitely many
differentiable curves, hence continuous, and its upper right Dini derivative
is bounded by the derivatives of the curves that attain it; the fencing
theorem `image_le_of_liminf_slope_right_lt_deriv_boundary` then holds `-r`
under the barrier `-r(s) + ε (· - s)` for every `ε > 0`, and letting `ε → 0`
gives `r(s) ≤ r(t)`.

The sign hypothesis is used only at the contact points of the barrier, where
`r` is still positive, so `hnn` is required only where `r ≥ 0` — which is what
the application can supply, the bracket of `eq: therighthandside` being
non-negative only in the hemisphere.
-/

import Mathlib.Analysis.Calculus.MeanValue

open scoped Topology

namespace Transformer
namespace Perspective

variable {ι : Type*} [Fintype ι] [Nonempty ι] {f f' : ι → ℝ → ℝ} {r : ℝ → ℝ}

/-- **The minimum of finitely many differentiable curves does not decrease
while it is positive.**

`r` is given by a specification rather than as a `Finset.inf'`: it is a lower
bound for the family and it is attained at every time.  `hnn` says that at a
time when `r` is non-negative, every curve attaining it has a non-negative
derivative.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`, step 1. -/
theorem le_min_curve_of_deriv_nonneg
    (hd : ∀ (i : ι) (t : ℝ), HasDerivAt (f i) (f' i t) t)
    (hmin : ∀ (t : ℝ) (i : ι), r t ≤ f i t)
    (hatt : ∀ t : ℝ, ∃ i : ι, r t = f i t)
    (hnn : ∀ (t : ℝ) (i : ι), 0 ≤ r t → f i t = r t → 0 ≤ f' i t)
    {s t : ℝ} (hs : 0 < r s) (hst : s ≤ t) : r s ≤ r t := by
  have hne : (Finset.univ : Finset ι).Nonempty := Finset.univ_nonempty
  -- `r` is the pointwise infimum of the family, hence continuous.
  have hreq : r = Finset.univ.inf' hne f := by
    funext z
    rw [Finset.inf'_apply]
    refine le_antisymm (Finset.le_inf' _ _ fun i _ => hmin z i) ?_
    obtain ⟨i, hi⟩ := hatt z
    exact hi ▸ Finset.inf'_le (fun i => f i z) (Finset.mem_univ i)
  have hcont : Continuous r := by
    rw [hreq]
    exact Continuous.finset_inf' hne fun i _ =>
      continuous_iff_continuousAt.mpr fun x => (hd i x).continuousAt
  -- The upper right Dini derivative of `-r`, bounded through the attaining curves.
  have hDini : ∀ x ρ : ℝ, (∀ i : ι, f i x = r x → -ρ < f' i x) →
      ∀ᶠ z in 𝓝[>] x, slope (fun y => -r y) x z < ρ := by
    intro x ρ hρ
    have hstep : ∀ i : ι, ∀ᶠ z in 𝓝[>] x, r x < f i z + ρ * (z - x) := by
      intro i
      by_cases hact : f i x = r x
      · have hg : HasDerivAt (fun z : ℝ => f i z + ρ * (z - x)) (f' i x + ρ) x := by
          have h1 : HasDerivAt (fun z : ℝ => ρ * (z - x)) ρ x := by
            simpa using ((hasDerivAt_id x).sub_const x).const_mul ρ
          exact (hd i x).add h1
        have hpos : 0 < f' i x + ρ := by have := hρ i hact; linarith
        have hslope : Filter.Tendsto
            (slope (fun z : ℝ => f i z + ρ * (z - x)) x) (𝓝[>] x) (𝓝 (f' i x + ρ)) :=
          (hasDerivAt_iff_tendsto_slope.mp hg).mono_left
            (nhdsWithin_mono x fun y hy => ne_of_gt hy)
        filter_upwards [hslope.eventually_const_lt hpos, self_mem_nhdsWithin] with z hz hzx
        have hzx' : 0 < z - x := sub_pos.mpr hzx
        rw [slope_def_field] at hz
        have hnum : 0 < (f i z + ρ * (z - x)) - (f i x + ρ * (x - x)) := by
          have := mul_pos hz hzx'
          rwa [div_mul_cancel₀ _ (ne_of_gt hzx')] at this
        rw [hact] at hnum
        linarith
      · have hlt : r x < f i x := lt_of_le_of_ne (hmin x i) (Ne.symm hact)
        have hc : Filter.Tendsto (fun z : ℝ => f i z + ρ * (z - x)) (𝓝[>] x)
            (𝓝 (f i x + ρ * (x - x))) :=
          (((hd i x).continuousAt).add (by fun_prop)).continuousWithinAt
        exact hc.eventually_const_lt (by simpa using hlt)
    filter_upwards [Filter.eventually_all.2 hstep, self_mem_nhdsWithin] with z hz hzx
    obtain ⟨i, hi⟩ := hatt z
    have hzx' : 0 < z - x := sub_pos.mpr hzx
    have h1 : r x < r z + ρ * (z - x) := by rw [hi]; exact hz i
    rw [slope_def_field, div_lt_iff₀ hzx']
    linarith
  rcases eq_or_lt_of_le hst with rfl | hlt
  · exact le_rfl
  have hts : 0 < t - s := sub_pos.mpr hlt
  -- The barrier `-r(s) + ε (· - s)`, for every `ε` small enough to keep `r` positive.
  have hkey : ∀ ε : ℝ, 0 < ε → ε * (t - s) < r s → r s - ε * (t - s) ≤ r t := by
    intro ε hε hsmall
    set B : ℝ → ℝ := fun z => -r s + ε * (z - s) with hB
    have hBd : ∀ x : ℝ, HasDerivAt B ε x := by
      intro x
      have h1 : HasDerivAt (fun z : ℝ => ε * (z - s)) ε x := by
        simpa using ((hasDerivAt_id x).sub_const s).const_mul ε
      simpa [hB] using h1.const_add (-r s)
    set F' : ℝ → ℝ := fun x =>
      if 0 ≤ r x then 0 else -(Finset.univ.inf' hne fun i => f' i x) with hF'
    have hfence : ∀ ⦃x : ℝ⦄, x ∈ Set.Icc s t → -r x ≤ B x := by
      refine image_le_of_liminf_slope_right_lt_deriv_boundary (f' := F')
        hcont.neg.continuousOn ?_ (by simp [hB]) hBd ?_
      · intro x _ ρ hρ
        refine Filter.Eventually.frequently (hDini x ρ ?_)
        intro i hi
        by_cases h0 : 0 ≤ r x
        · have hF : F' x = 0 := by simp [hF', h0]
          rw [hF] at hρ
          have := hnn x i h0 hi
          linarith
        · have hF : F' x = -(Finset.univ.inf' hne fun i => f' i x) := by simp [hF', h0]
          rw [hF] at hρ
          have hle : (Finset.univ.inf' hne fun i => f' i x) ≤ f' i x :=
            Finset.inf'_le _ (Finset.mem_univ i)
          linarith
      · intro x hx hcontact
        have hmul : ε * (x - s) < ε * (t - s) := by
          have hx2 : x - s < t - s := by have := hx.2; linarith
          exact mul_lt_mul_of_pos_left hx2 hε
        have hrx : r x = r s - ε * (x - s) := by
          have : -r x = -r s + ε * (x - s) := by simpa [hB] using hcontact
          linarith
        have hxpos : 0 ≤ r x := by linarith
        have hF : F' x = 0 := by simp [hF', hxpos]
        rw [hF]
        exact hε
    have hend := hfence (Set.mem_Icc.mpr ⟨hst, le_rfl⟩)
    simp only [hB] at hend
    linarith
  -- Letting `ε → 0`.
  rcases le_or_gt (r s) (r t) with hle | hcon
  · exact hle
  have h2ts : 0 < 2 * (t - s) := by linarith
  set ε := min ((r s - r t) / (2 * (t - s))) (r s / (2 * (t - s))) with hεdef
  have hε : 0 < ε := lt_min (div_pos (by linarith) h2ts) (div_pos hs h2ts)
  have h1 : ε * (t - s) ≤ (r s - r t) / 2 :=
    calc ε * (t - s) ≤ (r s - r t) / (2 * (t - s)) * (t - s) :=
          mul_le_mul_of_nonneg_right (min_le_left _ _) hts.le
      _ = (r s - r t) / 2 := by field_simp
  have h2 : ε * (t - s) ≤ r s / 2 :=
    calc ε * (t - s) ≤ r s / (2 * (t - s)) * (t - s) :=
          mul_le_mul_of_nonneg_right (min_le_right _ _) hts.le
      _ = r s / 2 := by field_simp
  have := hkey ε hε (by linarith)
  linarith

/-- The hypotheses of `le_min_curve_of_deriv_nonneg` are satisfiable: they all
hold at once for the single curve `t ↦ t`, which is its own minimum. -/
example : (1 : ℝ) ≤ 2 :=
  le_min_curve_of_deriv_nonneg (f := fun (_ : Fin 1) (t : ℝ) => t) (f' := fun _ _ => 1)
    (r := fun t => t) (fun _ t => hasDerivAt_id t) (fun _ _ => le_rfl) (fun _ => ⟨0, rfl⟩)
    (fun _ _ _ _ => zero_le_one) one_pos (by norm_num)

end Perspective
end Transformer
