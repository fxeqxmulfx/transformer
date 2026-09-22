/-
# The maximum of finitely many curves does not increase

Auxiliary calculus for arXiv:2312.10794v5, *A mathematical perspective on
Transformers*, §6.1, `lem: hemisphere.clustering`, where the width of the
configuration in a chart of the hemisphere is shown to decay exponentially.

A maximum of finitely many differentiable curves need not be differentiable
where the maximising index changes, so "`Ṙ ≤ 0` at a maximising index" is not
literally a derivative.  What is true is the one-sided statement proved here:
the upper right Dini derivative of `R` is bounded by the derivatives of the
curves attaining it, and the fencing theorem
`image_le_of_liminf_slope_right_lt_deriv_boundary` holds `R` under the barrier
`R(s) + ε (· - s)` for every `ε > 0`.

The curves need only be differentiable from time `a` on, which is what a chart
defined on the future of a trajectory can supply.
-/

import Mathlib.Analysis.Calculus.MeanValue

open scoped Topology

namespace Transformer
namespace Perspective

variable {ι : Type*} [Fintype ι] {f f' : ι → ℝ → ℝ} {R : ℝ → ℝ} {a : ℝ}

/-- **The maximum of finitely many curves does not increase** from time `a` on,
if every curve attaining it has a non-positive derivative there.

`R` is given by a specification: an upper bound for the family, attained at
every time.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`. -/
theorem max_curve_le_of_deriv_nonpos
    (hd : ∀ (i : ι) (t : ℝ), a ≤ t → HasDerivAt (f i) (f' i t) t)
    (hmax : ∀ (t : ℝ) (i : ι), f i t ≤ R t)
    (hatt : ∀ t : ℝ, ∃ i : ι, R t = f i t)
    (hnp : ∀ (t : ℝ), a ≤ t → ∀ i : ι, f i t = R t → f' i t ≤ 0)
    {s t : ℝ} (has : a ≤ s) (hst : s ≤ t) : R t ≤ R s := by
  obtain ⟨i₀, -⟩ := hatt s
  have hne : (Finset.univ : Finset ι).Nonempty := ⟨i₀, Finset.mem_univ _⟩
  have hreq : ∀ z, R z = Finset.univ.sup' hne (fun i => f i) z := by
    intro z
    rw [Finset.sup'_apply]
    refine le_antisymm ?_ (Finset.sup'_le _ _ fun i _ => hmax z i)
    obtain ⟨i, hi⟩ := hatt z
    exact hi ▸ Finset.le_sup' (fun i => f i z) (Finset.mem_univ i)
  have hcont : ContinuousOn R (Set.Icc s t) := by
    rw [show R = Finset.univ.sup' hne (fun i => f i) from funext hreq]
    exact ContinuousOn.finset_sup' hne fun i _ x hx =>
      (hd i x (has.trans hx.1)).continuousAt.continuousWithinAt
  -- The upper right Dini derivative of `R` is at most `0` from `a` on.
  have hDini : ∀ x : ℝ, a ≤ x → ∀ ρ : ℝ, 0 < ρ →
      ∀ᶠ z in 𝓝[>] x, slope R x z < ρ := by
    intro x hx ρ hρ
    have hstep : ∀ i : ι, ∀ᶠ z in 𝓝[>] x, f i z < R x + ρ * (z - x) := by
      intro i
      by_cases hact : f i x = R x
      · have hslope : Filter.Tendsto (slope (f i) x) (𝓝[>] x) (𝓝 (f' i x)) :=
          (hasDerivAt_iff_tendsto_slope.mp (hd i x hx)).mono_left
            (nhdsWithin_mono x fun y hy => ne_of_gt hy)
        have hlt : f' i x < ρ := (hnp x hx i hact).trans_lt hρ
        filter_upwards [hslope.eventually_lt_const hlt, self_mem_nhdsWithin] with z hz hzx
        have hzx' : 0 < z - x := sub_pos.mpr hzx
        rw [slope_def_field, div_lt_iff₀ hzx'] at hz
        linarith
      · have hlt : f i x < R x := lt_of_le_of_ne (hmax x i) hact
        have hc : Filter.Tendsto (f i) (𝓝[>] x) (𝓝 (f i x)) :=
          (hd i x hx).continuousAt.continuousWithinAt
        filter_upwards [hc.eventually_lt_const hlt, self_mem_nhdsWithin] with z hz hzx
        have : 0 < ρ * (z - x) := mul_pos hρ (sub_pos.mpr hzx)
        linarith
    filter_upwards [Filter.eventually_all.2 hstep, self_mem_nhdsWithin] with z hz hzx
    obtain ⟨i, hi⟩ := hatt z
    rw [slope_def_field, div_lt_iff₀ (sub_pos.mpr hzx), hi]
    linarith [hz i]
  -- The barrier `R(s) + ε (· - s)`.
  have hkey : ∀ ε : ℝ, 0 < ε → R t ≤ R s + ε * (t - s) := by
    intro ε hε
    have hBd : ∀ x : ℝ, HasDerivAt (fun z => R s + ε * (z - s)) ε x := by
      intro x
      simpa using (((hasDerivAt_id x).sub_const s).const_mul ε).const_add (R s)
    refine image_le_of_liminf_slope_right_lt_deriv_boundary (f' := fun _ => 0) hcont ?_
      (by simp) hBd (fun _ _ _ => hε) ⟨hst, le_rfl⟩
    intro x hx ρ hρ
    exact (hDini x (has.trans hx.1) ρ hρ).frequently
  refine le_of_forall_pos_le_add fun δ hδ => ?_
  have hts : 0 ≤ t - s := sub_nonneg.mpr hst
  have hpos : 0 < t - s + 1 := by linarith
  have h := hkey (δ / (t - s + 1)) (div_pos hδ hpos)
  have hle : δ / (t - s + 1) * (t - s) ≤ δ := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hpos]
    nlinarith
  linarith

/-- The hypotheses of `max_curve_le_of_deriv_nonpos` are satisfiable: they all
hold at once for the single curve `t ↦ -t`, which is its own maximum. -/
example : -(2 : ℝ) ≤ -1 :=
  max_curve_le_of_deriv_nonpos (f := fun (_ : Fin 1) (t : ℝ) => -t) (f' := fun _ _ => -1)
    (R := fun t => -t) (a := 0) (fun _ t _ => (hasDerivAt_id t).neg) (fun _ _ => le_rfl)
    (fun _ => ⟨0, rfl⟩) (fun _ _ _ _ => by norm_num) zero_le_one (by norm_num)

end Perspective
end Transformer
