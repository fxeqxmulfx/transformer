/-
# A mathematical perspective on Transformers — §5, the vanishing-of-integrand
lemma

`lem: ez.lemma` of arXiv:2312.10794v5, split off from `Section5_HighD` because
it is the one statement of the section that is pure real analysis and needs the
mean value theorem and the Lebesgue integral rather than the dynamics.

The proof is the standard one: if `|f(t)| ≥ ε` at some large `t`, the bound on
`f'` keeps `|f|` above `ε/2` on a whole interval of length `ε/(2C)` to the
right of `t`, so the tail `∫_t^∞ |f|` is at least `ε²/(4C)` — and the tail of a
convergent integral goes to zero.
-/

import Transformer.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.MeasureTheory.Integral.IntegralEqImproper
import Mathlib.MeasureTheory.Integral.ExpDecay

open Real MeasureTheory

namespace Transformer
namespace Perspective

/-- **Lemma (lem: ez.lemma).** *Vanishing-of-integrand lemma.*

If `f : ℝ_{≥0} → ℝ` is differentiable, `∫₀^∞ |f(t)| dt < ∞` and `f'` is
uniformly bounded, then `lim_{t→∞} f(t) = 0`.

Source: arXiv:2312.10794v5, §5, `lem: ez.lemma`. -/
lemma ez_lemma
    (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (hf_int : MeasureTheory.IntegrableOn (fun t => |f t|)
                (Set.Ici (0 : ℝ)) MeasureTheory.volume)
    (hf_dbnd : ∃ M : ℝ, ∀ t : ℝ, 0 ≤ t → |deriv f t| ≤ M) :
    Filter.Tendsto f Filter.atTop (nhds 0) := by
  obtain ⟨M, hM⟩ := hf_dbnd
  -- A bound that is positive, so that it can be divided by.
  set C : ℝ := max M 1 with hCdef
  have hC0 : (0 : ℝ) < C := lt_of_lt_of_le one_pos (le_max_right _ _)
  have hbound : ∀ x ∈ Set.Ici (0 : ℝ), ‖deriv f x‖ ≤ C := fun x hx =>
    (hM x hx).trans (le_max_left _ _)
  rw [Metric.tendsto_atTop]
  intro ε hε
  set δ : ℝ := ε / (2 * C) with hδdef
  have hδ0 : 0 < δ := by positivity
  have hCδ : C * δ = ε / 2 := by
    rw [hδdef]; field_simp
  have htail : Filter.Tendsto (fun T : ℝ => ∫ x in Set.Ici T, |f x|) Filter.atTop (nhds 0) :=
    MeasureTheory.tendsto_integral_Ici_zero Filter.tendsto_id
  obtain ⟨T₀, hT₀⟩ := Metric.tendsto_atTop.mp htail (ε * δ / 2) (by positivity)
  refine ⟨max T₀ 0, fun t ht => ?_⟩
  have ht0 : 0 ≤ t := le_trans (le_max_right T₀ 0) ht
  have htT : T₀ ≤ t := le_trans (le_max_left T₀ 0) ht
  rw [Real.dist_eq, sub_zero]
  by_contra hcon
  push Not at hcon
  -- On `[t, t + δ]` the function stays above `ε / 2`.
  have hlow : ∀ s ∈ Set.Icc t (t + δ), ε / 2 ≤ |f s| := by
    intro s hs
    have hs0 : (0 : ℝ) ≤ s := le_trans ht0 hs.1
    have hmvt : ‖f s - f t‖ ≤ C * ‖s - t‖ :=
      (convex_Ici (0 : ℝ)).norm_image_sub_le_of_norm_deriv_le
        (fun x _ => hf x) hbound ht0 hs0
    rw [Real.norm_eq_abs, Real.norm_eq_abs] at hmvt
    have hst : |s - t| ≤ δ := by
      rw [abs_of_nonneg (by linarith [hs.1])]
      linarith [hs.2]
    have h1 : |f s - f t| ≤ ε / 2 := by
      calc |f s - f t| ≤ C * |s - t| := hmvt
        _ ≤ C * δ := by exact mul_le_mul_of_nonneg_left hst hC0.le
        _ = ε / 2 := hCδ
    have h2 : |f t| - |f s| ≤ |f t - f s| := abs_sub_abs_le_abs_sub _ _
    rw [abs_sub_comm] at h2
    linarith
  -- Hence the tail integral at `t` is at least `δ ε / 2`.
  have hIci : IntegrableOn (fun x => |f x|) (Set.Ici t) volume :=
    hf_int.mono_set (Set.Ici_subset_Ici.mpr ht0)
  have hIcc : IntegrableOn (fun x => |f x|) (Set.Icc t (t + δ)) volume :=
    hIci.mono_set Set.Icc_subset_Ici_self
  have hconst : ∫ _x in Set.Icc t (t + δ), (ε / 2) = δ * (ε / 2) := by
    rw [MeasureTheory.setIntegral_const, Real.volume_real_Icc_of_le (by linarith),
      show t + δ - t = δ by ring, smul_eq_mul]
  have hstep : δ * (ε / 2) ≤ ∫ x in Set.Icc t (t + δ), |f x| := by
    rw [← hconst]
    exact MeasureTheory.setIntegral_mono_on
      (MeasureTheory.integrableOn_const (by rw [Real.volume_Icc]; exact ENNReal.ofReal_ne_top))
      hIcc measurableSet_Icc hlow
  have hmono : ∫ x in Set.Icc t (t + δ), |f x| ≤ ∫ x in Set.Ici t, |f x| :=
    MeasureTheory.setIntegral_mono_set hIci
      (Filter.Eventually.of_forall fun x => abs_nonneg _)
      (Filter.Eventually.of_forall Set.Icc_subset_Ici_self)
  have hsmall := hT₀ t htT
  rw [Real.dist_eq, sub_zero] at hsmall
  have hnn : (0 : ℝ) ≤ ∫ x in Set.Ici t, |f x| :=
    MeasureTheory.setIntegral_nonneg measurableSet_Ici fun x _ => abs_nonneg _
  rw [abs_of_nonneg hnn] at hsmall
  nlinarith

/-- The hypotheses of `ez_lemma` are satisfiable, and not only by functions
that vanish: `f(t) = e^{-t}` is differentiable, integrable on `[0, ∞)`, and has
`|f'| ≤ 1` there. -/
example :
    Differentiable ℝ (fun t : ℝ => Real.exp (-t)) ∧
      MeasureTheory.IntegrableOn (fun t : ℝ => |Real.exp (-t)|)
        (Set.Ici (0 : ℝ)) MeasureTheory.volume ∧
      ∀ t : ℝ, 0 ≤ t → |deriv (fun t : ℝ => Real.exp (-t)) t| ≤ 1 := by
  have hderiv : ∀ t : ℝ, HasDerivAt (fun t : ℝ => Real.exp (-t)) (-Real.exp (-t)) t := by
    intro t
    simpa using (hasDerivAt_neg t).exp
  refine ⟨fun t => (hderiv t).differentiableAt, ?_, fun t ht => ?_⟩
  · rw [integrableOn_Ici_iff_integrableOn_Ioi]
    have hcongr : (fun t : ℝ => |Real.exp (-t)|) = fun t : ℝ => Real.exp (-1 * t) := by
      funext t
      rw [abs_of_pos (Real.exp_pos _), neg_one_mul]
    rw [hcongr]
    exact exp_neg_integrableOn_Ioi 0 one_pos
  · rw [(hderiv t).deriv, abs_neg, abs_of_pos (Real.exp_pos _), Real.exp_le_one_iff]
    linarith

end Perspective
end Transformer
