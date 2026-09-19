/-
# Appendix D, `rem: usa.d` — the angle closes at rate `e^{β/2}` for `USA`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

One statement: `rem: usa.d`, the `USA` analogue of `e:ybetacloseto1`.  It is
integrated in two stages, both instances of `Perspective.decay_of_deriv_ge`:
a slow global rate `2/n` from time `0`, which is what carries the estimate up
to `t = n/2` and supplies the half-way value there, and the fast rate
`e^{β/2}` from `t = n/2` on, which is the rate the remark prints.
-/

import Transformer.Perspective.Gronwall
import Transformer.Perspective.Section5_HighDCurve
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Complex.ExponentialBounds

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- **Remark (rem: usa.d).** *The analogue for `USA`.*

The same argument as `e:ybetacloseto1` runs with `eq: ybeta` replaced by
`eq: ybetaUSA`; there the angle closes at the cleaner rate

  `1 - γ_β(t) ≤ (1/2) exp(-e^{β/2} (t - n/2))`,   `t ≥ 0`.

**What the source says and what is changed here.**  One change: the
`[0,1]`-invariance of the solution of `eq: ybetaUSA` — `0 ≤ γ_β ≤ 1` on
`[0, ∞)`, which the survey uses silently and which is a property of the
solution, not of the equation as a differential relation — is carried as the
hypothesis `hinv`.  The range `t ≥ 0` and the constant are the survey's, both
unchanged.

The hypothesis `2 ≤ n` is not decoration: at `n = 1`, `β = 0`, `t = 0` the
bound reads `1 - γ(0) = 1 ≤ (1/2) e^{1/2} ≈ 0.824`, which is false — see
`Perspective.not_forall_usa_analogue`.

**The proof.**  Below `t = n/2` the rate `e^{β/2}` is not available (nothing
forces `γ_β ≥ 1/2` there), but the crude rate `2/n` is: `e^{βγ} ≥ 1` and
`(n-1)γ + 1 ≥ 1` for `γ ∈ [0,1]`, `β ≥ 0`, so `1 - γ_β(t) ≤ e^{-2t/n}`.  At
`t = n/2` that reads `1 - γ_β(n/2) ≤ e^{-1} < 1/2`, which both starts the
second stage and, since `2/n ≤ 1 ≤ e^{β/2}` for `n ≥ 2`, dominates
`(1/2) e^{-e^{β/2}(t - n/2)}` on the whole of `[0, n/2]`.  Past `t = n/2` the
solution is above `1/2`, so `e^{βγ} ≥ e^{β/2}` and `(n-1)γ + 1 ≥ (n+1)/2`,
giving the rate `(2/n) e^{β/2} (n+1)/2 ≥ e^{β/2}`.

Source: arXiv:2312.10794v5, Appendix D, `rem: usa.d`. -/
theorem usa_analogue (hn : 2 ≤ n) (β : ℝ) (hβ : 0 ≤ β) (γ : ℝ → ℝ)
    (hγ : ybetaODE_USA n β γ)
    (hinv : ∀ t : ℝ, 0 ≤ t → 0 ≤ γ t ∧ γ t ≤ 1) :
    ∀ t : ℝ, 0 ≤ t →
      1 - γ t ≤ (1/2 : ℝ) * Real.exp (-(Real.exp (β / 2) * (t - (n : ℝ) / 2))) := by
  have hnR : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hn0 : (0 : ℝ) < (n : ℝ) := by linarith
  have hderiv := hγ.2
  have hexpneg : Real.exp (-1) < 1/2 := Real.exp_neg_one_lt_half
  -- Stage one: the crude rate `2/n`, valid from `t = 0` on.
  have hglob : ∀ t : ℝ, 0 ≤ t → 1 - γ t ≤ Real.exp (-(2 / (n : ℝ) * t)) := by
    have hd : ∀ t : ℝ, (0 : ℝ) ≤ t →
        ∃ c : ℝ, HasDerivAt γ c t ∧ 2 / (n : ℝ) * (1 - γ t) ≤ c := by
      intro t ht
      refine ⟨_, hderiv t, ?_⟩
      obtain ⟨hg0, hg1⟩ := hinv t ht
      have hu : (0 : ℝ) ≤ 1 - γ t := by linarith
      have hE : (1 : ℝ) ≤ Real.exp (β * γ t) :=
        Real.one_le_exp (mul_nonneg hβ hg0)
      have hL : (1 : ℝ) ≤ ((n : ℝ) - 1) * γ t + 1 := by nlinarith
      have hEL : (1 : ℝ) ≤ Real.exp (β * γ t) * (((n : ℝ) - 1) * γ t + 1) := by
        nlinarith
      calc 2 / (n : ℝ) * (1 - γ t) = 2 / (n : ℝ) * (1 - γ t) * 1 := by ring
        _ ≤ 2 / (n : ℝ) * (1 - γ t)
              * (Real.exp (β * γ t) * (((n : ℝ) - 1) * γ t + 1)) :=
            mul_le_mul_of_nonneg_left hEL (mul_nonneg (by positivity) hu)
        _ = 2 / (n : ℝ) * Real.exp (β * γ t) * (1 - γ t)
              * (((n : ℝ) - 1) * γ t + 1) := by ring
    intro t ht
    calc 1 - γ t ≤ (1 - γ 0) * Real.exp (2 / (n : ℝ) * (0 - t)) :=
          decay_of_deriv_ge γ 0 (2 / (n : ℝ)) hd t ht
      _ = Real.exp (-(2 / (n : ℝ) * t)) := by
          rw [hγ.1, sub_zero, one_mul]; congr 1; ring
  -- The solution is nondecreasing on `[0, ∞)`.
  have hmono : MonotoneOn γ (Set.Ici (0 : ℝ)) := by
    refine monotoneOn_of_deriv_nonneg (convex_Ici _) (fun s _ =>
      (hderiv s).continuousAt.continuousWithinAt) (fun s _ =>
      (hderiv s).differentiableAt.differentiableWithinAt) ?_
    intro s hs
    rw [interior_Ici] at hs
    rw [(hderiv s).deriv]
    obtain ⟨hg0, hg1⟩ := hinv s (Set.mem_Ioi.mp hs).le
    have hL : (0 : ℝ) ≤ ((n : ℝ) - 1) * γ s + 1 := by nlinarith
    exact mul_nonneg (mul_nonneg (mul_nonneg (by positivity)
      (Real.exp_pos _).le) (by linarith)) hL
  -- At `t = n/2` the crude rate has already brought `γ_β` above `1/2`.
  have hhalfpt : 1 - γ ((n : ℝ) / 2) ≤ Real.exp (-1) := by
    have h := hglob ((n : ℝ) / 2) (by linarith)
    have he : -(2 / (n : ℝ) * ((n : ℝ) / 2)) = -1 := by field_simp
    rwa [he] at h
  have hhalft : ∀ t : ℝ, (n : ℝ) / 2 ≤ t → 1/2 ≤ γ t := by
    intro t ht
    have h := hmono (Set.mem_Ici.mpr (by linarith : (0 : ℝ) ≤ (n : ℝ) / 2))
      (Set.mem_Ici.mpr (by linarith : (0 : ℝ) ≤ t)) ht
    linarith
  -- Stage two: the rate `e^{β/2}`, from `t = n/2` on.
  have hfast : ∀ t : ℝ, (n : ℝ) / 2 ≤ t →
      1 - γ t ≤ (1 - γ ((n : ℝ) / 2))
        * Real.exp (Real.exp (β / 2) * ((n : ℝ) / 2 - t)) := by
    refine decay_of_deriv_ge γ ((n : ℝ) / 2) (Real.exp (β / 2)) ?_
    intro t ht
    refine ⟨_, hderiv t, ?_⟩
    obtain ⟨hg0, hg1⟩ := hinv t (by linarith)
    have hhalf := hhalft t ht
    have hu : (0 : ℝ) ≤ 1 - γ t := by linarith
    have hq : (0 : ℝ) < Real.exp (β / 2) := Real.exp_pos _
    have hE : Real.exp (β / 2) ≤ Real.exp (β * γ t) :=
      Real.exp_le_exp.mpr (by nlinarith)
    have hL : ((n : ℝ) + 1) / 2 ≤ ((n : ℝ) - 1) * γ t + 1 := by nlinarith
    have hkey : Real.exp (β / 2)
        ≤ 2 / (n : ℝ) * Real.exp (β * γ t) * (((n : ℝ) - 1) * γ t + 1) := by
      rw [div_mul_eq_mul_div, div_mul_eq_mul_div, le_div_iff₀ hn0]
      nlinarith [mul_nonneg (sub_nonneg.mpr hE) (by linarith : (0 : ℝ) ≤ ((n : ℝ) - 1) * γ t + 1),
        mul_nonneg hq.le (sub_nonneg.mpr hL)]
    calc Real.exp (β / 2) * (1 - γ t)
        ≤ (2 / (n : ℝ) * Real.exp (β * γ t) * (((n : ℝ) - 1) * γ t + 1))
            * (1 - γ t) := mul_le_mul_of_nonneg_right hkey hu
      _ = 2 / (n : ℝ) * Real.exp (β * γ t) * (1 - γ t)
            * (((n : ℝ) - 1) * γ t + 1) := by ring
  -- The two stages, glued at `t = n/2`.
  intro t ht
  have h5 : Real.exp (β / 2) * ((n : ℝ) / 2 - t)
      = -(Real.exp (β / 2) * (t - (n : ℝ) / 2)) := by ring
  rw [← h5]
  rcases le_or_gt ((n : ℝ) / 2) t with hcase | hcase
  · have h2 : 1 - γ ((n : ℝ) / 2) ≤ 1/2 := le_of_lt (lt_of_le_of_lt hhalfpt hexpneg)
    calc 1 - γ t ≤ (1 - γ ((n : ℝ) / 2))
          * Real.exp (Real.exp (β / 2) * ((n : ℝ) / 2 - t)) := hfast t hcase
      _ ≤ (1/2 : ℝ) * Real.exp (Real.exp (β / 2) * ((n : ℝ) / 2 - t)) :=
          mul_le_mul_of_nonneg_right h2 (Real.exp_pos _).le
  · have hq1 : (1 : ℝ) ≤ Real.exp (β / 2) := Real.one_le_exp (by linarith)
    have hc1 : 2 / (n : ℝ) ≤ 1 := by rw [div_le_one hn0]; linarith
    have hcn : 2 / (n : ℝ) * ((n : ℝ) / 2) = 1 := by field_simp
    have hfle : (1 : ℝ)
        ≤ Real.exp (β / 2) * ((n : ℝ) / 2 - t) + 2 / (n : ℝ) * t := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hq1) (by linarith : (0 : ℝ) ≤ (n : ℝ) / 2 - t),
        mul_nonneg (by linarith : (0 : ℝ) ≤ 1 - 2 / (n : ℝ))
          (by linarith : (0 : ℝ) ≤ (n : ℝ) / 2 - t)]
    have hA : Real.exp (-(2 / (n : ℝ) * t) - Real.exp (β / 2) * ((n : ℝ) / 2 - t))
        ≤ 1/2 :=
      le_trans (Real.exp_le_exp.mpr (by linarith)) hexpneg.le
    calc 1 - γ t ≤ Real.exp (-(2 / (n : ℝ) * t)) := hglob t ht
      _ = Real.exp (-(2 / (n : ℝ) * t) - Real.exp (β / 2) * ((n : ℝ) / 2 - t))
            * Real.exp (Real.exp (β / 2) * ((n : ℝ) / 2 - t)) := by
          rw [← Real.exp_add]; congr 1; ring
      _ ≤ (1/2 : ℝ) * Real.exp (Real.exp (β / 2) * ((n : ℝ) / 2 - t)) :=
          mul_le_mul_of_nonneg_right hA (Real.exp_pos _).le

/-- The hypotheses of `usa_analogue` are satisfiable at the smallest `n` it
allows: `n = 2`, `β = 0`, `γ = tanh`, where at `t = 0` the conclusion reads
`1 ≤ e/2 ≈ 1.359`. -/
example : 2 ≤ 2 ∧ (0 : ℝ) ≤ 0 ∧ ybetaODE_USA 2 0 Real.tanh ∧
    ∀ t : ℝ, 0 ≤ t → 0 ≤ Real.tanh t ∧ Real.tanh t ≤ 1 := by
  refine ⟨le_rfl, le_rfl, ybetaODE_USA_two_zero, fun t ht => ⟨?_, (Real.tanh_lt_one t).le⟩⟩
  rw [Real.tanh_eq_sinh_div_cosh]
  exact div_nonneg (Real.sinh_nonneg_iff.mpr ht) (Real.cosh_pos t).le

/-- **`2 ≤ n` is not decoration.**  Dropping it from `usa_analogue` — keeping
everything else, `β ≥ 0`, the equation, and the `[0,1]`-invariance — makes the
remark false: at `n = 1`, `β = 0` the equation is `γ̇ = 2(1 - γ)`, solved by
`γ(t) = 1 - e^{-2t}`, and at `t = 0` the bound reads

  `1 = 1 - γ(0) ≤ (1/2) e^{1/2} ≈ 0.824`.

What fails is the gluing: the crude rate is `2/n = 2 > 1 = e^{β/2}`, so the
`[0, n/2]` half of the argument runs the wrong way and the bound is asked to
hold before the solution has had time to reach `1/2`. -/
theorem not_forall_usa_analogue :
    ¬ ∀ (n : ℕ) (β : ℝ) (γ : ℝ → ℝ), 0 ≤ β → ybetaODE_USA n β γ →
        (∀ t : ℝ, 0 ≤ t → 0 ≤ γ t ∧ γ t ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          1 - γ t
            ≤ (1/2 : ℝ) * Real.exp (-(Real.exp (β / 2) * (t - (n : ℝ) / 2))) := by
  intro h
  have hode : ybetaODE_USA 1 0 (fun t => 1 - Real.exp (-2 * t)) := by
    refine ⟨by simp, fun t => ?_⟩
    have hlin : HasDerivAt (fun s : ℝ => -2 * s) (-2 : ℝ) t := by
      simpa using HasDerivAt.const_mul (-2 : ℝ) (hasDerivAt_id t)
    have hd : HasDerivAt (fun s : ℝ => 1 - Real.exp (-2 * s))
        (-(Real.exp (-2 * t) * -2)) t := hlin.exp.const_sub 1
    refine hd.congr_deriv ?_
    norm_num
    ring
  have hinv : ∀ t : ℝ, (0 : ℝ) ≤ t →
      0 ≤ (fun t : ℝ => 1 - Real.exp (-2 * t)) t ∧
        (fun t : ℝ => 1 - Real.exp (-2 * t)) t ≤ 1 := by
    intro t ht
    have h1 : Real.exp (-2 * t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have h2 : (0 : ℝ) < Real.exp (-2 * t) := Real.exp_pos _
    exact ⟨by simpa using h1, by simpa using h2.le⟩
  have hbad := h 1 0 _ le_rfl hode hinv 0 le_rfl
  have hq : Real.exp (1/2 : ℝ) * Real.exp (1/2 : ℝ) = Real.exp 1 := by
    rw [← Real.exp_add]; norm_num
  norm_num at hbad
  nlinarith [Real.exp_one_lt_three, Real.exp_pos (1/2 : ℝ), hbad, hq]

end Perspective
end Transformer
