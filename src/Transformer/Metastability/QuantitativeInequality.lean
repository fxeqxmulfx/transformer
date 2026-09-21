/-
# `lem: quantitative inequality`, with the constant its proof supports

arXiv:2410.06833v1, "Dynamic metastability in the self-attention model", §3.2,
`lem: quantitative inequality`, adapted there from Otto–Villani, Proposition 1.

The lemma turns the *upper* bound of hypothesis (H1) — a Polyak-Łojasiewicz
inequality `𝖤(u) - 𝖤(v) ≤ (1/2c)‖∇𝖤(u)‖²` — into its *lower* bound, by
running the gradient flow `Ẋ = -∇𝖤(X)` from `u` to a point `v` of the slow
manifold and measuring how far it can travel:

  `‖u - v‖ ≤ ∫₀ᵀ ‖∇𝖤(X(t))‖ dt ≤ √(2/c) √(𝖤(u) - 𝖤(v))`.

Two corrections to the statement as printed.

* The sign: the lemma is printed for the *ascending* flow `Ẋ = ∇𝖤(X)`, with
  `𝖤(v) - 𝖤(u)` on both sides, while (H1), which it serves, is written for
  the descending flow `Ẋ = -∇𝖤(X)` of the Otto–Reznikoff framework, with
  `𝖤(u) - 𝖤(v)`.  The two conventions differ by `𝖤 ↦ -𝖤` (the constraint
  `𝖤 ≥ 0` is immaterial: both sides involve differences of `𝖤` only, and
  `𝖤` is bounded on the compact `ℳ` of the paper).  It is stated here in the
  convention of (H1).
* The constant: the argument gives `(c/2)‖u - v‖²`, not `2c‖u - v‖²`.  The
  printed proof differentiates `√(𝖤(v) - 𝖤(X(t)))` and drops the factor `2`
  of `(√f)' = f'/(2√f)`, which is exactly the factor `4` between the two.
  `not_quantitative_inequality_two_mul` refutes `2c`: the flow of `𝖤(x) = x²/2`
  on `ℝ` from `u = 1` to `v = e⁻¹` meets every hypothesis with `c = 1` and
  violates that conclusion.  At `c = 1` the corrected constant is `1/2`, which
  is exactly the lower bound (H1) asks for, so nothing downstream is lost.

The flow enters as a triple `(𝖤, ∇𝖤, X)`: `X` solves the gradient flow, and
`𝖤 ∘ X` falls at the rate `‖∇𝖤(X)‖²` — the chain rule for a smooth `𝖤`,
which this development does not carry for an abstract manifold.  The PL
inequality is asked for along the whole trajectory, as the paper's proof uses
it and its statement leaves implicit; at the single point `u` it says nothing
about where the flow goes, and no bound on `‖u - v‖` can follow.
-/

import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

namespace Transformer
namespace Metastability

/-- **Lemma (lem: quantitative inequality).**

Let `X` follow the gradient flow of `𝖤` from `u = X(0)` to `v = X(T)`, and let
the Polyak-Łojasiewicz bound `𝖤(X(t)) - 𝖤(v) ≤ (1/2c)‖∇𝖤(X(t))‖²` hold along
it.  Then

  `(c/2)‖u - v‖² ≤ 𝖤(u) - 𝖤(v)`.

Source: arXiv:2410.06833v1, §3.2, `lem: quantitative inequality`, with the sign
and the constant corrected (see the module docstring). -/
theorem quantitative_inequality
    {M : Type*} [NormedAddCommGroup M] [NormedSpace ℝ M] [CompleteSpace M]
    (E : M → ℝ) (gradE : M → M) (X : ℝ → M) (u v : M) (c T : ℝ)
    (hc : 0 < c) (hT : 0 ≤ T)
    (hflow : ∀ t : ℝ, HasDerivAt X (-gradE (X t)) t)
    (hgrad : Continuous fun t => gradE (X t))
    (hchain : ∀ t : ℝ, HasDerivAt (fun r => E (X r)) (-‖gradE (X t)‖ ^ 2) t)
    (hu : X 0 = u) (hv : X T = v)
    (hPL : ∀ t ∈ Set.Icc 0 T, E (X t) - E v ≤ 1 / (2 * c) * ‖gradE (X t)‖ ^ 2) :
    c / 2 * ‖u - v‖ ^ 2 ≤ E u - E v := by
  set g : ℝ → M := fun t => gradE (X t) with hgdef
  set F : ℝ → ℝ := fun t => E (X t) - E v with hFdef
  have hF : ∀ t : ℝ, HasDerivAt F (-‖g t‖ ^ 2) t := fun t => (hchain t).sub_const _
  have hFc : Continuous F := continuous_iff_continuousAt.2 fun t => (hF t).continuousAt
  have hFanti : Antitone F :=
    antitone_of_deriv_nonpos (fun t => (hF t).differentiableAt) fun t => by
      rw [(hF t).deriv]
      exact neg_nonpos.mpr (sq_nonneg _)
  have hFT0 : F T = 0 := by rw [hFdef]; simp [hv]
  have hFnonneg : ∀ t ∈ Set.Icc 0 T, 0 ≤ F t := fun t ht => by
    have h : F T ≤ F t := hFanti ht.2
    linarith
  have hgint : IntervalIntegrable (fun t => ‖g t‖) MeasureTheory.volume 0 T :=
    hgrad.norm.intervalIntegrable 0 T
  -- the flow cannot travel further than the length of its path
  have hdist : ‖u - v‖ ≤ ∫ t in (0 : ℝ)..T, ‖g t‖ := by
    have hint : ∫ t in (0 : ℝ)..T, -g t = X T - X 0 :=
      intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hflow t)
        (hgrad.neg.intervalIntegrable 0 T)
    calc ‖u - v‖ = ‖∫ t in (0 : ℝ)..T, -g t‖ := by
          rw [hint, hu, hv, norm_sub_rev]
      _ ≤ ∫ t in (0 : ℝ)..T, ‖-g t‖ := intervalIntegral.norm_integral_le_integral_norm hT
      _ = ∫ t in (0 : ℝ)..T, ‖g t‖ := by simp
  -- the length of the path, against the energy it spends, up to an `ε` that
  -- keeps the square root differentiable where the energy is exhausted
  have key : ∀ ε : ℝ, 0 < ε →
      Real.sqrt (c / 2) * ‖u - v‖ ≤ Real.sqrt (F 0) + Real.sqrt ε * (1 + c * T) := by
    intro ε hε
    have hSpos : ∀ t ∈ Set.Icc 0 T, 0 < F t + ε := fun t ht => by
      have := hFnonneg t ht
      linarith
    have hGderiv : ∀ t ∈ Set.uIcc (0 : ℝ) T,
        HasDerivAt (fun r => Real.sqrt (F r + ε))
          (-‖g t‖ ^ 2 / (2 * Real.sqrt (F t + ε))) t := by
      intro t ht
      rw [Set.uIcc_of_le hT] at ht
      exact ((hF t).add_const ε).sqrt (hSpos t ht).ne'
    have hGcont : ContinuousOn (fun t => -‖g t‖ ^ 2 / (2 * Real.sqrt (F t + ε)))
        (Set.uIcc (0 : ℝ) T) := by
      rw [Set.uIcc_of_le hT]
      refine ContinuousOn.div (by fun_prop) (by fun_prop) fun t ht => ?_
      have := hSpos t ht
      positivity
    have hGint : IntervalIntegrable (fun t => -‖g t‖ ^ 2 / (2 * Real.sqrt (F t + ε)))
        MeasureTheory.volume 0 T := hGcont.intervalIntegrable
    have hFTC : ∫ t in (0 : ℝ)..T, -‖g t‖ ^ 2 / (2 * Real.sqrt (F t + ε))
        = Real.sqrt (F T + ε) - Real.sqrt (F 0 + ε) :=
      intervalIntegral.integral_eq_sub_of_hasDerivAt hGderiv hGint
    -- pointwise: the Polyak-Łojasiewicz bound, with the square root regularized
    have hpt : ∀ t ∈ Set.Icc 0 T,
        Real.sqrt (c / 2) * ‖g t‖ - c * Real.sqrt ε
          ≤ -(-‖g t‖ ^ 2 / (2 * Real.sqrt (F t + ε))) := by
      intro t ht
      set a := ‖g t‖ with hadef
      set k := Real.sqrt (c / 2) with hkdef
      set e := Real.sqrt ε with hedef
      set S := Real.sqrt (F t + ε) with hSdef
      have ha : 0 ≤ a := norm_nonneg _
      have he : 0 ≤ e := Real.sqrt_nonneg _
      have hk : 0 < k := Real.sqrt_pos.mpr (by linarith)
      have hk2 : k ^ 2 = c / 2 := Real.sq_sqrt (by linarith)
      have he2 : e ^ 2 = ε := Real.sq_sqrt hε.le
      have hS : 0 < S := Real.sqrt_pos.mpr (hSpos t ht)
      have hc2 : c = 2 * k ^ 2 := by rw [hk2]; ring
      have hS2 : 2 * k * S ≤ a + 2 * k * e := by
        have hPLt : F t ≤ 1 / (2 * c) * a ^ 2 := hPL t ht
        have hSsq : S ^ 2 = F t + ε := Real.sq_sqrt (hSpos t ht).le
        have h2c : 2 * c * F t ≤ a ^ 2 := by
          calc 2 * c * F t ≤ 2 * c * (1 / (2 * c) * a ^ 2) :=
                mul_le_mul_of_nonneg_left hPLt (by positivity)
            _ = a ^ 2 := by field_simp
        have hsq : (2 * k * S) ^ 2 ≤ (a + 2 * k * e) ^ 2 := by
          have h1 : (2 * k * S) ^ 2 = 2 * c * (F t + ε) := by
            have h : (2 * k * S) ^ 2 = 4 * k ^ 2 * S ^ 2 := by ring
            rw [h, hk2, hSsq]; ring
          have h2 : (a + 2 * k * e) ^ 2 = a ^ 2 + 4 * k * a * e + 2 * c * ε := by
            have h : (a + 2 * k * e) ^ 2 = a ^ 2 + 4 * k * a * e + 4 * k ^ 2 * e ^ 2 := by ring
            rw [h, hk2, he2]; ring
          rw [h1, h2]
          have hae : 0 ≤ 4 * k * a * e := by positivity
          linarith
        nlinarith [hsq, mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hk.le) hS.le,
          mul_nonneg ha he, mul_nonneg hk.le he]
      have hgoal : (k * a - c * e) * (2 * S) ≤ a ^ 2 := by
        rcases le_or_gt (k * a - c * e) 0 with hneg | hpos
        · have hnp : (k * a - c * e) * (2 * S) ≤ 0 :=
            mul_nonpos_of_nonpos_of_nonneg hneg (by positivity)
          exact hnp.trans (sq_nonneg a)
        · have hmul : (k * a - c * e) * (2 * k * S) ≤ (k * a - c * e) * (a + 2 * k * e) :=
            mul_le_mul_of_nonneg_left hS2 hpos.le
          have hrhs : (k * a - c * e) * (a + 2 * k * e) = k * a ^ 2 - 4 * k ^ 3 * e ^ 2 := by
            rw [hc2]; ring
          have hlhs : (k * a - c * e) * (2 * k * S) = k * ((k * a - c * e) * (2 * S)) := by ring
          have hfin : k * ((k * a - c * e) * (2 * S)) ≤ k * a ^ 2 := by
            rw [← hlhs]
            have hstep : (k * a - c * e) * (2 * k * S) ≤ k * a ^ 2 - 4 * k ^ 3 * e ^ 2 :=
              hmul.trans_eq hrhs
            have hpos4 : 0 ≤ 4 * k ^ 3 * e ^ 2 := by positivity
            linarith
          exact le_of_mul_le_mul_left hfin hk
      rw [neg_div, neg_neg, le_div_iff₀ (by positivity)]
      linarith [hgoal]
    have hleft : IntervalIntegrable
        (fun t => Real.sqrt (c / 2) * ‖g t‖ - c * Real.sqrt ε) MeasureTheory.volume 0 T :=
      ((hgint.const_mul _).sub (intervalIntegrable_const))
    have hGneg : IntervalIntegrable
        (fun t => -(-‖g t‖ ^ 2 / (2 * Real.sqrt (F t + ε)))) MeasureTheory.volume 0 T :=
      hGint.neg
    have hmono := intervalIntegral.integral_mono_on hT hleft hGneg hpt
    rw [intervalIntegral.integral_sub (hgint.const_mul _) intervalIntegrable_const,
      intervalIntegral.integral_const_mul, intervalIntegral.integral_neg, hFTC,
      intervalIntegral.integral_const, smul_eq_mul] at hmono
    have hFT : Real.sqrt (F T + ε) ≥ 0 := Real.sqrt_nonneg _
    have hsub : Real.sqrt (F 0 + ε) ≤ Real.sqrt (F 0) + Real.sqrt ε := by
      have h0 : 0 ≤ F 0 := hFnonneg 0 ⟨le_rfl, hT⟩
      have hle : F 0 + ε ≤ (Real.sqrt (F 0) + Real.sqrt ε) ^ 2 := by
        have e1 := Real.sq_sqrt h0
        have e2 := Real.sq_sqrt hε.le
        nlinarith [mul_nonneg (Real.sqrt_nonneg (F 0)) (Real.sqrt_nonneg ε)]
      calc Real.sqrt (F 0 + ε) ≤ Real.sqrt ((Real.sqrt (F 0) + Real.sqrt ε) ^ 2) :=
            Real.sqrt_le_sqrt hle
        _ = Real.sqrt (F 0) + Real.sqrt ε := Real.sqrt_sq (by positivity)
    have hkpos : (0 : ℝ) ≤ Real.sqrt (c / 2) := Real.sqrt_nonneg _
    have hmul : Real.sqrt (c / 2) * ‖u - v‖ ≤ Real.sqrt (c / 2) * ∫ t in (0 : ℝ)..T, ‖g t‖ :=
      mul_le_mul_of_nonneg_left hdist hkpos
    nlinarith [hmono, hsub, hFT, hmul, Real.sqrt_nonneg ε]
  -- let the regularization go
  have hfinal : Real.sqrt (c / 2) * ‖u - v‖ ≤ Real.sqrt (F 0) := by
    refine le_of_forall_pos_le_add fun δ hδ => ?_
    have hden : 0 < 1 + c * T := by positivity
    have h := key ((δ / (1 + c * T)) ^ 2) (by positivity)
    rwa [Real.sqrt_sq (by positivity), div_mul_cancel₀ _ hden.ne'] at h
  have h0 : 0 ≤ F 0 := hFnonneg 0 ⟨le_rfl, hT⟩
  have hsq : (Real.sqrt (c / 2) * ‖u - v‖) ^ 2 ≤ Real.sqrt (F 0) ^ 2 :=
    pow_le_pow_left₀ (by positivity) hfinal 2
  rw [mul_pow, Real.sq_sqrt (by linarith), Real.sq_sqrt h0] at hsq
  simpa [hFdef, hu] using hsq

end Metastability
end Transformer
