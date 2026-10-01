/-
# Endpoint signs at an isolated polynomial zero

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.RootFlanks
import Transformer.Sturm.SignRelation

noncomputable section
open Filter Topology Polynomial

namespace Transformer.Sturm

/-- A negative derivative at a zero gives positivity on the left and
negativity on the right. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem negative_derivative_zero_flanks {f : ℝ → ℝ} {r d : ℝ}
    (h : HasDerivAt f d r) (hzero : f r = 0) (hd : d < 0) :
    (∀ᶠ x in 𝓝[<] r, 0 < f x) ∧ (∀ᶠ x in 𝓝[>] r, f x < 0) := by
  have hn := positive_derivative_zero_flanks h.neg
    (by change -f r = 0; rw [hzero, neg_zero]) (neg_pos.mpr hd)
  exact ⟨hn.1.mono (fun _ hx => neg_lt_zero.mp hx),
    hn.2.mono (fun _ hx => neg_pos.mp hx)⟩

/-- A sign valid just left of an isolated zero holds at any earlier
endpoint without intervening zeros. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomial_sign_left_endpoint {P : Polynomial ℝ} {a r : ℝ} (har : a < r)
    (hne : ∀ x ∈ Set.Ico a r, P.eval x ≠ 0) (s : SignType)
    (hs : ∀ᶠ x in 𝓝[<] r, SignType.sign (P.eval x) = s) :
    SignType.sign (P.eval a) = s := by
  obtain ⟨c, hc, hac, hcr⟩ := (hs.and (Ioo_mem_nhdsLT har)).exists
  have heq := eval_sign_eq_of_no_zero hac.le (fun x hx =>
    hne x ⟨hx.1, hx.2.trans_lt hcr⟩)
  exact heq.trans hc

/-- A sign valid just right of an isolated zero holds at any later
endpoint without intervening zeros. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomial_sign_right_endpoint {P : Polynomial ℝ} {r b : ℝ} (hrb : r < b)
    (hne : ∀ x ∈ Set.Ioc r b, P.eval x ≠ 0) (s : SignType)
    (hs : ∀ᶠ x in 𝓝[>] r, SignType.sign (P.eval x) = s) :
    SignType.sign (P.eval b) = s := by
  obtain ⟨c, hc, hrc, hcb⟩ := (hs.and (Ioo_mem_nhdsGT hrb)).exists
  have heq := eval_sign_eq_of_no_zero hcb.le (fun x hx =>
    hne x ⟨hrc.trans_le hx.1, hx.2⟩)
  exact heq.symm.trans hc

/-- The head-pair sign contribution drops by the sign of its nonzero
derivative at an isolated zero. Both crossing orientations are covered.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomial_zero_indicator_jump {P : Polynomial ℝ} {r a b : ℝ}
    (hr : P.eval r = 0) (hd : P.derivative.eval r ≠ 0) (har : a < r) (hrb : r < b)
    (hne : ∀ x ∈ Set.Icc a b, x ≠ r → P.eval x ≠ 0) :
    (if P.eval a < 0 then (1 : ℤ) else 0) - (if P.eval b < 0 then (1 : ℤ) else 0) =
      (SignType.sign (P.derivative.eval r) : ℤ) := by
  have hl : ∀ x ∈ Set.Ico a r, P.eval x ≠ 0 := fun x hx =>
    hne x ⟨hx.1, hx.2.le.trans hrb.le⟩ (ne_of_lt hx.2)
  have hh : ∀ x ∈ Set.Ioc r b, P.eval x ≠ 0 := fun x hx =>
    hne x ⟨har.le.trans hx.1.le, hx.2⟩ (ne_of_gt hx.1)
  rcases lt_or_gt_of_ne hd with hneg | hpos
  · obtain ⟨hL, hR⟩ := negative_derivative_zero_flanks (P.hasDerivAt r) hr hneg
    have ha := polynomial_sign_left_endpoint har hl 1 (hL.mono (fun _ h => sign_pos h))
    have hb := polynomial_sign_right_endpoint hrb hh (-1) (hR.mono (fun _ h => sign_neg h))
    have hap : 0 < P.eval a := sign_eq_one_iff.mp ha
    have hbn : P.eval b < 0 := sign_eq_neg_one_iff.mp hb
    simp only [not_lt_of_ge hap.le, hbn, ite_false, ite_true, sign_neg hneg,
      SignType.coe_neg_one]
    norm_num
  · obtain ⟨hL, hR⟩ := positive_derivative_zero_flanks (P.hasDerivAt r) hr hpos
    have ha := polynomial_sign_left_endpoint har hl (-1) (hL.mono (fun _ h => sign_neg h))
    have hb := polynomial_sign_right_endpoint hrb hh 1 (hR.mono (fun _ h => sign_pos h))
    have han : P.eval a < 0 := sign_eq_neg_one_iff.mp ha
    have hbp : 0 < P.eval b := sign_eq_one_iff.mp hb
    simp only [han, not_lt_of_ge hbp.le, ite_true, ite_false, sign_pos hpos,
      SignType.coe_one]
    norm_num

/-- The linear polynomial `X` and its negation witness both derivative
orientations and an interval with a single zero. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : HasDerivAt (fun x : ℝ => -x) (-1) 0 ∧ (fun x : ℝ => -x) 0 = 0 ∧
    (-1 : ℝ) < 0 ∧ (X : Polynomial ℝ).eval 0 = 0 ∧
    (X : Polynomial ℝ).derivative.eval 0 ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, x ≠ 0 → (X : Polynomial ℝ).eval x ≠ 0) := by
  refine ⟨(hasDerivAt_id 0).neg, by simp, by norm_num, by simp, by simp,
    by norm_num, fun _ _ hx => ?_⟩
  simpa only [eval_X] using hx

end Transformer.Sturm
