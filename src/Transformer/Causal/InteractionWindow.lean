/-
# Causal attention — The interaction window (§B of 2411.04990v2)

* `Lemma lem:interaction`  — the two inequalities on the interaction scale
  `β^{-1/2}` that the hypotheses of `thm: fixed_centers` ask of `h` and `g`;
* `Remark rem:interaction` — the explicit `ε, c, β, N` window in which those
  inequalities hold, so that only `β ≳ log N` is needed.

The functions themselves, and the Gaussian bounds the proof runs on, are in
`Causal.Interaction` and `Causal.InteractionBounds`.

**A missing hypothesis.**  `lem:interaction` is stated for every `ε > 0`.  It
is false there: at `ε = 30`, `c = 87.48`, `β = 490.07` and `N = 1.45·10^114`
— all within the lemma's own bounds — the first inequality reads
`1.35·10^{-22} < 1.67·10^{-167}`.  What goes wrong is that `ε β^{-1/2}` is
then far outside the first arch of the sine, so `h(εβ^{-1/2})` is no longer
of order `εβ^{-1/2}`.  The bound `ε < 0.1` that `rem:interaction` carries is
therefore taken as a hypothesis here too; the first failures a sweep finds
sit at `ε ≈ 0.97` (second inequality) and `ε ≈ 1.98` (first).
-/

import Transformer.Basic
import Transformer.Causal.Interaction
import Transformer.Causal.InteractionBounds
import Transformer.Causal.InteractionNumerics

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal


/-- **Lemma (lem:interaction), the core.**  The two inequalities in the
variables the proof actually runs in: `b = c - 1 - 2ε` and `s = β^{-1/2}`,
the latter entering only through `β s² = 1`.

Each of the three points `bs`, `(b+1)s` and `εs` is confined by the
hypotheses to the first arch of the sine, where the Gaussian sandwich of
`Causal.InteractionBounds` applies: `h(bs) < e^{-5b²/12} bs` and
`h(εs) > 0.99 εs`, while the hypothesis on `N` contributes `e^{3b²/8}` and
`e^{3b²/8 - 5b²/12} = e^{-b²/24} ≤ e^{-27/32} ≤ 1/2`.  For `g` the same
sandwich gives `-g((b+1)s) < e^{-365(b+1)²/972}(b+1)²`, which `e^{3b²/8}`
leaves below `e^{-365b/486} ≤ 71/(10b²)`, against `g(εs) > 0.9`.

Source: arXiv:2411.04990v2, §B, `lem:interaction`. -/
theorem interaction_inequalities_core
    (N ε β b s : ℝ) (hε : 0 < ε) (hsmall : ε < 1 / 10) (hN : 0 < N)
    (hb : 9 / 2 ≤ b) (hβ : b ^ 2 / 2 ≤ β) (hs : 0 < s) (hs2 : β * s ^ 2 = 1)
    (hNbound : N < Real.exp (3 * b ^ 2 / 8) * (ε / (b + 2 * ε))) :
    N * h_pot β (b * s) < h_pot β (ε * s) ∧
      -(N * g_pot β ((b + 1) * s)) < g_pot β (ε * s) := by
  have hbpos : (0 : ℝ) < b := by linarith
  have hb2 : 81 / 4 ≤ b ^ 2 := by nlinarith
  have hβ10 : 81 / 8 ≤ β := by linarith
  have hβpos : (0 : ℝ) < β := by linarith
  have hbs2 : b ^ 2 * s ^ 2 ≤ 2 := by
    linarith [mul_le_mul_of_nonneg_right hβ (sq_nonneg s)]
  -- the three points, and the arch of the sine they live in
  have hx1pos : (0 : ℝ) < b * s := mul_pos hbpos hs
  have hx1sq : β * (b * s) ^ 2 = b ^ 2 := by
    have h : β * (b * s) ^ 2 = b ^ 2 * (β * s ^ 2) := by ring
    rw [h, hs2, mul_one]
  have hx1pi : b * s ≤ Real.pi := by
    have h2 : b * s ≤ 3 / 2 := by nlinarith
    linarith [Real.pi_gt_three]
  have hx2pos : (0 : ℝ) < ε * s := mul_pos hε hs
  have hx2sq : β * (ε * s) ^ 2 = ε ^ 2 := by
    have h : β * (ε * s) ^ 2 = ε ^ 2 * (β * s ^ 2) := by ring
    rw [h, hs2, mul_one]
  have hεsq : ε ^ 2 ≤ 1 / 100 := by nlinarith
  have hx2small : (ε * s) ^ 2 ≤ 1 / 1000 := by nlinarith
  have hgauss2 : (199 : ℝ) / 200 ≤ Real.exp (-(ε ^ 2 / 2)) := by
    have h := Real.add_one_le_exp (-(ε ^ 2 / 2) : ℝ)
    linarith
  have hx2pi : ε * s ≤ Real.pi := by
    have h2 : ε * s ≤ 1 / 20 := by nlinarith
    linarith [Real.pi_gt_three]
  have hx3pos : (0 : ℝ) < (b + 1) * s := mul_pos (by linarith) hs
  have hx3sq : β * ((b + 1) * s) ^ 2 = (b + 1) ^ 2 := by
    have h : β * ((b + 1) * s) ^ 2 = (b + 1) ^ 2 * (β * s ^ 2) := by ring
    rw [h, hs2, mul_one]
  constructor
  · -- `N h(bs) < h(εs)`
    obtain ⟨-, hup⟩ := h_pot_bounds β hβpos hx1pos hx1pi
    have hE1 : -(β * (b * s) ^ 2 / 2) + β * (b * s) ^ 4 / 24 ≤ -(5 * b ^ 2 / 12) := by
      have h4 : β * (b * s) ^ 4 = b ^ 2 * (b ^ 2 * s ^ 2) := by
        have h : β * (b * s) ^ 4 = b ^ 2 * (b ^ 2 * s ^ 2) * (β * s ^ 2) := by ring
        rw [h, hs2, mul_one]
      rw [hx1sq, h4]
      nlinarith [mul_le_mul_of_nonneg_left hbs2 (sq_nonneg b)]
    have hh1 : h_pot β (b * s) < Real.exp (-(5 * b ^ 2 / 12)) * (b * s) :=
      lt_of_lt_of_le hup
        (mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr hE1) hx1pos.le)
    obtain ⟨hlow2, -⟩ := h_pot_bounds β hβpos hx2pos hx2pi
    have hh2 : (99 / 100 : ℝ) * (ε * s) < h_pot β (ε * s) := by
      refine lt_of_le_of_lt ?_ hlow2
      rw [hx2sq]
      exact lin_lower_of_small hx2pos hx2small hgauss2
    have hexp24 : Real.exp (3 * b ^ 2 / 8) * Real.exp (-(5 * b ^ 2 / 12))
        = Real.exp (-(b ^ 2 / 24)) := by
      rw [← Real.exp_add]; ring_nf
    have hhalf : Real.exp (-(b ^ 2 / 24)) ≤ 1 / 2 :=
      le_trans (Real.exp_le_exp.mpr (by linarith)) exp_neg_27_32_le_half
    have hratio : ε / (b + 2 * ε) * b ≤ ε := by
      rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith)]
      exact mul_le_mul_of_nonneg_left (by linarith) hε.le
    have key : N * h_pot β (b * s) < (99 / 100 : ℝ) * (ε * s) := by
      calc N * h_pot β (b * s)
          < N * (Real.exp (-(5 * b ^ 2 / 12)) * (b * s)) :=
            mul_lt_mul_of_pos_left hh1 hN
        _ < Real.exp (3 * b ^ 2 / 8) * (ε / (b + 2 * ε))
              * (Real.exp (-(5 * b ^ 2 / 12)) * (b * s)) :=
            mul_lt_mul_of_pos_right hNbound (by positivity)
        _ = Real.exp (3 * b ^ 2 / 8) * Real.exp (-(5 * b ^ 2 / 12))
              * (ε / (b + 2 * ε) * b) * s := by ring
        _ = Real.exp (-(b ^ 2 / 24)) * (ε / (b + 2 * ε) * b) * s := by rw [hexp24]
        _ ≤ 1 / 2 * ε * s :=
            mul_le_mul_of_nonneg_right
              (mul_le_mul hhalf hratio (by positivity) (by norm_num)) hs.le
        _ < 99 / 100 * (ε * s) := by linarith [mul_pos hε hs]
    exact lt_trans key hh2
  · -- `-N g((b+1)s) < g(εs)`
    have hβ1 : (1 : ℝ) ≤ β := by linarith
    have hb1b : (b + 1) ^ 2 ≤ 121 / 81 * b ^ 2 := by
      linarith [mul_nonneg (by linarith : (0 : ℝ) ≤ b - 9 / 2)
        (by linarith : (0 : ℝ) ≤ 40 * b + 18)]
    have hb1s : (b + 1) ^ 2 * s ^ 2 ≤ 242 / 81 := by
      linarith [mul_le_mul_of_nonneg_right hb1b (sq_nonneg s)]
    have hEbound : -(β * ((b + 1) * s) ^ 2 / 2) + β * ((b + 1) * s) ^ 4 / 24
        ≤ -(365 / 972 * (b + 1) ^ 2) := by
      have h4 : β * ((b + 1) * s) ^ 4 = (b + 1) ^ 2 * ((b + 1) ^ 2 * s ^ 2) := by
        have h : β * ((b + 1) * s) ^ 4
            = (b + 1) ^ 2 * ((b + 1) ^ 2 * s ^ 2) * (β * s ^ 2) := by ring
        rw [h, hs2, mul_one]
      rw [hx3sq, h4]
      exact quartic_exponent_le (sq_nonneg _) hb1s
    have hgauss : -(g_pot β ((b + 1) * s))
        < Real.exp (-(365 / 972 * (b + 1) ^ 2)) * (b + 1) ^ 2 := by
      have h := g_pot_lower_bound_gauss β hβ1 hx3pos
      have h1 := Real.exp_le_exp.mpr hEbound
      rw [hx3sq] at h h1
      linarith [mul_le_mul_of_nonneg_right h1 (sq_nonneg (b + 1))]
    -- `g` at the near point is essentially `1`
    have hq : (0 : ℝ) < (β + 1 / 2) ^ (-(1 / 2 : ℝ)) :=
      Real.rpow_pos_of_pos (by linarith) _
    have hq2 : ((β + 1 / 2) ^ (-(1 / 2 : ℝ)) : ℝ) ^ 2 = (β + 1 / 2)⁻¹ := by
      rw [← Real.rpow_natCast ((β + 1 / 2) ^ (-(1 / 2 : ℝ))) 2,
        ← Real.rpow_mul (by linarith)]
      norm_num
      rw [Real.rpow_neg_one]
    have hxb : ε * s < (β + 1 / 2) ^ (-(1 / 2 : ℝ)) := by
      have hkey : (ε * s) ^ 2 * (β + 1 / 2) < 1 := by
        have hexpand : (ε * s) ^ 2 * (β + 1 / 2) = β * (ε * s) ^ 2 + (ε * s) ^ 2 / 2 := by
          ring
        rw [hexpand, hx2sq]
        linarith
      have hlt : (ε * s) ^ 2 < ((β + 1 / 2) ^ (-(1 / 2 : ℝ))) ^ 2 := by
        rw [hq2, inv_eq_one_div, lt_div_iff₀ (by linarith : (0 : ℝ) < β + 1 / 2)]
        exact hkey
      exact lt_of_pow_lt_pow_left₀ 2 hq.le hlt
    have hg2 : (9 / 10 : ℝ) < g_pot β (ε * s) := by
      refine lt_of_le_of_lt ?_ (g_pot_lower_bound_near β hβpos hx2pos hxb)
      rw [hx2sq]
      exact const_lower_of_small hx2small hεsq hgauss2
    have hexpcomb : Real.exp (3 * b ^ 2 / 8) * Real.exp (-(365 / 972 * (b + 1) ^ 2))
        ≤ Real.exp (-(365 / 486 * b)) := by
      rw [← Real.exp_add]
      exact Real.exp_le_exp.mpr (by linarith)
    have hdecay : Real.exp (-(365 / 486 * b)) ≤ 71 / (10 * b ^ 2) := by
      refine le_trans (exp_neg_le_four_div_sq (by positivity)) ?_
      rw [div_le_div_iff₀ (by positivity) (by positivity)]
      linarith [sq_nonneg b]
    have hfrac : ε / (b + 2 * ε) ≤ 1 / (10 * b) := by
      rw [div_le_div_iff₀ (by linarith) (by positivity)]
      linarith [mul_lt_mul_of_pos_right hsmall hbpos]
    have hval : 1 / (10 * b) * (121 / 81 * b ^ 2) * (71 / (10 * b ^ 2))
        = 8591 / (8100 * b) := by
      field_simp; ring
    have hfin : 8591 / (8100 * b) < 9 / 10 := by
      rw [div_lt_div_iff₀ (by positivity) (by norm_num)]
      linarith
    have key : -(N * g_pot β ((b + 1) * s)) < (9 / 10 : ℝ) := by
      calc -(N * g_pot β ((b + 1) * s))
          = N * -(g_pot β ((b + 1) * s)) := by ring
        _ < N * (Real.exp (-(365 / 972 * (b + 1) ^ 2)) * (b + 1) ^ 2) :=
            mul_lt_mul_of_pos_left hgauss hN
        _ < Real.exp (3 * b ^ 2 / 8) * (ε / (b + 2 * ε))
              * (Real.exp (-(365 / 972 * (b + 1) ^ 2)) * (b + 1) ^ 2) :=
            mul_lt_mul_of_pos_right hNbound (by positivity)
        _ = ε / (b + 2 * ε) * (b + 1) ^ 2
              * (Real.exp (3 * b ^ 2 / 8) * Real.exp (-(365 / 972 * (b + 1) ^ 2))) := by
            ring
        _ ≤ ε / (b + 2 * ε) * (b + 1) ^ 2 * Real.exp (-(365 / 486 * b)) :=
            mul_le_mul_of_nonneg_left hexpcomb (by positivity)
        _ ≤ ε / (b + 2 * ε) * (b + 1) ^ 2 * (71 / (10 * b ^ 2)) :=
            mul_le_mul_of_nonneg_left hdecay (by positivity)
        _ ≤ 1 / (10 * b) * (121 / 81 * b ^ 2) * (71 / (10 * b ^ 2)) :=
            mul_le_mul_of_nonneg_right
              (mul_le_mul hfrac hb1b (sq_nonneg _) (by positivity)) (by positivity)
        _ = 8591 / (8100 * b) := hval
        _ < 9 / 10 := hfin
    exact lt_trans key hg2

/-- The hypotheses of `interaction_inequalities_core` are satisfiable:
`N = 1/100`, `ε = 1/20`, `β = 16`, `b = 9/2`, `s = 1/4`. -/
example : ∃ N ε β b s : ℝ, 0 < ε ∧ ε < 1 / 10 ∧ 0 < N ∧ 9 / 2 ≤ b ∧
    b ^ 2 / 2 ≤ β ∧ 0 < s ∧ β * s ^ 2 = 1 ∧
    N < Real.exp (3 * b ^ 2 / 8) * (ε / (b + 2 * ε)) := by
  refine ⟨1 / 100, 1 / 20, 16, 9 / 2, 1 / 4, by norm_num, by norm_num, by norm_num,
    le_rfl, by norm_num, by norm_num, by norm_num, ?_⟩
  have h := Real.add_one_le_exp (3 * (9 / 2 : ℝ) ^ 2 / 8)
  rw [show (3 : ℝ) * (9 / 2 : ℝ) ^ 2 / 8 = 243 / 32 by norm_num] at h ⊢
  rw [show ((1 : ℝ) / 20) / ((9 : ℝ) / 2 + 2 * (1 / 20)) = 1 / 92 by norm_num]
  nlinarith

/-- **Lemma (lem:interaction).** *The interaction inequalities of
`thm: fixed_centers`.*

If

  `ε < 0.1`,  `c ≥ 5.5 + 2ε`,  `β ≥ (c - 1 - 2ε)² / 2`,
  `N < e^{3(c-1-2ε)²/8} · ε/(c-1)`,

then

  `N h((c - 1 - 2ε) β^{-1/2}) < h(ε β^{-1/2})`  and
  `-N g((c - 2ε) β^{-1/2}) < g(ε β^{-1/2})`.

The second inequality is stated as the lemma states it, with the minus sign;
`thm: fixed_centers` asks for `N g((c-2ε)β^{-1/2}) < g(εβ^{-1/2})` without it,
and the two agree exactly when `g((c-2ε)β^{-1/2}) ≤ 0`, which the peak
location of `h_pot_unimodal` gives at `c - 2ε > 1`.

The bound `ε < 0.1` is not in the lemma but is needed — see the file header.
Everything else is `interaction_inequalities_core` at `b = c - 1 - 2ε` and
`s = β^{-1/2}`.

Source: arXiv:2411.04990v2, §B, `lem:interaction`. -/
theorem interaction_inequalities
    (N c ε β : ℝ) (hε : 0 < ε) (hsmall : ε < 1 / 10) (hN : 0 < N)
    (hc : 5.5 + 2 * ε ≤ c) (hβ : (c - 1 - 2 * ε) ^ 2 / 2 ≤ β)
    (hNbound : N < Real.exp (3 * (c - 1 - 2 * ε) ^ 2 / 8) * (ε / (c - 1))) :
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
        < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
      -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ))))
        < g_pot β (ε * β ^ (-(1 / 2 : ℝ))) := by
  have hb : 9 / 2 ≤ c - 1 - 2 * ε := by norm_num at hc ⊢; linarith
  have hβpos : (0 : ℝ) < β := by nlinarith
  have hs : (0 : ℝ) < β ^ (-(1 / 2 : ℝ)) := Real.rpow_pos_of_pos hβpos _
  have hs2 : β * (β ^ (-(1 / 2 : ℝ))) ^ 2 = 1 := by
    have h2 : (β ^ (-(1 / 2 : ℝ))) ^ 2 = β⁻¹ := by
      rw [← Real.rpow_natCast (β ^ (-(1 / 2 : ℝ))) 2, ← Real.rpow_mul hβpos.le]
      norm_num
      rw [Real.rpow_neg_one]
    rw [h2, mul_inv_cancel₀ (ne_of_gt hβpos)]
  have hNbound' : N < Real.exp (3 * (c - 1 - 2 * ε) ^ 2 / 8)
      * (ε / (c - 1 - 2 * ε + 2 * ε)) := by
    rw [show c - 1 - 2 * ε + 2 * ε = c - 1 by ring]; exact hNbound
  rw [show c - 2 * ε = c - 1 - 2 * ε + 1 by ring]
  exact interaction_inequalities_core N ε β _ _ hε hsmall hN hb hβ hs hs2 hNbound'

/-- The hypotheses of `interaction_inequalities` are satisfiable: the remark's
own example with `ε` halved, `ε = 0.05`, `c = 6.5`, `β = 15`, `N = 1`.  Note
`5.5 + 2ε = 5.6 ≤ 6.5` and `(c - 1 - 2ε)²/2 = 5.4²/2 = 14.58 ≤ 15`. -/
example :
    (0 : ℝ) < 0.05 ∧ (0.05 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 ∧ (5.5 : ℝ) + 2 * 0.05 ≤ 6.5 ∧
      ((6.5 : ℝ) - 1 - 2 * 0.05) ^ 2 / 2 ≤ 15 ∧
      (1 : ℝ) < Real.exp (3 * ((6.5 : ℝ) - 1 - 2 * 0.05) ^ 2 / 8) * (0.05 / (6.5 - 1)) := by
  refine ⟨by norm_num, by norm_num, one_pos, by norm_num, by norm_num, ?_⟩
  have h1 : (3.6 : ℝ) ≤ Real.exp 2.6 := by
    nlinarith [Real.add_one_le_exp (2.6 : ℝ)]
  have h2 : Real.exp 10.4 = Real.exp 2.6 ^ 4 := by
    rw [show (10.4 : ℝ) = (4 : ℕ) * 2.6 by norm_num, Real.exp_nat_mul]
  have h3 : (167 : ℝ) ≤ Real.exp 10.4 := by
    rw [h2]
    have := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 3.6) h1 4
    norm_num at this
    linarith
  have h4 : Real.exp 10.4
      ≤ Real.exp (3 * ((6.5 : ℝ) - 1 - 2 * 0.05) ^ 2 / 8) :=
    Real.exp_le_exp.mpr (by norm_num)
  nlinarith

/-- **Remark (rem:interaction).** *An explicit window for `thm: fixed_centers`.*

The hypotheses of `thm: fixed_centers` hold whenever

  `ε < 0.1`,  `c ≥ 5.5 + 2ε`,  `β ≥ (c - 1 - 2ε)²/2`,
  `N ≤ (ε / (c - 1)) e^{3(c-1-2ε)²/8}`,

so that only `β ≳ log N` is needed.  This is `interaction_inequalities` read
as a sufficient condition, with the strict bound on `N` relaxed to `≤`; the
remark's own example is `ε = 0.1`, `c = 6.5`, `β ≥ 14`, `N ≤ 700`.

A `Prop`-valued definition: it asserts that the window implies the two
inequalities, and is not proved here.

Source: arXiv:2411.04990v2, §B, `rem:interaction`. -/
def InteractionWindow : Prop :=
  ∀ N c ε β : ℝ, 0 < ε → ε < 0.1 → 0 < N →
    5.5 + 2 * ε ≤ c → (c - 1 - 2 * ε) ^ 2 / 2 ≤ β →
    N ≤ (ε / (c - 1)) * Real.exp (3 * (c - 1 - 2 * ε) ^ 2 / 8) →
      N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
          < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
        N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
          < g_pot β (ε * β ^ (-(1 / 2 : ℝ)))

end Causal
end Transformer
