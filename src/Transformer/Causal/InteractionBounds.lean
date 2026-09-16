/-
# Causal attention — The Gaussian bounds on `h` and `g` (§B of 2411.04990v2)

`lemma:interaction` (4) and (5): the interaction potential
`h(x) = e^{β(cos x - 1)} sin x` and its derivative `g = h'` are squeezed
between Gaussians of width `β^{-1/2}`.  Everything here is the pair of Taylor
bounds `x - x³/6 < sin x < x` and `1 - x²/2 ≤ cos x ≤ 1 - x²/2 + x⁴/24`
transported through `exp`, so `cos_le_quartic` — which Mathlib does not carry —
comes first.

Two of the three inequalities need a hypothesis the paper leaves implicit, and
Lean is what made that visible:

* the lower bound on `h` needs `x ≤ π`.  Past `π` the factor `sin x` turns
  negative while `x - x³/6` is already negative and exponentially smaller: at
  `β = 5`, `x = 3π/2` the claim reads `-10⁻²³ < -6.7·10⁻³`, which is false.
* the lower bound `-e^{-βx²/2 + βx⁴/24} βx² < g(x)` needs `1 ≤ β`.  Dividing
  by `e^{β(cos x - 1)} > 0` it comes down to `cos x + β(x² - sin²x) > 0`, and
  the supremum of `-cos x / (x² - sin²x)` over `x > 0` is `≈ 0.141`, attained
  near `x ≈ 2.26`; at `β = 0.05`, `x = 1.66` the claim is false.  `1 ≤ β` is
  the hypothesis `h_pot_unimodal` already carries, and `rem:interaction` works
  at `β ≥ 14`.

The bound on `g` near the origin holds for every `β > 0` as stated, so it is a
separate theorem rather than a conjunct.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (4) and (5).
-/

import Transformer.Causal.Interaction
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal

/-- **The quartic Taylor bound for the cosine:** `cos x ≤ 1 - x²/2 + x⁴/24`.

Mathlib has the quadratic companion `Real.one_sub_sq_div_two_le_cos` but not
this one.  It is `Real.sin_gt_sub_cube` integrated from `0`: the difference
`1 - x²/2 + x⁴/24 - cos x` vanishes at `0` and has derivative
`sin x - (x - x³/6) > 0`, hence is monotone on `[0, ∞)`; both sides are even.

Source: the alternating Taylor series of `cos`. -/
theorem cos_le_quartic (x : ℝ) : Real.cos x ≤ 1 - x ^ 2 / 2 + x ^ 4 / 24 := by
  have key : ∀ y : ℝ, 0 ≤ y → Real.cos y ≤ 1 - y ^ 2 / 2 + y ^ 4 / 24 := by
    intro y hy
    have hd : ∀ z : ℝ,
        HasDerivAt (fun w : ℝ => 1 - w ^ 2 / 2 + w ^ 4 / 24 - Real.cos w)
          (-z + z ^ 3 / 6 + Real.sin z) z := by
      intro z
      have h2 : HasDerivAt (fun w : ℝ => w ^ 2) (2 * z) z := by
        simpa using hasDerivAt_pow 2 z
      have h4 : HasDerivAt (fun w : ℝ => w ^ 4) (4 * z ^ 3) z := by
        simpa using hasDerivAt_pow 4 z
      have hp := (((hasDerivAt_const z (1 : ℝ)).sub (h2.div_const 2)).add
        (h4.div_const 24)).sub (Real.hasDerivAt_cos z)
      exact hp.congr_deriv (by ring)
    have hmono : MonotoneOn (fun w : ℝ => 1 - w ^ 2 / 2 + w ^ 4 / 24 - Real.cos w)
        (Set.Ici 0) := by
      refine monotoneOn_of_deriv_nonneg (convex_Ici 0)
        (fun z _ => (hd z).continuousAt.continuousWithinAt)
        (fun z _ => (hd z).differentiableAt.differentiableWithinAt) ?_
      intro z hz
      rw [interior_Ici] at hz
      rw [(hd z).deriv]
      have := Real.sin_gt_sub_cube (Set.mem_Ioi.mp hz)
      linarith
    have h0 := hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hy) hy
    norm_num at h0
    linarith
  rcases le_total 0 x with h | h
  · exact key x h
  · have hk := key (-x) (by linarith)
    rw [Real.cos_neg, show (-x) ^ 2 = x ^ 2 by ring, show (-x) ^ 4 = x ^ 4 by ring] at hk
    exact hk

/-- `sin² x < x²` away from the origin, the form both bounds use. -/
theorem sin_sq_lt_sq {x : ℝ} (hx : 0 < x) : Real.sin x ^ 2 < x ^ 2 := by
  have h := Real.abs_sin_lt_abs (ne_of_gt hx)
  rw [abs_of_pos hx] at h
  have h1 : |Real.sin x| ^ 2 < x ^ 2 := by
    have := abs_nonneg (Real.sin x)
    nlinarith
  rwa [sq_abs] at h1

/-- The hypothesis of `sin_sq_lt_sq` is satisfiable: `x = 1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- `e^{-βx²/2} ≤ e^{β(cos x - 1)} ≤ e^{-βx²/2 + βx⁴/24}`, the exponential
factor of `h` and `g` squeezed between the two Gaussians. -/
theorem exp_factor_bounds (β : ℝ) (hβ : 0 ≤ β) (x : ℝ) :
    Real.exp (-(β * x ^ 2 / 2)) ≤ Real.exp (β * (Real.cos x - 1)) ∧
      Real.exp (β * (Real.cos x - 1))
        ≤ Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) := by
  constructor
  · exact Real.exp_le_exp.mpr (by nlinarith [Real.one_sub_sq_div_two_le_cos (x := x)])
  · exact Real.exp_le_exp.mpr (by nlinarith [cos_le_quartic x])

/-- The hypothesis of `exp_factor_bounds` is satisfiable: `β = 1`. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

/-- **Lemma (lemma:interaction), 4.** *Gaussian bounds on `h`.*

For `0 < x ≤ π`,

  `e^{-βx²/2} (x - x³/6) < h(x) < e^{-βx²/2 + βx⁴/24} x`.

The paper states both for every `x > 0`.  The upper bound does hold there —
past `π` the left-hand side is negative and the right-hand side positive — but
the lower bound does not: see the file header.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (4). -/
theorem h_pot_bounds (β : ℝ) (hβ : 0 < β) {x : ℝ} (hx : 0 < x) (hxpi : x ≤ Real.pi) :
    Real.exp (-(β * x ^ 2 / 2)) * (x - x ^ 3 / 6) < h_pot β x ∧
      h_pot β x < Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) * x := by
  obtain ⟨hlow, hup⟩ := exp_factor_bounds β hβ.le x
  have hsin : 0 ≤ Real.sin x := Real.sin_nonneg_of_nonneg_of_le_pi hx.le hxpi
  constructor
  · have h1 : Real.exp (-(β * x ^ 2 / 2)) * (x - x ^ 3 / 6)
        < Real.exp (-(β * x ^ 2 / 2)) * Real.sin x :=
      mul_lt_mul_of_pos_left (Real.sin_gt_sub_cube hx) (Real.exp_pos _)
    have h2 : Real.exp (-(β * x ^ 2 / 2)) * Real.sin x ≤ h_pot β x :=
      mul_le_mul_of_nonneg_right hlow hsin
    linarith
  · have h1 : h_pot β x ≤ Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) * Real.sin x :=
      mul_le_mul_of_nonneg_right hup hsin
    have h2 : Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) * Real.sin x
        < Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) * x :=
      mul_lt_mul_of_pos_left (Real.sin_lt hx) (Real.exp_pos _)
    linarith

/-- The hypotheses of `h_pot_bounds` are satisfiable: `β = 1`, `x = 1 ≤ π`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ Real.pi :=
  ⟨one_pos, one_pos, by linarith [Real.pi_gt_three]⟩

/-- **Lemma (lemma:interaction), 5, first bound.** *`g` near the origin.*

  `e^{-βx²/2} (1 - x²/2 - βx²) < g(x)`   for `0 < x < (β + 1/2)^{-1/2}`.

The restriction on `x` is what makes the bracket positive, so that the two
factor-wise estimates may be multiplied.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (5). -/
theorem g_pot_lower_bound_near (β : ℝ) (hβ : 0 < β) {x : ℝ} (hx : 0 < x)
    (hxb : x < (β + 1 / 2) ^ (-(1 / 2 : ℝ))) :
    Real.exp (-(β * x ^ 2 / 2)) * (1 - x ^ 2 / 2 - β * x ^ 2) < g_pot β x := by
  have hb : 0 < β + 1 / 2 := by linarith
  have hsq : x ^ 2 * (β + 1 / 2) < 1 := by
    have h1 : x ^ 2 < ((β + 1 / 2) ^ (-(1 / 2 : ℝ))) ^ 2 := by nlinarith [hxb, hx]
    have h2 : ((β + 1 / 2) ^ (-(1 / 2 : ℝ)) : ℝ) ^ 2 = (β + 1 / 2)⁻¹ := by
      rw [← Real.rpow_natCast ((β + 1 / 2) ^ (-(1 / 2 : ℝ))) 2, ← Real.rpow_mul hb.le]
      norm_num
      rw [Real.rpow_neg_one]
    rw [h2] at h1
    calc x ^ 2 * (β + 1 / 2) < (β + 1 / 2)⁻¹ * (β + 1 / 2) := by nlinarith
      _ = 1 := inv_mul_cancel₀ (ne_of_gt hb)
  have hbracket : 0 < 1 - x ^ 2 / 2 - β * x ^ 2 := by nlinarith
  have hcos : 1 - x ^ 2 / 2 < Real.cos x := Real.one_sub_sq_div_two_lt_cos (ne_of_gt hx)
  have hsin := sin_sq_lt_sq hx
  have hfactor : 1 - x ^ 2 / 2 - β * x ^ 2 < Real.cos x - β * Real.sin x ^ 2 := by
    nlinarith
  obtain ⟨hlow, -⟩ := exp_factor_bounds β hβ.le x
  calc Real.exp (-(β * x ^ 2 / 2)) * (1 - x ^ 2 / 2 - β * x ^ 2)
      < Real.exp (-(β * x ^ 2 / 2)) * (Real.cos x - β * Real.sin x ^ 2) :=
        mul_lt_mul_of_pos_left hfactor (Real.exp_pos _)
    _ ≤ g_pot β x := mul_le_mul_of_nonneg_right hlow (by linarith)

/-- The hypotheses of `g_pot_lower_bound_near` are satisfiable: `β = 7/2` and
`x = 1/4 < 4^{-1/2} = 1/2`. -/
example : (0 : ℝ) < 7 / 2 ∧ (0 : ℝ) < 1 / 4 ∧
    (1 / 4 : ℝ) < ((7 : ℝ) / 2 + 1 / 2) ^ (-(1 / 2 : ℝ)) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  rw [show ((7 : ℝ) / 2 + 1 / 2) = 4 by norm_num, Real.rpow_neg (by norm_num),
    ← Real.sqrt_eq_rpow,
    show Real.sqrt 4 = 2 by
      rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
  norm_num

/-- **Lemma (lemma:interaction), 5, second bound.** *`g` is never far below
zero.*

  `-e^{-βx²/2 + βx⁴/24} βx² < g(x)`   for `x > 0` and `β ≥ 1`.

Written out, `g(x) + e^{-βx²/2 + βx⁴/24} βx²` equals

  `e^{β(cos x - 1)} (cos x + β(x² - sin²x)) + βx² (E - e^{β(cos x - 1)})`,

whose second summand is non-negative by `exp_factor_bounds`; the first is
positive because `x² - sin²x > 1/4` once `cos x < 0`, which forces `x > π/2`.
The paper states the bound for every `β > 0`; it fails there — see the file
header.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (5). -/
theorem g_pot_lower_bound_gauss (β : ℝ) (hβ : 1 ≤ β) {x : ℝ} (hx : 0 < x) :
    -(Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) * (β * x ^ 2)) < g_pot β x := by
  have hβ0 : (0 : ℝ) < β := by linarith
  have hsin := sin_sq_lt_sq hx
  obtain ⟨-, hup⟩ := exp_factor_bounds β hβ0.le x
  -- `cos x + β (x² - sin² x) > 0`
  have hkey : 0 < Real.cos x + β * (x ^ 2 - Real.sin x ^ 2) := by
    rcases le_or_gt 0 (Real.cos x) with hc | hc
    · nlinarith
    · have hxhalf : Real.pi / 2 < x := by
        by_contra hcon
        exact absurd (Real.cos_nonneg_of_mem_Icc
          ⟨by linarith [Real.pi_pos, hx], not_lt.mp hcon⟩) (not_le.mpr hc)
      have hx15 : (1.5 : ℝ) < x := by linarith [Real.pi_gt_three]
      have h1 : Real.sin x ^ 2 ≤ 1 := Real.sin_sq_le_one x
      have h2 : -1 ≤ Real.cos x := Real.neg_one_le_cos x
      nlinarith
  have hA : (0 : ℝ) < Real.exp (β * (Real.cos x - 1)) := Real.exp_pos _
  have hgap : 0 ≤ (Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24)
      - Real.exp (β * (Real.cos x - 1))) * (β * x ^ 2) := by
    have : (0 : ℝ) ≤ β * x ^ 2 := by positivity
    nlinarith
  have hexpand : g_pot β x + Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) * (β * x ^ 2)
      = Real.exp (β * (Real.cos x - 1)) * (Real.cos x + β * (x ^ 2 - Real.sin x ^ 2))
        + (Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24)
            - Real.exp (β * (Real.cos x - 1))) * (β * x ^ 2) := by
    unfold g_pot; ring
  nlinarith [mul_pos hA hkey]

/-- The hypotheses of `g_pot_lower_bound_gauss` are satisfiable: `β = 1`,
`x = 1`. -/
example : (1 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 := ⟨le_rfl, one_pos⟩

end Causal
end Transformer
