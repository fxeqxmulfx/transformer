/-
# IC-EoT: the three non-convex toy surfaces

arXiv:2603.22095v2, §3.4, Eqs. (28)–(30). The fractional exponent
in Eq. (30) is retained literally, using the real rational-power convention
`x^(4/3) = |x|^(4/3)` (even numerator, odd denominator), including negative
inputs. It is not silently replaced by the usual six-hump polynomial `x^4/3`.
All three functions fail Jensen's inequality.
-/

import Transformer.ICEoT.Section3_Encoder
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

noncomputable section

namespace Transformer.ICEoT

/-- The oscillatory cosine surface, §3.4, Eq. (28). -/
def toyCosine (v : ℝ × ℝ) : ℝ := -Real.cos (4 * v.1 ^ 2 + 4 * v.2 ^ 2)

/-- The min/max surface, §3.4, Eq. (29). -/
def toyComplex (v : ℝ × ℝ) : ℝ :=
  max (min (v.1 ^ 2 + v.2 ^ 2) ((2 * v.1 - 1) ^ 2 + (2 * v.2 - 1) ^ 2 - 2))
    (-(2 * v.1 + 1) ^ 2 - (2 * v.2 + 1) ^ 2 + 4)

/-- The literal fractional-exponent surface, §3.4, Eq. (30).
The non-convexity witness has `x = 0`, so it also refutes convexity
if the exponent was intended to be the standard six-hump term `x^4/3`. -/
def toySixHump (v : ℝ × ℝ) : ℝ :=
  v.1 ^ 2 * (4 - (21 / 10 : ℝ) * v.1 ^ 2 + Real.rpow |v.1| (4 / 3 : ℝ)) -
    4 * v.2 ^ 2 * (1 - v.2 ^ 2) + v.1 * v.2

/-- A Jensen counterexample to the cosine surface, §3.4, Eq. (28).
At `(r,±r)` it is `-1`, while at `(r,0)` it is `1`, for `r²=π/4`. -/
theorem toyCosine_not_convex : ¬ ConvexOn ℝ Set.univ toyCosine := by
  let r := Real.sqrt (Real.pi / 4)
  have hr : r ^ 2 = Real.pi / 4 := Real.sq_sqrt (by positivity)
  have harg : 4 * r ^ 2 + 4 * r ^ 2 = 2 * Real.pi := by linarith
  have hmid : 4 * r ^ 2 = Real.pi := by linarith
  have hp : toyCosine (r, r) = -1 := by
    simp only [toyCosine, harg, Real.cos_two_pi]
  have hn : toyCosine (r, -r) = -1 := by
    simp only [toyCosine, neg_sq, harg, Real.cos_two_pi]
  have hm : toyCosine ((1 / 2 : ℝ) • (r, r) + (1 / 2 : ℝ) • (r, -r)) = 1 := by
    have hc : (1 / 2 : ℝ) • (r, r) + (1 / 2 : ℝ) • (r, -r) = (r, 0) := by
      apply Prod.ext
      · change (1 / 2 : ℝ) * r + (1 / 2 : ℝ) * r = r
        ring
      · change (1 / 2 : ℝ) * r + (1 / 2 : ℝ) * (-r) = 0
        ring
    rw [hc]
    simp [toyCosine, hmid]
  intro h
  have hj := h.2 (Set.mem_univ (r, r)) (Set.mem_univ (r, -r))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  rw [hp, hn, hm] at hj
  norm_num at hj

/-- A Jensen counterexample to the min/max surface, §3.4, Eq. (29):
values `4`, `-2`, and midpoint `2 > (4-2)/2`. -/
theorem toyComplex_not_convex : ¬ ConvexOn ℝ Set.univ toyComplex := by
  intro h
  have hj := h.2 (Set.mem_univ ((-1 / 2 : ℝ), (-1 / 2 : ℝ)))
    (Set.mem_univ ((1 / 2 : ℝ), (1 / 2 : ℝ)))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  norm_num [toyComplex] at hj

/-- A Jensen counterexample along `x=0` in the third surface; §3.4,
Eq. (30). Values at `y=±1/2` are `-3/4`; the midpoint value is zero. -/
theorem toySixHump_not_convex : ¬ ConvexOn ℝ Set.univ toySixHump := by
  intro h
  have hj := h.2 (Set.mem_univ ((0 : ℝ), (-1 / 2 : ℝ)))
    (Set.mem_univ ((0 : ℝ), (1 / 2 : ℝ)))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  norm_num [toySixHump] at hj

end Transformer.ICEoT
