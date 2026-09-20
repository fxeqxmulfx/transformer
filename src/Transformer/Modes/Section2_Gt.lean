/-
# The number of modes of a Gaussian KDE — the summands `G(t)` and `G'(t)`

§2.2 of arXiv:2412.09080v3, `eq: Gt`: the pair whose independent copies `F_n`
and `F_n'` are sums of.

**What the source says and what is carried here.**

* `eq: Gt` is the pair
  `(G(t), G'(t)) = e^{-β(t-X)²/2} (t - X, 1 - β(t-X)²)`, for `X ~ N(0,1)`.
  `bigG` and `bigG'` are its two entries, as functions of the sample point.

* "we have `(F_n(t), F_n'(t)) = n^{-1/2} Σ_i (G_i(t), G_i'(t))`" is
  `fieldF_eq_sum_bigG` for the first entry, and `hasDerivAt_fieldF` for the
  second: the naming `G'` is justified, since `G'(t)` is literally `∂_t G(t)`
  (`hasDerivAt_bigG`).  Both are proved.

* The law of that vector is the density `p_t` of `thm:kac-rice`; that it is one
  is `lem: pt.bdd`, and it is not carried here.

Source: arXiv:2412.09080v3, `eq: Gt`, §2.2.
-/

import Transformer.Modes.Section2_Field

open scoped BigOperators
open Real

namespace Transformer
namespace Modes

variable {n : ℕ}

/-- **Equation (eq: Gt), first entry.**  `G(t) = e^{-β(t-X)²/2}(t - X)`.

Source: arXiv:2412.09080v3, `eq: Gt`. -/
noncomputable def bigG (β t x : ℝ) : ℝ := Real.exp (-(β / 2) * (t - x) ^ 2) * (t - x)

/-- **Equation (eq: Gt), second entry.**  `G'(t) = e^{-β(t-X)²/2}(1 - β(t - X)²)`.

Source: arXiv:2412.09080v3, `eq: Gt`. -/
noncomputable def bigG' (β t x : ℝ) : ℝ :=
  Real.exp (-(β / 2) * (t - x) ^ 2) * (1 - β * (t - x) ^ 2)

/-- The second entry of `eq: Gt` is the `t`-derivative of the first: the name
`G'` is not an abuse. -/
theorem hasDerivAt_bigG (β x t : ℝ) : HasDerivAt (fun s => bigG β s x) (bigG' β t x) t := by
  have hbase : HasDerivAt (fun s : ℝ => s - x) 1 t := (hasDerivAt_id t).sub_const x
  have h := (hasDerivAt_bump β x t).mul hbase
  rw [show -(β * (t - x)) * Real.exp (-(β / 2) * (t - x) ^ 2) * (t - x)
      + Real.exp (-(β / 2) * (t - x) ^ 2) * 1 = bigG' β t x from by rw [bigG']; ring] at h
  exact h

/-! ### `F_n` and `F_n'` as sums of copies of `eq: Gt` -/

/-- **`F_n(t) = n^{-1/2} Σ_i G_i(t)`.**

Source: arXiv:2412.09080v3, §2.2, after `eq: Gt`. -/
theorem fieldF_eq_sum_bigG (β : ℝ) (X : Idx n → ℝ) (t : ℝ) :
    fieldF β X t = (1 / Real.sqrt n) * ∑ i : Idx n, bigG β t (X i) := by
  rw [fieldF]
  exact congrArg _ (Finset.sum_congr rfl fun i _ => by rw [bigG]; ring)

/-- **`F_n'(t) = n^{-1/2} Σ_i G_i'(t)`.**

Source: arXiv:2412.09080v3, §2.2, after `eq: Gt`. -/
theorem hasDerivAt_fieldF (β : ℝ) (X : Idx n → ℝ) (t : ℝ) :
    HasDerivAt (fieldF β X) ((1 / Real.sqrt n) * ∑ i : Idx n, bigG' β t (X i)) t := by
  rw [show fieldF β X = fun s => (1 / Real.sqrt n) * ∑ i : Idx n, bigG β s (X i) from
    funext fun s => fieldF_eq_sum_bigG β X s]
  exact (hasDerivAt_fun_sum _ _ _ _ fun i _ => hasDerivAt_bigG β (X i) t).const_mul _

/-- **The pair of `eq: Gt` summed:**
`(F_n(t), F_n'(t)) = n^{-1/2} Σ_i (G_i(t), G_i'(t))`.

Source: arXiv:2412.09080v3, §2.2, the display after `eq: Gt`. -/
theorem fieldF_pair_eq (β : ℝ) (X : Idx n → ℝ) (t : ℝ) :
    (fieldF β X t, deriv (fieldF β X) t)
      = ((1 / Real.sqrt n) * ∑ i : Idx n, bigG β t (X i),
         (1 / Real.sqrt n) * ∑ i : Idx n, bigG' β t (X i)) :=
  Prod.ext (fieldF_eq_sum_bigG β X t) (hasDerivAt_fieldF β X t).deriv

/-! ### `G'` and the second derivative of the KDE -/

/-- `G'` is what `P̂_n''` is built from: `β(t - X)² - 1` is `-(1 - β(t - X)²)`,
so `P̂_n''` and `F_n'` differ by the constant `-√(2πn/β³)` of `eq:Fn`, as they
must.

Source: arXiv:2412.09080v3, `eq:Fn`, `eq: Gt`. -/
theorem beta_mul_bigG'_eq (β t x : ℝ) :
    β * bigG' β t x = -((β ^ 2 * (t - x) ^ 2 - β) * Real.exp (-(β / 2) * (t - x) ^ 2)) := by
  rw [bigG']
  ring

end Modes
end Transformer
