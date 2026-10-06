import Transformer.GPTMini.Semantics.ParityFeatures

/-!
# A homogeneous bounded-count decoder using ordinary ReLU2

Source: the unchanged bias-free ReLU2_FFN at f11b6e2 and the 0..16 ONE
counts allowed by Basis parity at cbafbe9. This new construction uses
four shifted quadratic hinges per integer count. Its compact support
is narrower than the distance between integers, so each count activates
exactly one bump. A signed sum has the required even/odd sign.

The shifts are multiplied by the internal BOS channel. They are linear
forms in the two hidden coordinates, not biases or an external length
input. The construction is homogeneous of degree two and therefore
retains its sign after the positive prenorm multiplier. Seventeen bumps
require 68 ReLU2 units, within the original 256-unit FFN. Realizing the
corresponding matrices and verifying EOS/tied readout are later steps;
this scalar result makes no convex-training or convergence claim.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- A compact quadratic bump equal to one at zero and zero at every other integer.
Source: the new four-hinge construction for the original ReLU2 FFN, with support in [-3/4,3/4]. -/
noncomputable def countBump (x : ℝ) : ℝ :=
  (8 / 3 : ℝ) * (relu2 (x + 3 / 4) - 3 * relu2 (x + 1 / 4) +
    3 * relu2 (x - 1 / 4) - relu2 (x - 3 / 4))

/-- The four actual hinge values normalize the center of the bump to one.
Source: the explicit quadratic activation, not a defined indicator of the count. -/
theorem countBump_zero : countBump 0 = 1 := by
  norm_num [countBump, relu2]

/-- All four hinges vanish below the left support endpoint.
Source: the nonpositive branch of the original ReLU2 activation. -/
theorem countBump_left_zero (x : ℝ) (hx : x ≤ -3 / 4) : countBump x = 0 := by
  have h1 : x + 3 / 4 ≤ 0 := by linarith
  have h2 : x + 1 / 4 ≤ 0 := by linarith
  have h3 : x - 1 / 4 ≤ 0 := by linarith
  have h4 : x - 3 / 4 ≤ 0 := by linarith
  rw [countBump, relu2_of_nonpos _ h1, relu2_of_nonpos _ h2,
    relu2_of_nonpos _ h3, relu2_of_nonpos _ h4]
  ring

example : (-1 : ℝ) ≤ -3 / 4 := by norm_num

/-- The four quadratic terms cancel exactly above the right support endpoint.
Source: finite differences of degree-two polynomials, using the actual ReLU2 positive branches. -/
theorem countBump_right_zero (x : ℝ) (hx : 3 / 4 ≤ x) : countBump x = 0 := by
  have h1 : 0 ≤ x + 3 / 4 := by linarith
  have h2 : 0 ≤ x + 1 / 4 := by linarith
  have h3 : 0 ≤ x - 1 / 4 := by linarith
  have h4 : 0 ≤ x - 3 / 4 := by linarith
  unfold countBump relu2
  rw [max_eq_right h1, max_eq_right h2, max_eq_right h3, max_eq_right h4]
  ring

example : (3 / 4 : ℝ) ≤ 1 := by norm_num

/-- On integer counts the continuous bump is an exact equality test.
Source: disjoint unit-spaced count centers, derived from the hinge support rather than built into its definition. -/
theorem countBump_nat (n k : ℕ) :
    countBump ((n : ℝ) - k) = if n = k then 1 else 0 := by
  by_cases heq : n = k
  · rw [heq, sub_self, countBump_zero, ite_eq_left rfl]
  · rw [ite_eq_right heq]
    by_cases hlt : n < k
    · have hn : n + 1 ≤ k := by omega
      have hr : (n : ℝ) + 1 ≤ k := by exact_mod_cast hn
      exact countBump_left_zero _ (by linarith)
    · have hn : k + 1 ≤ n := by omega
      have hr : (k : ℝ) + 1 ≤ n := by exact_mod_cast hn
      exact countBump_right_zero _ (by linarith)

/-- Positive scaling commutes with the quadratic hinges.
Source: ReLU2's positive homogeneity; this is the effect of the actual RMSNorm multiplier. -/
theorem relu2_mul_nonneg (s x : ℝ) (hs : 0 ≤ s) : relu2 (s * x) = s ^ 2 * relu2 x := by
  have hm : max 0 (s * x) = s * max 0 x := by
    simpa only [mul_zero] using (mul_max_of_nonneg (a := s) 0 x hs).symm
  unfold relu2
  rw [hm]
  ring

example : (0 : ℝ) ≤ 3 / 19 := by norm_num

/-- The count and denominator alone provide all four bias-free linear forms.
Source: the new homogeneous decoder in the unchanged ReLU2_FFN operator type. -/
noncomputable def countBumpHom (a b k : ℝ) : ℝ :=
  (8 / 3 : ℝ) * (relu2 (a - (k - 3 / 4) * b) - 3 * relu2 (a - (k - 1 / 4) * b) +
    3 * relu2 (a - (k + 1 / 4) * b) - relu2 (a - (k + 3 / 4) * b))

/-- The internal denominator converts the bias-free homogeneous bump to the unit-spaced count bump.
Source: the simultaneous ONE/BOS representation; no division operation is added to the FFN. -/
theorem countBumpHom_scale (s c k : ℝ) (hs : 0 ≤ s) :
    countBumpHom (s * c) s k = s ^ 2 * countBump (c - k) := by
  have h1 : s * c - (k - 3 / 4) * s = s * (c - k + 3 / 4) := by ring
  have h2 : s * c - (k - 1 / 4) * s = s * (c - k + 1 / 4) := by ring
  have h3 : s * c - (k + 1 / 4) * s = s * (c - k - 1 / 4) := by ring
  have h4 : s * c - (k + 3 / 4) * s = s * (c - k - 3 / 4) := by ring
  unfold countBumpHom countBump
  rw [h1, h2, h3, h4]
  simp only [relu2_mul_nonneg s _ hs]
  ring

example : (0 : ℝ) ≤ 3 / 19 := by norm_num

/-- Even counts receive a positive coefficient, odd counts a negative coefficient.
Source: Basis's EVEN/ODD semantics, applied only to seventeen count centers. -/
def countSign (k : ℕ) : ℝ := if k % 2 = 1 then -1 else 1

/-- A finite, linear-in-count-range sum of actual quadratic hinges.
Source: the new 68-unit decoder, not a table over all possible token prefixes. -/
noncomputable def paritySpline (a b : ℝ) : ℝ :=
  ∑ k : Fin 17, countSign k.val * countBumpHom a b (k.val : ℝ)

/-- Every bounded integer count selects exactly its own parity coefficient in the homogeneous FFN signal.
Source: the original 16-bit cap and the proved disjoint support of the four-hinge bumps. -/
theorem paritySpline_exact (s : ℝ) (hs : 0 ≤ s) (n : ℕ) (hn : n ≤ 16) :
    paritySpline (s * n) s = s ^ 2 * countSign n := by
  unfold paritySpline
  simp_rw [countBumpHom_scale s _ _ hs, countBump_nat]
  rw [Fintype.sum_eq_single (⟨n, by omega⟩ : Fin 17)]
  · simp
    ring
  · intro k hk
    have hne : n ≠ k.val := by
      intro h
      apply hk
      exact Fin.ext h.symm
    simp only [ite_eq_right hne, mul_zero]

example : (0 : ℝ) ≤ 3 / 19 ∧ (2 : ℕ) ≤ 16 := ⟨by norm_num, by decide⟩

/-- The decoder has a strictly signed margin for every legal count when the denominator is positive.
Source: the exact homogeneous formula; a zero ONE count still has a positive even margin. -/
theorem paritySpline_signed (s : ℝ) (hs : 0 < s) (n : ℕ) (hn : n ≤ 16) :
    if n % 2 = 1 then paritySpline (s * n) s < 0 else 0 < paritySpline (s * n) s := by
  rw [paritySpline_exact s hs.le n hn]
  unfold countSign
  split_ifs <;> nlinarith [sq_pos_of_pos hs]

example : (0 : ℝ) < 3 / 19 ∧ (0 : ℕ) ≤ 16 := ⟨by norm_num, by decide⟩

/-!
The count centers do not encode complete token prefixes: every permutation
and every length with the same ONE count uses the same center. The BOS
feature supplies the scale. This scalar construction addresses parity
decoding only; raw key/value binding and depth order require other features.
-/

end Transformer.GPTMini.Semantics
