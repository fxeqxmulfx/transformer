/-
# DASH — the algebraic restriction of square-root chaining

arXiv:2602.02016v2, §3.3, after `equation:NDB-Y-Z`. Arbitrary
choices of square or inverse square root still give only signed
dyadic exponents. These are compositions of the exact root functions;
finite numerical convergence requires the separately proved domains
and, when needed, normalization between calls.
-/

import Transformer.DASH.Section3_Spectral

noncomputable section

namespace Transformer.DASH

/-- Exponent obtained from a list of exact square-root calls. `true`
selects the inverse square root. The tail is evaluated before the head.
Source: arXiv:2602.02016v2, §3.3, only powers `±1/2^k` by chaining. -/
def ndbSelectedExponent : List Bool → ℝ
  | [] => 1
  | inverse :: stages =>
      ndbSelectedExponent stages * (if inverse then -(1 / 2 : ℝ) else 1 / 2)

/-- Composition of the exact root functions to which valid NDB calls
converge, allowing either output at each stage. This is not an assertion
that unscaled finite NDB iterates converge for every such intermediate input.
Source: arXiv:2602.02016v2, §3.3, chaining the two root outputs. -/
def ndbSelectedScalarChain (a : ℝ) : List Bool → ℝ
  | [] => a
  | inverse :: stages =>
      ndbSelectedScalarChain a stages ^ (if inverse then -(1 / 2 : ℝ) else 1 / 2)

/-- Every selected exact-root chain is a power with the product of
its stage exponents. Positivity is the real inverse-root domain.
Source: arXiv:2602.02016v2, §3.3, the algebraic chaining restriction. -/
theorem ndbSelectedScalarChain_eq (a : ℝ) (ha : 0 < a) (stages : List Bool) :
    ndbSelectedScalarChain a stages = a ^ ndbSelectedExponent stages := by
  induction stages with
  | nil => simp [ndbSelectedScalarChain, ndbSelectedExponent]
  | cons inverse stages ih =>
    rw [ndbSelectedScalarChain, ih, ← Real.rpow_mul ha.le, ndbSelectedExponent]

/-- Positive input assumptions hold for any selected root chain,
arXiv:2602.02016v2, §3.3. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Choosing either root output changes only the sign of the exponent;
its magnitude halves at each call.
Source: arXiv:2602.02016v2, §3.3, only signed dyadic powers. -/
theorem ndbSelectedExponent_abs (stages : List Bool) :
    |ndbSelectedExponent stages| = (1 / 2 : ℝ) ^ stages.length := by
  induction stages with
  | nil => norm_num [ndbSelectedExponent]
  | cons inverse stages ih =>
    cases inverse <;> norm_num [ndbSelectedExponent, abs_mul, ih, pow_succ]

/-- All possible exact-root chaining choices, including inverse roots
at intermediate stages, have exponent `+1/2^k` or `-1/2^k`, where `k`
is the number of calls. This proves the source's “only” restriction,
not just the construction of one particular dyadic chain.
Source: arXiv:2602.02016v2, §3.3, final paragraph. -/
theorem ndbSelectedExponent_only_dyadic (stages : List Bool) :
    ndbSelectedExponent stages = (1 / 2 : ℝ) ^ stages.length ∨
      ndbSelectedExponent stages = -((1 / 2 : ℝ) ^ stages.length) := by
  have habs := ndbSelectedExponent_abs stages
  rcases le_total 0 (ndbSelectedExponent stages) with h | h
  · rw [abs_of_nonneg h] at habs
    exact Or.inl habs
  · rw [abs_of_nonpos h] at habs
    exact Or.inr (by linarith)

end Transformer.DASH
