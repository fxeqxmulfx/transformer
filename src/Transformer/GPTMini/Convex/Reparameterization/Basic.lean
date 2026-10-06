import Transformer.GPTMini.Convex.Likelihood.Basic

/-!
# The prediction class of a scalar shared-key softmax head

New finite control derived from arXiv:2211.11052v1, §3's row-softmax
Q/K product. Two query rows read the same three keys. Values are the
three standard category vectors, so the predictions are the actual
attention probabilities. There is no residual, FFN, or learned value map.
This is a shared-memory head control, not the complete causal GPTMini.

The log-odds matrix against category two has rank at most one in this
class, independent of the choice of factor coordinates. Two concrete
heads are retained as coverage witnesses for a later likelihood test.
The membership predicate requires real original head factors; it does
not replace a substantive conclusion by a defining property.
-/

noncomputable section

namespace Transformer.GPTMini.Convex.Reparameterization

open scoped BigOperators

/-- Two query rows' categorical predictions over three shared memory keys.
Source: the new finite shared-memory specialization of §3's row-softmax operator. -/
abbrev SharedProbability := Fin 2 → Fin 3 → ℝ

/-- The actual positive row normalizer of one free scalar Q/K head.
Source: arXiv:2211.11052v1, §3, with two queries and three common keys. -/
def scalarHeadNormalizer (q : Fin 2 → ℝ) (k : Fin 3 → ℝ) (r : Fin 2) : ℝ :=
  ∑ c, Real.exp (q r * k c)

/-- Actual normalized predictions of the free scalar query/key factors.
Source: §3's row softmax; one-hot memory values expose its probabilities directly. -/
def scalarHeadProbability (q : Fin 2 → ℝ) (k : Fin 3 → ℝ) : SharedProbability :=
  fun r c => Real.exp (q r * k c) / scalarHeadNormalizer q k r

/-- Membership in the unchanged physical scalar-head prediction class.
Source: the new control's original Q/K factors; both are unrestricted real tables. -/
def ScalarHeadClass (p : SharedProbability) : Prop :=
  ∃ q : Fin 2 → ℝ, ∃ k : Fin 3 → ℝ, scalarHeadProbability q k = p

/-- Every actual finite softmax row has a strictly positive normalizer.
Source: §3's three positive exponential terms. -/
theorem scalarHeadNormalizer_pos (q : Fin 2 → ℝ) (k : Fin 3 → ℝ) (r : Fin 2) :
    0 < scalarHeadNormalizer q k r := by
  exact Finset.sum_pos (fun c hc => Real.exp_pos _) Finset.univ_nonempty

/-- Every category has positive mass for every original head assignment.
Source: §3's actual softmax formula, without a probability-domain assumption. -/
theorem scalarHeadProbability_pos (q : Fin 2 → ℝ) (k : Fin 3 → ℝ)
    (r : Fin 2) (c : Fin 3) : 0 < scalarHeadProbability q k r c := by
  exact div_pos (Real.exp_pos _) (scalarHeadNormalizer_pos q k r)

/-- All three actual probabilities sum to one in each row.
Source: §3's actual row softmax, not an independently postulated simplex table. -/
theorem scalarHeadProbability_sum (q : Fin 2 → ℝ) (k : Fin 3 → ℝ) (r : Fin 2) :
    ∑ c, scalarHeadProbability q k r c = 1 := by
  unfold scalarHeadProbability
  rw [← Finset.sum_div]
  change scalarHeadNormalizer q k r / scalarHeadNormalizer q k r = 1
  exact div_self (scalarHeadNormalizer_pos q k r).ne'

/-- An actual category log probability is its Q/K score minus the shared row normalizer.
Source: §3's row softmax and the finite positive normalizer identity. -/
theorem scalarHeadProbability_log (q : Fin 2 → ℝ) (k : Fin 3 → ℝ)
    (r : Fin 2) (c : Fin 3) :
    Real.log (scalarHeadProbability q k r c) = q r * k c - Real.log (scalarHeadNormalizer q k r) := by
  unfold scalarHeadProbability
  rw [Real.log_div (Real.exp_pos _).ne' (scalarHeadNormalizer_pos q k r).ne', Real.log_exp]

/-- The normalizer cancels in every category log-odds contrast.
Source: §3's shared-key softmax scores, before any change of factor coordinates. -/
theorem scalarHeadProbability_log_odds (q : Fin 2 → ℝ) (k : Fin 3 → ℝ)
    (r : Fin 2) (c : Fin 3) :
    Real.log (scalarHeadProbability q k r c) - Real.log (scalarHeadProbability q k r 2) =
      q r * (k c - k 2) := by
  rw [scalarHeadProbability_log, scalarHeadProbability_log]
  ring

/-- The log-odds minor of the two rows against the third category.
Source: the new prediction-space rank test, derived from §3's scalar Q/K product. -/
def logOddsMinor (p : SharedProbability) : ℝ :=
  (Real.log (p 0 0) - Real.log (p 0 2)) * (Real.log (p 1 1) - Real.log (p 1 2)) -
    (Real.log (p 0 1) - Real.log (p 0 2)) * (Real.log (p 1 0) - Real.log (p 1 2))

/-- Every original scalar head has zero log-odds minor, whatever its factor coordinates.
Source: the new control's actual normalized outputs and §3's scalar matching product. -/
theorem scalarHeadProbability_minor (q : Fin 2 → ℝ) (k : Fin 3 → ℝ) :
    logOddsMinor (scalarHeadProbability q k) = 0 := by
  unfold logOddsMinor
  rw [scalarHeadProbability_log_odds, scalarHeadProbability_log_odds,
    scalarHeadProbability_log_odds, scalarHeadProbability_log_odds]
  ring

/-- First coverage witness: only the first query has a nonzero score multiplier.
Source: the new §3 finite shared-memory counterexample construction. -/
def leftQueries (r : Fin 2) : ℝ := if r = 0 then 1 else 0

/-- The first witness favors category zero in its nonuniform query row.
Source: the new control's exact score log 2, giving probabilities (1/2, 1/4, 1/4). -/
def leftKeys (c : Fin 3) : ℝ := if c = 0 then Real.log 2 else 0

/-- Second coverage witness: only the second query has a nonzero score multiplier.
Source: the new §3 finite shared-memory counterexample construction. -/
def rightQueries (r : Fin 2) : ℝ := if r = 1 then 1 else 0

/-- The second witness favors category one in its nonuniform query row.
Source: the new control's exact score log 2. -/
def rightKeys (c : Fin 3) : ℝ := if c = 1 then Real.log 2 else 0

/-- Actual predictions of the first physical witness, including its uniform other row.
Source: the new §3 finite control, evaluated rather than stipulated. -/
def leftProbability : SharedProbability := scalarHeadProbability leftQueries leftKeys

/-- Actual predictions of the second physical witness, including its uniform other row.
Source: the new §3 finite control, evaluated rather than stipulated. -/
def rightProbability : SharedProbability := scalarHeadProbability rightQueries rightKeys

/-- The first physical head's probabilities are exactly the stated rational table.
Source: the new finite control's real Q/K scores and real softmax. -/
theorem leftProbability_eq (r : Fin 2) (c : Fin 3) :
    leftProbability r c = if r = 0 then (if c = 0 then 1 / 2 else 1 / 4) else 1 / 3 := by
  by_cases hr : r = 0 <;> by_cases hc : c = 0 <;>
    norm_num [leftProbability, scalarHeadProbability, scalarHeadNormalizer,
      leftQueries, leftKeys, Fin.sum_univ_three, hr, hc,
      Real.exp_log (by norm_num : (0 : ℝ) < 2)]

/-- The second physical head's probabilities are exactly the stated rational table.
Source: the new finite control's real Q/K scores and real softmax. -/
theorem rightProbability_eq (r : Fin 2) (c : Fin 3) :
    rightProbability r c = if r = 1 then (if c = 1 then 1 / 2 else 1 / 4) else 1 / 3 := by
  by_cases hr : r = 1 <;> by_cases hc : c = 1 <;>
    norm_num [rightProbability, scalarHeadProbability, scalarHeadNormalizer,
      rightQueries, rightKeys, Fin.sum_univ_three, hr, hc,
      Real.exp_log (by norm_num : (0 : ℝ) < 2)]

/-- Any assignment in the unchanged physical class must pass the zero-minor test.
Source: the new actual-softmax rank identity, independent of any chosen parameterization. -/
theorem scalarHeadClass_minor {p : SharedProbability} (hp : ScalarHeadClass p) :
    logOddsMinor p = 0 := by
  obtain ⟨q, k, h⟩ := hp
  rw [← h]
  exact scalarHeadProbability_minor q k

example : ScalarHeadClass leftProbability := by
  exact ⟨leftQueries, leftKeys, rfl⟩

end Transformer.GPTMini.Convex.Reparameterization
