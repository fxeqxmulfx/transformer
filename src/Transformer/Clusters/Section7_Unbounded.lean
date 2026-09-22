/-
# The emergence of clusters in self-attention dynamics — unbounded particles

§7 of arXiv:2305.05465v6, `s:unbounded`: in `d = 1`, a token that does not
stay bounded runs off to `±∞` at an exponential rate, and its row of the
self-attention matrix becomes a standard basis row.

**What the source says and what is carried here.**

* §7 orders the tokens once and for all by `x_1(0) < … < x_n(0)`, which
  `l:distnondec` then preserves; `IsOrderedConfig` is that hypothesis, and the
  extreme indices `1` and `n` are `0` and `Fin.last m` for `n = m + 1`.  The
  `d = 1` coordinate of a token is `X t i 0`.

* `l:auxiliary` names `A` as "the unique positive real number satisfying
  `A² = n² exp(-A²)`".  Its existence is proved — `exists_auxiliary_constant`
  — and `A` is then carried as a hypothesis, as the lemma states it.
  Uniqueness is not used by the lemma and is not carried.  The lemma itself
  is proved in `Section7_Auxiliary.lean`.

* "converges with doubly exponential rate" in `l:unboundedparticles` is read
  as a bound `|P_ij(t) - δ| ≤ exp(-c e^t)` valid eventually, for some `c > 0`:
  that is what the source's proof produces, from `P_ij(t) ≤ e^{-c x_i(t)}` and
  `x_i(t) ≥ c₁ e^t`.

* The witness for every hypothesis below is the one-token configuration
  `x(t) = e^t z`, which solves `e:Idnonresca` exactly — with `n = 1` the only
  attention weight is `1` and the equation is `ẋ = x`.

Source: arXiv:2305.05465v6, `l:auxiliary`, `l:unboundedparticles`,
`l:exactasymptotic`; the lemmas themselves are proved in
`Section7_Auxiliary.lean`, `Section7_UnboundedParticles.lean` and
`Section7_ExactAsymptotic.lean`.
-/

import Transformer.Clusters.Section7_LogSumExp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n m : ℕ}

/-! ### The ordering convention of §7 -/

/-- **The ordering convention of §7.**  In `d = 1` the tokens are relabelled
so that `x_1(0) < … < x_n(0)`; in particular they are pairwise distinct.

Source: arXiv:2305.05465v6, §7, the display before `e:infdist`. -/
def IsOrderedConfig (X : Idx n → EucSpace 1) : Prop :=
  ∀ i j : Idx n, i < j → X i 0 < X j 0

/-- A single token is trivially ordered. -/
theorem isOrderedConfig_subsingleton (X : Idx 1 → EucSpace 1) : IsOrderedConfig X := by
  intro i j hij
  exact absurd (Subsingleton.elim i j) hij.ne

/-- **The one-token solution of `e:Idnonresca`.**  With `n = 1` the only
attention weight is `1`, so the equation is `ẋ = x` and `x(t) = e^t z` solves
it.  This is the solution in closed form that witnesses the hypotheses of
§7. -/
theorem idNonrescaledDynamics_single (z : EucSpace d) :
    IdNonrescaledDynamics (n := 1) (fun t _ => Real.exp t • z) := by
  intro t i
  have hsum : ∑ j : Idx 1,
      attentionMatrix (1 : ParamMatrix d) 1 (fun _ => Real.exp t • z) i j •
        (Real.exp t • z) = Real.exp t • z := by
    rw [← Finset.sum_smul, sum_attentionMatrix one_pos, one_smul]
  rw [hsum]
  exact (Real.hasDerivAt_exp t).smul_const z

/-! ### `l:auxiliary` -/

/-- **The constant `A` of `l:auxiliary` exists.**  Substituting `u = A²`, the
equation `A² = n² e^{-A²}` reads `u e^u = n²`, and `u ↦ u e^u` runs
continuously from `0` to at least `n²` on `[0, n²]`.

Source: arXiv:2305.05465v6, `l:auxiliary`. -/
theorem exists_auxiliary_constant (n : ℕ) (hn : 0 < n) :
    ∃ A : ℝ, 0 < A ∧ A ^ 2 = (n : ℝ) ^ 2 * Real.exp (-A ^ 2) := by
  have hn' : (0 : ℝ) < (n : ℝ) ^ 2 := by positivity
  have hcont : ContinuousOn (fun u : ℝ => u * Real.exp u) (Set.Icc 0 ((n : ℝ) ^ 2)) :=
    (continuous_id.mul Real.continuous_exp).continuousOn
  have hmem : ((n : ℝ) ^ 2) ∈
      Set.Icc ((fun u : ℝ => u * Real.exp u) 0)
        ((fun u : ℝ => u * Real.exp u) ((n : ℝ) ^ 2)) := by
    constructor
    · show (0 : ℝ) * Real.exp 0 ≤ (n : ℝ) ^ 2
      simp
    · show (n : ℝ) ^ 2 ≤ (n : ℝ) ^ 2 * Real.exp ((n : ℝ) ^ 2)
      exact le_mul_of_one_le_right hn'.le (Real.one_le_exp hn'.le)
  obtain ⟨u, hu, hug⟩ := intermediate_value_Icc hn'.le hcont hmem
  have hupos : 0 < u := by
    rcases eq_or_lt_of_le hu.1 with h | h
    · exact absurd (by simpa [← h] using hug) hn'.ne
    · exact h
  refine ⟨Real.sqrt u, Real.sqrt_pos.mpr hupos, ?_⟩
  have hsq : Real.sqrt u ^ 2 = u := Real.sq_sqrt hupos.le
  rw [hsq, ← hug]
  rw [mul_assoc, ← Real.exp_add, add_neg_cancel, Real.exp_zero, mul_one]

/-! ### `l:unboundedparticles` and `l:exactasymptotic` -/

/-- A token is uniformly bounded in time when one radius holds it for every
`t ≥ 0`.  This is membership in `L^∞([0,+∞))` for a continuous curve.

Source: arXiv:2305.05465v6, `l:unboundedparticles`, `l:onlyone`. -/
def IsBoundedToken (X : ℝ → Idx n → EucSpace 1) (i : Idx n) : Prop :=
  ∃ R : ℝ, ∀ t : ℝ, 0 ≤ t → |X t i 0| ≤ R

/-- The one-token solution `x(t) = e^t` is not uniformly bounded. -/
theorem not_isBoundedToken_exp :
    ¬ IsBoundedToken
      (fun t (_ : Idx 1) => Real.exp t • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) 0 := by
  rintro ⟨R, hR⟩
  have h1 := hR (|R| + 1) (by positivity)
  have h2 : Real.exp (|R| + 1) ≤ R := by
    simpa [abs_of_pos (Real.exp_pos (|R| + 1))] using h1
  have h3 : |R| + 1 + 1 ≤ Real.exp (|R| + 1) := Real.add_one_le_exp _
  have h4 : R ≤ |R| := le_abs_self R
  linarith

end Clusters
end Transformer
