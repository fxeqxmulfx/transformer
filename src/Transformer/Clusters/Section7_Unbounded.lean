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
  Uniqueness is not used by the lemma and is not carried.

* "converges with doubly exponential rate" in `l:unboundedparticles` is read
  as a bound `|P_ij(t) - δ| ≤ exp(-c e^t)` valid eventually, for some `c > 0`:
  that is what the source's proof produces, from `P_ij(t) ≤ e^{-c x_i(t)}` and
  `x_i(t) ≥ c₁ e^t`.

* The witness for every hypothesis below is the one-token configuration
  `x(t) = e^t z`, which solves `e:Idnonresca` exactly — with `n = 1` the only
  attention weight is `1` and the equation is `ẋ = x`.

Source: arXiv:2305.05465v6, `l:auxiliary`, `l:unboundedparticles`,
`l:exactasymptotic`.
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

/-- **Lemma (l:auxiliary), the largest token.**  Let `A > 0` satisfy
`A² = n² exp(-A²)`.  If `x_n(t₀) > A` for some `t₀ ≥ 0`, then there is
`c₁ > 0` with `x_n(t) ≥ c₁ e^t` for every sufficiently large `t`.

Not proved here.

Source: arXiv:2305.05465v6, `l:auxiliary`. -/
theorem exists_exp_lower_bound_last (A : ℝ) (hA : 0 < A)
    (hAeq : A ^ 2 = ((m + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2))
    (X : ℝ → Idx (m + 1) → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (hord : IsOrderedConfig (X 0)) (t₀ : ℝ) (ht₀ : 0 ≤ t₀)
    (hlast : A < X t₀ (Fin.last m) 0) :
    ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, c * Real.exp t ≤ X t (Fin.last m) 0 := by
  sorry

/-- **Lemma (l:auxiliary), the smallest token.**  Symmetrically, if
`x_1(t₀) < -A` for some `t₀ ≥ 0` then `x_1(t) ≤ -c₁ e^t` for every
sufficiently large `t`.

Not proved here.

Source: arXiv:2305.05465v6, `l:auxiliary`. -/
theorem exists_exp_upper_bound_first (A : ℝ) (hA : 0 < A)
    (hAeq : A ^ 2 = ((m + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2))
    (X : ℝ → Idx (m + 1) → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (hord : IsOrderedConfig (X 0)) (t₀ : ℝ) (ht₀ : 0 ≤ t₀)
    (hfirst : X t₀ 0 0 < -A) :
    ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, X t 0 0 ≤ -(c * Real.exp t) := by
  sorry

/-- The hypotheses of `exists_exp_lower_bound_last` are satisfiable at
`m = 0`: the one-token solution `x(t) = e^t (A+1)` starts above `A`, and the
constant `A` itself exists by `exists_auxiliary_constant`. -/
example : ∃ A : ℝ, 0 < A ∧ A ^ 2 = ((0 + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2) ∧
    IdNonrescaledDynamics (n := 1)
        (fun t _ => Real.exp t • (EuclideanSpace.single 0 (A + 1) : EucSpace 1)) ∧
      IsOrderedConfig (n := 1)
        (fun _ => Real.exp 0 • (EuclideanSpace.single 0 (A + 1) : EucSpace 1)) ∧
      (0 : ℝ) ≤ 0 ∧
      A < (Real.exp 0 • (EuclideanSpace.single 0 (A + 1) : EucSpace 1)) 0 := by
  obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant 1 one_pos
  refine ⟨A, hA, by simpa using hAeq, idNonrescaledDynamics_single _,
    isOrderedConfig_subsingleton _, le_rfl, ?_⟩
  simp

/-! ### `l:unboundedparticles` and `l:exactasymptotic` -/

/-- A token is uniformly bounded in time when one radius holds it for every
`t ≥ 0`.  This is membership in `L^∞([0,+∞))` for a continuous curve.

Source: arXiv:2305.05465v6, `l:unboundedparticles`, `l:onlyone`. -/
def IsBoundedToken (X : ℝ → Idx n → EucSpace 1) (i : Idx n) : Prop :=
  ∃ R : ℝ, ∀ t : ℝ, 0 ≤ t → |X t i 0| ≤ R

/-- **Lemma (l:unboundedparticles).**  If `x_i(t)` is not uniformly bounded,
then it converges to `+∞` or to `-∞`; in the first case `P_ij(t) → δ_{nj}`, in
the second `P_ij(t) → δ_{1j}`, both with doubly exponential rate.

Not proved here.

Source: arXiv:2305.05465v6, `l:unboundedparticles`. -/
theorem unbounded_tendsto_atTop_or_atBot (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) (i : Idx (m + 1))
    (hub : ¬ IsBoundedToken X i) :
    (Tendsto (fun t => X t i 0) atTop atTop ∧
        ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, ∀ j : Idx (m + 1),
          |attentionMatrix (1 : ParamMatrix 1) 1 (X t) i j
              - (if j = Fin.last m then 1 else 0)| ≤ Real.exp (-(c * Real.exp t))) ∨
      (Tendsto (fun t => X t i 0) atTop atBot ∧
        ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, ∀ j : Idx (m + 1),
          |attentionMatrix (1 : ParamMatrix 1) 1 (X t) i j
              - (if j = 0 then 1 else 0)| ≤ Real.exp (-(c * Real.exp t))) := by
  sorry

/-- **Lemma (l:exactasymptotic).**  A token that is not uniformly bounded has
an exact exponential asymptotic: there is `γ_i ≠ 0` with
`x_i(t) = γ_i e^t + o(e^t)` as `t → +∞`.

Not proved here.

Source: arXiv:2305.05465v6, `l:exactasymptotic`. -/
theorem exists_exp_asymptotic (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) (i : Idx (m + 1))
    (hub : ¬ IsBoundedToken X i) :
    ∃ γ : ℝ, γ ≠ 0 ∧ Tendsto (fun t => X t i 0 / Real.exp t) atTop (nhds γ) := by
  sorry

/-- The hypotheses shared by `unbounded_tendsto_atTop_or_atBot` and
`exists_exp_asymptotic` are satisfiable at `m = 0`: the one-token solution
`x(t) = e^t` is not uniformly bounded. -/
example :
    IdNonrescaledDynamics (n := 1)
        (fun t _ => Real.exp t • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) ∧
      IsOrderedConfig (n := 1)
        (fun _ => Real.exp 0 • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) ∧
      ¬ IsBoundedToken
        (fun t (_ : Idx 1) => Real.exp t • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) 0 := by
  refine ⟨idNonrescaledDynamics_single _, isOrderedConfig_subsingleton _, ?_⟩
  rintro ⟨R, hR⟩
  have h1 := hR (|R| + 1) (by positivity)
  have h2 : Real.exp (|R| + 1) ≤ R := by
    simpa [abs_of_pos (Real.exp_pos (|R| + 1))] using h1
  have h3 : |R| + 1 + 1 ≤ Real.exp (|R| + 1) := Real.add_one_le_exp _
  have h4 : R ≤ |R| := le_abs_self R
  linarith

end Clusters
end Transformer
