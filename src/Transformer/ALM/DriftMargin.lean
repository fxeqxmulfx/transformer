/-
# What is true instead of exactness

`Transformer.ALM.CumSum` shows the cumulative sum coming back off by a
relative `2u` and the query inheriting it linearly, so the hypothesis `q : ℤ`
of `Transformer.ALM.FloatGrid.fp_exact_of_grid` does not hold of the running
machine: `todo3.md` §8a measures 2.8 % to 7.1 % of the hard-attention queries
of the six reference programs missing an integer, `collatz` by two units in the
last place at `2573.000000000001`.  Everything §0 proves about exactness
applies to the keys and not to these queries.

What is available instead is exactness *within a margin*, and the margin is
what this file bounds.  A query `ε` off its integer value moves each score by
at most `2Kε`, distinct integer keys at an integer query are a whole unit apart
(`Transformer.ALM.LatestWindow.one_le_sScore_sub`), so the retrieval is
unchanged as long as `4Kε < 1` — and `lookup_survives_drift` runs that from the
recovered counter through `5·cursor + 1` to the answer.

`symmetric_tie_broken_by_drift` is the other side, and the reason the margin
cannot be widened to cover everything: a query that misses every stored key
ties the two keys either side of it, that tie is exact at an integer query and
broken at a drifted one, and which way it breaks is decided by the sign of a
rounding error.  This is where `todo3.md` §3 finds `HullKVCache` disagreeing
with its own reference head, every observed disagreement on a miss.

Source: `todo3.md` §8, §8a and §3; `transformer_vm/wasm/interpreter.py:316-320`.
-/

import Transformer.ALM.CumSum

namespace Transformer
namespace ALM

/-! ### The score at a query that is not an integer -/

/-- The score of an integer key at a real query — `sScore` with the integrality
of the query given up, which is what `fetch_sum` hands the lookup. -/
noncomputable def qScore (q : ℝ) (k : ℤ) : ℝ := 2 * (k : ℝ) * q - (k : ℝ) ^ 2

@[simp] lemma qScore_intCast (q k : ℤ) : qScore (q : ℝ) k = sScore q k := by
  unfold qScore sScore; ring

/-- **The drift a key sees is its own size times the query's.**  The score is
linear in the query with slope `2k`, so a query `ε` off its integer value moves
the score of a key bounded by `K` by at most `2Kε`. -/
lemma qScore_drift {q : ℝ} {q₀ k : ℤ} {K ε : ℝ} (hq : |q - (q₀ : ℝ)| ≤ ε)
    (hk : |(k : ℝ)| ≤ K) : |qScore q k - sScore q₀ k| ≤ 2 * K * ε := by
  have h : qScore q k - sScore q₀ k = 2 * (k : ℝ) * (q - (q₀ : ℝ)) := by
    unfold qScore sScore; ring
  rw [h, abs_mul, abs_mul, abs_two]
  have hb : 0 ≤ |q - (q₀ : ℝ)| := abs_nonneg _
  have hK0 : 0 ≤ K := (abs_nonneg ((k : ℝ))).trans hk
  linarith [mul_le_mul hk hq hb hK0]

/-! ### The margin -/

/-- **A strict order between two keys survives the drift.**  Distinct integer
keys at an integer query are a whole unit apart, and each of the two scores
moves by at most `2Kε`, so `4Kε < 1` keeps the comparison the machine makes
the comparison the specification asks for. -/
theorem order_survives_drift {q : ℝ} {q₀ j k : ℤ} {K ε : ℝ}
    (hq : |q - (q₀ : ℝ)| ≤ ε) (hj : |(j : ℝ)| ≤ K) (hk : |(k : ℝ)| ≤ K)
    (hmargin : 4 * K * ε < 1) (hlt : sScore q₀ j < sScore q₀ k) :
    qScore q j < qScore q k := by
  have h1 := one_le_sScore_sub hlt
  have aj := abs_le.mp (qScore_drift hq hj)
  have ak := abs_le.mp (qScore_drift hq hk)
  linarith

/-- **So the winner is the same key.**  Whatever the head would have retrieved
at the integer query, it retrieves at the drifted one, provided every stored
key is inside the margin. -/
theorem winner_survives_drift {q : ℝ} {q₀ b : ℤ} {K ε : ℝ} {S : Finset ℤ}
    (hq : |q - (q₀ : ℝ)| ≤ ε) (hb : b ∈ S) (hK : ∀ k ∈ S, |(k : ℝ)| ≤ K)
    (hmargin : 4 * K * ε < 1)
    (hwin : ∀ k ∈ S, k ≠ b → sScore q₀ k < sScore q₀ b) :
    ∀ k ∈ S, k ≠ b → qScore q k < qScore q b :=
  fun k hk hne => order_survives_drift hq (hK k hk) (hK b hb) hmargin (hwin k hk hne)

/-- **The whole path, from the division to the answer.**  A counter recovered
`δ` off its integer value makes the query `5·cursor + 1` land `5δ` off the
integer query — `affine_drift`, nothing damps it — and the retrieval is
unchanged as long as `20Kδ < 1`.  This is what replaces exactness on the main
path of `wasm/interpreter.py:320`. -/
theorem lookup_survives_drift {r δ K : ℝ} {c₀ b : ℤ} {S : Finset ℤ}
    (hr : |r - (c₀ : ℝ)| ≤ δ) (hb : b ∈ S) (hK : ∀ k ∈ S, |(k : ℝ)| ≤ K)
    (hmargin : 20 * K * δ < 1)
    (hwin : ∀ k ∈ S, k ≠ b → sScore (5 * c₀ + 1) k < sScore (5 * c₀ + 1) b) :
    ∀ k ∈ S, k ≠ b → qScore (5 * r + 1) k < qScore (5 * r + 1) b := by
  have hq : |(5 * r + 1) - ((5 * c₀ + 1 : ℤ) : ℝ)| ≤ 5 * δ := by
    have h := affine_drift 5 1 hr
    rw [abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 5)] at h
    push_cast
    exact h
  exact winner_survives_drift (q₀ := 5 * c₀ + 1) (ε := 5 * δ) hq hb hK (by linarith) hwin

/-! ### And where the margin runs out -/

/-- **A miss ties the two keys around it, and the drift decides the tie.**  At
an integer query the keys `q₀ - d` and `q₀ + d` score alike, which is the exact
tie `todo3.md` §3 finds `HullKVCache` and `BruteAttentionHead` resolving
differently.  At a query pushed off by `δ > 0` the tie is gone and the larger
key wins — not because the machine chose it, but because one rounding of
`fetch_sum` happened to have that sign. -/
theorem symmetric_tie_broken_by_drift {q₀ j k : ℤ} {δ : ℝ} (hjk : j < k)
    (hsym : j + k = 2 * q₀) (hδ : 0 < δ) :
    sScore q₀ j = sScore q₀ k ∧ qScore ((q₀ : ℝ) + δ) j < qScore ((q₀ : ℝ) + δ) k := by
  have hc : (j : ℝ) + (k : ℝ) = 2 * (q₀ : ℝ) := by exact_mod_cast hsym
  have hlt : (j : ℝ) < (k : ℝ) := by exact_mod_cast hjk
  constructor
  · unfold sScore
    linear_combination ((k : ℝ) - (j : ℝ)) * hc
  · unfold qScore
    nlinarith [hlt, hδ]

/-! ### The hypotheses are satisfiable -/

/-- The margin's hypotheses, at a query that is exactly an integer and keys
well inside it: `q₀ = 3`, keys `2` and `3`, a drift of `10⁻⁹` against a key
bound of `100`. -/
example :
    |(3 : ℝ) - ((3 : ℤ) : ℝ)| ≤ 1e-9 ∧ |((2 : ℤ) : ℝ)| ≤ 100 ∧ |((3 : ℤ) : ℝ)| ≤ 100 ∧
      4 * (100 : ℝ) * 1e-9 < 1 ∧ sScore 3 2 < sScore 3 3 := by
  norm_num [sScore]

/-- And the whole path: the cursor `0`, the query `5·0 + 1`, the keys `{0, 1}`
of which `1` wins, a recovered counter `10⁻⁹` off. -/
example :
    |(0 : ℝ) - ((0 : ℤ) : ℝ)| ≤ 1e-9 ∧ (1 : ℤ) ∈ ({0, 1} : Finset ℤ) ∧
      (∀ k ∈ ({0, 1} : Finset ℤ), |(k : ℝ)| ≤ 1) ∧ 20 * (1 : ℝ) * 1e-9 < 1 ∧
      ∀ k ∈ ({0, 1} : Finset ℤ), k ≠ 1 →
        sScore (5 * 0 + 1) k < sScore (5 * 0 + 1) 1 := by
  refine ⟨by norm_num, by norm_num, ?_, by norm_num, ?_⟩
  · intro k hk
    simp only [Finset.mem_insert, Finset.mem_singleton] at hk
    rcases hk with rfl | rfl <;> norm_num
  · intro k hk hne
    simp only [Finset.mem_insert, Finset.mem_singleton] at hk
    rcases hk with rfl | rfl
    · norm_num [sScore]
    · exact absurd rfl hne

/-- The broken tie: the keys `2` and `4` around the query `3`. -/
example : (2 : ℤ) < 4 ∧ (2 : ℤ) + 4 = 2 * 3 ∧ (0 : ℝ) < 1e-9 := by norm_num

/-- And the margin the chain actually leaves.  At the working precision
`u = 2⁻⁵³`, a counter of `10⁷` recovered through `fetch_sum` drifts by at most
`(2u + u²)·10⁷`, and keys of the same size still clear `20Kδ < 1`: on this
bound alone the lookup is exact to ten million tokens. -/
example :
    20 * (10 ^ 7 : ℝ) * ((2 * (1 / 2 ^ 53) + (1 / 2 ^ 53 : ℝ) ^ 2) * 10 ^ 7) < 1 := by
  norm_num

end ALM
end Transformer
