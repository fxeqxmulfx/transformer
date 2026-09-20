/-
# The emergence of clusters in self-attention dynamics — asymptotic
  low-rankness

§2 of arXiv:2305.05465v6: in `d = 1` the self-attention matrix converges to a
Boolean matrix of low rank, `t:boolean`.

**What the source says and what is carried here.**

* The target set `𝒫` is given by a figure (`e:star`) rather than by a
  formula: "asterisks denote arbitrary non-negative reals which add up to 1",
  and "the row of asterisks may actually be any row between the first and the
  last one".  The proof in §7 says what this amounts to: all rows but at most
  one are `e_1` or `e_n`, and the remaining row is an arbitrary probability
  vector.  `IsBooleanLimit` is that, with the two distinguished columns and
  the exceptional row existentially quantified — the figure's permutation
  matrices are exactly this freedom of labelling.

* Such a matrix has at most three distinct rows, hence rank at most three;
  when the exceptional row is itself `e_1` or `e_n` the rank drops to two.
  `IsBooleanRows` is that stronger form, which is what the source's sentence
  "for almost all initial sequences of pairwise distinct tokens, `P*` is
  actually of rank 1 or 2" asserts.

* "For almost all initial sequences" is read as: outside a Lebesgue-null set
  of `ℝ^n`, the initial tokens being scalars.  `Idx n → EucSpace 1` carries no
  measure in Mathlib, so the initial datum is read through its single
  coordinate, `fun i => X 0 i 0`.

* `V > 0` and `QK > 0` are `IsPosDefOp V` and `IsPosDefQK Q K` at `d = 1`,
  where a map is multiplication by a scalar and positive definiteness is
  positivity of that scalar.

Source: arXiv:2305.05465v6, `t:boolean`, `e:star`.
-/

import Transformer.Clusters.Section1_Dynamics
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

open scoped BigOperators
open Real MeasureTheory Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The set `𝒫` -/

/-- **The set `𝒫` of Fig. `e:star`.**  A stochastic matrix all of whose rows
but at most one are standard basis rows `e_a` or `e_b`, the remaining row
being an arbitrary probability vector.

Source: arXiv:2305.05465v6, `e:star`, `t:boolean`. -/
def IsBooleanLimit (P : Idx n → Idx n → ℝ) : Prop :=
  ∃ a b i₀ : Idx n,
    (∀ i : Idx n, i ≠ i₀ → P i = Pi.single a (1 : ℝ) ∨ P i = Pi.single b (1 : ℝ)) ∧
      (∀ j : Idx n, 0 ≤ P i₀ j) ∧ ∑ j : Idx n, P i₀ j = 1

/-- **The rank `≤ 2` form.**  Every row is `e_a` or `e_b`: the exceptional row
of `𝒫` is itself one of the two basis rows.

Source: arXiv:2305.05465v6, §2, the sentence after `t:boolean`. -/
def IsBooleanRows (P : Idx n → Idx n → ℝ) : Prop :=
  ∃ a b : Idx n, ∀ i : Idx n, P i = Pi.single a (1 : ℝ) ∨ P i = Pi.single b (1 : ℝ)

/-- A standard basis row has non-negative entries. -/
theorem pi_single_one_nonneg (c j : Idx n) :
    0 ≤ (Pi.single c (1 : ℝ) : Idx n → ℝ) j := by
  rcases eq_or_ne j c with rfl | hj
  · simp
  · simp [Pi.single_eq_of_ne hj]

/-- A standard basis row is a probability vector. -/
theorem sum_pi_single_one (c : Idx n) :
    ∑ j : Idx n, (Pi.single c (1 : ℝ) : Idx n → ℝ) j = 1 := by
  simp

/-- The rank `≤ 2` form is a special case of `𝒫`: a basis row is in
particular a probability vector, so it may serve as the row of asterisks. -/
theorem isBooleanLimit_of_isBooleanRows {P : Idx n → Idx n → ℝ} (h : IsBooleanRows P) :
    IsBooleanLimit P := by
  obtain ⟨a, b, h⟩ := h
  refine ⟨a, b, a, fun i _ => h i, ?_, ?_⟩
  · intro j
    rcases h a with hh | hh <;> rw [hh] <;> exact pi_single_one_nonneg _ j
  · rcases h a with hh | hh <;> rw [hh] <;> exact sum_pi_single_one _

/-! ### `t:boolean` -/

/-- **Theorem (t:boolean).**  Let `d = 1`, `V > 0` and `QK > 0`.  For any
initial sequence of pairwise distinct tokens, the self-attention matrix `P(t)`
converges as `t → +∞` to a matrix of `𝒫`.

Not proved here.

Source: arXiv:2305.05465v6, `t:boolean`. -/
theorem boolean_tendsto_isBooleanLimit (Q K V : ParamMatrix 1) (hV : IsPosDefOp V)
    (hQK : IsPosDefQK Q K) (X : ℝ → Idx n → EucSpace 1) (hX : TransformerDynamics Q K V X)
    (hdist : ∀ i j : Idx n, i ≠ j → X 0 i ≠ X 0 j) :
    ∃ P : Idx n → Idx n → ℝ, IsBooleanLimit P ∧
      ∀ i j : Idx n,
        Tendsto (fun t => attentionMatrix Q K (X t) i j) atTop (nhds (P i j)) := by
  sorry

/-- The hypotheses of `boolean_tendsto_isBooleanLimit` are satisfiable: with a
single token, `Q = K = V = I_1` and the token pinned at the origin, the
distinctness condition is vacuous and the constant curve solves
`eq:trans_dyn` because `V 0 = 0`. -/
example :
    IsPosDefOp (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
      IsPosDefQK (ContinuousLinearMap.id ℝ (EucSpace 1))
        (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
      TransformerDynamics (n := 1) (ContinuousLinearMap.id ℝ (EucSpace 1))
        (ContinuousLinearMap.id ℝ (EucSpace 1)) (ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ _ => 0) ∧
      ∀ i j : Idx 1, i ≠ j → (0 : EucSpace 1) ≠ 0 := by
  refine ⟨isPosDefOp_id 1, isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1), ?_,
    fun i j hij => absurd (Subsingleton.elim i j) hij⟩
  intro t i
  simpa using hasDerivAt_const t (0 : EucSpace 1)

/-- **The genericity claim after `t:boolean`.**  Outside a Lebesgue-null set
of initial sequences, the limit matrix has every row equal to `e_a` or `e_b`,
hence rank at most two.

Not proved here.

Source: arXiv:2305.05465v6, §2, the paragraph after `t:boolean`. -/
theorem boolean_tendsto_isBooleanRows_ae (Q K V : ParamMatrix 1) (hV : IsPosDefOp V)
    (hQK : IsPosDefQK Q K) :
    ∃ N : Set (Idx n → ℝ), volume N = 0 ∧
      ∀ X : ℝ → Idx n → EucSpace 1, TransformerDynamics Q K V X →
        (∀ i j : Idx n, i ≠ j → X 0 i ≠ X 0 j) → (fun i => X 0 i 0) ∉ N →
          ∃ P : Idx n → Idx n → ℝ, IsBooleanRows P ∧
            ∀ i j : Idx n,
              Tendsto (fun t => attentionMatrix Q K (X t) i j) atTop (nhds (P i j)) := by
  sorry

/-- The hypotheses of `boolean_tendsto_isBooleanRows_ae` are satisfiable at
`Q = K = V = I_1`. -/
example :
    IsPosDefOp (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
      IsPosDefQK (ContinuousLinearMap.id ℝ (EucSpace 1))
        (ContinuousLinearMap.id ℝ (EucSpace 1)) :=
  ⟨isPosDefOp_id 1, isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)⟩

end Clusters
end Transformer
