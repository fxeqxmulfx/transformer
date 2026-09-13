/-
# Where the ties are, in every key dimension

`Transformer.ALM.TieBreak` locates the ties of the planar machine: two
distinct scalar integer keys score equally at the single query
`2q = k₁ + k₂`, so the merge path of `HullHalf::query`
(`transformer_vm/attention/hull2d_cht.h`, lines 252-310) is a set of measure
zero in the query, and is unreachable altogether when the query is one of the
stored keys.

In dimension `m` the same computation gives a *hyperplane* rather than a
point.  The tie locus of `k₁ ≠ k₂` is

  `⟪k₁ - k₂, 2q - (k₁ + k₂)⟫ = 0`,

the perpendicular bisector of the two keys — equivalently, the queries
equidistant from both.  That is what `score_eq_iff_inner_eq_zero` and
`score_eq_iff_norm_eq` say.  Two consequences for the machine:

* the lookup path is still safe — `tie_eq_of_query_mem` reproves
  `argmax_unique_of_query_mem` in every dimension, so `resolve` never sees a
  merged line when the query is a stored key;
* off that path the tie set *grows with the dimension*.  At `m = 1` it is one
  query, and `score_eq_iff_int_one` recovers `sScore_eq_iff` exactly; at
  `m = 2` it is already an unbounded line, as the final example exhibits.

The scores are `Transformer.ALM.score` of `Transformer.ALM.Defs`; the scalar
case is `Transformer.ALM.TieBreak`.
-/

import Transformer.ALM.TieBreak
import Transformer.ALM.VectorInt

open scoped BigOperators

namespace Transformer
namespace ALM

variable {m n : ℕ}

/-! ### The tie locus is the perpendicular bisector -/

/-- **Ties are equidistance.**  Two keys score equally at `q` exactly when `q`
is as far from one as from the other; by `score_gap` the score is `‖q‖²` minus
the squared distance. -/
theorem score_eq_iff_norm_eq (q k₁ k₂ : EucSpace m) :
    score q k₁ = score q k₂ ↔ ‖k₁ - q‖ = ‖k₂ - q‖ := by
  have h₁ := score_gap q k₁
  have h₂ := score_gap q k₂
  rw [← sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)]
  constructor
  · intro h; linarith
  · intro h; linarith

/-- **The tie locus is a hyperplane.**  The queries at which `k₁` and `k₂`
score equally are those on the perpendicular bisector, with normal `k₁ - k₂`
through the midpoint. -/
theorem score_eq_iff_inner_eq_zero (q k₁ k₂ : EucSpace m) :
    score q k₁ = score q k₂ ↔
      inner (𝕜 := ℝ) (k₁ - k₂) ((2 : ℝ) • q - (k₁ + k₂)) = 0 := by
  have hexp : inner (𝕜 := ℝ) (k₁ - k₂) ((2 : ℝ) • q - (k₁ + k₂))
      = score q k₁ - score q k₂ := by
    unfold score
    simp only [inner_sub_left, inner_sub_right, inner_add_right, real_inner_smul_right,
      real_inner_self_eq_norm_sq]
    rw [real_inner_comm k₂ k₁]
    ring
  rw [hexp, sub_eq_zero, eq_comm]

/-- The locus is never empty: the midpoint of the two keys is equidistant from
both, so `HullHalf::query` really can reach its merge path. -/
theorem score_midpoint_eq (k₁ k₂ : EucSpace m) :
    score ((2 : ℝ)⁻¹ • (k₁ + k₂)) k₁ = score ((2 : ℝ)⁻¹ • (k₁ + k₂)) k₂ := by
  rw [score_eq_iff_inner_eq_zero]
  rw [smul_smul]
  norm_num

/-- And it is never everything: distinct keys are separated at the query that
is one of them.  So the tie locus of `k₁ ≠ k₂` is a proper, nonempty affine
subspace — a hyperplane, not a degenerate case. -/
theorem exists_score_ne (k₁ k₂ : EucSpace m) (hne : k₁ ≠ k₂) :
    ∃ q : EucSpace m, score q k₁ ≠ score q k₂ :=
  ⟨k₁, fun h => hne (by
    have hnorm := (score_eq_iff_norm_eq k₁ k₁ k₂).mp h
    rw [sub_self, norm_zero] at hnorm
    have hz : k₂ - k₁ = 0 := by
      rw [← norm_eq_zero (E := EucSpace m)]
      exact hnorm.symm
    exact (sub_eq_zero.mp hz).symm)⟩

/-! ### The lookup path stays free of ties -/

/-- **The merge path is dead on the lookup path, in every dimension.**  If the
query is one of the stored keys, every key achieving the best score *is* that
query, so the scan merges nothing and `resolve` sees a single value.  This is
`argmax_unique_of_query_mem` of `Transformer.ALM.TieBreak` without the
restriction to scalar integer keys. -/
theorem tie_eq_of_query_mem (K : Fin n → EucSpace m) (q : EucSpace m) (i₀ : Fin n)
    (h₀ : K i₀ = q) (j : Fin n) (hj : score q (K j) = score q (K i₀)) :
    K j = q := by
  by_contra hne
  have hlt : score q (K j) < score q q := score_lt_of_ne hne
  rw [hj, h₀] at hlt
  exact lt_irrefl _ hlt

/-- The hypotheses are satisfiable: a key family containing the query. -/
example : ∃ (K : Fin 2 → EucSpace 3) (q : EucSpace 3) (i₀ j : Fin 2),
    K i₀ = q ∧ score q (K j) = score q (K i₀) :=
  ⟨fun _ => 0, 0, 0, 1, rfl, rfl⟩

/-! ### Dimension one is the published case -/

/-- **The scalar tie locus, recovered.**  On one-dimensional integer keys the
hyperplane degenerates to the single query `2q = k₁ + k₂`, which is
`sScore_eq_iff` of `Transformer.ALM.TieBreak`. -/
theorem score_eq_iff_int_one (q k₁ k₂ : Fin 1 → ℤ) (hne : k₁ 0 ≠ k₂ 0) :
    score (embInt q) (embInt k₁) = score (embInt q) (embInt k₂) ↔
      2 * q 0 = k₁ 0 + k₂ 0 := by
  rw [score_embInt_one, score_embInt_one]
  exact sScore_eq_iff (q 0) (k₁ 0) (k₂ 0) hne

/-- The hypothesis is satisfiable: two distinct scalar keys. -/
example : (fun _ : Fin 1 => (0 : ℤ)) 0 ≠ (fun _ : Fin 1 => (2 : ℤ)) 0 := by decide

/-! ### Dimension two is already unbounded -/

/-- **The tie set grows with the dimension.**  The keys `(0,0)` and `(2,0)`
tie along the whole line `q = (1, t)`: in the plane the merge path of
`HullHalf::query` is reachable from arbitrarily far away, where in dimension
one it was reachable from a single query. -/
example (t : ℝ) :
    score (WithLp.toLp 2 ![(1 : ℝ), t]) (WithLp.toLp 2 ![(0 : ℝ), 0])
      = score (WithLp.toLp 2 ![(1 : ℝ), t]) (WithLp.toLp 2 ![(2 : ℝ), 0]) := by
  unfold score
  simp [PiLp.inner_apply, RCLike.inner_apply, norm_sq_eq_sum, Fin.sum_univ_two]
  norm_num

end ALM
end Transformer
