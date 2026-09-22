/-
# The emergence of clusters in self-attention dynamics — bounded particles

§7 of arXiv:2305.05465v6, `s:bounded`: the tokens that stay in `L^∞` are at
most one, and the row of the self-attention matrix they carry is the only row
that need not become a standard basis row.

**What the source says and what is carried here.**

* `l:onlyone` is proved in `Transformer.Clusters.Section7_OnlyOne`, as
  `ncard_boundedTokens_le_one`.

* `l:boundedxn` is proved in `Transformer.Clusters.Section7_BoundedXN`, with
  the range of its second sentence corrected.

* `l:boundedother` is proved in `Transformer.Clusters.Section7_BoundedOther`,
  for every `j ∈ [n]`.  Stated over `j ∈ [n-1]`, the remaining entry would
  converge by the row sum — that is `tendsto_attention_of_tendsto_others` —
  and the limit row is a probability vector, which is the last sentence of the
  proof of `t:boolean` (`tendsto_row_isProbability`).

Source: arXiv:2305.05465v6, the proof of `t:boolean` in §7.
-/

import Transformer.Clusters.Section7_OnlyOne

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n m : ℕ}

/-! ### The row of `l:boundedother` is a probability vector -/

/-- **The remaining entry of a converging row.**  If every entry of the `i`-th
row of `eq:P` but the `a`-th converges, so does the `a`-th, to what the row sum
leaves.  This is what takes `l:boundedother`, stated over `j ∈ [n-1]`, to the
whole row.

Source: arXiv:2305.05465v6, the proof of `t:boolean` in §7. -/
theorem tendsto_attention_of_tendsto_others (hn : 0 < n) (Q K : ParamMatrix d)
    (X : ℝ → Idx n → EucSpace d) (i a : Idx n) (α : Idx n → ℝ)
    (hα : ∀ j : Idx n, j ≠ a → Tendsto (fun t => attentionMatrix Q K (X t) i j) atTop
      (nhds (α j))) :
    Tendsto (fun t => attentionMatrix Q K (X t) i a) atTop
      (nhds (1 - ∑ j ∈ Finset.univ.erase a, α j)) := by
  have hsum : ∀ t : ℝ, attentionMatrix Q K (X t) i a
      = 1 - ∑ j ∈ Finset.univ.erase a, attentionMatrix Q K (X t) i j := by
    intro t
    have := Finset.add_sum_erase Finset.univ (fun j => attentionMatrix Q K (X t) i j)
      (Finset.mem_univ a)
    rw [sum_attentionMatrix hn] at this
    linarith
  refine Tendsto.congr (fun t => (hsum t).symm) ?_
  exact tendsto_const_nhds.sub
    (tendsto_finsetSum _ fun j hj => hα j (Finset.ne_of_mem_erase hj))

/-- **The limit row is a probability vector.**  The last sentence of the proof
of `t:boolean`: "since the `i₀`-th row of `P(t)` has entries which sum up to
`1`, then so does `α`".

Source: arXiv:2305.05465v6, the proof of `t:boolean` in §7. -/
theorem tendsto_row_isProbability (hn : 0 < n) (Q K : ParamMatrix d)
    (X : ℝ → Idx n → EucSpace d) (i : Idx n) (α : Idx n → ℝ)
    (hα : ∀ j : Idx n, Tendsto (fun t => attentionMatrix Q K (X t) i j) atTop (nhds (α j))) :
    (∀ j : Idx n, 0 ≤ α j) ∧ ∑ j : Idx n, α j = 1 := by
  refine ⟨fun j => ge_of_tendsto' (hα j) fun t => attentionMatrix_nonneg Q K (X t) i j, ?_⟩
  refine tendsto_nhds_unique (tendsto_finsetSum _ fun j _ => hα j) ?_
  simp [sum_attentionMatrix hn]

/-- The hypotheses of `tendsto_attention_of_tendsto_others` and of
`tendsto_row_isProbability` are satisfiable: along a configuration that does
not move, every entry of `eq:P` is constant, hence convergent.  The whole
family is exhibited, which in particular gives the subfamily `j ≠ a` that
`tendsto_attention_of_tendsto_others` asks for. -/
example (Q K : ParamMatrix d) :
    0 < 1 ∧ ∀ j : Idx 1, Tendsto
      (fun t : ℝ => attentionMatrix Q K ((fun _ _ => (0 : EucSpace d)) t) (0 : Idx 1) j)
      atTop (nhds (attentionMatrix Q K (fun _ => (0 : EucSpace d)) (0 : Idx 1) j)) :=
  ⟨one_pos, fun _ => tendsto_const_nhds⟩

end Clusters
end Transformer
