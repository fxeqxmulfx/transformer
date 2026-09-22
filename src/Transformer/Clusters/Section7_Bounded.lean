/-
# The emergence of clusters in self-attention dynamics — bounded particles

§7 of arXiv:2305.05465v6, `s:bounded`: the tokens that stay in `L^∞` are at
most one, and the row of the self-attention matrix they carry is the only row
that need not become a standard basis row.

**What the source says and what is carried here.**

* `l:onlyone` is proved in `Transformer.Clusters.Section7_OnlyOne`, as
  `ncard_boundedTokens_le_one`.

* `l:boundedxn`'s second sentence says "if `x_1(t)` is bounded then
  `P_11(t) → 1` and `P_{1j}(t) → 0` for any `j ∈ [n-1]`".  As written it is
  false: `1 ∈ [n-1]` whenever `n ≥ 2`, so it asks for `P_11(t) → 0` and
  `P_11(t) → 1` at once.  The range is a copy of the first sentence's, where
  `[n-1] = [n] \ {n}` is correct; symmetrically it must be `[n] \ {1}`, and
  that is how it is carried.  Nothing else is changed.

* `l:boundedother` is carried over `j ∈ [n-1]`, as stated.  The remaining
  entry of the row then converges too, by the row sum — that is
  `tendsto_attention_of_tendsto_others` — and the limit row is again a
  probability vector, which is the last sentence of the proof of `t:boolean`;
  both are proved here, with the convergence of the other entries taken as an
  explicit hypothesis.

* The hypotheses of `l:boundedother` need three ordered tokens, one of them
  interior and bounded.  `Transformer.Clusters.Section7_Symmetric` supplies
  exactly that configuration, from a solution of the scalar equation
  `symDrift` it reduces to.

Source: arXiv:2305.05465v6, `l:boundedxn`, `l:boundedother`, and
the proof of `t:boolean` in §7.
-/

import Transformer.Clusters.Section7_OnlyOne

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n m : ℕ}

/-- The configuration pinned at the origin solves `e:Idnonresca`, and with one
token it is ordered and bounded; this witnesses the hypotheses of
both halves of `l:boundedxn`. -/
theorem isBoundedToken_zero (i : Idx n) :
    IsBoundedToken (fun _ _ => (0 : EucSpace 1)) i :=
  ⟨0, fun _ _ => by simp⟩

/-! ### `l:boundedxn` -/

/-- **Lemma (l:boundedxn), the largest token.**  If `x_n(t)` stays bounded
then `P_nn(t) → 1` and `P_nj(t) → 0` for every `j ∈ [n-1]`.

Not proved here.

Source: arXiv:2305.05465v6, `l:boundedxn`. -/
theorem tendsto_attention_of_bounded_last (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0))
    (hb : IsBoundedToken X (Fin.last m)) :
    Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) (Fin.last m) (Fin.last m))
        atTop (nhds 1) ∧
      ∀ j : Idx (m + 1), j ≠ Fin.last m →
        Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) (Fin.last m) j)
          atTop (nhds 0) := by
  sorry

/-- **Lemma (l:boundedxn), the smallest token.**  If `x_1(t)` stays bounded
then `P_11(t) → 1` and `P_{1j}(t) → 0` for every `j ∈ [n] \ {1}`.

The source writes `j ∈ [n-1]` here, copying the range of its first sentence;
that range contains `j = 1` and would contradict `P_11(t) → 1`.  The
symmetric range is `[n] \ {1}`, and that is what is stated.

Not proved here.

Source: arXiv:2305.05465v6, `l:boundedxn`, second sentence, corrected. -/
theorem tendsto_attention_of_bounded_first (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0))
    (hb : IsBoundedToken X 0) :
    Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) 0 0) atTop (nhds 1) ∧
      ∀ j : Idx (m + 1), j ≠ 0 →
        Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) 0 j) atTop (nhds 0) := by
  sorry

/-- The hypotheses of both halves of `l:boundedxn` are satisfiable. -/
example :
    IdNonrescaledDynamics (n := 1) (fun _ _ => (0 : EucSpace 1)) ∧
      IsOrderedConfig (n := 1) (fun _ => (0 : EucSpace 1)) ∧
      IsBoundedToken (fun _ _ => (0 : EucSpace 1)) (Fin.last 0) ∧
      IsBoundedToken (fun _ _ => (0 : EucSpace 1)) (0 : Idx 1) :=
  ⟨idNonrescaledDynamics_zero 1 1, isOrderedConfig_subsingleton _,
    isBoundedToken_zero _, isBoundedToken_zero _⟩

/-! ### `l:boundedother` -/

/-- **Lemma (l:boundedother).**  If the bounded token is interior,
`i₀ ∉ {1,n}`, then each entry `P_{i₀ j}(t)`, `j ∈ [n-1]`, converges to some
`α_j ∈ [0,1]`.

Not proved here.

Source: arXiv:2305.05465v6, `l:boundedother`. -/
theorem exists_tendsto_attention_of_bounded_interior (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) (i₀ : Idx (m + 1))
    (hfirst : i₀ ≠ 0) (hlast : i₀ ≠ Fin.last m) (hb : IsBoundedToken X i₀)
    (j : Idx (m + 1)) (hj : j ≠ Fin.last m) :
    ∃ α ∈ Set.Icc (0 : ℝ) 1,
      Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) i₀ j) atTop (nhds α) := by
  sorry

/-- The hypotheses of `exists_tendsto_attention_of_bounded_interior` are
satisfiable: the symmetric triple `(-u, 0, u)` of
`Transformer.Clusters.Section7_Symmetric` is ordered as soon as `u(0) > 0`,
and its interior token sits at the origin for all time, so it is the bounded
one.  The solution `u` of the scalar equation is the hypothesis; no solution
of `e:Idnonresca` with three distinct tokens is available in closed form. -/
example (u : ℝ → ℝ) (hu : ∀ t : ℝ, HasDerivAt u (symDrift (u t)) t) (hu0 : 0 < u 0) :
    IdNonrescaledDynamics (fun t => symTriple (u t)) ∧
      IsOrderedConfig (symTriple (u 0)) ∧
      (1 : Idx 3) ≠ 0 ∧ (1 : Idx 3) ≠ Fin.last 2 ∧
      IsBoundedToken (fun t => symTriple (u t)) 1 ∧ (0 : Idx 3) ≠ Fin.last 2 :=
  ⟨idNonrescaledDynamics_symTriple u hu, isOrderedConfig_symTriple hu0,
    symInterior_ne_first, symInterior_ne_last, isBoundedToken_symTriple u, by decide⟩

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
