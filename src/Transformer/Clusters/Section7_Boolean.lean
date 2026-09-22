/-
# The emergence of clusters in self-attention dynamics — `t:boolean`, the ordered case

§7 of arXiv:2305.05465v6, "Concluding the proof of Theorem `t:boolean`": for
the ordered solutions of `e:Idnonresca` in `d = 1`, every entry of `P(t)`
converges and the limit matrix lies in `𝒫`.

The proof is the source's.  A token that is not bounded has its row tend to
`e_1` or `e_n` (`l:unboundedparticles`); at most one token is bounded
(`l:onlyone`), and its row converges by `l:boundedxn` or `l:boundedother`; the
limit of that row is a probability vector because every row of `P(t)` is.

The source adds that the rows equal to `e_n` come after those equal to `e_1`;
`𝒫` as carried in `IsBooleanLimit` does not ask for it, and it is not proved.

Source: arXiv:2305.05465v6, proof of `t:boolean`, §7.
-/

import Transformer.Clusters.Section7_BoundedOther
import Transformer.Clusters.Section2_LowRank

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- The doubly exponential bound of `l:unboundedparticles` gives convergence. -/
theorem tendsto_of_abs_sub_le_exp {f : ℝ → ℝ} {L c : ℝ} (hc : 0 < c)
    (h : ∀ᶠ t in atTop, |f t - L| ≤ Real.exp (-(c * Real.exp t))) :
    Tendsto f atTop (𝓝 L) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  have h0 : Tendsto (fun t => Real.exp (-(c * Real.exp t))) atTop (𝓝 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero.comp (tendsto_exp_atTop.const_mul_atTop hc)
  exact squeeze_zero_norm' (by simpa only [Real.norm_eq_abs, abs_abs] using h) h0

/-- **Every row converges.**  The row of an unbounded token tends to `e_1` or
`e_n` (`l:unboundedparticles`); the row of a bounded token converges by
`l:boundedxn` or `l:boundedother`.

Source: arXiv:2305.05465v6, proof of `t:boolean`, §7. -/
theorem exists_row_limit (X : ℝ → Idx (m + 1) → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (hord : IsOrderedConfig (X 0)) (i : Idx (m + 1)) :
    ∃ α : Idx (m + 1) → ℝ,
      (∀ j, Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) i j) atTop
        (𝓝 (α j))) ∧
      (¬ IsBoundedToken X i → α = Pi.single 0 1 ∨ α = Pi.single (Fin.last m) 1) := by
  by_cases hb : IsBoundedToken X i
  · by_cases h0 : i = 0
    · subst h0
      obtain ⟨h1, h2⟩ := tendsto_attention_of_bounded_first X hX hord hb
      refine ⟨Pi.single 0 1, fun j => ?_, fun h => absurd hb h⟩
      rcases eq_or_ne j 0 with rfl | hj
      · simpa using h1
      · simpa [Pi.single_apply, hj] using h2 j hj
    by_cases hl : i = Fin.last m
    · subst hl
      obtain ⟨h1, h2⟩ := tendsto_attention_of_bounded_last X hX hord hb
      refine ⟨Pi.single (Fin.last m) 1, fun j => ?_, fun h => absurd hb h⟩
      rcases eq_or_ne j (Fin.last m) with rfl | hj
      · simpa using h1
      · simpa [Pi.single_apply, hj] using h2 j hj
    choose α _ hα using exists_tendsto_attention_of_bounded_interior X hX hord i h0 hl hb
    exact ⟨α, hα, fun h => absurd hb h⟩
  · rcases unbounded_tendsto_atTop_or_atBot X hX hord i hb with ⟨-, c, hc, h⟩ | ⟨-, c, hc, h⟩
    · refine ⟨Pi.single (Fin.last m) 1, fun j => ?_, fun _ => Or.inr rfl⟩
      exact tendsto_of_abs_sub_le_exp hc (h.mono fun t ht => by
        simpa only [Pi.single_apply] using ht j)
    · refine ⟨Pi.single 0 1, fun j => ?_, fun _ => Or.inl rfl⟩
      exact tendsto_of_abs_sub_le_exp hc (h.mono fun t ht => by
        simpa only [Pi.single_apply] using ht j)

/-- **Theorem (t:boolean), ordered case of `e:Idnonresca`.**  For an ordered
solution of `e:Idnonresca` in `d = 1`, `P(t)` converges entrywise to a matrix
of `𝒫`.

Source: arXiv:2305.05465v6, proof of `t:boolean`, §7. -/
theorem exists_isBooleanLimit_of_ordered (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) :
    ∃ P : Idx (m + 1) → Idx (m + 1) → ℝ, IsBooleanLimit P ∧
      ∀ i j, Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) i j) atTop
        (𝓝 (P i j)) := by
  choose P hP hPb using exists_row_limit X hX hord
  refine ⟨P, ?_, hP⟩
  have hunb : ∃ i₀, ∀ i, i ≠ i₀ → ¬ IsBoundedToken X i := by
    by_cases h : ∃ i₀, IsBoundedToken X i₀
    · obtain ⟨i₀, hi₀⟩ := h
      exact ⟨i₀, fun i hi => not_isBoundedToken_of_ne X hX hord hi₀ hi⟩
    · exact ⟨0, fun i _ hi => h ⟨i, hi⟩⟩
  obtain ⟨i₀, hi₀⟩ := hunb
  exact ⟨0, Fin.last m, i₀, fun i hi => hPb i (hi₀ i hi),
    (tendsto_row_isProbability (Nat.succ_pos m) 1 1 X i₀ (P i₀) (hP i₀)).1,
    (tendsto_row_isProbability (Nat.succ_pos m) 1 1 X i₀ (P i₀) (hP i₀)).2⟩

/-- The hypotheses of `exists_isBooleanLimit_of_ordered` are satisfiable: the
one-token solution `x(t) = e^t`. -/
example :
    IdNonrescaledDynamics (n := 1)
        (fun t _ => Real.exp t • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) ∧
      IsOrderedConfig (n := 1)
        (fun _ => Real.exp 0 • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) :=
  ⟨idNonrescaledDynamics_single _, isOrderedConfig_subsingleton _⟩

end Clusters
end Transformer
