/-
# The emergence of clusters in self-attention dynamics — `l:boundedother`

§7 of arXiv:2305.05465v6: if the bounded token is interior, its row of the
self-attention matrix converges.

The proof is the source's.  Every other token is unbounded (`l:onlyone`), so
`x_k e^{-t} → γ_k ≠ 0` (`l:exactasymptotic`), with `γ_n > 0` and `γ_1 < 0`
since `x_1 ≤ x_{i₀} ≤ x_n`; the bounded token has `x_{i₀} e^{-t} → 0`.  Then
`x_{i₀} e^t → θ₀` (`tendsto_rescaled`), so every score
`x_{i₀} x_k = (x_{i₀} e^t)(x_k e^{-t})` converges to `θ₀ γ_k`, and the row
converges to the softmax of these limits.

**What the source says and what is carried here.**  The source states the
convergence for `j ∈ [n-1]` only.  Its proof gives every `j ∈ [n]`, and that
is what is stated: the case `j = n` is no longer left to the row sum.

Source: arXiv:2305.05465v6, `l:boundedother`.
-/

import Transformer.Clusters.Section7_BoundedXN
import Transformer.Clusters.Section7_BoundedOtherEscape
import Transformer.Clusters.Section7_ExactAsymptotic

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- A bounded token is `o(e^t)`: `x_i e^{-t} → 0`. -/
theorem tendsto_div_exp_of_bounded (X : ℝ → Idx (m + 1) → EucSpace 1) {i : Idx (m + 1)}
    (hb : IsBoundedToken X i) : Tendsto (fun t => X t i 0 / Real.exp t) atTop (𝓝 0) := by
  obtain ⟨R, hR⟩ := hb
  refine squeeze_zero_norm' ?_ ((tendsto_const_nhds (x := R)).div_atTop tendsto_exp_atTop)
  filter_upwards [eventually_ge_atTop 0] with t ht
  rw [Real.norm_eq_abs, abs_div, abs_of_pos (Real.exp_pos t)]
  exact div_le_div_of_nonneg_right (hR t ht) (Real.exp_pos t).le

/-- **Lemma (l:boundedother).**  If the bounded token is interior,
`i₀ ∉ {1,n}`, then each entry `P_{i₀ j}(t)` converges to some `α_j ∈ [0,1]`.

The source states it for `j ∈ [n-1]`; its proof, carried here, gives every
`j ∈ [n]`, and the restriction is dropped.

Source: arXiv:2305.05465v6, `l:boundedother`. -/
theorem exists_tendsto_attention_of_bounded_interior (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) (i₀ : Idx (m + 1))
    (hfirst : i₀ ≠ 0) (hlast : i₀ ≠ Fin.last m) (hb : IsBoundedToken X i₀) (j : Idx (m + 1)) :
    ∃ α ∈ Set.Icc (0 : ℝ) 1,
      Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) i₀ j) atTop (nhds α) := by
  have hle := coord_mono X hX hord
  have hm : 0 < m := by
    rcases Nat.eq_zero_or_pos m with h | h
    · subst h; exact absurd (Fin.ext (by omega)) hfirst
    · exact h
  have hrate : ∀ k, ∃ g : ℝ, Tendsto (fun t => X t k 0 / Real.exp t) atTop (𝓝 g) ∧
      (k ≠ i₀ → g ≠ 0) := by
    intro k
    by_cases hk : k = i₀
    · exact ⟨0, hk ▸ tendsto_div_exp_of_bounded X hb, fun h => absurd hk h⟩
    · obtain ⟨g, hg, hlim⟩ := exists_exp_asymptotic X hX hord k
        (not_isBoundedToken_of_ne X hX hord hb hk)
      exact ⟨g, hlim, fun _ => hg⟩
  choose γ hγ hγ0 using hrate
  obtain ⟨R, hR⟩ := hb
  have hlow : Tendsto (fun t => -R / Real.exp t) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_exp_atTop
  have hp : 0 < γ (Fin.last m) := by
    refine lt_of_le_of_ne (le_of_tendsto_of_tendsto hlow (hγ _) ?_) (hγ0 _ (Ne.symm hlast)).symm
    filter_upwards [eventually_ge_atTop 0] with t ht
    exact div_le_div_of_nonneg_right
      ((abs_le.1 (hR t ht)).1.trans (hle t ht _ _ (Fin.le_last i₀))) (Real.exp_pos t).le
  have hq : γ 0 < 0 := by
    have hup : Tendsto (fun t => R / Real.exp t) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop tendsto_exp_atTop
    refine lt_of_le_of_ne (le_of_tendsto_of_tendsto (hγ _) hup ?_) (hγ0 _ (Ne.symm hfirst))
    filter_upwards [eventually_ge_atTop 0] with t ht
    exact div_le_div_of_nonneg_right
      ((hle t ht _ _ (Fin.zero_le i₀)).trans (abs_le.1 (hR t ht)).2) (Real.exp_pos t).le
  obtain ⟨θ₀, hθ₀⟩ := tendsto_rescaled (fun t k => X t k 0) (hasDerivAt_coord X hX) hm i₀ hR γ hγ
    hp hq
  refine ⟨Perspective.softmaxWeight (fun l => θ₀ * γ l) j,
    ⟨Perspective.softmaxWeight_nonneg _ _, ?_⟩, ?_⟩
  · rw [← Perspective.sum_softmaxWeight (Nat.succ_pos m) (fun l => θ₀ * γ l)]
    exact Finset.single_le_sum (f := Perspective.softmaxWeight (fun l => θ₀ * γ l))
      (fun k _ => Perspective.softmaxWeight_nonneg _ _) (Finset.mem_univ j)
  · simp only [attentionMatrix_one_eq]
    refine (tendsto_softmaxWeight (fun l => hθ₀.mul (hγ l)) j).congr fun t => ?_
    have he := Real.exp_pos t
    congr 1
    funext l
    field_simp

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
      IsBoundedToken (fun t => symTriple (u t)) 1 :=
  ⟨idNonrescaledDynamics_symTriple u hu, isOrderedConfig_symTriple hu0,
    symInterior_ne_first, symInterior_ne_last, isBoundedToken_symTriple u⟩

end Clusters
end Transformer
