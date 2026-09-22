/-
# The emergence of clusters in self-attention dynamics — `t:boolean`

§2 of arXiv:2305.05465v6 states `t:boolean`; §7 proves it.  The set `𝒫` and
the deviations from the source's statement are in
`Transformer.Clusters.Section2_LowRank`; the reduction to the ordered case of
`e:Idnonresca` is in `Section7_BooleanScaling`, and the ordered case in
`Section7_Boolean`.

Source: arXiv:2305.05465v6, `t:boolean`; the proof in §7.
-/

import Transformer.Clusters.Section7_Boolean
import Transformer.Clusters.Section7_BooleanScaling
import Mathlib.Data.Fin.Tuple.Sort

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {n : ℕ}

/-- **Theorem (t:boolean).**  Let `d = 1`, `V > 0` and `QK > 0`.  For any
initial sequence of pairwise distinct tokens, the self-attention matrix `P(t)`
converges as `t → +∞` to a matrix of `𝒫`.

The source leaves `n ≥ 1` implicit; at `n = 0` there is no index, so no
matrix is in `𝒫` and the statement is false as written.  `0 < n` is added.

The proof is the source's: time is reparametrized so that `V = 1`, the tokens
are relabelled so that they are increasing, and the factor `QK` is absorbed
into the tokens (`Section7_BooleanScaling`), which leaves the ordered case of
`e:Idnonresca` (`exists_isBooleanLimit_of_ordered`).

Source: arXiv:2305.05465v6, `t:boolean`; the proof in §7. -/
theorem boolean_tendsto_isBooleanLimit (Q K V : ParamMatrix 1) (hV : IsPosDefOp V)
    (hQK : IsPosDefQK Q K) (X : ℝ → Idx n → EucSpace 1) (hX : TransformerDynamics Q K V X)
    (hdist : ∀ i j : Idx n, i ≠ j → X 0 i ≠ X 0 j) (hn : 0 < n) :
    ∃ P : Idx n → Idx n → ℝ, IsBooleanLimit P ∧
      ∀ i j : Idx n,
        Tendsto (fun t => attentionMatrix Q K (X t) i j) atTop (nhds (P i j)) := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have hunit : unit1 ≠ 0 := fun h => by simpa using congrArg norm h
  have hv : 0 < (V unit1) 0 := by
    have := hV unit1 hunit
    rw [eq_smul_unit1 (V unit1), real_inner_smul_left, real_inner_self_eq_norm_sq,
      norm_unit1] at this
    simpa using this
  have hc : 0 < inner (𝕜 := ℝ) (Q unit1) (K unit1) := hQK.2 unit1 hunit
  set f : Idx (m + 1) → ℝ := fun i => X 0 i 0
  have hinj : Function.Injective f := by
    intro i j hij
    by_contra hne
    refine hdist i j hne ?_
    rw [eq_smul_unit1 (X 0 i), eq_smul_unit1 (X 0 j)]
    exact congrArg (· • unit1) hij
  set σ := Tuple.sort f
  have hmono : StrictMono (f ∘ σ) :=
    (Tuple.monotone_sort f).strictMono_of_injective (hinj.comp σ.injective)
  have hY := idNonrescaledDynamics_rescale hX hv hc σ
  have hord : IsOrderedConfig (fun i =>
      (√(inner (𝕜 := ℝ) (Q unit1) (K unit1)) * X (0 / (V unit1) 0) (σ i) 0) • unit1) := by
    intro i j hij
    simp only [coord_smul_unit1, zero_div]
    exact mul_lt_mul_of_pos_left (hmono hij) (Real.sqrt_pos.2 hc)
  obtain ⟨PY, hPY, hconv⟩ := exists_isBooleanLimit_of_ordered _ hY hord
  refine ⟨fun i j => PY (σ.symm i) (σ.symm j), isBooleanLimit_comp_perm σ.symm hPY,
    fun i j => ?_⟩
  refine ((hconv (σ.symm i) (σ.symm j)).comp (tendsto_id.const_mul_atTop hv)).congr
    fun t => ?_
  simp only [Function.comp_apply, id]
  exact (attentionMatrix_rescale Q K X hv hc σ t (σ.symm i) (σ.symm j)).trans
    (by rw [Equiv.apply_symm_apply, Equiv.apply_symm_apply])

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
      (∀ i j : Idx 1, i ≠ j → (0 : EucSpace 1) ≠ 0) ∧ 0 < 1 := by
  refine ⟨isPosDefOp_id 1, isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1), ?_,
    fun i j hij => absurd (Subsingleton.elim i j) hij, one_pos⟩
  intro t i
  simpa using hasDerivAt_const t (0 : EucSpace 1)

end Clusters
end Transformer
