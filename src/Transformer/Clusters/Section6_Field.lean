/-
# The emergence of clusters in self-attention dynamics — the vector field of
  `eq:trans_dyn`

§6 of arXiv:2305.05465v6 proves `p:wellposedparticles` from two properties of
the right-hand side `F(x)_i = Σ_j P_ij(x) V x_j` of `eq:trans_dyn`, seen as a
vector field on `(ℝ^d)^n`: it is smooth, and it grows at most linearly,
`‖F(x)‖ ≤ ‖V‖ ‖x‖`, because each `F(x)_i` is `V` applied to a convex
combination of the tokens.  This file proves both, and that a solution of
`eq:trans_dyn` is a solution of the ODE `x' = F(x)` on `(ℝ^d)^n` and back.

Source: arXiv:2305.05465v6, §6, `p:wellposedparticles`.
-/

import Transformer.Clusters.Section3_Rescaled
import Transformer.GlobalFlow.Existence
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.ContDiff.RCLike

open scoped BigOperators NNReal
open Set

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- The right-hand side of `eq:trans_dyn`, as a vector field on `(ℝ^d)^n`. -/
noncomputable def transformerField (Q K V : ParamMatrix d) (X : Idx n → EucSpace d) :
    Idx n → EucSpace d :=
  fun i => ∑ j, attentionMatrix Q K X i j • V (X j)

/-- **`eq:trans_dyn` is the ODE `x' = F(x)` on `(ℝ^d)^n`.** -/
theorem transformerDynamics_iff_hasDerivAt (Q K V : ParamMatrix d) (X : ℝ → Idx n → EucSpace d) :
    TransformerDynamics Q K V X ↔ ∀ t, HasDerivAt X (transformerField Q K V (X t)) t := by
  simp only [TransformerDynamics, hasDerivAt_pi (φ := X), transformerField]

/-- Each entry of the self-attention matrix is a smooth function of the tokens. -/
theorem contDiff_attentionMatrix (Q K : ParamMatrix d) (i j : Idx n) :
    ContDiff ℝ 1 fun X : Idx n → EucSpace d => attentionMatrix Q K X i j := by
  have hs : ∀ l, ContDiff ℝ 1 fun X : Idx n → EucSpace d =>
      Real.exp (inner (𝕜 := ℝ) (Q (X i)) (K (X l))) := fun l =>
    (ContDiff.inner ℝ (Q.contDiff.comp (contDiff_apply ℝ (EucSpace d) i))
      (K.contDiff.comp (contDiff_apply ℝ (EucSpace d) l))).exp
  unfold attentionMatrix Perspective.softmaxWeight
  exact (hs j).div (ContDiff.sum fun l _ => hs l) fun X =>
    (Finset.sum_pos (fun l _ => Real.exp_pos _) ⟨i, Finset.mem_univ _⟩).ne'

/-- **The field of `eq:trans_dyn` is smooth**, hence locally Lipschitz. -/
theorem contDiff_transformerField (Q K V : ParamMatrix d) :
    ContDiff ℝ 1 (transformerField (n := n) Q K V) :=
  contDiff_pi.2 fun i => ContDiff.sum fun j _ =>
    (contDiff_attentionMatrix Q K i j).smul (V.contDiff.comp (contDiff_apply ℝ (EucSpace d) j))

/-- **The field of `eq:trans_dyn` grows linearly**: `‖F(x)‖ ≤ ‖V‖ ‖x‖`, each
`F(x)_i` being `V` applied to a convex combination of the tokens. -/
theorem norm_transformerField_le (Q K V : ParamMatrix d) (X : Idx n → EucSpace d) :
    ‖transformerField Q K V X‖ ≤ ‖V‖ * ‖X‖ := by
  refine (pi_norm_le_iff_of_nonneg (by positivity)).2 fun i => ?_
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  calc ‖∑ j, attentionMatrix Q K X i j • V (X j)‖
      ≤ ∑ j, attentionMatrix Q K X i j * (‖V‖ * ‖X‖) := by
        refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun j _ => ?_)
        rw [norm_smul, Real.norm_of_nonneg (attentionMatrix_nonneg Q K X i j)]
        exact mul_le_mul_of_nonneg_left
          ((V.le_opNorm _).trans (mul_le_mul_of_nonneg_left (norm_le_pi_norm X j) (norm_nonneg _)))
          (attentionMatrix_nonneg Q K X i j)
    _ = ‖V‖ * ‖X‖ := by rw [← Finset.sum_mul, sum_attentionMatrix hn, one_mul]

/-- A curve with a continuous derivative is Lipschitz on every compact
interval. -/
theorem lipschitzOnWith_Icc_of_hasDerivAt {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {f f' : ℝ → E} (hf : ∀ t, HasDerivAt f (f' t) t) (hf' : Continuous f') (a b : ℝ) :
    ∃ L : ℝ≥0, LipschitzOnWith L f (Icc a b) := by
  obtain ⟨C, hC⟩ := isCompact_Icc.exists_bound_of_continuousOn (hf'.continuousOn (s := Icc a b))
  refine ⟨C.toNNReal, (convex_Icc a b).lipschitzOnWith_of_nnnorm_hasDerivWithin_le
    (fun t _ => (hf t).hasDerivWithinAt) fun t ht => ?_⟩
  rw [← NNReal.coe_le_coe, coe_nnnorm, Real.coe_toNNReal']
  exact (hC t ht).trans (le_max_left _ _)

/-- The hypotheses of `lipschitzOnWith_Icc_of_hasDerivAt` are satisfiable: the
identity of `ℝ`, with derivative `1`. -/
example : (∀ t : ℝ, HasDerivAt (fun s : ℝ => s) (1 : ℝ) t) ∧ Continuous fun _ : ℝ => (1 : ℝ) :=
  ⟨hasDerivAt_id, continuous_const⟩

end Clusters
end Transformer
