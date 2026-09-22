/-
# Homogenized Transformers — the attention average at a token

At a token `x_j` of a configuration `X`, the attention field of a head
`θ = (V, A)` is the value matrix applied to the attention average,

  `B_θ[μ_X](x_j) = V m_j(A)`,  `m_j(A) = Σ_k π^A_{j→k} x_k`

(`attnField_self_eq_valueMap`), which is the softmax barycenter
`m_{β,A}[μ_X](x_j)` of `eq: mbetaA` (`softBary_empMeasure`).  For unit tokens
`m_j(A)` is a convex combination of unit vectors, so `‖m_j(A)‖ ≤ 1`
(`norm_sum_attnProb_smul_le`), and it is the common token when all tokens
coincide (`sum_attnProb_smul_const`).  The remaining lemmas are the
measurability in `θ` that integrating against a head law requires.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`.
-/

import Transformer.Homogenized.Simplex
import Mathlib.Analysis.Matrix.MeasurableSpace
import Mathlib.Analysis.Normed.Lp.MeasurableSpace

open scoped BigOperators
open MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The attention average at a token -/

/-- **`B_θ[μ_X](x_j) = V m_j(A)`**, with `m_j(A) = Σ_k π^A_{j→k} x_k`: the
normalizer is a scalar and `V` is linear.  This is
`attnFieldOf_eq_valueMap_softBary` read at the empirical measure, where the
barycenter is the attention average (`softBary_empMeasure`).

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem attnField_self_eq_valueMap {d n : ℕ} (β : ℝ) (θ : HeadParam d)
    (x : Idx n → EucSpace d) (j : Idx n) :
    attnField β θ x (x j) = valueMap θ (∑ k, attnProb β θ.2 x j k • x k) := by
  rw [attnField, valueMap, map_sum, Finset.smul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [map_smul, attnProb, div_eq_inv_mul, smul_smul]
  rfl

/-- **`‖m_j(A)‖ ≤ 1`**: the attention average of unit tokens is a convex
combination of unit vectors.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula` ("a convex combination
of unit vectors"). -/
theorem norm_sum_attnProb_smul_le {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    {x : Idx n → EucSpace d} (hx : ∀ i, ‖x i‖ = 1) (j : Idx n) :
    ‖∑ k, attnProb β A x j k • x k‖ ≤ 1 := by
  refine (norm_sum_le _ _).trans ?_
  calc ∑ k, ‖attnProb β A x j k • x k‖ = ∑ k, attnProb β A x j k :=
        Finset.sum_congr rfl fun k _ => by
          rw [norm_smul, hx k, mul_one, Real.norm_of_nonneg (attnProb_nonneg β A x j k)]
    _ ≤ 1 := sum_attnProb_le_one β A x j

/-- When all tokens coincide, the attention average is the common token. -/
theorem sum_attnProb_smul_const {d m : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (e : EucSpace d) (j : Idx (m + 1)) :
    ∑ k, attnProb β A (fun _ => e) j k • e = e := by
  rw [← Finset.sum_smul, sum_attnProb, one_smul]

/-- The weight of a head `(0, A)` is continuous in `A`. -/
theorem continuous_attnWeight_qk {d : ℕ} (β : ℝ) (u v : EucSpace d) :
    Continuous fun A : Matrix (Fin d) (Fin d) ℝ => attnWeight β ((0, A) : HeadParam d) u v := by
  have h : Continuous fun A : Matrix (Fin d) (Fin d) ℝ => Matrix.toEuclideanLin A u :=
    (PiLp.continuous_toLp 2 _).comp (continuous_id.matrix_mulVec continuous_const)
  exact Real.continuous_exp.comp (continuous_const.mul (h.inner continuous_const))

/-- The attention average is measurable in `A`. -/
theorem measurable_sum_attnProb_smul {d n : ℕ} (β : ℝ) (x : Idx n → EucSpace d) (j : Idx n) :
    Measurable fun A : Matrix (Fin d) (Fin d) ℝ => ∑ k, attnProb β A x j k • x k := by
  refine Finset.measurable_sum _ fun k _ => ?_
  refine Measurable.smul_const ?_ (x k)
  exact (continuous_attnWeight_qk β (x j) (x k)).measurable.div
    (Finset.measurable_sum _ fun l _ => (continuous_attnWeight_qk β (x j) (x l)).measurable)

/-- The hypothesis of `norm_sum_attnProb_smul_le` is satisfiable: one token at
`basePoint d`. -/
example (d : ℕ) : ∀ _i : Idx 1, ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 :=
  fun _ => by simp [basePoint, PiLp.norm_single]

/-! ### Measurability in the head -/

/-- `w ↦ ‖Proj_e w‖²` is continuous. -/
theorem continuous_norm_proj_sq {d : ℕ} (e : EucSpace d) :
    Continuous fun w : EucSpace d => ‖proj d e w‖ ^ 2 := by
  unfold proj
  fun_prop

/-- `(M, a) ↦ M a` is continuous. -/
theorem continuous_toEuclideanLin_prod {d : ℕ} :
    Continuous fun p : Matrix (Fin d) (Fin d) ℝ × EucSpace d => Matrix.toEuclideanLin p.1 p.2 :=
  (PiLp.continuous_toLp 2 _).comp
    (continuous_fst.matrix_mulVec ((PiLp.continuous_ofLp 2 _).comp continuous_snd))

/-- `θ ↦ ‖Proj_e V Y(A)‖²` is measurable when `Y` is. -/
theorem measurable_norm_proj_sq {d : ℕ} (e : EucSpace d)
    {Y : Matrix (Fin d) (Fin d) ℝ → EucSpace d} (hY : Measurable Y) :
    Measurable fun θ : HeadParam d => ‖proj d e (Matrix.toEuclideanLin θ.1 (Y θ.2))‖ ^ 2 := by
  have hc : Measurable fun p : Matrix (Fin d) (Fin d) ℝ × EucSpace d =>
      ‖proj d e (Matrix.toEuclideanLin p.1 p.2)‖ ^ 2 :=
    ((continuous_norm_proj_sq e).comp continuous_toEuclideanLin_prod).measurable
  have hf : Measurable fun θ : HeadParam d =>
      ((θ.1, Y θ.2) : Matrix (Fin d) (Fin d) ℝ × EucSpace d) :=
    measurable_fst.prodMk (hY.comp measurable_snd)
  have h := hc.comp hf
  exact h

/-- The hypothesis of `measurable_norm_proj_sq` is satisfiable. -/
example (d : ℕ) : Measurable fun _ : Matrix (Fin d) (Fin d) ℝ => (0 : EucSpace d) :=
  measurable_const

end Homogenized
end Transformer
