/-
# Exclusive Self-Attention (XSA)

Formalization of:
  Shuangfei Zhai, "Exclusive Self Attention", arXiv:2603.09078v1.

The paper is empirical (no theorems).  Only one equation is introduced:
`eq:xsa`.  We record it here and note one mathematical observation that ties
XSA to the spherical-`SA` model already formalized in `Transformer.Section1_IPS`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS

open scoped BigOperators
open Real

namespace Transformer
namespace XSA

variable (d n : ℕ)

/-- **Equation (eq:sa).** Standard causal self-attention output:

  `y_i = Σ_{j ≤ i} a_{i,j} V x_j`,
  `a_{i,j} = exp(⟨Q x_i, K x_j⟩) / Σ_{j' ≤ i} exp(⟨Q x_i, K x_{j'}⟩)`. -/
noncomputable def SAOutput
    (Q K V : ParamMatrix d) (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  let Z := ∑ j : Idx n,
            if (j : ℕ) ≤ (i : ℕ) then
              Real.exp (inner (𝕜 := ℝ) (Q (x i)) (K (x j)))
            else 0
  Z⁻¹ • ∑ j : Idx n,
    if (j : ℕ) ≤ (i : ℕ) then
      Real.exp (inner (𝕜 := ℝ) (Q (x i)) (K (x j))) • V (x j)
    else 0

/-- **Equation (eq:xsa).**  *Exclusive self-attention.*

  `z_i = y_i - (⟨y_i, v_i⟩ / ‖v_i‖²) · v_i`,    `v_i = V x_i`.

`z_i` is the parallel projection of `y_i` onto `v_i^⊥`. -/
noncomputable def XSAOutput
    (Q K V : ParamMatrix d) (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  let y := SAOutput d n Q K V x i
  let v := V (x i)
  y - (inner (𝕜 := ℝ) y v / (‖v‖^2)) • v

/-- **Orthogonality property.**  `⟨z_i, v_i⟩ = 0`, whenever `v_i ≠ 0`. -/
theorem xsa_output_orthogonal_to_value
    (Q K V : ParamMatrix d) (x : Idx n → EucSpace d) (i : Idx n)
    (hv : V (x i) ≠ 0) :
    inner (𝕜 := ℝ) (XSAOutput d n Q K V x i) (V (x i)) = 0 := by
  sorry

/-- **Reduction to spherical `SA`.**  When `V = I_d` and `‖x_i‖ = 1`,
XSA's output coincides with the spherical-tangent projection used in the
canonical `SA` model formalized in §2 of arXiv:2312.10794v5:

  `z_i = y_i - ⟨y_i, x_i⟩ x_i = Proj_{x_i}(y_i)`. -/
theorem xsa_equals_spherical_SA_when_V_is_identity
    (Q K : ParamMatrix d) (x : Idx n → EucSpace d)
    (h_unit : ∀ i, ‖x i‖ = 1) (i : Idx n) :
    XSAOutput d n Q K (ContinuousLinearMap.id ℝ (EucSpace d)) x i
      = proj d (x i) (SAOutput d n Q K
                        (ContinuousLinearMap.id ℝ (EucSpace d)) x i) := by
  sorry

/-- *Attention similarity bias (informal).*

The empirical observation of the paper is that for *trained* transformers,
`⟨y_i, v_i⟩ / (‖y_i‖ ‖v_i‖)` is consistently positive and grows with depth.
XSA removes this component by construction.  As a formal statement this is
just a *training-time* observation; we record it here as a `True`-placeholder
to mark its role in motivating the construction. -/
theorem attention_similarity_bias_observation : True := trivial

/-- *Implicit attention sink (Remark in §4 of the paper).*

XSA reallocates "unused" attention mass to the diagonal entry `a_{i,i}`, which
behaves like an implicit attention sink à la Xiao–Tian–Han. -/
theorem implicit_attention_sink_remark : True := trivial

end XSA
end Transformer
