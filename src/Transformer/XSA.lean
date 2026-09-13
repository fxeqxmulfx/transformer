/-
# Exclusive Self-Attention (XSA)

Formalization of:
  Shuangfei Zhai, "Exclusive Self Attention", arXiv:2603.09078v1.

The paper is empirical (no theorems).  Only one equation is introduced:
`eq:xsa`.  We record it here and note one mathematical observation that ties
XSA to the spherical-`SA` model already formalized in `Transformer.Section1_IPS`.

Two claims of the paper are empirical and stay prose, because there is nothing
in them to prove: that `⟪y_i, v_i⟫ / (‖y_i‖ ‖v_i‖)` is consistently positive in
*trained* transformers and grows with depth (§1), and that the mass XSA frees
is reallocated to `a_{i,i}`, an implicit attention sink (§4).  Both are about
what training produces, not about the operator; the record runs of
`openai/parameter-golf` (2026-09-13) adopting XSA on the deepest layers and then
on all of them (1.1307 → 1.1099 BPB) are evidence for the first, and evidence is
not a theorem.  What *is* provable about `z_i` is `xsa_output_orthogonal_to_value`
below: the component XSA removes is removed exactly.
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
  have hnorm : ‖V (x i)‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hv)
  show inner (𝕜 := ℝ) (SAOutput d n Q K V x i
      - (inner (𝕜 := ℝ) (SAOutput d n Q K V x i) (V (x i)) / ‖V (x i)‖ ^ 2) • V (x i))
    (V (x i)) = 0
  rw [inner_sub_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
  field_simp
  ring

/-- The hypothesis is satisfiable, and the theorem is not vacuous on it: any
nonzero value vector will do, and `V = I` at a unit input is one. -/
example (x : EucSpace d) (hx : ‖x‖ = 1) :
    (ContinuousLinearMap.id ℝ (EucSpace d)) x ≠ 0 := by
  intro h
  have hx0 : x = 0 := h
  rw [hx0, norm_zero] at hx
  exact zero_ne_one hx

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
  have hx : ‖x i‖ ^ 2 = 1 := by rw [h_unit i]; norm_num
  show SAOutput d n Q K (ContinuousLinearMap.id ℝ (EucSpace d)) x i
      - (inner (𝕜 := ℝ) (SAOutput d n Q K (ContinuousLinearMap.id ℝ (EucSpace d)) x i)
          ((ContinuousLinearMap.id ℝ (EucSpace d)) (x i))
        / ‖(ContinuousLinearMap.id ℝ (EucSpace d)) (x i)‖ ^ 2)
        • (ContinuousLinearMap.id ℝ (EucSpace d)) (x i) = _
  rw [show (ContinuousLinearMap.id ℝ (EucSpace d)) (x i) = x i from rfl, hx, div_one,
    real_inner_comm]
  rfl

/-- The hypothesis is satisfiable: the constant unit tuple. -/
example : ∀ _i : Idx n, ‖(WithLp.toLp 2 (fun j => if j = (0 : Fin (d + 1)) then (1 : ℝ) else 0)
    : EucSpace (d + 1))‖ = 1 := by
  intro _
  rw [EuclideanSpace.norm_eq]
  simp

end XSA
end Transformer
