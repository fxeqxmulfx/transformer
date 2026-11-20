module

public import Transformer.AttentionResiduals.Section3_Attention
public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Mathlib.Analysis.Calculus.Deriv.Inv
public import Mathlib.Analysis.Calculus.FDeriv.Prod
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# LeRoPE, Appendix B.1: the softmax Jacobian and Lemma B.1

Karypis et al., arXiv:2607.10134v1, Appendix B.1. Without the scale `1/√d` and on one head, a
query attends to its keys `t` with the weights `α_t = softmax(z)_t` of its logits and outputs
`o = Σ_t α_t v_t`, before the output projection. Band `m`'s product `D_t` of the query and key
`t` enters the logit as a summand, `z_t = r_t + D_t`: Lemma B.1's `∂z_st/∂D_st = 1`.

* `hasFDerivAt_depthWeight`: the softmax Jacobian, `∂α_t/∂z_u = α_t (1[t = u] − α_u)`;
* `hasFDerivAt_attend`: Lemma B.1's display
  `∂o/∂z_t = Σ_{t'} α_{t'} (1[t = t'] − α_t) v_{t'} = α_t (v_t − o)`;
* `hasFDerivAt_loss_attend`: Lemma B.1, `∂L/∂D_t = α_t gᵀ(v_t − o)` for every loss `L` of the
  output with the gradient `g = ∂L/∂o` there.

The attention is `AttentionResiduals.attend`, over any finite set of keys. The paper writes
Lemma B.1 for the query at the position `s` among all of them; "the only effect of the value
`D_st` is through its effect on the output vector at position `s`", and its loss is a loss of
`o_s` with the other outputs held, its gradient `g_s`. Causal attention and Lemma B.3 are in
`SectionB1_Covariance`.
-/

@[expose] public section

open scoped InnerProductSpace
open Finset Transformer.AttentionResiduals

namespace Transformer.RoPE.LeRoPE

variable {ι : Type*} [Fintype ι]

/-- Appendix B.1: **the softmax Jacobian**, `∂α_t/∂z_u = α_t (1[t = u] − α_u)`. -/
theorem hasFDerivAt_depthWeight (z : ι → ℝ) (t : ι) :
    HasFDerivAt (fun z => depthWeight z t)
      (depthWeight z t • ((ContinuousLinearMap.proj t : (ι → ℝ) →L[ℝ] ℝ) -
        ∑ u, depthWeight z u • (ContinuousLinearMap.proj u : (ι → ℝ) →L[ℝ] ℝ))) z := by
  have : Nonempty ι := ⟨t⟩
  have hZ := depthPartition_pos z
  have hnum : HasFDerivAt (fun z : ι → ℝ => Real.exp (z t))
      (Real.exp (z t) • (ContinuousLinearMap.proj t : (ι → ℝ) →L[ℝ] ℝ)) z :=
    (hasFDerivAt_apply t z).exp
  have hden : HasFDerivAt (fun z : ι → ℝ => ∑ u, Real.exp (z u))
      (∑ u, Real.exp (z u) • (ContinuousLinearMap.proj u : (ι → ℝ) →L[ℝ] ℝ)) z :=
    HasFDerivAt.fun_sum fun u _ => (hasFDerivAt_apply u z).exp
  have hmul := hnum.fun_mul ((hasDerivAt_inv hZ.ne').comp_hasFDerivAt z hden)
  simp only [depthWeight, depthPartition, div_eq_mul_inv, Function.comp_def] at hmul hZ ⊢
  refine hmul.congr_fderiv (ContinuousLinearMap.ext fun h => ?_)
  have hs : ∑ u, Real.exp (z u) * (∑ l, Real.exp (z l))⁻¹ * h u
      = (∑ u, Real.exp (z u) * h u) * (∑ l, Real.exp (z l))⁻¹ := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun u _ => by ring
  simp only [smul_apply, sub_apply,
    add_apply, FunLike.coe_sum, Finset.sum_apply,
    ContinuousLinearMap.proj_apply, smul_eq_mul, mul_sub, hs]
  field_simp
  ring

/-- Lemma B.1's proof: the softmax Jacobian gives
`∂o/∂z_t = Σ_{t'} α_{t'} (1[t = t'] − α_t) v_{t'} = α_t (v_t − o)`, for values in any normed
space. -/
theorem hasFDerivAt_attend {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] (z : ι → ℝ)
    (v : ι → E) :
    HasFDerivAt (fun z => attend z v)
      (∑ t, (depthWeight z t • (ContinuousLinearMap.proj t : (ι → ℝ) →L[ℝ] ℝ)).smulRight
        (v t - attend z v)) z := by
  have h : HasFDerivAt (fun z => attend z v) (∑ t, (depthWeight z t •
      ((ContinuousLinearMap.proj t : (ι → ℝ) →L[ℝ] ℝ) - ∑ u, depthWeight z u •
        (ContinuousLinearMap.proj u : (ι → ℝ) →L[ℝ] ℝ))).smulRight (v t)) z :=
    HasFDerivAt.fun_sum fun t _ => (hasFDerivAt_depthWeight z t).smul_const (v t)
  refine h.congr_fderiv (ContinuousLinearMap.ext fun h => ?_)
  simp only [FunLike.coe_sum, Finset.sum_apply, ContinuousLinearMap.smulRight_apply,
    smul_apply, sub_apply, ContinuousLinearMap.proj_apply,
    smul_eq_mul, smul_sub, sub_smul, Finset.sum_sub_distrib]
  congr 1
  rw [← Finset.sum_smul, attend, Residual.weightedSum, Finset.smul_sum]
  exact Finset.sum_congr rfl fun t _ => by rw [smul_smul, mul_comm]

/-- Lemma B.1: **the gradient of a band's product**. For a loss `L` of the output
`o = Σ_t α_t v_t` of the logits `z = r + D`, with the gradient `g = ∂L/∂o` at `o`,
`∂L/∂D_t = α_t gᵀ(v_t − o)`. -/
theorem hasFDerivAt_loss_attend {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [CompleteSpace E] (r D : ι → ℝ) (v : ι → E) {L : E → ℝ} {g : E}
    (hL : HasGradientAt L g (attend (r + D) v)) :
    HasFDerivAt (fun D => L (attend (r + D) v))
      (∑ t, (depthWeight (r + D) t * ⟪g, v t - attend (r + D) v⟫_ℝ) •
        (ContinuousLinearMap.proj t : (ι → ℝ) →L[ℝ] ℝ)) D := by
  have hz : HasFDerivAt (fun D : ι → ℝ => r + D) (ContinuousLinearMap.id ℝ (ι → ℝ)) D :=
    (hasFDerivAt_id D).const_add r
  have h := (hasGradientAt_iff_hasFDerivAt.1 hL).comp D
    ((hasFDerivAt_attend (r + D) v).comp D hz)
  refine h.congr_fderiv (ContinuousLinearMap.ext fun h => ?_)
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.id_apply,
    InnerProductSpace.toDual_apply_apply, FunLike.coe_sum, Finset.sum_apply,
    ContinuousLinearMap.smulRight_apply, smul_apply,
    ContinuousLinearMap.proj_apply, smul_eq_mul, inner_sum, real_inner_smul_right]
  exact Finset.sum_congr rfl fun t _ => by ring

/-- The hypothesis of `hasFDerivAt_loss_attend` holds for the loss `⟪g, o⟫`, whose gradient is
`g` everywhere. -/
example (r D : Fin 2 → ℝ) (v : Fin 2 → EuclideanSpace ℝ (Fin 2)) (g : EuclideanSpace ℝ (Fin 2)) :=
  hasFDerivAt_loss_attend r D v (L := InnerProductSpace.toDual ℝ _ g)
    (hasGradientAt_iff_hasFDerivAt.2 (InnerProductSpace.toDual ℝ _ g).hasFDerivAt)

end Transformer.RoPE.LeRoPE
