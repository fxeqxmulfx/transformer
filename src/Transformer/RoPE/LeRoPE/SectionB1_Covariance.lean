module

public import Transformer.RoPE.LeRoPE.Section3_Gradient
public import Transformer.RoPE.LeRoPE.SectionB1_SoftmaxJacobian

/-!
# LeRoPE, Appendix B.1: Lemma B.3, a frequency's gradient as a covariance

Karypis et al., arXiv:2607.10134v1, Appendix B.1, Lemma B.3. Causal attention as §3.2 writes it:
the query at the position `s` attends to the keys `t < s` with the weights
`α_st = softmax_{t<s}(z_st)`, Lemma B.3's `p_s(t)`, and outputs `o_s = Σ_{t<s} α_st v_t`
(`causalWeight`, `causalOutput`). Lemma B.3 calls `v_t` "the attention value vector at
position s"; it is the value at the key position `t`, as in §3.2.

* `weightedCov`: `Cov_{t∼p}(X, Y) = E_p[(X − E_p X)(Y − E_p Y)]`;
* `sum_mul_sub_eq_weightedCov`: with weights summing to one, `E_p[Y (X − E_p X)] = Cov_p(X, Y)`,
  the proof's "the expected product of the two terms equals their covariance";
* `hasDerivAt_causalOutput`: Lemma B.1's softmax Jacobian along a path of logits,
  `do_s/dθ = Σ_{t<s} α_st (dz_st/dθ) (v_t − o_s)`;
* `hasDerivAt_loss_causal`: the proof's step for any path of logits: for a loss of the outputs
  with the gradients `g_s = ∂L/∂o_s`, `dL/dθ = Σ_s Cov_{t∼p_s}(v_tᵀ g_s, dz_st/dθ)`;
* `offsetSine`: Appendix B.1's `S_st = (s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + (s − t) θ)`;
* `hasDerivAt_loss_cov`: Lemma B.3, `∂L/∂θ = −Σ_s Cov_{t∼p_s}(v_tᵀ g_s, S_st)`.

The paper derives Lemma B.3 from Eq. (6) with Lemma B.1's `∂L/∂z_st = α_st (v_t − o_s)ᵀ g_s`;
here the chain rule runs along `θ` at once, through `∂z_st/∂θ = −S_st`. The paper's closing
remark, that "in the absence of periodic structure in downstream attention-logit gradients" the
term should "remain centered roughly around zero", is an expectation, not stated here.
-/

@[expose] public section

open scoped ComplexConjugate InnerProductSpace
open Finset Transformer.AttentionResiduals

namespace Transformer.RoPE.LeRoPE

/-- `Cov_{t∼p}(X, Y) = E_p[(X − E_p X)(Y − E_p Y)]`, the covariance of `X` and `Y` under the
weights `p` on a finite set. -/
noncomputable def weightedCov {κ : Type*} [Fintype κ] (p X Y : κ → ℝ) : ℝ :=
  ∑ t, p t * ((X t - ∑ u, p u * X u) * (Y t - ∑ u, p u * Y u))

/-- Lemma B.3's proof: with weights summing to one, `X − E_p X` has mean zero, so
`E_p[Y (X − E_p X)] = Cov_p(X, Y)`. -/
theorem sum_mul_sub_eq_weightedCov {κ : Type*} [Fintype κ] {p : κ → ℝ} (hp : ∑ t, p t = 1)
    (X Y : κ → ℝ) :
    ∑ t, p t * Y t * (X t - ∑ u, p u * X u) = weightedCov p X Y := by
  have h0 : ∑ t, p t * (X t - ∑ u, p u * X u) = 0 := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hp, one_mul, sub_self]
  have h : ∀ t, p t * ((X t - ∑ u, p u * X u) * (Y t - ∑ u, p u * Y u)) =
      p t * Y t * (X t - ∑ u, p u * X u) - (∑ u, p u * Y u) * (p t * (X t - ∑ u, p u * X u)) :=
    fun t => by ring
  rw [weightedCov, Finset.sum_congr rfl fun t _ => h t, Finset.sum_sub_distrib, ← Finset.mul_sum,
    h0, mul_zero, sub_zero]

/-- The hypothesis of `sum_mul_sub_eq_weightedCov` holds for the uniform weights on two points. -/
example (X Y : Fin 2 → ℝ) :=
  sum_mul_sub_eq_weightedCov (p := fun _ => 1 / 2) (by norm_num [Fin.sum_univ_two]) X Y

/-- `Cov_p(X, −Y) = −Cov_p(X, Y)`. -/
theorem weightedCov_neg {κ : Type*} [Fintype κ] (p X Y : κ → ℝ) :
    weightedCov p X (fun t => -Y t) = -weightedCov p X Y := by
  simp only [weightedCov, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Finset.sum_congr rfl fun u _ => mul_neg (p u) (Y u), Finset.sum_neg_distrib]
  ring

variable {n : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- §3.2: the weights `α_st = softmax_{t<s}(z_st)` of the query at the position `s` on the keys
`t < s`, Lemma B.3's `p_s(t)`. -/
noncomputable def causalWeight (z : Fin n × Fin n → ℝ) (s : Fin n) : Iio s → ℝ :=
  depthWeight fun t : Iio s => z (s, t)

/-- §3.2: the output `o_s = Σ_{t<s} α_st v_t` of the query at the position `s`. -/
noncomputable def causalOutput (z : Fin n × Fin n → ℝ) (v : Fin n → E) (s : Fin n) : E :=
  attend (fun t : Iio s => z (s, t)) fun t => v t

/-- Lemma B.1's softmax Jacobian along a path `θ ↦ z(θ)` of logits:
`do_s/dθ = Σ_{t<s} α_st (dz_st/dθ) (v_t − o_s)`. -/
theorem hasDerivAt_causalOutput {z : ℝ → Fin n × Fin n → ℝ} {z' : Fin n × Fin n → ℝ} {θ : ℝ}
    (hz : HasDerivAt z z' θ) (v : Fin n → E) (s : Fin n) :
    HasDerivAt (fun θ => causalOutput (z θ) v s)
      (∑ t : Iio s, (causalWeight (z θ) s t * z' (s, t)) • (v t - causalOutput (z θ) v s)) θ := by
  have hzs : HasDerivAt (fun θ (t : Iio s) => z θ (s, t)) (fun t => z' (s, t)) θ :=
    hasDerivAt_pi.2 fun t => hasDerivAt_pi.1 hz (s, t)
  refine ((hasFDerivAt_attend _ fun t : Iio s => v t).comp_hasDerivAt θ hzs).congr_deriv ?_
  simp only [FunLike.coe_sum, Finset.sum_apply, ContinuousLinearMap.smulRight_apply, smul_apply,
    ContinuousLinearMap.proj_apply, smul_eq_mul]
  rfl

/-- The hypothesis of `hasDerivAt_causalOutput` holds for the logits `θ c_st`. -/
example (c : Fin 2 × Fin 2 → ℝ) (v : Fin 2 → E) (θ : ℝ) :=
  hasDerivAt_causalOutput (z := fun θ p => θ * c p)
    (hasDerivAt_pi.2 fun p => (hasDerivAt_id' (x := θ)).mul_const (c p)) v 1

/-- **The gradient of causal attention along a path of logits**: for a loss `L` of the outputs,
differentiable at `o(z(θ))` with the gradients `g_s = ∂L/∂o_s`,
`dL/dθ = Σ_s Cov_{t∼p_s}(v_tᵀ g_s, dz_st/dθ)`. -/
theorem hasDerivAt_loss_causal {z : ℝ → Fin n × Fin n → ℝ} {z' : Fin n × Fin n → ℝ} {θ : ℝ}
    (hz : HasDerivAt z z' θ) (v : Fin n → E) {L : (Fin n → E) → ℝ} {L' : (Fin n → E) →L[ℝ] ℝ}
    {g : Fin n → E} (hL : HasFDerivAt L L' (causalOutput (z θ) v))
    (hg : ∀ s x, L' (Pi.single s x) = ⟪g s, x⟫_ℝ) :
    HasDerivAt (fun θ => L (causalOutput (z θ) v))
      (∑ s, weightedCov (causalWeight (z θ) s) (fun t => ⟪v t, g s⟫_ℝ) fun t => z' (s, t)) θ := by
  refine (hL.comp_hasDerivAt θ (hasDerivAt_pi.2 fun s =>
    hasDerivAt_causalOutput hz v s)).congr_deriv ?_
  conv_lhs => rw [← Finset.univ_sum_single (fun s => ∑ t : Iio s,
    (causalWeight (z θ) s t * z' (s, t)) • (v t - causalOutput (z θ) v s))]
  rw [map_sum]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [hg]
  rcases isEmpty_or_nonempty (Iio s) with hs | hs
  · simp [weightedCov]
  have hp : ∑ t, causalWeight (z θ) s t = 1 := depthWeight_sum _
  have ho : ⟪g s, causalOutput (z θ) v s⟫_ℝ = ∑ u, causalWeight (z θ) s u * ⟪v u, g s⟫_ℝ := by
    rw [causalOutput, attend, Residual.weightedSum, inner_sum]
    exact Finset.sum_congr rfl fun u _ => by rw [real_inner_smul_right, real_inner_comm]; rfl
  rw [← sum_mul_sub_eq_weightedCov hp]
  simp only [inner_sum, real_inner_smul_right, inner_sub_right, ho]
  exact Finset.sum_congr rfl fun t _ => by rw [real_inner_comm (v t) (g s)]

/-- The hypotheses of `hasDerivAt_loss_causal` hold for the logits `θ c_st` and the loss
`⟪d, o_1⟫` of two tokens, whose gradients are `d` at the position `1` and `0` at `0`. -/
example (c : Fin 2 × Fin 2 → ℝ) (v : Fin 2 → E) (d : E) (θ : ℝ) :=
  hasDerivAt_loss_causal (z := fun θ p => θ * c p)
    (hasDerivAt_pi.2 fun p => (hasDerivAt_id' (x := θ)).mul_const (c p)) v (g := Pi.single 1 d)
    ((innerSL ℝ d).comp (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => E) 1)).hasFDerivAt
    fun s x => by fin_cases s <;> simp

/-- Appendix B.1: the offset-weighted sine term `S_st = (s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + (s − t) θ)`,
with `φ_st = arg(q_s k̄_t)`; `∂D_st/∂θ = −S_st` (`hasDerivAt_bandProduct`). -/
noncomputable def offsetSine (q k : Fin n → ℂ) (θ : ℝ) (s t : Fin n) : ℝ :=
  ((s : ℕ) - (t : ℕ) : ℝ) * ‖q s‖ * ‖k t‖ *
    Real.sin (Complex.arg (q s * conj (k t)) + ((s : ℕ) - (t : ℕ) : ℝ) * θ)

/-- Lemma B.3: **a frequency's gradient is minus a sum of covariances**. For a loss `L` of the
outputs of causal attention on the logits `z_st = r_st + D_st(θ)`, differentiable there with the
gradients `g_s = ∂L/∂o_s`, `∂L/∂θ = −Σ_s Cov_{t∼p_s}(v_tᵀ g_s, S_st)`. -/
theorem hasDerivAt_loss_cov {r : Fin n × Fin n → ℝ} {q k : Fin n → ℂ} (v : Fin n → E)
    {L : (Fin n → E) → ℝ} {L' : (Fin n → E) →L[ℝ] ℝ} {g : Fin n → E} {θ : ℝ}
    (hL : HasFDerivAt L L' (causalOutput (bandLogits r q k θ) v))
    (hg : ∀ s x, L' (Pi.single s x) = ⟪g s, x⟫_ℝ) :
    HasDerivAt (fun θ => L (causalOutput (bandLogits r q k θ) v))
      (-∑ s, weightedCov (causalWeight (bandLogits r q k θ) s) (fun t => ⟪v t, g s⟫_ℝ)
        fun t => offsetSine q k θ s t) θ := by
  have hz : HasDerivAt (bandLogits r q k) (fun p => -offsetSine q k θ p.1 p.2) θ :=
    hasDerivAt_pi.2 fun p => ((hasDerivAt_bandProduct _ _ _ _ θ).const_add (r p)).congr_deriv
      (by rw [offsetSine]; ring)
  refine (hasDerivAt_loss_causal hz v hL hg).congr_deriv ?_
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun s _ => weightedCov_neg _ _ _

/-- The hypotheses of `hasDerivAt_loss_cov` hold for the loss `⟪c, o_1⟫` of two tokens, whose
gradients are `c` at the position `1` and `0` at `0`. -/
example (r : Fin 2 × Fin 2 → ℝ) (q k : Fin 2 → ℂ) (v : Fin 2 → E) (c : E) (θ : ℝ) :=
  hasDerivAt_loss_cov (r := r) (q := q) (k := k) v (g := Pi.single 1 c) (θ := θ)
    ((innerSL ℝ c).comp (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => E) 1)).hasFDerivAt
    fun s x => by fin_cases s <;> simp

end Transformer.RoPE.LeRoPE
