module

public import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies
public import Transformer.RoPE.RoFormer.Section3_Rotary
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# LeRoPE, §3.2: the gradient of a frequency

Karypis et al., arXiv:2607.10134v1, §3.2, Eq. (6). Band `m` adds to the logit `z_st` of the
query at the position `s` and the key at `t` the product `D_st` of its two chunks rotated by
`s θ` and `t θ` (Eq. (3), Lemma B.1's `∂z_st / ∂D_st = 1`); nothing else in the logits depends
on its frequency `θ`. The `n` tokens sit at the positions `0, …, n − 1`.

* `bandProduct_eq`, `hasDerivAt_bandProduct`: in polar form
  `D_st = ‖q_s‖ ‖k_t‖ cos(φ_st + (s − t) θ)`, with `φ_st = arg(q_s k̄_t)` the angle from the key
  to the query chunk, and `∂D_st/∂θ = −(s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + (s − t) θ)`;
* `hasDerivAt_loss`: Eq. (6),
  `∂L/∂θ = −Σ_s Σ_{t<s} (s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + (s − t) θ) ∂L/∂z_st`, for every loss
  differentiable in the logits whose partials at the masked pairs, `t > s`, vanish.

Lemma 3.1, the surrogate, is in `Section3_Surrogate`.
-/

@[expose] public section

open scoped ComplexConjugate InnerProductSpace
open Finset

namespace Transformer.RoPE.LeRoPE

open RoFormer

variable {n : ℕ}

/-- Eq. (3): band `m`'s product `D_st` of the query chunk `q` at the position `s` and the key chunk
`k` at `t`, rotated by `s θ` and `t θ`. -/
noncomputable def bandProduct (θ s t : ℝ) (q k : ℂ) : ℝ :=
  ⟪rotate θ s q, rotate θ t k⟫_ℝ

/-- §3.2, polar form: with `φ = arg(q k̄)`, the angle from the key to the query chunk,
`D_st = ‖q‖ ‖k‖ cos(φ + (s − t) θ)`. -/
theorem bandProduct_eq (θ s t : ℝ) (q k : ℂ) :
    bandProduct θ s t q k = ‖q‖ * ‖k‖ * Real.cos (Complex.arg (q * conj k) + (s - t) * θ) := by
  rw [bandProduct, inner_rotate]
  conv_lhs => rw [← Complex.norm_mul_exp_arg_mul_I (q * conj k)]
  rw [mul_assoc, ← Complex.exp_add, ← add_mul, ← Complex.ofReal_add, Complex.re_ofReal_mul,
    Complex.exp_ofReal_mul_I_re, norm_mul, Complex.norm_conj]

/-- Eq. (6), one pair: `∂D_st/∂θ = −(s − t) ‖q‖ ‖k‖ sin(φ + (s − t) θ)`. -/
theorem hasDerivAt_bandProduct (s t : ℝ) (q k : ℂ) (θ : ℝ) :
    HasDerivAt (fun θ => bandProduct θ s t q k)
      (-((s - t) * (‖q‖ * ‖k‖ * Real.sin (Complex.arg (q * conj k) + (s - t) * θ)))) θ := by
  simp only [bandProduct_eq]
  have h := ((((hasDerivAt_id' (x := θ)).const_mul (s - t)).const_add
    (Complex.arg (q * conj k))).cos).const_mul (‖q‖ * ‖k‖)
  convert h using 1
  ring

/-- The logits `z_st = r_st + D_st(θ)`, as a function of the pair `(s, t)`; `r` is all that does
not depend on band `m`'s frequency. -/
noncomputable def bandLogits (r : Fin n × Fin n → ℝ) (q k : Fin n → ℂ) (θ : ℝ) :
    Fin n × Fin n → ℝ :=
  fun p => r p + bandProduct θ (p.1 : ℕ) (p.2 : ℕ) (q p.1) (k p.2)

/-- `∂L/∂z_st`: the derivative `ℓ'` of a loss of the logits in the coordinate of the pair `p`. -/
noncomputable def logitPartial (ℓ' : (Fin n × Fin n → ℝ) →L[ℝ] ℝ) (p : Fin n × Fin n) : ℝ :=
  ℓ' fun p' => if p = p' then 1 else 0

/-- The derivative of a loss of the logits is the sum of its partials:
`ℓ' v = Σ_s Σ_t v_st ∂L/∂z_st`. -/
theorem apply_eq_sum_logitPartial (ℓ' : (Fin n × Fin n → ℝ) →L[ℝ] ℝ) (v : Fin n × Fin n → ℝ) :
    ℓ' v = ∑ s, ∑ t, v (s, t) * logitPartial ℓ' (s, t) := by
  have h := LinearMap.pi_apply_eq_sum_univ (ℓ' : (Fin n × Fin n → ℝ) →ₗ[ℝ] ℝ) v
  simp only [ContinuousLinearMap.coe_coe, smul_eq_mul] at h
  rw [h, Fintype.sum_prod_type]
  rfl

/-- The partials of the loss that reads the logit `z_p`: `1` at `p`, `0` elsewhere. -/
theorem logitPartial_proj (p p' : Fin n × Fin n) :
    logitPartial (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin n × Fin n => ℝ) p) p' =
      if p' = p then 1 else 0 := by
  rw [logitPartial, ContinuousLinearMap.proj_apply]

/-- The loss that reads a causal logit `z_p`, `p.2 ≤ p.1`, has no partial at a masked pair. -/
theorem logitPartial_proj_masked {p : Fin n × Fin n} (hp : p.2 ≤ p.1) {s t : Fin n}
    (hst : s < t) :
    logitPartial (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin n × Fin n => ℝ) p) (s, t) =
      0 := by
  rw [logitPartial_proj]
  refine ite_eq_right ?_
  rintro rfl
  exact absurd hp (not_le.2 hst)

/-- The hypotheses of `logitPartial_proj_masked` hold for the logit `z_10` and the pair `(0, 1)`. -/
example := logitPartial_proj_masked (n := 2) (p := (1, 0)) (by decide) (s := 0) (t := 1)
  (by decide)

/-- Eq. (6): **the gradient of a frequency**. For a loss `ℓ` of the logits `z_st = r_st + D_st(θ)`,
differentiable at `z(θ)` with the derivative `ℓ'` and whose partials at the masked pairs, `t > s`,
vanish, `∂L/∂θ = −Σ_s Σ_{t<s} (s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + (s − t) θ) ∂L/∂z_st`. -/
theorem hasDerivAt_loss {ℓ : (Fin n × Fin n → ℝ) → ℝ} {ℓ' : (Fin n × Fin n → ℝ) →L[ℝ] ℝ}
    {r : Fin n × Fin n → ℝ} {q k : Fin n → ℂ} {θ : ℝ}
    (hℓ : HasFDerivAt ℓ ℓ' (bandLogits r q k θ))
    (hmask : ∀ s t : Fin n, s < t → logitPartial ℓ' (s, t) = 0) :
    HasDerivAt (fun θ => ℓ (bandLogits r q k θ))
      (-∑ s : Fin n, ∑ t ∈ Iio s, ((s : ℕ) - (t : ℕ) : ℝ) * ‖q s‖ * ‖k t‖ *
        Real.sin (Complex.arg (q s * conj (k t)) + ((s : ℕ) - (t : ℕ) : ℝ) * θ) *
          logitPartial ℓ' (s, t)) θ := by
  have hz : HasDerivAt (bandLogits r q k) (fun p => -((((p.1 : ℕ) : ℝ) - (p.2 : ℕ)) *
      (‖q p.1‖ * ‖k p.2‖ * Real.sin (Complex.arg (q p.1 * conj (k p.2)) +
        (((p.1 : ℕ) : ℝ) - (p.2 : ℕ)) * θ)))) θ :=
    hasDerivAt_pi.2 fun p => (hasDerivAt_bandProduct _ _ _ _ θ).const_add (r p)
  refine (hℓ.comp_hasDerivAt θ hz).congr_deriv ?_
  rw [apply_eq_sum_logitPartial, ← sum_neg_distrib]
  refine sum_congr rfl fun s _ => ?_
  rw [← sum_subset (subset_univ (Iio s)), ← sum_neg_distrib]
  · exact sum_congr rfl fun t _ => by ring
  · intro t _ ht
    rcases (not_lt.1 (mem_Iio.not.1 ht)).lt_or_eq with hst | rfl
    · rw [hmask s t hst, mul_zero]
    · simp

/-- The hypotheses of `hasDerivAt_loss` hold for the loss that reads the logit `z_10` of two
tokens, `t = 0 < s = 1`. -/
example (r : Fin 2 × Fin 2 → ℝ) (q k : Fin 2 → ℂ) (θ : ℝ) :=
  hasDerivAt_loss (ℓ := fun z => z (1, 0)) (r := r) (q := q) (k := k) (θ := θ)
    (hasFDerivAt_apply (1, 0) _) fun _ _ hst => logitPartial_proj_masked (by decide) hst

end Transformer.RoPE.LeRoPE
