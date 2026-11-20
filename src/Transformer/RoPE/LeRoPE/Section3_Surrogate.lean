module

public import Transformer.RoPE.LeRoPE.Section3_Gradient

/-!
# LeRoPE, §3.2: the surrogate of a frequency's gradient

Karypis et al., arXiv:2607.10134v1, §3.2, Lemma 3.1, which is Lemma B.2 of Appendix B.1. Weights
`g_st` on the pairs and band `m`'s chunks define

* `offsetSignal`: the signal `C(δ) = Σ_{s−t=δ} ‖q_s‖ ‖k_t‖ e^{iφ_st} g_st` of the offset `δ`; in
  the paper `g_st = α_st (v_t − o_s)ᵀ g_s`, which is `∂L/∂z_st` (Lemma B.1);
* `surrogate`: `P(θ) = Re Σ_{δ=1}^{L−1} C(δ) e^{iδθ}`, the real part of the signal's discrete-time
  Fourier transform at `θ`, here of a context of `L = n` tokens.

Then:

* `surrogate_eq`: regrouped by pairs (`sum_offset`), `P(θ) = Σ_s Σ_{t<s} g_st D_st(θ)`;
* `hasDerivAt_surrogate`: `P'(θ) = −Σ_s Σ_{t<s} (s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + (s − t) θ) g_st`;
* `hasDerivAt_loss_deriv_surrogate`: Lemma 3.1, `∂L/∂θ = P'(θ)`, with the weights
  `g_st = ∂L/∂z_st` at `θ`, frozen;
* `hasDerivAt_loss_log_frequency`: LeRoPE trains `α`, with `θ̂ = e^α θ` (Eq. (5)); the chain rule,
  which the paper does not write out, gives `∂L/∂α = θ̂ P'(θ̂)`.
-/

@[expose] public section

open scoped ComplexConjugate
open Finset

namespace Transformer.RoPE.LeRoPE

variable {n : ℕ}

/-- §3.2: the offset signal `C(δ) = Σ_{s−t=δ} ‖q_s‖ ‖k_t‖ e^{iφ_st} g_st` of the weights `g` on
the pairs at the offset `δ`, `s = t + δ`, with `φ_st = arg(q_s k̄_t)`. -/
noncomputable def offsetSignal (q k : Fin n → ℂ) (g : Fin n × Fin n → ℝ) (δ : ℕ) : ℂ :=
  ∑ p ∈ univ.filter (fun p : Fin n × Fin n => (p.1 : ℕ) = p.2 + δ),
    ((‖q p.1‖ * ‖k p.2‖ * g p : ℝ) : ℂ) *
      Complex.exp (Complex.arg (q p.1 * conj (k p.2)) * Complex.I)

/-- §3.2: the surrogate `P(θ) = Re Σ_{δ=1}^{L−1} C(δ) e^{iδθ}` of a context of `L = n` tokens. -/
noncomputable def surrogate (q k : Fin n → ℂ) (g : Fin n × Fin n → ℝ) (θ : ℝ) : ℝ :=
  (∑ δ ∈ Ico 1 n, offsetSignal q k g δ * Complex.exp ((δ * θ : ℝ) * Complex.I)).re

/-- The causal pairs `t < s`, summed query by query. -/
theorem sum_causal {M : Type*} [AddCommMonoid M] (f : Fin n × Fin n → M) :
    ∑ p ∈ univ.filter (fun p : Fin n × Fin n => p.2 < p.1), f p = ∑ s, ∑ t ∈ Iio s, f (s, t) := by
  rw [sum_filter, Fintype.sum_prod_type]
  refine sum_congr rfl fun s _ => ?_
  rw [← sum_filter]
  congr 1
  ext t
  simp

/-- Lemma B.2's regrouping: the causal pairs `t < s` by their offset `δ = s − t ∈ [1, n − 1]`. -/
theorem sum_offset {M : Type*} [AddCommMonoid M] (f : Fin n × Fin n → M) :
    ∑ δ ∈ Ico 1 n, ∑ p ∈ univ.filter (fun p : Fin n × Fin n => (p.1 : ℕ) = p.2 + δ), f p =
      ∑ s, ∑ t ∈ Iio s, f (s, t) := by
  have hmaps : ∀ p ∈ univ.filter (fun p : Fin n × Fin n => p.2 < p.1),
      (p.1 : ℕ) - p.2 ∈ Ico 1 n := by
    intro p hp
    have h1 := p.1.isLt
    simp only [mem_filter, mem_univ, true_and, Fin.lt_def] at hp
    simp only [mem_Ico]
    omega
  rw [← sum_causal, ← sum_fiberwise_of_maps_to (g := fun p : Fin n × Fin n => (p.1 : ℕ) - p.2)
    hmaps f]
  refine sum_congr rfl fun δ hδ => sum_congr ?_ fun _ _ => rfl
  ext p
  simp only [mem_Ico] at hδ
  simp only [mem_filter, mem_univ, true_and, Fin.lt_def]
  constructor <;> intro h <;> omega

/-- The real part of the phasor `a e^{iφ} e^{ix}`, `a` real: `a cos(φ + x)`. -/
theorem re_ofReal_mul_exp_mul_exp (a φ x : ℝ) :
    ((a : ℂ) * Complex.exp (φ * Complex.I) * Complex.exp (x * Complex.I)).re =
      a * Real.cos (φ + x) := by
  rw [mul_assoc, ← Complex.exp_add, ← add_mul, ← Complex.ofReal_add, Complex.re_ofReal_mul,
    Complex.exp_ofReal_mul_I_re]

/-- Lemma B.2, regrouped by pairs: `P(θ) = Σ_s Σ_{t<s} g_st D_st(θ)`, the band products weighted
by `g`. -/
theorem surrogate_eq (q k : Fin n → ℂ) (g : Fin n × Fin n → ℝ) (θ : ℝ) :
    surrogate q k g θ =
      ∑ s : Fin n, ∑ t ∈ Iio s, g (s, t) * bandProduct θ (s : ℕ) (t : ℕ) (q s) (k t) := by
  have h := sum_offset fun p : Fin n × Fin n =>
    g p * bandProduct θ (p.1 : ℕ) (p.2 : ℕ) (q p.1) (k p.2)
  dsimp only at h
  rw [← h, surrogate, Complex.re_sum]
  refine sum_congr rfl fun δ _ => ?_
  rw [offsetSignal, sum_mul, Complex.re_sum]
  refine sum_congr rfl fun p hp => ?_
  have hδ : (δ : ℝ) = ((p.1 : ℕ) : ℝ) - (p.2 : ℕ) := by
    rw [(mem_filter.1 hp).2, Nat.cast_add]
    ring
  rw [hδ, re_ofReal_mul_exp_mul_exp, bandProduct_eq]
  ring

/-- §3.2: the surrogate's derivative,
`P'(θ) = −Σ_s Σ_{t<s} (s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + (s − t) θ) g_st`. -/
theorem hasDerivAt_surrogate (q k : Fin n → ℂ) (g : Fin n × Fin n → ℝ) (θ : ℝ) :
    HasDerivAt (surrogate q k g)
      (-∑ s : Fin n, ∑ t ∈ Iio s, ((s : ℕ) - (t : ℕ) : ℝ) * ‖q s‖ * ‖k t‖ *
        Real.sin (Complex.arg (q s * conj (k t)) + ((s : ℕ) - (t : ℕ) : ℝ) * θ) * g (s, t)) θ := by
  rw [funext (surrogate_eq q k g)]
  refine (HasDerivAt.fun_sum fun s _ => HasDerivAt.fun_sum fun t _ =>
    (hasDerivAt_bandProduct _ _ (q s) (k t) θ).const_mul (g (s, t))).congr_deriv ?_
  rw [← sum_neg_distrib]
  refine sum_congr rfl fun s _ => ?_
  rw [← sum_neg_distrib]
  exact sum_congr rfl fun t _ => by ring

/-- Lemma 3.1 (Lemma B.2): **the gradient of a frequency is the derivative of the surrogate**.
With the weights `g_st = ∂L/∂z_st` at `θ`, frozen, `∂L/∂θ = P'(θ)`. -/
theorem hasDerivAt_loss_deriv_surrogate {ℓ : (Fin n × Fin n → ℝ) → ℝ}
    {ℓ' : (Fin n × Fin n → ℝ) →L[ℝ] ℝ} {r : Fin n × Fin n → ℝ} {q k : Fin n → ℂ} {θ : ℝ}
    (hℓ : HasFDerivAt ℓ ℓ' (bandLogits r q k θ))
    (hmask : ∀ s t : Fin n, s < t → logitPartial ℓ' (s, t) = 0) :
    HasDerivAt (fun θ => ℓ (bandLogits r q k θ)) (deriv (surrogate q k (logitPartial ℓ')) θ)
      θ := by
  rw [(hasDerivAt_surrogate q k _ θ).deriv]
  exact hasDerivAt_loss hℓ hmask

/-- The hypotheses of `hasDerivAt_loss_deriv_surrogate` hold for the loss that reads the logit
`z_10` of two tokens. -/
example (r : Fin 2 × Fin 2 → ℝ) (q k : Fin 2 → ℂ) (θ : ℝ) :=
  hasDerivAt_loss_deriv_surrogate (ℓ := fun z => z (1, 0)) (r := r) (q := q) (k := k) (θ := θ)
    (hasFDerivAt_apply (1, 0) _) fun _ _ hst => logitPartial_proj_masked (by decide) hst

/-- §3.1 and §3.2: LeRoPE trains `α`, with `θ̂ = e^α θ` (Eq. (5)), and
**`∂L/∂α = θ̂ P'(θ̂)`**, with the weights `g_st = ∂L/∂z_st` at `θ̂`. -/
theorem hasDerivAt_loss_log_frequency {ℓ : (Fin n × Fin n → ℝ) → ℝ}
    {ℓ' : (Fin n × Fin n → ℝ) →L[ℝ] ℝ} {r : Fin n × Fin n → ℝ} {q k : Fin n → ℂ} {θ α : ℝ}
    (hℓ : HasFDerivAt ℓ ℓ' (bandLogits r q k (learnedFrequency θ α)))
    (hmask : ∀ s t : Fin n, s < t → logitPartial ℓ' (s, t) = 0) :
    HasDerivAt (fun α => ℓ (bandLogits r q k (learnedFrequency θ α)))
      (learnedFrequency θ α * deriv (surrogate q k (logitPartial ℓ')) (learnedFrequency θ α))
      α :=
  ((hasDerivAt_loss_deriv_surrogate hℓ hmask).comp α
    (learned_frequency_hasDerivAt θ α)).congr_deriv (mul_comm _ _)

/-- The hypotheses of `hasDerivAt_loss_log_frequency` hold for the loss that reads the logit
`z_10` of two tokens. -/
example (r : Fin 2 × Fin 2 → ℝ) (q k : Fin 2 → ℂ) (θ α : ℝ) :=
  hasDerivAt_loss_log_frequency (ℓ := fun z => z (1, 0)) (r := r) (q := q) (k := k) (θ := θ)
    (α := α) (hasFDerivAt_apply (1, 0) _) fun _ _ hst => logitPartial_proj_masked (by decide) hst

end Transformer.RoPE.LeRoPE
