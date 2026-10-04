/-
# The error signal under the regularizer

arXiv:1211.5063, §3.3.  The regularizer `Ω` (`Section3_Regularizer`) "prefers
solutions for which the error signal preserves norm as it travels back in
time": with `Ω = 0` each step keeps the norm of the nonzero error it sends back
(`norm_comp_of_omegaSum_eq_zero`), and for a loss `L = 𝓛(x_T)` with `∂𝓛 ≠ 0`,
whose errors are `∂L/∂x_k = ∂𝓛 ∂x_T/∂x_k` (eq. (5)), every error has the norm
of `∂𝓛` (`norm_comp_transport_of_omegaSum_eq_zero`).  "We are using a soft
constraint, therefore we are not ensured the norm of the error signal is
preserved": however small `Ω` is, the error can exceed `∂𝓛` any number of
times (`exists_omegaSum_le_le_norm`).
-/

import Transformer.RecurrentGradients.Section3_Regularizer

namespace Transformer.RecurrentGradients

/-- The product of eq. (5) split off at the right: `A (k + l) ⋯ A (k + 1) A k`. -/
theorem transport_succ' {M : Type*} [Monoid M] (A : ℕ → M) (k l : ℕ) :
    transport A k (l + 1) = transport A (k + 1) l * A k := by
  induction l with
  | zero =>
    show A (k + 0) * 1 = 1 * A k
    rw [add_zero, mul_one, one_mul]
  | succ l ih =>
    rw [transport, ih, transport, mul_assoc, show k + (l + 1) = k + 1 + l by omega]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **§3.3, for any errors**, as those of a loss `Σ_t L_t`: with `Ω = 0` each
step `∂x_{k+1}/∂x_k`, `k < T`, keeps the norm of the nonzero error
`∂L/∂x_{k+1}` it sends back. -/
theorem norm_comp_of_omegaSum_eq_zero {δ : ℕ → E →L[ℝ] ℝ} {A : ℕ → E →L[ℝ] E} {T : ℕ}
    (h : omegaSum δ A T = 0) {k : ℕ} (hk : k < T) (hδ : δ (k + 1) ≠ 0) :
    ‖δ (k + 1) ∘L A k‖ = ‖δ (k + 1)‖ :=
  (omegaTerm_eq_zero_iff hδ _).1 ((omegaSum_eq_zero_iff δ A T).1 h k hk)

/-- The hypotheses of `norm_comp_of_omegaSum_eq_zero` are satisfiable: the
identity errors and steps of `ℝ`. -/
example : ‖ContinuousLinearMap.id ℝ ℝ ∘L ContinuousLinearMap.id ℝ ℝ‖ =
    ‖ContinuousLinearMap.id ℝ ℝ‖ :=
  norm_comp_of_omegaSum_eq_zero (δ := fun _ => ContinuousLinearMap.id ℝ ℝ)
    (A := fun _ => ContinuousLinearMap.id ℝ ℝ) (T := 1) (by simp [omegaSum, omegaTerm])
    zero_lt_one id_ne_zero

/-- **§3.3: the regularizer "prefers solutions for which the error signal
preserves norm as it travels back in time".**  For a loss `L = 𝓛(x_T)` with
`∂𝓛 ≠ 0`, whose errors are `∂L/∂x_k = ∂𝓛 ∂x_T/∂x_k` (eq. (5)), `Ω = 0` gives
every error `∂L/∂x_k`, `k ≤ T`, the norm of `∂𝓛`. -/
theorem norm_comp_transport_of_omegaSum_eq_zero {ℓ' : E →L[ℝ] ℝ} (hℓ : ℓ' ≠ 0)
    {A : ℕ → E →L[ℝ] E} {T : ℕ} (h : omegaSum (fun k => ℓ' ∘L transport A k (T - k)) A T = 0)
    {k : ℕ} (hk : k ≤ T) : ‖ℓ' ∘L transport A k (T - k)‖ = ‖ℓ'‖ := by
  rw [omegaSum_eq_zero_iff] at h
  have key : ∀ j ≤ T, ‖ℓ' ∘L transport A (T - j) j‖ = ‖ℓ'‖ := by
    intro j
    induction j with
    | zero => exact fun _ => congrArg norm (ContinuousLinearMap.ext fun _ => rfl)
    | succ j ih =>
      intro hj
      have e : T - (j + 1) + 1 = T - j := by omega
      have hk := h (T - (j + 1)) (by omega)
      simp only [e, show T - (T - j) = j by omega] at hk
      have ih' := ih (by omega)
      have hne : ℓ' ∘L transport A (T - j) j ≠ 0 := by
        intro h0
        rw [h0, norm_zero] at ih'
        exact hℓ (norm_eq_zero.1 ih'.symm)
      rw [omegaTerm_eq_zero_iff hne] at hk
      rw [transport_succ', e, ContinuousLinearMap.mul_def, ← ContinuousLinearMap.comp_assoc, hk,
        ih']
  have := key (T - k) (by omega)
  rwa [show T - (T - k) = k by omega] at this

/-- The hypotheses of `norm_comp_transport_of_omegaSum_eq_zero` are satisfiable:
the identity steps and loss of `ℝ`. -/
example (T : ℕ) :
    ‖ContinuousLinearMap.id ℝ ℝ ∘L transport (fun _ => (1 : ℝ →L[ℝ] ℝ)) 0 (T - 0)‖ =
      ‖ContinuousLinearMap.id ℝ ℝ‖ := by
  refine norm_comp_transport_of_omegaSum_eq_zero id_ne_zero ?_ (Nat.zero_le T)
  rw [omegaSum_eq_zero_iff]
  intro k _
  simp only [transport_const, one_pow]
  rw [omegaTerm_eq_zero_iff (by simp)]
  rfl

/-- **§3.3: "we are using a soft constraint, therefore we are not ensured the
norm of the error signal is preserved", which may explode "as `t - k`
increases"**: however small `Ω` is, the error `∂L/∂x_0` of a loss `L = 𝓛(x_T)`
can exceed `∂𝓛` any number of times.  On `ℝ`, `T = N K` steps of Jacobian
`1 + 1/N` give `Ω = K/N` and `∂L/∂x_0 = (1 + 1/N)^{NK} ∂𝓛`, at least
`(1 + K) ∂𝓛`. -/
theorem exists_omegaSum_le_le_norm {ε : ℝ} (hε : 0 < ε) (M : ℝ) :
    ∃ (T : ℕ) (A : ℕ → ℝ →L[ℝ] ℝ),
      omegaSum (fun k => ContinuousLinearMap.id ℝ ℝ ∘L transport A k (T - k)) A T ≤ ε ∧
        M ≤ ‖ContinuousLinearMap.id ℝ ℝ ∘L transport A 0 T‖ := by
  obtain ⟨K, hK⟩ := exists_nat_ge M
  obtain ⟨N, hN⟩ := exists_nat_gt ((K : ℝ) / ε)
  have hN0 : (0 : ℝ) < N := (div_nonneg K.cast_nonneg hε.le).trans_lt hN
  have hc : (0 : ℝ) < 1 + 1 / N := by positivity
  have he : ∀ k l, ContinuousLinearMap.id ℝ ℝ ∘L transport
      (fun _ => (1 + 1 / (N : ℝ)) • (1 : ℝ →L[ℝ] ℝ)) k l = (1 + 1 / (N : ℝ)) ^ l • 1 :=
    fun k l => by rw [transport_const, smul_pow, one_pow, ContinuousLinearMap.id_comp]
  have hn : ∀ s : ℝ, ‖s • (1 : ℝ →L[ℝ] ℝ)‖ = |s| := fun s => by
    rw [norm_smul, norm_one, mul_one, Real.norm_eq_abs]
  refine ⟨N * K, fun _ => (1 + 1 / (N : ℝ)) • 1, ?_, ?_⟩
  · rw [omegaSum, Finset.sum_eq_card_nsmul (b := 1 / (N : ℝ) ^ 2) fun k _ => by
      rw [omegaTerm, he, ← ContinuousLinearMap.mul_def, smul_mul_smul_comm, one_mul, hn, hn,
        abs_mul, mul_div_cancel_left₀ _ (abs_pos.2 (pow_pos hc _).ne').ne', abs_of_pos hc]
      ring, Finset.card_range, nsmul_eq_mul]
    rw [div_lt_iff₀ hε] at hN
    push_cast
    rw [show (N : ℝ) * K * (1 / N ^ 2) = K / N by field_simp, div_le_iff₀ hN0]
    linarith
  · rw [he, hn, abs_of_pos (pow_pos hc _)]
    have hb := one_add_mul_le_pow (show (-2 : ℝ) ≤ 1 / N by
      linarith [one_div_pos.2 hN0]) (N * K)
    rw [show ((N * K : ℕ) : ℝ) * (1 / N) = K by push_cast; field_simp] at hb
    linarith

/-- The hypothesis of `exists_omegaSum_le_le_norm` is satisfiable: `ε = 1`. -/
example : ∃ (T : ℕ) (A : ℕ → ℝ →L[ℝ] ℝ),
    omegaSum (fun k => ContinuousLinearMap.id ℝ ℝ ∘L transport A k (T - k)) A T ≤ 1 ∧
      2 ≤ ‖ContinuousLinearMap.id ℝ ℝ ∘L transport A 0 T‖ :=
  exists_omegaSum_le_le_norm one_pos 2

end Transformer.RecurrentGradients
