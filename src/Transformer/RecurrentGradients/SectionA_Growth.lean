/-
# Exponential growth, and the entire gradient

arXiv:1211.5063, supplementary "Analytical analysis of the exploding and
vanishing gradients problem", from the power iteration of
`SectionA_PowerIteration`.  "Given certain conditions, `∂L_t/∂x_t (W_recᵀ)^l`
grows exponentially": the error sent back, `∂L_t/∂x_t ∂x_t/∂x_k`, has norm at
least `C |λ_j|^l` eventually (`eventually_le_norm_comp_factor_id`).  "If
`|λ_j| > 1`, it follows that `∂x_t/∂x_k` grows exponentially fast with `l`":
`‖∂x_t/∂x_k‖ ≥ |λ_j|^l` (`pow_le_norm_factor_id`).  "If `q_j` is not in the
null space of `∂⁺x_k/∂θ` the entire temporal component grows exponentially
with `l`" (`eventually_le_norm_temporal`).

"This approach extends easily to the entire gradient": `∂L_t/∂θ` is the sum
over `j` and `k` of the temporal components `c_j λ_j^{t-k} p_j ∂⁺x_k/∂θ`
(`hasFDerivAt_loss_eigen`, correcting an index).  The claim that the dominant
one makes the sum grow "exponentially fast to infinity with `t`" is false, as
the components of other `k` grow as fast and may cancel it
(`not_tendsto_abs_deriv_atTop`).
-/

import Transformer.RecurrentGradients.SectionA_PowerIteration

open Filter Topology

namespace Transformer.RecurrentGradients

variable {n : ℕ} {κ : Type*} [Fintype κ] {W : Matrix (Fin n) (Fin n) ℝ}
  {Win : Matrix (Fin n) κ ℝ} {b : EuclideanSpace ℝ (Fin n)}

/-- **"Given certain conditions, `∂L_t/∂x_t (W_recᵀ)^l` grows exponentially"**: for
`|λ_1| > ⋯ > |λ_n|`, `j` the first index with `c_j ≠ 0`, `p_j ≠ 0` and `|λ_j| > 1`,
`‖(Σ_i c_i p_i) ∂x_{k+l}/∂x_k‖ ≥ C |λ_j|^l` eventually, `C > 0`, "along the
direction `q_j`" by `tendsto_div_pow_comp_factor_id`. -/
theorem eventually_le_norm_comp_factor_id {p : Fin n → EuclideanSpace ℝ (Fin n) →L[ℝ] ℝ}
    {μ : Fin n → ℝ} (hp : ∀ i, p i ∘L Matrix.toEuclideanCLM (𝕜 := ℝ) W = μ i • p i)
    (hμ : StrictAnti fun i => |μ i|) {c : Fin n → ℝ} {j : Fin n} (hj : ∀ i < j, c i = 0)
    (hcj : c j ≠ 0) (hpj : p j ≠ 0) (hμj : 1 < |μ j|) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ (Fin n)) (k : ℕ) :
    ∃ C > 0, ∀ᶠ l in atTop, C * |μ j| ^ l ≤ ‖(∑ i, c i • p i) ∘L factor id W Win b u x₀ k l‖ := by
  simp only [comp_factor_id hp]
  exact eventually_le_norm_sum (abs_pos.1 (one_pos.trans hμj)) (abs_lt_of_strictAnti hμ hj)
    (smul_ne_zero hcj hpj)

/-- The hypotheses of `eventually_le_norm_comp_factor_id` are satisfiable:
`W_rec = diag(2, 1)`, the coordinates, the error `(1, 1)`. -/
example (Win : Matrix (Fin 2) κ ℝ) (b : EuclideanSpace ℝ (Fin 2)) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ (Fin 2)) (k : ℕ) : ∃ C > 0, ∀ᶠ l in atTop, C * |![(2 : ℝ), 1] 0| ^ l ≤
      ‖(∑ i, (fun _ => (1 : ℝ)) i • EuclideanSpace.proj i) ∘L
        factor id (Matrix.diagonal ![2, 1]) Win b u x₀ k l‖ :=
  eventually_le_norm_comp_factor_id (proj_comp_diagonal _) strictAnti_two_one
    (fun i h => absurd h (Fin.not_lt_zero i)) one_ne_zero
    (fun h => by simpa using congrArg (· (WithLp.toLp 2 ![1, 0])) h) (by norm_num) u x₀ k

/-- **"If `|λ_j| > 1`, it follows that `∂x_t/∂x_k` grows exponentially fast with
`l`"**: a left eigenvector `p ≠ 0` of `W_rec` with eigenvalue `λ` gives
`‖∂x_{k+l}/∂x_k‖ ≥ |λ|^l`. -/
theorem pow_le_norm_factor_id {p : EuclideanSpace ℝ (Fin n) →L[ℝ] ℝ} {μ : ℝ}
    (hp : p ∘L Matrix.toEuclideanCLM (𝕜 := ℝ) W = μ • p) (hp0 : p ≠ 0) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ (Fin n)) (k l : ℕ) : |μ| ^ l ≤ ‖factor id W Win b u x₀ k l‖ := by
  have h := ContinuousLinearMap.opNorm_comp_le (h := p) (Matrix.toEuclideanCLM (𝕜 := ℝ) W ^ l)
  rw [comp_pow_eq_smul hp, norm_smul, norm_pow, Real.norm_eq_abs, mul_comm ‖p‖] at h
  rw [factor_id, map_pow]
  exact le_of_mul_le_mul_right h (norm_pos_iff.2 hp0)

/-- The hypotheses of `pow_le_norm_factor_id` are satisfiable: `W_rec = diag(2, 1)`
and the first coordinate. -/
example (Win : Matrix (Fin 2) κ ℝ) (b : EuclideanSpace ℝ (Fin 2)) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ (Fin 2)) (k l : ℕ) :
    |![(2 : ℝ), 1] 0| ^ l ≤ ‖factor id (Matrix.diagonal ![2, 1]) Win b u x₀ k l‖ :=
  pow_le_norm_factor_id (proj_comp_diagonal _ 0)
    (fun h => by simpa using congrArg (· (WithLp.toLp 2 ![1, 0])) h) u x₀ k l

/-- **"If `q_j` is not in the null space of `∂⁺x_k/∂θ` the entire temporal
component grows exponentially with `l`"**: for `|λ_1| > ⋯ > |λ_n|`, `j` the first
index with `c_j ≠ 0`, `|λ_j| > 1` and `p_j ∂⁺x_k/∂θ ≠ 0`, the temporal component
`(Σ_i c_i p_i) ∂x_{k+l}/∂x_k ∂⁺x_k/∂θ` has norm at least `C |λ_j|^l`
eventually, `C > 0`. -/
theorem eventually_le_norm_temporal {Θ : Type*} [NormedAddCommGroup Θ] [NormedSpace ℝ Θ]
    {p : Fin n → EuclideanSpace ℝ (Fin n) →L[ℝ] ℝ} {μ : Fin n → ℝ}
    (hp : ∀ i, p i ∘L Matrix.toEuclideanCLM (𝕜 := ℝ) W = μ i • p i)
    (hμ : StrictAnti fun i => |μ i|) {c : Fin n → ℝ} {j : Fin n} (hj : ∀ i < j, c i = 0)
    (hcj : c j ≠ 0) (hμj : 1 < |μ j|) {B : Θ →L[ℝ] EuclideanSpace ℝ (Fin n)}
    (hB : p j ∘L B ≠ 0) (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ (Fin n)) (k : ℕ) :
    ∃ C > 0, ∀ᶠ l in atTop,
      C * |μ j| ^ l ≤ ‖((∑ i, c i • p i) ∘L factor id W Win b u x₀ k l) ∘L B‖ := by
  simp only [comp_factor_id hp, ContinuousLinearMap.finsetSum_comp,
    ContinuousLinearMap.smul_comp]
  exact eventually_le_norm_sum (x := fun i => p i ∘L B) (abs_pos.1 (one_pos.trans hμj))
    (abs_lt_of_strictAnti hμ hj) (smul_ne_zero hcj hB)

/-- The hypotheses of `eventually_le_norm_temporal` are satisfiable:
`W_rec = diag(2, 1)`, the coordinates, the error `(1, 1)` and `θ = b`, for which
`∂⁺x_k/∂θ` is the identity. -/
example (Win : Matrix (Fin 2) κ ℝ) (b : EuclideanSpace ℝ (Fin 2)) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ (Fin 2)) (k : ℕ) : ∃ C > 0, ∀ᶠ l in atTop,
      C * |![(2 : ℝ), 1] 0| ^ l ≤ ‖((∑ i, (fun _ => (1 : ℝ)) i • EuclideanSpace.proj i) ∘L
        factor id (Matrix.diagonal ![2, 1]) Win b u x₀ k l) ∘L
          ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin 2))‖ :=
  eventually_le_norm_temporal (proj_comp_diagonal _) strictAnti_two_one
    (fun i h => absurd h (Fin.not_lt_zero i)) one_ne_zero (by norm_num)
    (fun h => by simpa using congrArg (· (WithLp.toLp 2 ![1, 0])) h) u x₀ k

/-- **The supplementary's expansion of the entire gradient, corrected**: "If we
re-write it in terms of the eigen-decomposition of `W`, we get
`∂L_t/∂θ = Σ_{j=1}^n (Σ_{i=k}^t c_j λ_j^{t-k} q_jᵀ ∂⁺x_k/∂θ)`".  When every
Jacobian is `A`, as `W_rec` in the linear model (`factor_id`), and
`∂L_t/∂x_t = Σ_j c_j p_j` with `p_j A = λ_j p_j`,
`∂L_t/∂θ = Σ_j Σ_{1 ≤ k ≤ t} c_j λ_j^{t-k} p_j ∂⁺x_k/∂θ`, by eq. (4): the inner
sum runs over `1 ≤ k ≤ t`, where the paper's `Σ_{i=k}^t` sums a term in `k`. -/
theorem hasFDerivAt_loss_eigen {E Θ U : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup Θ] [NormedSpace ℝ Θ] {F : Θ → E → U → E} {u : ℕ → U} {x₀ : E}
    {θ : Θ} {A : E →L[ℝ] E} {B : ℕ → Θ →L[ℝ] E}
    (hF : ∀ i, HasFDerivAt (fun q : E × Θ => F q.2 q.1 (u (i + 1))) (A.coprod (B i))
      (states (F θ) u x₀ i, θ))
    {m : ℕ} {p : Fin m → E →L[ℝ] ℝ} {μ c : Fin m → ℝ} (hp : ∀ j, p j ∘L A = μ j • p j)
    {ℓ : E → ℝ} {t : ℕ} (hℓ : HasFDerivAt ℓ (∑ j, c j • p j) (states (F θ) u x₀ t)) :
    HasFDerivAt (fun θ' => ℓ (states (F θ') u x₀ t))
      (∑ j, ∑ k ∈ Finset.Icc 1 t, (c j * μ j ^ (t - k)) • (p j ∘L B (k - 1))) θ := by
  convert hasFDerivAt_loss (A := fun _ => A) hF hℓ using 1
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [transport_const, ContinuousLinearMap.finsetSum_comp]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [ContinuousLinearMap.smul_comp, ← ContinuousLinearMap.comp_assoc, comp_pow_eq_smul (hp j),
    ContinuousLinearMap.smul_comp, smul_smul]

/-- The hypotheses of `hasFDerivAt_loss_eigen` are satisfiable: `x_t = 2 x_{t-1} + θ`
on `ℝ`, `θ` a bias, and `L_t = x_t`. -/
example (x₀ θ : ℝ) (t : ℕ) :
    HasFDerivAt (fun θ' => id (states (fun x (_ : Unit) => 2 * x + θ') (fun _ => ()) x₀ t))
      (∑ j : Fin 1, ∑ k ∈ Finset.Icc 1 t, ((fun _ => (1 : ℝ)) j * (fun _ => (2 : ℝ)) j ^ (t - k)) •
        ((fun _ => ContinuousLinearMap.id ℝ ℝ) j ∘L (fun _ => ContinuousLinearMap.id ℝ ℝ) (k - 1)))
      θ :=
  hasFDerivAt_loss_eigen (F := fun (θ' x : ℝ) (_ : Unit) => 2 * x + θ') (u := fun _ => ())
    (x₀ := x₀) (θ := θ) (A := (2 : ℝ) • (1 : ℝ →L[ℝ] ℝ))
    (B := fun _ => ContinuousLinearMap.id ℝ ℝ) (p := fun _ => ContinuousLinearMap.id ℝ ℝ)
    (μ := fun _ => 2) (c := fun _ => 1) (ℓ := id) (t := t)
    (fun _ => ((hasFDerivAt_fst.const_mul (2 : ℝ)).add hasFDerivAt_snd).congr_fderiv
      (ContinuousLinearMap.ext fun q => by simp))
    (fun _ => ContinuousLinearMap.ext fun _ => by simp)
    ((hasFDerivAt_id _).congr_fderiv (ContinuousLinearMap.ext fun _ => by simp))

/-- The inputs `u_1 = 1`, `u_2 = -2` and `u_t = 0` otherwise. -/
noncomputable def cancelling (t : ℕ) : ℝ := if t = 1 then 1 else if t = 2 then -2 else 0

/-- With `w = 2` and the inputs `cancelling`, `x_t = 0` from `t = 2` on, for every
`θ`. -/
theorem states_cancelling (θ : ℝ) (t : ℕ) :
    states (fun x v => 2 * x + θ * v) cancelling 0 (t + 2) = 0 := by
  induction t with
  | zero =>
    show 2 * (2 * 0 + θ * cancelling 1) + θ * cancelling 2 = 0
    rw [show cancelling 1 = 1 by norm_num [cancelling],
      show cancelling 2 = -2 by norm_num [cancelling]]
    ring
  | succ t ih =>
    show 2 * states (fun x v => 2 * x + θ * v) cancelling 0 (t + 2) + θ * cancelling (t + 3) = 0
    simp [ih, cancelling, show t + 3 ≠ 1 by omega, show t + 3 ≠ 2 by omega]

/-- **The supplementary's "the same will happen to the sum" is false.**  "We can
now pick `j` and `k` such that `c_j q_jᵀ ∂⁺x_k/∂θ` does not have 0 norm, while
maximizing `|λ_j|`.  If for the chosen `j` it holds that `|λ_j| > 1` then
`λ_j^{t-k} c_j q_jᵀ ∂⁺x_k/∂θ` will dominate the sum and because this term grows
exponentially fast to infinity with `t`, the same will happen to the sum."  For
eq. (`standard_rnn`) with one linear unit, `W_rec = w`, `b = 0`, `x_0 = 0` and
the parameter `θ = W_in`, so `x_t = w x_{t-1} + θ u_t`, and the loss `L_t = x_t`,
the sum is `∂L_t/∂θ`, with `c_1 q_1 = 1`, `λ_1 = w` and `∂⁺x_k/∂θ = u_k`.  For
`w = 2`, `u_1 = 1` and `u_2 = -2` the temporal components `2^{t-1}` and
`-2^{t-1}` of `k = 1, 2` cancel: `x_t = 0` from `t = 2` on
(`states_cancelling`), so `∂L_t/∂θ = 0`. -/
theorem not_tendsto_abs_deriv_atTop :
    ¬ ∀ (w : ℝ) (u : ℕ → ℝ) (k : ℕ) (θ₀ : ℝ), 1 ≤ k → 1 < |w| → u k ≠ 0 →
      Tendsto (fun t => |deriv (fun θ => states (fun x v => w * x + θ * v) u 0 t) θ₀|) atTop
        atTop := by
  intro h
  have h1 := h 2 cancelling 1 0 le_rfl (by norm_num) (by norm_num [cancelling])
  refine not_tendsto_atTop_of_tendsto_nhds (a := 0) (tendsto_const_nhds.congr' ?_) h1
  filter_upwards [eventually_ge_atTop 2] with t ht
  obtain ⟨s, rfl⟩ : ∃ s, t = s + 2 := ⟨t - 2, by omega⟩
  rw [show (fun θ => states (fun x v => 2 * x + θ * v) cancelling 0 (s + 2)) = fun _ => 0 from
    funext fun θ => states_cancelling θ s, deriv_const, abs_zero]

end Transformer.RecurrentGradients
