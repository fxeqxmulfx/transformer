/-
# Previous solutions

arXiv:1211.5063, §3.1.  Of an L1 or L2 penalty on the recurrent weights: "the
spectral radius of `W_rec` is probably smaller than 1, from which it follows
that the gradient can not explode (see necessary condition found in section
2.1)", and the model is limited to a regime "where any information inserted in
the model has to die out exponentially fast in time".  Of Echo State Networks:
"Because usually the largest eigenvalue of the recurrent weight is, by
construction, smaller than 1, information fed in to the model has to die out
exponentially fast."  For tanh all three fail on the network of
`Section2_Counterexample`, whose `W_rec` has `λ₁ = 0`.  Its factors
`∂x_{k+l}/∂x_k` grow like `2^l` (`not_not_explode_of_spectralRadius_lt`).  With
`W_in` the identity and the inputs `0`, `x*` stays a fixed point
(`states_xn_zero`), and the derivative of `x_{k+1+l}` in the first coordinate
of `u_{k+1}` (`Section3_Inputs`) is `J^l (1, 0)` (`hasDerivAt_input_xn`), which
grows like `2^l` (`not_tendsto_input_of_spectralRadius_lt`).  Echo State
Networks update `y_t = σ(W_rec y_{t-1} + W_in u_t + b)`, the form of
footnote 1.  There `y* = tanh(x*)` is a fixed point (`statesAfter_yn`) with
Jacobian `K = [[8, 8], [-6, -6]]` (`Kn`), `K² = 2K`, and the derivative of
`y_{k+1+l}` in `u_{k+1}` grows like `2^l` too
(`not_tendsto_inputAfter_of_spectralRadius_lt`).  The counterexample uses a
bias `b = x* - W_rec tanh(x*)` that is not small, which the penalty on the
recurrent weights leaves free, and recurrent weights with `‖W_rec‖ > 1`
(`one_lt_norm_Wn`) whose spectral radius is below 1, all that the paper asks
of the penalty.  In the linear model the claims hold (`Section2_Linear`).
-/

import Transformer.RecurrentGradients.Section2_Counterexample
import Transformer.RecurrentGradients.Section3_Inputs

open scoped Matrix Matrix.Norms.L2Operator
open Filter Topology

namespace Transformer.RecurrentGradients

/-- **The penalty claim of §3.1 is false for tanh**: "the spectral radius of
`W_rec` is probably smaller than 1, from which it follows that the gradient can
not explode".  `λ₁ < 1` does not keep the factors `∂x_{k+l}/∂x_k` of eq. (2)
from growing exponentially. -/
theorem not_not_explode_of_spectralRadius_lt {κ : Type*} [Fintype κ] :
    ¬ ∀ (W : Matrix (Fin 2) (Fin 2) ℝ) (b x₀ : EuclideanSpace ℝ (Fin 2)) (u : ℕ → κ → ℝ)
      (k : ℕ), spectralRadius ℂ (W.map Complex.ofReal) < 1 →
        ¬ ∃ C > 0, ∃ α > 1, ∀ l, C * α ^ l ≤ ‖factor Real.tanh W 0 b u x₀ k l‖ := by
  intro h
  obtain ⟨C, hC, hl⟩ := exists_mul_two_pow_le (κ := κ) (fun _ _ => 0) 0
  exact h Wn bn xn (fun _ _ => 0) 0 (by simp [spectralRadius_Wn]) ⟨C, hC, 2, one_lt_two, hl⟩

/-- A sequence with `f (l + 1) = 2^l c`, `c > 0`, does not tend to `0`. -/
theorem not_tendsto_of_eq_two_pow {f : ℕ → ℝ} {c : ℝ} (hc : 0 < c)
    (hf : ∀ l, f (l + 1) = 2 ^ l * c) : ¬ Tendsto f atTop (𝓝 0) := by
  intro ht
  have h := (tendsto_add_atTop_iff_nat 1).2 ht
  simp only [hf] at h
  exact not_tendsto_nhds_of_tendsto_atTop
    ((tendsto_pow_atTop_atTop_of_one_lt one_lt_two).atTop_mul_const hc) 0 h

/-- The hypotheses of `not_tendsto_of_eq_two_pow` are satisfiable: `f l = 2^l`. -/
example : ¬ Tendsto (fun l : ℕ => (2 : ℝ) ^ l) atTop (𝓝 0) :=
  not_tendsto_of_eq_two_pow two_pos fun l => pow_succ 2 l

variable {κ : Type*} [Fintype κ]

/-- With the inputs `0`, `x*` is a fixed point of eq. (2) whatever `W_in`. -/
theorem states_xn_zero (Win : Matrix (Fin 2) κ ℝ) (t : ℕ) :
    states (step Real.tanh Wn Win bn) 0 xn t = xn := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change step Real.tanh Wn Win bn (states (step Real.tanh Wn Win bn) 0 xn t) 0 = xn
    rw [ih]
    simp [step, bn]

/-- With `W_in` the identity and the inputs `0`, the derivative of `x_{k+1+l}` in
the first coordinate of `u_{k+1}` is `J^l (1, 0)`. -/
theorem hasDerivAt_input_xn (k l : ℕ) :
    HasDerivAt (fun s : ℝ => states (step Real.tanh Wn (1 : Matrix (Fin 2) (Fin 2) ℝ) bn)
      (Function.update 0 (k + 1) ((0 : ℕ → Fin 2 → ℝ) (k + 1) + s • ![1, 0])) xn (k + 1 + l))
      (WithLp.toLp 2 (Jn ^ l *ᵥ ![1, 0])) 0 := by
  have h := hasDerivAt_states_input (F := step Real.tanh Wn (1 : Matrix (Fin 2) (Fin 2) ℝ) bn)
    (u := 0) (x₀ := xn) (k := k) (e := ![1, 0])
    (fun _ => hasFDerivAt_step (σ' := fun y => 1 - Real.tanh y ^ 2) fun _ => hasDerivAt_tanh _)
    (hasDerivAt_step_input _ _ _) l
  rw [← map_transport] at h
  simpa only [states_xn_zero, jacobian_xn, transport_const, Matrix.one_mulVec,
    Matrix.toEuclideanCLM_toLp] using h

/-- **"Any information inserted in the model has to die out exponentially fast in
time" is false for tanh** (§3.1, of the regime `λ₁ < 1` of the penalty): the
derivative of `x_{k+1+l}` in the input `u_{k+1}` need not even tend to `0`. -/
theorem not_tendsto_input_of_spectralRadius_lt :
    ¬ ∀ (W Win : Matrix (Fin 2) (Fin 2) ℝ) (b x₀ : EuclideanSpace ℝ (Fin 2))
      (u : ℕ → Fin 2 → ℝ) (k : ℕ) (e : Fin 2 → ℝ), spectralRadius ℂ (W.map Complex.ofReal) < 1 →
        Tendsto (fun l => ‖deriv (fun s : ℝ => states (step Real.tanh W Win b)
          (Function.update u (k + 1) (u (k + 1) + s • e)) x₀ (k + 1 + l)) 0‖) atTop (𝓝 0) := by
  intro h
  refine not_tendsto_of_eq_two_pow (c := ‖WithLp.toLp 2 (Jn *ᵥ ![1, 0])‖) ?_ (fun l => ?_)
    (h Wn 1 bn xn 0 0 ![1, 0] (by simp [spectralRadius_Wn]))
  · refine norm_pos_iff.2 fun h0 => ?_
    have := congrArg (fun w : EuclideanSpace ℝ (Fin 2) => w 0) h0
    norm_num [Jn, Matrix.mulVec, dotProduct, Fin.sum_univ_two] at this
  · rw [(hasDerivAt_input_xn 0 (l + 1)).deriv, Jn_pow_succ, Matrix.smul_mulVec,
      WithLp.toLp_smul, norm_smul, norm_pow, Real.norm_two]

/-- The state `y* = tanh(x*) = (0, 1/2)`. -/
noncomputable def yn : EuclideanSpace ℝ (Fin 2) := act Real.tanh xn

/-- `W_rec y* + b = x*`: with the input term `0`, the pre-activation at `y*` is `x*`. -/
theorem preactivation_yn : Matrix.toEuclideanCLM (𝕜 := ℝ) Wn yn + bn = xn := by
  simp [yn, bn]

/-- With the inputs `0`, `y*` is a fixed point of
`y_t = tanh(W_rec y_{t-1} + W_in u_t + b)` whatever `W_in`. -/
theorem statesAfter_yn (Win : Matrix (Fin 2) κ ℝ) (t : ℕ) :
    states (stepAfter Real.tanh Wn Win bn) 0 yn t = yn := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change stepAfter Real.tanh Wn Win bn (states (stepAfter Real.tanh Wn Win bn) 0 yn t) 0 = yn
    rw [ih, stepAfter, Matrix.mulVec_zero, WithLp.toLp_zero, add_zero, preactivation_yn]
    rfl

/-- The Jacobian `diag(tanh'(x*)) W_rec = [[8, 8], [-6, -6]]` at `y*`. -/
def Kn : Matrix (Fin 2) (Fin 2) ℝ := !![8, 8; -6, -6]

theorem jacobianAfter_yn : Matrix.diagonal (fun j => 1 - Real.tanh (xn j) ^ 2) * Wn = Kn := by
  have h : Real.tanh (Real.artanh (1 / 2)) = 1 / 2 :=
    Real.tanh_artanh ⟨by norm_num, by norm_num⟩
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Wn, Kn, xn, Matrix.mul_apply, Fin.sum_univ_two, h]

theorem Kn_pow_succ (l : ℕ) : Kn ^ (l + 1) = (2 : ℝ) ^ l • Kn := by
  have hK : Kn * Kn = (2 : ℝ) • Kn := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Kn, Matrix.mul_apply, Fin.sum_univ_two]
  induction l with
  | zero => simp
  | succ l ih => rw [pow_succ, ih, smul_mul_assoc, hK, smul_smul, pow_succ]

/-- With `W_in` the identity and the inputs `0`, the derivative of `y_{k+1+l}` in
the first coordinate of `u_{k+1}` is `K^l diag(tanh'(x*)) (1, 0)`. -/
theorem hasDerivAt_inputAfter_yn (k l : ℕ) :
    HasDerivAt (fun s : ℝ => states (stepAfter Real.tanh Wn (1 : Matrix (Fin 2) (Fin 2) ℝ) bn)
      (Function.update 0 (k + 1) ((0 : ℕ → Fin 2 → ℝ) (k + 1) + s • ![1, 0])) yn (k + 1 + l))
      (WithLp.toLp 2 (Kn ^ l *ᵥ (Matrix.diagonal (fun j => 1 - Real.tanh (xn j) ^ 2) *ᵥ
        ![1, 0]))) 0 := by
  have h := hasDerivAt_states_input
    (F := stepAfter Real.tanh Wn (1 : Matrix (Fin 2) (Fin 2) ℝ) bn) (u := 0) (x₀ := yn) (k := k)
    (e := ![1, 0]) (fun _ => hasFDerivAt_stepAfter hasDerivAt_tanh _ _)
    (hasDerivAt_stepAfter_input hasDerivAt_tanh _ _ _) l
  rw [← map_transport] at h
  simpa only [statesAfter_yn, Pi.zero_apply, Matrix.mulVec_zero, WithLp.toLp_zero, add_zero,
    preactivation_yn, jacobianAfter_yn, transport_const, Matrix.one_mulVec,
    Matrix.toEuclideanCLM_toLp] using h

/-- **The Echo State Network claim of §3.1 is false for tanh**: `λ₁ < 1` does
not make "information fed in to the model" die out; the derivative of `y_{k+1+l}`
in the input `u_{k+1}` of `y_t = tanh(W_rec y_{t-1} + W_in u_t + b)` need not
even tend to `0`. -/
theorem not_tendsto_inputAfter_of_spectralRadius_lt :
    ¬ ∀ (W Win : Matrix (Fin 2) (Fin 2) ℝ) (b y₀ : EuclideanSpace ℝ (Fin 2))
      (u : ℕ → Fin 2 → ℝ) (k : ℕ) (e : Fin 2 → ℝ), spectralRadius ℂ (W.map Complex.ofReal) < 1 →
        Tendsto (fun l => ‖deriv (fun s : ℝ => states (stepAfter Real.tanh W Win b)
          (Function.update u (k + 1) (u (k + 1) + s • e)) y₀ (k + 1 + l)) 0‖) atTop (𝓝 0) := by
  intro h
  refine not_tendsto_of_eq_two_pow (c := ‖WithLp.toLp 2 (Kn *ᵥ
    (Matrix.diagonal (fun j => 1 - Real.tanh (xn j) ^ 2) *ᵥ ![1, 0]))‖) ?_ (fun l => ?_)
    (h Wn 1 bn yn 0 0 ![1, 0] (by simp [spectralRadius_Wn]))
  · refine norm_pos_iff.2 fun h0 => ?_
    have := congrArg (fun w : EuclideanSpace ℝ (Fin 2) => w 0) h0
    norm_num [Kn, xn, Matrix.mulVec, dotProduct, Fin.sum_univ_two, Matrix.diagonal_apply] at this
  · rw [(hasDerivAt_inputAfter_yn 0 (l + 1)).deriv, Kn_pow_succ, Matrix.smul_mulVec,
      WithLp.toLp_smul, norm_smul, norm_pow, Real.norm_two]

end Transformer.RecurrentGradients
