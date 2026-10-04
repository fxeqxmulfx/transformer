/-
# The eigenvalue conditions of §2.1 are false

arXiv:1211.5063, §2.1.  For eq. (2) with `|σ'| ≤ γ`, the paper claims that
`λ₁ < 1/γ` is sufficient for the long term contributions to vanish, and that
`λ₁ > 1/γ` is necessary for them to explode, `λ₁` "the absolute value of the
largest eigenvalue of the recurrent weight matrix `W_rec`", which for a real
matrix is the spectral radius of its complexification.  Both fail for tanh
(`γ = 1`) on two units.  `W_rec = [[8, 8], [-8, -8]]` (`Wn`) is nilpotent, so
`λ₁ = 0` (`spectralRadius_Wn`), and `x* = (0, artanh(1/2))` (`xn`) is a fixed
point of eq. (2) for the bias `b = x* - W_rec tanh(x*)` (`bn`) and `W_in = 0`
(`states_xn`).  There `tanh'(x*) = (1, 3/4)`, so every Jacobian along the
orbit is `J = [[8, 6], [-8, -6]]` (`Jn`), with `J² = 2J`: the factor
`∂x_{k+l+1}/∂x_k` is `2^l J` (`norm_factor_xn_succ`), growing like `2^l`
(`exists_mul_two_pow_le`).  Hence neither claim holds
(`not_tendsto_of_spectralRadius_lt`, `not_one_lt_spectralRadius_of_explode`).
The 2-norm conditions of `Section2_Mechanics` are not contradicted:
`‖W_rec‖ γ > 1` (`one_lt_norm_Wn`).
-/

import Transformer.RecurrentGradients.Section2_Mechanics
import Mathlib.Analysis.Normed.Algebra.Spectrum
import Mathlib.Analysis.SpecialFunctions.Artanh

open scoped Matrix Matrix.Norms.L2Operator
open Filter Topology

namespace Transformer.RecurrentGradients

/-- `W_rec = [[8, 8], [-8, -8]]`: nilpotent, so all its eigenvalues are `0`. -/
def Wn : Matrix (Fin 2) (Fin 2) ℝ := !![8, 8; -8, -8]

/-- The state `x* = (0, artanh(1/2))`, at which `tanh' = (1, 3/4)`. -/
noncomputable def xn : EuclideanSpace ℝ (Fin 2) := WithLp.toLp 2 ![0, Real.artanh (1 / 2)]

/-- The bias that makes `x*` a fixed point of eq. (2): `b = x* - W_rec tanh(x*)`. -/
noncomputable def bn : EuclideanSpace ℝ (Fin 2) :=
  xn - Matrix.toEuclideanCLM (𝕜 := ℝ) Wn (act Real.tanh xn)

/-- The Jacobian `W_rec diag(tanh'(x*)) = [[8, 6], [-8, -6]]`. -/
def Jn : Matrix (Fin 2) (Fin 2) ℝ := !![8, 6; -8, -6]

/-- `W_rec² = 0`. -/
theorem Wn_mul_self : Wn * Wn = 0 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Wn, Matrix.mul_apply, Fin.sum_univ_two]

/-- `λ₁ = 0`: the spectral radius of the complexified `W_rec` is `0`. -/
theorem spectralRadius_Wn : spectralRadius ℂ (Wn.map Complex.ofReal) = 0 := by
  have h := spectrum.spectralRadius_pow_le (𝕜 := ℂ) (Wn.map Complex.ofReal) 2 two_ne_zero
  have e : Wn.map Complex.ofReal ^ 2 = 0 :=
    show Wn.map Complex.ofRealHom ^ 2 = 0 by
      rw [sq, ← Matrix.map_mul, Wn_mul_self, Matrix.map_zero _ (map_zero _)]
  rw [e, spectrum.spectralRadius_zero] at h
  exact pow_eq_zero_iff two_ne_zero |>.1 (le_antisymm h zero_le)

variable {κ : Type*} [Fintype κ]

/-- `x*` is a fixed point of eq. (2) with `W_in = 0`, whatever the input. -/
theorem step_xn (v : κ → ℝ) : step Real.tanh Wn 0 bn xn v = xn := by
  simp [step, bn]

theorem states_xn (u : ℕ → κ → ℝ) (t : ℕ) : states (step Real.tanh Wn 0 bn) u xn t = xn := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change step Real.tanh Wn 0 bn (states (step Real.tanh Wn 0 bn) u xn t) _ = xn
    rw [ih, step_xn]

/-- The Jacobian at `x*` is `J`. -/
theorem jacobian_xn : Wn * Matrix.diagonal (fun j => 1 - Real.tanh (xn j) ^ 2) = Jn := by
  have h : Real.tanh (Real.artanh (1 / 2)) = 1 / 2 :=
    Real.tanh_artanh ⟨by norm_num, by norm_num⟩
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Wn, Jn, xn, Matrix.mul_apply, Fin.sum_univ_two, h]

theorem Jn_mul_self : Jn * Jn = (2 : ℝ) • Jn := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Jn, Matrix.mul_apply, Fin.sum_univ_two]

theorem Jn_pow_succ (l : ℕ) : Jn ^ (l + 1) = (2 : ℝ) ^ l • Jn := by
  induction l with
  | zero => simp
  | succ l ih => rw [pow_succ, ih, smul_mul_assoc, Jn_mul_self, smul_smul, pow_succ]

/-- Along the fixed orbit the factor `∂x_{k+l}/∂x_k` is `J^l`. -/
theorem factor_xn (u : ℕ → κ → ℝ) (k l : ℕ) :
    factor Real.tanh Wn 0 bn u xn k l = Matrix.toEuclideanCLM (𝕜 := ℝ) (Jn ^ l) := by
  rw [factor, (hasFDerivAt_states_step (σ' := fun y => 1 - Real.tanh y ^ 2) hasDerivAt_tanh u xn
    k l).fderiv]
  simp only [states_xn, jacobian_xn, transport_const]

theorem norm_factor_xn_succ (u : ℕ → κ → ℝ) (k l : ℕ) :
    ‖factor Real.tanh Wn 0 bn u xn k (l + 1)‖ = 2 ^ l * ‖Jn‖ := by
  rw [factor_xn, Matrix.l2_opNorm_toEuclideanCLM, Jn_pow_succ, norm_smul, norm_pow,
    Real.norm_two]

/-- **The factors grow like `2^l`**: `C 2^l ≤ ‖∂x_{k+l}/∂x_k‖` for every `l`,
with `C > 0`. -/
theorem exists_mul_two_pow_le (u : ℕ → κ → ℝ) (k : ℕ) :
    ∃ C > 0, ∀ l, C * 2 ^ l ≤ ‖factor Real.tanh Wn 0 bn u xn k l‖ := by
  have hJ : 0 < ‖Jn‖ := norm_pos_iff.2 fun h => by
    simpa [Jn] using congrFun (congrFun h 0) 0
  refine ⟨min 1 (‖Jn‖ / 2), lt_min one_pos (half_pos hJ), fun l => ?_⟩
  cases l with
  | zero =>
    rw [factor_xn, pow_zero, pow_zero, map_one, norm_one, mul_one]
    exact min_le_left _ _
  | succ l =>
    rw [norm_factor_xn_succ, pow_succ, ← mul_assoc, mul_comm _ (2 ^ l), mul_assoc]
    exact mul_le_mul_of_nonneg_left
      ((mul_le_mul_of_nonneg_right (min_le_right _ _) zero_le_two).trans (by linarith))
      (by positivity)

/-- The corrected necessary condition holds for `Wn`: its factors are
unbounded, so `‖W_rec‖ γ > 1` with `γ = 1` (`one_lt_of_not_bddAbove`). -/
theorem one_lt_norm_Wn : 1 < ‖Wn‖ := by
  obtain ⟨C, hC, hl⟩ := exists_mul_two_pow_le (κ := Fin 1) (fun _ _ => 0) 0
  have hγ : ∀ y, |1 - Real.tanh y ^ 2| ≤ 1 := fun y => by
    have := Real.tanh_sq_lt_one y
    rw [abs_le]
    constructor <;> nlinarith [sq_nonneg (Real.tanh y)]
  have hu : ¬ BddAbove (Set.range fun l =>
      ‖factor Real.tanh Wn (0 : Matrix (Fin 2) (Fin 1) ℝ) bn (fun _ _ => 0) xn 0 l‖) := by
    rw [not_bddAbove_iff]
    intro r
    obtain ⟨l, hl'⟩ := pow_unbounded_of_one_lt (r / C) one_lt_two
    exact ⟨_, ⟨l, rfl⟩, ((div_lt_iff₀' hC).1 hl').trans_le (hl l)⟩
  simpa using one_lt_of_not_bddAbove hasDerivAt_tanh hγ _ xn 0 hu

/-- **§2.1's sufficient condition for vanishing is false.**  "We first prove that
it is sufficient for `λ₁ < 1/γ`, where `λ₁` is the absolute value of the largest
eigenvalue of the recurrent weight matrix `W_rec`, for the vanishing gradient
problem to occur."  For tanh, `γ = 1`: `W_rec = Wn` has `λ₁ = 0`, and its
factors `∂x_{k+l}/∂x_k` do not even tend to `0`. -/
theorem not_tendsto_of_spectralRadius_lt :
    ¬ ∀ (W : Matrix (Fin 2) (Fin 2) ℝ) (b x₀ : EuclideanSpace ℝ (Fin 2)) (u : ℕ → κ → ℝ)
      (k : ℕ), spectralRadius ℂ (W.map Complex.ofReal) < 1 →
        Tendsto (fun l => ‖factor Real.tanh W 0 b u x₀ k l‖) atTop (𝓝 0) := by
  intro h
  have ht := h Wn bn xn (fun _ _ => 0) 0 (by simp [spectralRadius_Wn])
  obtain ⟨C, hC, hl⟩ := exists_mul_two_pow_le (κ := κ) (fun _ _ => 0) 0
  have hlim : Tendsto (fun l : ℕ => C * 2 ^ l) atTop atTop :=
    (tendsto_pow_atTop_atTop_of_one_lt one_lt_two).const_mul_atTop hC
  exact not_tendsto_nhds_of_tendsto_atTop (tendsto_atTop_mono hl hlim) 0 ht

/-- **§2.1's necessary condition for exploding is false.**  "By inverting this
proof we get the necessary condition for exploding gradients, namely that the
largest eigenvalue `λ₁` is larger than `1/γ`."  For tanh, `γ = 1`: the factors
of `W_rec = Wn` grow like `2^l` while `λ₁ = 0`. -/
theorem not_one_lt_spectralRadius_of_explode :
    ¬ ∀ (W : Matrix (Fin 2) (Fin 2) ℝ) (b x₀ : EuclideanSpace ℝ (Fin 2)) (u : ℕ → κ → ℝ)
      (k : ℕ), (∃ C > 0, ∃ α > 1, ∀ l, C * α ^ l ≤ ‖factor Real.tanh W 0 b u x₀ k l‖) →
        1 < spectralRadius ℂ (W.map Complex.ofReal) := by
  intro h
  obtain ⟨C, hC, hl⟩ := exists_mul_two_pow_le (κ := κ) (fun _ _ => 0) 0
  have := h Wn bn xn (fun _ _ => 0) 0 ⟨C, hC, 2, one_lt_two, hl⟩
  simp [spectralRadius_Wn] at this

end Transformer.RecurrentGradients
