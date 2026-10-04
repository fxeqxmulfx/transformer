/-
# A toy model for the temperature

arXiv:1812.06162, Appendix C, "A Toy Model for the Temperature".

The loss `L(θ) = ½θᵀHθ` of a symmetric `H` (`toyLoss`) has gradient `Hθ`, and the per-example
loss `L_i(θ) = L(θ - c_i)` has gradient `H(θ - c_i)` (`fderiv_toyLoss_sub`).  For `c` of mean
zero and covariance `Σ_c`, the per-example gradient has mean `G = Hθ` and covariance
`Σ = HΣ_cH`, independent of `θ` (`integral_toyGrad`, `covMatrix_toyGrad`).

The paper cites [arXiv:1704.04289] for SGD sampling `θ` of mean zero and covariance `M` with
`MH + HM = (ε/B)Σ`.  One step `θ' = θ - ε(Hθ + ξ)`, of noise `ξ` independent of `θ`, has
covariance `(1 - εH)M(1 - εH) + ε² cov(ξ)` (`covMatrix_sgdStep`), so a stationary `M`, with
`cov(ξ) = Σ/B`, solves `MH + HM = (ε/B)Σ + εHMH` (`lyapunov_of_sgdStep`): the paper's equation
up to `εHMH`, which its continuous-time limit drops.  The equation gives
`2 tr(H^{k+1} M) = (ε/B) tr(H^k Σ)` (`trace_lyapunov`), and `θ` of mean zero gives
`E[(Hθ)ᵀ H^k (Hθ)] = tr(H^{k+2} M)` (`integral_toyGrad_quadForm`).
-/

import Transformer.NoiseScale.Section2_Quadratic

open MeasureTheory ProbabilityTheory
open scoped Matrix

namespace Transformer.NoiseScale

variable {Ω ι : Type*} [Fintype ι] [MeasurableSpace Ω] {P : Measure Ω}

/-- The toy loss `L(θ) = ½θᵀHθ` of Appendix C, least at `θ = 0` for `H ≥ 0`. -/
noncomputable def toyLoss (H : Matrix ι ι ℝ) (θ : ι → ℝ) : ℝ := θ ⬝ᵥ H *ᵥ θ / 2

/-- **Appendix C**: for symmetric `H`, the per-example loss `L_i(θ) = L(θ - c_i)` has gradient
`H(θ - c_i)`, and `L` the gradient `G = Hθ` (`c = 0`). -/
theorem fderiv_toyLoss_sub {H : Matrix ι ι ℝ} (hH : Hᵀ = H) (c θ v : ι → ℝ) :
    fderiv ℝ (fun θ => toyLoss H (θ - c)) θ v = H *ᵥ (θ - c) ⬝ᵥ v := by
  have e : (fun θ => toyLoss H (θ - c)) =
      fun θ : ι → ℝ => 2⁻¹ * ∑ a, (θ a - c a) * ∑ b, H a b * (θ b - c b) := by
    ext θ
    simp only [toyLoss, dotProduct, Matrix.mulVec, Pi.sub_apply]
    ring
  have h1 := fun a => HasFDerivAt.sub_const (c a) (hasFDerivAt_apply (𝕜 := ℝ) a θ)
  have h2 := fun a => HasFDerivAt.fun_sum (u := Finset.univ) fun b _ => (h1 b).const_mul (H a b)
  have hd := (HasFDerivAt.fun_sum fun a (_ : a ∈ Finset.univ) =>
    (h1 a).fun_mul (h2 a)).const_mul (2⁻¹ : ℝ)
  have hs : (θ - c) ⬝ᵥ H *ᵥ v = H *ᵥ (θ - c) ⬝ᵥ v := by
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, hH]
  rw [e, hd.fderiv]
  simp only [dotProduct, Matrix.mulVec, Pi.sub_apply] at hs ⊢
  simp [Finset.sum_add_distrib, hs]
  ring

/-- The hypothesis of `fderiv_toyLoss_sub` is satisfiable: `H = 1`. -/
example := fderiv_toyLoss_sub (H := (1 : Matrix Unit Unit ℝ)) Matrix.transpose_one 0 0 0

/-- **Appendix C**: for `c` of mean zero, the per-example gradient `H(θ - c)` has mean
`G = Hθ`. -/
theorem integral_toyGrad [IsProbabilityMeasure P] {c : Ω → ι → ℝ}
    (hc : ∀ a, Integrable (fun ω => c ω a) P) (h0 : ∀ a, ∫ ω, c ω a ∂P = 0)
    (H : Matrix ι ι ℝ) (θ : ι → ℝ) (a : ι) :
    ∫ ω, (H *ᵥ (θ - c ω)) a ∂P = (H *ᵥ θ) a := by
  have e : ∀ ω, (H *ᵥ (θ - c ω)) a = (H *ᵥ θ) a - (fun b => H a b) ⬝ᵥ c ω := fun ω => by
    rw [Matrix.mulVec_sub]
    rfl
  simp_rw [e]
  rw [integral_sub (integrable_const _) (integrable_dotProduct hc _), integral_dotProduct hc]
  simp [funext h0]

/-- The hypotheses of `integral_toyGrad` are satisfiable: `c = 0`. -/
example (H : Matrix ι ι ℝ) (θ : ι → ℝ) (a : ι) :=
  integral_toyGrad (P := Measure.dirac ()) (c := fun _ => 0) (fun _ => integrable_const 0)
    (fun _ => by simp) H θ a

/-- `cov(AY) = A cov(Y) Aᵀ`. -/
theorem covMatrix_mulVec [IsFiniteMeasure P] {κ : Type*} [Fintype κ] {Y : Ω → ι → ℝ}
    (hY : ∀ a, MemLp (fun ω => Y ω a) 2 P) (A : Matrix κ ι ℝ) :
    covMatrix (fun ω => A *ᵥ Y ω) P = A * covMatrix Y P * Aᵀ := by
  ext a b
  simp only [covMatrix, Matrix.of_apply, Matrix.mulVec, dotProduct, Matrix.mul_apply,
    Matrix.transpose_apply]
  rw [covariance_fun_sum_fun_sum (fun i => (hY i).const_mul (A a i))
    (fun j => (hY j).const_mul (A b j))]
  simp only [covariance_const_mul_left, covariance_const_mul_right, Finset.sum_mul]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => by ring

/-- **Appendix C**: for symmetric `H` and `c` of covariance `Σ_c`, the per-example gradient
`H(θ - c)` has covariance `Σ = HΣ_cH`, independent of `θ`. -/
theorem covMatrix_toyGrad [IsProbabilityMeasure P] {c : Ω → ι → ℝ}
    (hc : ∀ a, MemLp (fun ω => c ω a) 2 P) {H : Matrix ι ι ℝ} (hH : Hᵀ = H) (θ : ι → ℝ) :
    covMatrix (fun ω => H *ᵥ (θ - c ω)) P = H * covMatrix c P * H := by
  have hY : ∀ a, MemLp (fun ω => (θ - c ω) a) 2 P := fun a => (memLp_const (θ a)).sub (hc a)
  rw [covMatrix_mulVec hY, hH]
  congr 2
  ext a b
  simp [covMatrix, covariance_const_sub_left ((hc a).integrable one_le_two),
    covariance_const_sub_right ((hc b).integrable one_le_two)]

/-- The hypotheses of `covMatrix_mulVec` and `covMatrix_toyGrad` are satisfiable: a constant
and `H = 1`. -/
example [DecidableEq ι] (A : Matrix ι ι ℝ) (θ : ι → ℝ) :=
  And.intro (covMatrix_mulVec (P := Measure.dirac ()) (Y := fun _ : Unit => θ)
    (fun a => memLp_const (θ a)) A)
    (covMatrix_toyGrad (P := Measure.dirac ()) (c := fun _ : Unit => θ)
      (fun a => memLp_const (θ a)) (H := (1 : Matrix ι ι ℝ)) Matrix.transpose_one θ)

omit [Fintype ι] in
/-- `cov(X + Y) = cov(X) + cov(Y)` for `X`, `Y` of uncorrelated coordinates. -/
theorem covMatrix_add [IsFiniteMeasure P] {X Y : Ω → ι → ℝ}
    (hX : ∀ a, MemLp (fun ω => X ω a) 2 P) (hY : ∀ a, MemLp (fun ω => Y ω a) 2 P)
    (hXY : ∀ a b, cov[fun ω => X ω a, fun ω => Y ω b; P] = 0) :
    covMatrix (fun ω => X ω + Y ω) P = covMatrix X P + covMatrix Y P := by
  ext a b
  have e : ∀ c, (fun ω => (X ω + Y ω) c) = (fun ω => X ω c) + fun ω => Y ω c := fun c => rfl
  simp only [covMatrix, Matrix.of_apply, Matrix.add_apply, e]
  rw [covariance_add_left (hX a) (hY a) ((hX b).add (hY b)),
    covariance_add_right (hX a) (hX b) (hY b), covariance_add_right (hY a) (hX b) (hY b), hXY,
    covariance_comm (fun ω => Y ω a), hXY]
  ring

/-- One SGD step `θ' = θ - ε(Hθ + ξ)` on the toy loss, of noise `ξ` independent of `θ`, has
covariance `(1 - εH) cov(θ) (1 - εH)ᵀ + ε² cov(ξ)`. -/
theorem covMatrix_sgdStep [IsProbabilityMeasure P] [DecidableEq ι] {θ ξ : Ω → ι → ℝ}
    (hθ : ∀ a, MemLp (fun ω => θ ω a) 2 P) (hξ : ∀ a, MemLp (fun ω => ξ ω a) 2 P)
    (hind : IndepFun θ ξ P) (H : Matrix ι ι ℝ) (ε : ℝ) :
    covMatrix (fun ω => θ ω - ε • (H *ᵥ θ ω + ξ ω)) P =
      (1 - ε • H) * covMatrix θ P * (1 - ε • H)ᵀ + ε ^ 2 • covMatrix ξ P := by
  have e : (fun ω => θ ω - ε • (H *ᵥ θ ω + ξ ω)) =
      fun ω => (1 - ε • H) *ᵥ θ ω + ((-ε) • (1 : Matrix ι ι ℝ)) *ᵥ ξ ω := by
    ext ω a
    simp [Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.neg_mulVec]
    ring
  have hA : ∀ (A : Matrix ι ι ℝ) {Y : Ω → ι → ℝ}, (∀ a, MemLp (fun ω => Y ω a) 2 P) →
      ∀ a, MemLp (fun ω => (A *ᵥ Y ω) a) 2 P := fun A Y hY a => by
    simp only [Matrix.mulVec, dotProduct]
    exact memLp_finsetSum _ fun b _ => (hY b).const_mul (A a b)
  have hm : ∀ (A : Matrix ι ι ℝ) a, Measurable fun v : ι → ℝ => (A *ᵥ v) a := fun A a =>
    ((continuous_apply a).comp (continuous_const.matrix_mulVec continuous_id)).measurable
  rw [e, covMatrix_add (hA (1 - ε • H) hθ) (hA ((-ε) • 1) hξ) fun a b =>
    (hind.comp (hm _ a) (hm _ b)).covariance_eq_zero (hA _ hθ a) (hA _ hξ b),
    covMatrix_mulVec hθ, covMatrix_mulVec hξ]
  simp [Matrix.transpose_smul, smul_smul, sq]

/-- The hypotheses of `covMatrix_sgdStep` are satisfiable: constant `θ` and `ξ`. -/
example [DecidableEq ι] (θ ξ : ι → ℝ) (H : Matrix ι ι ℝ) (ε : ℝ) :=
  covMatrix_sgdStep (P := Measure.dirac ()) (θ := fun _ : Unit => θ) (ξ := fun _ => ξ)
    (fun a => memLp_const (θ a)) (fun a => memLp_const (ξ a)) (indepFun_const_left _ _) H ε

/-- A stationary covariance `M` of the step, with `cov(ξ) = Σ/B` and `ε ≠ 0`, solves
`MH + HM = (ε/B)Σ + εHMH`: the paper's `MH + HM = (ε/B)Σ` up to `εHMH`. -/
theorem lyapunov_of_sgdStep [DecidableEq ι] {H M S : Matrix ι ι ℝ} (hH : Hᵀ = H) {ε B : ℝ}
    (hε : ε ≠ 0) (h : M = (1 - ε • H) * M * (1 - ε • H)ᵀ + ε ^ 2 • (B⁻¹ • S)) :
    M * H + H * M = (ε / B) • S + ε • (H * M * H) := by
  rw [Matrix.transpose_sub, Matrix.transpose_one, Matrix.transpose_smul, hH] at h
  simp only [sub_mul, mul_sub, one_mul, mul_one, smul_mul_assoc, mul_smul_comm] at h
  apply smul_right_injective _ hε
  simp only
  linear_combination (norm := module) h

/-- The hypotheses of `lyapunov_of_sgdStep` are satisfiable: `H = 1`, `ε = B = 1`,
`M = Σ = 1`. -/
example := lyapunov_of_sgdStep (H := (1 : Matrix Unit Unit ℝ)) (M := 1) (S := 1) (B := 1)
  Matrix.transpose_one one_ne_zero (by simp)

/-- `MH + HM = τΣ` gives `2 tr(H^{k+1} M) = τ tr(H^k Σ)`. -/
theorem trace_lyapunov [DecidableEq ι] {H M S : Matrix ι ι ℝ} {τ : ℝ}
    (h : M * H + H * M = τ • S) (k : ℕ) :
    2 * (H ^ (k + 1) * M).trace = τ * (H ^ k * S).trace := by
  have h1 : (H ^ k * (M * H)).trace = (H ^ (k + 1) * M).trace := by
    rw [← Matrix.mul_assoc, Matrix.trace_mul_comm, ← Matrix.mul_assoc, ← pow_succ']
  have h2 : (H ^ k * (H * M)).trace = (H ^ (k + 1) * M).trace := by
    rw [← Matrix.mul_assoc, ← pow_succ]
  have := congrArg (fun X => (H ^ k * X).trace) h
  simp only [Matrix.mul_add, Matrix.trace_add, Matrix.mul_smul, Matrix.trace_smul,
    smul_eq_mul, h1, h2] at this
  linarith

/-- The hypothesis of `trace_lyapunov` is satisfiable: `H = M = Σ = 1`, `τ = 2`. -/
example := trace_lyapunov (H := (1 : Matrix Unit Unit ℝ)) (M := 1) (S := 1) (τ := 2)
  (by rw [two_smul]; simp) 0

/-- For symmetric `H` and `θ` of mean zero and covariance `M`,
`E[(Hθ)ᵀ H^k (Hθ)] = tr(H^{k+2} M)`: `E|G|² = tr(H²M)` and `E[GᵀHG] = tr(H³M)`. -/
theorem integral_toyGrad_quadForm [IsProbabilityMeasure P] [DecidableEq ι] {H : Matrix ι ι ℝ}
    (hH : Hᵀ = H) {θ : Ω → ι → ℝ} (hθ : ∀ a, MemLp (fun ω => θ ω a) 2 P)
    (h0 : ∀ a, ∫ ω, θ ω a ∂P = 0) (k : ℕ) :
    ∫ ω, H *ᵥ θ ω ⬝ᵥ H ^ k *ᵥ H *ᵥ θ ω ∂P = (H ^ (k + 2) * covMatrix θ P).trace := by
  have e : ∀ v : ι → ℝ, H *ᵥ v ⬝ᵥ H ^ k *ᵥ H *ᵥ v = v ⬝ᵥ H ^ (k + 2) *ᵥ v := fun v => by
    rw [Matrix.mulVec_mulVec, dotProduct_comm, Matrix.dotProduct_mulVec,
      ← Matrix.mulVec_transpose, hH, Matrix.mulVec_mulVec, dotProduct_comm, ← Matrix.mul_assoc,
      ← pow_succ', ← pow_succ]
  simp_rw [e]
  rw [integral_quadForm hθ]
  simp [funext h0]

/-- The hypotheses of `integral_toyGrad_quadForm` are satisfiable: `θ = 0`, `H = 1`. -/
example := integral_toyGrad_quadForm (P := Measure.dirac ()) (H := (1 : Matrix Unit Unit ℝ))
  Matrix.transpose_one (θ := fun _ _ => 0) (fun _ => memLp_const 0) (fun _ => by simp) 0

end Transformer.NoiseScale
