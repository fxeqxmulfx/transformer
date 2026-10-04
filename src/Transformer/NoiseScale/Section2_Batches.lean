/-
# The batch gradient

arXiv:1812.06162, §2.2.  The batch gradient `G_est = (1/B) Σ_{i=1}^B ∇_θ L_{x_i}(θ)`
of eq. (2.1) (`batchGrad`) has the mean `G` and the covariance `Σ/B` of eq. (2.2)
(`integral_batchGrad`, `covMatrix_batchGrad`), `Σ` the per-example covariance
`cov_{x∼ρ}(∇_θ L_x(θ)) = E[∇_θ L_x ∇_θ L_xᵀ] - G Gᵀ` of eq. (2.3) (`covMatrix`,
`covMatrix_apply_eq_sub`).  The per-example gradients are random vectors `X i`,
pairwise independent, of a common covariance `Σ` and a common mean `G`, the true
gradient `∇L = E_ρ[∇_θ L_x]` once `∇` and `E` are exchanged, as the paper does:
weaker hypotheses than its independent `x_i ∼ ρ`.

The moments §2.2 and Appendix A take, of a random vector `Y` of mean `m` and
covariance `C`: `E[WᵀY] = Wᵀm` (`integral_dotProduct`), `E[YᵀHY] = mᵀHm + tr(HC)`
(`integral_quadForm`) and `E[|Y - m|²] = tr C` (`integral_dotProduct_sub`).
-/

import Mathlib.Probability.Moments.Covariance

open MeasureTheory ProbabilityTheory
open scoped Matrix

namespace Transformer.NoiseScale

variable {Ω ι : Type*}

/-- The batch gradient `G_est = (1/B) Σ_i X_i` of eq. (2.1), for the per-example
gradients `X_i = ∇_θ L_{x_i}(θ)`. -/
noncomputable def batchGrad {B : ℕ} (X : Fin B → Ω → ι → ℝ) (ω : Ω) : ι → ℝ :=
  (B : ℝ)⁻¹ • ∑ i, X i ω

theorem batchGrad_apply {B : ℕ} (X : Fin B → Ω → ι → ℝ) (ω : Ω) (a : ι) :
    batchGrad X ω a = (B : ℝ)⁻¹ * ∑ i, X i ω a := by
  simp [batchGrad, Finset.sum_apply]

variable [MeasurableSpace Ω]

/-- The covariance matrix `cov(Y)_{ab} = cov(Y_a, Y_b)` of a random vector `Y`. -/
noncomputable def covMatrix (Y : Ω → ι → ℝ) (P : Measure Ω) : Matrix ι ι ℝ :=
  Matrix.of fun a b => cov[fun ω => Y ω a, fun ω => Y ω b; P]

theorem covMatrix_comm (Y : Ω → ι → ℝ) (P : Measure Ω) (a b : ι) :
    covMatrix Y P a b = covMatrix Y P b a :=
  covariance_comm _ _

variable {P : Measure Ω} {B : ℕ} {X : Fin B → Ω → ι → ℝ} {G : ι → ℝ} {S : Matrix ι ι ℝ}

theorem memLp_batchGrad (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P) (a : ι) :
    MemLp (fun ω => batchGrad X ω a) 2 P := by
  simp_rw [batchGrad_apply]
  exact (memLp_finsetSum _ fun i _ => hX i a).const_mul _

/-- **Eq. (2.2), the mean**: `E[G_est] = G`. -/
theorem integral_batchGrad (hB : B ≠ 0) (hX : ∀ i a, Integrable (fun ω => X i ω a) P)
    (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a) (a : ι) : ∫ ω, batchGrad X ω a ∂P = G a := by
  simp_rw [batchGrad_apply]
  rw [integral_const_mul, integral_finsetSum _ fun i _ => hX i a]
  simp only [hG, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact inv_mul_cancel_left₀ (Nat.cast_ne_zero.2 hB) _

/-- The hypotheses of `integral_batchGrad` are satisfiable: two copies of a constant. -/
example (G : ι → ℝ) (a : ι) :
    ∫ ω, batchGrad (fun (_ : Fin 2) (_ : Unit) => G) ω a ∂Measure.dirac () = G a :=
  integral_batchGrad (X := fun _ _ => G) (G := G) two_ne_zero (fun _ _ => integrable_const _)
    (fun _ _ => by simp) a

variable [IsProbabilityMeasure P]

/-- **Eq. (2.3)**: `Σ = E[Y Yᵀ] - E[Y] E[Y]ᵀ`. -/
theorem covMatrix_apply_eq_sub {Y : Ω → ι → ℝ} (hY : ∀ a, MemLp (fun ω => Y ω a) 2 P)
    (a b : ι) :
    covMatrix Y P a b = ∫ ω, Y ω a * Y ω b ∂P - (∫ ω, Y ω a ∂P) * ∫ ω, Y ω b ∂P :=
  covariance_eq_sub (hY a) (hY b)

/-- The hypothesis of `covMatrix_apply_eq_sub` is satisfiable: a constant. -/
example (G : ι → ℝ) (a b : ι) : covMatrix (fun _ : Unit => G) (Measure.dirac ()) a b =
    ∫ _, G a * G b ∂Measure.dirac () -
      (∫ _, G a ∂Measure.dirac ()) * ∫ _, G b ∂Measure.dirac () :=
  covMatrix_apply_eq_sub (Y := fun _ => G) (fun a => memLp_const (G a)) a b

/-- **Eq. (2.2), the covariance**: `cov(G_est) = Σ/B`. -/
theorem covMatrix_batchGrad (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hS : ∀ i, covMatrix (X i) P = S) :
    covMatrix (batchGrad X) P = (B : ℝ)⁻¹ • S := by
  ext a b
  have h : ∀ i, ∑ j, cov[fun ω => X i ω a, fun ω => X j ω b; P] = S a b := fun i => by
    rw [Finset.sum_eq_single i (fun j _ hji => ?_) (by simp)]
    · exact congrFun (congrFun (hS i) a) b
    · exact ((hind (Ne.symm hji)).comp (measurable_pi_apply a)
        (measurable_pi_apply b)).covariance_eq_zero (hX i a) (hX j b)
  simp only [covMatrix, Matrix.of_apply, batchGrad_apply, Matrix.smul_apply, smul_eq_mul]
  rw [covariance_const_mul_left, covariance_const_mul_right,
    covariance_fun_sum_fun_sum (fun i => hX i a) (fun j => hX j b)]
  simp only [h, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rcases eq_or_ne (B : ℝ) 0 with h0 | h0 <;> simp [h0]

/-- The constant gradient `G` has covariance `0`. -/
theorem covMatrix_const (G : ι → ℝ) : covMatrix (fun _ : Unit => G) (Measure.dirac ()) = 0 := by
  ext a b
  simp [covMatrix]

/-- The hypotheses of `covMatrix_batchGrad` are satisfiable: copies of a constant. -/
example (G : ι → ℝ) : covMatrix (batchGrad fun (_ : Fin 2) (_ : Unit) => G)
    (Measure.dirac ()) = ((2 : ℕ) : ℝ)⁻¹ • 0 :=
  covMatrix_batchGrad (X := fun _ _ => G) (fun _ _ => memLp_const _)
    (fun _ _ _ => indepFun_const_left _ _) fun _ => covMatrix_const G

variable [Fintype ι]

omit [IsProbabilityMeasure P] in
/-- `E[WᵀY] = Wᵀ E[Y]`. -/
theorem integral_dotProduct {Y : Ω → ι → ℝ} (hY : ∀ a, Integrable (fun ω => Y ω a) P)
    (W : ι → ℝ) : ∫ ω, W ⬝ᵥ Y ω ∂P = W ⬝ᵥ fun a => ∫ ω, Y ω a ∂P := by
  simp only [dotProduct]
  rw [integral_finsetSum _ fun a _ => (hY a).const_mul _]
  simp_rw [integral_const_mul]

/-- The hypothesis of `integral_dotProduct` is satisfiable: a constant. -/
example (G W : ι → ℝ) :
    ∫ ω, W ⬝ᵥ (fun _ : Unit => G) ω ∂Measure.dirac () =
      W ⬝ᵥ fun a => ∫ ω, (fun _ : Unit => G) ω a ∂Measure.dirac () :=
  integral_dotProduct (Y := fun _ => G) (fun _ => integrable_const _) W

omit [IsProbabilityMeasure P] in
theorem integrable_dotProduct {Y : Ω → ι → ℝ} (hY : ∀ a, Integrable (fun ω => Y ω a) P)
    (W : ι → ℝ) : Integrable (fun ω => W ⬝ᵥ Y ω) P :=
  integrable_finsetSum _ fun a _ => (hY a).const_mul _

omit [IsProbabilityMeasure P] in
theorem integrable_quadForm {Y : Ω → ι → ℝ} (hY : ∀ a, MemLp (fun ω => Y ω a) 2 P)
    (H : Matrix ι ι ℝ) : Integrable (fun ω => Y ω ⬝ᵥ H *ᵥ Y ω) P := by
  simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
  exact integrable_finsetSum _ fun a _ => integrable_finsetSum _ fun b _ =>
    (((hY a).integrable_mul (hY b)).const_mul (H a b)).congr (.of_forall fun ω => by
      simp only [Pi.mul_apply]; ring)

/-- The hypotheses of `integrable_dotProduct` and `integrable_quadForm` are satisfiable: a
constant. -/
example (G W : ι → ℝ) (H : Matrix ι ι ℝ) :
    Integrable (fun ω => W ⬝ᵥ (fun _ : Unit => G) ω) (Measure.dirac ()) ∧
      Integrable (fun ω => (fun _ : Unit => G) ω ⬝ᵥ H *ᵥ (fun _ : Unit => G) ω)
        (Measure.dirac ()) :=
  ⟨integrable_dotProduct (Y := fun _ => G) (fun _ => integrable_const _) W,
    integrable_quadForm (Y := fun _ => G) (fun a => memLp_const (G a)) H⟩

/-- `E[YᵀHY] = E[Y]ᵀ H E[Y] + tr(H cov(Y))`. -/
theorem integral_quadForm {Y : Ω → ι → ℝ} (hY : ∀ a, MemLp (fun ω => Y ω a) 2 P)
    (H : Matrix ι ι ℝ) :
    ∫ ω, Y ω ⬝ᵥ H *ᵥ Y ω ∂P = (fun a => ∫ ω, Y ω a ∂P) ⬝ᵥ H *ᵥ (fun a => ∫ ω, Y ω a ∂P) +
      (H * covMatrix Y P).trace := by
  have e : ∀ v : ι → ℝ, v ⬝ᵥ H *ᵥ v = ∑ a, ∑ b, H a b * (v a * v b) := fun v => by
    simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
    exact Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => by ring
  have hI : ∀ a b, Integrable (fun ω => Y ω a * Y ω b) P := fun a b =>
    (hY a).integrable_mul (hY b)
  have hC : ∀ a b, covMatrix Y P b a =
      ∫ ω, Y ω a * Y ω b ∂P - (∫ ω, Y ω a ∂P) * ∫ ω, Y ω b ∂P := fun a b => by
    rw [covMatrix_comm]
    exact covMatrix_apply_eq_sub hY a b
  simp_rw [e]
  rw [integral_finsetSum _ fun a _ => integrable_finsetSum _ fun b _ => (hI a b).const_mul (H a b)]
  simp_rw [integral_finsetSum _ fun b _ => (hI _ b).const_mul (H _ b), integral_const_mul]
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, hC, mul_sub, Finset.sum_sub_distrib]
  ring

/-- The hypothesis of `integral_quadForm` is satisfiable: a constant. -/
example (G : ι → ℝ) (H : Matrix ι ι ℝ) :
    ∫ ω, (fun _ : Unit => G) ω ⬝ᵥ H *ᵥ (fun _ : Unit => G) ω ∂Measure.dirac () =
      (fun a => ∫ ω, (fun _ : Unit => G) ω a ∂Measure.dirac ()) ⬝ᵥ
        H *ᵥ (fun a => ∫ ω, (fun _ : Unit => G) ω a ∂Measure.dirac ()) +
          (H * covMatrix (fun _ : Unit => G) (Measure.dirac ())).trace :=
  integral_quadForm (Y := fun _ => G) (fun a => memLp_const (G a)) H

/-- `E[|Y - E[Y]|²] = tr cov(Y)`. -/
theorem integral_dotProduct_sub {Y : Ω → ι → ℝ} (hY : ∀ a, MemLp (fun ω => Y ω a) 2 P) :
    ∫ ω, (Y ω - fun a => ∫ ω, Y ω a ∂P) ⬝ᵥ (Y ω - fun a => ∫ ω, Y ω a ∂P) ∂P =
      (covMatrix Y P).trace := by
  have hM : ∀ a, MemLp (fun ω => Y ω a - ∫ ω, Y ω a ∂P) 2 P := fun a =>
    (hY a).sub (memLp_const _)
  have hI : ∀ a, Integrable (fun ω => (Y ω a - ∫ ω, Y ω a ∂P) *
      (Y ω a - ∫ ω, Y ω a ∂P)) P := fun a => (hM a).integrable_mul (hM a)
  simp only [dotProduct, Pi.sub_apply]
  rw [integral_finsetSum _ fun a _ => hI a]
  rfl

/-- The hypothesis of `integral_dotProduct_sub` is satisfiable: a constant. -/
example (G : ι → ℝ) :
    ∫ ω, ((fun _ : Unit => G) ω - fun a => ∫ ω, (fun _ : Unit => G) ω a ∂Measure.dirac ()) ⬝ᵥ
      ((fun _ : Unit => G) ω - fun a => ∫ ω, (fun _ : Unit => G) ω a ∂Measure.dirac ())
        ∂Measure.dirac () = (covMatrix (fun _ : Unit => G) (Measure.dirac ())).trace :=
  integral_dotProduct_sub (Y := fun _ => G) fun a => memLp_const (G a)

end Transformer.NoiseScale
