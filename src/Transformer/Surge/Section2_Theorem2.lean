/-
# The optimal learning rate of the sign update

arXiv:2405.14578, §2.1, Theorem 2, proved in Appendix C.  Adam is approximated by the sign
update `V = sign(G_est)` of eq. (5).  When the per-sample gradient of each parameter `i` is
`N(μ_i, σ_i²)`, the true gradient is `G = μ`, eq. (37), and Appendix C gives `E[V]_i = 𝓔_i(B)`
and `cov(V) = diag(1 - 𝓔_i(B)²)`, with `𝓔_i(B) = erf(√(B/2)μ_i/σ_i)`, eq. (38) (`signMean`).
Lemma 1 then gives the optimal learning rate
`ε_opt = Σ𝓔_iμ_i/(Σ(1 - 𝓔_i²)H_ii + ΣΣ𝓔_i𝓔_jH_ij)` of eq. (9) (`lrOpt_sign_batchGrad`) and
the loss improvement `ΔL_opt = ½ΣΣ𝓔_i𝓔_jμ_iμ_j/(Σ(1 - 𝓔_i²)H_ii + ΣΣ𝓔_i𝓔_jH_ij)` of eq. (8)
(`integral_lossDrop_sign_batchGrad`).

As Lemma 1 does, Theorem 2 leaves implicit that the denominator is positive.  It is when the
Hessian is positive semidefinite and nonzero (`signDen_pos`), and then no learning rate improves
the loss more than eq. (8) (`integral_lossDrop_sign_le`).
-/

import Transformer.Surge.SectionC_SignMoments

open MeasureTheory ProbabilityTheory
open scoped Matrix

namespace Transformer.Surge

open Transformer.NoiseScale Transformer.BatchSize

/-- `𝓔(B) = erf(√(B/2)μ/σ)`, eq. (38), for a real batch size `B`. -/
noncomputable def signMean (μ σ B : ℝ) : ℝ := errorFunction (√(B / 2) * μ / σ)

variable {ι : Type*} [Fintype ι]

/-- The denominator `Σ(1 - 𝓔_i²)H_ii + ΣΣ𝓔_i𝓔_jH_ij` of eqs. (8) and (9). -/
noncomputable def signDen (E : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ :=
  ∑ i, (1 - E i ^ 2) * H i i + ∑ i, ∑ j, E i * E j * H i j

/-- The learning rate `Σ𝓔_iμ_i/(Σ(1 - 𝓔_i²)H_ii + ΣΣ𝓔_i𝓔_jH_ij)` of eq. (9). -/
noncomputable def lrSign (E μ : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ :=
  (∑ i, E i * μ i) / signDen E H

/-- The loss improvement `½ΣΣ𝓔_i𝓔_jμ_iμ_j/(Σ(1 - 𝓔_i²)H_ii + ΣΣ𝓔_i𝓔_jH_ij)` of eq. (8). -/
noncomputable def lossDropSign (E μ : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ :=
  1 / 2 * ((∑ i, ∑ j, E i * E j * μ i * μ j) / signDen E H)

/-- `ΣΣ𝓔_i𝓔_jH_ij = 𝓔ᵀH𝓔`. -/
theorem sum_sum_mul_eq (E : ι → ℝ) (H : Matrix ι ι ℝ) :
    ∑ i, ∑ j, E i * E j * H i j = E ⬝ᵥ H *ᵥ E := by
  simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- The denominator `tr(H cov(V)) + E[V]ᵀHE[V]` of eq. (6), for an update of mean `𝓔` and
covariance `diag(1 - 𝓔_i²)`, is that of eqs. (8) and (9). -/
theorem trace_mul_diagonal_add [DecidableEq ι] (E : ι → ℝ) (H : Matrix ι ι ℝ) :
    (H * Matrix.diagonal fun i => 1 - E i ^ 2).trace + E ⬝ᵥ H *ᵥ E = signDen E H := by
  rw [signDen, sum_sum_mul_eq]
  congr 1
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_diagonal]
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

/-- Eq. (6), for an update of mean `𝓔` and covariance `diag(1 - 𝓔_i²)`, is eq. (9). -/
theorem lrOpt_diagonal [DecidableEq ι] (E μ : ι → ℝ) (H : Matrix ι ι ℝ) :
    lrOpt μ E H (Matrix.diagonal fun i => 1 - E i ^ 2) = lrSign E μ H := by
  rw [lrOpt, trace_mul_diagonal_add, lrSign, dotProduct_comm]
  rfl

/-- Eq. (7), for an update of mean `𝓔`, at the learning rate of eq. (9), is eq. (8). -/
theorem dotProduct_div_two_mul_lrSign (E μ : ι → ℝ) (H : Matrix ι ι ℝ) :
    μ ⬝ᵥ E / 2 * lrSign E μ H = lossDropSign E μ H := by
  have h : ∑ i, ∑ j, E i * E j * μ i * μ j = (∑ i, E i * μ i) * ∑ j, E j * μ j := by
    rw [Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  rw [lossDropSign, lrSign, h, dotProduct_comm]
  simp only [dotProduct]
  ring

/-- The denominator of eqs. (8) and (9) is positive when every `|𝓔_i| < 1` and the Hessian is
positive semidefinite and nonzero. -/
theorem signDen_pos {E : ι → ℝ} (hE : ∀ i, |E i| < 1) {H : Matrix ι ι ℝ} (hH : H.PosSemidef)
    (hH0 : H ≠ 0) : 0 < signDen E H := by
  obtain ⟨i, hi⟩ : ∃ i, 0 < H i i := by
    by_contra h
    simp only [not_exists, not_lt] at h
    exact hH0 (hH.trace_eq_zero_iff.1
      (Finset.sum_eq_zero fun i _ => (h i).antisymm hH.diag_nonneg))
  have h1 (j : ι) : 0 < 1 - E j ^ 2 := sub_pos.2 ((sq_lt_one_iff_abs_lt_one _).2 (hE j))
  rw [signDen, sum_sum_mul_eq]
  refine add_pos_of_pos_of_nonneg (Finset.sum_pos' (fun j _ => mul_nonneg (h1 j).le
    hH.diag_nonneg) ⟨i, Finset.mem_univ _, mul_pos (h1 i) hi⟩) ?_
  simpa using hH.dotProduct_mulVec_nonneg E

/-- The hypotheses of `signDen_pos` are satisfiable: `𝓔 = 0`, `H = 1`. -/
example := signDen_pos (E := fun _ : Fin 2 => 0) (fun _ => by simp) Matrix.PosSemidef.one
  one_ne_zero

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {B : ℕ} {X : Fin B → Ω → ι → ℝ} {μ σ : ι → ℝ}
  (hind : iIndepFun (fun p : Fin B × ι => fun ω => X p.1 ω p.2) P)
  (hmeas : ∀ k i, Measurable fun ω => X k ω i)
  (hX : ∀ k i, HasLaw (fun ω => X k ω i) (gaussianReal (μ i) (.mk (σ i ^ 2) (sq_nonneg _))) P)
include hind hmeas hX

/-- **Theorem 2**, eq. (9): when the per-sample gradients `X_{k,i}` of all the parameters are
independent, with `X_{k,i} ∼ N(μ_i, σ_i²)`, Lemma 1's learning rate for the update `sign(G_est)`
is `ε_opt = Σ𝓔_iμ_i/(Σ(1 - 𝓔_i²)H_ii + ΣΣ𝓔_i𝓔_jH_ij)`. -/
theorem lrOpt_sign_batchGrad (hB : B ≠ 0) (hσ : ∀ i, 0 < σ i) (H : Matrix ι ι ℝ) :
    lrOpt μ (meanVec (fun ω i => Real.sign (batchGrad X ω i)) P) H
      (covMatrix (fun ω i => Real.sign (batchGrad X ω i)) P) =
      lrSign (fun i => signMean (μ i) (σ i) B) μ H := by
  classical
  rw [meanVec_sign_batchGrad hind hmeas hX hB hσ, covMatrix_sign_batchGrad hind hmeas hX hB hσ]
  exact lrOpt_diagonal _ μ H

/-- The hypotheses of `lrOpt_sign_batchGrad` are satisfiable: independent `N(0, 1)` gradients. -/
example := lrOpt_sign_batchGrad
  (P := Measure.pi fun _ : Fin 1 × Fin 2 => gaussianReal 0 (.mk (1 ^ 2) (sq_nonneg 1)))
  (X := fun k ω i => ω (k, i)) (μ := fun _ => 0) (σ := fun _ => 1)
  (iIndepFun_pi (X := fun _ x => x) fun _ => aemeasurable_id)
  (fun k i => measurable_pi_apply (k, i))
  (fun k i => (measurePreserving_eval (fun _ => _) (k, i)).hasLaw) one_ne_zero
  (fun _ => one_pos) 1

/-- **Theorem 2**, eq. (8): at that learning rate the expected loss improvement is
`ΔL_opt = ½ΣΣ𝓔_i𝓔_jμ_iμ_j/(Σ(1 - 𝓔_i²)H_ii + ΣΣ𝓔_i𝓔_jH_ij)`, in the model (30). -/
theorem integral_lossDrop_sign_batchGrad (hB : B ≠ 0) (hσ : ∀ i, 0 < σ i) (L : ℝ)
    (H : Matrix ι ι ℝ) :
    L - ∫ ω, quadModel L μ H (lrOpt μ (meanVec (fun ω i => Real.sign (batchGrad X ω i)) P) H
      (covMatrix (fun ω i => Real.sign (batchGrad X ω i)) P))
      (fun i => Real.sign (batchGrad X ω i)) ∂P =
      lossDropSign (fun i => signMean (μ i) (σ i) B) μ H := by
  rw [integral_lossDrop_lrOpt (memLp_sign_batchGrad hind hmeas hX hB) L μ H,
    lrOpt_sign_batchGrad hind hmeas hX hB hσ, meanVec_sign_batchGrad hind hmeas hX hB hσ]
  exact dotProduct_div_two_mul_lrSign _ μ H

/-- The hypotheses of `integral_lossDrop_sign_batchGrad` are satisfiable: independent `N(0, 1)`
gradients. -/
example (L : ℝ) := integral_lossDrop_sign_batchGrad
  (P := Measure.pi fun _ : Fin 1 × Fin 2 => gaussianReal 0 (.mk (1 ^ 2) (sq_nonneg 1)))
  (X := fun k ω i => ω (k, i)) (μ := fun _ => 0) (σ := fun _ => 1)
  (iIndepFun_pi (X := fun _ x => x) fun _ => aemeasurable_id)
  (fun k i => measurable_pi_apply (k, i))
  (fun k i => (measurePreserving_eval (fun _ => _) (k, i)).hasLaw) one_ne_zero
  (fun _ => one_pos) L 1

/-- **Theorem 2**: when the Hessian is positive semidefinite and nonzero, no learning rate
improves the loss more than eq. (8), so eq. (9) is the optimal one, in the model (30). -/
theorem integral_lossDrop_sign_le (hB : B ≠ 0) (hσ : ∀ i, 0 < σ i) (L : ℝ)
    {H : Matrix ι ι ℝ} (hH : H.PosSemidef) (hH0 : H ≠ 0) (ε : ℝ) :
    L - ∫ ω, quadModel L μ H ε (fun i => Real.sign (batchGrad X ω i)) ∂P ≤
      lossDropSign (fun i => signMean (μ i) (σ i) B) μ H := by
  classical
  refine (integral_lossDrop_le (memLp_sign_batchGrad hind hmeas hX hB) L ?_ ε).trans_eq
    (integral_lossDrop_sign_batchGrad hind hmeas hX hB hσ L H)
  rw [meanVec_sign_batchGrad hind hmeas hX hB hσ, covMatrix_sign_batchGrad hind hmeas hX hB hσ,
    trace_mul_diagonal_add]
  exact signDen_pos (fun _ => errorFunction_abs_lt_one _) hH hH0

/-- The hypotheses of `integral_lossDrop_sign_le` are satisfiable: independent `N(0, 1)`
gradients, `H = 1`. -/
example (L ε : ℝ) := integral_lossDrop_sign_le
  (P := Measure.pi fun _ : Fin 1 × Fin 2 => gaussianReal 0 (.mk (1 ^ 2) (sq_nonneg 1)))
  (X := fun k ω i => ω (k, i)) (μ := fun _ => 0) (σ := fun _ => 1)
  (iIndepFun_pi (X := fun _ x => x) fun _ => aemeasurable_id)
  (fun k i => measurable_pi_apply (k, i))
  (fun k i => (measurePreserving_eval (fun _ => _) (k, i)).hasLaw) one_ne_zero
  (fun _ => one_pos) L Matrix.PosSemidef.one one_ne_zero ε

end Transformer.Surge
