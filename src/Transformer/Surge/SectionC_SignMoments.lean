/-
# The sign of a Gaussian batch gradient

arXiv:2405.14578, Appendix C, the proof of Theorem 2.  For `x ∼ N(μ, σ²)`,
`E[sign(x)] = erf(μ/(√2σ))` and `var(sign(x)) = 1 - erf(μ/(√2σ))²`, eqs. (32) and (33)
(`integral_sign_gaussianReal`, `variance_sign_gaussianReal`), from `Transformer.BatchSize`.  The
mean of `B` independent samples of `N(μ, σ²)` is `N(μ, σ²/B)`, eq. (34) (`hasLaw_batchMean`).

So when the per-sample gradient of each parameter `i` is `N(μ_i, σ_i²)`, the update
`V = sign(G_est)` has `E[V]_i = erf(√(B/2)μ_i/σ_i)`, eq. (35) (`meanVec_sign_batchGrad`), and
`cov(V) = diag(1 - erf(√(B/2)μ_i/σ_i)²)`, eq. (36) (`covMatrix_sign_batchGrad`).  Eq. (36) needs
the coordinates of `G_est` independent, which Theorem 2 leaves implicit: here the per-sample
gradients of all the parameters are independent (`indepFun_batchGrad`).
-/

import Transformer.Surge.Section2_Lemma1
import Transformer.BatchSize.Section4_Coefficients

open MeasureTheory ProbabilityTheory
open scoped NNReal

namespace Transformer.Surge

open Transformer.NoiseScale Transformer.BatchSize

/-- `N(μ, σ²)` in the parametrization of `gaussian_sign_mean`, `s = √2σ`. -/
theorem gaussianReal_sqrt_two_mul (μ σ : ℝ) :
    gaussianReal μ (.mk ((√2 * σ) ^ 2 / 2) (by positivity)) =
      gaussianReal μ (.mk (σ ^ 2) (sq_nonneg σ)) := by
  congr 1
  exact NNReal.coe_injective (by simp only [NNReal.coe_mk]; rw [mul_pow,
    Real.sq_sqrt (by norm_num)]; ring)

/-- **Eq. (32)**: for `x ∼ N(μ, σ²)`, `E[sign(x)] = erf(μ/(√2σ))`. -/
theorem integral_sign_gaussianReal (μ : ℝ) {σ : ℝ} (hσ : 0 < σ) :
    ∫ x, Real.sign x ∂gaussianReal μ (.mk (σ ^ 2) (sq_nonneg σ)) =
      errorFunction (μ / (√2 * σ)) := by
  rw [← gaussianReal_sqrt_two_mul]
  exact gaussian_sign_mean μ (√2 * σ) (by positivity)

/-- The hypothesis of `integral_sign_gaussianReal` is satisfiable: `σ = 1`. -/
example (μ : ℝ) := integral_sign_gaussianReal μ one_pos

/-- **Eq. (33)**: for `x ∼ N(μ, σ²)`, `var(sign(x)) = 1 - erf(μ/(√2σ))²`. -/
theorem variance_sign_gaussianReal (μ : ℝ) {σ : ℝ} (hσ : 0 < σ) :
    variance Real.sign (gaussianReal μ (.mk (σ ^ 2) (sq_nonneg σ))) =
      1 - errorFunction (μ / (√2 * σ)) ^ 2 := by
  rw [← gaussianReal_sqrt_two_mul]
  exact gaussian_sign_variance μ (√2 * σ) (by positivity)

/-- The hypothesis of `variance_sign_gaussianReal` is satisfiable: `σ = 1`. -/
example (μ : ℝ) := variance_sign_gaussianReal μ one_pos

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- A sum of independent `N(μ, v)` variables over a finite set `s` is `N(|s|μ, |s|v)`. -/
theorem hasLaw_sum_gaussianReal {κ : Type*} {X : κ → Ω → ℝ} {μ : ℝ} {v : ℝ≥0}
    (hind : iIndepFun X P) (hmeas : ∀ k, Measurable (X k))
    (hX : ∀ k, HasLaw (X k) (gaussianReal μ v) P) (s : Finset κ) :
    HasLaw (fun ω => ∑ k ∈ s, X k ω) (gaussianReal (s.card * μ) (s.card * v)) P := by
  classical
  induction s using Finset.induction_on with
  | empty =>
    simp only [Finset.sum_empty, Finset.card_empty, Nat.cast_zero, zero_mul, gaussianReal_zero_var]
    exact ⟨aemeasurable_const, by simp⟩
  | insert a s ha ih =>
    have hI : IndepFun (X a) (fun ω => ∑ k ∈ s, X k ω) P := by
      have := (hind.indepFun_finsetSum_of_notMem hmeas ha).symm
      rwa [Finset.sum_fn] at this
    have hsum : (fun ω => ∑ k ∈ insert a s, X k ω) = X a + fun ω => ∑ k ∈ s, X k ω := by
      funext ω
      simp [Finset.sum_insert ha]
    rw [hsum, Finset.card_insert_of_notMem ha]
    refine ⟨(hX a).aemeasurable.add ih.aemeasurable, ?_⟩
    rw [gaussianReal_add_gaussianReal_of_indepFun hI (hX a) ih]
    congr 1 <;> push_cast <;> ring

/-- **Eq. (34)**: the mean of `B` independent samples of `N(μ, σ²)` is `N(μ, σ²/B)`. -/
theorem hasLaw_batchMean {B : ℕ} (hB : B ≠ 0) {X : Fin B → Ω → ℝ} {μ σ : ℝ}
    (hind : iIndepFun X P) (hmeas : ∀ k, Measurable (X k))
    (hX : ∀ k, HasLaw (X k) (gaussianReal μ (.mk (σ ^ 2) (sq_nonneg σ))) P) :
    HasLaw (fun ω => (B : ℝ)⁻¹ * ∑ k, X k ω) (gradientNoiseLaw B μ σ) P := by
  have h := gaussianReal_const_mul (hasLaw_sum_gaussianReal hind hmeas hX Finset.univ) (B : ℝ)⁻¹
  have hB' : (B : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hB
  rw [Finset.card_univ, Fintype.card_fin] at h
  unfold gradientNoiseLaw
  convert h using 2
  · field_simp
  · exact NNReal.coe_injective (by push_cast; field_simp)

/-- The hypotheses of `hasLaw_batchMean` are satisfiable: one standard Gaussian. -/
example : HasLaw (fun x : ℝ => ((1 : ℕ) : ℝ)⁻¹ * ∑ _ : Fin 1, x) (gradientNoiseLaw 1 0 1)
    (gaussianReal 0 1) :=
  hasLaw_batchMean one_ne_zero (X := fun _ x => x) iIndepFun.of_subsingleton
    (fun _ => measurable_id) (fun _ => ⟨aemeasurable_id, by simp⟩)

/-- The hypotheses of `hasLaw_sum_gaussianReal` are satisfiable: one standard Gaussian. -/
example := hasLaw_sum_gaussianReal (P := gaussianReal 0 1) (X := fun (_ : Unit) x => x) (μ := 0)
  (v := 1)
  iIndepFun.of_subsingleton (fun _ => measurable_id) (fun _ => ⟨aemeasurable_id, by simp⟩)
  Finset.univ

variable {ι : Type*} {B : ℕ} {X : Fin B → Ω → ι → ℝ} {μ σ : ι → ℝ}
  (hind : iIndepFun (fun p : Fin B × ι => fun ω => X p.1 ω p.2) P)
  (hmeas : ∀ k i, Measurable fun ω => X k ω i)
  (hX : ∀ k i, HasLaw (fun ω => X k ω i) (gaussianReal (μ i) (.mk (σ i ^ 2) (sq_nonneg _))) P)
include hind hmeas hX

/-- **Eq. (34)**, for each coordinate: when the per-sample gradients `X_{k,i}` of all the
parameters are independent, with `X_{k,i} ∼ N(μ_i, σ_i²)`, then `G_est,i ∼ N(μ_i, σ_i²/B)`. -/
theorem hasLaw_batchGrad (hB : B ≠ 0) (i : ι) :
    HasLaw (fun ω => batchGrad X ω i) (gradientNoiseLaw B (μ i) (σ i)) P := by
  simp only [batchGrad_apply]
  exact hasLaw_batchMean hB (hind.precomp (g := fun k : Fin B => (k, i))
    fun _ _ h => (Prod.mk.inj h).1) (fun k => hmeas k i) (fun k => hX k i)

/-- The hypotheses of `hasLaw_batchGrad` are satisfiable: independent `N(0, 1)` gradients. -/
example := hasLaw_batchGrad
  (P := Measure.pi fun _ : Fin 1 × Fin 2 => gaussianReal 0 (.mk (1 ^ 2) (sq_nonneg 1)))
  (X := fun k ω i => ω (k, i)) (μ := fun _ => 0) (σ := fun _ => 1)
  (iIndepFun_pi (X := fun _ x => x) fun _ => aemeasurable_id)
  (fun k i => measurable_pi_apply (k, i))
  (fun k i => (measurePreserving_eval (fun _ => _) (k, i)).hasLaw) one_ne_zero 0

omit [IsProbabilityMeasure P] hX in
/-- With independent per-sample gradients, distinct coordinates of `G_est` are independent. -/
theorem indepFun_batchGrad {i j : ι} (hij : i ≠ j) :
    IndepFun (fun ω => batchGrad X ω i) (fun ω => batchGrad X ω j) P := by
  classical
  have h := hind.indepFun_finset (Finset.univ ×ˢ {i}) (Finset.univ ×ˢ {j})
    (Finset.disjoint_product.2 (Or.inr (Finset.disjoint_singleton.2 hij))) fun p => hmeas p.1 p.2
  have hφ (a : ι) : Measurable fun z : (Finset.univ ×ˢ {a} : Finset (Fin B × ι)) → ℝ =>
      (B : ℝ)⁻¹ * ∑ k, z ⟨(k, a), by simp⟩ := by fun_prop
  simp only [batchGrad_apply]
  exact h.comp (hφ i) (hφ j)

/-- The hypotheses of `indepFun_batchGrad` are satisfiable: independent `N(0, 1)` gradients. -/
example := indepFun_batchGrad
  (P := Measure.pi fun _ : Fin 1 × Fin 2 => gaussianReal 0 1) (X := fun k ω i => ω (k, i))
  (iIndepFun_pi (X := fun _ x => x) fun _ => aemeasurable_id)
  (fun k i => measurable_pi_apply (k, i)) (i := 0) (j := 1) (by decide)

/-- `sign(G_est,i)` has a second moment. -/
theorem memLp_sign_batchGrad (hB : B ≠ 0) (i : ι) :
    MemLp (fun ω => Real.sign (batchGrad X ω i)) 2 P :=
  MemLp.of_bound (measurable_real_sign.comp_aemeasurable
    (hasLaw_batchGrad hind hmeas hX hB i).aemeasurable).aestronglyMeasurable 1
    (ae_of_all _ fun _ => by simpa [Real.norm_eq_abs] using sign_abs_le_one _)

/-- The hypotheses of `memLp_sign_batchGrad` are satisfiable: independent `N(0, 1)` gradients. -/
example := memLp_sign_batchGrad
  (P := Measure.pi fun _ : Fin 1 × Fin 2 => gaussianReal 0 (.mk (1 ^ 2) (sq_nonneg 1)))
  (X := fun k ω i => ω (k, i)) (μ := fun _ => 0) (σ := fun _ => 1)
  (iIndepFun_pi (X := fun _ x => x) fun _ => aemeasurable_id)
  (fun k i => measurable_pi_apply (k, i))
  (fun k i => (measurePreserving_eval (fun _ => _) (k, i)).hasLaw) one_ne_zero 0

/-- **Eq. (35)**: `E[sign(G_est)]_i = erf(√(B/2)μ_i/σ_i)`. -/
theorem meanVec_sign_batchGrad (hB : B ≠ 0) (hσ : ∀ i, 0 < σ i) :
    meanVec (fun ω i => Real.sign (batchGrad X ω i)) P = fun i => signResponse B (σ i) (μ i) := by
  funext i
  exact ((hasLaw_batchGrad hind hmeas hX hB i).integral_comp
    measurable_real_sign.aestronglyMeasurable).trans
    (signResponse_eq_mean B (μ i) (σ i) (Nat.pos_of_ne_zero hB) (hσ i))

/-- The hypotheses of `meanVec_sign_batchGrad` are satisfiable: independent `N(0, 1)` gradients. -/
example := meanVec_sign_batchGrad
  (P := Measure.pi fun _ : Fin 1 × Fin 2 => gaussianReal 0 (.mk (1 ^ 2) (sq_nonneg 1)))
  (X := fun k ω i => ω (k, i)) (μ := fun _ => 0) (σ := fun _ => 1)
  (iIndepFun_pi (X := fun _ x => x) fun _ => aemeasurable_id)
  (fun k i => measurable_pi_apply (k, i))
  (fun k i => (measurePreserving_eval (fun _ => _) (k, i)).hasLaw) one_ne_zero fun _ => one_pos

/-- **Eq. (36)**: `cov(sign(G_est)) = diag(1 - erf(√(B/2)μ_i/σ_i)²)`. -/
theorem covMatrix_sign_batchGrad [DecidableEq ι] (hB : B ≠ 0) (hσ : ∀ i, 0 < σ i) :
    covMatrix (fun ω i => Real.sign (batchGrad X ω i)) P =
      Matrix.diagonal fun i => 1 - signResponse B (σ i) (μ i) ^ 2 := by
  ext i j
  have hL := memLp_sign_batchGrad hind hmeas hX hB
  rcases eq_or_ne i j with rfl | hij
  · rw [covMatrix, Matrix.of_apply, Matrix.diagonal_apply_eq, covariance_self (hL i).aemeasurable]
    have hG := hasLaw_batchGrad hind hmeas hX hB i
    have h := variance_map (X := Real.sign) measurable_real_sign.aemeasurable hG.aemeasurable
    rw [hG.map_eq] at h
    exact h.symm.trans (signResponse_variance B (μ i) (σ i) (Nat.pos_of_ne_zero hB) (hσ i))
  · rw [covMatrix, Matrix.of_apply, Matrix.diagonal_apply_ne _ hij]
    exact ((indepFun_batchGrad hind hmeas hij).comp measurable_real_sign
      measurable_real_sign).covariance_eq_zero (hL i) (hL j)

/-- The hypotheses of `covMatrix_sign_batchGrad` are satisfiable: independent `N(0, 1)`
gradients. -/
example := covMatrix_sign_batchGrad
  (P := Measure.pi fun _ : Fin 1 × Fin 2 => gaussianReal 0 (.mk (1 ^ 2) (sq_nonneg 1)))
  (X := fun k ω i => ω (k, i)) (μ := fun _ => 0) (σ := fun _ => 1)
  (iIndepFun_pi (X := fun _ x => x) fun _ => aemeasurable_id)
  (fun k i => measurable_pi_apply (k, i))
  (fun k i => (measurePreserving_eval (fun _ => _) (k, i)).hasLaw) one_ne_zero fun _ => one_pos

end Transformer.Surge
