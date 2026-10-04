/-
# Unbiased estimates of the simple noise scale

arXiv:1812.06162, Appendix A.1.  The squared norm of the batch gradient has the mean
`E[|G_est|²] = |G|² + tr(Σ)/B`, eq. (A.1) (`integral_dotProduct_self_batchGrad`).  From the
squared norms at two batch sizes `B_small ≠ B_big`, eq. (A.2) builds
`|𝒢|² = (B_big|G_big|² - B_small|G_small|²)/(B_big - B_small)` and
`𝒮 = (|G_small|² - |G_big|²)/(1/B_small - 1/B_big)` (`gradSqEst`, `traceEst`), unbiased for
`|G|²` and `tr(Σ)` by eq. (A.1) alone (`integral_gradSqEst`, `integral_traceEst`), whatever
the joint law of the two batches.  In the paper's data-parallel use the local batch of
`B_small` examples is part of the global one of `B_big` (`integral_estimates_batchGrad`).

For `B_small = 1` and `B_big = n`, `𝒮` is `n/(n-1)` times `|G_small|² - |G_big|²`
(`traceEst_one`), and with the mean of the `n` per-example squared norms as `|G_small|²` it
is their sample variance with Bessel's correction, the footnote after eq. (A.2)
(`traceEst_one_mean`).
-/

import Transformer.NoiseScale.Section2_Batches

open MeasureTheory ProbabilityTheory
open scoped Matrix

namespace Transformer.NoiseScale

variable {Ω ι : Type*} [Fintype ι] [MeasurableSpace Ω] {P : Measure Ω}

/-- **Eq. (A.1)**: `E[|G_est|²] = |G|² + tr(Σ)/B`. -/
theorem integral_dotProduct_self_batchGrad [IsProbabilityMeasure P] {B : ℕ}
    {X : Fin B → Ω → ι → ℝ} {G : ι → ℝ} {S : Matrix ι ι ℝ} (hB : B ≠ 0)
    (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a)
    (hS : ∀ i, covMatrix (X i) P = S) :
    ∫ ω, batchGrad X ω ⬝ᵥ batchGrad X ω ∂P = G ⬝ᵥ G + S.trace / B := by
  classical
  have hm : (fun a => ∫ ω, batchGrad X ω a ∂P) = G :=
    funext (integral_batchGrad hB (fun i a => (hX i a).integrable (q := 2) (by norm_num)) hG)
  have h := integral_quadForm (memLp_batchGrad hX) (1 : Matrix ι ι ℝ)
  simp only [Matrix.one_mulVec, one_mul] at h
  rw [h, hm, covMatrix_batchGrad hX hind hS, Matrix.trace_smul, smul_eq_mul, inv_mul_eq_div]

/-- The hypotheses of `integral_dotProduct_self_batchGrad` are satisfiable: copies of a
constant. -/
example (G : ι → ℝ) :
    ∫ ω, batchGrad (fun (_ : Fin 2) (_ : Unit) => G) ω ⬝ᵥ
      batchGrad (fun (_ : Fin 2) (_ : Unit) => G) ω ∂Measure.dirac () =
        G ⬝ᵥ G + (0 : Matrix ι ι ℝ).trace / (2 : ℕ) :=
  integral_dotProduct_self_batchGrad (X := fun _ _ => G) two_ne_zero
    (fun _ _ => memLp_const _) (fun _ _ _ => indepFun_const_left _ _) (fun _ _ => by simp)
    (fun _ => covMatrix_const G)

/-- The estimate `|𝒢|² = (B_big|G_big|² - B_small|G_small|²)/(B_big - B_small)` of `|G|²`,
eq. (A.2), from the squared norms `|G_small|²`, `|G_big|²` of two batches. -/
noncomputable def gradSqEst (bs bb ns nb : ℝ) : ℝ := (bb * nb - bs * ns) / (bb - bs)

/-- The estimate `𝒮 = (|G_small|² - |G_big|²)/(1/B_small - 1/B_big)` of `tr(Σ)`, eq. (A.2). -/
noncomputable def traceEst (bs bb ns nb : ℝ) : ℝ := (ns - nb) / (1 / bs - 1 / bb)

/-- **Eq. (A.2)**: `E[|𝒢|²] = |G|²`, given eq. (A.1) at both batch sizes. -/
theorem integral_gradSqEst {ns nb : Ω → ℝ} (hns : Integrable ns P) (hnb : Integrable nb P)
    {bs bb g t : ℝ} (hbs : bs ≠ 0) (hbb : bb ≠ 0) (hne : bs ≠ bb)
    (hs : ∫ ω, ns ω ∂P = g + t / bs) (hb : ∫ ω, nb ω ∂P = g + t / bb) :
    ∫ ω, gradSqEst bs bb (ns ω) (nb ω) ∂P = g := by
  have : bb - bs ≠ 0 := sub_ne_zero.2 (Ne.symm hne)
  unfold gradSqEst
  rw [integral_div, integral_sub (hnb.const_mul bb) (hns.const_mul bs), integral_const_mul,
    integral_const_mul, hs, hb]
  field_simp
  ring

/-- The hypotheses of `integral_gradSqEst` are satisfiable: constant squared norms. -/
example (g t : ℝ) : ∫ ω, gradSqEst 1 2 ((fun _ : Unit => g + t / 1) ω)
    ((fun _ : Unit => g + t / 2) ω) ∂Measure.dirac () = g :=
  integral_gradSqEst (t := t) (integrable_const _) (integrable_const _) one_ne_zero
    two_ne_zero (by norm_num) (by simp) (by simp)

/-- **Eq. (A.2)**: `E[𝒮] = tr(Σ)`, given eq. (A.1) at both batch sizes. -/
theorem integral_traceEst {ns nb : Ω → ℝ} (hns : Integrable ns P) (hnb : Integrable nb P)
    {bs bb g t : ℝ} (hbs : bs ≠ 0) (hbb : bb ≠ 0) (hne : bs ≠ bb)
    (hs : ∫ ω, ns ω ∂P = g + t / bs) (hb : ∫ ω, nb ω ∂P = g + t / bb) :
    ∫ ω, traceEst bs bb (ns ω) (nb ω) ∂P = t := by
  have : bb - bs ≠ 0 := sub_ne_zero.2 (Ne.symm hne)
  unfold traceEst
  rw [integral_div, integral_sub hns hnb, hs, hb]
  field_simp
  ring

/-- The hypotheses of `integral_traceEst` are satisfiable: constant squared norms. -/
example (g t : ℝ) : ∫ ω, traceEst 1 2 ((fun _ : Unit => g + t / 1) ω)
    ((fun _ : Unit => g + t / 2) ω) ∂Measure.dirac () = t :=
  integral_traceEst (g := g) (integrable_const _) (integrable_const _) one_ne_zero
    two_ne_zero (by norm_num) (by simp) (by simp)

/-- **Eqs. (A.1)–(A.2)** in data-parallel training: with the first `b` of the `n` examples
as the local batch, `B_small = b`, and all of them as the global batch, `B_big = n`,
`E[|𝒢|²] = |G|²` and `E[𝒮] = tr(Σ)`. -/
theorem integral_estimates_batchGrad [IsProbabilityMeasure P] {b n : ℕ} (hb : b ≠ 0)
    (hbn : b < n) {X : Fin n → Ω → ι → ℝ} {G : ι → ℝ} {S : Matrix ι ι ℝ}
    (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a)
    (hS : ∀ i, covMatrix (X i) P = S) :
    (∫ ω, gradSqEst b n (batchGrad (X ∘ Fin.castLE hbn.le) ω ⬝ᵥ
        batchGrad (X ∘ Fin.castLE hbn.le) ω) (batchGrad X ω ⬝ᵥ batchGrad X ω) ∂P = G ⬝ᵥ G) ∧
      ∫ ω, traceEst b n (batchGrad (X ∘ Fin.castLE hbn.le) ω ⬝ᵥ
        batchGrad (X ∘ Fin.castLE hbn.le) ω) (batchGrad X ω ⬝ᵥ batchGrad X ω) ∂P = S.trace := by
  classical
  have hI : ∀ {B} (Y : Fin B → Ω → ι → ℝ), (∀ i a, MemLp (fun ω => Y i ω a) 2 P) →
      Integrable (fun ω => batchGrad Y ω ⬝ᵥ batchGrad Y ω) P := fun Y hY => by
    simpa only [Matrix.one_mulVec] using integrable_quadForm (memLp_batchGrad hY) 1
  have hXs : ∀ i a, MemLp (fun ω => (X ∘ Fin.castLE hbn.le) i ω a) 2 P := fun i => hX _
  have hs := integral_dotProduct_self_batchGrad hb hXs
    (fun i j hij => hind ((Fin.castLE_injective hbn.le).ne hij)) (fun i => hG _) (fun i => hS _)
  have hbig := integral_dotProduct_self_batchGrad (by omega) hX hind hG hS
  have hb' : (b : ℝ) ≠ 0 := by exact_mod_cast hb
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  have hne : (b : ℝ) ≠ n := by exact_mod_cast hbn.ne
  exact ⟨integral_gradSqEst (hI _ hXs) (hI _ hX) hb' hn' hne hs hbig,
    integral_traceEst (hI _ hXs) (hI _ hX) hb' hn' hne hs hbig⟩

/-- The hypotheses of `integral_estimates_batchGrad` are satisfiable: copies of a constant. -/
example (G : ι → ℝ) :=
  integral_estimates_batchGrad (P := Measure.dirac ()) (X := fun (_ : Fin 2) (_ : Unit) => G)
    (G := G) one_ne_zero one_lt_two (fun _ _ => memLp_const _)
    (fun _ _ _ => indepFun_const_left _ _) (fun _ _ => by simp) (fun _ => covMatrix_const G)

/-- The footnote after eq. (A.2): for `B_small = 1`, `B_big = n`,
`𝒮 = n/(n-1) (|G_small|² - |G_big|²)`. -/
theorem traceEst_one {n : ℝ} (hn : n ≠ 0) (hn1 : n ≠ 1) (ns nb : ℝ) :
    traceEst 1 n ns nb = n / (n - 1) * (ns - nb) := by
  have : n - 1 ≠ 0 := sub_ne_zero.2 hn1
  unfold traceEst
  field_simp

/-- The hypotheses of `traceEst_one` are satisfiable: `n = 2`. -/
example (ns nb : ℝ) : traceEst 1 2 ns nb = 2 / (2 - 1) * (ns - nb) :=
  traceEst_one two_ne_zero (by norm_num) ns nb

/-- The footnote after eq. (A.2): with `B_small = 1`, `B_big = n`, the mean `(1/n) Σ |x_i|²`
of the per-example squared norms for `|G_small|²` and `|x̄|²` for `|G_big|²`, `𝒮` is the
sample variance `(1/(n-1)) Σ |x_i - x̄|²` with Bessel's correction. -/
theorem traceEst_one_mean {n : ℕ} (hn : 2 ≤ n) (x : Fin n → ι → ℝ) :
    traceEst 1 n ((n : ℝ)⁻¹ * ∑ i, x i ⬝ᵥ x i)
        (((n : ℝ)⁻¹ • ∑ i, x i) ⬝ᵥ ((n : ℝ)⁻¹ • ∑ i, x i)) =
      ((n : ℝ) - 1)⁻¹ * ∑ i, (x i - (n : ℝ)⁻¹ • ∑ j, x j) ⬝ᵥ (x i - (n : ℝ)⁻¹ • ∑ j, x j) := by
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  have hn1 : (n : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ n := by exact_mod_cast hn
    linarith
  set m := (n : ℝ)⁻¹ • ∑ j, x j with hm
  have hsum : ∑ i, x i = (n : ℝ) • m := by rw [hm, smul_smul, mul_inv_cancel₀ hn0, one_smul]
  have key : ∑ i, (x i - m) ⬝ᵥ (x i - m) = ∑ i, x i ⬝ᵥ x i - n * (m ⬝ᵥ m) := by
    have h1 : ∑ i, x i ⬝ᵥ m = n * (m ⬝ᵥ m) := by
      rw [← sum_dotProduct, hsum, smul_dotProduct, smul_eq_mul]
    have h2 : ∑ i, m ⬝ᵥ x i = n * (m ⬝ᵥ m) := by
      rw [← dotProduct_sum, hsum, dotProduct_smul, smul_eq_mul]
    simp only [sub_dotProduct, dotProduct_sub, Finset.sum_sub_distrib, h1, h2,
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  rw [key, traceEst_one hn0 (sub_ne_zero.1 hn1)]
  field_simp

/-- The hypothesis of `traceEst_one_mean` is satisfiable: `n = 2`. -/
example (x : Fin 2 → ι → ℝ) :
    traceEst 1 (2 : ℕ) (((2 : ℕ) : ℝ)⁻¹ * ∑ i, x i ⬝ᵥ x i)
        ((((2 : ℕ) : ℝ)⁻¹ • ∑ i, x i) ⬝ᵥ (((2 : ℕ) : ℝ)⁻¹ • ∑ i, x i)) =
      (((2 : ℕ) : ℝ) - 1)⁻¹ * ∑ i, (x i - ((2 : ℕ) : ℝ)⁻¹ • ∑ j, x j) ⬝ᵥ
        (x i - ((2 : ℕ) : ℝ)⁻¹ • ∑ j, x j) :=
  traceEst_one_mean le_rfl x

end Transformer.NoiseScale
