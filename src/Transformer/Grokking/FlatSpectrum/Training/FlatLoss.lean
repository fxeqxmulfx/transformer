module

public import Mathlib.Analysis.Analytic.Constructions
public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Transformer.Landscape.SaddleAvoidance.Defs
public import Transformer.Grokking.FlatSpectrum.Training.FactorSpace
public import Transformer.Grokking.FlatSpectrum.Training.Gradient
public import Transformer.Grokking.FlatSpectrum.Training.SquaredLoss

/-!
# The weight-decayed squared loss on the factor space

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8.
`flatLoss` is the weight-decayed squared loss as a function of the vector that lists the
factors. It is a polynomial in the real and imaginary parts of the entries, hence real-analytic
(`analyticAt_flatLoss`). Its gradient for the real inner product is the vector listing
`M W = [G V + λ U; Gᴴ U + λ V]`, `G = P_Ω (U Vᴴ − K)` (`gradient_flatLoss`, from item WD3), so a
gradient step on the factor space is a step of `decayStep` on the factors
(`factors_gradientStep`), and the squared norm of the gradient is `‖M W‖²`
(`norm_gradient_flatLoss_sq`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open Transformer.Landscape.SaddleAvoidance (gradientStep)

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k]

/-- The weight-decayed squared loss `½ ‖P_Ω (U Vᴴ − K)‖² + λ · ½ (‖U‖² + ‖V‖²)` on the factor
space. -/
noncomputable def flatLoss (Ω : Set (m × n)) (K : Matrix m n ℂ) (lam : ℝ)
    (x : EuclideanSpace ℂ ((m ⊕ n) × k)) : ℝ :=
  decayedLoss (fitLoss Ω K) lam (factors x).1 (factors x).2

/-- The squared norm of a real-analytic complex function is real-analytic. -/
lemma analyticAt_norm_sq {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {c : E → ℂ}
    {x₀ : E} (h : AnalyticAt ℝ c x₀) : AnalyticAt ℝ (fun x => ‖c x‖ ^ 2) x₀ := by
  have e : (fun x => ‖c x‖ ^ 2) = fun x => (c x).re ^ 2 + (c x).im ^ 2 := by
    funext x
    rw [Complex.sq_norm, Complex.normSq_apply]
    ring
  rw [e]
  exact (((Complex.reCLM.analyticAt _).comp h).fun_pow 2).fun_add
    (((Complex.imCLM.analyticAt _).comp h).fun_pow 2)

example : AnalyticAt ℝ (fun z : ℂ => ‖z‖ ^ 2) 0 := analyticAt_norm_sq analyticAt_id

/-- **The weight-decayed squared loss is real-analytic** in the factors
(`review/flat_spectrum.md`, item WD8): a polynomial in the real and imaginary parts of the
entries. -/
theorem analyticAt_flatLoss (Ω : Set (m × n)) (K : Matrix m n ℂ) (lam : ℝ)
    (x₀ : EuclideanSpace ℂ ((m ⊕ n) × k)) : AnalyticAt ℝ (flatLoss Ω K lam) x₀ := by
  have hc (ij : (m ⊕ n) × k) :
      AnalyticAt ℝ (fun x : EuclideanSpace ℂ ((m ⊕ n) × k) => x ij) x₀ :=
    ((EuclideanSpace.proj ij).restrictScalars ℝ).analyticAt x₀
  have hstar (ij : (m ⊕ n) × k) :
      AnalyticAt ℝ (fun x : EuclideanSpace ℂ ((m ⊕ n) × k) => star (x ij)) x₀ :=
    (Complex.conjCLE.toContinuousLinearMap.analyticAt _).comp (hc ij)
  have hentry (a : m) (b : n) : AnalyticAt ℝ (fun x : EuclideanSpace ℂ ((m ⊕ n) × k) =>
      mask Ω ((factors x).1 * (factors x).2ᴴ - K) a b) x₀ := by
    by_cases hab : (a, b) ∈ Ω
    · simp only [mask, of_apply, hab, ↓reduceIte, Matrix.sub_apply, Matrix.mul_apply,
        conjTranspose_apply, factors]
      exact (Finset.analyticAt_fun_sum _ fun l _ =>
        (hc (Sum.inl a, l)).fun_mul (hstar (Sum.inr b, l))).fun_sub analyticAt_const
    · simp only [mask, of_apply, hab, ↓reduceIte]
      exact analyticAt_const
  unfold flatLoss decayedLoss fitLoss frobeniusSq
  refine AnalyticAt.fun_add (AnalyticAt.div_const (Finset.analyticAt_fun_sum _ fun a _ =>
    Finset.analyticAt_fun_sum _ fun b _ => analyticAt_norm_sq (hentry a b)))
    (analyticAt_const.fun_mul (AnalyticAt.div_const (AnalyticAt.fun_add
      (Finset.analyticAt_fun_sum _ fun i _ => Finset.analyticAt_fun_sum _ fun j _ =>
        analyticAt_norm_sq (hc (Sum.inl i, j)))
      (Finset.analyticAt_fun_sum _ fun i _ => Finset.analyticAt_fun_sum _ fun j _ =>
        analyticAt_norm_sq (hc (Sum.inr i, j))))))

/-- **The gradient of the weight-decayed squared loss on the factor space**
(`review/flat_spectrum.md`, items WD3 and WD8): the vector that lists
`[G V + λ U; Gᴴ U + λ V]` with `G = P_Ω (U Vᴴ − K)`. -/
theorem hasGradientAt_flatLoss (Ω : Set (m × n)) (K : Matrix m n ℂ) (lam : ℝ)
    (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    HasGradientAt (flatLoss Ω K lam) (flatten
      (mask Ω ((factors x).1 * (factors x).2ᴴ - K) * (factors x).2 + lam • (factors x).1,
        (mask Ω ((factors x).1 * (factors x).2ᴴ - K))ᴴ * (factors x).1 +
          lam • (factors x).2)) x := by
  have hd := (analyticAt_flatLoss Ω K lam x).differentiableAt
  rw [hasGradientAt_iff_hasFDerivAt]
  have key : fderiv ℝ (flatLoss Ω K lam) x = InnerProductSpace.toDual ℝ _ (flatten
      (mask Ω ((factors x).1 * (factors x).2ᴴ - K) * (factors x).2 + lam • (factors x).1,
        (mask Ω ((factors x).1 * (factors x).2ᴴ - K))ᴴ * (factors x).1 +
          lam • (factors x).2)) := by
    ext h
    have h0 : HasDerivAt (fun t : ℝ => x + t • h) h 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).smul_const h).const_add x
    have hline := hd.hasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) h0 (by simp)
    have hwd := hasDerivAt_decayedLoss (fitLoss_first_order Ω K) (continuous_mask_sub Ω K) lam
      (factors x).1 (factors h).1 (factors x).2 (factors h).2
    have e : (flatLoss Ω K lam ∘ fun t : ℝ => x + t • h) = fun t : ℝ =>
        decayedLoss (fitLoss Ω K) lam ((factors x).1 + t • (factors h).1)
          ((factors x).2 + t • (factors h).2) := by
      funext t
      simp only [Function.comp_apply, flatLoss, factors_add, factors_smul]
    rw [e] at hline
    rw [hline.unique hwd, InnerProductSpace.toDual_apply_apply, real_inner_eq_pairing_factors,
      factors_flatten]
  exact key ▸ hd.hasFDerivAt

/-- The gradient of `flatLoss` as an equation (`review/flat_spectrum.md`, item WD8). -/
theorem gradient_flatLoss (Ω : Set (m × n)) (K : Matrix m n ℂ) (lam : ℝ)
    (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    gradient (flatLoss Ω K lam) x = flatten
      (mask Ω ((factors x).1 * (factors x).2ᴴ - K) * (factors x).2 + lam • (factors x).1,
        (mask Ω ((factors x).1 * (factors x).2ᴴ - K))ᴴ * (factors x).1 +
          lam • (factors x).2) :=
  (hasGradientAt_flatLoss Ω K lam x).gradient

/-- **A gradient step on the factor space is a step of `decayStep`** on the factors
(`review/flat_spectrum.md`, items WD3 and WD8). -/
theorem factors_gradientStep (Ω : Set (m × n)) (K : Matrix m n ℂ) (lam η : ℝ)
    (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    factors (gradientStep (flatLoss Ω K lam) η x) =
      decayStep (fun X => mask Ω (X - K)) η lam (factors x) := by
  rw [gradientStep, gradient_flatLoss, factors_sub, factors_smul, factors_flatten]
  rfl

/-- The squared norm of the gradient is `‖M W‖²`. -/
theorem norm_gradient_flatLoss_sq [DecidableEq m] [DecidableEq n] (Ω : Set (m × n))
    (K : Matrix m n ℂ) (lam : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    ‖gradient (flatLoss Ω K lam) x‖ ^ 2 =
      frobeniusSq (lift lam (mask Ω ((factors x).1 * (factors x).2ᴴ - K)) *
        stack (factors x).1 (factors x).2) := by
  rw [gradient_flatLoss, norm_sq_eq_frobeniusSq_factors, factors_flatten, lift_mul_stack,
    frobeniusSq_stack]

end Transformer.Grokking.FlatSpectrum.Training
