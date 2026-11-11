module

public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Transformer.Landscape.SaddleAvoidance.Defs
public import Transformer.Grokking.FlatSpectrum.Training.Gradient
public import Transformer.Grokking.FlatSpectrum.Training.SquaredBounds

/-!
# The weight-decayed loss on the factor space

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8.
`flatLoss` is the weight-decayed loss as a function of the vector that lists the factors. If
the loss `L` is real-analytic in the entries of its matrix, `flatLoss` is real-analytic
(`analyticAt_flatLoss`): the entries of `U Vᴴ` are polynomials in the real and imaginary parts
of the entries of the factors (`analyticAt_productVector`), and so is the weight decay. If `L`
also satisfies the first-order inequality with a continuous `G`, the gradient of `flatLoss`
for the real inner product is the vector listing `M W = [G V + λ U; Gᴴ U + λ V]`,
`G = G (U Vᴴ)` (`gradient_flatLoss`, from item WD3). So a gradient step on the factor space is
a step of `decayStep` on the factors (`factors_gradientStep`), and the squared norm of the
gradient is `‖M W‖²` (`norm_gradient_flatLoss_sq`). The examples are the squared loss.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open Transformer.Landscape.SaddleAvoidance (gradientStep)

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] {L : Matrix m n ℂ → ℝ}
  {G : Matrix m n ℂ → Matrix m n ℂ}

/-- The weight-decayed loss `L (U Vᴴ) + λ · ½ (‖U‖² + ‖V‖²)` on the factor space. -/
noncomputable def flatLoss (L : Matrix m n ℂ → ℝ) (lam : ℝ)
    (x : EuclideanSpace ℂ ((m ⊕ n) × k)) : ℝ :=
  decayedLoss L lam (factors x).1 (factors x).2

/-- The vector that lists the entries of the product `U Vᴴ` of the factors. -/
noncomputable def productVector (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    EuclideanSpace ℂ (m × n) :=
  WithLp.toLp 2 fun ij => ((factors x).1 * (factors x).2ᴴ) ij.1 ij.2

/-- The entries of `U Vᴴ` are real-analytic in the factors: sums of products of an entry and a
conjugate entry. -/
lemma analyticAt_productVector (x₀ : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    AnalyticAt ℝ (productVector : EuclideanSpace ℂ ((m ⊕ n) × k) → EuclideanSpace ℂ (m × n))
      x₀ := by
  have hc (ij : (m ⊕ n) × k) :
      AnalyticAt ℝ (fun x : EuclideanSpace ℂ ((m ⊕ n) × k) => x ij) x₀ :=
    ((EuclideanSpace.proj ij).restrictScalars ℝ).analyticAt x₀
  have hpi : AnalyticAt ℝ (fun (x : EuclideanSpace ℂ ((m ⊕ n) × k)) (ij : m × n) =>
      ((factors x).1 * (factors x).2ᴴ) ij.1 ij.2) x₀ := by
    refine AnalyticAt.pi fun ij => ?_
    simp only [Matrix.mul_apply, conjTranspose_apply, factors, of_apply]
    exact Finset.analyticAt_fun_sum _ fun l _ => (hc (Sum.inl ij.1, l)).fun_mul
      ((Complex.conjCLE.toContinuousLinearMap.analyticAt _).comp (hc (Sum.inr ij.2, l)))
  exact ((PiLp.continuousLinearEquiv 2 ℝ fun _ : m × n => ℂ).symm.analyticAt _).comp hpi

/-- **The weight-decayed loss is real-analytic in the factors** (`review/flat_spectrum.md`,
item WD8) when the loss is real-analytic in the entries of its matrix. -/
theorem analyticAt_flatLoss
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (lam : ℝ) (x₀ : EuclideanSpace ℂ ((m ⊕ n) × k)) : AnalyticAt ℝ (flatLoss L lam) x₀ := by
  have hc (ij : (m ⊕ n) × k) :
      AnalyticAt ℝ (fun x : EuclideanSpace ℂ ((m ⊕ n) × k) => x ij) x₀ :=
    ((EuclideanSpace.proj ij).restrictScalars ℝ).analyticAt x₀
  have h1 : AnalyticAt ℝ (fun x : EuclideanSpace ℂ ((m ⊕ n) × k) =>
      L ((factors x).1 * (factors x).2ᴴ)) x₀ :=
    (hL (productVector x₀)).comp (analyticAt_productVector x₀)
  unfold flatLoss decayedLoss frobeniusSq
  exact h1.fun_add (analyticAt_const.fun_mul (AnalyticAt.div_const (AnalyticAt.fun_add
    (Finset.analyticAt_fun_sum _ fun i _ => Finset.analyticAt_fun_sum _ fun j _ =>
      analyticAt_norm_sq (hc (Sum.inl i, j)))
    (Finset.analyticAt_fun_sum _ fun i _ => Finset.analyticAt_fun_sum _ fun j _ =>
      analyticAt_norm_sq (hc (Sum.inr i, j))))))

/-- The gradient of the weight-decayed loss on the factor space, from its derivative along
every line of factors: no convexity is used (`review/flat_spectrum.md`, items WD3 and WD8). -/
theorem hasGradientAt_flatLoss_of_hasDerivAt
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (lam : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) {G₀ : Matrix m n ℂ}
    (hder : ∀ (A : Matrix m k ℂ) (B : Matrix n k ℂ), HasDerivAt
      (fun t : ℝ => decayedLoss L lam ((factors x).1 + t • A) ((factors x).2 + t • B))
      (pairing (G₀ * (factors x).2 + lam • (factors x).1) A +
        pairing (G₀ᴴ * (factors x).1 + lam • (factors x).2) B) 0) :
    HasGradientAt (flatLoss L lam) (flatten (G₀ * (factors x).2 + lam • (factors x).1,
      G₀ᴴ * (factors x).1 + lam • (factors x).2)) x := by
  have hd := (analyticAt_flatLoss hL lam x).differentiableAt
  rw [hasGradientAt_iff_hasFDerivAt]
  have key : fderiv ℝ (flatLoss L lam) x = InnerProductSpace.toDual ℝ _ (flatten
      (G₀ * (factors x).2 + lam • (factors x).1, G₀ᴴ * (factors x).1 + lam • (factors x).2)) := by
    ext h
    have h0 : HasDerivAt (fun t : ℝ => x + t • h) h 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).smul_const h).const_add x
    have hline := hd.hasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) h0 (by simp)
    have e : (flatLoss L lam ∘ fun t : ℝ => x + t • h) = fun t : ℝ =>
        decayedLoss L lam ((factors x).1 + t • (factors h).1)
          ((factors x).2 + t • (factors h).2) := by
      funext t
      simp only [Function.comp_apply, flatLoss, factors_add, factors_smul]
    rw [e] at hline
    rw [hline.unique (hder (factors h).1 (factors h).2), InnerProductSpace.toDual_apply_apply,
      real_inner_eq_pairing_factors, factors_flatten]
  exact key ▸ hd.hasFDerivAt

/-- **The gradient of the weight-decayed loss on the factor space**
(`review/flat_spectrum.md`, items WD3 and WD8): for `L` real-analytic in the entries with the
first-order inequality and a continuous `G`, it is the vector that lists
`[G V + λ U; Gᴴ U + λ V]` with `G = G (U Vᴴ)`. -/
theorem hasGradientAt_flatLoss (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (lam : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    HasGradientAt (flatLoss L lam) (flatten
      (G ((factors x).1 * (factors x).2ᴴ) * (factors x).2 + lam • (factors x).1,
        (G ((factors x).1 * (factors x).2ᴴ))ᴴ * (factors x).1 + lam • (factors x).2)) x :=
  hasGradientAt_flatLoss_of_hasDerivAt hL lam x fun A B =>
    hasDerivAt_decayedLoss hconv hG lam (factors x).1 A (factors x).2 B

/-- The gradient of `flatLoss` as an equation (`review/flat_spectrum.md`, item WD8). -/
theorem gradient_flatLoss (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (lam : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    gradient (flatLoss L lam) x = flatten
      (G ((factors x).1 * (factors x).2ᴴ) * (factors x).2 + lam • (factors x).1,
        (G ((factors x).1 * (factors x).2ᴴ))ᴴ * (factors x).1 + lam • (factors x).2) :=
  (hasGradientAt_flatLoss hconv hG hL lam x).gradient

/-- **A gradient step on the factor space is a step of `decayStep`** on the factors
(`review/flat_spectrum.md`, items WD3 and WD8). -/
theorem factors_gradientStep (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (lam η : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    factors (gradientStep (flatLoss L lam) η x) = decayStep G η lam (factors x) := by
  rw [gradientStep, gradient_flatLoss hconv hG hL, factors_sub, factors_smul, factors_flatten]
  rfl

/-- The squared norm of the gradient is `‖M W‖²`. -/
theorem norm_gradient_flatLoss_sq [DecidableEq m] [DecidableEq n]
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (lam : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    ‖gradient (flatLoss L lam) x‖ ^ 2 =
      frobeniusSq (lift lam (G ((factors x).1 * (factors x).2ᴴ)) *
        stack (factors x).1 (factors x).2) := by
  rw [gradient_flatLoss hconv hG hL, norm_sq_eq_frobeniusSq_factors, factors_flatten,
    lift_mul_stack, frobeniusSq_stack]

/-! The witness: the squared loss of the `1 × 1` kernel `1` at width `1`, at the origin. -/

example := analyticAt_flatLoss (k := Fin 1) (analyticAt_fitLoss (Set.univ : Set (Fin 1 × Fin 1)) 1)
  1 0

example := hasGradientAt_flatLoss (k := Fin 1)
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) 1 0

example := hasGradientAt_flatLoss_of_hasDerivAt (k := Fin 1)
  (analyticAt_fitLoss (Set.univ : Set (Fin 1 × Fin 1)) 1) 1 0 fun A B =>
    hasDerivAt_decayedLoss (fitLoss_first_order Set.univ 1) (continuous_mask_sub Set.univ 1) 1 _ A
      _ B

example := gradient_flatLoss (k := Fin 1)
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) 1 0

example := factors_gradientStep (k := Fin 1)
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) 1 (1 / 10) 0

example := norm_gradient_flatLoss_sq (k := Fin 1)
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) 1 0

end Transformer.Grokking.FlatSpectrum.Training
